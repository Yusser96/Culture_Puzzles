#!/usr/bin/env python3
"""
03_collect_topics.py
====================
Collect Wikipedia text for v0's 8 canonical topics, keyed on the lang_region
registry "when available". Wikipedia is language-keyed, so we collect **once per
unique `wiki` code** and replicate the corpus into each lang_region that shares it
(recording `shared_with` so downstream knows the baseline is shared). Regions with
`wiki: null` are skipped.

Topics are defined by English seed articles (`seed_en`) and resolved into each
target edition via interlanguage links (langlinks), so we only author 8 English
seed lists instead of 8 x N hand-translated title lists. Region-specific
"<country>" seeds are skipped here (they cannot be resolved per-language uniformly).

Output structure:
  vector_analysis/results/data/cultural/
    <topic_key>/<lang_region>.txt
    manifest.json    # topic -> lang_region -> {path, num_sentences, wiki, shared_with}

Usage:
    python 03_collect_topics.py [--config configs/riddles_config.yaml]
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List

import wikipediaapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import (
    load_config, ensure_dirs, setup_logging, load_registry,
    save_sentences, save_json,
)
from shared_utils.wiki import resolve_langlink

logger = setup_logging("collect_topics")


def sentence_split(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def fetch_article(wiki, title: str, max_chars: int = 50000) -> str:
    # Wikipedia's API can intermittently return non-JSON/HTML (rate limits,
    # transient errors); retry briefly then give up gracefully.
    for attempt in range(3):
        try:
            page = wiki.page(title)
            if not page.exists():
                return ""
            return page.text[:max_chars]
        except Exception as e:  # noqa: BLE001 - any network/parse error -> retry/skip
            if attempt == 2:
                logger.warning(f"      fetch failed for '{title}': {e}")
                return ""
            time.sleep(1.0 * (attempt + 1))
    return ""


def collect_topic_for_wiki(topic_cfg, wiki_code, n_sentences) -> List[str]:
    """Collect up to n_sentences for one (topic, wiki edition)."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="CultureRiddlesResearch/1.0 (research@example.com)",
        language=wiki_code,
    )
    seeds = [s for s in topic_cfg.get("seed_en", []) if "<country>" not in s]
    collected, seen = [], set()
    for seed in seeds:
        title = resolve_langlink(seed, wiki_code) or seed  # fall back to seed title
        text = fetch_article(wiki, title)
        if not text:
            logger.warning(f"      no article for seed '{seed}' -> '{title}' ({wiki_code})")
            continue
        for s in sentence_split(text):
            if s not in seen:
                seen.add(s)
                collected.append(s)
        time.sleep(0.5)
        if len(collected) >= n_sentences:
            break
    return collected[:n_sentences]


def main():
    parser = argparse.ArgumentParser(description="Collect topic data (registry-keyed)")
    parser.add_argument("--config", default="configs/riddles_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    registry = load_registry(cfg)
    out_dir = cfg["paths"]["va_cultural_dir"]
    n_sentences = cfg["data"]["topic_sentences_per_lang"]
    topics = cfg["cultural_topics"]

    # wiki code -> list of regions sharing it
    regions_by_wiki = defaultdict(list)
    for r in registry:
        if r.get("wiki"):
            regions_by_wiki[r["wiki"]].append(r["key"])
    wiki_codes = sorted(regions_by_wiki)
    logger.info(f"{len(wiki_codes)} unique wiki editions for {len(registry)} regions")

    manifest: Dict[str, Dict] = {}

    for topic_key, topic_cfg in topics.items():
        logger.info(f"Topic: {topic_key} -- {topic_cfg.get('description', '')}")
        manifest[topic_key] = {}
        topic_dir = os.path.join(out_dir, topic_key)
        os.makedirs(topic_dir, exist_ok=True)

        for wiki_code in wiki_codes:
            try:
                sentences = collect_topic_for_wiki(topic_cfg, wiki_code, n_sentences)
            except Exception as e:  # noqa: BLE001 - never let one edition kill the run
                logger.warning(f"  [{wiki_code}] collection failed: {e}; skipping.")
                sentences = []
            regions = regions_by_wiki[wiki_code]
            logger.info(f"  [{wiki_code:<6}] {len(sentences)} sentences "
                        f"-> {len(regions)} region(s)")
            for key in regions:
                path = os.path.join(topic_dir, f"{key}.txt")
                save_sentences(sentences, path)
                manifest[topic_key][key] = {
                    "path": path,
                    "wiki": wiki_code,
                    "num_sentences": len(sentences),
                    "shared_with": [k for k in regions if k != key],
                }

    save_json(manifest, os.path.join(out_dir, "manifest.json"))
    logger.info("\nTopic data collection complete.")


if __name__ == "__main__":
    main()
