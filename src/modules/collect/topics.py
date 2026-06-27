"""
src/modules/collect/topics.py
==============================
Collect Wikipedia text for the 8 canonical cultural topics, keyed on the
lang_region registry "when available". Wikipedia is language-keyed, so collection
runs once per unique wiki code and is replicated into every lang_region sharing it.
Regions with wiki: null are skipped.

Seed articles are in English and resolved into each target edition via
interlanguage links (langlinks) from src.shared_utils.wiki. Seeds with
"<country>" placeholders are skipped (cannot be resolved per-language uniformly).

Output structure under raw_dir:
  cultural/
    <topic_key>/<lang_region>.txt
    manifest.json
"""

import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List

from src.shared_utils.io import setup_logging, save_sentences, save_json, ensure_dir
from src.shared_utils.registry import load_registry
from src.shared_utils.wiki import resolve_langlink

logger = setup_logging("collect_topics")


def sentence_split(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def fetch_article(wiki, title: str, max_chars: int = 50000) -> str:
    """Fetch a Wikipedia article text; retry up to 3 times on transient errors."""
    for attempt in range(3):
        try:
            page = wiki.page(title)
            if not page.exists():
                return ""
            return page.text[:max_chars]
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                logger.warning(f"      fetch failed for '{title}': {e}")
                return ""
            time.sleep(1.0 * (attempt + 1))
    return ""


def collect_topic_for_wiki(topic_cfg: dict, wiki_code: str, n_sentences: int) -> List[str]:
    """Collect up to n_sentences for one (topic, wiki edition)."""
    import wikipediaapi  # type: ignore

    wiki = wikipediaapi.Wikipedia(
        user_agent="CultureRiddlesResearch/1.0 (research@example.com)",
        language=wiki_code,
    )
    seeds = [s for s in topic_cfg.get("seed_en", []) if "<country>" not in s]
    collected, seen = [], set()
    for seed in seeds:
        title = resolve_langlink(seed, wiki_code) or seed
        text = fetch_article(wiki, title)
        if not text:
            logger.warning(
                f"      no article for seed '{seed}' -> '{title}' ({wiki_code})"
            )
            continue
        for s in sentence_split(text):
            if s not in seen:
                seen.add(s)
                collected.append(s)
        time.sleep(0.5)
        if len(collected) >= n_sentences:
            break
    return collected[:n_sentences]


def collect(cfg: dict) -> None:
    """Collect topic corpora from Wikipedia into raw_dir/cultural/."""
    registry = load_registry(cfg)
    raw_dir = cfg["paths"]["raw_dir"]
    out_dir = os.path.join(raw_dir, "cultural")
    n_sentences = cfg.get("data", {}).get("topic_sentences_per_lang", 200)
    topics = cfg.get("cultural_topics", {})

    if not topics:
        logger.warning("No cultural_topics in config; nothing to collect.")
        return

    # wiki code -> list of region keys sharing it
    regions_by_wiki: Dict[str, List[str]] = defaultdict(list)
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
        ensure_dir(topic_dir)

        for wiki_code in wiki_codes:
            try:
                sentences = collect_topic_for_wiki(topic_cfg, wiki_code, n_sentences)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  [{wiki_code}] collection failed: {e}; skipping.")
                sentences = []
            regions = regions_by_wiki[wiki_code]
            logger.info(
                f"  [{wiki_code:<6}] {len(sentences)} sentences "
                f"-> {len(regions)} region(s)"
            )
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
    logger.info("Topic data collection complete.")
