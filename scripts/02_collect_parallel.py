#!/usr/bin/env python3
"""
02_collect_parallel.py
======================
Download parallel sentences (FLORES-200 and OPUS-100) keyed on the **lang_region**
registry, "when available". FLORES/OPUS are language-keyed, so several lang_regions
collapse onto one code (e.g. Spanish_* -> spa_Latn / es). We therefore download
**once per unique code** and replicate the corpus into each lang_region file that
maps to it. Regions whose code is null are skipped (logged).

Note: FLORES distinguishes the Arabic varieties (arz/ary/apc/arb), so the 4 Arabic
regions get genuinely different FLORES corpora; OPUS only has language-level `ar`,
so the 4 Arabic regions share one OPUS corpus.

Output structure (re-keyed on lang_region):
  vector_analysis/results/data/parallel/
    flores/   <lang_region>.txt   + manifest.json
    opus100/  <lang_region>.txt   + manifest.json
    manifest.json

Usage:
    python 02_collect_parallel.py [--config configs/riddles_config.yaml]
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import (
    load_config, ensure_dirs, setup_logging, load_registry,
    save_sentences, save_json, validate_flores_codes,
)

logger = setup_logging("collect_parallel")

FLORES_SOURCE = "eng_Latn"   # English is always the source side


def _filter(text, min_len, max_len):
    return bool(text) and min_len <= len(text) <= max_len


# -- FLORES-200 ----------------------------------------------------------------

def collect_flores_by_code(cfg: dict, codes: set) -> Dict[str, List[str]]:
    """Download FLORES-200 once per unique target code. Returns {flores_code: [sents]}."""
    n = cfg["data"]["parallel_samples_per_lang"]
    min_len = cfg["data"]["min_sentence_length"]
    max_len = cfg["data"]["max_sentence_length"]
    split = "dev"

    result: Dict[str, List[str]] = {}
    en_sentences: List[str] = []

    for code in sorted(c for c in codes if c and c != FLORES_SOURCE):
        pair = f"{FLORES_SOURCE}-{code}"
        logger.info(f"  FLORES pair: {pair}")
        try:
            ds = load_dataset("facebook/flores", name=pair, trust_remote_code=True,
                              token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            logger.warning(f"    Could not load {pair}: {e}")
            continue

        if not en_sentences:
            field = f"sentence_{FLORES_SOURCE}"
            for item in ds[split]:
                t = item.get(field, "")
                if _filter(t, min_len, max_len):
                    en_sentences.append(t.strip())
                if len(en_sentences) >= n:
                    break

        tgt_field = f"sentence_{code}"
        tgt = []
        for item in ds[split]:
            t = item.get(tgt_field, "")
            if _filter(t, min_len, max_len):
                tgt.append(t.strip())
            if len(tgt) >= n:
                break
        result[code] = tgt
        logger.info(f"    [{code}] {len(tgt)} sentences")

    if FLORES_SOURCE in codes:
        result[FLORES_SOURCE] = en_sentences
    return result


# -- OPUS-100 ------------------------------------------------------------------

def collect_opus_by_code(cfg: dict, codes: set) -> Dict[str, List[str]]:
    """Download OPUS-100 once per unique language code. Returns {opus_code: [sents]}."""
    n = cfg["data"]["parallel_samples_per_lang"]
    min_len = cfg["data"]["min_sentence_length"]
    max_len = cfg["data"]["max_sentence_length"]

    result: Dict[str, List[str]] = {}
    en_all: List[str] = []

    for code in sorted(c for c in codes if c and c != "en"):
        pair_sorted = "-".join(sorted(["en", code]))
        logger.info(f"  OPUS-100 pair: {pair_sorted}")
        try:
            ds = load_dataset("opus100", pair_sorted, split="test", trust_remote_code=True,
                              token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            logger.warning(f"    Could not load {pair_sorted}: {e}")
            continue

        sents, en_sents = [], []
        for ex in ds:
            tr = ex.get("translation", {})
            t = tr.get(code, "")
            en_t = tr.get("en", "")
            if _filter(t, min_len, max_len):
                sents.append(t.strip())
                if _filter(en_t, min_len, max_len):
                    en_sents.append(en_t.strip())
            if len(sents) >= n:
                break
        result[code] = sents
        en_all.extend(en_sents)
        logger.info(f"    [{code}] {len(sents)} sentences")

    if "en" in codes:
        seen, en_unique = set(), []
        for s in en_all:
            if s not in seen:
                seen.add(s)
                en_unique.append(s)
        result["en"] = en_unique[:n]
    return result


def _save_by_region(registry, code_field, by_code, out_dir, dataset_name):
    """Replicate each code's corpus into every lang_region file that maps to it."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"dataset": dataset_name, "lang_regions": {}}
    shared = defaultdict(list)
    for r in registry:
        shared[r.get(code_field)].append(r["key"])

    for r in registry:
        key = r["key"]
        code = r.get(code_field)
        if not code or code not in by_code:
            continue
        sents = by_code[code]
        path = os.path.join(out_dir, f"{key}.txt")
        save_sentences(sents, path)
        manifest["lang_regions"][key] = {
            "path": path,
            "code": code,
            "num_sentences": len(sents),
            "shared_with": [k for k in shared[code] if k != key],
        }
    save_json(manifest, os.path.join(out_dir, "manifest.json"))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Collect parallel data (registry-keyed)")
    parser.add_argument("--config", default="configs/riddles_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    validate_flores_codes(cfg, logger)

    registry = load_registry(cfg)
    parallel_dir = cfg["paths"]["va_parallel_dir"]

    flores_codes = {r.get("flores") for r in registry if r.get("flores")}
    opus_codes = {r.get("opus") for r in registry if r.get("opus")}
    logger.info(f"{len(flores_codes)} unique FLORES codes, {len(opus_codes)} unique OPUS codes")

    # -- FLORES ----------------------------------------------------------------
    flores_by_code = collect_flores_by_code(cfg, flores_codes)
    flores_manifest = _save_by_region(
        registry, "flores", flores_by_code,
        os.path.join(parallel_dir, "flores"), "flores",
    )

    # -- OPUS-100 --------------------------------------------------------------
    opus_by_code = collect_opus_by_code(cfg, opus_codes)
    opus_manifest = _save_by_region(
        registry, "opus", opus_by_code,
        os.path.join(parallel_dir, "opus100"), "opus100",
    )

    save_json(
        {"datasets": ["flores", "opus100"], "flores": flores_manifest, "opus100": opus_manifest},
        os.path.join(parallel_dir, "manifest.json"),
    )
    logger.info(
        f"\nParallel data complete: "
        f"flores {len(flores_manifest['lang_regions'])} regions, "
        f"opus100 {len(opus_manifest['lang_regions'])} regions."
    )


if __name__ == "__main__":
    main()
