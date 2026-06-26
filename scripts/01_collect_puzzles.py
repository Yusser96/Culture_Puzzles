#!/usr/bin/env python3
"""
01_collect_puzzles.py
=====================
Ingest the Cultural Riddles Benchmark .xlsx files (one per lang_region) into the
same on-disk corpus format used by the other collectors. This runs FIRST: its
manifest is the authoritative lang_region + topic inventory consumed by
02_collect_parallel.py and 03_collect_topics.py.

For each lang_region we write two parallel corpora (one line per riddle):
  - original.txt    : "<riddle original>  ||  <reference answer original>"
  - translation.txt : "<riddle english>   ||  <reference answer english>"
plus riddles.jsonl with the full structured rows.

Validation is strict (fail loudly): the script reports ALL problems at once and
exits non-zero before writing anything if any file has no resolvable riddle sheet
/ missing columns, a duplicate lang_region key, or a key absent from the config
registry.

Output structure:
  vector_analysis/results/data/puzzles/
    <lang_region>/
      original.txt
      translation.txt
      riddles.jsonl
    manifest.json

Usage:
    python 01_collect_puzzles.py [--config configs/riddles_config.yaml]
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import (
    load_config, ensure_dirs, setup_logging, load_registry,
    save_sentences, save_jsonl, save_json,
)
from shared_utils.riddles import (
    read_riddles_xlsx, parse_lang_region_key, RiddleSheetError,
)

logger = setup_logging("collect_puzzles")


def _clean(text):
    """Collapse internal whitespace/newlines so each riddle stays on one line."""
    return " ".join(text.split()) if text else text


def _build_line(text, answer, join):
    parts = [_clean(text)]
    if answer:
        parts.append(_clean(answer))
    return join.join(parts)


def main():
    parser = argparse.ArgumentParser(description="Collect riddle (puzzle) corpora")
    parser.add_argument("--config", default="configs/riddles_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    src_dir = cfg["paths"]["puzzles_src_dir"]
    out_dir = cfg["paths"]["va_puzzles_dir"]
    join = cfg["data"]["riddle_join"]
    label_map = cfg.get("topic_label_map", {})
    registry_keys = {r["key"] for r in load_registry(cfg)}

    files = sorted(glob.glob(os.path.join(src_dir, "*.xlsx")))
    if not files:
        logger.error(f"No .xlsx files found in {src_dir!r}")
        sys.exit(1)
    logger.info(f"Found {len(files)} .xlsx files in {src_dir!r}")

    # -- Validation pass (collect ALL problems before doing any work) ----------
    problems = defaultdict(list)
    rows_by_key = {}      # key -> (filepath, rows)
    seen_keys = {}        # key -> first filepath (to detect collisions)

    for path in files:
        fname = os.path.basename(path)
        key = parse_lang_region_key(fname)
        if not key:
            problems["no bracketed [lang_region] in filename"].append(fname)
            continue
        if key in seen_keys:
            problems["duplicate lang_region key"].append(
                f"{fname}  (collides with {os.path.basename(seen_keys[key])} -> key '{key}')"
            )
            continue
        seen_keys[key] = path
        if key not in registry_keys:
            problems["lang_region key not in config registry"].append(
                f"{fname}  (key '{key}')"
            )
            # keep going so we still report sheet/column problems
        try:
            rows = read_riddles_xlsx(path)
        except RiddleSheetError as e:
            problems["unresolvable riddle sheet / columns"].append(f"{fname}: {e}")
            continue
        if not rows:
            problems["no riddle rows found"].append(fname)
            continue
        rows_by_key[key] = (path, rows)

    if problems:
        logger.error("Validation failed. Fix these issues and re-run:")
        for kind, items in problems.items():
            logger.error(f"  [{kind}] ({len(items)})")
            for it in items:
                logger.error(f"      - {it}")
        sys.exit(1)

    # -- Write per-lang_region corpora -----------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}

    for key in sorted(rows_by_key):
        _, rows = rows_by_key[key]
        region_dir = os.path.join(out_dir, key)
        os.makedirs(region_dir, exist_ok=True)

        original_lines, translation_lines, jsonl_rows = [], [], []
        topic_counts = Counter()

        for r in rows:
            topic_raw = r.get("topic")
            topic_key = label_map.get(topic_raw) if topic_raw else None
            topic_counts[topic_key or (topic_raw or "UNKNOWN")] += 1

            if r.get("riddle_original"):
                original_lines.append(
                    _build_line(r["riddle_original"], r.get("ref_orig"), join)
                )
            if r.get("riddle_translation"):
                translation_lines.append(
                    _build_line(r["riddle_translation"], r.get("ref_en"), join)
                )

            jsonl_rows.append({
                "number": r.get("number"),
                "topic": topic_raw,
                "topic_key": topic_key,
                "author": r.get("author"),
                "riddle_original": r.get("riddle_original"),
                "riddle_translation": r.get("riddle_translation"),
                "ref_orig": r.get("ref_orig"),
                "ref_en": r.get("ref_en"),
            })

        original_path = os.path.join(region_dir, "original.txt")
        translation_path = os.path.join(region_dir, "translation.txt")
        jsonl_path = os.path.join(region_dir, "riddles.jsonl")
        save_sentences(original_lines, original_path)
        save_sentences(translation_lines, translation_path)
        save_jsonl(jsonl_rows, jsonl_path)

        manifest[key] = {
            "n_riddles": len(rows),
            "n_original": len(original_lines),
            "n_translation": len(translation_lines),
            "topics": dict(topic_counts),
            "paths": {
                "original": original_path,
                "translation": translation_path,
                "riddles": jsonl_path,
            },
        }
        logger.info(
            f"  [{key:<28}] {len(rows):>3} riddles "
            f"(orig {len(original_lines)}, trans {len(translation_lines)}, "
            f"{len(topic_counts)} topics)"
        )

    save_json(manifest, os.path.join(out_dir, "manifest.json"))
    logger.info(f"\nPuzzle collection complete: {len(manifest)} lang_regions -> {out_dir}")


if __name__ == "__main__":
    main()
