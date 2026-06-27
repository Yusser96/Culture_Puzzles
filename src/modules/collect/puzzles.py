"""
src/modules/collect/puzzles.py
==============================
Ingest Cultural Riddles Benchmark .xlsx files into the raw corpus tree.

For each lang_region writes:
  raw_dir/puzzles/<lang_region>/original.txt
  raw_dir/puzzles/<lang_region>/translation.txt
  raw_dir/puzzles/<lang_region>/riddles.jsonl
  raw_dir/puzzles/manifest.json

Reads xlsx from cfg['paths']['puzzles_src_dir'].
"""

import glob
import logging
import os
from collections import Counter, defaultdict
from typing import Dict

from src.shared_utils.io import (
    setup_logging, save_sentences, save_jsonl, save_json, ensure_dir,
)
from src.shared_utils.registry import load_registry
from src.shared_utils.riddles import (
    read_riddles_xlsx, parse_lang_region_key, RiddleSheetError,
)

logger = setup_logging("collect_puzzles")


def _clean(text: str) -> str:
    """Collapse internal whitespace/newlines so each riddle stays on one line."""
    return " ".join(text.split()) if text else text


def _build_line(text, answer, join: str) -> str:
    parts = [_clean(text)]
    if answer:
        parts.append(_clean(answer))
    return join.join(parts)


def collect(cfg: dict) -> None:
    """Collect puzzle (riddle) corpora from xlsx files into raw_dir/puzzles/."""
    src_dir = cfg["paths"]["puzzles_src_dir"]
    raw_dir = cfg["paths"]["raw_dir"]
    out_dir = os.path.join(raw_dir, "puzzles")
    join = cfg.get("data", {}).get("riddle_join", "  ||  ")
    label_map = cfg.get("topic_label_map", {})
    registry_keys = {r["key"] for r in load_registry(cfg)}

    files = sorted(glob.glob(os.path.join(src_dir, "*.xlsx")))
    if not files:
        logger.error(f"No .xlsx files found in {src_dir!r}")
        return
    logger.info(f"Found {len(files)} .xlsx files in {src_dir!r}")

    # -- Validation pass -------------------------------------------------------
    problems: Dict[str, list] = defaultdict(list)
    rows_by_key = {}
    seen_keys: Dict[str, str] = {}

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
        logger.error("Validation problems detected:")
        for kind, items in problems.items():
            logger.error(f"  [{kind}] ({len(items)})")
            for it in items:
                logger.error(f"      - {it}")
        # Non-fatal: write what we can, but log all issues.

    # -- Write per-lang_region corpora -----------------------------------------
    ensure_dir(out_dir)
    manifest = {}

    for key in sorted(rows_by_key):
        _, rows = rows_by_key[key]
        region_dir = os.path.join(out_dir, key)
        ensure_dir(region_dir)

        original_lines, translation_lines, jsonl_rows = [], [], []
        topic_counts: Counter = Counter()

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
    logger.info(f"Puzzle collection complete: {len(manifest)} lang_regions -> {out_dir}")
