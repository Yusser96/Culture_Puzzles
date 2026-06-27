"""
src/modules/metadata/build.py
==============================
Build the unified per-sample metadata table from the raw corpora.

Produces a DataFrame with exactly MetadataTable.COLUMNS (15 columns).

Sources handled:
  puzzles/<region>/riddles.jsonl   -> puzzles_original / puzzles_translation
  parallel/{flores,opus100}/<region>.jsonl  -> flores / opus100
  cultural/<topic>/<region>.txt    -> cultural
  sib200/<topic>/<lang>.txt        -> sib200
"""

import hashlib
import json
import os
from typing import List, Optional

import pandas as pd

from src.shared_utils.registry import region_factors
from src.shared_utils.store import MetadataTable
from src.shared_utils.text import detect_script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_split(sample_id: str) -> str:
    """Deterministic 80/20 train/test split based on MD5 hash of sample_id."""
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
    return "train" if (h % 100) < 80 else "test"


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def _row(sample_id, text, source, topic, topic_canonical, topic_raw,
         factors, domain, token_count, translation_group_id):
    return {
        "sample_id": sample_id,
        "text": text,
        "source": source,
        "topic": topic,
        "topic_canonical": topic_canonical,
        "topic_raw": topic_raw,
        "language": factors["base_language"],
        "region": factors["region"],
        "language_region": factors["language_region"],
        "script": detect_script(text),
        "domain": domain,
        "prompt_template": "raw",
        "token_count": token_count,
        "translation_group_id": translation_group_id,
        "split": _deterministic_split(sample_id),
    }


# ---------------------------------------------------------------------------
# Per-source builders
# ---------------------------------------------------------------------------

def _build_puzzles(raw_dir: str, cfg: dict, tokenizer) -> List[dict]:
    """Walk raw_dir/puzzles/<region>/riddles.jsonl."""
    rows: List[dict] = []
    puzzles_dir = os.path.join(raw_dir, "puzzles")
    if not os.path.isdir(puzzles_dir):
        return rows

    canonical_topics = set(cfg.get("canonical_topics", []))
    topic_label_map = cfg.get("topic_label_map", {})

    for region_key in sorted(os.listdir(puzzles_dir)):
        region_dir = os.path.join(puzzles_dir, region_key)
        if not os.path.isdir(region_dir):
            continue
        jsonl_path = os.path.join(region_dir, "riddles.jsonl")
        if not os.path.isfile(jsonl_path):
            continue

        factors = region_factors(region_key, cfg)

        with open(jsonl_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                topic_raw = rec.get("topic")
                mapped = topic_label_map.get(topic_raw) if topic_raw else None
                topic_canonical = mapped if (mapped in canonical_topics) else None

                for source, text_key in [
                    ("puzzles_original", "riddle_original"),
                    ("puzzles_translation", "riddle_translation"),
                ]:
                    text = rec.get(text_key)
                    if not text:
                        continue
                    sample_id = f"{source}/{region_key}/{i}"
                    rows.append(_row(
                        sample_id=sample_id,
                        text=text,
                        source=source,
                        topic=topic_raw,
                        topic_canonical=topic_canonical,
                        topic_raw=topic_raw,
                        factors=factors,
                        domain=source,
                        token_count=_count_tokens(text, tokenizer),
                        translation_group_id=None,
                    ))
    return rows


def _build_parallel(raw_dir: str, cfg: dict, tokenizer) -> List[dict]:
    """Walk raw_dir/parallel/{flores,opus100}/<region>.jsonl."""
    rows: List[dict] = []
    parallel_dir = os.path.join(raw_dir, "parallel")
    if not os.path.isdir(parallel_dir):
        return rows

    for dataset in ("flores", "opus100"):
        dataset_dir = os.path.join(parallel_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.endswith(".jsonl"):
                continue
            region_key = fname[: -len(".jsonl")]
            fpath = os.path.join(dataset_dir, fname)
            factors = region_factors(region_key, cfg)

            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = rec.get("text", "")
                    if not text:
                        continue
                    translation_group_id = rec.get("translation_group_id")
                    sample_id = f"{dataset}/{region_key}/{i}"
                    rows.append(_row(
                        sample_id=sample_id,
                        text=text,
                        source=dataset,
                        topic=None,
                        topic_canonical=None,
                        topic_raw=None,
                        factors=factors,
                        domain=dataset,
                        token_count=_count_tokens(text, tokenizer),
                        translation_group_id=translation_group_id,
                    ))
    return rows


def _build_cultural(raw_dir: str, cfg: dict, tokenizer) -> List[dict]:
    """Walk raw_dir/cultural/<topic>/<region>.txt."""
    rows: List[dict] = []
    cultural_dir = os.path.join(raw_dir, "cultural")
    if not os.path.isdir(cultural_dir):
        return rows

    for topic in sorted(os.listdir(cultural_dir)):
        topic_dir = os.path.join(cultural_dir, topic)
        if not os.path.isdir(topic_dir):
            continue

        for fname in sorted(os.listdir(topic_dir)):
            if not fname.endswith(".txt"):
                continue
            region_key = fname[: -len(".txt")]
            fpath = os.path.join(topic_dir, fname)
            factors = region_factors(region_key, cfg)

            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    text = line.strip()
                    if not text:
                        continue
                    sample_id = f"cultural/{topic}/{region_key}/{i}"
                    rows.append(_row(
                        sample_id=sample_id,
                        text=text,
                        source="cultural",
                        topic=topic,
                        topic_canonical=topic,
                        topic_raw=topic,
                        factors=factors,
                        domain="cultural",
                        token_count=_count_tokens(text, tokenizer),
                        translation_group_id=None,
                    ))
    return rows


def _sib_code_to_region_key(sib_code: str, cfg: dict) -> str:
    """Convert a SIB-200 code (e.g. 'arz-Arab') back to a lang_region key.

    SIB codes are FLORES codes with '_' replaced by '-'.  We reverse that
    substitution and search the registry for the first matching flores code.
    Falls back to the raw sib_code if no match is found.
    """
    flores_code = sib_code.replace("-", "_")
    for r in cfg.get("lang_regions", []):
        if r.get("flores") == flores_code:
            return r["key"]
    return sib_code


def _build_sib200(raw_dir: str, cfg: dict, tokenizer) -> List[dict]:
    """Walk raw_dir/sib200/<topic>/<lang>.txt."""
    rows: List[dict] = []
    sib_dir = os.path.join(raw_dir, "sib200")
    if not os.path.isdir(sib_dir):
        return rows

    for topic in sorted(os.listdir(sib_dir)):
        topic_dir = os.path.join(sib_dir, topic)
        if not os.path.isdir(topic_dir):
            continue

        for fname in sorted(os.listdir(topic_dir)):
            if not fname.endswith(".txt"):
                continue
            lang_code = fname[: -len(".txt")]
            region_key = _sib_code_to_region_key(lang_code, cfg)
            fpath = os.path.join(topic_dir, fname)
            factors = region_factors(region_key, cfg)

            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    text = line.strip()
                    if not text:
                        continue
                    sample_id = f"sib200/{topic}/{lang_code}/{i}"
                    rows.append(_row(
                        sample_id=sample_id,
                        text=text,
                        source="sib200",
                        topic=topic,
                        topic_canonical=topic,
                        topic_raw=topic,
                        factors=factors,
                        domain="sib200",
                        token_count=_count_tokens(text, tokenizer),
                        translation_group_id=None,
                    ))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_metadata(cfg: dict, tokenizer=None) -> pd.DataFrame:
    """Build the unified metadata table from all raw corpora.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration (must contain ``paths.raw_dir``).
    tokenizer : optional
        If provided, ``token_count`` is ``len(tokenizer.encode(text))``;
        otherwise falls back to whitespace split length.

    Returns
    -------
    pd.DataFrame with exactly ``MetadataTable.COLUMNS`` (15 columns).
    """
    raw_dir = cfg["paths"]["raw_dir"]
    rows: List[dict] = []
    rows.extend(_build_puzzles(raw_dir, cfg, tokenizer))
    rows.extend(_build_parallel(raw_dir, cfg, tokenizer))
    rows.extend(_build_cultural(raw_dir, cfg, tokenizer))
    rows.extend(_build_sib200(raw_dir, cfg, tokenizer))

    if rows:
        df = pd.DataFrame(rows)[MetadataTable.COLUMNS]
    else:
        df = pd.DataFrame(columns=MetadataTable.COLUMNS)
    return df
