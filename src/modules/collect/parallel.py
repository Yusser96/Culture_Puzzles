"""
src/modules/collect/parallel.py
================================
Download parallel sentences (FLORES-200 and OPUS-100) keyed on the lang_region
registry. FLORES/OPUS are language-keyed, so several lang_regions may collapse
onto one code (e.g. Spanish_* -> spa_Latn / es). Each unique code is downloaded
once and replicated into every lang_region that maps to it.

KEY DIFFERENCE from scripts/02_collect_parallel.py:
  - Output format is .jsonl (not .txt).
  - Each line: {"text": <sentence>, "translation_group_id": <int>}
  - FLORES: translation_group_id = row index in the dev-split dataset
    (assigned from the original dataset position before length filtering,
    so IDs stay aligned across all FLORES language pairs).
  - OPUS-100: translation_group_id = running accept-index (0-based).

Output structure under raw_dir:
  parallel/
    flores/<lang_region>.jsonl  + manifest.json
    opus100/<lang_region>.jsonl + manifest.json
    manifest.json
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List

from src.shared_utils.io import setup_logging, save_jsonl, save_json, ensure_dir
from src.shared_utils.registry import load_registry

logger = setup_logging("collect_parallel")

FLORES_SOURCE = "eng_Latn"   # English is always the source side


def _filter(text: str, min_len: int, max_len: int) -> bool:
    return bool(text) and min_len <= len(text) <= max_len


# -- FLORES-200 ----------------------------------------------------------------

def collect_flores_by_code(
    cfg: dict, codes: set
) -> Dict[str, List[Dict]]:
    """
    Download FLORES-200 once per unique target code.
    Returns {flores_code: [{"text": str, "translation_group_id": int}]}.

    translation_group_id is the row index within the dev split (0-based),
    assigned BEFORE length filtering so it stays aligned across language pairs.
    """
    from datasets import load_dataset  # type: ignore

    n = cfg["data"]["parallel_samples_per_lang"]
    min_len = cfg["data"]["min_sentence_length"]
    max_len = cfg["data"]["max_sentence_length"]
    split = "dev"

    result: Dict[str, List[Dict]] = {}
    en_records: List[Dict] = []

    for code in sorted(c for c in codes if c and c != FLORES_SOURCE):
        pair = f"{FLORES_SOURCE}-{code}"
        logger.info(f"  FLORES pair: {pair}")
        try:
            ds = load_dataset(
                "facebook/flores",
                name=pair,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            logger.warning(f"    Could not load {pair}: {e}")
            continue

        # Collect English side once (row-index aligned).
        if not en_records:
            field = f"sentence_{FLORES_SOURCE}"
            for row_idx, item in enumerate(ds[split]):
                t = item.get(field, "")
                if _filter(t, min_len, max_len):
                    en_records.append({
                        "text": t.strip(),
                        "translation_group_id": row_idx,
                    })
                if len(en_records) >= n:
                    break

        # Collect target side; row_idx from the original dataset position.
        tgt_field = f"sentence_{code}"
        tgt: List[Dict] = []
        for row_idx, item in enumerate(ds[split]):
            t = item.get(tgt_field, "")
            if _filter(t, min_len, max_len):
                tgt.append({
                    "text": t.strip(),
                    "translation_group_id": row_idx,
                })
            if len(tgt) >= n:
                break
        result[code] = tgt
        logger.info(f"    [{code}] {len(tgt)} records")

    if FLORES_SOURCE in codes:
        result[FLORES_SOURCE] = en_records
    return result


# -- OPUS-100 ------------------------------------------------------------------

def collect_opus_by_code(
    cfg: dict, codes: set
) -> Dict[str, List[Dict]]:
    """
    Download OPUS-100 once per unique language code.
    Returns {opus_code: [{"text": str, "translation_group_id": int}]}.

    translation_group_id is the running accept-index (0-based) within the
    collected sentences for that language (no cross-language alignment).
    """
    from datasets import load_dataset  # type: ignore

    n = cfg["data"]["parallel_samples_per_lang"]
    min_len = cfg["data"]["min_sentence_length"]
    max_len = cfg["data"]["max_sentence_length"]

    result: Dict[str, List[Dict]] = {}
    en_all: List[Dict] = []

    for code in sorted(c for c in codes if c and c != "en"):
        pair_sorted = "-".join(sorted(["en", code]))
        logger.info(f"  OPUS-100 pair: {pair_sorted}")
        ds = None
        for split in ("test", "dev", "train"):
            try:
                ds = load_dataset(
                    "opus100",
                    pair_sorted,
                    split=split,
                    token=os.environ.get("HF_TOKEN"),
                )
                break
            except Exception as e:
                if "split" in str(e).lower():
                    continue
                logger.warning(f"    Could not load {pair_sorted}: {e}")
                break
        if ds is None:
            logger.warning(f"    Could not load {pair_sorted} (no usable split)")
            continue

        sents: List[Dict] = []
        en_sents: List[Dict] = []
        accept_idx = 0
        for ex in ds:
            tr = ex.get("translation", {})
            t = tr.get(code, "")
            en_t = tr.get("en", "")
            if _filter(t, min_len, max_len):
                sents.append({"text": t.strip(), "translation_group_id": accept_idx})
                if _filter(en_t, min_len, max_len):
                    en_sents.append({"text": en_t.strip(), "translation_group_id": accept_idx})
                accept_idx += 1
            if len(sents) >= n:
                break
        result[code] = sents
        en_all.extend(en_sents)
        logger.info(f"    [{code}] {len(sents)} records")

    if "en" in codes:
        seen, en_unique = set(), []
        for rec in en_all:
            if rec["text"] not in seen:
                seen.add(rec["text"])
                en_unique.append(rec)
        result["en"] = en_unique[:n]
    return result


def _save_by_region(
    registry: list,
    code_field: str,
    by_code: Dict[str, List[Dict]],
    out_dir: str,
    dataset_name: str,
) -> dict:
    """Replicate each code's records into every lang_region .jsonl that maps to it."""
    ensure_dir(out_dir)
    manifest = {"dataset": dataset_name, "lang_regions": {}}
    shared = defaultdict(list)
    for r in registry:
        shared[r.get(code_field)].append(r["key"])

    for r in registry:
        key = r["key"]
        code = r.get(code_field)
        if not code or code not in by_code:
            continue
        records = by_code[code]
        path = os.path.join(out_dir, f"{key}.jsonl")
        save_jsonl(records, path)
        manifest["lang_regions"][key] = {
            "path": path,
            "code": code,
            "num_records": len(records),
            "shared_with": [k for k in shared[code] if k != key],
        }
    save_json(manifest, os.path.join(out_dir, "manifest.json"))
    return manifest


def collect(cfg: dict) -> None:
    """Collect FLORES-200 and OPUS-100 parallel data into raw_dir/parallel/."""
    registry = load_registry(cfg)
    raw_dir = cfg["paths"]["raw_dir"]
    parallel_dir = os.path.join(raw_dir, "parallel")

    flores_codes = {r.get("flores") for r in registry if r.get("flores")}
    opus_codes = {r.get("opus") for r in registry if r.get("opus")}
    logger.info(
        f"{len(flores_codes)} unique FLORES codes, {len(opus_codes)} unique OPUS codes"
    )

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
        {
            "datasets": ["flores", "opus100"],
            "flores": flores_manifest,
            "opus100": opus_manifest,
        },
        os.path.join(parallel_dir, "manifest.json"),
    )
    logger.info(
        f"Parallel data complete: "
        f"flores {len(flores_manifest['lang_regions'])} regions, "
        f"opus100 {len(opus_manifest['lang_regions'])} regions."
    )
