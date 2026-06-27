"""
src/modules/collect/sib200.py
==============================
Collect SIB-200 (Sentence Inference Benchmark, 200+ languages) text data.

SIB-200 is a FLORES-derived topic-classification dataset published on HuggingFace
as "Davlan/sib200". Each language config has fields:
  - "text"     : sentence (str)
  - "category" : topic label (str), e.g. "science/technology", "health", ...

Collection strategy:
  For each registry language that has a FLORES code, attempt to load the
  corresponding SIB-200 language config. The SIB-200 lang code is the FLORES
  code with '_' replaced by '-' and lowercased (e.g. "arz_Arab" -> "arz-Arab").
  We try both "all" split and individual split names ("train", "test", "dev").
  Languages without SIB-200 coverage are skipped (try/except + log).

Output structure under raw_dir:
  sib200/
    <category>/<lang_code>.txt   (one sentence per line)
    manifest.json                 {lang_code -> {category -> {path, n}}}

Assumption: dataset id = "Davlan/sib200", fields = "text" and "category".
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List

from src.shared_utils.io import setup_logging, save_sentences, save_json, ensure_dir
from src.shared_utils.registry import load_registry

logger = setup_logging("collect_sib200")


def _flores_to_sib_code(flores_code: str) -> str:
    """
    Convert a FLORES-200 code to the SIB-200 config name.
    E.g. "arz_Arab" -> "arz-Arab", "zho_Hans" -> "zho-Hans".
    """
    return flores_code.replace("_", "-")


def collect_language(sib_code: str) -> Dict[str, List[str]]:
    """
    Load SIB-200 for one language config, returning {category: [sentences]}.
    Raises on any failure (caller wraps in try/except).
    """
    from datasets import load_dataset  # type: ignore

    ds = None
    # Try the combined "all" split first; fall back to individual splits.
    for split in ("test", "train", "dev", "all"):
        try:
            ds = load_dataset(
                "Davlan/sib200",
                sib_code,
                split=split,
                trust_remote_code=True,
            )
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"No usable split found for SIB-200 config '{sib_code}'")

    by_category: Dict[str, List[str]] = defaultdict(list)
    for item in ds:
        text = item.get("text", "").strip()
        category = item.get("category", "").strip()
        if text and category:
            by_category[category].append(text)
    return dict(by_category)


def collect(cfg: dict) -> None:
    """Collect SIB-200 topic sentences into raw_dir/sib200/."""
    registry = load_registry(cfg)
    raw_dir = cfg["paths"]["raw_dir"]
    out_dir = os.path.join(raw_dir, "sib200")

    # Deduplicate: collect once per unique FLORES code.
    seen_flores: Dict[str, str] = {}   # flores_code -> sib_code
    for r in registry:
        flores = r.get("flores")
        if flores and flores not in seen_flores:
            seen_flores[flores] = _flores_to_sib_code(flores)

    logger.info(
        f"Attempting SIB-200 for {len(seen_flores)} unique FLORES codes "
        f"across {len(registry)} registry entries."
    )

    # manifest: sib_code -> {category -> {"path": ..., "n": ...}}
    manifest: Dict[str, Dict] = {}
    # collected: flores_code -> {category: [sentences]}
    collected: Dict[str, Dict[str, List[str]]] = {}

    for flores_code, sib_code in sorted(seen_flores.items()):
        try:
            by_cat = collect_language(sib_code)
        except Exception as e:
            logger.warning(f"  Skipping '{sib_code}' ('{flores_code}'): {e}")
            continue

        total = sum(len(v) for v in by_cat.values())
        logger.info(
            f"  [{sib_code}] {len(by_cat)} categories, {total} sentences"
        )
        collected[flores_code] = by_cat

        # Write files per category.
        lang_manifest: Dict[str, Dict] = {}
        for category, sentences in by_cat.items():
            cat_dir = os.path.join(out_dir, category)
            ensure_dir(cat_dir)
            path = os.path.join(cat_dir, f"{sib_code}.txt")
            save_sentences(sentences, path)
            lang_manifest[category] = {"path": path, "n": len(sentences)}
        manifest[sib_code] = lang_manifest

    # Also write per lang_region (replicate for regions sharing a flores code).
    region_manifest: Dict[str, Dict] = {}
    for r in registry:
        flores = r.get("flores")
        if not flores or flores not in collected:
            continue
        key = r["key"]
        sib_code = seen_flores[flores]
        by_cat = collected[flores]
        region_manifest[key] = {"sib_code": sib_code, "categories": list(by_cat.keys())}

    save_json(
        {"by_sib_code": manifest, "by_lang_region": region_manifest},
        os.path.join(out_dir, "manifest.json"),
    )
    logger.info(
        f"SIB-200 collection complete: "
        f"{len(manifest)} language configs written to {out_dir}"
    )
