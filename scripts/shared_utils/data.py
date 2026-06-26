"""
shared_utils/data.py
I/O helpers, config loading, logging setup.
"""

import os
import json
import logging
from typing import List

import yaml


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    for key, path in cfg.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_sentences(sentences: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.strip() + "\n")


def load_sentences(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_jsonl(rows, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_registry(cfg: dict) -> List[dict]:
    """Return the lang_regions registry list from the config."""
    return cfg.get("lang_regions", [])


# Cache the live FLORES-200 code set so validation hits the network at most once.
_FLORES_CODES = None


def _fetch_flores_codes():
    global _FLORES_CODES
    if _FLORES_CODES is not None:
        return _FLORES_CODES
    import re
    import urllib.request
    url = "https://raw.githubusercontent.com/facebookresearch/flores/main/flores200/README.md"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "flores-check/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            txt = resp.read().decode("utf-8", "ignore")
        _FLORES_CODES = set(re.findall(r"\b([a-z]{2,4}_[A-Z][a-z]{3})\b", txt))
    except Exception:
        _FLORES_CODES = set()  # offline: skip validation gracefully
    return _FLORES_CODES


def validate_flores_codes(cfg: dict, logger=None) -> List[str]:
    """
    Warn about any non-null `flores` codes in the registry that are absent from the
    live FLORES-200 language set. Returns the list of offending codes (empty if all
    valid, or if the code list could not be fetched).
    """
    codes = _fetch_flores_codes()
    if not codes:
        if logger:
            logger.warning("Could not fetch FLORES-200 code list; skipping validation.")
        return []
    bad = []
    for r in load_registry(cfg):
        fc = r.get("flores")
        if fc and fc not in codes:
            bad.append(fc)
            if logger:
                logger.warning(f"FLORES code '{fc}' for {r['key']} not in FLORES-200.")
    return bad
