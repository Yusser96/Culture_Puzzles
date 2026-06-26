"""
shared_utils/embeddings.py
Embedding-store I/O and group maps for embedding-layer analysis.

The store holds RAW mean representations (not normalized DiffMean directions):
  <store_dir>/layer_embed.npz, layer_000.npz, ...   keys = "<prefix><key>"
  <store_dir>/metadata.json                          layers, keys, groups
"""

import os
from typing import Dict, List

import numpy as np

from shared_utils.data import save_json, load_json


def _layer_label(layer) -> str:
    return "embed" if layer == "embed" else f"{int(layer):03d}"


def build_group_map(cfg: dict, source: str, keys: List[str]) -> Dict[str, str]:
    """Map each key to its grouping label: base_language for region sources,
    the topic for 'topics'/'culture' sources."""
    if source in ("topics", "culture"):
        out = {}
        for k in keys:
            out[k] = k.split("_")[0] if source == "culture" else k
        return out
    base = {r["key"]: r["base_language"] for r in cfg.get("lang_regions", [])}
    return {k: base.get(k, "UNKNOWN") for k in keys}


def save_embedding_store(out_dir, acts_by_key, layers, key_prefix, group_map):
    """Write per-key MEAN representation at each layer (incl. 'embed')."""
    os.makedirs(out_dir, exist_ok=True)
    for layer in layers:
        arrays = {f"{key_prefix}{k}": acts_by_key[k][layer].mean(axis=0)
                  for k in acts_by_key}
        np.savez(os.path.join(out_dir, f"layer_{_layer_label(layer)}.npz"), **arrays)
    meta = {
        "layers": ["embed" if l == "embed" else int(l) for l in layers],
        "keys": sorted(f"{key_prefix}{k}" for k in acts_by_key),
        "groups": {f"{key_prefix}{k}": group_map.get(k, "UNKNOWN") for k in acts_by_key},
    }
    save_json(meta, os.path.join(out_dir, "metadata.json"))
    return meta


def load_embedding_store(store_dir):
    """Return ({layer_label: {prefixed_key: vec}}, metadata). layer_label is 'embed' or int."""
    meta = load_json(os.path.join(store_dir, "metadata.json"))
    by_layer = {}
    for fn in sorted(os.listdir(store_dir)):
        if not (fn.startswith("layer_") and fn.endswith(".npz")):
            continue
        tag = fn[len("layer_"):-len(".npz")]
        label = "embed" if tag == "embed" else int(tag)
        z = np.load(os.path.join(store_dir, fn))
        by_layer[label] = {k: z[k] for k in z.files}
    return by_layer, meta
