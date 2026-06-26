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
from shared_utils.vectors import cosine_similarity, pairwise_distance_matrix
from shared_utils.clustering import hierarchical_cluster, cluster_agreement_scores


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


def structure_score(emb_by_key, group_map):
    keys = [k for k in sorted(emb_by_key) if k in group_map]
    if len(keys) < 3:
        return {"silhouette": None, "within_minus_cross": None, "n": len(keys)}
    X = np.stack([np.asarray(emb_by_key[k], dtype=float) for k in keys])
    groups = [group_map[k] for k in keys]

    sil = None
    if len(set(groups)) >= 2 and len(keys) > len(set(groups)):
        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(X, groups, metric="cosine"))
        except Exception:
            sil = None

    wi = wc = cr = cc = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = cosine_similarity(X[i], X[j])
            if groups[i] == groups[j]:
                wi += c; wc += 1
            else:
                cr += c; cc += 1
    wmc = (wi / wc - cr / cc) if wc and cc else None
    return {"silhouette": sil, "within_minus_cross": wmc, "n": len(keys)}


def _layer_sort_key(label):
    return (0, -1) if label == "embed" else (1, int(label))


def depth_structure(by_layer, group_map):
    rows = []
    for label in sorted(by_layer, key=_layer_sort_key):
        s = structure_score(by_layer[label], group_map)
        rows.append({"layer": "embed" if label == "embed" else str(label), **s})
    return rows


def cluster_embeddings(emb_by_key, n_clusters, group_map):
    keys = [k for k in sorted(emb_by_key) if k in group_map]
    vecs = {k: np.asarray(emb_by_key[k], dtype=float) for k in keys}
    dm, labels = pairwise_distance_matrix(vecs, metric="cosine")
    res = hierarchical_cluster(dm, labels, n_clusters=min(n_clusters, len(labels)))
    assignments = dict(zip(labels, res["cluster_labels"]))
    true = [group_map[k] for k in labels]
    scores = cluster_agreement_scores([assignments[k] for k in labels], true)
    return {"assignments": assignments, **scores}
