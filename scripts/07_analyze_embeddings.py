#!/usr/bin/env python3
"""
07_analyze_embeddings.py
========================
Structure (embedding layer) + depth (embed -> deep layers) analysis of the
per-region/per-topic embedding store written by 04_compute_vectors.py.

Outputs CSVs to va_analysis_dir and figures to va_plot_dir.

Usage: python 07_analyze_embeddings.py [--config configs/riddles_config.yaml]
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import load_config, ensure_dirs, setup_logging, save_json
from shared_utils.embeddings import (
    load_embedding_store, structure_score, depth_structure, cluster_embeddings,
)
from shared_utils.vectors import cosine_similarity_matrix

logger = setup_logging("analyze_embeddings")

REGION_SOURCES = ["flores", "opus100", "puzzles_original", "puzzles_translation"]


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def analyze_source(name, store_dir, analysis_dir, plot_dir):
    if not os.path.isdir(store_dir):
        logger.warning(f"  [{name}] no embedding store; skipping.")
        return None
    by_layer, meta = load_embedding_store(store_dir)
    groups = meta["groups"]
    emb = by_layer.get("embed")
    if emb is None or len(emb) < 3:
        logger.warning(f"  [{name}] <3 keys at embed layer; skipping structure.")
        return None

    # Similarity heatmap (embed layer)
    mat, labels = cosine_similarity_matrix(emb)
    _write_csv(os.path.join(analysis_dir, f"embedding_similarity_{name}.csv"),
               ["vec_a", "vec_b", "cosine_sim"],
               [[labels[i], labels[j], f"{mat[i, j]:.6f}"]
                for i in range(len(labels)) for j in range(len(labels))])
    plt.figure(figsize=(8, 7))
    plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.colorbar(label="cosine"); plt.title(f"Embedding similarity: {name}")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(plot_dir, f"emb_fig1_similarity_{name}.{ext}"))
    plt.close()

    # PCA scatter colored by group
    X = np.stack([emb[k] for k in labels])
    coords = PCA(n_components=2).fit(X)
    xy = coords.transform(X)
    glist = [groups.get(k, "UNKNOWN") for k in labels]
    save_json({"labels": labels, "groups": glist,
               "coordinates": xy.tolist(),
               "explained_variance": coords.explained_variance_ratio_.tolist()},
              os.path.join(analysis_dir, f"embedding_pca_{name}.json"))
    plt.figure(figsize=(8, 7))
    for g in sorted(set(glist)):
        idx = [i for i, gg in enumerate(glist) if gg == g]
        plt.scatter(xy[idx, 0], xy[idx, 1], label=g, s=30)
    plt.legend(fontsize=6, ncol=2); plt.title(f"Embedding PCA: {name}")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(plot_dir, f"emb_fig2_pca_{name}.{ext}"))
    plt.close()

    # Clustering vs base_language
    n_groups = len(set(groups.values()))
    clu = cluster_embeddings(emb, n_clusters=max(2, n_groups), group_map=groups)
    _write_csv(os.path.join(analysis_dir, f"embedding_clusters_{name}.csv"),
               ["key", "cluster", "group"],
               [[k, clu["assignments"][k], groups.get(k, "UNKNOWN")] for k in labels])

    # Dendrogram (fig3)
    plt.figure(figsize=(9, 5))
    dendrogram(np.asarray(clu["linkage"]), labels=clu["labels"], leaf_font_size=6)
    plt.title(f"Embedding clustering: {name}"); plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(plot_dir, f"emb_fig3_dendrogram_{name}.{ext}"))
    plt.close()

    s = structure_score(emb, groups)
    return {"source": name, "silhouette": s["silhouette"],
            "within_minus_cross": s["within_minus_cross"],
            "ari": clu["ari"], "nmi": clu["nmi"], "n": s["n"]}


def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings")
    parser.add_argument("--config", default="configs/riddles_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    emb_dir = cfg["paths"]["va_embeddings_dir"]
    analysis_dir = cfg["paths"]["va_analysis_dir"]
    plot_dir = cfg["paths"]["va_plot_dir"]
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    summary = []
    depth_rows = []
    for name in REGION_SOURCES:
        store_dir = os.path.join(emb_dir, name)
        row = analyze_source(name, store_dir, analysis_dir, plot_dir)
        if row:
            summary.append(row)
            by_layer, meta = load_embedding_store(store_dir)
            for d in depth_structure(by_layer, meta["groups"]):
                depth_rows.append([name, d["layer"], d["silhouette"],
                                   d["within_minus_cross"], d["n"]])

    if summary:
        _write_csv(os.path.join(analysis_dir, "embedding_structure_summary.csv"),
                   ["source", "silhouette", "within_minus_cross", "ari", "nmi", "n"],
                   [[r["source"], r["silhouette"], r["within_minus_cross"],
                     r["ari"], r["nmi"], r["n"]] for r in summary])
    if depth_rows:
        _write_csv(os.path.join(analysis_dir, "depth_structure.csv"),
                   ["source", "layer", "silhouette", "within_minus_cross", "n"], depth_rows)
        # Depth figure: silhouette vs layer, one line per source
        plt.figure(figsize=(9, 6))
        for name in sorted(set(r[0] for r in depth_rows)):
            rows = [r for r in depth_rows if r[0] == name and r[2] is not None]
            xs = [r[1] for r in rows]; ys = [r[2] for r in rows]
            plt.plot(range(len(xs)), ys, marker="o", label=name)
            plt.xticks(range(len(xs)), xs, rotation=90, fontsize=6)
        plt.ylabel("silhouette (group=base_language)"); plt.xlabel("layer")
        plt.legend(); plt.title("Structure vs depth"); plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(plot_dir, f"emb_fig4_depth_structure.{ext}"))
        plt.close()

    logger.info(f"Embedding analysis complete. Sources analyzed: {[r['source'] for r in summary]}")


if __name__ == "__main__":
    main()
