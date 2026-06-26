#!/usr/bin/env python3
"""
05_analyze_vectors.py
======================
Comprehensive analysis of language, topic, and culture vectors.

Analyses performed:
  1. Within-dataset language vector similarity
  2. **Cross-dataset consistency** -- do FLORES and OPUS-100 yield the same
     language vectors? (cosine similarity per language across datasets)
  3. Topic vector similarity
  4. Cross-similarity: language vs. topic vectors (orthogonality test)
  5. Decomposition test: culture ~ language + topic?
  6. PCA projections
  7. Layer-by-layer progression of all metrics

Usage:
    python vector_analysis/scripts/05_analyze_vectors.py [--config configs/config.yaml]
"""

import argparse
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import load_config, ensure_dirs, setup_logging, save_json, load_json
from shared_utils.vectors import load_vectors, cosine_similarity, cosine_similarity_matrix, subspace_angle

logger = setup_logging("analyze_vectors")


# -- Helpers -------------------------------------------------------------------

def analyze_within_similarity(vectors: dict, label: str) -> pd.DataFrame:
    """Pairwise cosine similarity within a set of vectors. Skips NaN vectors."""
    clean = {k: v for k, v in vectors.items() if np.isfinite(v).all()}
    mat, labels = cosine_similarity_matrix(clean)
    rows = []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            rows.append({"vec_a": li, "vec_b": lj, "cosine_sim": mat[i, j], "group": label})
    return pd.DataFrame(rows)


def analyze_cross_similarity(lang_vectors: dict, topic_vectors: dict) -> pd.DataFrame:
    """Cross cosine similarity between language and topic vectors. Skips NaN vectors."""
    rows = []
    for lname, lvec in lang_vectors.items():
        if not np.isfinite(lvec).all():
            continue
        for tname, tvec in topic_vectors.items():
            if not np.isfinite(tvec).all():
                continue
            sim = cosine_similarity(lvec, tvec)
            rows.append({
                "language": lname, "topic": tname,
                "cosine_sim": sim, "abs_cosine": abs(sim),
            })
    return pd.DataFrame(rows)


def decomposition_analysis(lang_vectors, topic_vectors, culture_vectors) -> pd.DataFrame:
    """Test whether culture ~ language + topic. Skips NaN vectors."""
    rows = []
    for culture_key, culture_vec in culture_vectors.items():
        if not np.isfinite(culture_vec).all():
            continue
        parts = culture_key.replace("culture_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        topic, lang = parts

        lang_key = f"lang_{lang}"
        topic_key = f"topic_{topic}"
        if lang_key not in lang_vectors or topic_key not in topic_vectors:
            continue

        lang_vec = lang_vectors[lang_key]
        topic_vec = topic_vectors[topic_key]
        if not np.isfinite(lang_vec).all() or not np.isfinite(topic_vec).all():
            continue
        composite = lang_vec + topic_vec
        composite_norm = composite / (np.linalg.norm(composite) + 1e-10)

        sim_composite = cosine_similarity(culture_vec, composite_norm)
        sim_lang_only = cosine_similarity(culture_vec, lang_vec)
        sim_topic_only = cosine_similarity(culture_vec, topic_vec)

        residual = culture_vec - composite_norm
        residual_magnitude = np.linalg.norm(residual)

        rows.append({
            "topic": topic, "language": lang,
            "sim_composite": sim_composite,
            "sim_lang_only": sim_lang_only,
            "sim_topic_only": sim_topic_only,
            "residual_norm": residual_magnitude,
        })
    return pd.DataFrame(rows)


def pca_projection(lang_vectors, topic_vectors, culture_vectors, n_components=3) -> dict:
    """Project all vectors into shared PCA space. Skips any NaN vectors."""
    all_vecs, all_labels, all_types = [], [], []

    for name, vec in lang_vectors.items():
        if not np.isfinite(vec).all():
            continue
        all_vecs.append(vec)
        all_labels.append(name.replace("lang_", ""))
        all_types.append("language")
    for name, vec in topic_vectors.items():
        if not np.isfinite(vec).all():
            continue
        all_vecs.append(vec)
        all_labels.append(name.replace("topic_", ""))
        all_types.append("topic")
    for name, vec in culture_vectors.items():
        if not np.isfinite(vec).all():
            continue
        all_vecs.append(vec)
        all_labels.append(name.replace("culture_", ""))
        all_types.append("culture")

    if len(all_vecs) < 2:
        return {}

    X = np.stack(all_vecs, axis=0)
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    coords = pca.fit_transform(X)

    return {
        "coordinates": coords.tolist(),
        "labels": all_labels,
        "types": all_types,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }


# ==============================================================================
# Cross-dataset consistency analysis
# ==============================================================================

def analyze_cross_dataset_consistency(
    lang_vec_dir: str,
    dataset_names: list,
    layers: list,
) -> pd.DataFrame:
    """
    For each language and layer, compute cosine similarity of the language
    vector computed from dataset A vs. dataset B.

    A high similarity (-> 1.0) means the language direction is a robust
    property of the model rather than an artifact of a specific corpus.
    """
    if len(dataset_names) < 2:
        logger.warning("Need >= 2 datasets for cross-dataset consistency. Skipping.")
        return pd.DataFrame()

    rows = []

    for layer in layers:
        # Load vectors for each dataset at this layer
        ds_vectors = {}
        for ds_name in dataset_names:
            path = os.path.join(lang_vec_dir, ds_name, f"layer_{layer:03d}.npz")
            if os.path.exists(path):
                ds_vectors[ds_name] = load_vectors(path)

        if len(ds_vectors) < 2:
            continue

        # Pairwise dataset comparison
        ds_list = sorted(ds_vectors.keys())
        for i in range(len(ds_list)):
            for j in range(i + 1, len(ds_list)):
                ds_a, ds_b = ds_list[i], ds_list[j]
                vecs_a = ds_vectors[ds_a]
                vecs_b = ds_vectors[ds_b]

                # Find common languages
                langs_a = {k.replace("lang_", "") for k in vecs_a}
                langs_b = {k.replace("lang_", "") for k in vecs_b}
                common_langs = sorted(langs_a & langs_b)

                for lang in common_langs:
                    va = vecs_a[f"lang_{lang}"]
                    vb = vecs_b[f"lang_{lang}"]
                    if not np.isfinite(va).all() or not np.isfinite(vb).all():
                        continue
                    sim = cosine_similarity(va, vb)
                    angle = subspace_angle(va, vb)

                    rows.append({
                        "layer": layer,
                        "language": lang,
                        "dataset_a": ds_a,
                        "dataset_b": ds_b,
                        "cosine_sim": sim,
                        "angle_degrees": angle,
                    })

    return pd.DataFrame(rows)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze vectors")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    lang_vec_dir = cfg["paths"]["va_language_vector_dir"]
    topic_dir = cfg["paths"]["va_topic_vector_dir"]
    out_dir = cfg["paths"]["va_analysis_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # -- Load metadata ---------------------------------------------------------

    lang_meta = load_json(os.path.join(lang_vec_dir, "metadata.json"))
    topic_meta = load_json(os.path.join(topic_dir, "metadata.json"))

    layers = lang_meta.get("layers", [])
    dataset_names = lang_meta.get("datasets", ["flores", "opus100"])
    # Use first dataset as "primary" for topic/culture cross-analysis
    primary_ds = dataset_names[0]

    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"Primary dataset for cross-analysis: {primary_ds}")
    logger.info(f"Analyzing {len(layers)} layers...")

    # ==========================================================================
    # Analysis 0: CROSS-DATASET CONSISTENCY
    # ==========================================================================

    logger.info("\n=== Cross-Dataset Consistency ===")
    df_consistency = analyze_cross_dataset_consistency(
        lang_vec_dir, dataset_names, layers,
    )
    if not df_consistency.empty:
        df_consistency.to_csv(
            os.path.join(out_dir, "cross_dataset_consistency.csv"), index=False
        )
        mean_sim = df_consistency["cosine_sim"].mean()
        std_sim = df_consistency["cosine_sim"].std()
        logger.info(f"  Mean cross-dataset cos sim: {mean_sim:.4f} (+/-{std_sim:.4f})")

        # Per-language breakdown
        per_lang = df_consistency.groupby("language")["cosine_sim"].agg(["mean", "std"])
        logger.info("  Per-language consistency:")
        for lang, row in per_lang.iterrows():
            logger.info(f"    {lang}: {row['mean']:.4f} (+/-{row['std']:.4f})")
    else:
        logger.info("  Skipped (need >=2 datasets).")

    # ==========================================================================
    # Per-layer analyses (using primary dataset for language vectors)
    # ==========================================================================

    all_within_lang = []
    all_within_topic = []
    all_cross = []
    all_decomp = []
    all_pca = {}
    layer_summary = []

    # Also accumulate per-dataset within-language similarity
    all_within_lang_by_ds = {ds: [] for ds in dataset_names}

    for layer in layers:
        logger.info(f"Layer {layer}...")

        # -- Load primary language vectors -------------------------------------
        primary_lang_path = os.path.join(
            lang_vec_dir, primary_ds, f"layer_{layer:03d}.npz"
        )
        if not os.path.exists(primary_lang_path):
            logger.warning(f"  No primary language vectors at layer {layer}. Skipping.")
            continue
        lang_vecs = load_vectors(primary_lang_path)

        # -- Load topic / culture vectors --------------------------------------
        topic_path = os.path.join(topic_dir, f"topic_layer_{layer:03d}.npz")
        culture_path = os.path.join(topic_dir, f"culture_layer_{layer:03d}.npz")
        topic_vecs = load_vectors(topic_path) if os.path.exists(topic_path) else {}
        culture_vecs = load_vectors(culture_path) if os.path.exists(culture_path) else {}

        # -- 1. Within-language similarity (per dataset) -----------------------
        for ds_name in dataset_names:
            ds_lang_path = os.path.join(
                lang_vec_dir, ds_name, f"layer_{layer:03d}.npz"
            )
            if not os.path.exists(ds_lang_path):
                continue
            ds_lang_vecs = load_vectors(ds_lang_path)
            df_wl = analyze_within_similarity(ds_lang_vecs, f"language_{ds_name}")
            df_wl["layer"] = layer
            df_wl["dataset"] = ds_name
            all_within_lang_by_ds[ds_name].append(df_wl)

        # Primary dataset for the rest
        df_lang = analyze_within_similarity(lang_vecs, "language")
        df_lang["layer"] = layer
        all_within_lang.append(df_lang)

        # -- 2. Within-topic similarity ----------------------------------------
        if topic_vecs:
            df_topic = analyze_within_similarity(topic_vecs, "topic")
            df_topic["layer"] = layer
            all_within_topic.append(df_topic)

        # -- 3. Cross similarity (orthogonality test) --------------------------
        if topic_vecs:
            df_cross = analyze_cross_similarity(lang_vecs, topic_vecs)
            df_cross["layer"] = layer
            all_cross.append(df_cross)

        # -- 4. Decomposition analysis -----------------------------------------
        if topic_vecs and culture_vecs:
            df_decomp = decomposition_analysis(lang_vecs, topic_vecs, culture_vecs)
            df_decomp["layer"] = layer
            all_decomp.append(df_decomp)

        # -- 5. PCA ------------------------------------------------------------
        if topic_vecs and culture_vecs:
            pca_result = pca_projection(
                lang_vecs, topic_vecs, culture_vecs,
                n_components=cfg["analysis"]["n_pca_components"],
            )
            all_pca[layer] = pca_result

        # -- Layer summary -----------------------------------------------------
        summary_row = {"layer": layer}

        if not df_lang.empty and "vec_a" in df_lang.columns:
            lang_off = df_lang[df_lang["vec_a"] != df_lang["vec_b"]]["cosine_sim"]
            summary_row["mean_lang_within_sim"] = float(lang_off.abs().mean()) if len(lang_off) else 0
        else:
            summary_row["mean_lang_within_sim"] = 0

        if all_within_topic and not all_within_topic[-1].empty and "vec_a" in all_within_topic[-1].columns:
            df_t = all_within_topic[-1]
            topic_off = df_t[df_t["vec_a"] != df_t["vec_b"]]["cosine_sim"]
            summary_row["mean_topic_within_sim"] = float(topic_off.abs().mean()) if len(topic_off) else 0
        else:
            summary_row["mean_topic_within_sim"] = 0

        if all_cross and not all_cross[-1].empty and "abs_cosine" in all_cross[-1].columns:
            summary_row["mean_cross_sim"] = float(all_cross[-1]["abs_cosine"].mean())
        else:
            summary_row["mean_cross_sim"] = 0

        if all_decomp and not all_decomp[-1].empty and "sim_composite" in all_decomp[-1].columns:
            summary_row["mean_decomp_sim"] = float(all_decomp[-1]["sim_composite"].mean())
            summary_row["mean_residual_norm"] = float(all_decomp[-1]["residual_norm"].mean())
        else:
            summary_row["mean_decomp_sim"] = 0
            summary_row["mean_residual_norm"] = 0

        # Cross-dataset consistency at this layer
        if not df_consistency.empty:
            layer_consist = df_consistency[df_consistency["layer"] == layer]
            summary_row["mean_cross_dataset_sim"] = float(layer_consist["cosine_sim"].mean()) \
                if len(layer_consist) else 0
        else:
            summary_row["mean_cross_dataset_sim"] = 0

        layer_summary.append(summary_row)

    # ==========================================================================
    # Save all results
    # ==========================================================================

    logger.info("\nSaving analysis results...")

    if all_within_lang:
        df_wl = pd.concat(all_within_lang)
        if not df_wl.empty:
            df_wl.to_csv(os.path.join(out_dir, "within_language_similarity.csv"), index=False)

    # Per-dataset within-language similarity
    for ds_name in dataset_names:
        if all_within_lang_by_ds[ds_name]:
            df_ds = pd.concat(all_within_lang_by_ds[ds_name])
            if not df_ds.empty:
                df_ds.to_csv(
                    os.path.join(out_dir, f"within_language_similarity_{ds_name}.csv"),
                    index=False,
                )

    if all_within_topic:
        df_wt = pd.concat(all_within_topic)
        if not df_wt.empty:
            df_wt.to_csv(os.path.join(out_dir, "within_topic_similarity.csv"), index=False)
    if all_cross:
        df_cr = pd.concat(all_cross)
        if not df_cr.empty:
            df_cr.to_csv(os.path.join(out_dir, "cross_similarity.csv"), index=False)
    if all_decomp:
        df_dc = pd.concat(all_decomp)
        if not df_dc.empty:
            df_dc.to_csv(os.path.join(out_dir, "decomposition_analysis.csv"), index=False)

    pd.DataFrame(layer_summary).to_csv(
        os.path.join(out_dir, "layer_summary.csv"), index=False
    )
    save_json(all_pca, os.path.join(out_dir, "pca_projections.json"))

    # -- Print final summary ---------------------------------------------------

    logger.info("\n" + "=" * 60)
    logger.info("  ANALYSIS SUMMARY  (averaged across layers)")
    logger.info("=" * 60)
    df_summary = pd.DataFrame(layer_summary)
    for col in df_summary.columns:
        if col != "layer":
            logger.info(f"  {col}: {df_summary[col].mean():.4f} (+/-{df_summary[col].std():.4f})")

    if not df_consistency.empty:
        logger.info("\n=== Key Finding: Cross-Dataset Consistency ===")
        logger.info(f"  Mean cos(v_lang^FLORES, v_lang^OPUS): "
                     f"{df_consistency['cosine_sim'].mean():.4f}")
        logger.info(f"  -> {'Highly consistent' if df_consistency['cosine_sim'].mean() > 0.8 else 'Moderately consistent' if df_consistency['cosine_sim'].mean() > 0.5 else 'Low consistency'}")

    if all_cross:
        df_all_cross = pd.concat(all_cross)
        if not df_all_cross.empty and "abs_cosine" in df_all_cross.columns:
            logger.info("\n=== Key Finding: Language perpendicular to Topic? ===")
            mean_abs = df_all_cross["abs_cosine"].mean()
            logger.info(f"  Mean |cos(lang, topic)|: {mean_abs:.4f}")
            logger.info(f"  -> {'Near-orthogonal' if mean_abs < 0.2 else 'NOT orthogonal'}")

    if all_decomp:
        df_all_decomp = pd.concat(all_decomp)
        if not df_all_decomp.empty and "sim_composite" in df_all_decomp.columns:
            logger.info("\n=== Key Finding: Culture ~ Lang + Topic? ===")
            mean_comp = df_all_decomp["sim_composite"].mean()
            logger.info(f"  Mean cos(culture, lang+topic): {mean_comp:.4f}")
            logger.info(f"  -> {'Supports decomposition' if mean_comp > 0.5 else 'Weak decomposition'}")

    logger.info("\nAnalysis complete.")


if __name__ == "__main__":
    main()
