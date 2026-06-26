#!/usr/bin/env python3
"""
06_generate_plots.py
=====================
Generate publication-quality figures for the culture vector analysis.

Figures:
  1. Language vector similarity heatmap (representative layer)
  2. Topic vector similarity heatmap
  3. Cross-similarity heatmap: language x topic
  4. PCA projection scatter
  5. Layer progression plot for all key metrics
  6. Decomposition analysis: culture vs. lang+topic composite
  7. Violin plots of cross-similarity distributions
  8. **Cross-dataset consistency**: FLORES vs OPUS-100 language vectors
  9. **Per-language consistency across layers**

Usage:
    python vector_analysis/scripts/06_generate_plots.py [--config configs/config.yaml]
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import load_config, ensure_dirs, setup_logging

logger = setup_logging("generate_plots")

# -- Style Configuration ------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

LANG_COLORS = {
    "en": "#1f77b4", "fr": "#ff7f0e", "de": "#2ca02c", "ar": "#d62728",
    "zh": "#9467bd", "ja": "#8c564b", "hi": "#e377c2", "es": "#7f7f7f",
    "ru": "#bcbd22", "ko": "#17becf",
}
TOPIC_COLORS = {
    "marriage": "#e41a1c", "religion": "#377eb8",
    "food": "#4daf4a", "festivals": "#984ea3",
}
TYPE_COLORS = {"language": "#1f77b4", "topic": "#ff7f0e", "culture": "#2ca02c"}


def pick_representative_layer(summary_df: pd.DataFrame) -> int:
    """Pick a middle-to-late layer where vectors are most informative."""
    if "mean_decomp_sim" in summary_df.columns:
        idx = summary_df["mean_decomp_sim"].idxmax()
        return int(summary_df.loc[idx, "layer"])
    layers = sorted(summary_df["layer"].tolist())
    return layers[int(len(layers) * 0.75)]


# -- Figure 1: Language Similarity Heatmap -------------------------------------

def plot_language_heatmap(analysis_dir, plot_dir, rep_layer):
    df = pd.read_csv(os.path.join(analysis_dir, "within_language_similarity.csv"))
    df_layer = df[df["layer"] == rep_layer]
    if df_layer.empty:
        return

    labels = sorted(df_layer["vec_a"].unique())
    clean = [l.replace("lang_", "") for l in labels]
    n = len(labels)
    mat = np.zeros((n, n))
    for _, row in df_layer.iterrows():
        i = labels.index(row["vec_a"])
        j = labels.index(row["vec_b"])
        mat[i, j] = row["cosine_sim"]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        mat, xticklabels=clean, yticklabels=clean,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )
    ax.set_title(f"Language Vector Similarity (Layer {rep_layer})")

    path = os.path.join(plot_dir, "fig1_language_similarity_heatmap.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Figure 2: Topic Similarity Heatmap ---------------------------------------

def plot_topic_heatmap(analysis_dir, plot_dir, rep_layer):
    path_csv = os.path.join(analysis_dir, "within_topic_similarity.csv")
    if not os.path.exists(path_csv):
        return
    df = pd.read_csv(path_csv)
    df_layer = df[df["layer"] == rep_layer]
    if df_layer.empty:
        return

    labels = sorted(df_layer["vec_a"].unique())
    clean = [l.replace("topic_", "") for l in labels]
    n = len(labels)
    mat = np.zeros((n, n))
    for _, row in df_layer.iterrows():
        i = labels.index(row["vec_a"])
        j = labels.index(row["vec_b"])
        mat[i, j] = row["cosine_sim"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        mat, xticklabels=clean, yticklabels=clean,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )
    ax.set_title(f"Topic Vector Similarity (Layer {rep_layer})")

    path = os.path.join(plot_dir, "fig2_topic_similarity_heatmap.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Figure 3: Cross-Similarity Heatmap ---------------------------------------

def plot_cross_similarity(analysis_dir, plot_dir, rep_layer):
    path_csv = os.path.join(analysis_dir, "cross_similarity.csv")
    if not os.path.exists(path_csv):
        return
    df = pd.read_csv(path_csv)
    df_layer = df[df["layer"] == rep_layer].copy()
    if df_layer.empty:
        return

    df_layer["language"] = df_layer["language"].str.replace("lang_", "")
    df_layer["topic"] = df_layer["topic"].str.replace("topic_", "")
    pivot = df_layer.pivot(index="language", columns="topic", values="cosine_sim")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        annot=True, fmt=".2f", linewidths=0.5, ax=ax,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )
    ax.set_title(f"Language x Topic Cross-Similarity (Layer {rep_layer})")
    mean_abs = df_layer["abs_cosine"].mean()
    ax.set_xlabel(f"topic\nMean |cos|: {mean_abs:.3f}")

    path = os.path.join(plot_dir, "fig3_cross_similarity.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Figure 4: PCA Projections ------------------------------------------------

def plot_pca_projections(analysis_dir, plot_dir, rep_layer):
    pca_path = os.path.join(analysis_dir, "pca_projections.json")
    if not os.path.exists(pca_path):
        return
    with open(pca_path) as f:
        all_pca = json.load(f)

    key = str(rep_layer)
    if key not in all_pca or not all_pca[key]:
        return

    pca_data = all_pca[key]
    coords = np.array(pca_data["coordinates"])
    labels = pca_data["labels"]
    types = pca_data["types"]
    var_explained = pca_data.get("explained_variance", [0, 0])

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"language": "o", "topic": "s", "culture": "^"}
    sizes = {"language": 120, "topic": 120, "culture": 50}
    alphas = {"language": 1.0, "topic": 1.0, "culture": 0.6}

    for i, (label, vtype) in enumerate(zip(labels, types)):
        color = TYPE_COLORS.get(vtype, "#999999")
        if vtype == "language" and label in LANG_COLORS:
            color = LANG_COLORS[label]
        elif vtype == "topic" and label in TOPIC_COLORS:
            color = TOPIC_COLORS[label]
        elif vtype == "culture":
            parts = label.rsplit("_", 1)
            if len(parts) == 2 and parts[0] in TOPIC_COLORS:
                color = TOPIC_COLORS[parts[0]]

        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, marker=markers.get(vtype, "o"),
            s=sizes.get(vtype, 60), alpha=alphas.get(vtype, 0.8),
            edgecolors="black", linewidths=0.5,
            zorder=3 if vtype != "culture" else 2,
        )
        if vtype != "culture":
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                        fontsize=8, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    legend_elements = [
        Patch(facecolor=TYPE_COLORS["language"], label="Language"),
        Patch(facecolor=TYPE_COLORS["topic"], label="Topic"),
        Patch(facecolor=TYPE_COLORS["culture"], label="Culture (Topic x Lang)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var.)")
    ax.set_title(f"PCA Projection of Steering Vectors (Layer {rep_layer})")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")

    path = os.path.join(plot_dir, "fig4_pca_projections.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Figure 5: Layer Progression -----------------------------------------------

def plot_layer_progression(analysis_dir, plot_dir):
    df = pd.read_csv(os.path.join(analysis_dir, "layer_summary.csv"))

    has_consistency = "mean_cross_dataset_sim" in df.columns and df["mean_cross_dataset_sim"].sum() > 0
    ncols = 3 if has_consistency else 2
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8), sharex=True)
    axes = axes.flatten()

    # (a) Within-similarity
    ax = axes[0]
    ax.plot(df["layer"], df["mean_lang_within_sim"], "o-", label="Language",
            color="#1f77b4", markersize=3)
    ax.plot(df["layer"], df["mean_topic_within_sim"], "s-", label="Topic",
            color="#ff7f0e", markersize=3)
    ax.set_ylabel("Mean |Cosine Similarity|")
    ax.set_title("(a) Within-Group Similarity")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (b) Cross-similarity
    ax = axes[1]
    ax.plot(df["layer"], df["mean_cross_sim"], "D-", color="#2ca02c", markersize=3)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Mean |cos(lang, topic)|")
    ax.set_title("(b) Cross-Similarity (Orthogonality)")
    ax.grid(True, alpha=0.3)

    # (c) Cross-dataset consistency (if available)
    if has_consistency:
        ax = axes[2]
        ax.plot(df["layer"], df["mean_cross_dataset_sim"], "P-",
                color="#d62728", markersize=4)
        ax.axhline(y=0.9, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel("cos(FLORES, OPUS)")
        ax.set_title("(c) Cross-Dataset Consistency")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)

    # (d) Decomposition similarity
    ax = axes[ncols]
    if "mean_decomp_sim" in df.columns:
        ax.plot(df["layer"], df["mean_decomp_sim"], "^-", color="#d62728", markersize=3)
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--", label="threshold")
        ax.legend()
    ax.set_ylabel("cos(culture, lang+topic)")
    ax.set_title(f"({'d' if has_consistency else 'c'}) Culture ~ Lang + Topic?")
    ax.set_xlabel("Layer"); ax.grid(True, alpha=0.3)

    # (e) Residual norm
    ax = axes[ncols + 1]
    if "mean_residual_norm" in df.columns:
        ax.plot(df["layer"], df["mean_residual_norm"], "v-", color="#9467bd", markersize=3)
    ax.set_ylabel("||culture - (lang+topic)||")
    ax.set_title(f"({'e' if has_consistency else 'd'}) Decomposition Residual")
    ax.set_xlabel("Layer"); ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(ncols + 2, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Vector Properties Across Layers", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(plot_dir, "fig5_layer_progression.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Figure 6: Decomposition Detail -------------------------------------------

def plot_decomposition_detail(analysis_dir, plot_dir, rep_layer):
    path_csv = os.path.join(analysis_dir, "decomposition_analysis.csv")
    if not os.path.exists(path_csv):
        return
    df = pd.read_csv(path_csv)
    df_layer = df[df["layer"] == rep_layer].sort_values(["topic", "language"])
    if df_layer.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = [f"{r['topic']}\n{r['language']}" for _, r in df_layer.iterrows()]
    x = np.arange(len(x_labels))
    width = 0.25

    ax.bar(x - width, df_layer["sim_lang_only"], width,
           label="Lang only", color="#1f77b4", alpha=0.8)
    ax.bar(x, df_layer["sim_topic_only"], width,
           label="Topic only", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width, df_layer["sim_composite"], width,
           label="Lang + Topic", color="#2ca02c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Cosine Similarity with Culture Vector")
    ax.set_title(f"Culture Vector Decomposition (Layer {rep_layer})")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y"); ax.axhline(y=0, color="gray", linewidth=0.5)

    path = os.path.join(plot_dir, "fig6_decomposition_detail.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Figure 7: Violin Plot ----------------------------------------------------

def plot_cross_violin(analysis_dir, plot_dir):
    path_csv = os.path.join(analysis_dir, "cross_similarity.csv")
    if not os.path.exists(path_csv):
        return
    df = pd.read_csv(path_csv)
    if df.empty:
        return

    layers = sorted(df["layer"].unique())
    step = max(1, len(layers) // 10)
    selected = layers[::step]
    df_sub = df[df["layer"].isin(selected)]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(data=df_sub, x="layer", y="cosine_sim",
                   color="#2ca02c", alpha=0.7, inner="quartile", ax=ax)
    ax.axhline(y=0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(Language Vector, Topic Vector)")
    ax.set_title("Distribution of Language-Topic Cross-Similarities")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(plot_dir, "fig7_cross_similarity_violin.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# ==============================================================================
# Figures 8 & 9 -- Cross-Dataset Consistency
# ==============================================================================

def plot_cross_dataset_heatmap(analysis_dir, plot_dir, rep_layer):
    """
    Figure 8: Heatmap of cross-dataset consistency per language
    at the representative layer.
    """
    path_csv = os.path.join(analysis_dir, "cross_dataset_consistency.csv")
    if not os.path.exists(path_csv):
        logger.info("  Skipping fig8 (no cross-dataset data).")
        return
    df = pd.read_csv(path_csv)
    df_layer = df[df["layer"] == rep_layer]
    if df_layer.empty:
        return

    # One row per language
    languages = sorted(df_layer["language"].unique())
    datasets_a = df_layer["dataset_a"].unique()
    datasets_b = df_layer["dataset_b"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 5]})

    # (a) Bar chart: per-language cosine similarity
    ax = axes[0]
    per_lang = df_layer.groupby("language")["cosine_sim"].mean().sort_values(ascending=True)
    colors = [LANG_COLORS.get(l, "#999") for l in per_lang.index]
    bars = ax.barh(per_lang.index, per_lang.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cosine Similarity (FLORES vs OPUS-100)")
    ax.set_title(f"(a) Cross-Dataset Consistency\n(Layer {rep_layer})")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate bars with values
    for bar, val in zip(bars, per_lang.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    # (b) Per-language consistency across ALL layers (line plot)
    ax = axes[1]
    for lang in languages:
        df_lang = df[df["language"] == lang]
        per_layer = df_lang.groupby("layer")["cosine_sim"].mean()
        color = LANG_COLORS.get(lang, "#999")
        ax.plot(per_layer.index, per_layer.values, "-", color=color,
                label=lang, markersize=2, linewidth=1.5, alpha=0.8)

    ax.axhline(y=0.9, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(FLORES, OPUS-100)")
    ax.set_title("(b) Consistency Across Layers")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    fig.suptitle("Language Vector Consistency Across Datasets", fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(plot_dir, "fig8_cross_dataset_consistency.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


def plot_consistency_vs_metrics(analysis_dir, plot_dir):
    """
    Figure 9: Scatter -- does cross-dataset consistency correlate with
    orthogonality to topic vectors? (per layer)
    """
    summary_path = os.path.join(analysis_dir, "layer_summary.csv")
    df = pd.read_csv(summary_path)

    if "mean_cross_dataset_sim" not in df.columns or df["mean_cross_dataset_sim"].sum() == 0:
        logger.info("  Skipping fig9 (no cross-dataset data).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Consistency vs orthogonality
    ax = axes[0]
    sc = ax.scatter(df["mean_cross_dataset_sim"], df["mean_cross_sim"],
                    c=df["layer"], cmap="viridis", s=30, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("Cross-Dataset Consistency\ncos(FLORES, OPUS)")
    ax.set_ylabel("Cross-Similarity\n|cos(lang, topic)|")
    ax.set_title("(a) Consistency vs Orthogonality")
    plt.colorbar(sc, ax=ax, label="Layer")
    ax.grid(True, alpha=0.3)

    # (b) Consistency vs decomposition
    ax = axes[1]
    if "mean_decomp_sim" in df.columns:
        sc = ax.scatter(df["mean_cross_dataset_sim"], df["mean_decomp_sim"],
                        c=df["layer"], cmap="viridis", s=30, edgecolors="black", linewidths=0.3)
        ax.set_ylabel("Decomposition Quality\ncos(culture, lang+topic)")
        plt.colorbar(sc, ax=ax, label="Layer")
    ax.set_xlabel("Cross-Dataset Consistency\ncos(FLORES, OPUS)")
    ax.set_title("(b) Consistency vs Decomposition")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Relating Cross-Dataset Consistency to Other Metrics", fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(plot_dir, "fig9_consistency_vs_metrics.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved {path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    analysis_dir = cfg["paths"]["va_analysis_dir"]
    plot_dir = cfg["paths"]["va_plot_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    summary_df = pd.read_csv(os.path.join(analysis_dir, "layer_summary.csv"))
    rep_layer = pick_representative_layer(summary_df)
    logger.info(f"Representative layer: {rep_layer}")

    logger.info("Generating figures...")
    plot_language_heatmap(analysis_dir, plot_dir, rep_layer)
    plot_topic_heatmap(analysis_dir, plot_dir, rep_layer)
    plot_cross_similarity(analysis_dir, plot_dir, rep_layer)
    plot_pca_projections(analysis_dir, plot_dir, rep_layer)
    plot_layer_progression(analysis_dir, plot_dir)
    plot_decomposition_detail(analysis_dir, plot_dir, rep_layer)
    plot_cross_violin(analysis_dir, plot_dir)
    plot_cross_dataset_heatmap(analysis_dir, plot_dir, rep_layer)    # NEW
    plot_consistency_vs_metrics(analysis_dir, plot_dir)               # NEW

    logger.info(f"All figures saved to {plot_dir}")


if __name__ == "__main__":
    main()
