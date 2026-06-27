"""
src/modules/data_stats/run.py
==============================
Dataset statistics over the unified MetadataTable — a groupby-based port of
scripts/data_stats.py that reads from metadata.parquet instead of raw dirs.

Public API
----------
run(cfg) -> dict
    Produces CSVs + Matplotlib Agg plots under
    ``cfg['paths']['analysis_dir']/data_stats`` and returns a summary dict.

Requires metadata to exist at ``cfg['paths']['metadata']``.  Do NOT call
``run`` in unit tests; the function is smoke-validated by the CI pipeline.

Outputs
-------
CSVs:
    overview_by_source.csv         n / token stats per source
    language_by_source.csv         per-language breakdown per source
    script_coverage.csv            script × source sample counts
    topic_region_matrix.csv        topic_canonical × region heatmap data
    confounds.csv                  factor-balance diagnostics

Plots (PNG):
    fig_counts_by_source.png
    fig_token_length_by_source.png
    fig_topic_region_heatmap.png
    fig_script_distribution.png
    fig_length_by_script.png
"""

import logging
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.shared_utils.io import ensure_dir, save_csv
from src.shared_utils.store import MetadataTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar(path: str, labels, values, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _boxplot(path: str, data: dict, title: str, xlabel: str, ylabel: str) -> None:
    keys = sorted(data.keys())
    vals = [data[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 0.6), 5))
    ax.boxplot(vals, labels=keys, showfliers=False)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _heatmap(path: str, mat: np.ndarray, rowlabels, collabels, title: str) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8, len(collabels) * 0.3), max(4, len(rowlabels) * 0.35))
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(rowlabels)))
    ax.set_yticklabels(rowlabels, fontsize=6)
    ax.set_xticks(range(len(collabels)))
    ax.set_xticklabels(collabels, rotation=90, fontsize=5)
    fig.colorbar(im, ax=ax, label="count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _hist_overlay(path: str, data: dict, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, vals in data.items():
        if len(vals):
            ax.hist(vals, bins=40, alpha=0.5, label=label, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(cfg: dict) -> Dict:
    """
    Compute and write dataset statistics from the unified metadata table.

    Parameters
    ----------
    cfg : dict
        Must contain:
          cfg['paths']['metadata']     — path to metadata.parquet
          cfg['paths']['analysis_dir'] — root output directory

    Returns
    -------
    dict
        Summary with keys ``out_dir``, ``n_samples``, ``sources``,
        ``n_scripts``, ``topics``, ``figures``.
    """
    meta_path: str = cfg["paths"]["metadata"]
    analysis_dir: str = cfg["paths"]["analysis_dir"]
    out_dir = os.path.join(analysis_dir, "data_stats")
    ensure_dir(out_dir)

    logger.info("Loading metadata from %s", meta_path)
    df: pd.DataFrame = MetadataTable.load(meta_path)
    logger.info("Loaded %d rows", len(df))

    produced_figures = []

    # -- 1. Overview by source -------------------------------------------------
    overview_rows = []
    token_by_source: Dict[str, list] = {}
    for src, grp in df.groupby("source"):
        toks = grp["token_count"].dropna().tolist()
        token_by_source[src] = [float(t) for t in toks]
        arr = np.array(toks, dtype=float) if toks else np.array([0.0])
        overview_rows.append([
            src,
            len(grp),
            round(float(arr.mean()), 2),
            round(float(arr.std()), 2),
        ])
    save_csv(
        os.path.join(out_dir, "overview_by_source.csv"),
        ["source", "n_samples", "token_len_mean", "token_len_std"],
        overview_rows,
    )

    src_labels = [r[0] for r in overview_rows]
    src_counts = [r[1] for r in overview_rows]
    fig_counts = os.path.join(out_dir, "fig_counts_by_source.png")
    _bar(fig_counts, src_labels, src_counts, "Sample count by source", "n_samples")
    produced_figures.append(fig_counts)

    fig_token = os.path.join(out_dir, "fig_token_length_by_source.png")
    _hist_overlay(fig_token, token_by_source, "Token length distribution by source", "token length")
    produced_figures.append(fig_token)

    # -- 2. Language breakdown by source ---------------------------------------
    lang_rows = []
    for (src, lang), grp in df.groupby(["source", "language"]):
        lang_rows.append([src, lang, len(grp),
                          round(float(grp["token_count"].mean()), 2)])
    save_csv(
        os.path.join(out_dir, "language_by_source.csv"),
        ["source", "language", "n_samples", "token_len_mean"],
        lang_rows,
    )

    # -- 3. Script coverage ----------------------------------------------------
    script_rows = []
    for (src, script), grp in df.groupby(["source", "script"]):
        script_rows.append([src, script, len(grp)])
    save_csv(
        os.path.join(out_dir, "script_coverage.csv"),
        ["source", "script", "n_samples"],
        script_rows,
    )

    # Length by script (all sources)
    tok_by_script: Dict[str, list] = {}
    for script, grp in df.groupby("script"):
        tok_by_script[script] = grp["token_count"].dropna().tolist()
    if tok_by_script:
        fig_script = os.path.join(out_dir, "fig_length_by_script.png")
        _boxplot(
            fig_script, tok_by_script,
            "Token length by script", "script", "token length",
        )
        produced_figures.append(fig_script)

    # Script distribution bar
    scripts_all = df["script"].value_counts()
    if len(scripts_all):
        fig_sd = os.path.join(out_dir, "fig_script_distribution.png")
        _bar(
            fig_sd,
            scripts_all.index.tolist(),
            scripts_all.values.tolist(),
            "Script distribution (all sources)", "n_samples",
        )
        produced_figures.append(fig_sd)

    # -- 4. Topic × region heatmap ---------------------------------------------
    topic_col = "topic_canonical"
    topic_region_rows = []
    if topic_col in df.columns and "region" in df.columns:
        pivot = (
            df.dropna(subset=[topic_col])
            .groupby([topic_col, "region"])
            .size()
            .unstack(fill_value=0)
        )
        if not pivot.empty:
            topics_sorted = list(pivot.index)
            regions_sorted = list(pivot.columns)
            mat = pivot.values.astype(float)
            save_csv(
                os.path.join(out_dir, "topic_region_matrix.csv"),
                [topic_col] + regions_sorted,
                [[t] + list(pivot.loc[t]) for t in topics_sorted],
            )
            if mat.size > 0:
                fig_hm = os.path.join(out_dir, "fig_topic_region_heatmap.png")
                _heatmap(fig_hm, mat, topics_sorted, regions_sorted,
                         "Topic × region sample counts")
                produced_figures.append(fig_hm)
            topic_region_rows = topics_sorted
    else:
        topics_sorted, regions_sorted = [], []

    # -- 5. Confounds CSV ------------------------------------------------------
    conf_rows = []

    # Length mean/std per source
    for src, toks in token_by_source.items():
        arr = np.array(toks, dtype=float) if toks else np.array([0.0])
        conf_rows.append(["length_token_mean", src, round(float(arr.mean()), 3)])
        conf_rows.append(["length_token_std", src, round(float(arr.std()), 3)])

    # n_scripts
    n_scripts = df["script"].nunique()
    conf_rows.append(["n_scripts", "all", n_scripts])

    # topic coverage (fraction of regions with >= 1 sample for each topic)
    if "topic_canonical" in df.columns and "region" in df.columns:
        all_regions = df["region"].nunique()
        covered_topics = df.dropna(subset=["topic_canonical"])["topic_canonical"].nunique()
        conf_rows.append(["n_covered_topics", "all", covered_topics])
        conf_rows.append(["n_regions", "all", all_regions])

    # languages with multiple regions
    if "language" in df.columns and "region" in df.columns:
        lang_region_counts = (
            df.groupby("language")["region"].nunique()
        )
        multi_region_langs = (lang_region_counts > 1).sum()
        conf_rows.append(["languages_with_multiple_regions", "registry",
                          int(multi_region_langs)])
        conf_rows.append(["regions_in_multi_region_languages", "registry",
                          int(lang_region_counts[lang_region_counts > 1].sum())])

    save_csv(
        os.path.join(out_dir, "confounds.csv"),
        ["metric", "scope", "value"],
        conf_rows,
    )

    n_topics = len(topics_sorted)
    summary = {
        "out_dir": out_dir,
        "n_samples": len(df),
        "sources": src_labels,
        "n_scripts": n_scripts,
        "topics": topics_sorted,
        "figures": produced_figures,
    }

    logger.info(
        "data_stats complete: %d samples, %d sources, %d scripts, %d topics -> %s",
        len(df), len(src_labels), n_scripts, n_topics, out_dir,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.shared_utils.io import load_config, setup_logging

    setup_logging(__name__)
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/configs/config.yaml"
    run(load_config(cfg_path))
