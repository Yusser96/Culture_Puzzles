"""
src/shared_utils/plotting.py
Matplotlib Agg-backend helpers for saving figures to disk.
All functions write to *path* and close the figure; they never display.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe without a display

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple


def heatmap(
    path: str,
    mat: np.ndarray,
    rows: List[str],
    cols: List[str],
    title: str = "",
    cmap: str = "viridis",
) -> None:
    """
    Save a heatmap of *mat* to *path*.

    Parameters
    ----------
    path  : output file path (e.g. "out.png")
    mat   : 2-D array, shape (len(rows), len(cols))
    rows  : y-axis tick labels
    cols  : x-axis tick labels
    title : figure title
    cmap  : matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=(max(4, len(cols) * 0.6), max(3, len(rows) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lines(
    path: str,
    x: List,
    series: Dict[str, List],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """
    Save a line chart to *path*.

    Parameters
    ----------
    path   : output file path
    x      : shared x-axis values
    series : dict mapping label -> list of y values (same length as x)
    title  : figure title
    xlabel : x-axis label
    ylabel : y-axis label
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, y in series.items():
        ax.plot(x, y, marker="o", label=label)
    if series:
        ax.legend(fontsize=8, loc="best")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(
    path: str,
    xy: np.ndarray,
    groups: List,
    title: str = "",
    alpha: float = 0.7,
) -> None:
    """
    Save a 2-D scatter plot to *path*, coloured by group labels.

    Parameters
    ----------
    path   : output file path
    xy     : array of shape (N, 2) — x/y coordinates
    groups : list of N group labels (any hashable type)
    title  : figure title
    alpha  : point transparency
    """
    xy = np.asarray(xy)
    unique_groups = list(dict.fromkeys(groups))  # preserve insertion order
    cmap = plt.get_cmap("tab20", max(len(unique_groups), 1))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for g in unique_groups:
        mask = np.array([gr == g for gr in groups])
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            label=str(g),
            color=cmap(group_to_idx[g]),
            alpha=alpha,
            s=20,
        )
    if len(unique_groups) <= 20:
        ax.legend(fontsize=7, loc="best", markerscale=1.5)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
