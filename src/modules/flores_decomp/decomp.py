"""
decomp.variance_partition
--------------------------
Sequential (type-I) variance partition of hidden-state matrix H into
contributions from four factors in fixed order:

    sentence  → translation_group_id column
    language  → language column
    region    → region column
    script    → script column
    residual  → unexplained after all four

Algorithm (averaged over hidden dimensions):
  1. total_ss  = sum_i sum_d (H[i,d] - mean_d)^2
  2. D_0       = column of ones (intercept)
  3. For factor k in order:
       D_k = hstack(D_{k-1}, onehot(factor_k))
       rss_k = ||H - D_k @ lstsq(D_k, H)||^2_F
       fraction_k = max(0, (rss_{k-1} - rss_k) / total_ss)
       prev = rss_k
  4. residual  = max(0, rss_last / total_ss)

One-hot encoding: all categories (no dropped column); numpy lstsq handles
rank-deficient systems via SVD so the projection H_pred is unique.
Constant factors (single unique value) contribute zero columns → 0 fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

# Ordered pairs: (dataframe column name, output key)
_FACTORS = [
    ("translation_group_id", "sentence"),
    ("language",             "language"),
    ("region",               "region"),
    ("script",               "script"),
]


def _onehot(series: pd.Series) -> np.ndarray:
    """Return float32 one-hot matrix (n, k).  Returns (n, 0) for constant factor."""
    uniq = sorted(series.dropna().unique())
    if len(uniq) <= 1:
        return np.zeros((len(series), 0), dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(uniq)}
    mat = np.zeros((len(series), len(uniq)), dtype=np.float32)
    for row_i, val in enumerate(series):
        idx = cat_to_idx.get(val, -1)
        if idx >= 0:
            mat[row_i, idx] = 1.0
    return mat


def _rss(H: np.ndarray, D: np.ndarray) -> float:
    """Residual sum of squares from projecting H onto column space of D."""
    # lstsq returns the minimum-norm solution; H_pred = D @ beta is the projection.
    beta, _, _, _ = np.linalg.lstsq(D, H, rcond=None)
    H_pred = D @ beta
    return float(np.sum((H - H_pred) ** 2))


def variance_partition(
    H: np.ndarray,
    factors_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Sequential variance partition of H (n_samples × n_dims).

    Parameters
    ----------
    H : np.ndarray, shape (n, d)
    factors_df : pd.DataFrame
        Must contain columns translation_group_id, language, region, script
        (missing columns silently contribute 0).

    Returns
    -------
    dict with keys sentence, language, region, script, residual;
    values are fractions in [0, 1] summing to ≈ 1.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]

    total_ss = float(np.sum((H - H.mean(axis=0)) ** 2))
    if total_ss == 0.0 or n == 0:
        return {k: 0.0 for _, k in _FACTORS} | {"residual": 0.0}

    # Start from intercept-only model (predicted = column means of H)
    D_cum = np.ones((n, 1), dtype=np.float64)
    prev = _rss(H, D_cum)   # should equal total_ss

    result: Dict[str, float] = {}
    for col, key in _FACTORS:
        if col not in factors_df.columns:
            result[key] = 0.0
            continue
        oh = _onehot(factors_df[col]).astype(np.float64)
        if oh.shape[1] == 0:
            result[key] = 0.0
            continue
        D_cum = np.concatenate([D_cum, oh], axis=1)
        curr = _rss(H, D_cum)
        result[key] = max(0.0, (prev - curr) / total_ss)
        prev = curr

    result["residual"] = max(0.0, prev / total_ss)
    return result
