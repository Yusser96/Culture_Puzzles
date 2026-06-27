"""
Probe runner — train linear probes across the full combination grid and
write layer_probe_scores.csv / transfer_scores.csv to analysis_dir.

Public API
----------
run(cfg, store=None, meta=None, factors=None) -> (probe_rows, transfer_rows)
"""
import logging
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np

from src.shared_utils.io import save_csv
from src.shared_utils.probes import make_splits, probe_score, train_probe
from src.shared_utils.store import ActivationStore, MetadataTable
from src.modules.normalize.run import representation as get_representation

logger = logging.getLogger(__name__)

# CSV column order (same for both output files)
_HEADER = [
    "model", "readout", "representation", "layer",
    "factor", "kind", "split", "macro_f1", "auroc",
]

# Default factor list (token_bin added if token_count is available)
DEFAULT_FACTORS: List[str] = [
    "topic_canonical",
    "language",
    "region",
    "language_region",
    "script",
    "source",
    "prompt_template",
]


def _add_token_bin(meta, n_bins: int = 4):
    """Optionally add a token_bin column from token_count quantiles (in-place)."""
    if "token_count" in meta.columns and "token_bin" not in meta.columns:
        try:
            import pandas as pd
            meta["token_bin"] = pd.qcut(
                meta["token_count"], q=n_bins, labels=False, duplicates="drop"
            ).astype(str)
        except Exception:
            pass  # silently skip if qcut fails (e.g. too few distinct values)
    return meta


def run(cfg, store=None, meta=None, factors=None):
    """
    Run probes for every cell of the
    model × readout × representation × layer × factor × kind × split grid.

    Parameters
    ----------
    cfg : dict
        Must contain keys: paths.analysis_dir, probes.kinds, probes.splits,
        representations, analysis.seed.
        paths.store_dir and paths.metadata are used only when store / meta
        are not supplied.
    store : ActivationStore-like, optional
        Object with .models(), .readouts(model), .layers(model, readout),
        .load_layer(model, readout, layer).  Defaults to ActivationStore.
    meta : pd.DataFrame, optional
        Row-aligned with the store arrays.  Defaults to MetadataTable.load.
    factors : list of str, optional
        Column names in meta to probe.  Defaults to DEFAULT_FACTORS (+token_bin
        if token_count is present).

    Returns
    -------
    (probe_rows, transfer_rows) : tuple of lists
        Raw row tuples in _HEADER order (also written to CSV).
    """
    # --- resolve defaults ---------------------------------------------------
    if store is None:
        store = ActivationStore(cfg["paths"]["store_dir"])
    if meta is None:
        meta = MetadataTable.load(cfg["paths"]["metadata"])
    if factors is None:
        factors = list(DEFAULT_FACTORS)
        meta = _add_token_bin(meta)
        if "token_bin" in meta.columns:
            factors.append("token_bin")

    seed: int = cfg.get("analysis", {}).get("seed", 42)
    kinds: List[str] = cfg["probes"]["kinds"]
    splits: List[str] = cfg["probes"]["splits"]
    representations: List[str] = cfg["representations"]
    analysis_dir: str = cfg["paths"]["analysis_dir"]

    probe_rows: List[Tuple] = []
    transfer_rows: List[Tuple] = []

    # --- main loop ----------------------------------------------------------
    for model in store.models():
        for readout in store.readouts(model):
            for repname in representations:
                for layer in store.layers(model, readout):
                    # Load representation once per (model, readout, repname, layer)
                    try:
                        X = get_representation(
                            store, meta, model, readout, layer, repname
                        )
                    except Exception as exc:
                        logger.warning(
                            "Skipping representation %s/%s/%s/%s: %s",
                            model, readout, repname, layer, exc,
                        )
                        continue

                    for factor in factors:
                        if factor not in meta.columns:
                            warnings.warn(
                                f"Factor {factor!r} not in meta columns; skipping.",
                                stacklevel=2,
                            )
                            continue

                        y = meta[factor].astype(str).values
                        n_classes = len(np.unique(y))
                        if n_classes < 2:
                            warnings.warn(
                                f"Factor {factor!r} has <2 classes; skipping.",
                                stacklevel=2,
                            )
                            continue

                        for kind in kinds:
                            for split in splits:
                                split_pairs = make_splits(meta, split, seed)
                                for tr, te in split_pairs:
                                    if len(tr) == 0 or len(te) == 0:
                                        continue
                                    if len(np.unique(y[tr])) < 2:
                                        continue

                                    try:
                                        fitted = train_probe(
                                            X[tr], y[tr], kind, seed
                                        )
                                        scores = probe_score(fitted, X[te], y[te])
                                    except Exception as exc:
                                        logger.warning(
                                            "Probe failed %s/%s/%s/%s/%s/%s/%s: %s",
                                            model, readout, repname, layer,
                                            factor, kind, split, exc,
                                        )
                                        continue

                                    row = (
                                        model, readout, repname, layer,
                                        factor, kind, split,
                                        scores["macro_f1"],
                                        scores.get("auroc"),
                                    )
                                    probe_rows.append(row)
                                    if split.startswith("heldout"):
                                        transfer_rows.append(row)

    # --- write CSVs ---------------------------------------------------------
    probe_path = os.path.join(analysis_dir, "layer_probe_scores.csv")
    transfer_path = os.path.join(analysis_dir, "transfer_scores.csv")
    save_csv(probe_path, _HEADER, probe_rows)
    save_csv(transfer_path, _HEADER, transfer_rows)

    logger.info(
        "Wrote %d probe rows to %s, %d transfer rows to %s",
        len(probe_rows), probe_path, len(transfer_rows), transfer_path,
    )
    return probe_rows, transfer_rows


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.shared_utils.io import load_config, setup_logging

    setup_logging(__name__)
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/configs/config.yaml"
    run(load_config(cfg_path))
