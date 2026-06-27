"""
run.py — FLORES variance decomposition runner
----------------------------------------------
For every (model, readout, layer) in the ActivationStore, filters metadata to
FLORES rows (source == 'flores'), loads the activation matrix, runs
variance_partition, and appends a row to flores_decomposition.csv.

Public API
----------
run(cfg, store=None, meta=None) -> list of row tuples
"""
import logging
import os
from typing import List, Optional

import numpy as np

from src.shared_utils.io import save_csv
from src.shared_utils.store import ActivationStore, MetadataTable
from src.modules.flores_decomp.decomp import variance_partition

logger = logging.getLogger(__name__)

_HEADER = ["model", "readout", "layer", "sentence", "language", "region", "script", "residual"]


def run(cfg, store=None, meta=None):
    """
    Compute FLORES variance decomposition across all model/readout/layer cells.

    Parameters
    ----------
    cfg : dict
        Keys used:
          paths.store_dir, paths.metadata, paths.analysis_dir
    store : ActivationStore-like, optional
    meta  : pd.DataFrame, optional

    Returns
    -------
    list of tuples in _HEADER order (also written to flores_decomposition.csv)
    """
    if store is None:
        store = ActivationStore(cfg["paths"]["store_dir"])
    if meta is None:
        meta = MetadataTable.load(cfg["paths"]["metadata"])

    analysis_dir: str = cfg["paths"]["analysis_dir"]

    # Filter to FLORES rows once
    flores_mask = meta["source"].str.lower() == "flores"
    flores_meta = meta[flores_mask].reset_index(drop=True)
    flores_indices = np.where(flores_mask.values)[0]

    if len(flores_meta) == 0:
        logger.warning("No FLORES rows found in metadata; writing empty CSV.")
        rows: List = []
        out_path = os.path.join(analysis_dir, "flores_decomposition.csv")
        save_csv(out_path, _HEADER, rows)
        return rows

    rows = []
    for model in store.models():
        for readout in store.readouts(model):
            for layer in store.layers(model, readout):
                try:
                    H_full = store.load_layer(model, readout, layer)
                    H = H_full[flores_indices]
                except Exception as exc:
                    logger.warning(
                        "Skipping %s/%s/layer=%s: %s", model, readout, layer, exc
                    )
                    continue

                try:
                    vp = variance_partition(H, flores_meta)
                except Exception as exc:
                    logger.warning(
                        "variance_partition failed for %s/%s/layer=%s: %s",
                        model, readout, layer, exc,
                    )
                    continue

                row = (
                    model,
                    readout,
                    layer,
                    vp.get("sentence", 0.0),
                    vp.get("language", 0.0),
                    vp.get("region", 0.0),
                    vp.get("script", 0.0),
                    vp.get("residual", 0.0),
                )
                rows.append(row)

    out_path = os.path.join(analysis_dir, "flores_decomposition.csv")
    save_csv(out_path, _HEADER, rows)
    logger.info("Wrote %d rows to %s", len(rows), out_path)
    return rows


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.shared_utils.io import load_config, setup_logging

    setup_logging(__name__)
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/configs/config.yaml"
    run(load_config(cfg_path))
