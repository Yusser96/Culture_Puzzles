"""
Representational similarity analysis across languages and layers.

Public API
----------
run(cfg, store=None, meta=None) -> list of row tuples
    Writes:
      {analysis_dir}/cka_matrices/cka_<model>_<readout>_layer_<L>.npy
      {analysis_dir}/rep_similarity_summary.csv

CSV columns: model, readout, layer, measure, group_a, group_b, value

Notes
-----
- Language matrices with <2 rows or fewer rows than dims are skipped (warn).
- Procrustes requires equal row counts; the shorter matrix is used for both.
- Cross-layer CKA uses the full (all-language) activation matrix.
"""
import logging
import os
import warnings

import numpy as np

from src.shared_utils.io import ensure_dir, save_csv
from src.shared_utils.similarity import linear_cka, procrustes_disparity, svcca
from src.modules.normalize.run import representation as get_representation

logger = logging.getLogger(__name__)

_HEADER = ["model", "readout", "layer", "measure", "group_a", "group_b", "value"]


def _layer_tag(layer):
    """Convert layer index to a filename-safe string."""
    return "embed" if layer == "embed" else str(int(layer))


def _check_lang(X_lang, lang):
    """Return (ok, reason) for a single language's activation matrix."""
    n, d = X_lang.shape
    if n < 2:
        return False, f"only {n} row(s) — need ≥2"
    if n < d:
        return False, f"{n} rows < {d} dims — need n ≥ dims for SVCCA"
    return True, None


def run(cfg, store=None, meta=None):
    """
    Compute CKA / SVCCA / Procrustes between language-conditioned activation
    matrices at each layer, and cross-layer CKA for the full activation pool.

    Parameters
    ----------
    cfg : dict
        Required keys:
          cfg['paths']['analysis_dir']   — output directory.
        Optional (under cfg['analysis']):
          'repname'  — representation variant passed to normalize.run.representation
                       (default: 'raw').
          'svcca_k'  — number of SVCCA components (default: 10).
        If store / meta are None:
          cfg['paths']['store_dir'] and cfg['paths']['metadata'] are used.
    store : ActivationStore-like, optional
        Exposes .models(), .readouts(model), .layers(model, readout),
        .load_layer(model, readout, layer).
    meta : pd.DataFrame, optional
        Row-aligned with store arrays.  Must contain column 'language'.

    Returns
    -------
    rows : list of tuples
        Each tuple matches _HEADER order (also written to CSV).
    """
    # ---- resolve store / meta from config if not provided --------------------
    if store is None:
        from src.shared_utils.store import ActivationStore
        store = ActivationStore(cfg["paths"]["store_dir"])
    if meta is None:
        from src.shared_utils.store import MetadataTable
        meta = MetadataTable.load(cfg["paths"]["metadata"])

    analysis_cfg = cfg.get("analysis", {})
    repname: str = analysis_cfg.get("repname", "raw")
    svcca_k: int = int(analysis_cfg.get("svcca_k", 10))
    analysis_dir: str = cfg["paths"]["analysis_dir"]
    cka_dir = os.path.join(analysis_dir, "cka_matrices")
    ensure_dir(cka_dir)

    all_languages = sorted(meta["language"].astype(str).unique())
    rows = []

    for model in store.models():
        for readout in store.readouts(model):
            all_layers = store.layers(model, readout)

            # Cache full-matrix representations for cross-layer CKA
            layer_reps: dict = {}

            # -----------------------------------------------------------------
            # Per-layer cross-language measures
            # -----------------------------------------------------------------
            for layer in all_layers:
                try:
                    X = get_representation(store, meta, model, readout, layer, repname)
                except Exception as exc:
                    logger.warning(
                        "Skipping %s/%s/%s — representation load failed: %s",
                        model, readout, layer, exc,
                    )
                    continue

                X = np.asarray(X, dtype=float)
                layer_reps[layer] = X

                # Split by language and apply guards
                lang_mats: dict = {}
                for lang in all_languages:
                    mask = meta["language"].astype(str).values == lang
                    X_lang = X[mask]
                    ok, reason = _check_lang(X_lang, lang)
                    if not ok:
                        warnings.warn(
                            f"rep_similarity: skipping lang='{lang}' at layer {layer}: {reason}",
                            stacklevel=2,
                        )
                        continue
                    lang_mats[lang] = X_lang

                valid_langs = sorted(lang_mats.keys())
                n_l = len(valid_langs)

                # ---- Language × Language CKA matrix --------------------------
                cka_matrix = np.full((n_l, n_l), np.nan)
                for i, la in enumerate(valid_langs):
                    for j, lb in enumerate(valid_langs):
                        try:
                            cka_matrix[i, j] = linear_cka(lang_mats[la], lang_mats[lb])
                        except Exception as exc:
                            logger.warning(
                                "linear_cka(%s, %s) @ layer %s failed: %s",
                                la, lb, layer, exc,
                            )

                ltag = _layer_tag(layer)
                npy_path = os.path.join(
                    cka_dir, f"cka_{model}_{readout}_layer_{ltag}.npy"
                )
                np.save(npy_path, cka_matrix)
                logger.info("Saved %dx%d CKA matrix to %s", n_l, n_l, npy_path)

                # ---- Pairwise summary rows (upper triangle only) --------------
                for i, la in enumerate(valid_langs):
                    for j, lb in enumerate(valid_langs):
                        if i >= j:
                            continue  # skip diagonal and lower triangle

                        Xa, Xb = lang_mats[la], lang_mats[lb]

                        # linear CKA
                        try:
                            val = linear_cka(Xa, Xb)
                            rows.append((
                                model, readout, layer, "linear_cka", la, lb, val
                            ))
                        except Exception as exc:
                            logger.warning(
                                "linear_cka(%s, %s) @ layer %s: %s", la, lb, layer, exc
                            )

                        # SVCCA — adapt k to data dimensions
                        k = min(svcca_k, Xa.shape[0], Xb.shape[0], Xa.shape[1])
                        if k < 1:
                            warnings.warn(
                                f"rep_similarity: SVCCA k={k} < 1 for "
                                f"'{la}' vs '{lb}' at layer {layer} — skipped",
                                stacklevel=2,
                            )
                        else:
                            try:
                                val = svcca(Xa, Xb, k=k)
                                rows.append((
                                    model, readout, layer, "svcca", la, lb, val
                                ))
                            except Exception as exc:
                                logger.warning(
                                    "svcca(%s, %s) @ layer %s: %s", la, lb, layer, exc
                                )

                        # Procrustes — equalise row counts
                        n_proc = min(len(Xa), len(Xb))
                        try:
                            val = procrustes_disparity(Xa[:n_proc], Xb[:n_proc])
                            rows.append((
                                model, readout, layer,
                                "procrustes_disparity", la, lb, val,
                            ))
                        except Exception as exc:
                            logger.warning(
                                "procrustes_disparity(%s, %s) @ layer %s: %s",
                                la, lb, layer, exc,
                            )

            # -----------------------------------------------------------------
            # Cross-layer CKA (full activation matrix, upper triangle)
            # -----------------------------------------------------------------
            layer_list = [l for l in all_layers if l in layer_reps]
            for i in range(len(layer_list)):
                for j in range(i + 1, len(layer_list)):
                    li, lj = layer_list[i], layer_list[j]
                    try:
                        val = linear_cka(layer_reps[li], layer_reps[lj])
                        rows.append((
                            model, readout, "cross",
                            "linear_cka",
                            f"layer_{_layer_tag(li)}",
                            f"layer_{_layer_tag(lj)}",
                            val,
                        ))
                    except Exception as exc:
                        logger.warning(
                            "Cross-layer CKA layer %s vs %s: %s", li, lj, exc
                        )

    out_path = os.path.join(analysis_dir, "rep_similarity_summary.csv")
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
