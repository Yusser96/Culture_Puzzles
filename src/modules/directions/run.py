"""
Direction analysis — compare DiffMean vectors to logistic/SVM probe normals.

Public API
----------
run(cfg, store=None, meta=None) -> list of row tuples
    Writes topic_vector_cosines.csv to cfg['paths']['analysis_dir'].

CSV columns
-----------
model, readout, layer, topic, language,
cos_diffmean_logistic, cos_diffmean_svm, cos_logistic_svm

One row per (model, readout, layer, topic, language):
  - language == "ALL"  → directions computed across all languages.
  - language == <lang> → directions computed within that language's rows.
"""
import logging
import os
import warnings
from typing import Optional, Tuple

import numpy as np

from src.shared_utils.io import save_csv
from src.shared_utils.probes import probe_normal, train_probe
from src.shared_utils.vectors import balanced_background, cosine, diffmean
from src.modules.normalize.run import representation as get_representation

logger = logging.getLogger(__name__)

_HEADER = [
    "model", "readout", "layer", "topic", "language",
    "cos_diffmean_logistic", "cos_diffmean_svm", "cos_logistic_svm",
]


def _directions_for_slice(
    X: np.ndarray,
    labels: np.ndarray,
    topic: str,
    groups: np.ndarray,
    seed: int,
    rng: np.random.Generator,
) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
    """
    Compute DiffMean and probe-normal directions for *topic* vs rest,
    within the supplied data slice (X, labels, groups).

    Returns
    -------
    ((cos_dm_log, cos_dm_svm, cos_log_svm), None) on success, or
    (None, reason_string) if the slice should be skipped.
    """
    target_mask = labels == topic
    n_target = int(target_mask.sum())

    if n_target < 2:
        return None, f"only {n_target} sample(s) for topic '{topic}' — need ≥2"

    y_bin = target_mask  # boolean True/False  (topic vs rest)
    n_classes = len(np.unique(y_bin))
    if n_classes < 2:
        return None, f"only one class present for topic '{topic}' (no background rows)"

    # --- DiffMean direction ---------------------------------------------------
    bg = balanced_background(X, labels, topic, groups, rng)
    v_dm = diffmean(X[target_mask], bg)

    # --- Probe normals --------------------------------------------------------
    try:
        p_log = train_probe(X, y_bin, "logistic", seed)
        p_svm = train_probe(X, y_bin, "svm", seed)
    except Exception as exc:
        return None, f"probe training failed for topic '{topic}': {exc}"

    v_log = probe_normal(p_log, "logistic")
    v_svm = probe_normal(p_svm, "svm")

    return (
        cosine(v_dm, v_log),
        cosine(v_dm, v_svm),
        cosine(v_log, v_svm),
    ), None


def run(cfg, store=None, meta=None):
    """
    Compute DiffMean vs logistic / SVM probe-normal cosines for every
    (model × readout × layer × topic) combination, across all languages
    ("ALL") and per individual language.

    Parameters
    ----------
    cfg : dict
        Required keys:
          cfg['paths']['analysis_dir'] — output directory.
        Optional:
          cfg['analysis']['seed']      — integer RNG seed (default 42).
        When store / meta are None:
          cfg['paths']['store_dir'] and cfg['paths']['metadata'] are used.
    store : ActivationStore-like, optional
        Must expose .models(), .readouts(model), .layers(model, readout),
        and .load_layer(model, readout, layer).
    meta : pd.DataFrame, optional
        Row-aligned with store arrays.  Must contain columns
        ``topic_canonical`` and ``language``.

    Returns
    -------
    rows : list of tuples
        Each tuple matches _HEADER order (also written to CSV).
    """
    # --- resolve defaults ----------------------------------------------------
    if store is None:
        from src.shared_utils.store import ActivationStore
        store = ActivationStore(cfg["paths"]["store_dir"])
    if meta is None:
        from src.shared_utils.store import MetadataTable
        meta = MetadataTable.load(cfg["paths"]["metadata"])

    seed: int = cfg.get("analysis", {}).get("seed", 42)
    analysis_dir: str = cfg["paths"]["analysis_dir"]
    rng = np.random.default_rng(seed)

    rows = []

    for model in store.models():
        for readout in store.readouts(model):
            for layer in store.layers(model, readout):
                # Load the "raw" representation once per combo
                try:
                    X = get_representation(store, meta, model, readout, layer, "raw")
                except Exception as exc:
                    logger.warning(
                        "Skipping %s/%s/%s — representation load failed: %s",
                        model, readout, layer, exc,
                    )
                    continue

                labels = meta["topic_canonical"].astype(str).values
                groups = meta["language"].astype(str).values
                topics = sorted(set(labels))
                languages = sorted(set(groups))

                for topic in topics:
                    # ---- cross-language ("ALL") row --------------------------
                    result, err = _directions_for_slice(
                        X, labels, topic, groups, seed, rng
                    )
                    if err:
                        warnings.warn(
                            f"Skipping topic={topic!r} lang=ALL: {err}",
                            stacklevel=2,
                        )
                    else:
                        cos_dm_log, cos_dm_svm, cos_log_svm = result
                        rows.append((
                            model, readout, layer, topic, "ALL",
                            cos_dm_log, cos_dm_svm, cos_log_svm,
                        ))

                    # ---- per-language rows -----------------------------------
                    for lang in languages:
                        lang_mask = groups == lang
                        X_lang = X[lang_mask]
                        labels_lang = labels[lang_mask]
                        groups_lang = groups[lang_mask]

                        result, err = _directions_for_slice(
                            X_lang, labels_lang, topic, groups_lang, seed, rng
                        )
                        if err:
                            warnings.warn(
                                f"Skipping topic={topic!r} lang={lang}: {err}",
                                stacklevel=2,
                            )
                        else:
                            cos_dm_log, cos_dm_svm, cos_log_svm = result
                            rows.append((
                                model, readout, layer, topic, lang,
                                cos_dm_log, cos_dm_svm, cos_log_svm,
                            ))

    out_path = os.path.join(analysis_dir, "topic_vector_cosines.csv")
    save_csv(out_path, _HEADER, rows)
    logger.info("Wrote %d direction rows to %s", len(rows), out_path)
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
