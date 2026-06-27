"""
Cross-language / region / topic analysis.

Public API
----------
run(cfg, store=None, meta=None) -> dict
    Writes four output files to cfg['paths']['analysis_dir']:

    cross_language_topic_cosine.csv
        (model, readout, layer, topic, lang_a, lang_b, cosine)
        Per-language DiffMean topic vectors; pairwise cosine between languages.

    region_contrasts.csv
        (model, readout, layer, contrast_type, key_a, key_b, cosine, shared_text)
        Cosines between language_region mean vectors for:
          - same-language / different-region pairs
          - different-language / same-region pairs
        shared_text=True when the two mean vectors are identical (np.allclose),
        flagging the degenerate shared-corpus (FLORES/Wiki) case.

    topic_rdm_<model>_<readout>_<layer>.npy
        RDM matrix over per-topic mean vectors.

    heldout_language_transfer.csv
        (model, readout, layer, topic, heldout_language, macro_f1)
        Topic-vs-rest logistic probe trained on all-but-one language,
        evaluated on the held-out language.

Guards
------
- Skips when only one language is present.
- Skips (topic, language) pairs with <2 samples.
- Emits warnings for all skipped cases; never raises.
"""
import itertools
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.shared_utils.io import save_csv
from src.shared_utils.probes import probe_score, train_probe
from src.shared_utils.similarity import rdm as compute_rdm
from src.shared_utils.vectors import balanced_background, cosine, diffmean
from src.modules.normalize.run import representation as get_representation

logger = logging.getLogger(__name__)

_COSINE_HEADER = ["model", "readout", "layer", "topic", "lang_a", "lang_b", "cosine"]
_REGION_HEADER = [
    "model", "readout", "layer", "contrast_type", "key_a", "key_b", "cosine", "shared_text"
]
_TRANSFER_HEADER = ["model", "readout", "layer", "topic", "heldout_language", "macro_f1"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _topic_lang_vector(
    X: np.ndarray,
    topic_labels: np.ndarray,
    lang_labels: np.ndarray,
    topic: str,
    lang: str,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    DiffMean topic vector for *topic* within *lang*.

    Returns (vector, None) on success, (None, reason) when the slice must be
    skipped.
    """
    lang_mask = lang_labels == lang
    X_lang = X[lang_mask]
    tl_lang = topic_labels[lang_mask]

    target_mask = tl_lang == topic
    n_target = int(target_mask.sum())
    if n_target < 2:
        return None, f"topic={topic!r} lang={lang}: only {n_target} sample(s) — need ≥2"

    # balanced background within this language, grouped by topic label
    bg = balanced_background(X_lang, tl_lang, topic, tl_lang, rng)
    if bg.shape[0] == 0:
        return None, f"topic={topic!r} lang={lang}: no background samples in language"

    v = diffmean(X_lang[target_mask], bg)
    return v, None


def _cross_language_cosines(
    X: np.ndarray,
    meta,
    model: str,
    readout: str,
    layer,
    rng: np.random.Generator,
) -> List[Tuple]:
    topic_labels = meta["topic_canonical"].astype(str).values
    lang_labels = meta["language"].astype(str).values
    topics = sorted(set(topic_labels))
    languages = sorted(set(lang_labels))
    layer_str = str(layer)

    if len(languages) < 2:
        warnings.warn(
            f"Only one language present for {model}/{readout}/{layer}; "
            "skipping cross-language topic cosines.",
            stacklevel=4,
        )
        return []

    rows: List[Tuple] = []

    for topic in topics:
        lang_vectors: Dict[str, np.ndarray] = {}
        for lang in languages:
            v, err = _topic_lang_vector(X, topic_labels, lang_labels, topic, lang, rng)
            if err:
                warnings.warn(f"cross-language cosine — {err}", stacklevel=4)
            else:
                lang_vectors[lang] = v

        for lang_a, lang_b in itertools.combinations(sorted(lang_vectors.keys()), 2):
            cos = cosine(lang_vectors[lang_a], lang_vectors[lang_b])
            rows.append((model, readout, layer_str, topic, lang_a, lang_b, cos))

    return rows


def _region_contrasts(
    X: np.ndarray,
    meta,
    model: str,
    readout: str,
    layer,
) -> List[Tuple]:
    lang_labels = meta["language"].astype(str).values
    region_labels = meta["region"].astype(str).values
    lr_labels = meta["language_region"].astype(str).values
    layer_str = str(layer)

    # Per-language_region mean vectors
    lr_groups = sorted(set(lr_labels))
    lr_means: Dict[str, np.ndarray] = {}
    lr_info: Dict[str, Tuple[str, str]] = {}   # lr -> (language, region)

    for lr in lr_groups:
        mask = lr_labels == lr
        if not mask.any():
            continue
        lr_means[lr] = X[mask].mean(0)
        lr_info[lr] = (lang_labels[mask][0], region_labels[mask][0])

    rows: List[Tuple] = []
    lr_list = sorted(lr_means.keys())

    for i, lr_a in enumerate(lr_list):
        for lr_b in lr_list[i + 1:]:
            lang_a, reg_a = lr_info[lr_a]
            lang_b, reg_b = lr_info[lr_b]

            same_lang = lang_a == lang_b
            same_region = reg_a == reg_b

            if same_lang and not same_region:
                contrast_type = "same_language_different_region"
            elif not same_lang and same_region:
                contrast_type = "different_language_same_region"
            else:
                # same-lang/same-region or diff-lang/diff-region — skip
                continue

            v_a = lr_means[lr_a]
            v_b = lr_means[lr_b]
            cos = cosine(v_a, v_b)
            shared = bool(np.allclose(v_a, v_b))

            rows.append((
                model, readout, layer_str,
                contrast_type, lr_a, lr_b,
                cos, shared,
            ))

    return rows


def _topic_rdm(
    X: np.ndarray,
    meta,
    model: str,
    readout: str,
    layer,
    analysis_dir: str,
) -> None:
    topic_labels = meta["topic_canonical"].astype(str).values
    topics = sorted(set(topic_labels))

    topic_vecs = []
    for topic in topics:
        mask = topic_labels == topic
        if not mask.any():
            continue
        topic_vecs.append(X[mask].mean(0))

    if len(topic_vecs) < 2:
        warnings.warn(
            f"Fewer than 2 topics for RDM in {model}/{readout}/{layer}; skipping.",
            stacklevel=4,
        )
        return

    mat = np.stack(topic_vecs)
    rdm_mat = compute_rdm(mat)

    model_safe = model.replace("/", "_")
    readout_safe = readout.replace("/", "_")
    layer_safe = str(layer)
    fn = f"topic_rdm_{model_safe}_{readout_safe}_{layer_safe}.npy"
    out_path = os.path.join(analysis_dir, fn)
    np.save(out_path, rdm_mat)
    logger.info("Wrote topic RDM to %s", out_path)


def _heldout_language_transfer(
    X: np.ndarray,
    meta,
    model: str,
    readout: str,
    layer,
    seed: int,
) -> List[Tuple]:
    topic_labels = meta["topic_canonical"].astype(str).values
    lang_labels = meta["language"].astype(str).values
    topics = sorted(set(topic_labels))
    languages = sorted(set(lang_labels))
    layer_str = str(layer)

    if len(languages) < 2:
        warnings.warn(
            f"Only one language for {model}/{readout}/{layer}; "
            "cannot run held-out language transfer.",
            stacklevel=4,
        )
        return []

    rows: List[Tuple] = []

    for topic in topics:
        y_bin = (topic_labels == topic).astype(int)

        for heldout_lang in languages:
            train_mask = lang_labels != heldout_lang
            test_mask = lang_labels == heldout_lang

            X_tr = X[train_mask]
            y_tr = y_bin[train_mask]
            X_te = X[test_mask]
            y_te = y_bin[test_mask]

            if len(X_tr) < 2 or len(X_te) < 2:
                warnings.warn(
                    f"heldout_language_transfer — topic={topic!r} "
                    f"heldout={heldout_lang}: <2 train or test samples; skipping.",
                    stacklevel=4,
                )
                continue

            if len(set(y_tr)) < 2:
                warnings.warn(
                    f"heldout_language_transfer — topic={topic!r} "
                    f"heldout={heldout_lang}: only one class in training data; skipping.",
                    stacklevel=4,
                )
                continue

            try:
                fitted = train_probe(X_tr, y_tr, "logistic", seed)
                scores = probe_score(fitted, X_te, y_te)
            except Exception as exc:
                logger.warning(
                    "Probe failed topic=%r heldout_lang=%r: %s", topic, heldout_lang, exc
                )
                continue

            rows.append((
                model, readout, layer_str, topic, heldout_lang,
                scores["macro_f1"],
            ))

    return rows


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(cfg, store=None, meta=None):
    """
    Run cross-language / region / topic analysis.

    Parameters
    ----------
    cfg : dict
        Required: cfg['paths']['analysis_dir'].
        Optional: cfg['analysis']['seed'] (default 42).
        When store / meta are None, uses cfg['paths']['store_dir'] and
        cfg['paths']['metadata'].
    store : ActivationStore-like, optional
        Must expose .models(), .readouts(model), .layers(model, readout),
        and .load_layer(model, readout, layer).
    meta : pd.DataFrame, optional
        Row-aligned with the store arrays.  Must contain columns:
        topic_canonical, language, region, language_region.

    Returns
    -------
    dict
        {'cosine_rows': list, 'region_rows': list, 'transfer_rows': list}
    """
    if store is None:
        from src.shared_utils.store import ActivationStore
        store = ActivationStore(cfg["paths"]["store_dir"])
    if meta is None:
        from src.shared_utils.store import MetadataTable
        meta = MetadataTable.load(cfg["paths"]["metadata"])

    seed: int = cfg.get("analysis", {}).get("seed", 42)
    analysis_dir: str = cfg["paths"]["analysis_dir"]
    os.makedirs(analysis_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_cosine_rows: List[Tuple] = []
    all_region_rows: List[Tuple] = []
    all_transfer_rows: List[Tuple] = []

    for model in store.models():
        for readout in store.readouts(model):
            for layer in store.layers(model, readout):
                # Load representation once per (model, readout, layer)
                try:
                    X = get_representation(store, meta, model, readout, layer, "raw")
                except Exception as exc:
                    logger.warning(
                        "Skipping %s/%s/%s — representation load failed: %s",
                        model, readout, layer, exc,
                    )
                    continue

                # 1. Cross-language topic cosines
                rows = _cross_language_cosines(X, meta, model, readout, layer, rng)
                all_cosine_rows.extend(rows)

                # 2. Region contrasts
                rows = _region_contrasts(X, meta, model, readout, layer)
                all_region_rows.extend(rows)

                # 3. Topic RDM
                _topic_rdm(X, meta, model, readout, layer, analysis_dir)

                # 4. Held-out language transfer
                rows = _heldout_language_transfer(X, meta, model, readout, layer, seed)
                all_transfer_rows.extend(rows)

    # Write output CSVs
    save_csv(
        os.path.join(analysis_dir, "cross_language_topic_cosine.csv"),
        _COSINE_HEADER, all_cosine_rows,
    )
    save_csv(
        os.path.join(analysis_dir, "region_contrasts.csv"),
        _REGION_HEADER, all_region_rows,
    )
    save_csv(
        os.path.join(analysis_dir, "heldout_language_transfer.csv"),
        _TRANSFER_HEADER, all_transfer_rows,
    )

    logger.info(
        "cross: %d cosine rows, %d region rows, %d transfer rows",
        len(all_cosine_rows), len(all_region_rows), len(all_transfer_rows),
    )

    return {
        "cosine_rows": all_cosine_rows,
        "region_rows": all_region_rows,
        "transfer_rows": all_transfer_rows,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.shared_utils.io import load_config, setup_logging

    setup_logging(__name__)
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/configs/config.yaml"
    run(load_config(cfg_path))
