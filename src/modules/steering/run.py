"""
Steering reliability diagnostics and activation-addition alpha-sweep runner.

Public API
----------
reliability(contrast_vectors, pos, neg) -> dict
    Pure function (no model).  Given a stack of per-example contrast
    vectors (n, d) and positive/negative activation matrices, return:

    ``mean_pairwise_cosine``
        Mean cosine similarity among all pairs of contrast vectors
        (uses src.shared_utils.vectors.cosine).

    ``pos_neg_centroid_distance``
        Euclidean distance between pos and neg centroids:
        ||mean(pos) - mean(neg)||.

    ``within_class_variance``
        Mean of the total variance of pos and neg:
        0.5 * (sum_of_per_dim_var(pos) + sum_of_per_dim_var(neg)).

    ``probe_margin``
        Signed gap between pos/neg means projected onto the mean
        contrast direction (unit-normalised).

run(cfg) -> list of row tuples
    Model-dependent.  For each (model × layer × topic), sweeps alpha
    values from cfg['steering']['alpha'], calls add_and_generate for a
    small prompt set, and writes steering_results.csv.

CSV columns
-----------
model, layer, topic, alpha, prompt, generation,
mean_pairwise_cosine, pos_neg_centroid_distance,
within_class_variance, probe_margin
"""

import logging
import os
from typing import List, Optional

import numpy as np

from src.shared_utils.io import ensure_dir, save_csv
from src.shared_utils.vectors import cosine, diffmean

logger = logging.getLogger(__name__)

_HEADER = [
    "model", "layer", "topic", "alpha", "prompt", "generation",
    "mean_pairwise_cosine", "pos_neg_centroid_distance",
    "within_class_variance", "probe_margin",
]

# Default prompts used when none are provided in cfg.
_DEFAULT_PROMPTS = [
    "Tell me about the cultural traditions of this region.",
    "What are the most important historical events here?",
    "Describe everyday life in this culture.",
]


# ---------------------------------------------------------------------------
# Pure reliability function (unit-tested, no model required)
# ---------------------------------------------------------------------------

def reliability(
    contrast_vectors: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
) -> dict:
    """
    Compute steering-direction reliability diagnostics.

    Parameters
    ----------
    contrast_vectors : np.ndarray, shape (n, d)
        Per-example contrast vectors (e.g. pos[i] - neg[i], normalised or
        raw).  At least one vector required.
    pos : np.ndarray, shape (n_pos, d)
        Positive-class activation matrix.
    neg : np.ndarray, shape (n_neg, d)
        Negative-class activation matrix.

    Returns
    -------
    dict
        Keys: mean_pairwise_cosine, pos_neg_centroid_distance,
              within_class_variance, probe_margin.
    """
    contrast_vectors = np.asarray(contrast_vectors, dtype=float)
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)

    if contrast_vectors.ndim != 2:
        raise ValueError(
            f"contrast_vectors must be 2-D (n, d), got shape {contrast_vectors.shape}"
        )
    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError("pos and neg must be 2-D arrays (n_samples, d)")

    n = contrast_vectors.shape[0]

    # ------------------------------------------------------------------
    # mean pairwise cosine among contrast vectors
    # ------------------------------------------------------------------
    if n < 2:
        mean_pairwise_cosine = 1.0  # single vector — trivially similar to itself
    else:
        cos_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                cos_sum += cosine(contrast_vectors[i], contrast_vectors[j])
                count += 1
        mean_pairwise_cosine = cos_sum / count

    # ------------------------------------------------------------------
    # pos/neg centroid distance
    # ------------------------------------------------------------------
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    pos_neg_centroid_distance = float(np.linalg.norm(mu_pos - mu_neg))

    # ------------------------------------------------------------------
    # within-class variance  (total variance = trace of covariance)
    # ------------------------------------------------------------------
    var_pos = float(np.sum(np.var(pos, axis=0)))
    var_neg = float(np.sum(np.var(neg, axis=0)))
    within_class_variance = (var_pos + var_neg) / 2.0

    # ------------------------------------------------------------------
    # probe margin  (projection of centroid gap onto mean contrast dir)
    # ------------------------------------------------------------------
    mean_dir = contrast_vectors.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm > 1e-10:
        mean_dir_hat = mean_dir / norm
        probe_margin = float(np.dot(mu_pos - mu_neg, mean_dir_hat))
    else:
        probe_margin = 0.0

    return {
        "mean_pairwise_cosine": float(mean_pairwise_cosine),
        "pos_neg_centroid_distance": pos_neg_centroid_distance,
        "within_class_variance": within_class_variance,
        "probe_margin": probe_margin,
    }


# ---------------------------------------------------------------------------
# Model-dependent alpha-sweep runner  (smoke-validated, not unit-tested)
# ---------------------------------------------------------------------------

def _compute_topic_direction(
    store,
    meta,
    model_name: str,
    readout: str,
    layer: int,
    topic: str,
) -> Optional[np.ndarray]:
    """
    Compute the DiffMean direction vector for *topic* vs rest at *layer*.

    Returns None if the slice has fewer than 2 target samples.
    """
    from src.modules.normalize.run import representation as get_representation

    try:
        X = get_representation(store, meta, model_name, readout, layer, "raw")
    except Exception as exc:
        logger.warning("Cannot load activations for %s/%s/%s: %s", model_name, readout, layer, exc)
        return None

    labels = meta["topic_canonical"].astype(str).values
    target_mask = labels == topic
    if target_mask.sum() < 2:
        return None

    bg_mask = ~target_mask
    if bg_mask.sum() == 0:
        return None

    vec = diffmean(X[target_mask], X[bg_mask], normalize=True)
    return vec


def run(cfg: dict) -> List[tuple]:
    """
    Activation-addition alpha-sweep runner.

    For each (model × layer × topic) combination, computes the DiffMean
    steering direction, evaluates reliability diagnostics, then generates
    text for each (alpha × prompt) pair.

    Parameters
    ----------
    cfg : dict
        Required keys:
          cfg['paths']['analysis_dir']   — output directory.
          cfg['steering']['alpha']       — list of alpha values to sweep.
        Optional:
          cfg['models']                  — list of model names.
          cfg['steering']['max_new_tokens'] (default: 50).
          cfg['steering']['layer']       — single layer to steer at
                                           (default: middle layer).
          cfg['steering']['topics']      — list of topics to steer.
          cfg['steering']['prompts']     — list of prompts (default built-in).
          cfg['model']['device']         — device string (default 'cpu').
          cfg['model']['dtype']          — dtype string (default 'float16').

    Returns
    -------
    rows : list of tuples
        Each tuple matches _HEADER order (also written to CSV).
    """
    from src.shared_utils.models import load_decoder
    from src.shared_utils.steering_utils import add_and_generate
    from src.shared_utils.store import ActivationStore
    from src.shared_utils.store import MetadataTable

    # ---- resolve config -----------------------------------------------------
    steering_cfg = cfg.get("steering", {})
    alphas: List[float] = [float(a) for a in steering_cfg.get("alpha", [1.0])]
    max_new_tokens: int = int(steering_cfg.get("max_new_tokens", 50))
    prompts: List[str] = steering_cfg.get("prompts", _DEFAULT_PROMPTS)
    analysis_dir: str = cfg["paths"]["analysis_dir"]
    ensure_dir(analysis_dir)

    model_names: List[str] = cfg.get("models", [])
    if not model_names:
        logger.warning("No models specified in cfg['models']; nothing to do.")
        return []

    store = ActivationStore(cfg["paths"]["store_dir"])
    meta = MetadataTable.load(cfg["paths"]["metadata"])
    all_labels = meta["topic_canonical"].astype(str).values
    all_topics: List[str] = sorted(set(all_labels))

    # Allow config to restrict topics / topics for steering
    topics_filter = steering_cfg.get("topics", None)
    if topics_filter is not None:
        all_topics = [t for t in all_topics if t in topics_filter]

    rows = []

    for model_name in model_names:
        logger.info("Loading model: %s", model_name)
        try:
            handle = load_decoder(cfg, model_name)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            continue

        # Determine layers to sweep
        readout = "mean_content"
        available_layers = store.layers(model_name, readout)
        steer_layer_cfg = steering_cfg.get("layer", None)
        if steer_layer_cfg is not None:
            steer_layers = [int(steer_layer_cfg)]
        elif available_layers:
            mid = available_layers[len(available_layers) // 2]
            steer_layers = [mid]
        else:
            steer_layers = [handle.num_layers // 2]

        for layer in steer_layers:
            for topic in all_topics:
                # ---- compute direction + reliability diagnostics -------------
                vec = _compute_topic_direction(
                    store, meta, model_name, readout, layer, topic
                )
                if vec is None:
                    logger.warning(
                        "Skipping topic=%r layer=%s — could not compute direction",
                        topic, layer,
                    )
                    continue

                # Build pos / neg matrices for reliability
                target_mask = all_labels == topic
                X_all = store.load_layer(model_name, readout, layer)
                pos_mat = X_all[target_mask]
                neg_mat = X_all[~target_mask]

                # Contrast vectors: pos[i] - neg centroid repeated
                mu_neg = neg_mat.mean(axis=0, keepdims=True)
                contrast_vecs = pos_mat - mu_neg  # (n_pos, d)

                rel = reliability(contrast_vecs, pos_mat, neg_mat)

                # ---- alpha sweep --------------------------------------------
                for alpha in alphas:
                    for prompt in prompts:
                        try:
                            generation = add_and_generate(
                                handle=handle,
                                prompt=prompt,
                                layer=layer,
                                vec=vec,
                                alpha=alpha,
                                max_new_tokens=max_new_tokens,
                            )
                        except Exception as exc:
                            logger.warning(
                                "add_and_generate failed (model=%s layer=%s topic=%r alpha=%s): %s",
                                model_name, layer, topic, alpha, exc,
                            )
                            generation = ""

                        rows.append((
                            model_name,
                            layer,
                            topic,
                            alpha,
                            prompt,
                            generation,
                            rel["mean_pairwise_cosine"],
                            rel["pos_neg_centroid_distance"],
                            rel["within_class_variance"],
                            rel["probe_margin"],
                        ))

    out_path = os.path.join(analysis_dir, "steering_results.csv")
    save_csv(out_path, _HEADER, rows)
    logger.info("Wrote %d steering rows to %s", len(rows), out_path)
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
