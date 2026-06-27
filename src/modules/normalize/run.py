import numpy as np
from src.shared_utils.normalize import fit_stats, standardize, center


def representation(store, meta, model, readout, layer, repname):
    """
    Load and transform a representation variant on the fly.

    Args:
        store: Object with load_layer(model, readout, layer) -> ndarray
        meta: DataFrame with columns for 'split' and grouping factors
        model: Model name (passed to store.load_layer)
        readout: Readout name (passed to store.load_layer)
        layer: Layer index (passed to store.load_layer)
        repname: Representation variant name
            - "raw": return X unchanged
            - "standardized": standardize with train-row stats
            - "language_centered": center by language
            - "language_region_centered": center by language_region
            - "topic_centered": center by topic_canonical
            - "source_centered": center by source

    Returns:
        ndarray: The transformed representation
    """
    X = store.load_layer(model, readout, layer)
    X = np.asarray(X, dtype=float)

    if repname == "raw":
        return X

    elif repname == "standardized":
        # Fit stats on train rows only
        train_mask = meta["split"] == "train"
        X_train = X[train_mask]
        mu, std = fit_stats(X_train)
        return standardize(X, mu, std)

    elif repname == "language_centered":
        return center(X, meta["language"])

    elif repname == "language_region_centered":
        return center(X, meta["language_region"])

    elif repname == "topic_centered":
        return center(X, meta["topic_canonical"])

    elif repname == "source_centered":
        return center(X, meta["source"])

    else:
        raise ValueError(f"Unknown representation variant: {repname}")


def run(cfg, store=None, meta=None):
    """
    Validation step: confirm all representations are computable for every
    model/readout/layer in the store, and that the store index matches metadata.

    Args:
        cfg: Configuration dict with cfg['paths']['store_dir'] and
             cfg['paths']['metadata']; optional cfg['representations'] list.
        store: Optional ActivationStore override (for testing).
        meta:  Optional metadata DataFrame override (for testing).
    """
    from src.shared_utils.io import setup_logging
    from src.shared_utils.store import ActivationStore, MetadataTable

    log = setup_logging("normalize")

    if store is None:
        store = ActivationStore(cfg["paths"]["store_dir"])
    if meta is None:
        meta = MetadataTable.load(cfg["paths"]["metadata"])

    reps = cfg.get("representations", ["raw"])

    for model in store.models():
        # Spec hard-error contract: store index must match metadata sample_ids.
        idx = store.load_index(model)
        if list(idx) != meta["sample_id"].tolist():
            raise ValueError(
                f"sample_id mismatch between store[{model}] and metadata "
                f"({len(idx)} vs {len(meta)})"
            )
        for readout in store.readouts(model):
            for layer in store.layers(model, readout):
                for rep in reps:
                    representation(store, meta, model, readout, layer, rep)  # must not raise
        log.info(
            f"[{model}] representations validated for "
            f"{len(store.readouts(model))} readouts."
        )

    log.info("normalize validation complete.")
