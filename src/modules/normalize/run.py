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


def run(cfg):
    """
    Thin CLI that validates representations are computable.

    Args:
        cfg: Configuration dict with 'store', 'meta', 'models', 'readouts', 'layers'
    """
    store = cfg.get("store")
    meta = cfg.get("meta")
    models = cfg.get("models", [])
    readouts = cfg.get("readouts", [])
    layers = cfg.get("layers", [])
    representations = cfg.get("representations", ["raw", "language_centered"])

    if not (store and meta is not None and models and readouts):
        raise ValueError("Config must include: store, meta, models, readouts")

    # Validate that each representation can be loaded
    for model in models:
        for readout in readouts:
            for layer in layers:
                for repname in representations:
                    try:
                        representation(store, meta, model, readout, layer, repname)
                    except Exception as e:
                        print(
                            f"Failed to load {model}/{readout}/{layer}/{repname}: {e}"
                        )
                        raise

    print("All representations loaded successfully")
