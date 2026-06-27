"""
src/modules/extract/run
Orchestrate multi-readout extraction into the ActivationStore.
"""

from src.shared_utils.models import load_decoder
from src.shared_utils.extraction import extract
from src.shared_utils.store import ActivationStore, MetadataTable


def run(cfg):
    """
    Extract activations for all models and save to ActivationStore.

    Parameters
    ----------
    cfg : dict
        Pipeline config with keys:
        - models: list of model names to load
        - readouts: list of readout types
        - model: dict with keys max_seq_len, batch_size
        - paths: dict with keys metadata, store_dir
    """
    # Load metadata (row order = sample order)
    meta = MetadataTable.load(cfg["paths"]["metadata"])
    texts = meta["text"].tolist()
    sample_ids = meta["sample_id"].tolist()

    # Extract for each model
    for model_name in cfg["models"]:
        # Load decoder
        h = load_decoder(cfg, model_name)

        # Get layer indices
        layers = list(range(h.num_layers))

        # Extract activations
        out = extract(
            h,
            texts,
            layers,
            cfg["readouts"],
            max_seq_len=cfg["model"]["max_seq_len"],
            batch_size=cfg["model"]["batch_size"],
            answers=None,
        )

        # Save to store
        store = ActivationStore(cfg["paths"]["store_dir"])
        store.save_index(model_name, sample_ids)

        # Save each layer for each readout
        for readout in out:
            for layer in out[readout]:
                store.save_layer(model_name, readout, layer, out[readout][layer])
