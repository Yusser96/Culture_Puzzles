import json, os
import numpy as np

def _label(layer):
    return "embed" if layer == "embed" else f"{int(layer):03d}"

class ActivationStore:
    def __init__(self, store_dir):
        self.dir = store_dir
    def _mdir(self, model):
        return os.path.join(self.dir, model)
    def save_index(self, model, sample_ids):
        os.makedirs(self._mdir(model), exist_ok=True)
        with open(os.path.join(self._mdir(model), "sample_ids.json"), "w") as f:
            json.dump(list(sample_ids), f)
    def load_index(self, model):
        with open(os.path.join(self._mdir(model), "sample_ids.json")) as f:
            return json.load(f)
    def save_layer(self, model, readout, layer, X):
        d = os.path.join(self._mdir(model), readout); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"layer_{_label(layer)}.npy"), np.asarray(X, dtype=np.float32))
    def load_layer(self, model, readout, layer):
        return np.load(os.path.join(self._mdir(model), readout, f"layer_{_label(layer)}.npy"))
    def models(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(m for m in os.listdir(self.dir) if os.path.isdir(self._mdir(m)))
    def readouts(self, model):
        md = self._mdir(model)
        return sorted(r for r in os.listdir(md) if os.path.isdir(os.path.join(md, r)))
    def layers(self, model, readout):
        d = os.path.join(self._mdir(model), readout); out = []
        for fn in sorted(os.listdir(d)):
            if fn.startswith("layer_") and fn.endswith(".npy"):
                tag = fn[len("layer_"):-len(".npy")]
                out.append("embed" if tag == "embed" else int(tag))
        return out

class MetadataTable:
    COLUMNS = ["sample_id", "text", "source", "topic", "topic_canonical", "topic_raw",
               "language", "region", "language_region", "script", "domain",
               "prompt_template", "token_count", "translation_group_id", "split"]
    @staticmethod
    def save(df, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df[MetadataTable.COLUMNS].to_parquet(path, index=False)
    @staticmethod
    def load(path):
        import pandas as pd
        return pd.read_parquet(path)[MetadataTable.COLUMNS]
