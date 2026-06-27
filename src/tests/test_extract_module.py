import os, tempfile, unittest
import numpy as np, pandas as pd
import src.modules.extract.run as R
from src.shared_utils.store import ActivationStore, MetadataTable

class TestExtractModule(unittest.TestCase):
    def test_writes_store(self):
        with tempfile.TemporaryDirectory() as d:
            mp = os.path.join(d, "m.parquet")
            df = pd.DataFrame([{c: "x" for c in MetadataTable.COLUMNS} for _ in range(3)])
            df["sample_id"] = ["a", "b", "c"]; df["text"] = ["t1", "t2", "t3"]
            MetadataTable.save(df, mp)
            cfg = {"models": ["fake"], "readouts": ["mean_content", "embed"],
                   "model": {"layers": "all", "batch_size": 2, "max_seq_len": 16},
                   "paths": {"metadata": mp, "store_dir": os.path.join(d, "store")}}
            class H: num_layers = 2; hidden_size = 4; name = "fake"
            R.load_decoder = lambda cfg, name: H()
            R.extract = lambda h, texts, layers, readouts, **k: {
                ro: {**{l: np.ones((len(texts), 4)) for l in layers}, "embed": np.zeros((len(texts), 4))}
                for ro in readouts}
            R.run(cfg)
            s = ActivationStore(cfg["paths"]["store_dir"])
            self.assertEqual(s.load_index("fake"), ["a", "b", "c"])
            self.assertIn("embed", s.layers("fake", "mean_content"))

if __name__ == "__main__":
    unittest.main()
