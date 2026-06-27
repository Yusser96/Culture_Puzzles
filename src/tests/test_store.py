import os, tempfile, unittest
import numpy as np, pandas as pd
from src.shared_utils.store import ActivationStore, MetadataTable

class TestStore(unittest.TestCase):
    def test_activation_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            s = ActivationStore(d)
            s.save_index("m1", ["a", "b", "c"])
            s.save_layer("m1", "mean_content", "embed", np.ones((3, 4)))
            s.save_layer("m1", "mean_content", 0, np.zeros((3, 4)))
            self.assertEqual(s.load_index("m1"), ["a", "b", "c"])
            np.testing.assert_allclose(s.load_layer("m1", "mean_content", "embed"), np.ones((3, 4)))
            self.assertEqual(s.models(), ["m1"])
            self.assertEqual(set(s.layers("m1", "mean_content")), {"embed", 0})
    def test_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            df = pd.DataFrame([{c: "x" for c in MetadataTable.COLUMNS}])
            p = os.path.join(d, "m.parquet"); MetadataTable.save(df, p)
            back = MetadataTable.load(p)
            self.assertEqual(list(back.columns), MetadataTable.COLUMNS)

if __name__ == "__main__":
    unittest.main()
