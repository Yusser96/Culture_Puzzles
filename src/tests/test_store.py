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


class TestStoreSlashModel(unittest.TestCase):
    def test_slash_model_name_roundtrips(self):
        import tempfile, os
        import numpy as np
        from src.shared_utils.store import ActivationStore
        with tempfile.TemporaryDirectory() as d:
            s = ActivationStore(d)
            s.save_index("Qwen/Qwen3-1.7B", ["a", "b"])
            s.save_layer("Qwen/Qwen3-1.7B", "mean_content", "embed", np.ones((2, 3)))
            self.assertEqual(s.models(), ["Qwen__Qwen3-1.7B"])
            # load via the original slashed name and via the sanitized name both work
            self.assertEqual(s.load_index("Qwen/Qwen3-1.7B"), ["a", "b"])
            self.assertEqual(s.load_index("Qwen__Qwen3-1.7B"), ["a", "b"])
            self.assertIn("embed", s.layers("Qwen__Qwen3-1.7B", "mean_content"))
