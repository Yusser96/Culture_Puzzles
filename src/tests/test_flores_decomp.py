import unittest
import numpy as np, pandas as pd
from src.modules.flores_decomp.decomp import variance_partition

class TestDecomp(unittest.TestCase):
    def test_language_dominates(self):
        rng = np.random.default_rng(0); n = 60
        lang = np.array(["en", "fr", "de"])[rng.integers(0, 3, n)]
        H = np.zeros((n, 4))
        for i, L in enumerate(lang):
            H[i] = {"en": [5, 0, 0, 0], "fr": [0, 5, 0, 0], "de": [0, 0, 5, 0]}[L]
        H += rng.normal(0, 0.1, H.shape)
        df = pd.DataFrame({"translation_group_id": rng.integers(0, 10, n).astype(str),
                           "language": lang, "region": ["x"]*n, "script": ["LATIN"]*n})
        out = variance_partition(H, df)
        self.assertGreater(out["language"], 0.7)

if __name__ == "__main__":
    unittest.main()
