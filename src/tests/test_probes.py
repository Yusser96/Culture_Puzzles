import unittest
import numpy as np, pandas as pd
from src.shared_utils.probes import train_probe, probe_score, probe_normal, make_splits

def _sep(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    Xa = rng.normal(+2, 0.3, (n, d)); Xb = rng.normal(-2, 0.3, (n, d))
    return np.vstack([Xa, Xb]), np.array(["a"]*n + ["b"]*n)

class TestProbes(unittest.TestCase):
    def test_logistic_separable(self):
        X, y = _sep(); p = train_probe(X, y, "logistic", 0)
        self.assertGreater(probe_score(p, X, y)["macro_f1"], 0.95)
    def test_diffmean_normal_unit(self):
        X, y = _sep(); p = train_probe(X, y, "diffmean", 0)
        v = probe_normal(p, "diffmean"); self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, 5)
    def test_heldout_split_disjoint_groups(self):
        df = pd.DataFrame({"language": ["en","en","fr","fr","de","de"]})
        splits = make_splits(df, "heldout_language", 0)
        for tr, te in splits:
            self.assertTrue(set(df.language.iloc[tr]).isdisjoint(set(df.language.iloc[te])))

if __name__ == "__main__":
    unittest.main()
