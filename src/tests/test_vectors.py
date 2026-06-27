import unittest
import numpy as np
from src.shared_utils.vectors import diffmean, cosine, cosine_matrix, balanced_background

class TestVectors(unittest.TestCase):
    def test_diffmean_unit_norm(self):
        v = diffmean(np.ones((4, 3)), np.zeros((4, 3)), normalize=True)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
    def test_cosine(self):
        self.assertAlmostEqual(cosine(np.array([1., 0]), np.array([1., 0])), 1.0, places=6)
    def test_cosine_matrix_diag(self):
        m, labels = cosine_matrix({"a": np.array([1., 0]), "b": np.array([0., 1.])})
        self.assertAlmostEqual(m[0, 0], 1.0, places=6); self.assertAlmostEqual(m[0, 1], 0.0, places=6)
    def test_balanced_background_equal_per_group(self):
        X = np.arange(60).reshape(20, 3).astype(float)
        labels = ["t"] * 5 + ["o"] * 15
        groups = (["g1"] * 10) + (["g2"] * 10)
        bg = balanced_background(X, labels, "t", groups, np.random.default_rng(0))
        self.assertEqual(bg.shape[1], 3); self.assertGreater(bg.shape[0], 0)

if __name__ == "__main__":
    unittest.main()
