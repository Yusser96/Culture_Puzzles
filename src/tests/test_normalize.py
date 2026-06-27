import unittest
import numpy as np
from src.shared_utils.normalize import fit_stats, standardize, center

class TestNormalize(unittest.TestCase):
    def test_standardize_zero_mean_unit_std(self):
        X = np.random.default_rng(0).normal(5, 3, (200, 4))
        mu, std = fit_stats(X); Z = standardize(X, mu, std)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-6)
        np.testing.assert_allclose(Z.std(0), 1, atol=1e-2)
    def test_center_removes_group_mean(self):
        X = np.array([[10., 0], [12., 0], [0., 5], [2., 5]])
        g = ["a", "a", "b", "b"]; C = center(X, g)
        np.testing.assert_allclose(C, [[-1, 0], [1, 0], [-1, 0], [1, 0]])

if __name__ == "__main__":
    unittest.main()
