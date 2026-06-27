import unittest
import numpy as np
from src.shared_utils.similarity import linear_cka, procrustes_disparity, rdm

class TestSim(unittest.TestCase):
    def test_cka_identity(self):
        X = np.random.default_rng(0).normal(size=(50, 8))
        self.assertAlmostEqual(linear_cka(X, X), 1.0, places=5)
    def test_cka_rotation_invariant(self):
        rng = np.random.default_rng(1); X = rng.normal(size=(50, 6))
        Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        self.assertAlmostEqual(linear_cka(X, X @ Q), 1.0, places=4)
    def test_procrustes_rotation_zero(self):
        rng = np.random.default_rng(2); X = rng.normal(size=(40, 5))
        Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        self.assertLess(procrustes_disparity(X, X @ Q), 1e-6)
    def test_rdm_shape(self):
        self.assertEqual(rdm(np.random.default_rng(0).normal(size=(7, 4))).shape, (7, 7))

if __name__ == "__main__":
    unittest.main()
