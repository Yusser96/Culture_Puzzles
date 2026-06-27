import unittest
import numpy as np
import pandas as pd
from src.modules.normalize.run import representation


class FakeStore:
    def __init__(self, X):
        self.X = X

    def load_layer(self, m, r, l):
        return self.X


class TestNormModule(unittest.TestCase):
    def test_language_centered(self):
        X = np.array([[10., 0], [12., 0], [0., 4], [2., 4]])
        meta = pd.DataFrame(
            {"language": ["a", "a", "b", "b"], "split": ["train"] * 4}
        )
        C = representation(FakeStore(X), meta, "m", "mean_content", 0, "language_centered")
        np.testing.assert_allclose(C, [[-1, 0], [1, 0], [-1, 0], [1, 0]])

    def test_raw_passthrough(self):
        X = np.arange(8).reshape(4, 2).astype(float)
        meta = pd.DataFrame(
            {"language": ["a"] * 4, "split": ["train"] * 4}
        )
        np.testing.assert_allclose(
            representation(FakeStore(X), meta, "m", "mean_content", 0, "raw"), X
        )


if __name__ == "__main__":
    unittest.main()
