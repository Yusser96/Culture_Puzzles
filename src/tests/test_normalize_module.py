import unittest
import numpy as np
import pandas as pd
from src.modules.normalize.run import representation, run


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


class FakeStoreRun:
    """Minimal ActivationStore stand-in for run() tests."""

    def __init__(self, index=None):
        self._index = index if index is not None else ["a", "b"]

    def models(self):
        return ["m"]

    def readouts(self, model):
        return ["mean_content"]

    def layers(self, model, readout):
        return [0]

    def load_index(self, model):
        return list(self._index)

    def load_layer(self, model, readout, layer):
        return np.zeros((len(self._index), 3))


def _make_meta(sample_ids=None):
    if sample_ids is None:
        sample_ids = ["a", "b"]
    n = len(sample_ids)
    return pd.DataFrame({
        "sample_id": sample_ids,
        "split": ["train"] * n,
        "language": ["x"] * n,
        "language_region": ["x_R"] * n,
        "topic_canonical": ["topic"] * n,
        "source": ["src"] * n,
    })


class TestNormalizeRun(unittest.TestCase):

    def test_run_succeeds(self):
        """run() with matching index completes without error."""
        cfg = {"representations": ["raw"]}
        store = FakeStoreRun(index=["a", "b"])
        meta = _make_meta(["a", "b"])
        # Must not raise
        run(cfg, store=store, meta=meta)

    def test_run_mismatch_raises(self):
        """run() raises ValueError when store index != metadata sample_ids."""
        cfg = {"representations": ["raw"]}
        store = FakeStoreRun(index=["a", "c"])   # "c" != "b"
        meta = _make_meta(["a", "b"])
        with self.assertRaises(ValueError):
            run(cfg, store=store, meta=meta)


if __name__ == "__main__":
    unittest.main()
