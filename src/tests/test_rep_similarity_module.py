"""
Tests for src/modules/rep_similarity/run.py.

Synthetic store + meta; asserts:
  - cka_matrices/cka_<model>_<readout>_layer_<L>.npy is written
  - diagonal of the CKA matrix is ~1.0 (self-similarity of a language vs itself)
  - off-diagonal < 1.0 for clearly separated language clusters
  - rep_similarity_summary.csv is written with all three measures
  - languages with <2 rows are silently skipped
  - cross-layer CKA rows appear in the summary
"""
import csv
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import src.modules.rep_similarity.run as R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeStore:
    """Minimal store stub that returns pre-built activation matrices per layer."""

    def __init__(self, layer_data: dict):
        """layer_data: {layer_index -> ndarray of shape (n_samples, d)}"""
        self._data = layer_data

    def load_layer(self, model, readout, layer):
        return self._data[layer]

    def models(self):
        return ["m"]

    def readouts(self, model):
        return ["mean"]

    def layers(self, model, readout):
        return sorted(self._data.keys())


def _make_data(rng, n_per_lang: int = 20, d: int = 8):
    """
    Two languages ('en', 'fr') with clearly different activation patterns:
      'en': cluster around [+3, 0, 0, ...]  (noise std=0.1)
      'fr': cluster around [-3, 0, 0, ...]  (noise std=0.1)

    Returns (X, meta) where X is (2*n_per_lang, d) and meta has 'language'
    and 'split' columns.
    """
    center_en = np.zeros(d)
    center_en[0] = 3.0
    center_fr = np.zeros(d)
    center_fr[0] = -3.0

    X_en = rng.normal(center_en, 0.1, (n_per_lang, d))
    X_fr = rng.normal(center_fr, 0.1, (n_per_lang, d))
    X = np.vstack([X_en, X_fr])

    meta = pd.DataFrame({
        "language": ["en"] * n_per_lang + ["fr"] * n_per_lang,
        "split":    ["train"] * n_per_lang + ["test"] * n_per_lang,
    })
    return X, meta


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestRepSimilarityModule(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        n_per, d = 20, 8
        X0, self.meta = _make_data(rng, n_per, d)
        # Layer 1 is a slightly perturbed version of layer 0 (for cross-layer test)
        X1 = X0 + rng.normal(0, 0.05, X0.shape)
        self.store = FakeStore({0: X0, 1: X1})

    # ---- primary assertion from brief ----------------------------------------

    def test_cka_matrix_written_with_unit_diagonal(self):
        """cka_matrices/ is created and .npy diagonal ≈ 1.0 (language vs itself)."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=self.store, meta=self.meta)

            cka_path = os.path.join(tmpd, "cka_matrices", "cka_m_mean_layer_0.npy")
            self.assertTrue(
                os.path.exists(cka_path),
                f"CKA matrix file not found: {cka_path}",
            )

            mat = np.load(cka_path)
            self.assertEqual(
                mat.shape, (2, 2),
                f"Expected shape (2, 2) for 2 languages, got {mat.shape}",
            )
            np.testing.assert_allclose(
                np.diag(mat),
                np.ones(2),
                atol=1e-6,
                err_msg="Diagonal of CKA matrix should be exactly 1.0 (self-similarity)",
            )

    # ---- secondary structural tests ------------------------------------------

    def test_cka_offdiagonal_less_than_one(self):
        """Off-diagonal CKA < 1.0 for clearly separated language clusters."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=self.store, meta=self.meta)

            mat = np.load(os.path.join(tmpd, "cka_matrices", "cka_m_mean_layer_0.npy"))
            off_diag = mat[0, 1]
            self.assertLess(
                off_diag, 1.0,
                f"Off-diagonal CKA should be < 1.0 for different clusters, got {off_diag}",
            )

    def test_summary_csv_written_and_non_empty(self):
        """rep_similarity_summary.csv is written and has at least one data row."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=self.store, meta=self.meta)

            csv_path = os.path.join(tmpd, "rep_similarity_summary.csv")
            self.assertTrue(os.path.exists(csv_path), "rep_similarity_summary.csv not written")

            with open(csv_path) as fh:
                data_rows = list(csv.DictReader(fh))
            self.assertGreater(len(data_rows), 0, "Summary CSV has no data rows")

    def test_summary_csv_header(self):
        """Summary CSV has exactly the expected columns in order."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=self.store, meta=self.meta)

            with open(os.path.join(tmpd, "rep_similarity_summary.csv")) as fh:
                header = next(csv.reader(fh))

            self.assertEqual(
                header,
                ["model", "readout", "layer", "measure", "group_a", "group_b", "value"],
            )

    def test_all_three_measures_present(self):
        """linear_cka, svcca, and procrustes_disparity all appear in the CSV."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=self.store, meta=self.meta)

            with open(os.path.join(tmpd, "rep_similarity_summary.csv")) as fh:
                measures = {r["measure"] for r in csv.DictReader(fh)}

            for expected in ("linear_cka", "svcca", "procrustes_disparity"):
                self.assertIn(expected, measures, f"Missing measure '{expected}'")

    def test_language_with_too_few_rows_is_skipped(self):
        """A language with 1 row is silently skipped; no crash; matrix is 2×2 not 3×3."""
        rng = np.random.default_rng(7)
        d = 4
        # 'en': 10 rows, 'fr': 1 row (skipped — < 2 rows), 'de': 10 rows
        X = rng.normal(0, 1, (21, d))
        meta = pd.DataFrame({
            "language": ["en"] * 10 + ["fr"] * 1 + ["de"] * 10,
            "split":    ["train"] * 21,
        })
        store = FakeStore({0: X})

        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=store, meta=meta)   # must not raise

            mat = np.load(os.path.join(tmpd, "cka_matrices", "cka_m_mean_layer_0.npy"))
            self.assertEqual(
                mat.shape, (2, 2),
                f"Expected (2, 2) after skipping 'fr', got {mat.shape}",
            )

    def test_cross_layer_cka_rows_written(self):
        """Cross-layer CKA rows (layer='cross') appear in the summary CSV."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=self.store, meta=self.meta)

            with open(os.path.join(tmpd, "rep_similarity_summary.csv")) as fh:
                cross_rows = [r for r in csv.DictReader(fh) if r["layer"] == "cross"]

            self.assertGreater(
                len(cross_rows), 0,
                "No cross-layer CKA rows found in summary CSV",
            )

    def test_run_returns_row_list(self):
        """run() returns a list of tuples matching the CSV header length."""
        with tempfile.TemporaryDirectory() as tmpd:
            cfg = {"paths": {"analysis_dir": tmpd}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = R.run(cfg, store=self.store, meta=self.meta)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for row in result:
            self.assertEqual(
                len(row), len(R._HEADER),
                f"Row has {len(row)} fields, expected {len(R._HEADER)}: {row}",
            )


if __name__ == "__main__":
    unittest.main()
