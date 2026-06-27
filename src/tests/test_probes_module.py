"""
Tests for src/modules/probes/run.py.

Exercises a clearly separable `topic_canonical` factor (clusters at +3 vs -3
with std 0.2) on the random split and asserts that the written CSV contains
at least one row with macro_f1 > 0.9.
"""
import csv
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import src.modules.probes.run as R


class FakeStore:
    """Minimal store stub that always returns the same X matrix."""

    def __init__(self, X):
        self.X = X

    def load_layer(self, m, r, l):
        return self.X

    def models(self):
        return ["m"]

    def readouts(self, m):
        return ["mean_content"]

    def layers(self, m, r):
        return [0]


class TestProbesModule(unittest.TestCase):
    def test_writes_scores(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(3, 0.2, (20, 4)), rng.normal(-3, 0.2, (20, 4))])
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(40)],
            "topic_canonical": ["a"] * 20 + ["b"] * 20,
            "language": (["en", "fr"] * 20)[:40],
            "region": ["x"] * 40,
            "language_region": ["x"] * 40,
            "source": ["s"] * 40,
            "prompt_template": ["raw"] * 40,
            "token_count": [5] * 40,
            "script": ["LATIN"] * 40,
            "split": ["train"] * 32 + ["test"] * 8,
        })
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "probes": {"kinds": ["logistic"], "splits": ["random"]},
                "representations": ["raw"],
                "analysis": {"seed": 0},
                "paths": {"analysis_dir": d},
            }
            R.run(cfg, store=FakeStore(X), meta=meta, factors=["topic_canonical"])
            probe_csv = os.path.join(d, "layer_probe_scores.csv")
            self.assertTrue(os.path.exists(probe_csv), "layer_probe_scores.csv not written")
            with open(probe_csv) as fh:
                rows = list(csv.DictReader(fh))
            self.assertGreater(len(rows), 0, "No rows written to layer_probe_scores.csv")
            self.assertTrue(
                any(float(r["macro_f1"]) > 0.9 for r in rows),
                f"Expected macro_f1 > 0.9 but got: {[r['macro_f1'] for r in rows]}",
            )

    def test_transfer_csv_written(self):
        """transfer_scores.csv is always created (empty when no heldout splits)."""
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(3, 0.2, (20, 4)), rng.normal(-3, 0.2, (20, 4))])
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(40)],
            "topic_canonical": ["a"] * 20 + ["b"] * 20,
            "language": (["en", "fr"] * 20)[:40],
            "region": ["x"] * 40,
            "language_region": ["x"] * 40,
            "source": ["s"] * 40,
            "prompt_template": ["raw"] * 40,
            "token_count": [5] * 40,
            "script": ["LATIN"] * 40,
            "split": ["train"] * 32 + ["test"] * 8,
        })
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "probes": {"kinds": ["logistic"], "splits": ["random"]},
                "representations": ["raw"],
                "analysis": {"seed": 0},
                "paths": {"analysis_dir": d},
            }
            R.run(cfg, store=FakeStore(X), meta=meta, factors=["topic_canonical"])
            self.assertTrue(
                os.path.exists(os.path.join(d, "transfer_scores.csv")),
                "transfer_scores.csv not written",
            )

    def test_factor_fewer_than_two_classes_skipped(self):
        """Factors with <2 distinct classes are skipped without crashing."""
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (20, 4))
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(20)],
            "topic_canonical": ["only_one"] * 20,
            "language": ["en"] * 20,
            "region": ["x"] * 20,
            "language_region": ["x"] * 20,
            "source": ["s"] * 20,
            "prompt_template": ["raw"] * 20,
            "token_count": [5] * 20,
            "script": ["LATIN"] * 20,
            "split": ["train"] * 16 + ["test"] * 4,
        })
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "probes": {"kinds": ["logistic"], "splits": ["random"]},
                "representations": ["raw"],
                "analysis": {"seed": 0},
                "paths": {"analysis_dir": d},
            }
            import warnings
            with warnings.catch_warnings(record=True):
                R.run(
                    cfg, store=FakeStore(X), meta=meta,
                    factors=["topic_canonical"],
                )
            with open(os.path.join(d, "layer_probe_scores.csv")) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 0, "Expected 0 rows for single-class factor")

    def test_heldout_rows_in_transfer_csv(self):
        """Rows with a heldout_* split appear in transfer_scores.csv."""
        rng = np.random.default_rng(3)
        # Use 4 languages so heldout_language produces valid folds
        langs = ["en", "fr", "de", "es"]
        n_per = 10
        n = n_per * len(langs)
        X = np.vstack([
            rng.normal(float(i), 0.2, (n_per, 4)) for i in range(len(langs))
        ])
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(n)],
            "topic_canonical": (["a"] * (n // 2) + ["b"] * (n // 2))[:n],
            "language": [l for l in langs for _ in range(n_per)],
            "region": ["x"] * n,
            "language_region": ["x"] * n,
            "source": ["s"] * n,
            "prompt_template": ["raw"] * n,
            "token_count": [5] * n,
            "script": ["LATIN"] * n,
            "split": ["train"] * (n - n_per) + ["test"] * n_per,
        })
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "probes": {"kinds": ["logistic"], "splits": ["heldout_language"]},
                "representations": ["raw"],
                "analysis": {"seed": 0},
                "paths": {"analysis_dir": d},
            }
            R.run(
                cfg, store=FakeStore(X), meta=meta,
                factors=["topic_canonical"],
            )
            with open(os.path.join(d, "transfer_scores.csv")) as fh:
                transfer_rows = list(csv.DictReader(fh))
            self.assertGreater(
                len(transfer_rows), 0,
                "Expected transfer rows for heldout_language split",
            )
            for r in transfer_rows:
                self.assertEqual(r["split"], "heldout_language")


if __name__ == "__main__":
    unittest.main()
