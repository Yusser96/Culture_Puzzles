"""
Tests for src/modules/directions/run.py.

Uses a clearly separable ``topic_canonical`` (clusters at +5 vs -5 in dim 0)
across two languages and asserts that the written CSV contains a row for the
separable topic where cos_diffmean_logistic > 0.8.
"""
import csv
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import src.modules.directions.run as R


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


class TestDirectionsModule(unittest.TestCase):

    def _make_separable_data(self, rng):
        """
        Two topics 'A' and 'B', each with 20 samples across 2 languages.

        Topic 'A': cluster at [+5, 0, 0, 0]  (noise std=0.1)
        Topic 'B': cluster at [-5, 0, 0, 0]  (noise std=0.1)

        DiffMean('A') and probe normal('A') should both point in the +dim0
        direction, yielding a cosine very close to 1.
        """
        n_per = 20  # 20 samples per topic (10 per language)
        X_A = rng.normal([5.0, 0, 0, 0], 0.1, (n_per, 4))
        X_B = rng.normal([-5.0, 0, 0, 0], 0.1, (n_per, 4))
        X = np.vstack([X_A, X_B])

        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(2 * n_per)],
            "topic_canonical": ["A"] * n_per + ["B"] * n_per,
            "language": (["en"] * 10 + ["fr"] * 10) * 2,
        })
        return X, meta

    def test_writes_csv_with_high_cosine_for_separable_topic(self):
        """cos_diffmean_logistic > 0.8 for a cleanly separable topic."""
        rng = np.random.default_rng(42)
        X, meta = self._make_separable_data(rng)

        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "analysis": {"seed": 42},
                "paths": {"analysis_dir": d},
            }
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)

            csv_path = os.path.join(d, "topic_vector_cosines.csv")
            self.assertTrue(os.path.exists(csv_path), "topic_vector_cosines.csv not written")

            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))

            self.assertGreater(len(rows), 0, "No rows written to topic_vector_cosines.csv")

            # Find the cross-language ("ALL") row for topic "A"
            topic_a_all = [
                r for r in rows
                if r["topic"] == "A" and r["language"] == "ALL"
            ]
            self.assertEqual(
                len(topic_a_all), 1,
                f"Expected exactly 1 ALL row for topic A, got: {topic_a_all}",
            )
            row = topic_a_all[0]
            cos_val = float(row["cos_diffmean_logistic"])
            self.assertGreater(
                cos_val, 0.8,
                f"Expected cos_diffmean_logistic > 0.8 for separable topic A, got {cos_val}",
            )

    def test_csv_header(self):
        """Written CSV has exactly the expected columns in order."""
        rng = np.random.default_rng(0)
        X, meta = self._make_separable_data(rng)

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 0}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)

            csv_path = os.path.join(d, "topic_vector_cosines.csv")
            with open(csv_path) as fh:
                reader = csv.reader(fh)
                header = next(reader)

        expected = [
            "model", "readout", "layer", "topic", "language",
            "cos_diffmean_logistic", "cos_diffmean_svm", "cos_logistic_svm",
        ]
        self.assertEqual(header, expected)

    def test_per_language_rows_written(self):
        """Both 'ALL' and per-language rows exist for each topic."""
        rng = np.random.default_rng(1)
        X, meta = self._make_separable_data(rng)

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 1}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)

            with open(os.path.join(d, "topic_vector_cosines.csv")) as fh:
                rows = list(csv.DictReader(fh))

        languages = {r["language"] for r in rows}
        # Expect "ALL", "en", "fr"
        self.assertIn("ALL", languages)
        self.assertIn("en", languages)
        self.assertIn("fr", languages)

    def test_skips_topic_with_insufficient_samples(self):
        """A topic with <2 samples is skipped (no crash)."""
        rng = np.random.default_rng(2)
        # Only 1 sample for topic "rare", 19 for "common"
        X = rng.normal(0, 1, (20, 4))
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(20)],
            "topic_canonical": ["rare"] + ["common"] * 19,
            "language": (["en"] * 10 + ["fr"] * 10),
        })

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 2}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)

            with open(os.path.join(d, "topic_vector_cosines.csv")) as fh:
                rows = list(csv.DictReader(fh))

        rare_rows = [r for r in rows if r["topic"] == "rare" and r["language"] == "ALL"]
        self.assertEqual(len(rare_rows), 0, "Expected topic 'rare' to be skipped")

    def test_cosines_in_neg1_to_1(self):
        """All cosine values are in [-1, 1]."""
        rng = np.random.default_rng(3)
        X, meta = self._make_separable_data(rng)

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 3}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)

            with open(os.path.join(d, "topic_vector_cosines.csv")) as fh:
                rows = list(csv.DictReader(fh))

        for r in rows:
            for col in ("cos_diffmean_logistic", "cos_diffmean_svm", "cos_logistic_svm"):
                val = float(r[col])
                self.assertGreaterEqual(val, -1.0 - 1e-6)
                self.assertLessEqual(val, 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
