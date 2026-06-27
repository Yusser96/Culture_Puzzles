"""
Tests for src/modules/cross/run.py.

Synthetic data: 2 topics ('food', 'sport'), 2 languages ('en', 'fr'),
2 regions per language ('US', 'UK').  'en_US' and 'en_UK' are given
identical mean activation vectors to exercise the shared_text flag.

Assertions
----------
(a) cross_language_topic_cosine.csv contains exactly one row per
    (topic, lang_pair) combination.
(b) The region_contrasts row for the two language_regions that share
    identical activations has shared_text == True.
(c) CSV headers match spec.
(d) topic_rdm_*.npy is written and has shape (n_topics, n_topics).
(e) heldout_language_transfer.csv is written with the expected columns.
(f) Only one language -> cross-language cosines are skipped (no crash).
"""
import csv
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import src.modules.cross.run as R


# ---------------------------------------------------------------------------
# Minimal store stub
# ---------------------------------------------------------------------------

class FakeStore:
    """Always returns the same X for any (model, readout, layer) triple."""

    def __init__(self, X: np.ndarray):
        self._X = X

    def load_layer(self, model, readout, layer):
        return self._X

    def models(self):
        return ["model_a"]

    def readouts(self, model):
        return ["mean_content"]

    def layers(self, model, readout):
        return [0]


# ---------------------------------------------------------------------------
# Shared data factory
# ---------------------------------------------------------------------------

def make_data(rng=None):
    """
    Build synthetic activations and metadata.

    Layout (8 groups of 6 samples each = 48 rows):
      topic  lang  region  lr
      food   en    US      en_US   -> activations = [3.0, 1.0, 0.0, 0.0]  (constant)
      food   en    UK      en_UK   -> activations = [3.0, 1.0, 0.0, 0.0]  (IDENTICAL to en_US)
      food   fr    US      fr_US   -> activations = [0.0, 3.0, 1.0, 0.0]
      food   fr    UK      fr_UK   -> activations = [0.0, 3.0, 2.0, 0.0]
      sport  en    US      en_US   -> activations = [-3.0, 1.0, 0.0, 0.0] (constant)
      sport  en    UK      en_UK   -> activations = [-3.0, 1.0, 0.0, 0.0] (IDENTICAL to en_US)
      sport  fr    US      fr_US   -> activations = [0.0, -3.0, 1.0, 0.0]
      sport  fr    UK      fr_UK   -> activations = [0.0, -3.0, 2.0, 0.0]

    Mean vectors per language_region:
      en_US: mean([3,1,0,0]*6 + [-3,1,0,0]*6) = [0, 1, 0, 0]
      en_UK: mean([3,1,0,0]*6 + [-3,1,0,0]*6) = [0, 1, 0, 0]  <- IDENTICAL -> shared_text=True
      fr_US: mean([0,3,1,0]*6 + [0,-3,1,0]*6) = [0, 0, 1, 0]
      fr_UK: mean([0,3,2,0]*6 + [0,-3,2,0]*6) = [0, 0, 2, 0]  <- different
    """
    n_per = 6  # samples per (topic, language, region)
    d = 4

    groups = [
        ("food",  "en", "US", "en_US",  [ 3.0,  1.0, 0.0, 0.0]),
        ("food",  "en", "UK", "en_UK",  [ 3.0,  1.0, 0.0, 0.0]),  # identical to en_US
        ("food",  "fr", "US", "fr_US",  [ 0.0,  3.0, 1.0, 0.0]),
        ("food",  "fr", "UK", "fr_UK",  [ 0.0,  3.0, 2.0, 0.0]),
        ("sport", "en", "US", "en_US",  [-3.0,  1.0, 0.0, 0.0]),
        ("sport", "en", "UK", "en_UK",  [-3.0,  1.0, 0.0, 0.0]),  # identical to en_US
        ("sport", "fr", "US", "fr_US",  [ 0.0, -3.0, 1.0, 0.0]),
        ("sport", "fr", "UK", "fr_UK",  [ 0.0, -3.0, 2.0, 0.0]),
    ]

    rows_X = []
    rows_meta = []
    sid = 0
    for topic, lang, region, lr, vec in groups:
        for _ in range(n_per):
            rows_X.append(vec)
            rows_meta.append({
                "sample_id": str(sid),
                "topic_canonical": topic,
                "language": lang,
                "region": region,
                "language_region": lr,
                "split": "train",
            })
            sid += 1

    X = np.array(rows_X, dtype=float)
    meta = pd.DataFrame(rows_meta)
    return X, meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrossLanguageTopicCosine(unittest.TestCase):

    def _run(self, X, meta):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = R.run(cfg, store=FakeStore(X), meta=meta)
            csv_path = os.path.join(d, "cross_language_topic_cosine.csv")
            self.assertTrue(os.path.exists(csv_path), "cross_language_topic_cosine.csv not written")
            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))
            return rows, result, d

    def test_row_per_topic_language_pair(self):
        """(a) CSV has exactly one row per (topic, lang_a, lang_b) pair."""
        X, meta = make_data()
        rows, _, d = self._run(X, meta)

        topics = sorted(meta["topic_canonical"].unique())       # ['food', 'sport']
        languages = sorted(meta["language"].unique())            # ['en', 'fr']
        import itertools
        expected_pairs = list(itertools.combinations(languages, 2))  # [('en', 'fr')]

        expected_count = len(topics) * len(expected_pairs)      # 2 * 1 = 2
        self.assertEqual(
            len(rows), expected_count,
            f"Expected {expected_count} rows but got {len(rows)}: {rows}",
        )

        # Check that every (topic, lang_pair) exists exactly once
        for topic in topics:
            for la, lb in expected_pairs:
                matching = [
                    r for r in rows
                    if r["topic"] == topic and r["lang_a"] == la and r["lang_b"] == lb
                ]
                self.assertEqual(
                    len(matching), 1,
                    f"Expected 1 row for topic={topic!r} ({la},{lb}), got {len(matching)}",
                )

    def test_csv_header(self):
        """CSV header matches the spec exactly."""
        X, meta = make_data()
        rows, _, _ = self._run(X, meta)
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 0}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            with open(os.path.join(d, "cross_language_topic_cosine.csv")) as fh:
                header = next(csv.reader(fh))
        self.assertEqual(
            header, ["model", "readout", "layer", "topic", "lang_a", "lang_b", "cosine"]
        )

    def test_cosine_values_in_range(self):
        """All cosine values lie in [-1, 1]."""
        X, meta = make_data()
        rows, _, _ = self._run(X, meta)
        for r in rows:
            val = float(r["cosine"])
            self.assertGreaterEqual(val, -1.0 - 1e-6)
            self.assertLessEqual(val, 1.0 + 1e-6)


class TestRegionContrasts(unittest.TestCase):

    def _run(self, X, meta):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            csv_path = os.path.join(d, "region_contrasts.csv")
            self.assertTrue(os.path.exists(csv_path), "region_contrasts.csv not written")
            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))
            return rows

    def test_shared_text_flag_when_identical_activations(self):
        """
        (b) When en_US and en_UK have the same mean activation vector,
        the region_contrasts row for that pair has shared_text == 'True'.
        """
        X, meta = make_data()
        rows = self._run(X, meta)

        # There must be at least one row for the same-language/different-region contrast
        same_lang_rows = [r for r in rows if r["contrast_type"] == "same_language_different_region"]
        self.assertGreater(len(same_lang_rows), 0, "No same_language_different_region rows found")

        # Find en_US vs en_UK (or en_UK vs en_US)
        en_identical = [
            r for r in same_lang_rows
            if set((r["key_a"], r["key_b"])) == {"en_US", "en_UK"}
        ]
        self.assertEqual(
            len(en_identical), 1,
            f"Expected exactly 1 row for en_US/en_UK, got: {en_identical}",
        )
        self.assertEqual(
            en_identical[0]["shared_text"], "True",
            f"Expected shared_text=True for identical en_US/en_UK vectors, "
            f"got: {en_identical[0]['shared_text']}",
        )

    def test_shared_text_false_when_different_activations(self):
        """fr_US and fr_UK have different mean vectors -> shared_text == False."""
        X, meta = make_data()
        rows = self._run(X, meta)

        same_lang_rows = [r for r in rows if r["contrast_type"] == "same_language_different_region"]
        fr_rows = [
            r for r in same_lang_rows
            if set((r["key_a"], r["key_b"])) == {"fr_US", "fr_UK"}
        ]
        self.assertEqual(len(fr_rows), 1, f"Expected 1 row for fr_US/fr_UK, got {fr_rows}")
        self.assertEqual(
            fr_rows[0]["shared_text"], "False",
            f"Expected shared_text=False for fr_US/fr_UK, got {fr_rows[0]['shared_text']}",
        )

    def test_different_language_same_region_rows_present(self):
        """Rows of type 'different_language_same_region' are also written."""
        X, meta = make_data()
        rows = self._run(X, meta)

        diff_lang_rows = [r for r in rows if r["contrast_type"] == "different_language_same_region"]
        self.assertGreater(
            len(diff_lang_rows), 0, "Expected at least one different_language_same_region row"
        )

    def test_region_contrasts_csv_header(self):
        """region_contrasts.csv header matches the spec."""
        X, meta = make_data()
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 0}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            with open(os.path.join(d, "region_contrasts.csv")) as fh:
                header = next(csv.reader(fh))
        self.assertEqual(
            header,
            ["model", "readout", "layer", "contrast_type", "key_a", "key_b", "cosine", "shared_text"],
        )


class TestTopicRDM(unittest.TestCase):

    def test_rdm_file_written_and_shape(self):
        """topic_rdm_*.npy is written and has shape (n_topics, n_topics)."""
        X, meta = make_data()
        topics = meta["topic_canonical"].unique()

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)

            rdm_files = [f for f in os.listdir(d) if f.startswith("topic_rdm_") and f.endswith(".npy")]
            self.assertEqual(len(rdm_files), 1, f"Expected 1 RDM file, got {rdm_files}")

            mat = np.load(os.path.join(d, rdm_files[0]))
            n = len(topics)
            self.assertEqual(
                mat.shape, (n, n),
                f"Expected shape ({n},{n}), got {mat.shape}",
            )


class TestHeldoutLanguageTransfer(unittest.TestCase):

    def test_transfer_csv_written(self):
        """heldout_language_transfer.csv is written."""
        X, meta = make_data()
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            transfer_path = os.path.join(d, "heldout_language_transfer.csv")
            self.assertTrue(os.path.exists(transfer_path), "heldout_language_transfer.csv missing")

    def test_transfer_csv_header(self):
        """heldout_language_transfer.csv header matches spec."""
        X, meta = make_data()
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            with open(os.path.join(d, "heldout_language_transfer.csv")) as fh:
                header = next(csv.reader(fh))
        self.assertEqual(
            header,
            ["model", "readout", "layer", "topic", "heldout_language", "macro_f1"],
        )

    def test_transfer_rows_per_topic_and_language(self):
        """One row per (topic, heldout_language) combination."""
        X, meta = make_data()
        topics = sorted(meta["topic_canonical"].unique())
        languages = sorted(meta["language"].unique())

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            with open(os.path.join(d, "heldout_language_transfer.csv")) as fh:
                rows = list(csv.DictReader(fh))

        expected = len(topics) * len(languages)
        self.assertEqual(
            len(rows), expected,
            f"Expected {expected} transfer rows, got {len(rows)}",
        )

    def test_macro_f1_in_range(self):
        """macro_f1 values are in [0, 1]."""
        X, meta = make_data()
        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 42}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            with open(os.path.join(d, "heldout_language_transfer.csv")) as fh:
                rows = list(csv.DictReader(fh))

        for r in rows:
            val = float(r["macro_f1"])
            self.assertGreaterEqual(val, 0.0 - 1e-9, f"macro_f1 out of range: {val}")
            self.assertLessEqual(val, 1.0 + 1e-9, f"macro_f1 out of range: {val}")


class TestGuards(unittest.TestCase):

    def test_single_language_no_crash(self):
        """When only one language is present, cross-language outputs are empty but no crash."""
        rng = np.random.default_rng(99)
        n = 20
        X = rng.normal(0, 1, (n, 4))
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(n)],
            "topic_canonical": ["food"] * 10 + ["sport"] * 10,
            "language": ["en"] * n,
            "region": ["US"] * n,
            "language_region": ["en_US"] * n,
            "split": ["train"] * n,
        })

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 0}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = R.run(cfg, store=FakeStore(X), meta=meta)

        # cross-language cosines: nothing to compute
        self.assertEqual(result["cosine_rows"], [],
                         "Expected empty cosine rows for single language")
        # region contrasts: nothing (only one lr)
        self.assertEqual(result["region_rows"], [],
                         "Expected empty region rows for single language_region")

    def test_fewer_than_two_samples_skipped(self):
        """A (topic, language) pair with <2 samples is silently skipped."""
        rng = np.random.default_rng(7)
        # topic 'rare' has only 1 sample in language 'en'
        n = 21
        X = rng.normal(0, 1, (n, 4))
        meta = pd.DataFrame({
            "sample_id": [str(i) for i in range(n)],
            "topic_canonical": ["rare"] + ["common"] * 10 + ["rare"] * 2 + ["common"] * 8,
            "language": ["en"] + ["en"] * 10 + ["fr"] * 10,
            "region": ["US"] * n,
            "language_region": (["en_US"] * 11) + (["fr_US"] * 10),
            "split": ["train"] * n,
        })

        with tempfile.TemporaryDirectory() as d:
            cfg = {"analysis": {"seed": 0}, "paths": {"analysis_dir": d}}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                R.run(cfg, store=FakeStore(X), meta=meta)
            # Must not crash; CSV must exist
            self.assertTrue(
                os.path.exists(os.path.join(d, "cross_language_topic_cosine.csv"))
            )


if __name__ == "__main__":
    unittest.main()
