"""
Tests for src/modules/steering/run.py — reliability() function only.

No model is loaded; all tests use synthetic numpy arrays.

Test strategy
-------------
1. Tight cluster of nearly-identical contrast vectors → mean_pairwise_cosine ≈ 1.
2. Well-separated pos/neg classes → pos_neg_centroid_distance > 0.
3. Overlapping pos/neg → within_class_variance is a finite non-negative float.
4. probe_margin is positive when pos centroid is further along the contrast
   direction than neg centroid.
5. Edge cases: single contrast vector, zero-variance classes.
"""

import unittest

import numpy as np

from src.modules.steering.run import reliability


class TestReliabilityTightCluster(unittest.TestCase):
    """Tight cluster of contrast vectors → mean_pairwise_cosine near 1."""

    def setUp(self):
        rng = np.random.default_rng(0)
        d = 16
        n = 8

        # Nearly-identical contrast vectors: base + tiny noise
        base = np.array([1.0] + [0.0] * (d - 1))
        self.contrast_vectors = base + rng.normal(0, 1e-4, (n, d))

        # Pos cluster around +5 in dim 0, neg around -5
        self.pos = rng.normal([5.0] + [0.0] * (d - 1), 0.1, (20, d))
        self.neg = rng.normal([-5.0] + [0.0] * (d - 1), 0.1, (20, d))

    def test_mean_pairwise_cosine_near_one(self):
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        self.assertGreater(
            result["mean_pairwise_cosine"], 0.99,
            f"Expected mean_pairwise_cosine > 0.99 for tight cluster, "
            f"got {result['mean_pairwise_cosine']:.6f}",
        )

    def test_keys_present(self):
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        expected_keys = {
            "mean_pairwise_cosine",
            "pos_neg_centroid_distance",
            "within_class_variance",
            "probe_margin",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_all_values_are_finite(self):
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        for key, val in result.items():
            self.assertTrue(
                np.isfinite(val),
                f"Expected finite value for {key!r}, got {val}",
            )


class TestReliabilitySeparatedCentroids(unittest.TestCase):
    """Well-separated pos/neg → positive centroid distance."""

    def setUp(self):
        rng = np.random.default_rng(1)
        d = 8
        self.pos = rng.normal([10.0] + [0.0] * (d - 1), 0.5, (15, d))
        self.neg = rng.normal([-10.0] + [0.0] * (d - 1), 0.5, (15, d))
        # Contrast vectors ~ pos - mean(neg)
        mu_neg = self.neg.mean(axis=0)
        self.contrast_vectors = self.pos - mu_neg

    def test_positive_centroid_distance(self):
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        self.assertGreater(
            result["pos_neg_centroid_distance"], 0.0,
            "Expected positive centroid distance for separated classes.",
        )

    def test_centroid_distance_is_approximately_correct(self):
        # ||mean(pos) - mean(neg)|| should be close to 20 (10 - (-10))
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        self.assertGreater(result["pos_neg_centroid_distance"], 15.0)
        self.assertLess(result["pos_neg_centroid_distance"], 25.0)

    def test_probe_margin_positive(self):
        # pos centroid is in the same direction as the contrast vectors,
        # so the projection should be positive
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        self.assertGreater(
            result["probe_margin"], 0.0,
            f"Expected positive probe_margin, got {result['probe_margin']:.4f}",
        )

    def test_mean_pairwise_cosine_in_range(self):
        result = reliability(self.contrast_vectors, self.pos, self.neg)
        self.assertGreaterEqual(result["mean_pairwise_cosine"], -1.0 - 1e-9)
        self.assertLessEqual(result["mean_pairwise_cosine"], 1.0 + 1e-9)


class TestReliabilityWithinClassVariance(unittest.TestCase):
    """within_class_variance is non-negative and scales with actual spread."""

    def test_low_variance_classes(self):
        rng = np.random.default_rng(2)
        d = 4
        pos = rng.normal(0, 1e-6, (10, d))  # nearly constant
        neg = rng.normal(1, 1e-6, (10, d))
        cvecs = pos - neg.mean(axis=0)
        result = reliability(cvecs, pos, neg)
        self.assertAlmostEqual(result["within_class_variance"], 0.0, places=5)

    def test_high_variance_classes(self):
        rng = np.random.default_rng(3)
        d = 4
        pos = rng.normal(0, 10.0, (50, d))
        neg = rng.normal(5, 10.0, (50, d))
        cvecs = pos - neg.mean(axis=0)
        result_high = reliability(cvecs, pos, neg)

        pos_low = rng.normal(0, 0.01, (50, d))
        neg_low = rng.normal(5, 0.01, (50, d))
        cvecs_low = pos_low - neg_low.mean(axis=0)
        result_low = reliability(cvecs_low, pos_low, neg_low)

        self.assertGreater(
            result_high["within_class_variance"],
            result_low["within_class_variance"],
            "High-spread classes should have larger within_class_variance.",
        )

    def test_variance_is_non_negative(self):
        rng = np.random.default_rng(4)
        d = 6
        pos = rng.standard_normal((12, d))
        neg = rng.standard_normal((12, d))
        cvecs = pos - neg.mean(axis=0)
        result = reliability(cvecs, pos, neg)
        self.assertGreaterEqual(result["within_class_variance"], 0.0)


class TestReliabilitySingleContrastVector(unittest.TestCase):
    """Single contrast vector: mean_pairwise_cosine defaults to 1.0."""

    def test_single_vector_cosine(self):
        rng = np.random.default_rng(5)
        d = 8
        cvecs = rng.standard_normal((1, d))
        pos = rng.standard_normal((5, d))
        neg = rng.standard_normal((5, d))
        result = reliability(cvecs, pos, neg)
        self.assertEqual(
            result["mean_pairwise_cosine"], 1.0,
            "Single contrast vector should give mean_pairwise_cosine == 1.0",
        )


class TestReliabilityInputValidation(unittest.TestCase):
    """Invalid inputs raise ValueError."""

    def test_1d_contrast_vectors_raises(self):
        rng = np.random.default_rng(6)
        with self.assertRaises(ValueError):
            reliability(rng.standard_normal(8), rng.standard_normal((4, 8)), rng.standard_normal((4, 8)))

    def test_1d_pos_raises(self):
        rng = np.random.default_rng(7)
        with self.assertRaises(ValueError):
            reliability(rng.standard_normal((3, 8)), rng.standard_normal(8), rng.standard_normal((4, 8)))


class TestReliabilityOppositeDirectionCluster(unittest.TestCase):
    """
    Contrast vectors pointing in alternating opposite directions.
    mean_pairwise_cosine should be well below 1.
    """

    def test_opposite_vectors_low_cosine(self):
        d = 4
        n = 6
        base = np.array([1.0, 0.0, 0.0, 0.0])
        # Alternate between +base and -base
        cvecs = np.array([base if i % 2 == 0 else -base for i in range(n)])
        pos = np.random.default_rng(8).standard_normal((10, d))
        neg = np.random.default_rng(9).standard_normal((10, d))

        result = reliability(cvecs, pos, neg)
        # Cosine between opposite vectors is -1; mean of all pairs should be ~ -1
        self.assertLess(
            result["mean_pairwise_cosine"], 0.0,
            f"Expected negative cosine for alternating directions, "
            f"got {result['mean_pairwise_cosine']:.4f}",
        )


class TestImportability(unittest.TestCase):
    """run.py is importable without a model being loaded."""

    def test_module_imports(self):
        import src.modules.steering.run as steering_run  # noqa: F401
        self.assertTrue(callable(steering_run.reliability))
        self.assertTrue(callable(steering_run.run))

    def test_init_imports(self):
        import src.modules.steering  # noqa: F401


if __name__ == "__main__":
    unittest.main()
