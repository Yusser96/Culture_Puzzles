"""
Tests for src/modules/report/run.py — success_criteria only.

Unit-tests for the pure function success_criteria(record) -> dict[str, bool].
No I/O, no metadata, no model.

Thresholds used (see run.py docstring):
    decodable            : probe_macro_f1 > chance + 0.1  (default: > 0.6)
    persists_after_controls : macro_f1_centered > 0.55
    transfers            : heldout_macro_f1 > transfer_threshold (default 0.5)
    layer_stable         : n_stable_layers >= 2
    coherent             : mean_contrast_cosine > 0.3
    not_confounded       : NOT (script_macro_f1 > 0.8 AND topic_macro_f1 < script_macro_f1)
    causal               : steering_effect > 0
"""

import unittest

from src.modules.report.run import success_criteria


# ---------------------------------------------------------------------------
# A "passing" record that satisfies every criterion
# ---------------------------------------------------------------------------
_PASSING_RECORD = {
    "probe_macro_f1": 0.75,        # > 0.6  → decodable
    "macro_f1_centered": 0.65,     # > 0.55 → persists_after_controls
    "heldout_macro_f1": 0.62,      # > 0.5  → transfers
    "n_stable_layers": 3,          # >= 2   → layer_stable
    "mean_contrast_cosine": 0.55,  # > 0.3  → coherent
    "script_macro_f1": 0.60,       # 0.60 ≤ 0.8 → not_confounded (condition false)
    "topic_macro_f1": 0.75,
    "steering_effect": 0.25,       # > 0    → causal
}


class TestSuccessCriteriaAllPass(unittest.TestCase):
    """Passing record must return all True."""

    def setUp(self):
        self.result = success_criteria(dict(_PASSING_RECORD))

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_all_keys(self):
        expected_keys = {
            "decodable", "persists_after_controls", "transfers",
            "layer_stable", "coherent", "not_confounded", "causal",
        }
        self.assertEqual(set(self.result.keys()), expected_keys)

    def test_all_true(self):
        for key, val in self.result.items():
            self.assertTrue(val, f"Expected {key!r} to be True for passing record")


class TestSuccessCriteriaFlipTransfers(unittest.TestCase):
    """Lowering heldout_macro_f1 below the default transfer threshold (0.5)
    must flip 'transfers' to False while leaving everything else True."""

    def setUp(self):
        rec = dict(_PASSING_RECORD)
        rec["heldout_macro_f1"] = 0.40  # below default threshold 0.5
        self.result = success_criteria(rec)

    def test_transfers_false(self):
        self.assertFalse(self.result["transfers"])

    def test_other_criteria_unchanged(self):
        for key in ("decodable", "persists_after_controls",
                    "layer_stable", "coherent", "not_confounded", "causal"):
            self.assertTrue(self.result[key],
                            f"Expected {key!r} to still be True")


class TestSuccessCriteriaFlipCausal(unittest.TestCase):
    """steering_effect = 0 must flip 'causal' to False."""

    def setUp(self):
        rec = dict(_PASSING_RECORD)
        rec["steering_effect"] = 0  # not > 0
        self.result = success_criteria(rec)

    def test_causal_false(self):
        self.assertFalse(self.result["causal"])

    def test_other_criteria_unchanged(self):
        for key in ("decodable", "persists_after_controls", "transfers",
                    "layer_stable", "coherent", "not_confounded"):
            self.assertTrue(self.result[key],
                            f"Expected {key!r} to still be True")


class TestSuccessCriteriaFlipDecodable(unittest.TestCase):
    """probe_macro_f1 at 0.55 is below the default threshold of 0.60."""

    def test_decodable_false_below_threshold(self):
        rec = dict(_PASSING_RECORD)
        rec["probe_macro_f1"] = 0.55
        result = success_criteria(rec)
        self.assertFalse(result["decodable"])

    def test_decodable_true_with_custom_threshold(self):
        rec = dict(_PASSING_RECORD)
        rec["probe_macro_f1"] = 0.55
        rec["decodable_threshold"] = 0.50  # explicit lower threshold
        result = success_criteria(rec)
        self.assertTrue(result["decodable"])


class TestSuccessCriteriaFlipLayerStable(unittest.TestCase):
    """n_stable_layers = 1 must flip 'layer_stable' to False."""

    def test_layer_stable_false(self):
        rec = dict(_PASSING_RECORD)
        rec["n_stable_layers"] = 1
        result = success_criteria(rec)
        self.assertFalse(result["layer_stable"])


class TestSuccessCriteriaFlipCoherent(unittest.TestCase):
    """mean_contrast_cosine = 0.2 must flip 'coherent' to False."""

    def test_coherent_false(self):
        rec = dict(_PASSING_RECORD)
        rec["mean_contrast_cosine"] = 0.2
        result = success_criteria(rec)
        self.assertFalse(result["coherent"])


class TestSuccessCriteriaConfounded(unittest.TestCase):
    """When script_macro_f1 > 0.8 and topic_macro_f1 < script_macro_f1,
    'not_confounded' must be False."""

    def test_confounded(self):
        rec = dict(_PASSING_RECORD)
        rec["script_macro_f1"] = 0.85
        rec["topic_macro_f1"] = 0.70  # topic < script → confounded
        result = success_criteria(rec)
        self.assertFalse(result["not_confounded"])

    def test_not_confounded_when_script_below_threshold(self):
        rec = dict(_PASSING_RECORD)
        rec["script_macro_f1"] = 0.75  # ≤ 0.8 → condition fails → not_confounded True
        rec["topic_macro_f1"] = 0.60
        result = success_criteria(rec)
        self.assertTrue(result["not_confounded"])

    def test_not_confounded_when_topic_above_script(self):
        rec = dict(_PASSING_RECORD)
        rec["script_macro_f1"] = 0.90
        rec["topic_macro_f1"] = 0.92  # topic ≥ script → not dominated → not_confounded True
        result = success_criteria(rec)
        self.assertTrue(result["not_confounded"])


class TestSuccessCriteriaPersists(unittest.TestCase):
    """macro_f1_centered at 0.50 is below threshold of 0.55."""

    def test_persists_after_controls_false(self):
        rec = dict(_PASSING_RECORD)
        rec["macro_f1_centered"] = 0.50
        result = success_criteria(rec)
        self.assertFalse(result["persists_after_controls"])


class TestSuccessCriteriaCustomTransferThreshold(unittest.TestCase):
    """record['transfer_threshold'] overrides the default 0.5."""

    def test_custom_threshold_tighter(self):
        rec = dict(_PASSING_RECORD)
        rec["heldout_macro_f1"] = 0.60
        rec["transfer_threshold"] = 0.70  # tighter → should fail
        result = success_criteria(rec)
        self.assertFalse(result["transfers"])

    def test_custom_threshold_looser(self):
        rec = dict(_PASSING_RECORD)
        rec["heldout_macro_f1"] = 0.35
        rec["transfer_threshold"] = 0.30  # looser → should pass
        result = success_criteria(rec)
        self.assertTrue(result["transfers"])


if __name__ == "__main__":
    unittest.main()
