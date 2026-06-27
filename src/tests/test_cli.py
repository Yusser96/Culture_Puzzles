"""
src/tests/test_cli.py
=====================
Unit tests for the unified CLI dispatcher (src/run.py).

Tests verify that:
1. STEPS dict exists and contains exactly 12 step names.
2. Each step maps to a callable.
3. All required step names are present.
"""

import unittest


class TestCLI(unittest.TestCase):
    """Test the unified CLI module."""

    def test_steps_dict_exists(self):
        """STEPS dict should be importable from src.run."""
        from src.run import STEPS
        self.assertIsNotNone(STEPS)
        self.assertIsInstance(STEPS, dict)

    def test_steps_has_12_entries(self):
        """STEPS should map exactly 12 step names."""
        from src.run import STEPS
        expected_steps = {
            "collect",
            "metadata",
            "extract",
            "normalize",
            "probes",
            "directions",
            "cross",
            "flores-decomp",
            "rep-similarity",
            "steering",
            "data-stats",
            "report",
        }
        self.assertEqual(set(STEPS.keys()), expected_steps)

    def test_steps_all_callable(self):
        """Each step should map to a callable."""
        from src.run import STEPS
        for step_name, step_func in STEPS.items():
            self.assertTrue(
                callable(step_func),
                f"Step '{step_name}' maps to non-callable: {type(step_func)}"
            )

    def test_import_does_not_load_model(self):
        """Importing src.run should not require a model."""
        # This test simply ensures the import succeeds and doesn't
        # trigger model loading at module level.
        try:
            import src.run  # noqa
        except Exception as e:
            self.fail(f"Importing src.run raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
