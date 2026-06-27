"""
src/tests/test_collect.py
Tests for the collect sub-pipeline (Task 9).

Unit tests that do NOT require network access or HuggingFace datasets.
"""

import os
import unittest

from src.shared_utils.riddles import parse_lang_region_key


class TestCollect(unittest.TestCase):
    def test_key_parse(self):
        self.assertEqual(
            parse_lang_region_key("Cultural Riddles Benchmark [Arabic_Egypt].xlsx"),
            "Arabic_Egypt",
        )

    def test_key_parse_uganda(self):
        self.assertEqual(
            parse_lang_region_key("Cultural Riddles Benchmark [Luganda_Uganda].xlsx"),
            "Luganda_Uganda",
        )

    def test_key_parse_no_bracket(self):
        self.assertIsNone(parse_lang_region_key("SomeOtherFile.xlsx"))

    def test_key_parse_space_in_key(self):
        # Sinhala has a space in the region: "Sinhala_Sri Lanka"
        self.assertEqual(
            parse_lang_region_key(
                "Cultural Riddles Benchmark [Sinhala_Sri Lanka].xlsx"
            ),
            "Sinhala_Sri Lanka",
        )


if __name__ == "__main__":
    unittest.main()
