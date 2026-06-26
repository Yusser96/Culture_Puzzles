"""Tests for the riddles_config.yaml registry and shared_utils/data.py helpers."""

import os
import tempfile
import unittest

from tests.helpers import CONFIG_PATH

from shared_utils.data import (
    load_config, load_registry, save_jsonl, load_jsonl,
)


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG_PATH)
        cls.reg = load_registry(cls.cfg)

    def test_46_unique_regions(self):
        keys = [r["key"] for r in self.reg]
        self.assertEqual(len(keys), 46)
        self.assertEqual(len(set(keys)), 46)

    def test_every_region_has_required_fields(self):
        for r in self.reg:
            for field in ("key", "base_language", "wiki", "flores", "opus"):
                self.assertIn(field, r, f"{r.get('key')} missing {field}")

    def test_eight_canonical_topics(self):
        self.assertEqual(len(self.cfg["cultural_topics"]), 8)
        self.assertEqual(len(self.cfg["topic_label_map"]), 8)
        for t in self.cfg["cultural_topics"].values():
            self.assertIn("seed_en", t)

    def test_arabic_flores_varieties_distinct(self):
        # The 4 Arabic regions must map to 4 distinct FLORES codes.
        arabic = [r["flores"] for r in self.reg if r["base_language"] == "Arabic"]
        self.assertEqual(len(arabic), 4)
        self.assertEqual(len(set(arabic)), 4)

    def test_spanish_shares_one_flores_code(self):
        spanish = [r["flores"] for r in self.reg if r["base_language"] == "Spanish"]
        self.assertTrue(len(spanish) >= 3)
        self.assertEqual(set(spanish), {"spa_Latn"})

    def test_null_sources_present(self):
        # Mongondow has no wiki/flores/opus.
        m = next(r for r in self.reg if r["key"] == "Mongondow_Indonesia")
        self.assertIsNone(m["wiki"])
        self.assertIsNone(m["flores"])
        self.assertIsNone(m["opus"])


class TestJsonl(unittest.TestCase):
    def test_roundtrip_unicode(self):
        rows = [
            {"id": 1, "text": "Tahrir التحرير"},
            {"id": 2, "text": "音樂", "n": None},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            save_jsonl(rows, p)
            back = load_jsonl(p)
        self.assertEqual(rows, back)


if __name__ == "__main__":
    unittest.main()
