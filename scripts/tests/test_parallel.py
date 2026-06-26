"""Tests for 02_collect_parallel.py — dedup-by-code replication into lang_region files."""

import os
import tempfile
import unittest

from tests.helpers import CONFIG_PATH, load_script
from shared_utils.data import load_config, load_sentences, load_json


class TestSaveByRegion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = load_script("02_collect_parallel.py", "collect_parallel_mod")
        cls.reg = load_config(CONFIG_PATH)["lang_regions"]

    def _run(self):
        by_code = {
            "arz_Arab": ["e1", "e2"],   # Egypt only
            "ary_Arab": ["m1"],         # Morocco only
            "apc_Arab": ["s1"],         # Syria only
            "arb_Arab": ["su1"],        # Sudan only
            "spa_Latn": ["x1", "x2", "x3"],  # 3 Spanish regions share
            "eng_Latn": ["en1"],        # 2 English regions share
        }
        d = tempfile.mkdtemp()
        man = self.cp._save_by_region(self.reg, "flores", by_code,
                                      os.path.join(d, "flores"), "flores")
        return d, man

    def test_arabic_varieties_not_shared(self):
        _, man = self._run()
        lr = man["lang_regions"]
        self.assertEqual(lr["Arabic_Egypt"]["code"], "arz_Arab")
        self.assertEqual(lr["Arabic_Egypt"]["shared_with"], [])
        self.assertEqual(lr["Arabic_Syria"]["code"], "apc_Arab")

    def test_spanish_regions_shared(self):
        _, man = self._run()
        sw = set(man["lang_regions"]["Spanish_Argentina"]["shared_with"])
        self.assertEqual(sw, {"Spanish_Cuba", "Spanish_Peru"})

    def test_null_flores_region_excluded(self):
        _, man = self._run()
        # Konkani has flores: null -> no file/manifest entry.
        self.assertNotIn("Konkani_India", man["lang_regions"])

    def test_files_written_with_replicated_content(self):
        d, man = self._run()
        cuba = os.path.join(d, "flores", "Spanish_Cuba.txt")
        peru = os.path.join(d, "flores", "Spanish_Peru.txt")
        self.assertEqual(load_sentences(cuba), ["x1", "x2", "x3"])
        self.assertEqual(load_sentences(cuba), load_sentences(peru))

    def test_manifest_file_saved(self):
        d, _ = self._run()
        man = load_json(os.path.join(d, "flores", "manifest.json"))
        self.assertEqual(man["dataset"], "flores")


if __name__ == "__main__":
    unittest.main()
