import unittest
from src.shared_utils.text import detect_script, sentence_split
from src.shared_utils.registry import region_factors
from src.shared_utils.io import load_config
from src.tests.helpers import CONFIG

class TestText(unittest.TestCase):
    def test_script(self):
        self.assertEqual(detect_script("hello world"), "LATIN")
        self.assertEqual(detect_script("مرحبا"), "ARABIC")
    def test_sentence_split(self):
        self.assertEqual(len(sentence_split("One sentence here is quite long indeed. Two sentence here is also quite long indeed!")), 2)
    def test_region_factors(self):
        cfg = load_config(CONFIG)
        f = region_factors("Arabic_Egypt", cfg)
        self.assertEqual(f["base_language"], "Arabic")
        self.assertEqual(f["region"], "Egypt")
        self.assertEqual(f["language_region"], "Arabic_Egypt")

if __name__ == "__main__":
    unittest.main()
