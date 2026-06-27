import unittest
from src.shared_utils.text import detect_script, sentence_split, content_token_offsets
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


class FakeTokenizer:
    """Minimal fake tokenizer for testing content_token_offsets."""
    def __init__(self, input_ids, offset_mapping, all_special_ids):
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping
        self.all_special_ids = all_special_ids

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        """Return mock tokenization result."""
        return {
            "input_ids": self.input_ids,
            "offset_mapping": self.offset_mapping
        }


class TestContentMask(unittest.TestCase):
    """Test suite for content_token_offsets function."""

    def test_content_mask_no_answer_basic(self):
        """Test that special tokens and (0,0) offsets are marked False, others True."""
        # Text: "I am Paris"
        # Tokens: [CLS] "I" "am" "Paris" [SEP]
        # IDs: 101 1 2 100 102
        # Offsets: (0,0) (0,1) (2,4) (5,10) (0,0)
        tokenizer = FakeTokenizer(
            input_ids=[101, 1, 2, 100, 102],
            offset_mapping=[(0, 0), (0, 1), (2, 4), (5, 10), (0, 0)],
            all_special_ids={101, 102}
        )
        text = "I am Paris"
        mask = content_token_offsets(tokenizer, text, answer=None)

        # [CLS] is special (id 101) → False
        self.assertFalse(mask[0])
        # "I" is content (not special, offset (0,1)) → True
        self.assertTrue(mask[1])
        # "am" is content (not special, offset (2,4)) → True
        self.assertTrue(mask[2])
        # "Paris" is content (not special, offset (5,10)) → True
        self.assertTrue(mask[3])
        # [SEP] is special (id 102) → False
        self.assertFalse(mask[4])

    def test_content_mask_with_answer_overlap(self):
        """Test that tokens overlapping the answer substring are marked False."""
        tokenizer = FakeTokenizer(
            input_ids=[101, 1, 2, 100, 102],
            offset_mapping=[(0, 0), (0, 1), (2, 4), (5, 10), (0, 0)],
            all_special_ids={101, 102}
        )
        text = "I am Paris"
        # answer="Paris" has span (5, 10) in the text
        mask = content_token_offsets(tokenizer, text, answer="Paris")

        # [CLS] is special → False
        self.assertFalse(mask[0])
        # "I" does not overlap (0,1) with (5,10) → True
        self.assertTrue(mask[1])
        # "am" does not overlap (2,4) with (5,10) → True
        self.assertTrue(mask[2])
        # "Paris" overlaps (5,10) with (5,10) → False
        self.assertFalse(mask[3])
        # [SEP] is special → False
        self.assertFalse(mask[4])

    def test_content_mask_answer_not_found(self):
        """Test behavior when answer substring is not found in text."""
        tokenizer = FakeTokenizer(
            input_ids=[101, 1, 2, 100, 102],
            offset_mapping=[(0, 0), (0, 1), (2, 4), (5, 10), (0, 0)],
            all_special_ids={101, 102}
        )
        text = "I am Paris"
        # answer="London" is not in text → should be treated as no answer
        mask = content_token_offsets(tokenizer, text, answer="London")

        # Should behave like answer=None for content tokens
        # [CLS] is special → False
        self.assertFalse(mask[0])
        # "I", "am", "Paris" are all content → True
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        # [SEP] is special → False
        self.assertFalse(mask[4])

    def test_content_mask_zero_offset_is_special(self):
        """Test that tokens with (0,0) offset are marked False."""
        tokenizer = FakeTokenizer(
            input_ids=[101, 1, 2],
            offset_mapping=[(0, 0), (0, 1), (0, 0)],
            all_special_ids={101}
        )
        text = "I am"
        mask = content_token_offsets(tokenizer, text, answer=None)

        # Token with (0,0) should be False even if not in all_special_ids
        self.assertFalse(mask[0])  # id 101, special
        self.assertTrue(mask[1])   # id 1, content
        self.assertFalse(mask[2])  # id 2, offset (0,0)

    def test_content_mask_partial_answer_overlap(self):
        """Test that tokens partially overlapping the answer are marked False."""
        # Text: "I am Paris today"
        # Tokens: [CLS] "I" "am" "Paris" "today" [SEP]
        # IDs: 101 1 2 100 200 102
        # Offsets: (0,0) (0,1) (2,4) (5,10) (11,16) (0,0)
        tokenizer = FakeTokenizer(
            input_ids=[101, 1, 2, 100, 200, 102],
            offset_mapping=[(0, 0), (0, 1), (2, 4), (5, 10), (11, 16), (0, 0)],
            all_special_ids={101, 102}
        )
        text = "I am Paris today"
        # answer="Paris t" has span (5, 12) - overlaps both "Paris" and "today"
        mask = content_token_offsets(tokenizer, text, answer="Paris t")

        self.assertFalse(mask[0])  # [CLS] special
        self.assertTrue(mask[1])   # "I" no overlap
        self.assertTrue(mask[2])   # "am" no overlap
        self.assertFalse(mask[3])  # "Paris" overlaps (5,10) with (5,12)
        self.assertFalse(mask[4])  # "today" overlaps (11,16) with (5,12)
        self.assertFalse(mask[5])  # [SEP] special

if __name__ == "__main__":
    unittest.main()
