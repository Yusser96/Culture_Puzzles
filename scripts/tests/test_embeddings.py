"""Tests for embedding extraction + analysis (no model/GPU)."""
import os, tempfile, unittest
import numpy as np
from tests.helpers import SCRIPTS_DIR  # noqa: F401 (ensures sys.path setup)
from shared_utils.activation_extraction import get_embedding_module


class _FakeInner:
    class model:  # noqa: N801
        embed_tokens = "EMBED_MODULE"


class _FakeViaInputEmb:
    def get_input_embeddings(self):
        return "INPUT_EMB"


class TestEmbeddingModule(unittest.TestCase):
    def test_resolves_embed_tokens(self):
        self.assertEqual(get_embedding_module(_FakeInner()), "EMBED_MODULE")

    def test_falls_back_to_input_embeddings(self):
        self.assertEqual(get_embedding_module(_FakeViaInputEmb()), "INPUT_EMB")

    def test_raises_when_absent(self):
        with self.assertRaises(RuntimeError):
            get_embedding_module(object())


if __name__ == "__main__":
    unittest.main()
