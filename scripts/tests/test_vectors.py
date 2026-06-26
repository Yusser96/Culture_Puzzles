"""Tests for 04_compute_vectors.py — loaders, DiffMean, content dedup, save layout.

Uses a fake activation extractor so no model/GPU is needed.
"""

import os
import tempfile
import unittest

import numpy as np

from tests.helpers import load_script

D = 8


def fake_extract(model, tok, sentences, layers, max_seq_len, batch_size, desc="x"):
    """Deterministic per-sentence vectors: identical corpora -> identical activations."""
    seed = abs(hash(tuple(sentences))) % (2 ** 32)
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((len(sentences), D))
    return {l: base + l * 0.01 for l in layers}


class VectorTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cv = load_script("04_compute_vectors.py", "compute_vectors_mod")
        cls.cv.extract_activations_batch = fake_extract  # patch module-bound name

    def _write(self, path, lines):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


class TestLoaders(VectorTestBase):
    def test_load_flat_source(self):
        d = tempfile.mkdtemp()
        self._write(os.path.join(d, "Arabic_Egypt.txt"), ["a", "b"])
        self._write(os.path.join(d, "French_France.txt"), ["c"])
        out = self.cv.load_flat_source(d)
        self.assertEqual(set(out), {"Arabic_Egypt", "French_France"})
        self.assertEqual(out["Arabic_Egypt"], ["a", "b"])

    def test_load_puzzles_source(self):
        d = tempfile.mkdtemp()
        self._write(os.path.join(d, "English_USA", "original.txt"), ["o1", "o2"])
        self._write(os.path.join(d, "English_USA", "translation.txt"), ["t1", "t2"])
        orig = self.cv.load_puzzles_source(d, "original")
        self.assertEqual(orig, {"English_USA": ["o1", "o2"]})

    def test_load_cultural_source(self):
        d = tempfile.mkdtemp()
        self._write(os.path.join(d, "politics", "English_USA.txt"), ["p1"])
        self._write(os.path.join(d, "sports", "French_France.txt"), ["s1"])
        out = self.cv.load_cultural_source(d)
        self.assertIn(("politics", "English_USA"), out)
        self.assertIn(("sports", "French_France"), out)


class TestDiffMean(VectorTestBase):
    def _acts(self, keys, layers):
        return {k: fake_extract(None, None, [f"{k}_{i}" for i in range(5)], layers, 0, 0)
                for k in keys}

    def test_unit_norm(self):
        layers = [0, 1]
        acts = self._acts(["a", "b", "c"], layers)
        vecs = self.cv.diffmean_set(acts, layers)
        for layer in layers:
            for k in ("a", "b", "c"):
                self.assertAlmostEqual(float(np.linalg.norm(vecs[layer][k])), 1.0, places=5)

    def test_identical_inputs_identical_vectors(self):
        layers = [0]
        shared = fake_extract(None, None, ["same1", "same2"], layers, 0, 0)
        acts = {"x": shared, "y": shared, "z": self._acts(["z"], layers)["z"]}
        vecs = self.cv.diffmean_set(acts, layers)
        np.testing.assert_allclose(vecs[0]["x"], vecs[0]["y"])


class TestExtractDedup(VectorTestBase):
    def test_shared_corpora_extracted_once(self):
        calls = {"n": 0}
        layers = [0]

        def counting_extract(*a, **k):
            calls["n"] += 1
            return fake_extract(*a, **k)
        self.cv.extract_activations_batch = counting_extract

        spa = ["x1", "x2", "x3"]
        key_to_sents = {
            "Spanish_Argentina": spa,
            "Spanish_Cuba": list(spa),   # identical content
            "Arabic_Egypt": ["e1", "e2"],
        }
        acts, shared = self.cv.extract_for_keys(None, None, key_to_sents, layers, 0, 0)

        # 3 keys but only 2 unique corpora -> 2 extractions.
        self.assertEqual(calls["n"], 2)
        self.assertIn("Spanish_Argentina", shared)
        self.assertEqual(set(shared["Spanish_Argentina"]),
                         {"Spanish_Argentina", "Spanish_Cuba"})
        np.testing.assert_allclose(acts["Spanish_Argentina"][0], acts["Spanish_Cuba"][0])

        self.cv.extract_activations_batch = fake_extract  # restore


class TestSaveVectorSet(VectorTestBase):
    def test_layout_keys_and_mirror(self):
        layers = [0, 1]
        acts = {k: fake_extract(None, None, [f"{k}{i}" for i in range(4)], layers, 0, 0)
                for k in ("Arabic_Egypt", "French_France")}
        vecs = self.cv.diffmean_set(acts, layers)
        out = tempfile.mkdtemp()
        mirror = tempfile.mkdtemp()
        meta = self.cv.save_vector_set(out, vecs, acts, layers, "lang_",
                                       {"Arabic_Egypt": ["a", "b"]},
                                       {"source": "flores"}, mirror)
        npz = np.load(os.path.join(out, "layer_000.npz"))
        self.assertEqual(set(npz.files), {"lang_Arabic_Egypt", "lang_French_France"})
        self.assertTrue(os.path.exists(os.path.join(out, "raw_means",
                                                    "Arabic_Egypt_layer_000.npy")))
        self.assertTrue(os.path.exists(os.path.join(mirror, "layer_001.npz")))
        self.assertEqual(meta["source"], "flores")
        self.assertIn("shared_groups", meta)


if __name__ == "__main__":
    unittest.main()
