"""Tests for embedding extraction + analysis (no model/GPU)."""
import os, tempfile, unittest
import numpy as np
from tests.helpers import SCRIPTS_DIR, CONFIG_PATH  # noqa: F401 (ensures sys.path setup)
from shared_utils.activation_extraction import get_embedding_module
from shared_utils.embeddings import (
    build_group_map, save_embedding_store, load_embedding_store,
)
from shared_utils.data import load_config


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


class TestEmbeddingStore(unittest.TestCase):
    def test_build_group_map_regions(self):
        cfg = load_config(CONFIG_PATH)
        gm = build_group_map(cfg, "flores", ["Arabic_Egypt", "French_France"])
        self.assertEqual(gm["Arabic_Egypt"], "Arabic")
        self.assertEqual(gm["French_France"], "French")

    def test_build_group_map_topics(self):
        cfg = load_config(CONFIG_PATH)
        gm = build_group_map(cfg, "topics", ["politics", "sports"])
        self.assertEqual(gm["politics"], "politics")

    def test_save_and_load_roundtrip(self):
        acts = {
            "Arabic_Egypt": {"embed": np.ones((3, 4)), 0: np.full((3, 4), 2.0)},
            "French_France": {"embed": np.zeros((2, 4)), 0: np.full((2, 4), 5.0)},
        }
        with tempfile.TemporaryDirectory() as d:
            save_embedding_store(d, acts, ["embed", 0], "lang_",
                                 {"Arabic_Egypt": "Arabic", "French_France": "French"})
            self.assertTrue(os.path.exists(os.path.join(d, "layer_embed.npz")))
            self.assertTrue(os.path.exists(os.path.join(d, "layer_000.npz")))
            by_layer, meta = load_embedding_store(d)
        # embed-layer mean of ones == 1.0
        np.testing.assert_allclose(by_layer["embed"]["lang_Arabic_Egypt"], np.ones(4))
        np.testing.assert_allclose(by_layer[0]["lang_French_France"], np.full(4, 5.0))
        self.assertEqual(meta["groups"]["lang_Arabic_Egypt"], "Arabic")


from shared_utils.embeddings import structure_score, depth_structure, cluster_embeddings


def _two_clusters():
    # 3 vectors near +x, 3 near -x in 4-D -> two clear groups
    base = {
        "a1": np.array([1., 0, 0, 0]), "a2": np.array([0.9, 0.1, 0, 0]),
        "a3": np.array([0.95, 0, 0.1, 0]),
        "b1": np.array([-1., 0, 0, 0]), "b2": np.array([-0.9, 0.1, 0, 0]),
        "b3": np.array([-0.95, 0, 0.1, 0]),
    }
    groups = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "b3": "B"}
    return base, groups


class TestStructureMetrics(unittest.TestCase):
    def test_structure_score_separable(self):
        emb, groups = _two_clusters()
        s = structure_score(emb, groups)
        self.assertGreater(s["silhouette"], 0.5)
        self.assertGreater(s["within_minus_cross"], 0.5)
        self.assertEqual(s["n"], 6)

    def test_structure_score_too_few(self):
        s = structure_score({"a": np.ones(4), "b": np.ones(4)}, {"a": "A", "b": "B"})
        self.assertIsNone(s["silhouette"])

    def test_depth_structure_orders_embed_first(self):
        emb, groups = _two_clusters()
        by_layer = {0: emb, "embed": emb}
        rows = depth_structure(by_layer, groups)
        self.assertEqual(rows[0]["layer"], "embed")
        self.assertEqual(rows[1]["layer"], "0")

    def test_cluster_recovers_groups(self):
        emb, groups = _two_clusters()
        r = cluster_embeddings(emb, n_clusters=2, group_map=groups)
        self.assertAlmostEqual(r["ari"], 1.0, places=5)
        # New contract: linkage matrix and labels are returned for dendrogram rendering
        self.assertIn("linkage", r)
        self.assertGreater(len(r["linkage"]), 0)
        self.assertIn("labels", r)
        self.assertGreater(len(r["labels"]), 0)


import importlib.util
from tests.helpers import SCRIPTS_DIR


def _fake_extract_with_embed(model, tok, sentences, layers, max_seq_len, batch_size,
                             desc="x", include_embedding=False):
    D = 4
    rng = np.random.default_rng(abs(hash(tuple(sentences))) % (2**32))
    base = rng.standard_normal((len(sentences), D))
    out = {l: base + l * 0.01 for l in layers}
    if include_embedding:
        out["embed"] = base - 1.0
    return out


class TestZeroFourEmbeddingStore(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "cv_emb", os.path.join(SCRIPTS_DIR, "04_compute_vectors.py"))
        self.cv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cv)
        self.cv.extract_activations_batch = _fake_extract_with_embed

    def test_process_flat_writes_embedding_store(self):
        cfg = load_config(CONFIG_PATH)
        cfg["model"]["include_embedding_layer"] = True
        d = tempfile.mkdtemp()
        data = os.path.join(d, "flores")
        os.makedirs(data)
        for r in ["Arabic_Egypt", "French_France", "German_Germany"]:
            with open(os.path.join(data, f"{r}.txt"), "w") as f:
                f.write("\n".join(f"{r} s{i}" for i in range(4)) + "\n")
        self.cv.process_flat(None, None, cfg, [0], "flores", data, "lang_",
                             os.path.join(d, "vectors"), None,
                             emb_base=os.path.join(d, "embeddings"))
        store = os.path.join(d, "embeddings", "flores")
        self.assertTrue(os.path.exists(os.path.join(store, "layer_embed.npz")))
        z = np.load(os.path.join(store, "layer_embed.npz"))
        self.assertIn("lang_Arabic_Egypt", z.files)


if __name__ == "__main__":
    unittest.main()
