# Embedding-layer Extraction + Structure & Depth Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the token-embedding-layer representation (`embed_tokens` output, pre-block-0), persist per-region/per-topic mean embeddings for every source, and add an analysis script for embedding-space structure and across-depth structure.

**Architecture:** Extend the existing single forward pass (`shared_utils/activation_extraction.py`, used by `04_compute_vectors.py`) to also capture the embedding layer under the sentinel key `"embed"`. A new pure-function module `shared_utils/embeddings.py` saves/loads a consolidated embedding store and computes structure metrics. A new CLI `07_analyze_embeddings.py` turns those into CSVs + figures. Steering-vector outputs and `05`/`06` are untouched.

**Tech Stack:** Python 3.9, NNsight + transformers (Qwen3), numpy, scipy, scikit-learn, matplotlib. Tests use `unittest` (stdlib). Virtualenv at `.venv`.

## Global Constraints

- Tests are **unittest**, run from the `scripts/` dir: `cd scripts && ../.venv/bin/python -m unittest tests.<module> -v`.
- Scripts run with **cwd = `scripts/`**; configs are under `configs/`.
- Python **3.9**; no bash-4 features in shell scripts (macOS bash 3.2).
- Pure functions go in `shared_utils/embeddings.py` so they unit-test **without a model/GPU**. The actual `embed_tokens` capture (NNsight trace) is validated only by the smoke run (Task 7), mirroring how `extract_activations_batch` is already untested at unit level.
- Embedding store holds **raw mean representations** (NOT normalized DiffMean directions).
- Sentinel embedding-layer key is the string `"embed"`; transformer layers are ints. Layer labels are opaque downstream.
- Group labels: region-keyed sources group by `base_language` (from the `lang_regions` registry); cultural sources group by topic.
- `env.sh` is tokens-only; device/model come from config.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

---

### Task 1: Embedding-layer capture in activation extraction

**Files:**
- Modify: `scripts/shared_utils/activation_extraction.py`
- Test: `scripts/tests/test_embeddings.py` (create)

**Interfaces:**
- Produces: `get_embedding_module(model)` → the model's embedding module/envoy; `extract_activations_batch(..., include_embedding: bool = False)` → adds key `"embed"` to the returned `{layer: ndarray}` dict when true.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: FAIL — `ImportError: cannot import name 'get_embedding_module'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/shared_utils/activation_extraction.py`, add after the imports:

```python
def get_embedding_module(model):
    """Return the token-embedding module/envoy (embed_tokens), with fallbacks."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    inner = getattr(model, "_model", None)
    if inner is not None and hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
        return inner.model.embed_tokens
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    raise RuntimeError(f"Cannot locate embedding module on {type(model)}")
```

Then change the `extract_activations_batch` signature and body. Replace the signature line:

```python
def extract_activations_batch(
    model,
    tokenizer,
    sentences: List[str],
    layers: List[int],
    max_seq_len: int = 128,
    batch_size: int = 4,
    desc: str = "Batches",
    include_embedding: bool = False,
) -> Dict[int, np.ndarray]:
```

Replace `layer_activations = {l: [] for l in layers}` with:

```python
    probe_keys = list(layers) + (["embed"] if include_embedding else [])
    layer_activations = {k: [] for k in probe_keys}
```

Inside the `with model.trace(...)` block, after the existing `for layer_idx in layers:` loop that fills `saved`, add:

```python
            if include_embedding:
                saved["embed"] = get_embedding_module(model).output.save()
```

Replace the post-trace `for layer_idx in layers:` pooling loop header with `for layer_idx in probe_keys:` (the body that mean-pools `saved[layer_idx]` is unchanged). Replace the final `for l in layers:` stacking loop header with `for l in probe_keys:`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: PASS (3 tests, OK).

- [ ] **Step 5: Commit**

```bash
git add scripts/shared_utils/activation_extraction.py scripts/tests/test_embeddings.py
git commit -m "feat: capture embed_tokens layer in activation extraction

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Embedding store save/load + group map

**Files:**
- Create: `scripts/shared_utils/embeddings.py`
- Test: `scripts/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `shared_utils.data.save_json`, `load_json`.
- Produces:
  - `build_group_map(cfg, source, keys)` → `{key: group_label}` (base_language for region sources; topic for `topics`/`culture`).
  - `save_embedding_store(out_dir, acts_by_key, layers, key_prefix, group_map)` → writes `layer_<L>.npz` (`L` = `"embed"` or `"000"`...) with arrays keyed `key_prefix+key`, plus `metadata.json` (`source` optional, `layers`, `keys`, `groups`).
  - `load_embedding_store(store_dir)` → `(by_layer, meta)` where `by_layer = {layer_label: {prefixed_key: ndarray}}`, `layer_label` is `"embed"` or `int`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_embeddings.py`:

```python
from shared_utils.embeddings import (
    build_group_map, save_embedding_store, load_embedding_store,
)
from shared_utils.data import load_config
from tests.helpers import CONFIG_PATH


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings.TestEmbeddingStore -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared_utils.embeddings'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/shared_utils/embeddings.py`:

```python
"""
shared_utils/embeddings.py
Embedding-store I/O and group maps for embedding-layer analysis.

The store holds RAW mean representations (not normalized DiffMean directions):
  <store_dir>/layer_embed.npz, layer_000.npz, ...   keys = "<prefix><key>"
  <store_dir>/metadata.json                          layers, keys, groups
"""

import os
from typing import Dict, List

import numpy as np

from shared_utils.data import save_json, load_json


def _layer_label(layer) -> str:
    return "embed" if layer == "embed" else f"{int(layer):03d}"


def build_group_map(cfg: dict, source: str, keys: List[str]) -> Dict[str, str]:
    """Map each key to its grouping label: base_language for region sources,
    the topic for 'topics'/'culture' sources."""
    if source in ("topics", "culture"):
        out = {}
        for k in keys:
            out[k] = k.split("_")[0] if source == "culture" else k
        return out
    base = {r["key"]: r["base_language"] for r in cfg.get("lang_regions", [])}
    return {k: base.get(k, "UNKNOWN") for k in keys}


def save_embedding_store(out_dir, acts_by_key, layers, key_prefix, group_map):
    """Write per-key MEAN representation at each layer (incl. 'embed')."""
    os.makedirs(out_dir, exist_ok=True)
    for layer in layers:
        arrays = {f"{key_prefix}{k}": acts_by_key[k][layer].mean(axis=0)
                  for k in acts_by_key}
        np.savez(os.path.join(out_dir, f"layer_{_layer_label(layer)}.npz"), **arrays)
    meta = {
        "layers": ["embed" if l == "embed" else int(l) for l in layers],
        "keys": sorted(f"{key_prefix}{k}" for k in acts_by_key),
        "groups": {f"{key_prefix}{k}": group_map.get(k, "UNKNOWN") for k in acts_by_key},
    }
    save_json(meta, os.path.join(out_dir, "metadata.json"))
    return meta


def load_embedding_store(store_dir):
    """Return ({layer_label: {prefixed_key: vec}}, metadata). layer_label is 'embed' or int."""
    meta = load_json(os.path.join(store_dir, "metadata.json"))
    by_layer = {}
    for fn in sorted(os.listdir(store_dir)):
        if not (fn.startswith("layer_") and fn.endswith(".npz")):
            continue
        tag = fn[len("layer_"):-len(".npz")]
        label = "embed" if tag == "embed" else int(tag)
        z = np.load(os.path.join(store_dir, fn))
        by_layer[label] = {k: z[k] for k in z.files}
    return by_layer, meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings.TestEmbeddingStore -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: PASS (3 tests, OK).

- [ ] **Step 5: Commit**

```bash
git add scripts/shared_utils/embeddings.py scripts/tests/test_embeddings.py
git commit -m "feat: embedding store save/load + group map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Structure & depth metrics

**Files:**
- Modify: `scripts/shared_utils/embeddings.py`
- Test: `scripts/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `shared_utils.vectors.cosine_similarity_matrix`, `pairwise_distance_matrix`, `cosine_similarity`; `shared_utils.clustering.hierarchical_cluster`, `cluster_agreement_scores`.
- Produces:
  - `structure_score(emb_by_key, group_map)` → `{"silhouette": float|None, "within_minus_cross": float|None, "n": int}`.
  - `depth_structure(by_layer, group_map)` → `[{"layer": str, "silhouette": float|None, "within_minus_cross": float|None, "n": int}, ...]` (layer `"embed"` first, then ascending ints).
  - `cluster_embeddings(emb_by_key, n_clusters, group_map)` → `{"assignments": {key: int}, "ari": float, "nmi": float}`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings.TestStructureMetrics -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: FAIL — `ImportError: cannot import name 'structure_score'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/shared_utils/embeddings.py`:

```python
from shared_utils.vectors import cosine_similarity, pairwise_distance_matrix
from shared_utils.clustering import hierarchical_cluster, cluster_agreement_scores


def structure_score(emb_by_key, group_map):
    keys = [k for k in sorted(emb_by_key) if k in group_map]
    if len(keys) < 3:
        return {"silhouette": None, "within_minus_cross": None, "n": len(keys)}
    X = np.stack([np.asarray(emb_by_key[k], dtype=float) for k in keys])
    groups = [group_map[k] for k in keys]

    sil = None
    if len(set(groups)) >= 2 and len(keys) > len(set(groups)):
        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(X, groups, metric="cosine"))
        except Exception:
            sil = None

    wi = wc = cr = cc = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = cosine_similarity(X[i], X[j])
            if groups[i] == groups[j]:
                wi += c; wc += 1
            else:
                cr += c; cc += 1
    wmc = (wi / wc - cr / cc) if wc and cc else None
    return {"silhouette": sil, "within_minus_cross": wmc, "n": len(keys)}


def _layer_sort_key(label):
    return (0, -1) if label == "embed" else (1, int(label))


def depth_structure(by_layer, group_map):
    rows = []
    for label in sorted(by_layer, key=_layer_sort_key):
        s = structure_score(by_layer[label], group_map)
        rows.append({"layer": "embed" if label == "embed" else str(label), **s})
    return rows


def cluster_embeddings(emb_by_key, n_clusters, group_map):
    keys = [k for k in sorted(emb_by_key) if k in group_map]
    vecs = {k: np.asarray(emb_by_key[k], dtype=float) for k in keys}
    dm, labels = pairwise_distance_matrix(vecs, metric="cosine")
    res = hierarchical_cluster(dm, labels, n_clusters=min(n_clusters, len(labels)))
    assignments = dict(zip(labels, res["cluster_labels"]))
    true = [group_map[k] for k in labels]
    scores = cluster_agreement_scores([assignments[k] for k in labels], true)
    return {"assignments": assignments, **scores}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings.TestStructureMetrics -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: PASS (4 tests, OK).

- [ ] **Step 5: Commit**

```bash
git add scripts/shared_utils/embeddings.py scripts/tests/test_embeddings.py
git commit -m "feat: embedding structure + depth metrics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire 04 to emit the embedding store + config

**Files:**
- Modify: `scripts/04_compute_vectors.py`
- Modify: `scripts/configs/riddles_config.yaml`
- Modify: `scripts/configs/riddles_config_1.7b.yaml`
- Test: `scripts/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `shared_utils.embeddings.save_embedding_store`, `build_group_map`; `extract_activations_batch(include_embedding=...)`.
- Produces: `<va_embeddings_dir>/<source>/layer_*.npz` for every source.

- [ ] **Step 1: Add config keys (no test — config plumbing folded into this task)**

In `scripts/configs/riddles_config.yaml`, under `model:` add `include_embedding_layer: true`. Under `analysis:` add `depth_layers: null  # null = all probed layers; or a list like [0, 7, 14, 21, 27]`. Under `paths:` add (using the SSD root already used by the other `va_*` paths):

```yaml
  va_embeddings_dir: "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/vector_analysis/results/embeddings"
  va_analysis_dir:   "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/vector_analysis/results/analysis"
  va_plot_dir:       "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/vector_analysis/results/plots"
```

In `scripts/configs/riddles_config_1.7b.yaml` add the same `model.include_embedding_layer: true` and `analysis.depth_layers: null`, and the same three `paths` entries (same SSD root).

- [ ] **Step 2: Write the failing test**

Append to `scripts/tests/test_embeddings.py` (drives the new `04` helper via a fake extractor that returns an `"embed"` key):

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings.TestZeroFourEmbeddingStore -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: FAIL — `TypeError: process_flat() got an unexpected keyword argument 'emb_base'`.

- [ ] **Step 4: Write minimal implementation**

In `scripts/04_compute_vectors.py`:

Add import near the top imports:

```python
from shared_utils.embeddings import save_embedding_store, build_group_map
```

Change `extract_for_keys` to forward `include_embedding`. Update its signature to:

```python
def extract_for_keys(model, tokenizer, key_to_sents, layers, max_seq_len, batch_size,
                     include_embedding=False):
```

and in its body change the `extract_activations_batch(...)` call to pass
`include_embedding=include_embedding`. (The returned dict then also has an `"embed"` key.)

Add a helper:

```python
def _emb_layers(cfg, layers):
    depth = cfg.get("analysis", {}).get("depth_layers")
    chosen = layers if not depth else [l for l in depth if l in layers]
    return ["embed"] + list(chosen)
```

Change `process_flat` signature to add `emb_base=None` and a `source` group name (its `name` already serves), and after computing `acts`:

```python
def process_flat(model, tok, cfg, layers, name, data_dir, key_prefix, out_base,
                 mirror_base=None, emb_base=None):
    key_to_sents = load_flat_source(data_dir)
    if len(key_to_sents) < 2:
        logger.warning(f"  [{name}] need >=2 classes, found {len(key_to_sents)}; skipping.")
        return None
    inc = cfg["model"].get("include_embedding_layer", False)
    acts, shared = extract_for_keys(
        model, tok, key_to_sents, layers,
        cfg["model"]["max_seq_len"], cfg["model"]["batch_size"], include_embedding=inc,
    )
    vectors = diffmean_set(acts, layers)
    mirror = os.path.join(mirror_base, name) if mirror_base else None
    meta = save_vector_set(os.path.join(out_base, name), vectors, acts, layers,
                           key_prefix, shared, {"source": name}, mirror)
    if inc and emb_base:
        gm = build_group_map(cfg, name, list(key_to_sents.keys()))
        save_embedding_store(os.path.join(emb_base, name), acts, _emb_layers(cfg, layers),
                             key_prefix, gm)
    return meta
```

Apply the same `include_embedding` + `save_embedding_store` pattern to `process_puzzles`
(add `emb_base=None` param; `name = f"puzzles_{variant}"`; group map via `build_group_map(cfg, name, keys)`) and to `process_topics` (add `emb_base=None`; after extracting `acts`, write two stores: `topics` keyed by topic and `culture` keyed by `topic_region`). For `process_topics`, build the topic-keyed activations by pooling per topic before saving, mirroring how topic vectors pool — concretely, after the existing extraction add:

```python
    inc = cfg["model"].get("include_embedding_layer", False)
    if inc and emb_base:
        emb_layers = _emb_layers(cfg, layers)
        # culture: each (topic, region) pair
        culture_acts = {f"{t}_{r}": acts[(t, r)] for (t, r) in acts}
        save_embedding_store(os.path.join(emb_base, "culture"), culture_acts, emb_layers,
                             "culture_", build_group_map(cfg, "culture", list(culture_acts)))
        # topics: pool all regions per topic, per layer
        topic_acts = {}
        for t in topics:
            per_layer = {}
            for L in emb_layers:
                parts = [acts[(tt, r)][L] for (tt, r) in acts if tt == t]
                if parts:
                    per_layer[L] = np.concatenate(parts, axis=0)
            if per_layer:
                topic_acts[t] = per_layer
        save_embedding_store(os.path.join(emb_base, "topics"), topic_acts, emb_layers,
                             "topic_", build_group_map(cfg, "topics", list(topic_acts)))
```

Finally, in `main`, read `emb_dir = cfg["paths"]["va_embeddings_dir"]` and pass
`emb_base=emb_dir` to every `process_flat`/`process_puzzles`/`process_topics` call.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings.TestZeroFourEmbeddingStore -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"`
Expected: PASS.

- [ ] **Step 6: Run the full embeddings + vectors test suites (no regressions)**

Run: `cd scripts && ../.venv/bin/python -m unittest tests.test_embeddings tests.test_vectors -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn" | tail -5`
Expected: OK (all pass).

- [ ] **Step 7: Commit**

```bash
git add scripts/04_compute_vectors.py scripts/configs/riddles_config.yaml scripts/configs/riddles_config_1.7b.yaml scripts/tests/test_embeddings.py
git commit -m "feat: 04 emits per-source embedding store; config paths + flag

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `07_analyze_embeddings.py` CLI + figures

**Files:**
- Create: `scripts/07_analyze_embeddings.py`
- Test: covered by Task 3 unit tests (pure funcs) + Task 7 smoke (figures/CSVs).

**Interfaces:**
- Consumes: `shared_utils.embeddings.{load_embedding_store, structure_score, depth_structure, cluster_embeddings}`; `shared_utils.vectors.cosine_similarity_matrix`; `shared_utils.data.{load_config, ensure_dirs, setup_logging, save_json}`.
- Produces: CSVs in `va_analysis_dir`, figures in `va_plot_dir`.

- [ ] **Step 1: Write the script**

Create `scripts/07_analyze_embeddings.py`:

```python
#!/usr/bin/env python3
"""
07_analyze_embeddings.py
========================
Structure (embedding layer) + depth (embed -> deep layers) analysis of the
per-region/per-topic embedding store written by 04_compute_vectors.py.

Outputs CSVs to va_analysis_dir and figures to va_plot_dir.

Usage: python 07_analyze_embeddings.py [--config configs/riddles_config.yaml]
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import load_config, ensure_dirs, setup_logging, save_json
from shared_utils.embeddings import (
    load_embedding_store, structure_score, depth_structure, cluster_embeddings,
)
from shared_utils.vectors import cosine_similarity_matrix

logger = setup_logging("analyze_embeddings")

REGION_SOURCES = ["flores", "opus100", "puzzles_original", "puzzles_translation"]


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def analyze_source(name, store_dir, analysis_dir, plot_dir):
    if not os.path.isdir(store_dir):
        logger.warning(f"  [{name}] no embedding store; skipping.")
        return None
    by_layer, meta = load_embedding_store(store_dir)
    groups = meta["groups"]
    emb = by_layer.get("embed")
    if emb is None or len(emb) < 3:
        logger.warning(f"  [{name}] <3 keys at embed layer; skipping structure.")
        return None

    # Similarity heatmap (embed layer)
    mat, labels = cosine_similarity_matrix(emb)
    _write_csv(os.path.join(analysis_dir, f"embedding_similarity_{name}.csv"),
               ["vec_a", "vec_b", "cosine_sim"],
               [[labels[i], labels[j], f"{mat[i, j]:.6f}"]
                for i in range(len(labels)) for j in range(len(labels))])
    plt.figure(figsize=(8, 7))
    plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.colorbar(label="cosine"); plt.title(f"Embedding similarity: {name}")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(plot_dir, f"emb_fig1_similarity_{name}.{ext}"))
    plt.close()

    # PCA scatter colored by group
    X = np.stack([emb[k] for k in labels])
    coords = PCA(n_components=2).fit(X)
    xy = coords.transform(X)
    glist = [groups.get(k, "UNKNOWN") for k in labels]
    save_json({"labels": labels, "groups": glist,
               "coordinates": xy.tolist(),
               "explained_variance": coords.explained_variance_ratio_.tolist()},
              os.path.join(analysis_dir, f"embedding_pca_{name}.json"))
    plt.figure(figsize=(8, 7))
    for g in sorted(set(glist)):
        idx = [i for i, gg in enumerate(glist) if gg == g]
        plt.scatter(xy[idx, 0], xy[idx, 1], label=g, s=30)
    plt.legend(fontsize=6, ncol=2); plt.title(f"Embedding PCA: {name}")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(plot_dir, f"emb_fig2_pca_{name}.{ext}"))
    plt.close()

    # Clustering vs base_language
    n_groups = len(set(groups.values()))
    clu = cluster_embeddings(emb, n_clusters=max(2, n_groups), group_map=groups)
    _write_csv(os.path.join(analysis_dir, f"embedding_clusters_{name}.csv"),
               ["key", "cluster", "group"],
               [[k, clu["assignments"][k], groups.get(k, "UNKNOWN")] for k in labels])

    s = structure_score(emb, groups)
    return {"source": name, "silhouette": s["silhouette"],
            "within_minus_cross": s["within_minus_cross"],
            "ari": clu["ari"], "nmi": clu["nmi"], "n": s["n"]}


def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings")
    parser.add_argument("--config", default="configs/riddles_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    emb_dir = cfg["paths"]["va_embeddings_dir"]
    analysis_dir = cfg["paths"]["va_analysis_dir"]
    plot_dir = cfg["paths"]["va_plot_dir"]
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    summary = []
    depth_rows = []
    for name in REGION_SOURCES:
        store_dir = os.path.join(emb_dir, name)
        row = analyze_source(name, store_dir, analysis_dir, plot_dir)
        if row:
            summary.append(row)
            by_layer, meta = load_embedding_store(store_dir)
            for d in depth_structure(by_layer, meta["groups"]):
                depth_rows.append([name, d["layer"], d["silhouette"],
                                   d["within_minus_cross"], d["n"]])

    if summary:
        _write_csv(os.path.join(analysis_dir, "embedding_structure_summary.csv"),
                   ["source", "silhouette", "within_minus_cross", "ari", "nmi", "n"],
                   [[r["source"], r["silhouette"], r["within_minus_cross"],
                     r["ari"], r["nmi"], r["n"]] for r in summary])
    if depth_rows:
        _write_csv(os.path.join(analysis_dir, "depth_structure.csv"),
                   ["source", "layer", "silhouette", "within_minus_cross", "n"], depth_rows)
        # Depth figure: silhouette vs layer, one line per source
        plt.figure(figsize=(9, 6))
        for name in sorted(set(r[0] for r in depth_rows)):
            rows = [r for r in depth_rows if r[0] == name and r[2] is not None]
            xs = [r[1] for r in rows]; ys = [r[2] for r in rows]
            plt.plot(range(len(xs)), ys, marker="o", label=name)
            plt.xticks(range(len(xs)), xs, rotation=90, fontsize=6)
        plt.ylabel("silhouette (group=base_language)"); plt.xlabel("layer")
        plt.legend(); plt.title("Structure vs depth"); plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(plot_dir, f"emb_fig4_depth_structure.{ext}"))
        plt.close()

    logger.info(f"Embedding analysis complete. Sources analyzed: {[r['source'] for r in summary]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile-check**

Run: `cd scripts && ../.venv/bin/python -m py_compile 07_analyze_embeddings.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/07_analyze_embeddings.py
git commit -m "feat: 07_analyze_embeddings CLI (structure + depth, figures)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: pipeline.sh stage

**Files:**
- Modify: `pipeline.sh`

**Interfaces:**
- Produces: `./pipeline.sh embed-analysis` (alias `07`) → `07_analyze_embeddings.py`.

- [ ] **Step 1: Add the stage mapping**

In `pipeline.sh`, inside `script_for()`, add a case arm before the `*)` default:

```bash
    embed-analysis|07) echo "07_analyze_embeddings.py" ;;
```

Update the header usage comment line `# Stage names: ...` to include `embed-analysis`, and the `Unknown stage` error message list likewise.

- [ ] **Step 2: Verify**

Run: `bash -n pipeline.sh && echo "syntax OK" && bash -c 'f(){ case "$1" in embed-analysis|07) echo 07_analyze_embeddings.py;; esac; }; f embed-analysis'`
Expected: `syntax OK` then `07_analyze_embeddings.py`.

- [ ] **Step 3: Commit**

```bash
git add pipeline.sh
git commit -m "feat: pipeline.sh embed-analysis stage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Smoke verification on the 1.7B sample (MPS)

**Files:** none (verification only). Uses `/tmp/smoke/data` and `/tmp/smoke/config_17b.yaml` from earlier sessions; recreate them if absent (see plan notes).

- [ ] **Step 1: Point the smoke config at an embeddings/analysis/plots dir**

Edit `/tmp/smoke/config_17b.yaml` to add under `model:` `include_embedding_layer: true`, under a new `analysis:` block `depth_layers: null`, and under `paths:` `va_embeddings_dir: "/tmp/smoke/embeddings_17b"`, `va_analysis_dir: "/tmp/smoke/analysis_17b"`, `va_plot_dir: "/tmp/smoke/plots_17b"`.

- [ ] **Step 2: Run extraction (writes embedding store)**

Run: `cd /Users/yual01-admin/Desktop/dfki/2026/Culture_Puzzles && ./pipeline.sh --config /tmp/smoke/config_17b.yaml vectors > /tmp/emb_extract.log 2>&1; grep "EXIT" /tmp/emb_extract.log`
Expected: the wrapper prints an exit marker; confirm `find /tmp/smoke/embeddings_17b -name 'layer_embed.npz' | wc -l` ≥ 4 (flores, opus100, puzzles_original, puzzles_translation; topics/culture also present).

- [ ] **Step 3: Run analysis**

Run: `cd /Users/yual01-admin/Desktop/dfki/2026/Culture_Puzzles && ./pipeline.sh --config /tmp/smoke/config_17b.yaml embed-analysis > /tmp/emb_analyze.log 2>&1; grep -iE "complete|Error|Traceback" /tmp/emb_analyze.log | tail`
Expected: "Embedding analysis complete"; CSVs in `/tmp/smoke/analysis_17b/` (`embedding_structure_summary.csv`, `depth_structure.csv`) and figures in `/tmp/smoke/plots_17b/` (`emb_fig1_*`, `emb_fig2_*`, `emb_fig4_depth_structure.png`).

- [ ] **Step 4: Sanity-check outputs**

Run: `cd /Users/yual01-admin/Desktop/dfki/2026/Culture_Puzzles && .venv/bin/python -c "import csv;rows=list(csv.DictReader(open('/tmp/smoke/analysis_17b/depth_structure.csv')));print('layers per source ok:', {r['source'] for r in rows}); print('has embed row:', any(r['layer']=='embed' for r in rows))"`
Expected: prints the source set and `has embed row: True`.

- [ ] **Step 5: Commit (docs note only, if any)**

No code change; if anything was fixed during smoke, commit it with an appropriate message. Otherwise nothing to commit.

---

## Self-Review

**Spec coverage:**
- Embedding capture (`embed_tokens`) → Task 1. ✓
- Per-region/topic mean store for all sources incl. topics/culture → Tasks 2, 4. ✓
- Structure (similarity/PCA/clustering + silhouette/ARI/NMI) → Tasks 3, 5. ✓
- Depth (structure vs layer, embed-first) → Tasks 3, 5. ✓
- Config gaps `va_analysis_dir`/`va_plot_dir` + `va_embeddings_dir` + `include_embedding_layer` + `depth_layers` → Task 4. ✓
- `pipeline.sh` stage → Task 6. ✓
- Tests without model + smoke on 1.7B sample → Tasks 1–4 (unit), Task 7 (smoke). ✓
- Error handling (missing embed module, <2/<3 keys, degenerate silhouette, absent store) → Task 1 (raise), Task 3 (`None` guards), Task 5 (`analyze_source` skips). ✓

**Placeholder scan:** none — all steps carry concrete code/commands.

**Type consistency:** `structure_score`/`depth_structure`/`cluster_embeddings` signatures and return dict keys match between Task 3 definitions and Task 5 usage; `save_embedding_store`/`build_group_map` signatures match between Task 2 and Task 4; `process_flat(..., emb_base=...)` matches between Task 4 test and impl.
