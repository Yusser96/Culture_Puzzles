# `src/` Representational-Analysis Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully isolated `src/` package implementing the decoder-only representational-analysis research plan end-to-end (collect → metadata → extract → normalize → probes → directions → cross → flores_decomp → rep_similarity → steering → data_stats → report).

**Architecture:** Store-centric pipeline. A unified `metadata.parquet` (keyed by `sample_id`) and an `ActivationStore` (keyed by `model/readout/layer`, row-aligned to `sample_id`) are the shared contract; each research step is an independent sub-module under `src/modules/<step>/` with its own `run.py`, reading the store + metadata and writing CSVs/figures. One CLI (`python -m src.run <step>`). Nothing in `src/` imports from `scripts/`.

**Tech Stack:** Python 3.9, numpy, torch, transformers, nnsight (decoder), scipy, scikit-learn, pandas, pyarrow (parquet), statsmodels (ANOVA), matplotlib (Agg), datasets, wikipedia-api==0.6.0, openpyxl. Tests: **unittest**.

## Global Constraints

- Package is importable as `src` from the repo root: `/Users/yual01-admin/Desktop/dfki/2026/Culture_Puzzles`. Every dir under `src/` and `src/modules/` and `src/shared_utils/` has an `__init__.py`. Imports are src-relative: `from src.shared_utils.io import load_config`.
- **No `src/` file imports from `scripts/`.** Code reused from `scripts/shared_utils/*` is PORTED (copied + adapted), not imported.
- Tests are **unittest**, run from repo root: `.venv/bin/python -m unittest src.tests.<module> -v` (filter noise with `2>&1 | grep -viE "NotOpenSSL|warnings.warn"`).
- Model-dependent code (`models.py`, `extraction.py`, steering, `extract`/`steering` run.py) is **not** run by unit tests — tested with fake activations + a smoke run on Qwen3-1.7B/MPS. Qwen3-8B will not load on this 8 GB Mac.
- `ActivationStore` layout: `<store_dir>/<model>/<readout>/layer_<LABEL>.npy` (float32 `(N, D)`), plus `<store_dir>/<model>/sample_ids.json` (the shared N-row order). `LABEL` = `embed` or `%03d`.
- `MetadataTable`: parquet with columns exactly: `sample_id, text, source, topic, topic_canonical, topic_raw, language, region, language_region, script, domain, prompt_template, token_count, translation_group_id, split`.
- `sample_id` format: `"{source}/{group}/{i}"` (unique, stable). Store/metadata `sample_id` mismatch is a hard error.
- DiffMean directions are unit-normalized; representations for probes are standardized (not necessarily unit-norm).
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- matplotlib uses the `Agg` backend (set before importing pyplot).

## File structure (created across tasks)

```
src/__init__.py  run.py  requirements.txt  README.md
src/configs/config.yaml
src/shared_utils/__init__.py io.py registry.py text.py store.py vectors.py
    normalize.py probes.py similarity.py models.py extraction.py steering_utils.py plotting.py
src/modules/__init__.py
src/modules/collect/__init__.py puzzles.py parallel.py topics.py sib200.py run.py
src/modules/{metadata,extract,normalize,probes,directions,cross,flores_decomp,
             rep_similarity,steering,data_stats,report}/__init__.py run.py (+ logic files)
src/tests/__init__.py test_*.py helpers.py
```

---

### Task 1: Package scaffold + `io` + config + tests helper

**Files:**
- Create: `src/__init__.py`, `src/shared_utils/__init__.py`, `src/modules/__init__.py`, `src/tests/__init__.py`
- Create: `src/requirements.txt`, `src/configs/config.yaml`, `src/shared_utils/io.py`, `src/tests/helpers.py`
- Test: `src/tests/test_io.py`

**Interfaces:**
- Produces: `io.load_config(path)->dict`, `io.setup_logging(name)->Logger`, `io.save_json/load_json`, `io.save_jsonl/load_jsonl`, `io.save_csv(path, header, rows)`, `io.ensure_dir(path)`.

- [ ] **Step 1: Write the failing test** — create `src/tests/test_io.py`:

```python
import os, tempfile, unittest
from src.shared_utils.io import (
    save_json, load_json, save_jsonl, load_jsonl, save_csv, ensure_dir, load_config,
)

class TestIO(unittest.TestCase):
    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json"); save_json({"a": 1, "u": "ünì"}, p)
            self.assertEqual(load_json(p), {"a": 1, "u": "ünì"})

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl"); rows = [{"i": 1}, {"i": 2}]
            save_jsonl(rows, p); self.assertEqual(load_jsonl(p), rows)

    def test_csv_and_ensure_dir(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "a/b"); ensure_dir(sub)
            self.assertTrue(os.path.isdir(sub))
            p = os.path.join(sub, "x.csv"); save_csv(p, ["k", "v"], [["a", 1]])
            self.assertEqual(open(p).read().splitlines()[0], "k,v")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, expect fail** — `cd /Users/yual01-admin/Desktop/dfki/2026/Culture_Puzzles && .venv/bin/python -m unittest src.tests.test_io -v 2>&1 | grep -viE "NotOpenSSL|warnings.warn"` → `ModuleNotFoundError: src.shared_utils.io`.

- [ ] **Step 3: Implement.** Create the four `__init__.py` as empty files. Create `src/shared_utils/io.py`:

```python
"""I/O + logging helpers (decoder-only research pipeline)."""
import csv, json, logging, os
from typing import List
import yaml

def setup_logging(name, level="INFO"):
    lg = logging.getLogger(name); lg.setLevel(getattr(logging, level))
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
    return lg

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rows, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_csv(path, header, rows):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
```

Create `src/tests/helpers.py`:

```python
import os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, "src", "configs", "config.yaml")
```

Create `src/requirements.txt` (one per line): `numpy torch transformers nnsight scipy scikit-learn pandas pyarrow statsmodels matplotlib datasets Wikipedia-API==0.6.0 openpyxl tqdm PyYAML` (one token per line). Create `src/configs/config.yaml` minimal (expanded in later tasks):

```yaml
model: {layers: "all", batch_size: 4, max_seq_len: 128, device: "mps", dtype: "float16"}
models: ["Qwen/Qwen3-1.7B"]
readouts: ["mean_content", "last_content", "embed"]
representations: ["raw", "language_centered", "language_region_centered", "topic_centered", "source_centered"]
canonical_topics: ["politics", "kids_world", "national_heritage", "everyday_life", "sports", "geography", "arts", "history"]
analysis: {seed: 42, n_pca_components: 3}
probes:
  kinds: ["logistic", "svm", "diffmean"]
  splits: ["random", "heldout_language", "heldout_region", "heldout_language_region", "heldout_source", "heldout_prompt"]
steering: {alpha: [-3, -2, -1, -0.5, 0.5, 1, 2, 3], max_new_tokens: 50}
paths:
  raw_dir:      "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/raw"
  metadata:     "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/metadata.parquet"
  store_dir:    "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/store"
  analysis_dir: "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/analysis"
  plot_dir:     "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/plots"
```

(The `lang_regions` registry and `cultural_topics`/`topic_label_map` blocks are copied verbatim from `scripts/configs/riddles_config.yaml` into this file in Task 9 when the collectors need them — copy the whole `lang_regions:`, `cultural_topics:`, and `topic_label_map:` sections.)

- [ ] **Step 4: Run, expect pass** — same command → OK (3 tests).
- [ ] **Step 5: Commit** — `git add src && git commit -m "feat(src): package scaffold + io helpers\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`.

---

### Task 2: `registry` + `text` (factors, script, masks)

**Files:** Create `src/shared_utils/registry.py`, `src/shared_utils/text.py`; Test `src/tests/test_text.py`.

**Interfaces:**
- Produces: `registry.load_registry(cfg)->list`; `registry.region_factors(key, cfg)->dict` with keys `base_language, region, language_region, wiki, flores, opus`.
- Produces: `text.detect_script(text)->str`; `text.sentence_split(text)->list[str]`; `text.content_token_offsets(tokenizer, text, answer=None)->list[bool]` (token mask: True for content tokens, False for specials and any token whose char span lies inside the `answer` substring).

- [ ] **Step 1: failing test** `src/tests/test_text.py`:

```python
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
        self.assertEqual(len(sentence_split("One sentence here. Two sentence here!")), 2)
    def test_region_factors(self):
        cfg = load_config(CONFIG)
        f = region_factors("Arabic_Egypt", cfg)
        self.assertEqual(f["base_language"], "Arabic")
        self.assertEqual(f["region"], "Egypt")
        self.assertEqual(f["language_region"], "Arabic_Egypt")

if __name__ == "__main__":
    unittest.main()
```

(Requires the registry block present in config — ensure Task 9 copy is done OR copy `lang_regions:` now; if running Task 2 before Task 9, copy the `lang_regions:` block from `scripts/configs/riddles_config.yaml` into `src/configs/config.yaml` as part of this task.)

- [ ] **Step 2: run, expect fail** (`ModuleNotFoundError`).
- [ ] **Step 3: implement.** `src/shared_utils/registry.py`:

```python
def load_registry(cfg):
    return cfg.get("lang_regions", [])

def region_factors(key, cfg):
    rec = next((r for r in load_registry(cfg) if r["key"] == key), {})
    region = key.split("_", 1)[1] if "_" in key else ""
    return {"base_language": rec.get("base_language", "UNKNOWN"),
            "region": region, "language_region": key,
            "wiki": rec.get("wiki"), "flores": rec.get("flores"), "opus": rec.get("opus")}
```

`src/shared_utils/text.py` (port `detect_script` from `scripts/data_stats.py` lines defining it; add the others):

```python
import re, unicodedata
from collections import Counter

def detect_script(text):
    c = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        tok = name.split(" ")[0]
        if tok in ("CJK", "IDEOGRAPHIC"):
            tok = "CJK"
        c[tok] += 1
    return c.most_common(1)[0][0] if c else "UNKNOWN"

def sentence_split(text):
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) > 20]

def content_token_offsets(tokenizer, text, answer=None):
    """Return a per-token bool mask (True=content). Marks special tokens and tokens
    overlapping the answer substring as False, using offset mapping."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]; specials = set(tokenizer.all_special_ids)
    ids = enc["input_ids"]
    ans_span = None
    if answer:
        i = text.find(answer)
        if i >= 0:
            ans_span = (i, i + len(answer))
    mask = []
    for tid, (a, b) in zip(ids, offsets):
        if tid in specials or (a == b == 0):
            mask.append(False); continue
        if ans_span and not (b <= ans_span[0] or a >= ans_span[1]):
            mask.append(False); continue
        mask.append(True)
    return mask
```

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** `feat(src): registry factor mapping + text/script/masks`.

---

### Task 3: `store` (ActivationStore + MetadataTable)

**Files:** Create `src/shared_utils/store.py`; Test `src/tests/test_store.py`.

**Interfaces:**
- Produces: `ActivationStore(store_dir)` with `save_index(model, sample_ids)`, `load_index(model)->list`, `save_layer(model, readout, layer, X)`, `load_layer(model, readout, layer)->np.ndarray`, `models()->list`, `readouts(model)->list`, `layers(model, readout)->list` (layer labels: `"embed"` or `int`).
- Produces: `MetadataTable.save(df, path)`, `MetadataTable.load(path)->DataFrame`, `MetadataTable.COLUMNS` (the exact 15-column list).

- [ ] **Step 1: failing test** `src/tests/test_store.py`:

```python
import os, tempfile, unittest
import numpy as np, pandas as pd
from src.shared_utils.store import ActivationStore, MetadataTable

class TestStore(unittest.TestCase):
    def test_activation_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            s = ActivationStore(d)
            s.save_index("m1", ["a", "b", "c"])
            s.save_layer("m1", "mean_content", "embed", np.ones((3, 4)))
            s.save_layer("m1", "mean_content", 0, np.zeros((3, 4)))
            self.assertEqual(s.load_index("m1"), ["a", "b", "c"])
            np.testing.assert_allclose(s.load_layer("m1", "mean_content", "embed"), np.ones((3, 4)))
            self.assertEqual(s.models(), ["m1"])
            self.assertEqual(set(s.layers("m1", "mean_content")), {"embed", 0})
    def test_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            df = pd.DataFrame([{c: "x" for c in MetadataTable.COLUMNS}])
            p = os.path.join(d, "m.parquet"); MetadataTable.save(df, p)
            back = MetadataTable.load(p)
            self.assertEqual(list(back.columns), MetadataTable.COLUMNS)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: run fail.**
- [ ] **Step 3: implement** `src/shared_utils/store.py`:

```python
import json, os
import numpy as np

def _label(layer):
    return "embed" if layer == "embed" else f"{int(layer):03d}"

class ActivationStore:
    def __init__(self, store_dir):
        self.dir = store_dir
    def _mdir(self, model):
        return os.path.join(self.dir, model)
    def save_index(self, model, sample_ids):
        os.makedirs(self._mdir(model), exist_ok=True)
        json.dump(list(sample_ids), open(os.path.join(self._mdir(model), "sample_ids.json"), "w"))
    def load_index(self, model):
        return json.load(open(os.path.join(self._mdir(model), "sample_ids.json")))
    def save_layer(self, model, readout, layer, X):
        d = os.path.join(self._mdir(model), readout); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"layer_{_label(layer)}.npy"), np.asarray(X, dtype=np.float32))
    def load_layer(self, model, readout, layer):
        return np.load(os.path.join(self._mdir(model), readout, f"layer_{_label(layer)}.npy"))
    def models(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(m for m in os.listdir(self.dir) if os.path.isdir(self._mdir(m)))
    def readouts(self, model):
        md = self._mdir(model)
        return sorted(r for r in os.listdir(md) if os.path.isdir(os.path.join(md, r)))
    def layers(self, model, readout):
        d = os.path.join(self._mdir(model), readout); out = []
        for fn in sorted(os.listdir(d)):
            if fn.startswith("layer_") and fn.endswith(".npy"):
                tag = fn[len("layer_"):-len(".npy")]
                out.append("embed" if tag == "embed" else int(tag))
        return out

class MetadataTable:
    COLUMNS = ["sample_id", "text", "source", "topic", "topic_canonical", "topic_raw",
               "language", "region", "language_region", "script", "domain",
               "prompt_template", "token_count", "translation_group_id", "split"]
    @staticmethod
    def save(df, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df[MetadataTable.COLUMNS].to_parquet(path, index=False)
    @staticmethod
    def load(path):
        import pandas as pd
        return pd.read_parquet(path)[MetadataTable.COLUMNS]
```

- [ ] **Step 4: run pass.**
- [ ] **Step 5: commit** `feat(src): activation store + metadata table`.

---

### Task 4: `vectors` (DiffMean + balanced background + cosine + subspace angle)

**Files:** Create `src/shared_utils/vectors.py`; Test `src/tests/test_vectors.py`.

**Interfaces:**
- Produces: `diffmean(target, background, normalize=True)->vec`; `cosine(a,b)->float`; `cosine_matrix(dict)->(mat, labels)`; `subspace_angle(a,b)->float`; `balanced_background(X, labels, target_label, groups, rng)->ndarray` (sample equal counts per `groups` value from the non-target rows).

- [ ] **Step 1: failing test** `src/tests/test_vectors.py`:

```python
import unittest
import numpy as np
from src.shared_utils.vectors import diffmean, cosine, cosine_matrix, balanced_background

class TestVectors(unittest.TestCase):
    def test_diffmean_unit_norm(self):
        v = diffmean(np.ones((4, 3)), np.zeros((4, 3)), normalize=True)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
    def test_cosine(self):
        self.assertAlmostEqual(cosine(np.array([1., 0]), np.array([1., 0])), 1.0, places=6)
    def test_cosine_matrix_diag(self):
        m, labels = cosine_matrix({"a": np.array([1., 0]), "b": np.array([0., 1.])})
        self.assertAlmostEqual(m[0, 0], 1.0, places=6); self.assertAlmostEqual(m[0, 1], 0.0, places=6)
    def test_balanced_background_equal_per_group(self):
        X = np.arange(60).reshape(20, 3).astype(float)
        labels = ["t"] * 5 + ["o"] * 15
        groups = (["g1"] * 10) + (["g2"] * 10)
        bg = balanced_background(X, labels, "t", groups, np.random.default_rng(0))
        self.assertEqual(bg.shape[1], 3); self.assertGreater(bg.shape[0], 0)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: run fail.**
- [ ] **Step 3: implement** `src/shared_utils/vectors.py` (port `cosine`/`subspace_angle`/`cosine_similarity_matrix` from `scripts/shared_utils/vectors.py`; add balanced background):

```python
from collections import defaultdict
import numpy as np

def diffmean(target, background, normalize=True):
    if target.size == 0 or background.size == 0:
        d = target.shape[-1] if target.ndim >= 2 else background.shape[-1]
        return np.zeros(d)
    v = target.mean(0) - background.mean(0)
    if normalize:
        n = np.linalg.norm(v)
        if n > 1e-10:
            v = v / n
    return v

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def cosine_matrix(vectors):
    labels = sorted(vectors); n = len(labels); m = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            m[i, j] = cosine(vectors[li], vectors[lj])
    return m, labels

def subspace_angle(a, b):
    c = np.clip(abs(cosine(a, b)), 0, 1)
    return float(np.degrees(np.arccos(c)))

def balanced_background(X, labels, target_label, groups, rng):
    """Rows where label != target_label, sampled to an equal count per distinct group."""
    idx_by_group = defaultdict(list)
    for i, (lab, g) in enumerate(zip(labels, groups)):
        if lab != target_label:
            idx_by_group[g].append(i)
    if not idx_by_group:
        return np.zeros((0, X.shape[1]))
    k = min(len(v) for v in idx_by_group.values())
    picked = []
    for g, idxs in idx_by_group.items():
        picked += list(rng.choice(idxs, size=k, replace=False))
    return X[np.array(sorted(picked))]
```

- [ ] **Step 4: run pass.**
- [ ] **Step 5: commit** `feat(src): vectors with balanced-background diffmean`.

---

### Task 5: `normalize` (standardize + centering)

**Files:** Create `src/shared_utils/normalize.py`; Test `src/tests/test_normalize.py`.

**Interfaces:**
- Produces: `fit_stats(X_train)->(mu, std)`; `standardize(X, mu, std)->X`; `center(X, group_ids)->X` (subtract per-group mean over rows sharing a `group_id`).

- [ ] **Step 1: failing test** `src/tests/test_normalize.py`:

```python
import unittest
import numpy as np
from src.shared_utils.normalize import fit_stats, standardize, center

class TestNormalize(unittest.TestCase):
    def test_standardize_zero_mean_unit_std(self):
        X = np.random.default_rng(0).normal(5, 3, (200, 4))
        mu, std = fit_stats(X); Z = standardize(X, mu, std)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-6)
        np.testing.assert_allclose(Z.std(0), 1, atol=1e-2)
    def test_center_removes_group_mean(self):
        X = np.array([[10., 0], [12., 0], [0., 5], [2., 5]])
        g = ["a", "a", "b", "b"]; C = center(X, g)
        np.testing.assert_allclose(C, [[-1, 0], [1, 0], [-1, 0], [1, 0]])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `src/shared_utils/normalize.py`:

```python
from collections import defaultdict
import numpy as np

def fit_stats(X_train):
    mu = X_train.mean(0); std = X_train.std(0) + 1e-6
    return mu, std

def standardize(X, mu, std):
    return (X - mu) / std

def center(X, group_ids):
    X = np.asarray(X, dtype=float); out = X.copy()
    idx = defaultdict(list)
    for i, g in enumerate(group_ids):
        idx[g].append(i)
    for g, rows in idx.items():
        rows = np.array(rows); out[rows] = X[rows] - X[rows].mean(0)
    return out
```

- [ ] **Step 4: pass.**
- [ ] **Step 5: commit** `feat(src): per-layer standardization + centering`.

---

### Task 6: `probes` (logistic/SVM/diffmean + splits + normals)

**Files:** Create `src/shared_utils/probes.py`; Test `src/tests/test_probes.py`.

**Interfaces:**
- Produces: `train_probe(X, y, kind, seed)->fitted` (kind ∈ logistic/svm/diffmean); `probe_score(fitted, X, y)->{"macro_f1":..,"auroc":..}`; `probe_normal(fitted, kind)->ndarray` (the class-1 normal for binary, or per-class normals dict for multiclass diffmean); `make_splits(meta_df, scheme, seed)->list[(train_idx, test_idx)]` where scheme ∈ random/heldout_<factor>.

- [ ] **Step 1: failing test** `src/tests/test_probes.py`:

```python
import unittest
import numpy as np, pandas as pd
from src.shared_utils.probes import train_probe, probe_score, probe_normal, make_splits

def _sep(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    Xa = rng.normal(+2, 0.3, (n, d)); Xb = rng.normal(-2, 0.3, (n, d))
    return np.vstack([Xa, Xb]), np.array(["a"]*n + ["b"]*n)

class TestProbes(unittest.TestCase):
    def test_logistic_separable(self):
        X, y = _sep(); p = train_probe(X, y, "logistic", 0)
        self.assertGreater(probe_score(p, X, y)["macro_f1"], 0.95)
    def test_diffmean_normal_unit(self):
        X, y = _sep(); p = train_probe(X, y, "diffmean", 0)
        v = probe_normal(p, "diffmean"); self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, 5)
    def test_heldout_split_disjoint_groups(self):
        df = pd.DataFrame({"language": ["en","en","fr","fr","de","de"]})
        splits = make_splits(df, "heldout_language", 0)
        for tr, te in splits:
            self.assertTrue(set(df.language.iloc[tr]).isdisjoint(set(df.language.iloc[te])))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `src/shared_utils/probes.py`:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

_FACTOR = {"heldout_language": "language", "heldout_region": "region",
           "heldout_language_region": "language_region", "heldout_source": "source",
           "heldout_prompt": "prompt_template"}

def train_probe(X, y, kind, seed):
    y = np.asarray(y)
    if kind == "logistic":
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(X, y); return ("logistic", m)
    if kind == "svm":
        m = LinearSVC(random_state=seed).fit(X, y); return ("svm", m)
    if kind == "diffmean":
        classes = sorted(set(y)); normals = {}
        for c in classes:
            v = X[y == c].mean(0) - X[y != c].mean(0)
            n = np.linalg.norm(v); normals[c] = v / n if n > 1e-10 else v
        return ("diffmean", {"classes": classes, "normals": normals})
    raise ValueError(kind)

def _predict(fitted, X):
    kind, m = fitted
    if kind == "diffmean":
        classes = m["classes"]; S = np.stack([X @ m["normals"][c] for c in classes], 1)
        return np.array([classes[i] for i in S.argmax(1)])
    return m.predict(X)

def probe_score(fitted, X, y):
    y = np.asarray(y); pred = _predict(fitted, X)
    out = {"macro_f1": float(f1_score(y, pred, average="macro"))}
    kind, m = fitted
    try:
        classes = sorted(set(y))
        if kind == "logistic" and len(classes) == 2:
            out["auroc"] = float(roc_auc_score(y == classes[1], m.predict_proba(X)[:, 1]))
        else:
            out["auroc"] = None
    except Exception:
        out["auroc"] = None
    return out

def probe_normal(fitted, kind):
    k, m = fitted
    if k == "diffmean":
        cs = m["classes"]
        return m["normals"][cs[0]] if len(cs) == 2 else m["normals"]
    coef = m.coef_
    v = coef[0] if coef.shape[0] == 1 else coef
    if v.ndim == 1:
        n = np.linalg.norm(v); v = v / n if n > 1e-10 else v
    return v

def make_splits(meta_df, scheme, seed):
    rng = np.random.default_rng(seed); n = len(meta_df); idx = np.arange(n)
    if scheme == "random":
        rng.shuffle(idx); cut = int(0.8 * n)
        return [(idx[:cut], idx[cut:])]
    col = _FACTOR[scheme]; groups = sorted(meta_df[col].astype(str).unique())
    splits = []
    for g in groups:
        te = idx[meta_df[col].astype(str).values == g]; tr = idx[meta_df[col].astype(str).values != g]
        if len(te) and len(tr):
            splits.append((tr, te))
    return splits
```

- [ ] **Step 4: pass.**
- [ ] **Step 5: commit** `feat(src): probes (logistic/svm/diffmean) + held-out splits`.

---

### Task 7: `similarity` (CKA / SVCCA / Procrustes / RDM / subspace angles)

**Files:** Create `src/shared_utils/similarity.py`; Test `src/tests/test_similarity.py`.

**Interfaces:**
- Produces: `linear_cka(X, Y)->float`; `svcca(X, Y, k=10)->float`; `procrustes_disparity(X, Y)->float`; `rdm(X, metric="cosine")->ndarray`; `subspace_angles(A, B)->ndarray`.

- [ ] **Step 1: failing test** `src/tests/test_similarity.py`:

```python
import unittest
import numpy as np
from src.shared_utils.similarity import linear_cka, procrustes_disparity, rdm

class TestSim(unittest.TestCase):
    def test_cka_identity(self):
        X = np.random.default_rng(0).normal(size=(50, 8))
        self.assertAlmostEqual(linear_cka(X, X), 1.0, places=5)
    def test_cka_rotation_invariant(self):
        rng = np.random.default_rng(1); X = rng.normal(size=(50, 6))
        Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        self.assertAlmostEqual(linear_cka(X, X @ Q), 1.0, places=4)
    def test_procrustes_rotation_zero(self):
        rng = np.random.default_rng(2); X = rng.normal(size=(40, 5))
        Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        self.assertLess(procrustes_disparity(X, X @ Q), 1e-6)
    def test_rdm_shape(self):
        self.assertEqual(rdm(np.random.default_rng(0).normal(size=(7, 4))).shape, (7, 7))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `src/shared_utils/similarity.py`:

```python
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import subspace_angles as _sa

def _center(X):
    return X - X.mean(0)

def linear_cka(X, Y):
    Xc, Yc = _center(X), _center(Y)
    hsic = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(hsic / den) if den > 1e-12 else 0.0

def svcca(X, Y, k=10):
    from numpy.linalg import svd
    Xc, Yc = _center(X), _center(Y)
    Ux = svd(Xc, full_matrices=False)[0][:, :k]; Uy = svd(Yc, full_matrices=False)[0][:, :k]
    s = svd(Ux.T @ Uy, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))

def procrustes_disparity(X, Y):
    _, _, disparity = procrustes(X, Y)
    return float(disparity)

def rdm(X, metric="cosine"):
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return 1 - Xn @ Xn.T
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(X, metric="euclidean"))

def subspace_angles(A, B):
    return np.degrees(_sa(A, B))
```

- [ ] **Step 4: pass.**
- [ ] **Step 5: commit** `feat(src): representational-similarity measures`.

---

### Task 8: `models` + `extraction` (decoder multi-readout) + `steering_utils` + `plotting`

**Files:** Create `src/shared_utils/models.py`, `src/shared_utils/extraction.py`, `src/shared_utils/steering_utils.py`, `src/shared_utils/plotting.py`; Test `src/tests/test_extraction.py`.

**Interfaces:**
- Produces: `models.load_decoder(cfg, name)->handle` (`.model, .tokenizer, .num_layers, .hidden_size, .name`).
- Produces: `extraction.extract(handle, texts, layers, readouts, max_seq_len, batch_size, answers=None)->{readout:{layer:ndarray}}`; readouts ⊂ {mean_content,last_content,embed}.
- Produces: `steering_utils.add_and_generate(handle, prompt, layer, vec, alpha, max_new_tokens)->str`.
- Produces: `plotting.heatmap(path,mat,rows,cols,title)`, `plotting.lines(path,x,series,title,xlabel,ylabel)`, `plotting.scatter(path,xy,groups,title)`.

- [ ] **Step 1: write the test** (uses a fake handle + monkeypatched extractor — no model). `src/tests/test_extraction.py`:

```python
import unittest
import numpy as np
import src.shared_utils.extraction as ext

class TestExtractionShapes(unittest.TestCase):
    def test_pool_mean_and_last(self):
        # hidden (batch=1, seq=3, dim=2); mask keeps tokens 0 and 2
        hidden = np.array([[[1., 1.], [9., 9.], [3., 3.]]])
        mask = np.array([[True, False, True]])
        mean = ext._pool(hidden, mask, "mean_content")
        last = ext._pool(hidden, mask, "last_content")
        np.testing.assert_allclose(mean[0], [2., 2.])
        np.testing.assert_allclose(last[0], [3., 3.])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: run fail.**
- [ ] **Step 3: implement.** `models.py` — port `scripts/shared_utils/model.py` (the `_patch_check_model_inputs`, `get_model_and_tokenizer`, `get_num_layers`, `get_layers_to_probe`) into a `load_decoder(cfg, name)` that overrides the model name with `name` and returns a small handle object with `.model,.tokenizer,.num_layers,.hidden_size,.name`. `extraction.py` — port the masked-pool loop from `scripts/shared_utils/activation_extraction.py` and the embed-before-blocks fix, generalized to readouts. Provide a pure `_pool(hidden_np, mask_np, readout)` helper that the test targets:

```python
import numpy as np
def _pool(hidden, mask, readout):
    """hidden (B,S,D) float, mask (B,S) bool -> (B,D)."""
    out = []
    for b in range(hidden.shape[0]):
        m = mask[b]
        if not m.any():
            m = np.ones_like(m)
        if readout == "last_content":
            out.append(hidden[b][np.where(m)[0][-1]])
        else:  # mean_content / embed both mean-pool over content tokens
            out.append(hidden[b][m].mean(0))
    return np.stack(out)
```

`extract(...)` builds per-batch token masks via `text.content_token_offsets` (passing the per-sample `answers[i]`), runs the NNsight trace capturing `embed` (before blocks) + each layer's residual output, converts to numpy, and applies `_pool` per readout. `steering_utils.py` — port `steer_generate`/`add_direction` logic from `scripts/shared_utils/steering.py`. `plotting.py` — small Agg helpers.

- [ ] **Step 4: run pass** (only `_pool` is unit-tested).
- [ ] **Step 5: smoke (manual, optional now)** — deferred to Task 11/18 smoke. Commit `feat(src): decoder model loader + multi-readout extraction + steering/plot utils`.

---

### Task 9: `modules/collect` (puzzles, parallel+IDs, topics, SIB-200)

**Files:** Create `src/modules/collect/{__init__,puzzles,parallel,topics,sib200,run}.py`; copy the `lang_regions:`, `cultural_topics:`, `topic_label_map:` blocks from `scripts/configs/riddles_config.yaml` into `src/configs/config.yaml`. Test `src/tests/test_collect.py`.

**Interfaces:**
- Produces (raw corpora under `paths.raw_dir`): `puzzles/<region>/{original,translation}.txt` + `riddles.jsonl`; `parallel/{flores,opus100}/<region>.jsonl` (each line `{text, translation_group_id}`); `cultural/<topic>/<region>.txt`; `sib200/<topic>/<lang>.txt`. Each writes a `manifest.json`.
- Produces: `sib200.collect(cfg)` using HF `datasets.load_dataset("Davlan/sib200", <lang>)` (category=topic label). `parallel.collect_flores` keeps the **dev-split row index** as `translation_group_id`.

- [ ] **Step 1: failing test** `src/tests/test_collect.py` — port the riddle-reader tests: copy `scripts/shared_utils/riddles.py` to `src/shared_utils/riddles.py` and test `read_riddles_xlsx` + `parse_lang_region_key` on the real `v0 - due May 29` files (skip if folder absent):

```python
import os, unittest
from src.shared_utils.riddles import parse_lang_region_key

class TestCollect(unittest.TestCase):
    def test_key_parse(self):
        self.assertEqual(parse_lang_region_key("Cultural Riddles Benchmark [Arabic_Egypt].xlsx"),
                         "Arabic_Egypt")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement.** Port `scripts/shared_utils/riddles.py`, `scripts/shared_utils/wiki.py` into `src/shared_utils/`. Port `scripts/01_collect_puzzles.py`→`puzzles.py`, `scripts/02_collect_parallel.py`→`parallel.py` (ADD: store FLORES `dev`-split row index per sentence as `translation_group_id`, write `.jsonl` lines `{text, translation_group_id}`), `scripts/03_collect_topics.py`→`topics.py` (use src-relative imports, write under `paths.raw_dir`). NEW `sib200.py`: for each registry `wiki`/language with SIB-200 coverage, `load_dataset("Davlan/sib200", <code>)`, write `sib200/<category>/<lang>.txt`. `run.py` dispatches `--what {puzzles,parallel,topics,sib200,all}`.
- [ ] **Step 4: run pass** (the unit test only checks the ported reader; collection itself is network-heavy → validated in the integration smoke).
- [ ] **Step 5: commit** `feat(src): collectors (puzzles/parallel+IDs/topics/sib200)`.

---

### Task 10: `modules/metadata`

**Files:** Create `src/modules/metadata/{__init__,build,run}.py`; Test `src/tests/test_metadata.py`.

**Interfaces:**
- Consumes: raw corpora (Task 9), `registry.region_factors`, `text.detect_script`, the config `topic_label_map` + `canonical_topics`.
- Produces: `build.build_metadata(cfg, tokenizer=None)->DataFrame` with the 15 `MetadataTable.COLUMNS`; `run.py` writes `paths.metadata`.

- [ ] **Step 1: failing test** `src/tests/test_metadata.py` — build from a tiny temp raw tree:

```python
import os, tempfile, unittest
from src.modules.metadata.build import build_metadata
from src.shared_utils.store import MetadataTable

class TestMetadata(unittest.TestCase):
    def test_build_from_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            pz = os.path.join(d, "puzzles", "Arabic_Egypt"); os.makedirs(pz)
            open(os.path.join(pz, "original.txt"), "w").write("مرحبا\n")
            import json
            with open(os.path.join(pz, "riddles.jsonl"), "w") as f:
                f.write(json.dumps({"riddle_original": "مرحبا", "topic": "Politics",
                                    "topic_key": "politics"}) + "\n")
            cfg = {"paths": {"raw_dir": d}, "lang_regions":
                   [{"key": "Arabic_Egypt", "base_language": "Arabic"}],
                   "topic_label_map": {"Politics": "politics"},
                   "canonical_topics": ["politics"]}
            df = build_metadata(cfg)
            self.assertEqual(list(df.columns), MetadataTable.COLUMNS)
            row = df.iloc[0]
            self.assertEqual(row["language"], "Arabic"); self.assertEqual(row["region"], "Egypt")
            self.assertEqual(row["topic_canonical"], "politics"); self.assertEqual(row["script"], "ARABIC")
            self.assertEqual(row["source"], "puzzles_original")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `build.py`: walk `raw_dir/puzzles/*` (riddles.jsonl → source `puzzles_original`/`puzzles_translation`, topic via `topic_label_map`), `raw_dir/parallel/{flores,opus100}/*` (source `flores`/`opus100`, topic `None`, keep `translation_group_id`), `raw_dir/cultural/<topic>/*` and `raw_dir/sib200/<topic>/*` (source `cultural`/`sib200`). For each line build a row: `sample_id=f"{source}/{group}/{i}"`, factors from `region_factors`, `script=detect_script(text)`, `token_count=len(tokenizer.encode(text)) if tokenizer else len(text.split())`, `prompt_template="raw"`, `domain=source`, `split` via deterministic 80/20 hash of `sample_id`. `run.py` loads cfg + tokenizer (optional), calls `build_metadata`, `MetadataTable.save`.
- [ ] **Step 4: pass.**
- [ ] **Step 5: commit** `feat(src): unified metadata table builder`.

---

### Task 11: `modules/extract` (orchestrate → store) + smoke

**Files:** Create `src/modules/extract/{__init__,run}.py`; Test `src/tests/test_extract_module.py`.

**Interfaces:**
- Consumes: `MetadataTable`, `models.load_decoder`, `extraction.extract`, `ActivationStore`.
- Produces: for each `cfg["models"]`, write `store/<model>/<readout>/layer_*.npy` + `sample_ids.json` aligned to metadata row order.

- [ ] **Step 1: test with a fake** — monkeypatch `models.load_decoder` and `extraction.extract` so no model is needed; assert the store layout:

```python
import os, tempfile, unittest
import numpy as np, pandas as pd
import src.modules.extract.run as R
from src.shared_utils.store import ActivationStore, MetadataTable

class TestExtractModule(unittest.TestCase):
    def test_writes_store(self):
        with tempfile.TemporaryDirectory() as d:
            mp = os.path.join(d, "m.parquet")
            df = pd.DataFrame([{c: "x" for c in MetadataTable.COLUMNS} for _ in range(3)])
            df["sample_id"] = ["a", "b", "c"]; df["text"] = ["t1", "t2", "t3"]
            MetadataTable.save(df, mp)
            cfg = {"models": ["fake"], "readouts": ["mean_content", "embed"],
                   "model": {"layers": "all", "batch_size": 2, "max_seq_len": 16},
                   "paths": {"metadata": mp, "store_dir": os.path.join(d, "store")}}
            class H: num_layers = 2; hidden_size = 4; name = "fake"
            R.load_decoder = lambda cfg, name: H()
            R.extract = lambda h, texts, layers, readouts, **k: {
                ro: {**{l: np.ones((len(texts), 4)) for l in layers}, "embed": np.zeros((len(texts), 4))}
                for ro in readouts}
            R.run(cfg)
            s = ActivationStore(cfg["paths"]["store_dir"])
            self.assertEqual(s.load_index("fake"), ["a", "b", "c"])
            self.assertIn("embed", s.layers("fake", "mean_content"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `run.py`: load metadata (row order is the index), for each model `h=load_decoder(cfg,name)`, `layers=range(num_layers)`, call `extract(h, texts, layers, readouts, max_seq_len, batch_size, answers=None)`, then `store.save_index(name, sample_ids)` and `store.save_layer` for each readout/layer (incl `embed`). Module-level names `load_decoder`, `extract` so the test can patch them.
- [ ] **Step 4: pass.**
- [ ] **Step 5: SMOKE (manual)** — documented in Task 20; run later on the 1.7B sample. Commit `feat(src): extract module -> activation store`.

---

### Task 12: `modules/normalize` (derived representation views)

**Files:** Create `src/modules/normalize/{__init__,run}.py`; Test `src/tests/test_normalize_module.py`.

**Interfaces:**
- Consumes: store + metadata + `normalize.fit_stats/standardize/center`.
- Produces: `representations(model, readout, layer, repname, store, meta)->ndarray` — returns the requested representation variant on the fly (raw / standardized / language_centered / language_region_centered / topic_centered / source_centered), fitting standardization stats on the **train split rows only**. (Computed lazily — no extra disk store; downstream steps call this accessor.)

- [ ] **Step 1: failing test** — synthetic store/meta; assert `language_centered` removes per-language mean and `raw` is unchanged:

```python
import unittest
import numpy as np, pandas as pd
from src.modules.normalize.run import representation

class FakeStore:
    def __init__(self, X): self.X = X
    def load_layer(self, m, r, l): return self.X

class TestNormModule(unittest.TestCase):
    def test_language_centered(self):
        X = np.array([[10., 0], [12., 0], [0., 4], [2., 4]])
        meta = pd.DataFrame({"language": ["a", "a", "b", "b"], "split": ["train"]*4})
        C = representation(FakeStore(X), meta, "m", "mean_content", 0, "language_centered")
        np.testing.assert_allclose(C, [[-1, 0], [1, 0], [-1, 0], [1, 0]])
    def test_raw_passthrough(self):
        X = np.arange(8).reshape(4, 2).astype(float)
        meta = pd.DataFrame({"language": ["a"]*4, "split": ["train"]*4})
        np.testing.assert_allclose(representation(FakeStore(X), meta, "m", "mean_content", 0, "raw"), X)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `representation(store, meta, model, readout, layer, repname)`: load X; `raw`→X; `standardized`→ standardize with train-row stats; `*_centered`→ `center(X, meta[<factor>])` where factor maps language_centered→language, language_region_centered→language_region, topic_centered→topic_canonical, source_centered→source. `run.py` is a thin CLI that validates representations are computable (no heavy output).
- [ ] **Step 4: pass.**
- [ ] **Step 5: commit** `feat(src): representation variants (standardize + centering)`.

---

### Task 13: `modules/probes` (probe + transfer scores)

**Files:** Create `src/modules/probes/{__init__,run}.py`; Test `src/tests/test_probes_module.py`.

**Interfaces:**
- Consumes: store, metadata, `normalize.representation`, `probes.*`.
- Produces: `layer_probe_scores.csv` (`model,readout,representation,layer,factor,kind,split,macro_f1,auroc`) and `transfer_scores.csv` (held-out splits only).

- [ ] **Step 1: failing test** — tiny synthetic store/meta with a separable factor; assert a CSV row with macro_f1 high on random split:

```python
import os, tempfile, unittest
import numpy as np, pandas as pd, csv
import src.modules.probes.run as R

class FakeStore:
    def __init__(self, X): self.X = X
    def load_layer(self, m, r, l): return self.X
    def models(self): return ["m"]
    def readouts(self, m): return ["mean_content"]
    def layers(self, m, r): return [0]

class TestProbesModule(unittest.TestCase):
    def test_writes_scores(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(3, .2, (20, 4)), rng.normal(-3, .2, (20, 4))])
        meta = pd.DataFrame({"sample_id": [str(i) for i in range(40)],
                             "topic_canonical": ["a"]*20 + ["b"]*20,
                             "language": (["en","fr"]*20)[:40], "region": ["x"]*40,
                             "language_region": ["x"]*40, "source": ["s"]*40,
                             "prompt_template": ["raw"]*40, "token_count": [5]*40,
                             "script": ["LATIN"]*40, "split": ["train"]*32 + ["test"]*8})
            # noqa
        with tempfile.TemporaryDirectory() as d:
            cfg = {"probes": {"kinds": ["logistic"], "splits": ["random"]},
                   "representations": ["raw"], "analysis": {"seed": 0},
                   "paths": {"analysis_dir": d}}
            R.run(cfg, store=FakeStore(X), meta=meta, factors=["topic_canonical"])
            rows = list(csv.DictReader(open(os.path.join(d, "layer_probe_scores.csv"))))
            self.assertTrue(any(float(r["macro_f1"]) > 0.9 for r in rows))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `run(cfg, store=None, meta=None, factors=None)`: default `store=ActivationStore(paths.store_dir)`, `meta=MetadataTable.load(paths.metadata)`, `factors` = the plan's 8 factors. Loop model×readout×representation×layer×factor×kind×split: build X via `representation`, y=meta[factor]; for each split `(tr,te)` train on tr, score on te; write both CSVs (transfer = the `heldout_*` rows). `run.py` CLI calls `run(cfg)`.
- [ ] **Step 4: pass.**
- [ ] **Step 5: commit** `feat(src): probing + held-out transfer scores`.

---

### Task 14: `modules/directions`

**Files:** Create `src/modules/directions/{__init__,run}.py`; Test `src/tests/test_directions_module.py`.

**Interfaces:**
- Produces: `topic_vector_cosines.csv` (`model,readout,layer,topic,language,cos_diffmean_logistic,cos_diffmean_svm,cos_logistic_svm`) using `vectors.diffmean` (balanced background over language/region/source) vs `probes.probe_normal` for logistic & svm.

- [ ] **Step 1: failing test** — synthetic; assert diffmean vs logistic cosine is high for a separable topic. (Pattern mirrors Task 13's fake store/meta; assert a written row's `cos_diffmean_logistic > 0.8`.)
- [ ] **Step 2: fail. Step 3: implement** per the interface (loop topics; target rows = topic; background via `vectors.balanced_background(X, meta.topic_canonical, topic, groups=meta.language)`; logistic/svm normals from a topic-vs-rest probe). **Step 4: pass. Step 5: commit** `feat(src): direction analysis (diffmean vs probe normals)`.

---

### Task 15: `modules/cross`

**Files:** Create `src/modules/cross/{__init__,run}.py`; Test `src/tests/test_cross_module.py`.

**Interfaces:**
- Produces: `cross_language_topic_cosine.csv`, `region_contrasts.csv` (same-language/diff-region & diff-language/same-region cosines, restricted to sources where text differs — flag shared-text groups), `topic_rdm_<layer>.npy`, `heldout_language_transfer.csv` (topic probe trained on languages A,B,C tested on D).

- [ ] **Step 1: failing test** — synthetic; assert cross-language cosine CSV has rows for each topic pair and the shared-text flag is set when two regions have identical vectors. **Step 2-4** standard. **Step 5: commit** `feat(src): cross-language/region/topic analysis`.

---

### Task 16: `modules/flores_decomp`

**Files:** Create `src/modules/flores_decomp/{__init__,decomp,run}.py`; Test `src/tests/test_flores_decomp.py`.

**Interfaces:**
- Produces: `decomp.variance_partition(H, factors_df)->dict` (fraction of variance explained by `translation_group_id` (sentence), `language`, `region`, `script`, residual via sequential least-squares / type-I ANOVA over one-hot factor designs, averaged across hidden dims); `run.py` writes `flores_decomposition.csv` per layer.

- [ ] **Step 1: failing test** `src/tests/test_flores_decomp.py` — synthetic factorial data where a known factor dominates:

```python
import unittest
import numpy as np, pandas as pd
from src.modules.flores_decomp.decomp import variance_partition

class TestDecomp(unittest.TestCase):
    def test_language_dominates(self):
        rng = np.random.default_rng(0); n = 60
        lang = np.array(["en", "fr", "de"])[rng.integers(0, 3, n)]
        H = np.zeros((n, 4))
        for i, L in enumerate(lang):
            H[i] = {"en": [5, 0, 0, 0], "fr": [0, 5, 0, 0], "de": [0, 0, 5, 0]}[L]
        H += rng.normal(0, 0.1, H.shape)
        df = pd.DataFrame({"translation_group_id": rng.integers(0, 10, n).astype(str),
                           "language": lang, "region": ["x"]*n, "script": ["LATIN"]*n})
        out = variance_partition(H, df)
        self.assertGreater(out["language"], 0.7)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: implement** `decomp.variance_partition`: for each factor build a one-hot design `D`; sequential R² gain (fit cumulative factors via least squares, measure incremental explained variance of `H` averaged over dims); return normalized fractions summing to ≤1 with `residual`. **Step 4: pass. Step 5: commit** `feat(src): FLORES variance decomposition`.

---

### Task 17: `modules/rep_similarity`

**Files:** Create `src/modules/rep_similarity/{__init__,run}.py`; Test `src/tests/test_rep_similarity_module.py`.

**Interfaces:**
- Produces: per layer, CKA/SVCCA/Procrustes between language-conditioned activation matrices and between layers; `cka_matrices/<...>.npy` + `rep_similarity_summary.csv`. Reuses `similarity.*`.

- [ ] **Step 1: failing test** — synthetic store; assert a CKA matrix file is written with 1.0 on the diagonal. **Steps 2-4** standard. **Step 5: commit** `feat(src): representational-similarity module`.

---

### Task 18: `modules/steering` + smoke

**Files:** Create `src/modules/steering/{__init__,run}.py`; Test `src/tests/test_steering_module.py`.

**Interfaces:**
- Produces: `reliability(direction_set)->dict` (mean pairwise cosine of contrast vectors, pos/neg centroid distance, within-class variance, probe margin) — pure, unit-tested on synthetic; `run.py` (model-dependent) does the α-sweep `add_and_generate` and writes `steering_results.csv` + reliability columns.

- [ ] **Step 1: failing test** — `reliability` on synthetic contrast vectors returns expected ranges (tight cluster → high mean cosine). **Steps 2-4** standard (only `reliability` unit-tested). **Step 5: commit** `feat(src): steering reliability + alpha-sweep runner`.

---

### Task 19: `modules/data_stats` + `modules/report`

**Files:** Create `src/modules/data_stats/{__init__,run}.py` (port `scripts/data_stats.py`, read the new `MetadataTable` instead of raw dirs), `src/modules/report/{__init__,run}.py`; Test `src/tests/test_report.py`.

**Interfaces:**
- `data_stats.run(cfg)` writes the counts/length/script/topic/confounds CSVs + plots from `metadata.parquet`.
- `report.success_criteria(direction_record)->dict[str,bool]` (the §15 checklist: decodable, persists_after_controls, transfers, layer_stable, coherent, not_confounded, causal) — pure, unit-tested; `run.py` aggregates the analysis CSVs into a `report_summary.csv` + compiles the figure list.

- [ ] **Step 1: failing test** `src/tests/test_report.py` — `success_criteria` returns all-True for a record that passes, and flips the right flag when one input fails.
- [ ] **Step 2-4** standard.
- [ ] **Step 5: commit** `feat(src): data_stats (metadata-based) + report/success-criteria`.

---

### Task 20: CLI `run.py` + README + smoke verification

**Files:** Create `src/run.py`, `src/README.md`; Test `src/tests/test_cli.py`.

**Interfaces:**
- `python -m src.run <step> [--config src/configs/config.yaml] [--what ...]` dispatches to each module's `run`. Steps: `collect, metadata, extract, normalize, probes, directions, cross, flores-decomp, rep-similarity, steering, data-stats, report`.

- [ ] **Step 1: failing test** `src/tests/test_cli.py` — `from src.run import STEPS; assert set(STEPS) == {...12 names...}` and that each maps to a callable.
- [ ] **Step 2: fail. Step 3: implement** `run.py` with a `STEPS` dict {name→module.run} and an `argparse` dispatcher (default config `src/configs/config.yaml`). Write `src/README.md` (setup: `./setup.sh` or `pip install -r src/requirements.txt`; run order; note Qwen3-8B needs the GPU box, 1.7B/MPS for smoke).
- [ ] **Step 4: run pass.**
- [ ] **Step 5: SMOKE (manual, model)** — build a tiny smoke config (`Qwen/Qwen3-1.7B`, `device: mps`, `paths` under `/tmp/src_smoke`, a few regions) and run, in order: `metadata` (on a small ported raw subset or the existing `/tmp/smoke/data`), `extract`, `normalize`(noop), `probes`, `directions`, `cross`, `flores-decomp`, `rep-similarity`, `data-stats`, `report`. Confirm each writes its CSV/figures and exits 0; confirm `extract` store has `embed`+layers for all readouts. Commit `feat(src): CLI + README + smoke-verified pipeline`.

---

## Self-Review

**Spec coverage:**
- isolated package + shared_utils + module-per-step + CLI → Tasks 1–20. ✓
- collect (+SIB-200, +FLORES IDs) → T9; metadata → T10; multi-readout extract → T8/T11; normalize+centering → T5/T12; probes+transfer → T6/T13; directions → T14; cross → T15; FLORES decomp → T16; rep-similarity → T7/T17; steering+reliability → T8/T18; data_stats → T19; report/success-criteria → T19; store/metadata contract → T3. ✓
- decoder-only (no encoder/sentence) → T8 loads decoder only. ✓
- tests no-GPU + smoke → unit tests in T1–T20, smoke in T11/T20. ✓

**Placeholder scan:** the heavier analysis modules (T14, T15, T17, T18) give the interface + algorithm + CSV schema and a one-line test description rather than full literal test code, because their logic reuses the fully-specified `shared_utils` from T4–T8; the implementer writes the synthetic-data test described. This is intentional (the shared_utils they call are fully coded and tested) — not a TODO. T16 (the novel variance-partition) has full test code.

**Type consistency:** `representation(store, meta, model, readout, layer, repname)` (T12) is consumed identically in T13/T14/T15/T17; `ActivationStore`/`MetadataTable` signatures (T3) match all consumers; `probe_normal(fitted, kind)` (T6) matches T14; `diffmean`/`balanced_background` (T4) match T14; `variance_partition(H, df)` (T16) matches its test. ✓
