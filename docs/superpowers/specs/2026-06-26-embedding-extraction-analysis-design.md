# Design: Embedding-layer extraction + structure & depth analysis

Date: 2026-06-26
Status: Approved design (pending spec review)

## Context

The vector pipeline (`scripts/04_compute_vectors.py`) extracts **DiffMean steering
vectors** (normalized difference-of-means directions) from post-transformer-block
hidden states, and `05`/`06` analyze those directions. We now also want the model's
**embedding-space representation** — the output of the token embedding layer
(`embed_tokens`), i.e. the residual stream **before** transformer block 0 — and to run
analysis on it.

Two analysis goals (both requested):
1. **Structure** — how `lang_region`s / languages / scripts cluster in raw embedding space.
2. **Depth** — how that structure changes from the embedding layer through the
   transformer layers (does region/language structure emerge, strengthen, or wash out
   with depth?).

Key realization that shapes the design: `04`'s `extract_activations_batch` already
produces each region's per-layer mean representation; `save_vector_set` already writes
`raw_means/<key>_layer_XXX.npy` for flores/opus/puzzles. So this work is mostly (a) add
the **embedding layer** to the probed set, (b) save per-region mean representations for
**all** sources (topics/culture currently skip `raw_means`), and (c) a new analysis
script. One forward pass yields both the existing steering vectors and the embeddings.

### Goals
- Capture the `embed_tokens` output as an extra probed layer (`"embed"`).
- Persist per-region/per-topic **mean representations** at the embed layer and the
  transformer layers, for every source, in a consolidated store.
- New `07_analyze_embeddings.py`: embedding-space **structure** + across-depth analysis.
- Fill config gaps (`va_analysis_dir`, `va_plot_dir`) and add `va_embeddings_dir`.

### Non-goals (YAGNI)
- Per-sentence embedding export (only aggregated per-region/per-topic means).
- Changing the DiffMean steering-vector math or `05`/`06`.
- Re-running full extraction locally (8 GB Mac can't; full runs go to the GPU box).

## Design

### Unit 1 — Embedding capture (`shared_utils/activation_extraction.py`)

Extend `extract_activations_batch(...)` with `include_embedding: bool = False`.
When true, additionally capture `model.model.embed_tokens.output` within the same
`model.trace(...)` block and store it under the sentinel layer key `"embed"` in the
returned dict, mean-pooled over tokens with the same attention mask as the other layers.

- Embedding output is a plain tensor (not a tuple) — handle that shape directly.
- Resolution helper `get_embedding_module(model)`: try `inner.model.embed_tokens`,
  fall back to `inner.get_input_embeddings()`; raise a clear error if neither exists.
- Return dict keys become `{"embed", 0, 1, ...}`. `"embed"` sorts/handles as a string;
  downstream code treats layer keys as opaque labels.

What does it do / how used / depends on: produces masked-mean activations per sentence
per layer incl. the embedding layer; called by `04`; depends on NNsight tracing.

### Unit 2 — Per-region mean store (`scripts/04_compute_vectors.py`)

- Add `model.include_embedding_layer` (config, default true) → pass `include_embedding`
  through `process_flat` / `process_puzzles` / `process_topics`.
- Write a consolidated **embeddings store** (raw mean representations, NOT normalized):
  ```
  <va_embeddings_dir>/<source>/layer_embed.npz     # keys = region/topic keys
  <va_embeddings_dir>/<source>/layer_XXX.npz       # one per probed transformer layer
  <va_embeddings_dir>/<source>/metadata.json       # source, layers (incl "embed"), keys, group map
  ```
  Sources: `flores`, `opus100`, `puzzles_original`, `puzzles_translation`, and for
  cultural — `topics` (per-topic pooled mean) and `culture` (per topic×region mean).
- New helper `save_embedding_store(out_dir, acts_by_key, layers, key_prefix, group_map)`:
  for each layer (incl `"embed"`), write `{prefix+key: acts_by_key[key][layer].mean(0)}`.
  `group_map` records each key's `base_language` (region sources) or topic (cultural),
  pulled from the `lang_regions` registry, into `metadata.json`.
- `process_topics` is refactored to also emit the embedding store (it currently writes
  only steering npz). Existing steering-vector outputs and `raw_means` are unchanged.
- Depth layer set: `analysis.depth_layers` (config, optional) selects which transformer
  layers enter the store; default = the probed layers. `"embed"` is always included.

### Unit 3 — Analysis (`scripts/07_analyze_embeddings.py`)

Reads `<va_embeddings_dir>/<source>/`. Group labels from `metadata.json` (`base_language`
for region sources; topic for cultural). Operates per region-keyed source (default:
`flores`, `opus100`, `puzzles_original`, `puzzles_translation`).

**Structure (at the embed layer):**
- Cosine-similarity matrix between region embeddings (`vectors.cosine_similarity_matrix`)
  → `embedding_similarity_<source>.csv` (`vec_a, vec_b, cosine_sim`).
- PCA to 2-D (`analysis.n_pca_components`) → `embedding_pca_<source>.json`
  (`coordinates, labels, groups, explained_variance`).
- Clustering: `pairwise_distance_matrix(..., "cosine")` →
  `clustering.hierarchical_cluster(dm, labels, n_clusters=#base_languages)`; score against
  the true `base_language` grouping with `clustering.cluster_agreement_scores` (ARI/NMI)
  and `sklearn.metrics.silhouette_score` (cosine) → `embedding_clusters_<source>.csv`
  (`key, cluster, base_language`) + scores in `embedding_structure_summary.csv`.

**Depth (across layers, per source):**
- For each layer in the store (`embed`, then transformer layers), compute a structure
  score = silhouette of region embeddings grouped by `base_language` (cosine metric);
  also record mean within-group minus cross-group cosine.
- Output `depth_structure.csv` (`source, layer, silhouette, within_minus_cross`) and an
  explicit `embed` vs deepest-layer delta. Optional: `clustering.mantel_test` between the
  embed-layer and deepest-layer distance matrices (do the relations persist with depth?).

**Figures** (`06`-style, written to `va_plot_dir`):
- `emb_fig1_similarity_<source>.{png,pdf}` — embedding similarity heatmap
- `emb_fig2_pca_<source>.{png,pdf}` — 2-D PCA scatter, colored by `base_language`
- `emb_fig3_dendrogram_<source>.{png,pdf}` — hierarchical clustering dendrogram
- `emb_fig4_depth_structure.{png,pdf}` — structure score vs layer (one line per source)

What does it do / how used / depends on: consumes the embedding store + registry groups,
emits CSVs and figures; depends on `shared_utils/{data,vectors,clustering}.py`, sklearn,
matplotlib.

### Unit 4 — Config & pipeline

- `riddles_config.yaml`:
  - `model.include_embedding_layer: true`
  - `analysis.depth_layers: <list|null>` (null → all probed layers)
  - `paths.va_embeddings_dir`, `paths.va_analysis_dir`, `paths.va_plot_dir`
    (the latter two were missing — also unblocks `05`/`06`), all under the SSD root.
- `pipeline.sh`: add stage `embed-analysis` → `07_analyze_embeddings.py`
  (alias `07`). Extraction stays in the `vectors`/`04` stage.

## Data flow

```
04 (extended): corpora --forward pass--> per-layer masked-mean activations
   --> steering vectors  (vectors/<source>/...)            [unchanged]
   --> embedding store   (embeddings/<source>/layer_*.npz) [new]
07: embeddings/<source>/ + registry groups
   --> CSVs (va_analysis_dir) + figures (va_plot_dir)
```

## Error handling
- Embedding module not found → raise `RuntimeError` with the model's module names.
- A source/layer with <2 keys → skip its structure/clustering with a logged warning
  (DiffMean/silhouette undefined); depth metric for that layer recorded as null.
- Silhouette/PCA on degenerate input (identical vectors, single group) → guarded, emit
  null and warn rather than crash.
- `07` skips a source whose embedding store is absent (warn), so partial runs work.

## Testing
- **Unit (synthetic activations, no model/GPU)** in `scripts/tests/`:
  - `extract_activations_batch(include_embedding=True)` returns an `"embed"` key
    (monkeypatched trace) — or test the pure mean-pool path with a fake.
  - `save_embedding_store` layout: npz per layer incl `layer_embed.npz`, correct
    prefixed keys, `metadata.json` group map.
  - structure metrics: cosine matrix shape/symmetry, silhouette on separable synthetic
    groups is high, `depth_structure.csv` columns.
- **Smoke (MPS, 1.7B sample):** run extended `04` then `07` on `/tmp/smoke/data`
  → embedding store for all sources, CSVs + 4 figures produced, exit 0.

## Scope / constraints
- Meaningful clustering needs many regions → real results require the **full data on the
  GPU box**. Locally we only smoke-test the scripts on the 1.7B sample (few regions).
- The embed-layer representation is a bag-of-token-embedding mean, so embedding-space
  structure is expected to track script/vocabulary; the depth analysis is what reveals
  whether deeper semantic/region structure emerges. This is the intended finding, not a bug.
