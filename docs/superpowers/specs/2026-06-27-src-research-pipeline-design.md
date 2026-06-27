# Design: `src/` — isolated representational-analysis research pipeline

Date: 2026-06-27
Status: Approved design (pending spec review)
Requirements source: `research/IMPLEMENTATION_CHANGE_PLAN.md` (P0–P3) and
`research/representation_analysis_research_plan.docx` (§1–§17, refs R1–R24).

## Context

The existing `scripts/` pipeline collects multilingual cultural data and computes
DiffMean steering/embedding vectors, but the analysis layer is DiffMean-only and misses
most of the research plan: factor metadata, multi-readout extraction, per-layer
standardization + centering, linear/SVM probes with held-out transfer, FLORES variance
decomposition, representational-similarity geometry (CKA/SVCCA/Procrustes/RDM), and
steering causal validation. This project builds a **new, fully isolated `src/` package**
that implements the change plan end-to-end (all of P0–P3) and runs on its own
(re-collecting data from scratch, including SIB-200).

### Goals
- A self-contained `src/` package: own `configs/`, `shared_utils/`, a `modules/`
  sub-package **per research step**, one CLI; **no imports from `scripts/`**.
- Implement every change-plan item fully: collection (+SIB-200, +FLORES aligned IDs),
  unified metadata, multi-readout all-layer extraction across 3 model families, per-layer
  standardization + centering variants, probing + held-out transfer, direction analysis,
  cross-language/region/topic analysis, FLORES decomposition, representational similarity,
  steering + reliability, data stats, and a report step.
- Unit-test all non-GPU logic; smoke-test model-dependent steps on Qwen3-1.7B/MPS.

### Non-goals
- Modifying or deleting the existing `scripts/` pipeline (left as-is).
- Producing final full-scale results here (heavy/full runs target a CUDA box).

## Architecture & isolation

`src/` is a Python package run as `python -m src.run <step> [--config src/configs/config.yaml]`.
Imports are src-relative (`from src.shared_utils...`). It has its own
`src/requirements.txt`. The **store-centric** contract: `collect` writes raw corpora →
`metadata` writes one table keyed by `sample_id` → `extract` writes an activation store
keyed by `(model, layer, readout)` aligned to `sample_id` → analysis steps read
store+metadata and write CSVs/figures → `report` aggregates.

```
src/
  __init__.py  run.py  requirements.txt  README.md
  configs/config.yaml
  shared_utils/  __init__.py io.py registry.py text.py models.py extraction.py
                 normalize.py vectors.py probes.py similarity.py steering_utils.py
                 store.py plotting.py
  modules/
    collect/  metadata/  extract/  normalize/  probes/  directions/  cross/
    flores_decomp/  rep_similarity/  steering/  data_stats/  report/
      (each: __init__.py, run.py, + focused logic files)
  tests/
```

## shared_utils (ported/adapted from `scripts/shared_utils`, made src-relative)

- `io.py` — `load_config`, `save_json/load_json`, `save_jsonl/load_jsonl`,
  `save_csv/load_csv`, `setup_logging`, `ensure_dirs`. (from `data.py`)
- `registry.py` — `load_registry(cfg)`, `region_to_factors(key, cfg)` →
  `{base_language, region, language_region, wiki, flores, opus}`; FLORES-code validation.
- `text.py` — `detect_script(text)`, `sentence_split(text)`, and
  `content_token_mask(tokenizer, text, answer_span=None)` → boolean mask excluding
  BOS/EOS/PAD and (for riddles) the reference-answer span. (§4)
- `models.py` — `load_model(cfg, family)` for `decoder` (NNsight `LanguageModel`),
  `encoder` (HF `AutoModel`, output_hidden_states), `sentence` (sentence-transformers).
  Returns a uniform handle with `num_layers`, `hidden_size`, `family`.
- `extraction.py` — `extract(model_handle, texts, layers, readouts, tokenizer, masks)` →
  `{readout: {layer: ndarray(n, d)}}`; readouts ∈ {`mean_content`, `last_content`,
  `embed`}; decoder uses NNsight trace (embed captured before blocks — known ordering
  rule), encoder uses `hidden_states`, sentence uses `.encode`. (§3, §4)
- `normalize.py` — `fit_stats(H_train)`→`(mu_l, std_l)`; `standardize(H, mu, std)`;
  `center(H, group_ids)` for language/language_region/topic/source centering. Train-split
  stats only. (§6)
- `vectors.py` — `diffmean(target, background, normalize=True)` with **balanced
  background** sampling; `cosine`, `cosine_matrix`, `subspace_angle`. (§6.3, §9)
- `probes.py` — `train_probe(X, y, kind)` for `logistic`/`svm`/`diffmean`;
  `make_splits(metadata, scheme)` for random/held-out-language/region/language_region/
  source/prompt; `probe_normal(probe)` → the linear normal vector. (§8, §9)
- `similarity.py` — `linear_cka(X, Y)`, `svcca(X, Y)`, `procrustes_align(X, Y)`,
  `rdm(X)`, `subspace_angles(A, B)`. (§12)
- `steering_utils.py` — `add_direction(model, layer, v, alpha)`, `generate(...)`,
  reliability helpers. (from `steering.py`, §13)
- `store.py` — `ActivationStore` writing/reading `(model, layer, readout)` arrays +
  `sample_id` index (npz/parquet); `MetadataTable` (parquet/csv) load/save/join.
- `plotting.py` — heatmap/line/scatter/box helpers (Agg backend).

## modules (one sub-module per research step)

Each has `run.py` (`python -m src.run <step>` dispatches here) + logic files. Outputs go
to config `paths` dirs.

1. **collect** — `puzzles.py`, `parallel.py`, `topics.py`, `sib200.py`. Port the
   `scripts/` collectors (puzzles riddles, FLORES/OPUS parallel **keeping the aligned
   sentence id `translation_group_id`**, Wikipedia topics) and add **SIB-200** (HF
   `Davlan/sib200` or equivalent: topic-labeled, FLORES-derived, 200+ langs). Writes raw
   corpora under `paths.raw_dir`. (§7, R18–R20)
2. **metadata** — build `metadata.parquet` keyed by `sample_id` with columns
   `text, source, topic, topic_canonical, topic_raw, language, region, language_region,
   script, domain, prompt_template, token_count, translation_group_id, split`. Maps the
   70 messy puzzle topics → 8 canonical via the topic map; assigns train/test split. (§7)
3. **extract** — for each configured `(model_family, model)` run `extraction.extract`
   over all layers + readouts; write the activation store. (§3, §4, §5)
   *Note:* `decoder`/`encoder` yield per-layer readouts; the `sentence` family yields a
   single pooled embedding stored as one pseudo-layer (`layer = "sentence"`), so downstream
   steps treat layer labels as opaque.
4. **normalize** — fit per-layer train stats; emit standardized + centered representation
   variants (raw, language_centered, language_region_centered, topic_centered,
   source_centered) as derived views in the store. (§6)
5. **probes** — train logistic + SVM + diffmean probes for every factor
   (topic, language, region, language_region, script, source, token_count-bin,
   prompt_template) × layer × readout × representation; evaluate on random +
   held-out-{language,region,language_region,source,prompt}. Output
   `layer_probe_scores.csv`, `transfer_scores.csv`. (§8)
6. **directions** — diffmean vectors (balanced background) + logistic/SVM normals;
   per-(topic), per-(topic,language), per-(language), per-(region) directions; cosine
   agreement `cos(v_diffmean, v_logistic/svm)`. Output `topic_vector_cosines.csv`. (§9)
7. **cross** — cross-language topic-vector alignment + held-out-language transfer;
   same-language/different-region and different-language/same-region contrasts (valid
   where text differs: puzzles everywhere, Arabic FLORES varieties); cross-topic RDMs,
   centroid-distance + probe-confusion matrices; interpret post-centering. (§10)
8. **flores_decomp** — using `translation_group_id`, per-layer variance partition
   `h = sentence + language + region + script + residual` (ANOVA via statsmodels / numpy
   least squares). Output `flores_decomposition.csv`. (§11)
9. **rep_similarity** — CKA, SVCCA/PWCCA, Procrustes, RDMs, centroid cosine, subspace
   angles across languages/datasets/layers. Output `cka_matrices/` + CSVs. (§12)
10. **steering** — activation addition/removal with α∈{−3,−2,−1,−0.5,0.5,1,2,3} on the
    decoder model for directions passing earlier gates; reliability diagnostics (contrast
    cosine, centroid distance, within-class variance, probe margin, cross-language/template
    cosine). Output `steering_results.csv`. (§13, R7–R9)
11. **data_stats** — port `scripts/data_stats.py` (counts/length/script/topic coverage +
    confounds + plots), reading the new metadata table.
12. **report** — apply §15 success-criteria checklist to candidate directions; compile the
    §14 figure set and a results summary incl. **negative results**. (§14, §15)

## Config schema (`src/configs/config.yaml`)

`models:` list of `{family, name}` (decoder Qwen3, encoder XLM-R/mBERT, sentence LaBSE/E5);
`readouts: [mean_content, last_content, embed]`;
`representations: [raw, language_centered, language_region_centered, topic_centered, source_centered]`;
`probes: {kinds: [logistic, svm, diffmean], splits: [random, heldout_language, heldout_region, heldout_language_region, heldout_source, heldout_prompt]}`;
`steering: {alpha: [...], max_new_tokens, ...}`;
`canonical_topics: [...8...]`; `lang_regions:` registry (ported); `cultural_topics:`/`seed_en`;
`data: {samples_per_*}`; `paths:` (raw_dir, metadata, store_dir, analysis_dir, plot_dir — SSD root, configurable).

## Error handling
- Missing model family deps (sentence-transformers/encoder) → clear error naming the dep;
  the step skips that family with a logged warning rather than aborting the whole run.
- Degenerate probe/similarity inputs (<2 classes, single group, zero-variance) → return
  null metric + warn (no crash).
- Same-language shared-text regions → analysis steps detect identical corpora and **flag**
  the degenerate region term (esp. FLORES decomposition) rather than reporting fake signal.
- Store/metadata `sample_id` mismatch → hard error (it's a contract violation).
- Heavy steps (`extract`, `steering`) are not run by unit tests; CPU/MPS OOM is surfaced.

## Testing
- **Unit (no GPU)**: `text` masks + script; `registry` factor mapping; `metadata` build
  from a tiny fixture corpus; `normalize` standardize/center identities; `vectors` diffmean
  balanced background + unit norm; `probes` recover separable synthetic classes + split
  builders produce disjoint held-out groups; `similarity` CKA(X,X)=1, Procrustes on a known
  rotation; `flores_decomp` variance partition on synthetic factorial data; `directions`
  cosine agreement on synthetic; `report` criteria logic.
- **Smoke (MPS, Qwen3-1.7B, tiny subset)**: `extract` produces a store with all readouts +
  layers; `steering` runs an α sweep on one direction; one analysis end-to-end on the smoke
  store. Reuse the existing `/tmp/smoke` sample.
- Run from repo root: `python -m pytest src/tests` or unittest discovery; tests must not
  require a GPU.

## Dependencies (`src/requirements.txt`)
Reuse: numpy, torch, transformers, nnsight, scipy, scikit-learn, pandas, matplotlib,
openpyxl, datasets, wikipedia-api==0.6.0, tqdm, PyYAML. Add: `sentence-transformers`,
`statsmodels`, `pyarrow` (parquet).

## Build order (informs the plan)
shared_utils foundations → store/metadata → collect → extract → normalize → probes →
directions → cross → flores_decomp → rep_similarity → steering → data_stats → report → CLI.
