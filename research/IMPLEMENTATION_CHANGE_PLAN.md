# Implementation Change Plan — aligning Culture_Puzzles to the Research Plan

Date: 2026-06-26
Inputs: `representation_analysis_research_plan.docx` (17 sections, refs R1–R24),
`..._literature_map_ANNOTATED.xlsx` (168 papers), and `data_stats.py` output.

This document compares the **current implementation** to the **research plan**, grounds
the comparison in our **data statistics** (and their confounds), and gives a prioritized,
concrete plan of changes.

---

## A. What the data says (from `data_stats.py`) — drives priorities

| Fact (measured) | Plan implication |
|---|---|
| 4 sources with very different lengths (token mean: flores **67.6**, cultural **53**, puzzles **51.4**, opus **31**; cultural std **≈95**) | **length & source are confounds** (§4, §7, §8): need length/source as probed+controlled factors and **balanced backgrounds** for steering means (§6.3). |
| **6 base languages span 15 regions** (Arabic×4, Spanish×3, Bengali×2, English×2, Chinese×3, Portuguese×2) | For flores/opus/cultural, same-language regions get **identical (shared) text** → their region signal is **degenerate**; only **puzzles** carry true per-region variation. Region vs language_region tests (§10.2) are only valid where text actually differs (puzzles everywhere; Arabic via FLORES varieties arz/ary/apc/arb). |
| **12 scripts** (Arabic, Bengali, Buginese, CJK, Cyrillic, Devanagari, Latin, Malayalam, Sinhala, Tamil, Telugu, Thai) | **script/tokenization confound** (§3, §11): need a **script probe** and the **layer-0 baseline** (already added) to separate script from semantics. |
| Puzzle topics: **70 distinct labels**, only **8 canonical** present in all 46 regions; cultural = clean 8 topics but **Wikipedia-derived (language-level)** | **topic confounded with source/region** (§7). Use the 8 canonical topics as the analysis label set; treat wildcard labels separately; add a **clean topic-labeled multilingual corpus (SIB-200)** (§7.3). |

---

## B. Alignment scorecard (plan section → current state)

Legend: ✅ done · 🟡 partial · ❌ missing

| § | Plan requirement | Current state |
|---|---|---|
| 1 | Factorized representation (topic/lang/region/lang_region/script/source/prompt + interactions) | 🟡 we model topic, language(region), culture(topic×region); **no script/source/length/prompt factors** |
| 2 | decoder-only **+ encoder (mBERT/XLM-R) + sentence-embedding (LaBSE/E5/SBERT)** | ❌ decoder-only Qwen3 only |
| 3 | residual stream every layer; resid_pre/attn/mlp/resid_post; layer-0 baseline | 🟡 every layer `resid_post` + **embed baseline ✅ (new)**; **no attn/mlp/resid_pre split** |
| 4 | mean over **content** tokens **+ last content token**; content masks | 🟡 masked-mean over all non-pad tokens; **no last-token readout**; **no content-vs-answer/instruction masks** (riddle lines include the answer) |
| 5 | sweep all layers, report curves | ✅ `layers: all`; depth curves in `07` |
| 6 | per-layer standardization (train stats); raw + language/lang_region/topic-centered; mean-centred steering; unit-norm for cosine | ❌ no standardization; ❌ no centering variants; ✅ unit-norm DiffMean; 🟡 steering mean not balanced-background |
| 7 | balanced metadata table; FLORES decomposition; SIB-200 topics | 🟡 lang_region+topic+source implicit, `shared_groups`; ❌ no unified metadata (script/length/domain/prompt/translation_group_id); ❌ no SIB-200 |
| 8 | linear probes + diff-in-means; probe **all factors**; held-out splits | ❌ **only diff-in-means**; no logistic/SVM probes, no probe accuracy, no held-out transfer |
| 9 | diff-in-means **vs logistic/SVM normals** (cosine agreement) | 🟡 diff-in-means only; ❌ no probe-normal comparison |
| 10 | cross-language topic alignment + held-out transfer; **separate language from region**; cross-topic RDMs | 🟡 cross/within cosine in `05`; ❌ no held-out-language transfer; ❌ no explicit same-lang/diff-region contrasts; ❌ no RDMs |
| 11 | FLORES variance decomposition (sentence_id+language+region+script) | ❌ FLORES collected but **aligned sentence IDs not kept**; no decomposition |
| 12 | CKA, SVCCA/PWCCA, Procrustes, RDM, centroid cosine, subspace angles | 🟡 cosine matrices + PCA; ❌ CKA/SVCCA/Procrustes/RDM/subspace-angle (note: `vectors.py` has `subspace_angle`, unused) |
| 13 | activation-addition steering (α sweep) + reliability diagnostics | ❌ not implemented in the riddles pipeline (config has a steering block, unused) |
| 14 | outputs: layer_probe_scores, transfer_scores, topic_vector_cosines, flores_decomposition, cka_matrices, steering_results, figures | 🟡 vector cosines + embedding analysis; ❌ probe/transfer/flores/cka/steering CSVs |
| 15 | pre-registered success criteria | ❌ not evaluated |

**Net:** the collection layer is in good shape; the **analysis layer is mostly DiffMean-only** and misses the plan's probing, normalization/centering, transfer, similarity-geometry, FLORES decomposition, and steering-validation machinery — plus the metadata/factor scaffolding everything else depends on.

---

## C. Change plan (prioritized)

Each item: what to add/change, where, and the plan section it satisfies.

### P0 — Foundations everything else needs

1. **Unified per-sample metadata table** (`08_build_metadata.py` → `metadata.parquet/csv`).
   Columns from §7.1: `sample_id, text, source, topic, topic_canonical, language(base),
   region, language_region, script, domain, prompt_template, token_count,
   translation_group_id`. Derive `script` (already done in `data_stats.detect_script`),
   `token_count` (tokenizer), `language/region` from the registry, `topic_canonical` via
   the existing `topic_label_map` (map the 70 puzzle labels → 8 canonical, keep raw as
   `topic_raw`). This is the join key for all probing/centering/decomposition. (§7)

2. **Multi-readout activation extraction.** Extend `extract_activations_batch` to return,
   per layer: `mean_content`, `last_content`, and keep `embed` (✅). Add **content-token
   masks** that exclude BOS/EOS/PAD and, for riddles, the reference-answer span (store the
   riddle/answer split already present in `riddles.jsonl`). (§3, §4) — touches
   `shared_utils/activation_extraction.py`, `04`, and the embedding/vector stores gain a
   `readout` dimension.

3. **Per-layer standardization + centering variants** (`shared_utils/normalize.py`).
   `standardize(H, mu_l, std_l)` using **train-split** stats; produce `raw`,
   `language_centered`, `language_region_centered`, `topic_centered`, `source_centered`
   representations (subtract the conditional centroid). Persist `mu_l/std_l` from train
   only. (§6) Used by probing, similarity, and direction analysis.

### P1 — Core analyses the plan is built around

4. **Probing module** (`09_probes.py`). For each (model, layer, readout, representation
   variant): train **logistic-regression** and **linear-SVM** probes (+ keep diff-in-means)
   for `topic, language, region, language_region, script, source, token_count(bin),
   prompt_template`. Report macro-F1/AUROC. Splits: **random, held-out-language,
   held-out-region, held-out-language_region, held-out-source, held-out-prompt** (§8).
   Outputs `layer_probe_scores.csv`, `transfer_scores.csv` (§14). Reuse
   `sklearn`. This is the single biggest gap.

5. **Direction analysis upgrade** (extend `04`/new `10_directions.py`). Keep DiffMean
   (`v = mean(topic) − mean(balanced_background)`, **balanced over language/region/source/
   length** per §6.3) and add the **logistic & SVM normal vectors**; report
   `cos(v_diffmean, v_logistic)`, `cos(·, v_svm)` (§9). Also compute per-(topic,language)
   and per-region vectors explicitly (§9 formulas). Output `topic_vector_cosines.csv`.

6. **Cross-language / region / topic analysis** (`11_cross_analysis.py`). Cross-language
   topic-vector cosine + **held-out-language transfer**; **same-language/different-region**
   and **different-language/same-region** contrasts (valid set: puzzles everywhere; Arabic
   via FLORES varieties); cross-topic **RDMs**, centroid distance matrices, probe confusion
   matrices — interpret only **after** language/language_region centering (§10).

7. **FLORES decomposition** (`12_flores_decomp.py`). Requires P0.1 `translation_group_id`
   (keep FLORES aligned sentence IDs in `02_collect_parallel.py` — currently dropped).
   Fit `h(sentence_id, language, region, script) = sentence + language + region + script +
   residual` (per-layer ANOVA / variance partition); output `flores_decomposition.csv`
   (§11). **Note from data:** for same-language regions sharing FLORES text the region
   term is null by construction except Arabic varieties — report this explicitly.

### P2 — Geometry, steering, and the science framing

8. **Representational-similarity module** (`13_rep_similarity.py`). Linear **CKA**,
   **SVCCA/PWCCA**, **Procrustes**, **RDMs**, centroid cosine, **subspace angles** (wire up
   the existing unused `vectors.subspace_angle`). Compare across languages, datasets, and
   layers. Output `cka_matrices/` (§12). These measures have different invariances — report
   several, don't treat as interchangeable [R22–R24].

9. **Steering + reliability** (`14_steering_eval.py`). Activation addition/removal with
   **α ∈ {−3,−2,−1,−0.5,0.5,1,2,3}** on the decoder-only model, **only for directions that
   pass probing/transfer/geometry** (§13). Reliability diagnostics: mean pairwise cosine of
   contrast vectors, pos/neg centroid distance, within-class variance, probe margin,
   cross-language & cross-template vector cosine. Output `steering_results.csv`.
   Heed [R7][R8][R9]: good detection ≠ good steering; prompting may beat steering.

10. **Success-criteria evaluation + report** (`15_report.py`). Apply §15 checklist to each
    candidate direction (decodable / persists after controls / transfers / layer-stable /
    coherent / not script-source-length-prompt explained / causal). Emit the §14 figure set
    and a results summary, including **negative results** (where topic is confounded, where
    steering fails, where vectors are language-specific).

### P3 — Models & datasets (broaden scope)

11. **Encoder + sentence-embedding baselines** (`shared_utils/model.py` + config). Add
    mBERT/XLM-R (encoder, mean-pooled + CLS) and LaBSE/multilingual-E5 (sentence embeddings)
    as additional `model.family` entries; run the same probing/similarity (§2). NNsight
    path is decoder-specific — encoders can use plain HF hidden_states.

12. **SIB-200 collector** (`16_collect_sib200.py`). Add SIB-200 (topic labels across 200+
    languages, FLORES-derived) as a **clean topic-labeled multilingual corpus** to
    complement the Wikipedia topics, addressing the topic-confound (§7.3) [R20]. Map its
    topics/regions into the same metadata schema.

13. **Resid_pre/attn_out/mlp_out capture** (optional, `activation_extraction.py`). For the
    decoder model, additionally save attention- and MLP-output streams to localize where
    factors emerge (§3).

---

## D. Concrete edits to existing files

- `02_collect_parallel.py`: **keep the aligned FLORES sentence index** (`translation_group_id`)
  per sentence (needed for §11); currently it stores only the text lines.
- `04_compute_vectors.py`: switch DiffMean background from "all other keys" to a
  **balanced background** (§6.3); add `readout` dimension (mean_content + last_content);
  consume the metadata table; emit standardized/centered variants.
- `shared_utils/activation_extraction.py`: multi-readout + content/answer masks (§4).
- `shared_utils/vectors.py`: expose `subspace_angle` use; add CKA/Procrustes helpers (or a
  new `similarity.py`).
- `configs/riddles_config.yaml`: add `models:` list (decoder/encoder/sentence-embedding),
  `readouts: [mean_content, last_content, embed]`, `representations: [raw, lang_centered,
  lang_region_centered, topic_centered, source_centered]`, `probes:`, `splits:`,
  `steering: alpha_range`, and a `canonical_topics:` list for analysis.
- `pipeline.sh`: add stages `metadata`, `probes`, `directions`, `cross`, `flores-decomp`,
  `rep-sim`, `steering`, `report` (and `sib200`).

---

## E. Recommended order of execution

1. P0 (metadata, multi-readout, normalization) — unblocks everything.
2. P1.4 probing + P1.5 directions — the plan's empirical core; produces
   `layer_probe_scores.csv` + `transfer_scores.csv` + `topic_vector_cosines.csv`.
3. P1.6 cross-analysis + P1.7 FLORES decomposition.
4. P2.8 similarity + P2.9 steering + P2.10 report.
5. P3 (encoder/sentence-embedding models, SIB-200, sub-stream capture) as scope allows.

**Caveats to bake into every analysis (from §B + data stats):** report whether a result
survives language/language_region centering and held-out-source; never claim region
structure from sources where same-language regions share text; always show the layer-0
baseline alongside deep-layer claims; treat the 8 canonical topics as the label set and
keep length/script/source as explicit probed confounds.
```
