# Culture Puzzles — Cultural Riddles Vector Pipeline

Computes DiffMean steering vectors over cultural data keyed on **lang_region**
(e.g. `Arabic_Egypt`), from three sources: cultural riddles (puzzles), FLORES/OPUS
parallel text, and Wikipedia topic text.

## Setup

```bash
./setup.sh                 # creates .venv, installs requirements, seeds env.sh
# then edit env.sh -> fill in HF_TOKEN / OPENAI_API_KEY
```

## Run

```bash
./pipeline.sh              # puzzles -> parallel -> topics -> vectors
./pipeline.sh puzzles      # a single stage
./pipeline.sh topics vectors   # a subset, in order
./pipeline.sh all          # everything incl. analyze + plots
```

Both scripts use `.venv` and `source env.sh` automatically.

> FLORES (`facebook/flores`) is a **gated** HF dataset: the `HF_TOKEN` account must
> accept its terms, or `02` returns 403 (it skips gracefully).
> Python 3.9 note: `wikipedia-api` is pinned to `0.6.0` (newer needs 3.10+).

## Pipeline (run from `scripts/`, all take `--config configs/riddles_config.yaml`)

| Order | Script | Output |
|-------|--------|--------|
| 1 | `01_collect_puzzles.py` | `…/data/puzzles/<lang_region>/{original,translation}.txt`, `riddles.jsonl`, manifest |
| 2 | `02_collect_parallel.py` | `…/data/parallel/{flores,opus100}/<lang_region>.txt` |
| 3 | `03_collect_topics.py` | `…/data/cultural/<topic>/<lang_region>.txt` |
| 4 | `04_compute_vectors.py` | `…/vectors/<dataname>/layer_XXX.npz` (+ `language_vectors/`,`topic_vectors/` mirrors) |
| 5–6 | `05_analyze_vectors.py`, `06_generate_plots.py` | analysis & plots |

`01` runs first: its manifest is the authoritative lang_region/topic inventory.
`04` needs Qwen3-8B on CUDA. Config registry (`configs/riddles_config.yaml`) is
variety-aware: each region maps to the most region-specific wiki/flores code a
source offers, `null` when unavailable.

## Tests

```bash
cd scripts && ../.venv/bin/python -m unittest discover -s tests -v
```

Covers riddle xlsx parsing (sheet/column resolution), the config registry,
parallel dedup/replication, and the vector builder (DiffMean, content dedup,
output layout) — all without network or GPU.

## Source data

`v0 - due May 29/` — 46 `Cultural Riddles Benchmark [<lang_region>].xlsx` files.
**Not included in this repo** (gitignored); place the folder at the project root locally.
Before running `01`, the folder must be clean (no duplicate-key files, no
filenames whose bracket key isn't in the registry) — `01` fails loudly otherwise.
