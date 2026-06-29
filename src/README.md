# Isolated `src/` Research Pipeline

This is a self-contained Python package implementing a 12-step neural representation analysis pipeline for multilingual cultural probing. The pipeline processes text through language models (LLMs) and performs a series of analyses to identify and characterize cultural direction vectors.

## Quickstart

```bash
src/setup.sh                 # create repo-root .venv + install src/requirements.txt + seed env.sh
# edit env.sh (HF_TOKEN for FLORES/SIB-200), then:
src/pipeline.sh              # run the full pipeline in order
src/pipeline.sh data         # group: collect metadata extract
src/pipeline.sh analyze      # group: normalize..report
src/pipeline.sh probes       # a single step
src/pipeline.sh --config src/configs/smoke.yaml extract   # alternate config
```

`pipeline.sh` wraps the Python CLI (`python -m src.run <step>`); use the CLI directly for finer control. Both run from the repo root and use the repo-root `.venv`.

## Installation

### Requirements
- Python 3.9+
- Virtual environment (recommended)
- Dependencies listed in `requirements.txt`

### Setup

1. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

2. Create or update your config YAML file (see [Configuration](#configuration) below).

## Quick Start

Run a single pipeline step:
```bash
python -m src.run <step> --config src/configs/config.yaml
```

### Available Steps
The 12 steps of the pipeline, in recommended execution order:

1. **collect** — Gather raw text data from Wikipedia, parallel corpora, and topical sources
2. **metadata** — Build unified metadata table with language, region, topic, script labels
3. **extract** — Extract model activations (embeddings + hidden layer outputs)
4. **normalize** — Validate representation variants (standardized, centered by language/region/topic)
5. **probes** — Train linear probes for each cultural factor (topic, language, region, script, source, prompt)
6. **directions** — Compute topic direction vectors (DiffMean vs. logistic/SVM probe normals)
7. **cross** — Cross-language/region analysis: cosine similarities, regional contrasts, heldout-language transfer
8. **flores-decomp** — Variance partition: FLORES-derived vs. Wikipedia-based representation differences
9. **rep-similarity** — Representation similarity analysis (RDMs, correlations across layers)
10. **steering** — Activation-addition steering experiments: reliability diagnostics + alpha-sweep generation
11. **data-stats** — Dataset statistics: sample counts, token distributions, script coverage, confound analysis
12. **report** — Final success-criteria aggregation and summary report

### Example: Full Pipeline Run

```bash
# Sequential execution of all steps (requires model + GPU)
for step in collect metadata extract normalize probes directions cross flores-decomp rep-similarity steering data-stats report; do
    python -m src.run $step --config src/configs/config.yaml
    if [ $? -ne 0 ]; then
        echo "Step '$step' failed"
        exit 1
    fi
done
```

## Configuration

The pipeline reads from a YAML configuration file (default: `src/configs/config.yaml`). Key sections:

### paths
Specifies input/output directories on SSD:
- `raw_dir` — Raw collected text (from step 1: collect)
- `metadata` — Parquet file with unified metadata table (output of step 2: metadata)
- `store_dir` — ActivationStore directory for extracted activations (output of step 3: extract)
- `analysis_dir` — Analysis outputs (CSVs, plots) from steps 5–12
- `plot_dir` — Additional plotting outputs

Example (local SSD):
```yaml
paths:
  raw_dir:      "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/raw"
  metadata:     "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/metadata.parquet"
  store_dir:    "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/store"
  analysis_dir: "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/analysis"
  plot_dir:     "/Volumes/Extreme SSD/dfki/2026/Culture_puzzles/src_results/plots"
```

### model
LLM settings:
- `batch_size` — Batch size for extraction (4 recommended)
- `max_seq_len` — Max token sequence length (128)
- `device` — Device: `"mps"` (Apple GPU), `"cuda"` (NVIDIA), or `"cpu"`
- `dtype` — Data type: `"float16"` or `"float32"`

### models
List of model identifiers to process. Example:
```yaml
models: ["Qwen/Qwen3-1.7B"]  # Local smoke test
# OR
models: ["Qwen/Qwen3-8B"]    # Full pipeline (GPU box)
```

### readouts
Activation readout types (default: `["mean_content", "last_content", "embed"]`).

### representations
Representation variants for normalization (default: `["raw", "language_centered", "language_region_centered", "topic_centered", "source_centered"]`).

### probes.kinds & probes.splits
Probe types (`logistic`, `svm`) and cross-validation splits (`random`, `heldout_language`, etc.).

### steering
Activation-addition alpha parameters and generation token limit.

### lang_regions
List of language+region combinations (e.g., `Arabic_Egypt`, `Bengali_India`). Each entry specifies Wikipedia, FLORES, and Opus language codes.

## Model Selection

### Local Smoke Test (Recommended for Development)
- **Model:** `Qwen/Qwen3-1.7B`
- **Device:** `mps` (Apple Silicon) or `cpu`
- **Data:** Small subset or `/tmp/smoke/` 
- **Time:** ~10 minutes for all 12 steps
- **No GPU box required**

Configuration:
```yaml
models: ["Qwen/Qwen3-1.7B"]
model:
  device: "mps"
  dtype: "float16"
  batch_size: 2
```

### Full Pipeline (GPU Box Required)
- **Model:** `Qwen/Qwen3-8B` or larger
- **Device:** `cuda` (NVIDIA GPU)
- **Data:** Full multilingual dataset
- **Time:** Several hours
- **GPU box required** (24+ GB VRAM)

Configuration:
```yaml
models: ["Qwen/Qwen3-8B"]
model:
  device: "cuda"
  dtype: "float16"
  batch_size: 4
```

## Output Files

Each step produces CSV files and plots in `cfg['paths']['analysis_dir']`:

| Step | Output Files |
|------|--------------|
| collect | Raw text files in `raw_dir` |
| metadata | `metadata.parquet` in `paths.metadata` |
| extract | ActivationStore (layer HDF5 files) in `store_dir` |
| normalize | (validation only; no output) |
| probes | `layer_probe_scores.csv`, `transfer_scores.csv` |
| directions | `topic_vector_cosines.csv` |
| cross | `cross_language_topic_cosine.csv`, `region_contrasts.csv`, `heldout_language_transfer.csv` |
| flores-decomp | `variance_partition_<model>_<readout>_<layer>.csv` |
| rep-similarity | `rdm_<model>_<readout>_<layer>.csv`, correlation plots |
| steering | `steering_results.csv` |
| data-stats | `overview_by_source.csv`, `language_by_source.csv`, `script_coverage.csv`, plots |
| report | `report_summary.csv` (success-criteria per direction) |

## Testing

Run the CLI unit tests (no model required):
```bash
python -m unittest src.tests.test_cli -v
```

This verifies:
- The STEPS dict is correctly defined with exactly 12 entries
- Each step maps to a callable function
- Importing `src.run` does not trigger model loading

## Module Structure

- `src/run.py` — Main CLI dispatcher with STEPS mapping
- `src/modules/` — Each step has a module with `run(cfg)` function:
  - `collect/`, `metadata/`, `extract/`, `normalize/`, `probes/`, `directions/`, `cross/`, `flores_decomp/`, `rep_similarity/`, `steering/`, `data_stats/`, `report/`
- `src/shared_utils/` — Common utilities (I/O, model loading, storage, math)
- `src/configs/config.yaml` — Default configuration
- `src/tests/` — Unit tests for all modules

## Troubleshooting

### CUDA out of memory
Reduce `batch_size` or `max_seq_len` in the config.

### SSD path not found
Verify `paths` in `config.yaml` point to mounted SSD volumes.

### Model download timeout
Transformers will cache models in `~/.cache/huggingface/`. For offline use, pre-download models.

### Import errors
Ensure `pip install -r src/requirements.txt` completes successfully and you're using the correct Python 3.9+ interpreter.

## References

- **Task 20:** CLI + README + smoke verification
- **Pipeline:** 12 steps from collection through final reporting
- **Models:** Qwen series (1.7B for local, 8B+ for GPU box)
- **Data:** Multilingual Wikipedia + parallel corpora + cultural riddle puzzles

## License

Part of the DFKI Culture Puzzles research project (2026).
