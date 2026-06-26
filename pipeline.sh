#!/usr/bin/env bash
# Run pipeline stages with the project's .venv.
#
#   ./pipeline.sh                       # run: puzzles parallel topics vectors
#   ./pipeline.sh puzzles               # run a single stage
#   ./pipeline.sh topics vectors        # run a subset, in the given order
#   ./pipeline.sh collect               # data collection only (puzzles parallel topics)
#   ./pipeline.sh all                   # run everything incl. analyze + plots
#   ./pipeline.sh --config configs/riddles_config_1.7b.yaml vectors   # alt config
#
# Stage names: puzzles parallel topics vectors analyze plots embed-analysis  (or 01..07)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "No .venv found. Run ./setup.sh first." >&2
  exit 1
fi
PY="$ROOT/.venv/bin/python"
CONFIG="configs/riddles_config.yaml"

# Load env (HF_TOKEN for FLORES, OPENAI_API_KEY for judge steps) if present.
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/env.sh"
fi

script_for() {
  case "$1" in
    puzzles|01)  echo "01_collect_puzzles.py" ;;
    parallel|02) echo "02_collect_parallel.py" ;;
    topics|03)   echo "03_collect_topics.py" ;;
    vectors|04)  echo "04_compute_vectors.py" ;;
    analyze|05)  echo "05_analyze_vectors.py" ;;
    plots|06)    echo "06_generate_plots.py" ;;
    embed-analysis|07) echo "07_analyze_embeddings.py" ;;
    *)           echo "" ;;
  esac
}

# Optional config override (must precede stage names):
#   ./pipeline.sh --config configs/riddles_config_1.7b.yaml vectors
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config=*) CONFIG="${1#*=}"; shift ;;
    --config|-c) CONFIG="$2"; shift 2 ;;
    *) break ;;
  esac
done

# Determine stages to run.
if [ "$#" -eq 0 ]; then
  STAGES="puzzles parallel topics vectors"
elif [ "$1" = "all" ]; then
  STAGES="puzzles parallel topics vectors analyze plots"
elif [ "$1" = "collect" ]; then
  STAGES="puzzles parallel topics"          # data collection only (no vectors)
else
  STAGES="$*"
fi

# Validate before running anything.
for s in $STAGES; do
  if [ -z "$(script_for "$s")" ]; then
    echo "Unknown stage: '$s'" >&2
    echo "Valid: puzzles parallel topics vectors analyze plots embed-analysis (or 01..07)" >&2
    exit 2
  fi
done

cd "$ROOT/scripts"
echo "Config: $CONFIG"
for s in $STAGES; do
  script="$(script_for "$s")"
  echo ""
  echo "=================================================="
  echo "  Stage: $s  ->  $script"
  echo "=================================================="
  "$PY" "$script" --config "$CONFIG"
done

echo ""
echo "Pipeline finished: $STAGES"
