#!/usr/bin/env bash
# Run src/ pipeline steps via the Python CLI (python -m src.run), using the repo-root .venv.
#
#   src/pipeline.sh                          # full pipeline, in order
#   src/pipeline.sh collect                  # a single step
#   src/pipeline.sh data                     # group: collect metadata extract
#   src/pipeline.sh analyze                  # group: normalize..report (no collect/extract)
#   src/pipeline.sh probes directions        # a subset, in the given order
#   src/pipeline.sh --config src/configs/smoke.yaml extract   # alternate config
#
# Steps: collect metadata extract normalize probes directions cross
#        flores-decomp rep-similarity steering data-stats report
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src
ROOT="$(cd "$HERE/.." && pwd)"                          # repo root

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "No .venv found. Run src/setup.sh first." >&2
  exit 1
fi
PY="$ROOT/.venv/bin/python"
CONFIG="src/configs/config.yaml"

# Load env (HF_TOKEN for FLORES/SIB-200, OPENAI_API_KEY for judges) if present.
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/env.sh"
fi

ALL="collect metadata extract normalize probes directions cross flores-decomp rep-similarity steering data-stats report"
DATA="collect metadata extract"
ANALYZE="normalize probes directions cross flores-decomp rep-similarity steering data-stats report"

# Optional --config override (must precede step names).
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config=*) CONFIG="${1#*=}"; shift ;;
    --config|-c) CONFIG="$2"; shift 2 ;;
    *) break ;;
  esac
done

if [ "$#" -eq 0 ]; then
  STAGES="$ALL"
elif [ "$1" = "all" ]; then
  STAGES="$ALL"
elif [ "$1" = "data" ]; then
  STAGES="$DATA"
elif [ "$1" = "analyze" ]; then
  STAGES="$ANALYZE"
else
  STAGES="$*"
fi

cd "$ROOT"
echo "Config: $CONFIG"
for s in $STAGES; do
  echo ""
  echo "=================================================="
  echo "  Step: $s"
  echo "=================================================="
  "$PY" -m src.run "$s" --config "$CONFIG"   # the CLI rejects unknown step names
done

echo ""
echo "Pipeline finished: $STAGES"
