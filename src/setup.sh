#!/usr/bin/env bash
# Set up the venv for the src/ pipeline.
#   src/setup.sh            # create repo-root .venv (if missing) + install src/requirements.txt
#   PYTHON=python3.11 src/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src
ROOT="$(cd "$HERE/.." && pwd)"                          # repo root (src runs as `python -m src.run`)
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "Creating virtualenv at .venv (using $PYTHON) ..."
  "$PYTHON" -m venv .venv
else
  echo ".venv already exists — reusing it."
fi

echo "Upgrading pip ..."
.venv/bin/python -m pip install --quiet --upgrade pip

echo "Installing src/requirements.txt ..."
.venv/bin/python -m pip install -r src/requirements.txt

if [ ! -f env.sh ] && [ -f env.sh.example ]; then
  cp env.sh.example env.sh
  echo "Created env.sh from env.sh.example — fill in HF_TOKEN / OPENAI_API_KEY."
fi

echo ""
echo "Setup complete."
echo "  1. Edit env.sh (HF_TOKEN needed for FLORES/SIB-200 in the 'collect' step)."
echo "  2. Run the pipeline:  src/pipeline.sh        (or a subset: src/pipeline.sh analyze)"
