#!/usr/bin/env bash
# Create the .venv and install dependencies.
#   ./setup.sh            # create .venv (if missing) and install requirements.txt
#   PYTHON=python3.11 ./setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

echo "Installing requirements.txt ..."
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f env.sh ] && [ -f env.sh.example ]; then
  cp env.sh.example env.sh
  echo "Created env.sh from env.sh.example — fill in HF_TOKEN / OPENAI_API_KEY."
fi

echo ""
echo "Setup complete."
echo "  1. Edit env.sh (HF_TOKEN is needed for FLORES in stage 'parallel')."
echo "  2. Run the pipeline:  ./pipeline.sh"
