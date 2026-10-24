#!/usr/bin/env bash
# Run the test suite.
#
# Works from any directory - it does not depend on the current working
# directory, and it finds the venv on its own.
#
# Usage:
#   ./test.sh              # logic tests only, no GPU
#   ./test.sh --hardware   # also the hardware verification (loads a model)

set -euo pipefail

REPO="C:/Users/willow/Documents/Cpt_Mem0"
PY="$REPO/.venv/Scripts/python.exe"

if [ ! -f "$PY" ]; then
    echo "The project environment is missing: $PY" >&2
    echo "Create it with:" >&2
    echo "  cd $REPO && uv venv .venv && uv pip install --python .venv/Scripts/python.exe pytest" >&2
    exit 2
fi

exec "$PY" "$REPO/scripts/run_tests.py" "$@"
