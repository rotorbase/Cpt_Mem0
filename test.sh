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

# The directory of this script, following symlinks. The project path is
# derived from it, so the script works in any checkout and on any
# machine.
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO="$(cd -P "$(dirname "$SOURCE")" && pwd)"

# On Git Bash for Windows the path comes back as /c/Users/..., which the
# native Python cannot open. The conversion turns it into C:/Users/...
case "$REPO" in
    /[A-Za-z]/*)
        REPO="$(printf '%s' "$REPO" | sed -E 's|^/([A-Za-z])/(.*)|\1:/\2|')"
        ;;
esac

# The venv: the Windows layout or the POSIX one.
if [ -f "$REPO/.venv/Scripts/python.exe" ]; then
    PY="$REPO/.venv/Scripts/python.exe"
elif [ -f "$REPO/.venv/bin/python" ]; then
    PY="$REPO/.venv/bin/python"
else
    echo "The project environment is missing: $REPO/.venv" >&2
    echo "Create it with:" >&2
    echo "  cd \"$REPO\" && uv venv .venv && uv pip install --python .venv/Scripts/python.exe pytest" >&2
    exit 2
fi

exec "$PY" "$REPO/scripts/run_tests.py" "$@"
