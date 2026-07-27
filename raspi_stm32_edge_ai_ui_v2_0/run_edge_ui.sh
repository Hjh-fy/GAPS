#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Keep the launcher portable across acquisition-only and AI-enabled installs.
# Examples:
#   EDGE_UI_PYTHON="$HOME/GAPS/gaps_rpi_env/bin/python" \
#   EDGE_UI_EXTRA_PYTHONPATH="/usr/lib/python3/dist-packages" \
#   ./run_edge_ui.sh --ai-package /path/to/package
PYTHON_BIN="${EDGE_UI_PYTHON:-python3}"
if [[ -n "${EDGE_UI_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EDGE_UI_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || {
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 1
  }
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter was not found: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" edge_ui_app.py "$@"
