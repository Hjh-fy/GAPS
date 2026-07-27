#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${HC04_PORT:-/dev/rfcomm0}"
if [[ ! -e "$PORT" ]]; then
  echo "Bluetooth serial device $PORT does not exist."
  echo "Run ./bind_hc04.sh first, or pass --port /dev/ttyUSB0 for USB serial."
  exit 1
fi
exec ./run_edge_ui.sh --port "$PORT" --fullscreen "$@"
