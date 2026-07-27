#!/usr/bin/env bash
set -euo pipefail
MAC="${HC04_MAC:-04:25:01:09:0A:EB}"
CHANNEL="${HC04_CHANNEL:-1}"
DEVICE="${HC04_DEVICE:-0}"

if ! command -v rfcomm >/dev/null 2>&1; then
  echo "rfcomm is missing. Install bluez first: sudo apt install bluez"
  exit 1
fi

sudo rfcomm release "${DEVICE}" >/dev/null 2>&1 || true
sudo rfcomm bind "${DEVICE}" "${MAC}" "${CHANNEL}"
echo "HC-04 bound: /dev/rfcomm${DEVICE} -> ${MAC} channel ${CHANNEL}"
ls -l "/dev/rfcomm${DEVICE}"
