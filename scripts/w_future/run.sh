#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
dataset="${1:-Sdwpfh1}"
length="${2:-short}"
exec "${PYTHON:-python}" run.py --mode w_future --dataset "$dataset" --length "$length" \
  --seeds 1 2 3 --evaluate-test "${@:3}"
