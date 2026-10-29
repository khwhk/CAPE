#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python}" run.py --mode wo_future --dataset Sdwpfh2 --length "${1:-short}" --seeds 1 2 3 --evaluate-test "${@:2}"
