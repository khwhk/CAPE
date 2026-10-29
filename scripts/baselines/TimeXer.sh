#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/scripts/baselines/run.py" --model TimeXer --mode "${1:-w_future}" --dataset "${2:-Sdwpfh1}" --length "${3:-short}" "${@:4}"
