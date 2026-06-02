#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_ROOT"
"$PYTHON" scripts/probe_ctc_loss.py \
  --model_name_or_path facebook/wav2vec2-base-960h \
  --num_samples 2 \
  "$@"
