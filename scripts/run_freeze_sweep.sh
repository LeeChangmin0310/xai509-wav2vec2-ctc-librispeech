#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_training "freeze_feature_lr1e-4" \
  --learning_rate 1e-4 \
  --freeze_feature_encoder
run_training "freeze3_lr1e-4" \
  --learning_rate 1e-4 \
  --freeze_n_layers 3
run_training "freeze6_lr1e-4" \
  --learning_rate 1e-4 \
  --freeze_n_layers 6
