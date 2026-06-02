#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_training "layerwise_lr_decay" \
  --disable_spec_augment \
  --loss_impl hf \
  --ctc_zero_infinity \
  --learning_rate 5e-5 \
  --layerwise_lr_decay \
  --layerwise_lr_decay_rate 0.9 \
  --head_learning_rate 1e-4
