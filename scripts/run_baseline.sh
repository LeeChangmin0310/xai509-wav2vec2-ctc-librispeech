#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_training "baseline_lr1e-4" \
  --disable_spec_augment \
  --loss_impl hf \
  --ctc_zero_infinity \
  --learning_rate 1e-4
