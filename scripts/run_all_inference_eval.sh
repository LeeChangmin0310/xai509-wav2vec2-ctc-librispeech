#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiments=(
  baseline_lr1e-4
  lr1e-5
  lr5e-5
  freeze_feature_lr1e-4
  freeze3_lr1e-4
  freeze6_lr1e-4
)

for experiment in "${experiments[@]}"; do
  bash "$SCRIPT_DIR/run_inference.sh" "$experiment"
  bash "$SCRIPT_DIR/run_eval.sh" "$experiment"
done
