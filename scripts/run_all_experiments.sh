#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_baseline.sh"
bash "$SCRIPT_DIR/run_lr_sweep.sh"
bash "$SCRIPT_DIR/run_freeze_sweep.sh"
