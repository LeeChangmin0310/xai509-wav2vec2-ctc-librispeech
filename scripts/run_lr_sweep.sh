#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_training "lr1e-5" --learning_rate 1e-5
run_training "lr5e-5" --learning_rate 5e-5
