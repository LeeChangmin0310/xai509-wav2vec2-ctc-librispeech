#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

experiment="${1:-baseline_lr1e-4}"
output_dir="$OUTPUTS_DIR/$experiment"
result_dir="$RESULTS_DIR/$experiment"
clean_result="$result_dir/test_clean_result.txt"
other_result="$result_dir/test_other_result.txt"

if [[ "$FORCE" != "1" ]] && [[ -f "$clean_result" ]] && [[ -f "$other_result" ]]; then
  echo "Skipping $experiment inference: result files already exist. Set FORCE=1 to rerun."
  exit 0
fi

if ! model_path="$(resolve_model_path "$output_dir")"; then
  echo "No trained checkpoint found for $experiment in $output_dir." >&2
  exit 1
fi

mkdir -p "$result_dir"
"$PYTHON" "$PROJECT_ROOT/wav2vec_inference.py" \
  --test_clean_shards "$TEST_CLEAN_SHARDS" \
  --test_other_shards "$TEST_OTHER_SHARDS" \
  --model_name_or_path "$model_path" \
  --output_dir "$result_dir" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --seed "$SEED" \
  "${FP16_ARGS[@]}" 2>&1 | tee "$LOGS_DIR/${experiment}_inference.log"
