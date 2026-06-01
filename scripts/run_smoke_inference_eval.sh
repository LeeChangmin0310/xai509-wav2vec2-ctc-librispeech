#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
FP16="${FP16:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
source "$SCRIPT_DIR/common.sh"

require_data

output_dir="$OUTPUTS_DIR/smoke"
result_dir="$RESULTS_DIR/smoke"
log_file="$LOGS_DIR/smoke_inference_eval.log"
clean_result="$result_dir/test_clean_result.txt"
other_result="$result_dir/test_other_result.txt"
dry_run_args=()

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  model_path="$MODEL_NAME_OR_PATH"
  dry_run_args+=(--dry_run)
elif ! model_path="$(resolve_model_path "$output_dir")"; then
  echo "No smoke checkpoint found. Run scripts/run_smoke_train.sh first." >&2
  exit 1
fi

mkdir -p "$result_dir"
if [[ "${DRY_RUN:-0}" != "1" ]] && [[ "$FORCE" != "1" ]] && \
    [[ -f "$clean_result" ]] && [[ -f "$other_result" ]]; then
  echo "Skipping smoke inference: result files already exist. Set FORCE=1 to rerun."
else
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "$PYTHON" "$PROJECT_ROOT/wav2vec_inference.py" \
    --test_clean_shards "$TEST_CLEAN_SHARDS" \
    --test_other_shards "$TEST_OTHER_SHARDS" \
    --model_name_or_path "$model_path" \
    --output_dir "$result_dir" \
    --per_device_eval_batch_size "${SMOKE_EVAL_BATCH_SIZE:-1}" \
    --max_test_samples "${SMOKE_TEST_SAMPLES:-4}" \
    --seed "$SEED" \
    "${FP16_ARGS[@]}" \
    "${dry_run_args[@]}" 2>&1 | tee -a "$log_file"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/evaluate_wer.py" \
    --experiment_name smoke \
    --summary_csv "$result_dir/wer_summary.csv" \
    "$clean_result" \
    "$other_result" 2>&1 | tee -a "$log_file"
fi
