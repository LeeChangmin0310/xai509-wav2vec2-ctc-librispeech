#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
FP16="${FP16:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
source "$SCRIPT_DIR/common.sh"

require_data

source_experiment="${BEST_EXPERIMENT:-baseline_lr1e-4}"
beam_width="${BEAM_WIDTH:-100}"
result_experiment="${source_experiment}_beam"
output_dir="$OUTPUTS_DIR/$source_experiment"
result_dir="$RESULTS_DIR/$result_experiment"
log_file="$LOGS_DIR/$result_experiment.log"
clean_result="$result_dir/test_clean_result.txt"
other_result="$result_dir/test_other_result.txt"

if ! model_path="$(resolve_model_path "$output_dir")"; then
  echo "No trained checkpoint found for $source_experiment in $output_dir." >&2
  exit 1
fi

mkdir -p "$result_dir"
if [[ "$FORCE" != "1" ]] && [[ -f "$clean_result" ]] && [[ -f "$other_result" ]]; then
  echo "Skipping beam inference: result files already exist. Set FORCE=1 to rerun."
else
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "$PYTHON" "$PROJECT_ROOT/wav2vec_inference.py" \
    --test_clean_shards "$TEST_CLEAN_SHARDS" \
    --test_other_shards "$TEST_OTHER_SHARDS" \
    --model_name_or_path "$model_path" \
    --output_dir "$result_dir" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --decoding_method beam \
    --beam_width "$beam_width" \
    --seed "$SEED" \
    "${FP16_ARGS[@]}" 2>&1 | tee -a "$log_file"
fi

experiment_summary_args "$source_experiment"
"$PYTHON" "$PROJECT_ROOT/evaluate_wer.py" \
  --experiment_name "$result_experiment" \
  --summary_csv "$RESULTS_DIR/wer_summary.csv" \
  "${SUMMARY_ARGS[@]}" \
  --decoding_method beam \
  --beam_width "$beam_width" \
  --checkpoint_path "$model_path" \
  "$clean_result" \
  "$other_result" 2>&1 | tee -a "$log_file"

"$PYTHON" "$SCRIPT_DIR/summarize_results.py"
