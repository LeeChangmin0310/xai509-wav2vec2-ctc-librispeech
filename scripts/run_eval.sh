#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

experiment="${1:-baseline_lr1e-4}"
output_dir="$OUTPUTS_DIR/$experiment"
result_dir="$RESULTS_DIR/$experiment"
clean_result="$result_dir/test_clean_result.txt"
other_result="$result_dir/test_other_result.txt"

if [[ ! -f "$clean_result" ]] || [[ ! -f "$other_result" ]]; then
  echo "Missing inference results for $experiment. Run scripts/run_inference.sh first." >&2
  exit 1
fi

experiment_summary_args "$experiment"
checkpoint_args=()
if model_path="$(resolve_model_path "$output_dir")"; then
  checkpoint_args+=(--checkpoint_path "$model_path")
fi

"$PYTHON" "$PROJECT_ROOT/evaluate_wer.py" \
  --experiment_name "$experiment" \
  --summary_csv "$RESULTS_DIR/wer_summary.csv" \
  "${SUMMARY_ARGS[@]}" \
  "${checkpoint_args[@]}" \
  "$clean_result" \
  "$other_result" 2>&1 | tee "$LOGS_DIR/${experiment}_eval.log"
