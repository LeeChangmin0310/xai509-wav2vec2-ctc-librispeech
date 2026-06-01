#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
FP16="${FP16:-1}"
FORCE="${FORCE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export FP16 FORCE
source "$SCRIPT_DIR/common.sh"

require_data

failed_file="$LOGS_DIR/failed_experiments.txt"
: > "$failed_file"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_stage() {
  local experiment="$1"
  shift
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOGS_DIR/$experiment.log"
}

run_stage() {
  local experiment="$1"
  local stage="$2"
  shift 2

  log_stage "$experiment" "START $stage"
  if "$@"; then
    log_stage "$experiment" "DONE  $stage"
    return 0
  else
    local status=$?
    log_stage "$experiment" "FAIL  $stage (exit $status)"
    return "$status"
  fi
}

training_args_for() {
  case "$1" in
    baseline_lr1e-4)
      TRAINING_ARGS=(--learning_rate 1e-4)
      ;;
    lr1e-5)
      TRAINING_ARGS=(--learning_rate 1e-5)
      ;;
    lr5e-5)
      TRAINING_ARGS=(--learning_rate 5e-5)
      ;;
    freeze_feature_lr1e-4)
      TRAINING_ARGS=(--learning_rate 1e-4 --freeze_feature_encoder)
      ;;
    freeze3_lr1e-4)
      TRAINING_ARGS=(--learning_rate 1e-4 --freeze_n_layers 3)
      ;;
    freeze6_lr1e-4)
      TRAINING_ARGS=(--learning_rate 1e-4 --freeze_n_layers 6)
      ;;
    *)
      echo "Unknown experiment: $1" >&2
      return 1
      ;;
  esac
}

run_inference_logged() {
  local experiment="$1"
  bash "$SCRIPT_DIR/run_inference.sh" "$experiment" 2>&1 |
    tee -a "$LOGS_DIR/$experiment.log"
}

run_eval_logged() {
  local experiment="$1"
  bash "$SCRIPT_DIR/run_eval.sh" "$experiment" 2>&1 |
    tee -a "$LOGS_DIR/$experiment.log"
}

run_experiment() {
  local experiment="$1"
  training_args_for "$experiment" || return $?
  run_stage "$experiment" training \
    run_training "$experiment" "${TRAINING_ARGS[@]}" || return $?
  run_stage "$experiment" inference \
    run_inference_logged "$experiment" || return $?
  run_stage "$experiment" wer_evaluation \
    run_eval_logged "$experiment" || return $?
}

experiments=(
  baseline_lr1e-4
  lr1e-5
  lr5e-5
  freeze_feature_lr1e-4
  freeze3_lr1e-4
  freeze6_lr1e-4
)

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for experiment in "${experiments[@]}"; do
  log_stage "$experiment" "BEGIN experiment"
  if run_experiment "$experiment"; then
    log_stage "$experiment" "END   experiment"
  else
    printf '%s\n' "$experiment" | tee -a "$failed_file"
    log_stage "$experiment" "FAILED experiment"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping queue after failure. Set CONTINUE_ON_ERROR=1 to continue." >&2
      exit 1
    fi
  fi
done

if [[ -f "$RESULTS_DIR/wer_summary.csv" ]]; then
  "$PYTHON" "$SCRIPT_DIR/summarize_results.py"
else
  echo "WER summary was not created; skipping result summarization."
fi

if [[ -s "$failed_file" ]]; then
  echo "Queue completed with failures listed in $failed_file"
  exit 1
fi
echo "Queue completed successfully."
