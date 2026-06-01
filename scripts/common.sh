#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$PROJECT_ROOT/outputs}"
LOGS_DIR="${LOGS_DIR:-$PROJECT_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"

TRAIN_SHARDS="${TRAIN_SHARDS:-$DATA_DIR/train}"
TEST_CLEAN_SHARDS="${TEST_CLEAN_SHARDS:-$DATA_DIR/test-clean}"
TEST_OTHER_SHARDS="${TEST_OTHER_SHARDS:-$DATA_DIR/test-other}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-facebook/wav2vec2-base}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SEED="${SEED:-42}"
FP16="${FP16:-0}"
FORCE="${FORCE:-0}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR" "$RESULTS_DIR"

FP16_ARGS=()
if [[ "$FP16" == "1" ]]; then
  FP16_ARGS+=(--fp16)
fi

has_checkpoint() {
  local output_dir="$1"
  [[ -d "$output_dir" ]] || return 1
  [[ -f "$output_dir/final_model/config.json" ]] || \
    find "$output_dir" -maxdepth 2 -type f \
      -path '*/checkpoint-*/config.json' -print -quit | grep -q .
}

resolve_model_path() {
  local output_dir="$1"
  local checkpoint
  [[ -d "$output_dir" ]] || return 1
  if [[ -f "$output_dir/final_model/config.json" ]]; then
    printf '%s\n' "$output_dir/final_model"
    return 0
  fi
  checkpoint="$(
    find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print |
      sort -V |
      tail -n 1
  )"
  [[ -n "$checkpoint" ]] || return 1
  printf '%s\n' "$checkpoint"
}

require_data() {
  bash "$SCRIPT_DIR/check_data.sh"
}

run_training() {
  local experiment="$1"
  shift
  local output_dir="$OUTPUTS_DIR/$experiment"
  mkdir -p "$output_dir"

  if [[ "$FORCE" != "1" ]] && has_checkpoint "$output_dir"; then
    echo "Skipping $experiment: checkpoint already exists. Set FORCE=1 to rerun."
    return 0
  fi

  "$PYTHON" "$PROJECT_ROOT/wav2vec_finetuning.py" \
    --train_shards "$TRAIN_SHARDS" \
    --test_clean_shards "$TEST_CLEAN_SHARDS" \
    --test_other_shards "$TEST_OTHER_SHARDS" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$output_dir" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --seed "$SEED" \
    "${FP16_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$LOGS_DIR/$experiment.log"
}
