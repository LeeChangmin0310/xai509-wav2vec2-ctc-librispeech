#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
FORCE="${FORCE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
INFERENCE_FP16="${INFERENCE_FP16:-1}"
FP16=0
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-facebook/wav2vec2-base-960h}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export FORCE FP16 MODEL_NAME_OR_PATH
source "$SCRIPT_DIR/common.sh"

require_data

summary_csv="$RESULTS_DIR/wer_summary_asrinit.csv"
failed_file="$LOGS_DIR/failed_asrinit_experiments.txt"
touch "$failed_file"
failure_count=0

inference_fp16_args=()
if [[ "$INFERENCE_FP16" == "1" ]]; then
  inference_fp16_args+=(--fp16)
fi

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
    asrinit_lr1e-6_fp32)
      TRAINING_ARGS=(--learning_rate 1e-6)
      ;;
    asrinit_lr3e-6_fp32)
      TRAINING_ARGS=(--learning_rate 3e-6)
      ;;
    asrinit_lr1e-5_fp32_fixed)
      TRAINING_ARGS=(--learning_rate 1e-5)
      ;;
    asrinit_freeze_feature_lr3e-6_fp32)
      TRAINING_ARGS=(--learning_rate 3e-6 --freeze_feature_encoder)
      ;;
    asrinit_freeze3_lr3e-6_fp32)
      TRAINING_ARGS=(--learning_rate 3e-6 --freeze_n_layers 3)
      ;;
    asrinit_layerwise_lr_decay_fixed)
      TRAINING_ARGS=(--learning_rate 3e-6 --layerwise_lr_decay --layerwise_lr_decay_rate 0.9 --head_learning_rate 1e-5)
      ;;
    *)
      echo "Unknown ASR-init training experiment: $1" >&2
      return 1
      ;;
  esac
}

summary_args_for() {
  case "$1" in
    asr_pretrained_960h_full)
      SUMMARY_ARGS=(--train_setting pretrained_asr_control --freeze_setting none --layerwise_lr_decay false)
      ;;
    asrinit_lr1e-6_fp32)
      SUMMARY_ARGS=(--train_setting asr_initialized_finetuning --learning_rate 1e-6 --freeze_setting none --layerwise_lr_decay false)
      ;;
    asrinit_lr3e-6_fp32)
      SUMMARY_ARGS=(--train_setting asr_initialized_finetuning --learning_rate 3e-6 --freeze_setting none --layerwise_lr_decay false)
      ;;
    asrinit_lr1e-5_fp32_fixed)
      SUMMARY_ARGS=(--train_setting asr_initialized_finetuning --learning_rate 1e-5 --freeze_setting none --layerwise_lr_decay false)
      ;;
    asrinit_freeze_feature_lr3e-6_fp32)
      SUMMARY_ARGS=(--train_setting asr_initialized_finetuning --learning_rate 3e-6 --freeze_setting feature_encoder --layerwise_lr_decay false)
      ;;
    asrinit_freeze3_lr3e-6_fp32)
      SUMMARY_ARGS=(--train_setting asr_initialized_finetuning --learning_rate 3e-6 --freeze_setting first_3_encoder_layers --layerwise_lr_decay false)
      ;;
    asrinit_layerwise_lr_decay_fixed)
      SUMMARY_ARGS=(--train_setting asr_initialized_finetuning --learning_rate 3e-6 --freeze_setting feature_extractor --layerwise_lr_decay 0.9)
      ;;
    *)
      echo "Unknown ASR-init summary experiment: $1" >&2
      return 1
      ;;
  esac
}

resolve_experiment_model_path() {
  local experiment="$1"
  if [[ "$experiment" == "asr_pretrained_960h_full" ]]; then
    printf '%s\n' "$MODEL_NAME_OR_PATH"
  else
    resolve_model_path "$OUTPUTS_DIR/$experiment"
  fi
}

run_asrinit_training() {
  local experiment="$1"
  training_args_for "$experiment"
  run_training "$experiment" \
    --disable_spec_augment \
    --loss_impl hf \
    --ctc_zero_infinity \
    --no_attention_mask_for_loss \
    "${TRAINING_ARGS[@]}"
}

run_asrinit_inference() {
  local experiment="$1"
  local model_path result_dir clean_result other_result
  result_dir="$RESULTS_DIR/$experiment"
  clean_result="$result_dir/test_clean_result.txt"
  other_result="$result_dir/test_other_result.txt"

  if [[ "$FORCE" != "1" ]] && [[ -f "$clean_result" ]] && [[ -f "$other_result" ]]; then
    echo "Skipping $experiment inference: result files already exist. Set FORCE=1 to rerun."
    return 0
  fi
  if ! model_path="$(resolve_experiment_model_path "$experiment")"; then
    echo "No checkpoint found for $experiment." >&2
    return 1
  fi

  mkdir -p "$result_dir"
  "$PYTHON" "$PROJECT_ROOT/wav2vec_inference.py" \
    --test_clean_shards "$TEST_CLEAN_SHARDS" \
    --test_other_shards "$TEST_OTHER_SHARDS" \
    --model_name_or_path "$model_path" \
    --output_dir "$result_dir" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --seed "$SEED" \
    "${inference_fp16_args[@]}" 2>&1 | tee -a "$LOGS_DIR/$experiment.log"
}

summary_has_experiment() {
  local experiment="$1"
  [[ -f "$summary_csv" ]] && grep -q "^${experiment}," "$summary_csv"
}

run_asrinit_eval() {
  local experiment="$1"
  local model_path result_dir clean_result other_result
  result_dir="$RESULTS_DIR/$experiment"
  clean_result="$result_dir/test_clean_result.txt"
  other_result="$result_dir/test_other_result.txt"

  if [[ ! -f "$clean_result" ]] || [[ ! -f "$other_result" ]]; then
    echo "Missing inference results for $experiment." >&2
    return 1
  fi
  if [[ "$FORCE" != "1" ]] && summary_has_experiment "$experiment"; then
    echo "Skipping $experiment evaluation: summary row already exists. Set FORCE=1 to rerun."
    return 0
  fi
  if ! model_path="$(resolve_experiment_model_path "$experiment")"; then
    echo "No checkpoint found for $experiment." >&2
    return 1
  fi
  summary_args_for "$experiment"

  "$PYTHON" "$PROJECT_ROOT/evaluate_wer.py" \
    --experiment_name "$experiment" \
    --summary_csv "$summary_csv" \
    --decoding_method greedy \
    --checkpoint_path "$model_path" \
    "${SUMMARY_ARGS[@]}" \
    "$clean_result" \
    "$other_result" 2>&1 | tee -a "$LOGS_DIR/$experiment.log"
}

run_experiment() {
  local experiment="$1"
  if [[ "$experiment" == "asr_pretrained_960h_full" ]]; then
    log_stage "$experiment" "SKIP  training (pretrained ASR inference control)"
  else
    run_stage "$experiment" training run_asrinit_training "$experiment" || return $?
  fi
  run_stage "$experiment" inference run_asrinit_inference "$experiment" || return $?
  run_stage "$experiment" wer_evaluation run_asrinit_eval "$experiment" || return $?
}

experiments=(
  asr_pretrained_960h_full
  asrinit_lr1e-6_fp32
  asrinit_lr3e-6_fp32
  asrinit_lr1e-5_fp32_fixed
  asrinit_freeze_feature_lr3e-6_fp32
  asrinit_freeze3_lr3e-6_fp32
  asrinit_layerwise_lr_decay_fixed
)

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Training fp16=$FP16; inference fp16=$INFERENCE_FP16"
for experiment in "${experiments[@]}"; do
  log_stage "$experiment" "BEGIN experiment"
  if run_experiment "$experiment"; then
    log_stage "$experiment" "END   experiment"
  else
    printf '[%s] %s\n' "$(timestamp)" "$experiment" | tee -a "$failed_file"
    failure_count=$((failure_count + 1))
    log_stage "$experiment" "FAILED experiment"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping queue after failure. Set CONTINUE_ON_ERROR=1 to continue." >&2
      exit 1
    fi
  fi
done

if [[ -f "$summary_csv" ]]; then
  "$PYTHON" "$SCRIPT_DIR/summarize_results.py" --input_csv "$summary_csv"
else
  echo "ASR-init WER summary was not created; skipping result summarization."
fi

echo "ASR-init queue finished. Review $failed_file for any recorded failures."
if (( failure_count > 0 )); then
  exit 1
fi
