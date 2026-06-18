#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:?Set GPU_ID for this fold}"
CANDIDATE="${CANDIDATE:?Set CANDIDATE to H or F}"
FOLD="${FOLD:?Set FOLD from 0 through 4}"
FORCE="${FORCE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
PRINT_CONFIG_ONLY="${PRINT_CONFIG_ONLY:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SEED="${SEED:-42}"
MASK_TIME_PROB="${MASK_TIME_PROB:-0.01}"
MASK_TIME_LENGTH="${MASK_TIME_LENGTH:-5}"
MASK_TIME_MIN_MASKS="${MASK_TIME_MIN_MASKS:-1}"

if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
  echo "FOLD must be between 0 and 4" >&2
  exit 2
fi
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "FORCE must be 0 or 1" >&2
  exit 2
fi

case "$CANDIDATE" in
  H)
    candidate_name="two_stage_head_warmup"
    ;;
  F)
    candidate_name="lr2e-4_freeze_feature"
    ;;
  *)
    echo "Unknown CANDIDATE=$CANDIDATE; expected H or F" >&2
    exit 2
    ;;
esac
candidate_name="${candidate_name}${RUN_SUFFIX}"

train_shards=()
for shard_index in 0 1 2 3 4; do
  if [[ "$shard_index" != "$FOLD" ]]; then
    train_shards+=(
      "$PROJECT_ROOT/data/train/shard-$(printf '%06d' "$shard_index").tar"
    )
  fi
done
train_shard_spec="$(IFS=,; printf '%s' "${train_shards[*]}")"
eval_shard="$PROJECT_ROOT/data/train/shard-$(printf '%06d' "$FOLD").tar"

output_dir="$PROJECT_ROOT/outputs/base_strict_cv/$candidate_name/fold_$FOLD"
result_dir="$PROJECT_ROOT/results/base_strict_cv/$candidate_name/fold_$FOLD"
log_dir="$PROJECT_ROOT/logs/base_strict_cv/$candidate_name"
log_file="$log_dir/fold_$FOLD.log"
metadata_path="$result_dir/run_metadata.json"

print_configuration() {
  printf 'candidate=%s\n' "$candidate_name"
  printf 'fold=%s\n' "$FOLD"
  printf 'gpu=%s\n' "$GPU_ID"
  printf 'train_shards=%s\n' "$train_shard_spec"
  printf 'eval_shards=%s\n' "$eval_shard"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'result_dir=%s\n' "$result_dir"
  printf 'log_file=%s\n' "$log_file"
  printf 'model_source=facebook/wav2vec2-base\n'
  printf 'specaugment=on mask_time_prob=%s mask_time_length=%s min_masks=%s\n' \
    "$MASK_TIME_PROB" "$MASK_TIME_LENGTH" "$MASK_TIME_MIN_MASKS"
  if [[ "$CANDIDATE" == "H" ]]; then
    printf 'training=stage1_head_only_10_epochs,stage2_encoder_40_epochs\n'
  else
    printf 'training=feature_encoder_frozen_lr2e-4_50_epochs\n'
  fi
}

if [[ "$PRINT_CONFIG_ONLY" == "1" ]]; then
  print_configuration
  exit 0
fi

mkdir -p "$output_dir" "$result_dir" "$log_dir"

if [[ "$FORCE" != "1" ]] \
  && [[ -f "$output_dir/best_model/config.json" ]] \
  && [[ -f "$metadata_path" ]]; then
  echo "Skipping $candidate_name fold $FOLD: completed artifacts exist."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_training() {
  local model_source="$1"
  local stage_output_dir="$2"
  local stage_result_dir="$3"
  local epochs="$4"
  local eval_delay="$5"
  local early_stop_start="$6"
  local patience="$7"
  local load_best_flag="$8"
  shift 8
  local -a extra_args=("$@")
  local -a resume_args=()

  mkdir -p "$stage_output_dir" "$stage_result_dir"
  if [[ "$FORCE" != "1" ]] \
    && find "$stage_output_dir" -maxdepth 1 -type d -name 'checkpoint-*' \
      -print -quit | grep -q .; then
    resume_args=(--resume_from_checkpoint latest)
    printf '[%s] resuming_from_latest=%s\n' \
      "$(date -Is)" "$stage_output_dir" | tee -a "$log_file"
  fi

  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$PROJECT_ROOT/wav2vec_finetuning.py" \
    --experiment_role main \
    --model_name_or_path "$model_source" \
    --train_shards "$train_shard_spec" \
    --eval_shards "$eval_shard" \
    --output_dir "$stage_output_dir" \
    --final_model_subdir best_model \
    --num_train_epochs "$epochs" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --early_stopping_patience "$patience" \
    --early_stopping_threshold 0.0 \
    --early_stopping_start_epoch "$early_stop_start" \
    --eval_delay "$eval_delay" \
    --logging_steps 5 \
    --save_total_limit 3 \
    --loss_impl hf \
    --enable_spec_augment \
    --mask_time_prob "$MASK_TIME_PROB" \
    --mask_time_length "$MASK_TIME_LENGTH" \
    --mask_time_min_masks "$MASK_TIME_MIN_MASKS" \
    --ctc_zero_infinity \
    --no_attention_mask_for_loss \
    --finite_loss_check_samples 2 \
    --seed "$SEED" \
    --local_files_only \
    --training_log_csv "$stage_result_dir/training_log.csv" \
    --validation_history_csv "$stage_result_dir/validation_wer_history.csv" \
    --run_metadata_json "$stage_result_dir/run_metadata.json" \
    "$load_best_flag" \
    "${extra_args[@]}" \
    "${resume_args[@]}"
}

{
  printf '[%s] candidate=%s fold=%s gpu=%s\n' \
    "$(date -Is)" "$candidate_name" "$FOLD" "$GPU_ID"
  printf '[%s] train_shards=%s\n' "$(date -Is)" "$train_shard_spec"
  printf '[%s] eval_shards=%s\n' "$(date -Is)" "$eval_shard"
  printf '[%s] model_source=facebook/wav2vec2-base\n' "$(date -Is)"
  printf '[%s] specaugment=on mask_time_prob=%s mask_time_length=%s min_masks=%s\n' \
    "$(date -Is)" "$MASK_TIME_PROB" "$MASK_TIME_LENGTH" "$MASK_TIME_MIN_MASKS"
} | tee -a "$log_file"

if [[ "$CANDIDATE" == "H" ]]; then
  stage1_output="$output_dir/stage1"
  stage1_result="$result_dir/stage1"
  {
    if [[ "$FORCE" != "1" ]] \
      && [[ -f "$stage1_output/best_model/config.json" ]] \
      && [[ -f "$stage1_result/run_metadata.json" ]]; then
      printf '[%s] stage=1 already_complete; reusing=%s\n' \
        "$(date -Is)" "$stage1_output/best_model"
    else
      printf '[%s] stage=1 head-only epochs=10 head_lr=1e-3\n' "$(date -Is)"
      run_training \
        facebook/wav2vec2-base \
        "$stage1_output" \
        "$stage1_result" \
        10 \
        1 \
        20 \
        0 \
        --no_load_best_model_at_end \
        --learning_rate 1e-3 \
        --freeze_wav2vec2
    fi

    printf '[%s] stage=2 encoder_lr=1e-4 head_lr=1e-3 epochs=40\n' \
      "$(date -Is)"
    run_training \
      "$stage1_output/best_model" \
      "$output_dir" \
      "$result_dir" \
      40 \
      0 \
      10 \
      8 \
      --load_best_model_at_end \
      --learning_rate 1e-4 \
      --freeze_feature_encoder \
      --layerwise_lr_decay \
      --layerwise_lr_decay_rate 1.0 \
      --head_learning_rate 1e-3
  } 2>&1 | tee -a "$log_file"
else
  run_training \
    facebook/wav2vec2-base \
    "$output_dir" \
    "$result_dir" \
    50 \
    5 \
    20 \
    8 \
    --load_best_model_at_end \
    --learning_rate 2e-4 \
    --freeze_feature_encoder 2>&1 | tee -a "$log_file"
fi

printf '[%s] completed candidate=%s fold=%s gpu=%s\n' \
  "$(date -Is)" "$candidate_name" "$FOLD" "$GPU_ID" | tee -a "$log_file"
