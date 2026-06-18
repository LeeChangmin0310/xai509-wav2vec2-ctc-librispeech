#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:?Set GPU_ID}"
CANDIDATE="${CANDIDATE:?Set CANDIDATE F, G, H, I, or J}"
FORCE="${FORCE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SEED="${SEED:-42}"
MASK_TIME_PROB="${MASK_TIME_PROB:-0.01}"
MASK_TIME_LENGTH="${MASK_TIME_LENGTH:-5}"
MASK_TIME_MIN_MASKS="${MASK_TIME_MIN_MASKS:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
H_STAGE1_EPOCHS="${H_STAGE1_EPOCHS:-10}"
H_STAGE2_EPOCHS="${H_STAGE2_EPOCHS:-40}"

train_shards=(
  "$PROJECT_ROOT/data/train/shard-000001.tar"
  "$PROJECT_ROOT/data/train/shard-000002.tar"
  "$PROJECT_ROOT/data/train/shard-000003.tar"
  "$PROJECT_ROOT/data/train/shard-000004.tar"
)
train_shard_spec="$(IFS=,; printf '%s' "${train_shards[*]}")"
eval_shard="$PROJECT_ROOT/data/train/shard-000000.tar"

case "$CANDIDATE" in
  F)
    candidate_name="lr2e-4_freeze_feature"
    candidate_args=(--learning_rate 2e-4 --freeze_feature_encoder)
    ;;
  G)
    candidate_name="lr5e-4_freeze_feature"
    candidate_args=(--learning_rate 5e-4 --freeze_feature_encoder)
    ;;
  H)
    candidate_name="two_stage_head_warmup"
    candidate_args=()
    ;;
  I)
    candidate_name="encoder3e-5_head1e-3"
    candidate_args=(
      --learning_rate 3e-5
      --freeze_feature_encoder
      --layerwise_lr_decay
      --layerwise_lr_decay_rate 1.0
      --head_learning_rate 1e-3
    )
    ;;
  J)
    candidate_name="encoder1e-4_head1e-3_blankbias-2"
    candidate_args=(
      --learning_rate 1e-4
      --freeze_feature_encoder
      --layerwise_lr_decay
      --layerwise_lr_decay_rate 1.0
      --head_learning_rate 1e-3
      --blank_bias_init -2.0
    )
    ;;
  *)
    echo "Unknown CANDIDATE=$CANDIDATE; expected F, G, H, I, or J" >&2
    exit 2
    ;;
esac
candidate_name="${candidate_name}${RUN_SUFFIX}"

output_dir="$PROJECT_ROOT/outputs/base_strict_cv/$candidate_name/fold_0"
result_dir="$PROJECT_ROOT/results/base_strict_cv/$candidate_name/fold_0"
log_dir="$PROJECT_ROOT/logs/base_strict_cv/$candidate_name"
log_file="$log_dir/fold_0.log"
metadata_path="$result_dir/run_metadata.json"
mkdir -p "$output_dir" "$result_dir" "$log_dir"

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

  mkdir -p "$stage_output_dir" "$stage_result_dir"
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
    "${extra_args[@]}"
}

if [[ "$FORCE" != "1" ]] && [[ -f "$output_dir/best_model/config.json" ]] \
  && [[ -f "$metadata_path" ]]; then
  echo "Skipping $candidate_name fold 0: completed artifacts exist."
  exit 0
fi

{
  printf '[%s] candidate=%s fold=0 gpu=%s\n' \
    "$(date -Is)" "$candidate_name" "$GPU_ID"
  printf '[%s] train_shards=%s\n' "$(date -Is)" "$train_shard_spec"
  printf '[%s] eval_shards=%s\n' "$(date -Is)" "$eval_shard"
  printf '[%s] specaugment=on mask_time_prob=%s mask_time_length=%s min_masks=%s\n' \
    "$(date -Is)" "$MASK_TIME_PROB" "$MASK_TIME_LENGTH" "$MASK_TIME_MIN_MASKS"
} | tee -a "$log_file"

if [[ "$CANDIDATE" == "H" ]]; then
  stage1_output="$output_dir/stage1"
  stage1_result="$result_dir/stage1"
  {
    printf '[%s] stage=1 head-only epochs=%s head_lr=1e-3\n' \
      "$(date -Is)" "$H_STAGE1_EPOCHS"
    run_training \
      facebook/wav2vec2-base \
      "$stage1_output" \
      "$stage1_result" \
      "$H_STAGE1_EPOCHS" \
      1 \
      20 \
      0 \
      --no_load_best_model_at_end \
      --learning_rate 1e-3 \
      --freeze_wav2vec2

    printf '[%s] stage=2 encoder_lr=1e-4 head_lr=1e-3 epochs=%s\n' \
      "$(date -Is)" "$H_STAGE2_EPOCHS"
    run_training \
      "$stage1_output/best_model" \
      "$output_dir" \
      "$result_dir" \
      "$H_STAGE2_EPOCHS" \
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
    "$MAX_EPOCHS" \
    5 \
    20 \
    8 \
    --load_best_model_at_end \
    "${candidate_args[@]}" 2>&1 | tee -a "$log_file"
fi

printf '[%s] completed candidate=%s fold=0 gpu=%s\n' \
  "$(date -Is)" "$candidate_name" "$GPU_ID" | tee -a "$log_file"
