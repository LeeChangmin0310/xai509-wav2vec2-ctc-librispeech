#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:?Set GPU_ID for this fold}"
CANDIDATE="${CANDIDATE:?Set CANDIDATE A, B, C, D, or E}"
FOLD="${FOLD:?Set FOLD from 0 through 4}"
FORCE="${FORCE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SEED="${SEED:-42}"

if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
  echo "FOLD must be between 0 and 4" >&2
  exit 2
fi

case "$CANDIDATE" in
  A)
    candidate_name="lr3e-5_freeze_feature"
    candidate_args=(--learning_rate 3e-5 --freeze_feature_encoder)
    ;;
  B)
    candidate_name="lr5e-5_freeze_feature"
    candidate_args=(--learning_rate 5e-5 --freeze_feature_encoder)
    ;;
  C)
    candidate_name="lr1e-4_freeze_feature"
    candidate_args=(--learning_rate 1e-4 --freeze_feature_encoder)
    ;;
  D)
    candidate_name="lr3e-5_full"
    candidate_args=(--learning_rate 3e-5)
    ;;
  E)
    candidate_name="lr5e-5_full"
    candidate_args=(--learning_rate 5e-5)
    ;;
  *)
    echo "Unknown CANDIDATE=$CANDIDATE; expected A, B, C, D, or E" >&2
    exit 2
    ;;
esac

train_shards=()
for shard_index in 0 1 2 3 4; do
  if [[ "$shard_index" != "$FOLD" ]]; then
    train_shards+=("$PROJECT_ROOT/data/train/shard-$(printf '%06d' "$shard_index").tar")
  fi
done
train_shard_spec="$(IFS=,; printf '%s' "${train_shards[*]}")"
eval_shard="$PROJECT_ROOT/data/train/shard-$(printf '%06d' "$FOLD").tar"
output_dir="$PROJECT_ROOT/outputs/base_strict_cv/$candidate_name/fold_$FOLD"
log_dir="$PROJECT_ROOT/logs/base_strict_cv/$candidate_name"
log_file="$log_dir/fold_$FOLD.log"
result_dir="$PROJECT_ROOT/results/base_strict_cv/$candidate_name/fold_$FOLD"
metadata_path="$result_dir/run_metadata.json"

mkdir -p "$output_dir" "$log_dir" "$result_dir"
if [[ "$FORCE" != "1" ]] && [[ -f "$output_dir/best_model/config.json" ]] && [[ -f "$metadata_path" ]]; then
  echo "Skipping $candidate_name fold $FOLD: completed artifacts exist."
  exit 0
fi

resume_args=()
if [[ "$FORCE" != "1" ]] && find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  resume_args=(--resume_from_checkpoint latest)
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

{
  printf '[%s] candidate=%s fold=%s gpu=%s\n' "$(date -Is)" "$candidate_name" "$FOLD" "$GPU_ID"
  printf '[%s] train_shards=%s\n' "$(date -Is)" "$train_shard_spec"
  printf '[%s] eval_shards=%s\n' "$(date -Is)" "$eval_shard"
} | tee -a "$log_file"

conda run --no-capture-output -n "$CONDA_ENV" python "$PROJECT_ROOT/wav2vec_finetuning.py" \
  --experiment_role main \
  --model_name_or_path facebook/wav2vec2-base \
  --train_shards "$train_shard_spec" \
  --eval_shards "$eval_shard" \
  --output_dir "$output_dir" \
  --final_model_subdir best_model \
  --num_train_epochs "$MAX_EPOCHS" \
  --max_train_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --warmup_ratio 0.1 \
  --max_grad_norm 1.0 \
  --early_stopping_patience 3 \
  --early_stopping_threshold 0.0 \
  --eval_delay 5 \
  --logging_steps 5 \
  --save_total_limit 2 \
  --loss_impl hf \
  --enable_spec_augment \
  --ctc_zero_infinity \
  --no_attention_mask_for_loss \
  --finite_loss_check_samples 2 \
  --seed "$SEED" \
  --local_files_only \
  --training_log_csv "$result_dir/training_log.csv" \
  --validation_history_csv "$result_dir/validation_wer_history.csv" \
  --run_metadata_json "$metadata_path" \
  "${candidate_args[@]}" \
  "${resume_args[@]}" 2>&1 | tee -a "$log_file"

printf '[%s] completed candidate=%s fold=%s gpu=%s\n' \
  "$(date -Is)" "$candidate_name" "$FOLD" "$GPU_ID" | tee -a "$log_file"
