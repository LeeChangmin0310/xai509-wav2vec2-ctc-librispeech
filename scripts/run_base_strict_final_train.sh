#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:-4}"
FORCE="${FORCE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SEED="${SEED:-42}"
selected_path="$PROJECT_ROOT/results/base_strict_cv/selected_config.json"

if [[ ! -f "$selected_path" ]]; then
  echo "Missing $selected_path. Run scripts/run_base_strict_cv.sh first." >&2
  exit 1
fi

candidate="$(
  conda run -n "$CONDA_ENV" python -c \
    "import json; d=json.load(open('$selected_path')); print(d['candidate'] if d else '')"
)"
if [[ -z "$candidate" ]]; then
  echo "No five-fold candidate is selected in $selected_path" >&2
  exit 1
fi

case "$candidate" in
  lr3e-5_freeze_feature)
    candidate_args=(--learning_rate 3e-5 --freeze_feature_encoder)
    ;;
  lr5e-5_freeze_feature)
    candidate_args=(--learning_rate 5e-5 --freeze_feature_encoder)
    ;;
  lr1e-4_freeze_feature)
    candidate_args=(--learning_rate 1e-4 --freeze_feature_encoder)
    ;;
  lr3e-5_full)
    candidate_args=(--learning_rate 3e-5)
    ;;
  lr5e-5_full)
    candidate_args=(--learning_rate 5e-5)
    ;;
  *)
    echo "Unsupported selected candidate: $candidate" >&2
    exit 2
    ;;
esac

train_shards="$PROJECT_ROOT/data/train/shard-000000.tar,$PROJECT_ROOT/data/train/shard-000001.tar,$PROJECT_ROOT/data/train/shard-000002.tar,$PROJECT_ROOT/data/train/shard-000003.tar"
eval_shard="$PROJECT_ROOT/data/train/shard-000004.tar"
output_dir="$PROJECT_ROOT/outputs/base_strict_final"
result_dir="$PROJECT_ROOT/results/base_strict_final"
log_file="$PROJECT_ROOT/logs/base_strict_final.log"
mkdir -p "$output_dir" "$result_dir" "$PROJECT_ROOT/logs"

if [[ "$FORCE" != "1" ]] && [[ -f "$output_dir/best_model/config.json" ]] && [[ -f "$result_dir/run_metadata.json" ]]; then
  echo "Strict final training is already complete. Set FORCE=1 to rerun."
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

printf '[%s] selected_candidate=%s gpu=%s\n' \
  "$(date -Is)" "$candidate" "$GPU_ID" | tee -a "$log_file"

conda run --no-capture-output -n "$CONDA_ENV" python "$PROJECT_ROOT/wav2vec_finetuning.py" \
  --experiment_role main \
  --model_name_or_path facebook/wav2vec2-base \
  --train_shards "$train_shards" \
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
  --training_log_csv "$result_dir/final_training_log.csv" \
  --validation_history_csv "$result_dir/validation_wer_history.csv" \
  --run_metadata_json "$result_dir/run_metadata.json" \
  "${candidate_args[@]}" \
  "${resume_args[@]}" 2>&1 | tee -a "$log_file"

printf '[%s] strict final training complete\n' "$(date -Is)" | tee -a "$log_file"
