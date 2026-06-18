#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
FORCE="${FORCE:-0}"
OUTPUT="$ROOT/outputs/strict_final"
TRAIN_SHARDS="$ROOT/data/train/shard-000000.tar,$ROOT/data/train/shard-000001.tar,$ROOT/data/train/shard-000002.tar,$ROOT/data/train/shard-000003.tar"
VALIDATION_SHARD="$ROOT/data/train/shard-000004.tar"
LOCAL_ARGS=()

if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_ARGS=(--local_files_only)
fi
if [[ "$FORCE" == "1" ]]; then
  rm -rf "$OUTPUT"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

COMMON_ARGS=(
  --experiment_role main
  --train_shards "$TRAIN_SHARDS"
  --eval_shards "$VALIDATION_SHARD"
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 2
  --warmup_ratio 0.1
  --max_grad_norm 1.0
  --logging_steps 5
  --save_total_limit 3
  --loss_impl hf
  --enable_spec_augment
  --mask_time_prob 0.01
  --mask_time_length 5
  --mask_time_min_masks 1
  --ctc_zero_infinity
  --no_attention_mask_for_loss
  --finite_loss_check_samples 2
  --seed 42
)

if [[ ! -f "$OUTPUT/stage1/best_model/config.json" ]]; then
  "$PYTHON" -m src.train \
    "${COMMON_ARGS[@]}" \
    --model_name_or_path facebook/wav2vec2-base \
    --output_dir "$OUTPUT/stage1" \
    --final_model_subdir best_model \
    --num_train_epochs 10 \
    --early_stopping_patience 0 \
    --early_stopping_start_epoch 20 \
    --eval_delay 1 \
    --no_load_best_model_at_end \
    --learning_rate 1e-3 \
    --freeze_wav2vec2 \
    "${LOCAL_ARGS[@]}"
fi

"$PYTHON" -m src.train \
  "${COMMON_ARGS[@]}" \
  --model_name_or_path "$OUTPUT/stage1/best_model" \
  --output_dir "$OUTPUT" \
  --final_model_subdir best_model \
  --num_train_epochs 40 \
  --early_stopping_patience 8 \
  --early_stopping_start_epoch 10 \
  --eval_delay 0 \
  --load_best_model_at_end \
  --learning_rate 1e-4 \
  --freeze_feature_encoder \
  --layerwise_lr_decay \
  --layerwise_lr_decay_rate 1.0 \
  --head_learning_rate 1e-3 \
  "${LOCAL_ARGS[@]}"
