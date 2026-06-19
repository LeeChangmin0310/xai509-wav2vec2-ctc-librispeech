#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT="$ROOT/outputs/strict_final"
TRAIN_SHARDS="$ROOT/data/train/shard-000000.tar,$ROOT/data/train/shard-000001.tar,$ROOT/data/train/shard-000002.tar,$ROOT/data/train/shard-000003.tar"
VALIDATION_SHARD="$ROOT/data/train/shard-000004.tar"
LOCAL_ARGS=()

validate_modules() {
  "$PYTHON" -c \
    "import importlib.util,sys; missing=[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]; assert not missing, f'missing modules: {missing}'; print('module paths OK:', ', '.join(sys.argv[1:]))" \
    "$@"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_ARGS=(--local_files_only)
fi
if [[ "$FORCE" == "1" && "$DRY_RUN" != "1" ]]; then
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

STAGE1_COMMAND=(
  "$PYTHON" -m src.train
  "${COMMON_ARGS[@]}"
  --model_name_or_path facebook/wav2vec2-base
  --output_dir "$OUTPUT/stage1"
  --final_model_subdir best_model
  --num_train_epochs 10
  --early_stopping_patience 0
  --early_stopping_start_epoch 20
  --eval_delay 1
  --no_load_best_model_at_end
  --learning_rate 1e-3
  --freeze_wav2vec2
  "${LOCAL_ARGS[@]}"
)

STAGE2_COMMAND=(
  "$PYTHON" -m src.train
  "${COMMON_ARGS[@]}"
  --model_name_or_path "$OUTPUT/stage1/best_model"
  --output_dir "$OUTPUT"
  --final_model_subdir best_model
  --num_train_epochs 40
  --early_stopping_patience 8
  --early_stopping_start_epoch 10
  --eval_delay 0
  --load_best_model_at_end
  --learning_rate 1e-4
  --freeze_feature_encoder
  --layerwise_lr_decay
  --layerwise_lr_decay_rate 1.0
  --head_learning_rate 1e-3
  "${LOCAL_ARGS[@]}"
)

if [[ "$DRY_RUN" == "1" ]]; then
  validate_modules src.train src.data src.normalization src.guard
  echo "DRY RUN: Staged CTC Fine-tuning"
  print_command "${STAGE1_COMMAND[@]}"
  print_command "${STAGE2_COMMAND[@]}"
  exit 0
fi

if [[ ! -f "$OUTPUT/stage1/best_model/config.json" ]]; then
  "${STAGE1_COMMAND[@]}"
fi

"${STAGE2_COMMAND[@]}"
