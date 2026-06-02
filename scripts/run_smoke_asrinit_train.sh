#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
FP16=0
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-facebook/wav2vec2-base-960h}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
source "$SCRIPT_DIR/common.sh"

require_data

output_dir="${OUTPUT_DIR:-$OUTPUTS_DIR/smoke_asrinit}"
log_file="$LOGS_DIR/smoke_asrinit_train.log"
dry_run_args=()
debug_first_batch_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  dry_run_args+=(--dry_run)
fi
if [[ "${DEBUG_FIRST_BATCH:-0}" == "1" ]]; then
  debug_first_batch_args+=(--debug_first_batch)
fi

if [[ "${DRY_RUN:-0}" != "1" ]] && [[ "$FORCE" != "1" ]] && has_checkpoint "$output_dir"; then
  echo "Skipping ASR-init smoke training: checkpoint already exists. Set FORCE=1 to rerun."
  exit 0
fi

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using fp16=$FP16"
"$PYTHON" "$PROJECT_ROOT/wav2vec_finetuning.py" \
  --train_shards "$TRAIN_SHARDS" \
  --test_clean_shards "$TEST_CLEAN_SHARDS" \
  --test_other_shards "$TEST_OTHER_SHARDS" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$output_dir" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --num_train_epochs 1 \
  --max_train_steps 2 \
  --max_eval_samples 4 \
  --per_device_train_batch_size "${SMOKE_TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${SMOKE_EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps 1 \
  --loss_impl hf \
  --ctc_zero_infinity \
  --disable_spec_augment \
  --no_attention_mask_for_loss \
  --seed "$SEED" \
  "${FP16_ARGS[@]}" \
  "${debug_first_batch_args[@]}" \
  "${dry_run_args[@]}" 2>&1 | tee -a "$log_file"
