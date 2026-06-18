#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:?Set GPU_ID}"
VARIANT="${VARIANT:?Set VARIANT=H2 or H3}"
FORCE="${FORCE:-0}"
train_shards="$PROJECT_ROOT/data/train/shard-000000.tar,$PROJECT_ROOT/data/train/shard-000001.tar,$PROJECT_ROOT/data/train/shard-000002.tar,$PROJECT_ROOT/data/train/shard-000003.tar"
eval_shard="$PROJECT_ROOT/data/train/shard-000004.tar"
lm_path="$PROJECT_ROOT/results/base_strict_final/train_text_trigram_lm.json"
variant_lower="$(printf '%s' "$VARIANT" | tr '[:upper:]' '[:lower:]')"
output_dir="$PROJECT_ROOT/outputs/base_strict_exploratory/head_refinement_$variant_lower"
result_dir="$PROJECT_ROOT/results/base_strict_exploratory/head_refinement_$variant_lower"
log_dir="$PROJECT_ROOT/logs/base_strict_exploratory"
log_file="$log_dir/head_refinement_$variant_lower.log"
prediction_path="$result_dir/validation_predictions.txt"
summary_path="$result_dir/validation_summary.csv"

if [[ "$VARIANT" != "H2" && "$VARIANT" != "H3" ]]; then
  echo "VARIANT must be H2 or H3" >&2
  exit 2
fi

mkdir -p "$output_dir" "$result_dir" "$log_dir"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_training() {
  local model_source="$1"
  local stage_output="$2"
  local stage_result="$3"
  local epochs="$4"
  local load_best_flag="$5"
  local patience="$6"
  local early_stop_start="$7"
  shift 7
  local -a extra_args=("$@")

  mkdir -p "$stage_output" "$stage_result"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$PROJECT_ROOT/wav2vec_finetuning.py" \
    --experiment_role main \
    --model_name_or_path "$model_source" \
    --train_shards "$train_shards" \
    --eval_shards "$eval_shard" \
    --output_dir "$stage_output" \
    --final_model_subdir best_model \
    --num_train_epochs "$epochs" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --early_stopping_patience "$patience" \
    --early_stopping_threshold 0.0 \
    --early_stopping_start_epoch "$early_stop_start" \
    --eval_delay 0 \
    --logging_steps 5 \
    --save_total_limit 3 \
    "$load_best_flag" \
    --loss_impl hf \
    --enable_spec_augment \
    --mask_time_prob 0.01 \
    --mask_time_length 5 \
    --mask_time_min_masks 1 \
    --ctc_zero_infinity \
    --no_attention_mask_for_loss \
    --finite_loss_check_samples 2 \
    --seed 42 \
    --local_files_only \
    --training_log_csv "$stage_result/training_log.csv" \
    --validation_history_csv "$stage_result/validation_wer_history.csv" \
    --run_metadata_json "$stage_result/run_metadata.json" \
    "${extra_args[@]}"
}

if [[ "$FORCE" == "1" ]] || [[ ! -f "$output_dir/best_model/config.json" ]]; then
  if [[ "$VARIANT" == "H2" ]]; then
    stage1_source="$output_dir/stage1/best_model"
    if [[ "$FORCE" == "1" ]] || [[ ! -f "$stage1_source/config.json" ]]; then
      run_training \
        facebook/wav2vec2-base \
        "$output_dir/stage1" \
        "$result_dir/stage1" \
        15 \
        --no_load_best_model_at_end \
        0 \
        20 \
        --learning_rate 1e-3 \
        --freeze_wav2vec2 2>&1 | tee -a "$log_file"
    fi
    stage2_extra=(
      --learning_rate 1e-4
      --freeze_feature_encoder
      --layerwise_lr_decay
      --layerwise_lr_decay_rate 1.0
      --head_learning_rate 1e-3
    )
  else
    stage1_source="$PROJECT_ROOT/outputs/base_strict_cv/two_stage_head_warmup/fold_4/stage1/best_model"
    stage2_extra=(
      --learning_rate 1e-4
      --freeze_feature_encoder
      --freeze_n_layers 6
      --layerwise_lr_decay
      --layerwise_lr_decay_rate 1.0
      --head_learning_rate 1e-3
    )
  fi

  run_training \
    "$stage1_source" \
    "$output_dir" \
    "$result_dir" \
    40 \
    --load_best_model_at_end \
    8 \
    10 \
    "${stage2_extra[@]}" 2>&1 | tee -a "$log_file"
fi

if [[ "$FORCE" == "1" ]] || [[ ! -f "$summary_path" ]]; then
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$PROJECT_ROOT/wav2vec_inference.py" \
    --experiment_role main \
    --model_name_or_path "$output_dir/best_model" \
    --input_shards "$eval_shard" \
    --result_file "$prediction_path" \
    --output_dir "$result_dir/inference" \
    --per_device_eval_batch_size 4 \
    --normalize_text \
    --fp16 \
    --no_attention_mask_for_forward \
    --local_files_only \
    --decoding_method beam \
    --beam_width 50 \
    --language_model_path "$lm_path" \
    --alpha 0.3 \
    --beta 1.5 2>&1 | tee -a "$log_file"

  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/summarize_exploratory_validation.py" \
    --variant "$VARIANT" \
    --metadata "$result_dir/run_metadata.json" \
    --predictions "$prediction_path" \
    --output "$summary_path" 2>&1 | tee -a "$log_file"
fi
