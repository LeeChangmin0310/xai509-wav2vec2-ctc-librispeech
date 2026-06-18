#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:-1}"
FORCE="${FORCE:-0}"
output_dir="$PROJECT_ROOT/outputs/base_strict_exploratory/h_alltrain"
result_root="$PROJECT_ROOT/results/base_strict_exploratory"
result_dir="$result_root/h_alltrain"
log_dir="$PROJECT_ROOT/logs/base_strict_exploratory"
log_file="$log_dir/h_alltrain.log"
model_path="$output_dir/best_model"
decoder_path="$PROJECT_ROOT/results/base_strict_final/selected_decoder_config.json"
lm_path="$PROJECT_ROOT/results/base_strict_final/train_text_trigram_lm.json"
train_shards="$PROJECT_ROOT/data/train/shard-000000.tar,$PROJECT_ROOT/data/train/shard-000001.tar,$PROJECT_ROOT/data/train/shard-000002.tar,$PROJECT_ROOT/data/train/shard-000003.tar,$PROJECT_ROOT/data/train/shard-000004.tar"
clean_predictions="$result_root/h_alltrain_test_clean_predictions.txt"
other_predictions="$result_root/h_alltrain_test_other_predictions.txt"
summary_path="$result_root/h_alltrain_wer_summary.csv"

mkdir -p "$output_dir" "$result_dir" "$log_dir"
cd "$PROJECT_ROOT"

conda run -n "$CONDA_ENV" python -c \
  "import json; d=json.load(open('$decoder_path')); assert d['decoding_method']=='beam_lm'; assert d['beam_width']==50; assert d['alpha']==0.3; assert d['beta']==1.5; assert d['test_splits_used_for_selection'] is False"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_training() {
  local model_source="$1"
  local stage_output="$2"
  local stage_result="$3"
  local epochs="$4"
  shift 4
  local -a extra_args=("$@")

  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$PROJECT_ROOT/wav2vec_finetuning.py" \
    --experiment_role main \
    --model_name_or_path "$model_source" \
    --train_shards "$train_shards" \
    --eval_shards "$train_shards" \
    --output_dir "$stage_output" \
    --final_model_subdir best_model \
    --num_train_epochs "$epochs" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --early_stopping_patience 0 \
    --eval_delay 0 \
    --logging_steps 5 \
    --save_total_limit 1 \
    --no_load_best_model_at_end \
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
    --validation_history_csv "$stage_result/training_set_diagnostics.csv" \
    --run_metadata_json "$stage_result/run_metadata.json" \
    "${extra_args[@]}"
}

if [[ "$FORCE" == "1" ]] || [[ ! -f "$model_path/config.json" ]]; then
  if [[ "$FORCE" == "1" ]] || \
    [[ ! -f "$output_dir/stage1/best_model/config.json" ]]; then
    mkdir -p "$output_dir/stage1" "$result_dir/stage1"
    printf '[%s] stage1=fixed_head_only epochs=10\n' "$(date -Is)" \
      | tee -a "$log_file"
    run_training \
      facebook/wav2vec2-base \
      "$output_dir/stage1" \
      "$result_dir/stage1" \
      10 \
      --learning_rate 1e-3 \
      --freeze_wav2vec2 2>&1 | tee -a "$log_file"
  fi

  printf '[%s] stage2=fixed_encoder epochs=40\n' "$(date -Is)" \
    | tee -a "$log_file"
  run_training \
    "$output_dir/stage1/best_model" \
    "$output_dir" \
    "$result_dir" \
    40 \
    --learning_rate 1e-4 \
    --freeze_feature_encoder \
    --layerwise_lr_decay \
    --layerwise_lr_decay_rate 1.0 \
    --head_learning_rate 1e-3 2>&1 | tee -a "$log_file"
else
  echo "Reusing completed H all-train model: $model_path" | tee -a "$log_file"
fi

run_test_split() {
  local shards="$1"
  local prediction_path="$2"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$PROJECT_ROOT/wav2vec_inference.py" \
    --experiment_role main \
    --model_name_or_path "$model_path" \
    --input_shards "$shards" \
    --result_file "$prediction_path" \
    --output_dir "$result_root/h_alltrain_inference" \
    --per_device_eval_batch_size 8 \
    --normalize_text \
    --fp16 \
    --no_attention_mask_for_forward \
    --local_files_only \
    --decoding_method beam \
    --beam_width 50 \
    --language_model_path "$lm_path" \
    --alpha 0.3 \
    --beta 1.5
}

if [[ "$FORCE" == "1" ]] || [[ ! -f "$summary_path" ]]; then
  if [[ "$FORCE" == "1" ]] || [[ ! -f "$clean_predictions" ]]; then
    run_test_split "$PROJECT_ROOT/data/test-clean" "$clean_predictions" \
      2>&1 | tee -a "$log_file"
  fi
  if [[ "$FORCE" == "1" ]] || [[ ! -f "$other_predictions" ]]; then
    run_test_split "$PROJECT_ROOT/data/test-other" "$other_predictions" \
      2>&1 | tee -a "$log_file"
  fi
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/summarize_h_alltrain.py" \
    --metadata "$result_dir/run_metadata.json" \
    --decoder "$decoder_path" \
    --test-clean "$clean_predictions" \
    --test-other "$other_predictions" \
    --output "$summary_path" 2>&1 | tee -a "$log_file"
else
  echo "H all-train test summary already exists." | tee -a "$log_file"
fi
