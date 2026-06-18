#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:-4}"
FORCE="${FORCE:-0}"
result_dir="$PROJECT_ROOT/results/base_strict_final"
model_path="$PROJECT_ROOT/outputs/base_strict_final/best_model"
lm_path="$result_dir/train_text_trigram_lm.json"
sweep_path="$result_dir/validation_decoding_sweep.csv"
log_file="$PROJECT_ROOT/logs/base_strict_decoder_tuning.log"
train_shards="$PROJECT_ROOT/data/train/shard-000000.tar,$PROJECT_ROOT/data/train/shard-000001.tar,$PROJECT_ROOT/data/train/shard-000002.tar,$PROJECT_ROOT/data/train/shard-000003.tar"

if [[ ! -f "$model_path/config.json" ]]; then
  echo "Missing final acoustic checkpoint: $model_path" >&2
  exit 1
fi
mkdir -p "$result_dir" "$PROJECT_ROOT/logs"

if [[ "$FORCE" == "1" ]] || [[ ! -f "$lm_path" ]]; then
  conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/train_simple_ngram_lm.py" \
    --train_shards "$train_shards" \
    --output_path "$lm_path" \
    --text_output_path "$result_dir/lm_training_text.txt" \
    --order 3 \
    --add_k 0.1 2>&1 | tee -a "$log_file"
fi

if [[ "$FORCE" != "1" ]] \
  && [[ -f "$sweep_path" ]] \
  && [[ -f "$result_dir/selected_decoder_config.json" ]]; then
  echo "Decoder sweep already exists. Set FORCE=1 to rerun."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/tune_base_strict_decoder.py" \
  --model_name_or_path "$model_path" \
  --validation_shards "$PROJECT_ROOT/data/train/shard-000004.tar" \
  --lm_path "$lm_path" \
  --output_csv "$sweep_path" \
  --best_decoder_json "$result_dir/selected_decoder_config.json" \
  --greedy_output_csv "$result_dir/validation_greedy_wer.csv" \
  --best_predictions "$result_dir/validation_ref_hyp_examples.txt" \
  --beam_widths 50,100,200,300 \
  --alphas 0.0,0.3,0.5,0.7,1.0,1.5 \
  --betas=-1.0,0.0,0.5,1.0,1.5,2.0 \
  --batch_size 4 \
  --fp16 \
  --no_attention_mask_for_forward \
  --local_files_only 2>&1 | tee -a "$log_file"
