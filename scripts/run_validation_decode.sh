#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
MODEL="$ROOT/outputs/strict_final/best_model"
LM="$ROOT/outputs/strict_final/train_text_trigram_lm.json"
TRAIN_SHARDS="$ROOT/data/train/shard-000000.tar,$ROOT/data/train/shard-000001.tar,$ROOT/data/train/shard-000002.tar,$ROOT/data/train/shard-000003.tar"
LOCAL_ARGS=()

if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_ARGS=(--local_files_only)
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Missing $MODEL; run scripts/run_train.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m src.lm \
  --train_shards "$TRAIN_SHARDS" \
  --output_path "$LM" \
  --order 3 \
  --add_k 0.1

"$PYTHON" -m src.infer \
  --tune_decoder \
  --experiment_role main \
  --model_name_or_path outputs/strict_final/best_model \
  --output_dir "$ROOT/outputs/strict_final/validation" \
  --validation_shards data/train/shard-000004.tar \
  --language_model_path outputs/strict_final/train_text_trigram_lm.json \
  --decoder_config_output "$ROOT/results/selected_decoder_config.json" \
  --decoder_sweep_output "$ROOT/outputs/strict_final/validation_decoding_sweep.csv" \
  --beam_widths 50,100,200,300 \
  --alphas 0.0,0.3,0.5,0.7,1.0,1.5 \
  --betas=-1.0,0.0,0.5,1.0,1.5,2.0 \
  --per_device_eval_batch_size 4 \
  --fp16 \
  --no_attention_mask_for_forward \
  "${LOCAL_ARGS[@]}"
