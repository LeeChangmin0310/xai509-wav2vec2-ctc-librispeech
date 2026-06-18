#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
MODEL="$ROOT/outputs/strict_final/best_model"
DECODER_CONFIG="$ROOT/results/selected_decoder_config.json"
PREDICTION_DIR="$ROOT/outputs/strict_final/test_predictions"
LOCAL_ARGS=()

if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_ARGS=(--local_files_only)
fi
if [[ ! -f "$MODEL/config.json" || ! -f "$DECODER_CONFIG" ]]; then
  echo "Run scripts/run_train.sh and scripts/run_validation_decode.sh first." >&2
  exit 1
fi

mapfile -t DECODER < <(
  "$PYTHON" -c "import json; d=json.load(open('$DECODER_CONFIG')); print(d['decoding_method']); print(d.get('beam_width', '')); print(d.get('alpha', '')); print(d.get('beta', '')); print(d.get('language_model_path', '')); print(d['validation_wer'])"
)
METHOD="${DECODER[0]}"
VALIDATION_WER="${DECODER[5]}"
DECODER_ARGS=(--decoding_method greedy)
if [[ "$METHOD" == "beam" ]]; then
  DECODER_ARGS=(--decoding_method beam --beam_width "${DECODER[1]}")
elif [[ "$METHOD" == "beam_lm" ]]; then
  DECODER_ARGS=(
    --decoding_method beam
    --beam_width "${DECODER[1]}"
    --alpha "${DECODER[2]}"
    --beta "${DECODER[3]}"
    --language_model_path "${DECODER[4]}"
  )
fi

mkdir -p "$PREDICTION_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

run_split() {
  local shards="$1"
  local result_file="$2"
  "$PYTHON" -m src.infer \
    --experiment_role main \
    --model_name_or_path "$MODEL" \
    --input_shards "$shards" \
    --result_file "$result_file" \
    --output_dir "$PREDICTION_DIR" \
    --per_device_eval_batch_size 8 \
    --normalize_text \
    --fp16 \
    --no_attention_mask_for_forward \
    "${DECODER_ARGS[@]}" \
    "${LOCAL_ARGS[@]}"
}

run_split "$ROOT/data/test-clean" "$PREDICTION_DIR/test_clean_predictions.txt"
run_split "$ROOT/data/test-other" "$PREDICTION_DIR/test_other_predictions.txt"

"$PYTHON" -m src.eval \
  "$PREDICTION_DIR/test_clean_predictions.txt" \
  "$PREDICTION_DIR/test_other_predictions.txt" \
  --normalize_text \
  --summary_csv "$ROOT/results/strict_final_summary.csv" \
  --experiment_name strict_final \
  --acoustic_config "H / two_stage_head_warmup" \
  --decoder "beam + train-text trigram LM" \
  --validation_wer "$VALIDATION_WER" \
  --notes "Validation-only checkpoint and decoder selection"
