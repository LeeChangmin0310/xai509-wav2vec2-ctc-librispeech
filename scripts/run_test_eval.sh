#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
MODEL="$ROOT/outputs/strict_final/best_model"
DECODER_CONFIG="$ROOT/results/selected_decoder_config.json"
PREDICTION_DIR="$ROOT/outputs/strict_final/test_predictions"
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
if [[ ! -f "$DECODER_CONFIG" ]]; then
  echo "Missing $DECODER_CONFIG; run scripts/run_validation_decode.sh first." >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -f "$MODEL/config.json" ]]; then
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

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$PREDICTION_DIR"
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

run_split() {
  local shards="$1"
  local result_file="$2"
  local -a command=(
    "$PYTHON" -m src.infer
    --experiment_role main
    --model_name_or_path "$MODEL"
    --input_shards "$shards"
    --result_file "$result_file"
    --output_dir "$PREDICTION_DIR"
    --per_device_eval_batch_size 8
    --normalize_text
    --fp16
    --no_attention_mask_for_forward
    "${DECODER_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  validate_modules src.infer src.eval src.data src.normalization src.guard
  echo "DRY RUN: final test decoding and WER evaluation"
fi

run_split "$ROOT/data/test-clean" "$PREDICTION_DIR/test_clean_predictions.txt"
run_split "$ROOT/data/test-other" "$PREDICTION_DIR/test_other_predictions.txt"

EVAL_COMMAND=(
  "$PYTHON" -m src.eval
  "$PREDICTION_DIR/test_clean_predictions.txt"
  "$PREDICTION_DIR/test_other_predictions.txt"
  --normalize_text
  --summary_csv "$ROOT/results/strict_final_summary.csv"
  --experiment_name main_staged_ctc
  --setting "Staged CTC Fine-tuning"
  --internal_id H
  --decoder "beam + train-text trigram LM"
  --validation_wer "$VALIDATION_WER"
  --notes "Main reproducible result selected using train shard 000004 only"
)

if [[ "$DRY_RUN" == "1" ]]; then
  print_command "${EVAL_COMMAND[@]}"
  exit 0
fi

"${EVAL_COMMAND[@]}"
