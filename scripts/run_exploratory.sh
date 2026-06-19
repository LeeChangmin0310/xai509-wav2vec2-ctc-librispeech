#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT="$ROOT/outputs/exploratory_h_alltrain"
DECODER_CONFIG="$ROOT/results/selected_decoder_config.json"
ALL_TRAIN_SHARDS="$ROOT/data/train/shard-000000.tar,$ROOT/data/train/shard-000001.tar,$ROOT/data/train/shard-000002.tar,$ROOT/data/train/shard-000003.tar,$ROOT/data/train/shard-000004.tar"
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
  echo "Missing decoder config; run scripts/run_validation_decode.sh first." >&2
  exit 1
fi
if [[ "$FORCE" == "1" && "$DRY_RUN" != "1" ]]; then
  rm -rf "$OUTPUT"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

COMMON_ARGS=(
  --experiment_role main
  --train_shards "$ALL_TRAIN_SHARDS"
  --eval_shards "$ALL_TRAIN_SHARDS"
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 2
  --warmup_ratio 0.1
  --max_grad_norm 1.0
  --early_stopping_patience 0
  --eval_delay 0
  --logging_steps 5
  --save_total_limit 1
  --no_load_best_model_at_end
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
  --learning_rate 1e-4
  --freeze_feature_encoder
  --layerwise_lr_decay
  --layerwise_lr_decay_rate 1.0
  --head_learning_rate 1e-3
  "${LOCAL_ARGS[@]}"
)

if [[ "$DRY_RUN" == "1" ]]; then
  validate_modules src.train src.infer src.eval src.data src.normalization src.guard
  echo "DRY RUN: Staged CTC All-Train Model"
  print_command "${STAGE1_COMMAND[@]}"
  print_command "${STAGE2_COMMAND[@]}"
else
  if [[ ! -f "$OUTPUT/stage1/best_model/config.json" ]]; then
    "${STAGE1_COMMAND[@]}"
  fi

  if [[ ! -f "$OUTPUT/best_model/config.json" ]]; then
    "${STAGE2_COMMAND[@]}"
  fi
fi

mapfile -t DECODER < <(
  "$PYTHON" -c "import json; d=json.load(open('$DECODER_CONFIG')); print(d['beam_width']); print(d['alpha']); print(d['beta']); print(d['language_model_path'])"
)
PREDICTION_DIR="$OUTPUT/test_predictions"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$PREDICTION_DIR"
fi

run_split() {
  local shards="$1"
  local result_file="$2"
  local -a command=(
    "$PYTHON" -m src.infer
    --experiment_role main
    --model_name_or_path "$OUTPUT/best_model"
    --input_shards "$shards"
    --result_file "$result_file"
    --output_dir "$PREDICTION_DIR"
    --per_device_eval_batch_size 8
    --normalize_text
    --fp16
    --no_attention_mask_for_forward
    --decoding_method beam
    --beam_width "${DECODER[0]}"
    --alpha "${DECODER[1]}"
    --beta "${DECODER[2]}"
    --language_model_path "${DECODER[3]}"
    "${LOCAL_ARGS[@]}"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

run_split "$ROOT/data/test-clean" "$PREDICTION_DIR/test_clean_predictions.txt"
run_split "$ROOT/data/test-other" "$PREDICTION_DIR/test_other_predictions.txt"

EVAL_COMMAND=(
  "$PYTHON" -m src.eval
  "$PREDICTION_DIR/test_clean_predictions.txt"
  "$PREDICTION_DIR/test_other_predictions.txt"
  --normalize_text
  --summary_csv "$ROOT/results/exploratory_summary.csv"
  --experiment_name staged_ctc_all_train
  --setting "Staged CTC All-Train Model"
  --internal_id H-all-train
  --decoder "frozen main beam + train-text trigram LM"
  --notes "Exploratory all-train run with fixed final epoch and no untouched validation split"
)

if [[ "$DRY_RUN" == "1" ]]; then
  print_command "${EVAL_COMMAND[@]}"
  echo "ROVER remains report-only because folds 0-3 trained on shard 000004."
  exit 0
fi

"${EVAL_COMMAND[@]}"

echo "ROVER remains report-only because folds 0-3 trained on shard 000004."
