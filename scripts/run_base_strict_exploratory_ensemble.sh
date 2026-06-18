#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:-0}"
PHASE="${PHASE:?Set PHASE=validation or PHASE=test}"
result_dir="$PROJECT_ROOT/results/base_strict_exploratory"
log_dir="$PROJECT_ROOT/logs/base_strict_exploratory"
log_file="$log_dir/h_fold_ensemble_${PHASE}.log"

if [[ "$PHASE" != "validation" && "$PHASE" != "test" ]]; then
  echo "PHASE must be validation or test" >&2
  exit 2
fi
if [[ "$PHASE" == "test" ]] \
  && [[ ! -f "$result_dir/selected_h_fold_ensemble.json" ]]; then
  echo "Missing frozen validation selection for H-fold ensemble." >&2
  exit 1
fi

mkdir -p "$result_dir" "$log_dir" \
  "$PROJECT_ROOT/outputs/base_strict_exploratory"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/evaluate_h_fold_ensemble.py" \
  --phase "$PHASE" \
  --validation_shards "$PROJECT_ROOT/data/train/shard-000004.tar" \
  --test_clean_shards "$PROJECT_ROOT/data/test-clean" \
  --test_other_shards "$PROJECT_ROOT/data/test-other" \
  --language_model_path \
    "$PROJECT_ROOT/results/base_strict_final/train_text_trigram_lm.json" \
  --result_dir "$result_dir" \
  --beam_width 50 \
  --alpha 0.3 \
  --beta 1.5 \
  --batch_size 4 \
  --fp16 \
  --no_attention_mask_for_forward \
  --local_files_only 2>&1 | tee -a "$log_file"
