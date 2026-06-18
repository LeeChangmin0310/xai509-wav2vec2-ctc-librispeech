#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:?Set GPU_ID}"
MODE="${MODE:?Set MODE to noaug or specaug}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
MASK_TIME_PROB="${MASK_TIME_PROB:-0.05}"
MASK_TIME_LENGTH="${MASK_TIME_LENGTH:-10}"
MASK_TIME_MIN_MASKS="${MASK_TIME_MIN_MASKS:-2}"

case "$MODE" in
  noaug)
    output_name="tiny_overfit_noaug"
    augment_args=()
    ;;
  specaug)
    output_name="tiny_overfit_specaug"
    augment_args=(--enable_spec_augment)
    ;;
  *)
    echo "MODE must be noaug or specaug" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/run_tiny_overfit_sanity.py" \
  --train_shards "$PROJECT_ROOT/data/train/shard-000001.tar" \
  --output_dir "$PROJECT_ROOT/results/base_strict_debug/$output_name" \
  --num_samples "$NUM_SAMPLES" \
  --num_epochs "$NUM_EPOCHS" \
  --batch_size 2 \
  --encoder_learning_rate 1e-4 \
  --head_learning_rate 1e-3 \
  --eval_every_epochs 5 \
  --mask_time_prob "$MASK_TIME_PROB" \
  --mask_time_length "$MASK_TIME_LENGTH" \
  --mask_time_min_masks "$MASK_TIME_MIN_MASKS" \
  --device cuda \
  --local_files_only \
  "${augment_args[@]}"
