#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_ID="${GPU_ID:-4}"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conda run -n "$CONDA_ENV" python scripts/check_specaugment_finite_loss.py \
  --model_name_or_path facebook/wav2vec2-base \
  --train_shards data/train/shard-000000.tar \
  --num_samples 2 \
  --device cuda \
  --local_files_only
