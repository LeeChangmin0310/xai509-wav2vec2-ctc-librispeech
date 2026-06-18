#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_IDS="${GPU_IDS:-4 5}"

cd "$PROJECT_ROOT"
GPU_IDS="$GPU_IDS" CONDA_ENV="$CONDA_ENV" bash "$SCRIPT_DIR/run_base_strict_cv.sh"
GPU_ID="${FINAL_GPU_ID:-4}" CONDA_ENV="$CONDA_ENV" bash "$SCRIPT_DIR/run_base_strict_final_train.sh"
GPU_ID="${FINAL_GPU_ID:-4}" CONDA_ENV="$CONDA_ENV" bash "$SCRIPT_DIR/run_base_strict_decoder_tuning.sh"
GPU_ID="${FINAL_GPU_ID:-4}" CONDA_ENV="$CONDA_ENV" bash "$SCRIPT_DIR/run_base_strict_final_evaluation.sh"
