#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:-6}"
CANDIDATE="${CANDIDATE:?Set CANDIDATE A, B, C, D, or E}"
FOLD="${FOLD:?Set FOLD from 0 through 4}"
CHECKPOINTS_TEXT="${CHECKPOINTS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
  echo "FOLD must be between 0 and 4" >&2
  exit 2
fi

case "$CANDIDATE" in
  A) candidate_name="lr3e-5_freeze_feature" ;;
  B) candidate_name="lr5e-5_freeze_feature" ;;
  C) candidate_name="lr1e-4_freeze_feature" ;;
  D) candidate_name="lr3e-5_full" ;;
  E) candidate_name="lr5e-5_full" ;;
  *)
    echo "Unknown CANDIDATE=$CANDIDATE; expected A, B, C, D, or E" >&2
    exit 2
    ;;
esac

checkpoint_root="$PROJECT_ROOT/outputs/base_strict_cv/$candidate_name/fold_$FOLD"
result_dir="$PROJECT_ROOT/results/base_strict_cv/$candidate_name/fold_$FOLD"
eval_shard="$PROJECT_ROOT/data/train/shard-$(printf '%06d' "$FOLD").tar"

checkpoint_args=()
if [[ -n "$CHECKPOINTS_TEXT" ]]; then
  read -r -a checkpoint_steps <<< "$CHECKPOINTS_TEXT"
  for checkpoint_step in "${checkpoint_steps[@]}"; do
    checkpoint_args+=(--checkpoint "$checkpoint_root/checkpoint-$checkpoint_step")
  done
else
  mapfile -t checkpoint_paths < <(
    find "$checkpoint_root" -maxdepth 1 -type d -name 'checkpoint-*' \
      -printf '%f\n' | sort -t- -k2,2n
  )
  if (( ${#checkpoint_paths[@]} == 0 )); then
    echo "No checkpoints found under $checkpoint_root" >&2
    exit 1
  fi
  for checkpoint_path in "${checkpoint_paths[@]}"; do
    checkpoint_args+=(--checkpoint "$checkpoint_root/$checkpoint_path")
  done
fi

mkdir -p "$result_dir"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/diagnose_validation_decoding.py" \
  "${checkpoint_args[@]}" \
  --validation_shards "$eval_shard" \
  --output_csv "$result_dir/blank_diagnostics.csv" \
  --examples_output "$result_dir/ref_hyp_examples.txt" \
  --batch_size "$BATCH_SIZE" \
  --device cuda \
  --local_files_only
