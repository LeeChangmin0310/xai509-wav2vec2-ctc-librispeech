#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_IDS_TEXT="${GPU_IDS:-4 5}"
CV_CANDIDATES_TEXT="${CV_CANDIDATES:-A B C D E}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
FORCE="${FORCE:-0}"

read -r -a gpu_ids <<< "$GPU_IDS_TEXT"
read -r -a candidates <<< "$CV_CANDIDATES_TEXT"
if (( ${#gpu_ids[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
bash "$SCRIPT_DIR/run_specaugment_finite_loss_check.sh"

tasks=()
for candidate in "${candidates[@]}"; do
  for fold in 0 1 2 3 4; do
    tasks+=("$candidate:$fold")
  done
done

failed=()
task_index=0
while (( task_index < ${#tasks[@]} )); do
  pids=()
  names=()
  for gpu_id in "${gpu_ids[@]}"; do
    if (( task_index >= ${#tasks[@]} )); then
      break
    fi
    task="${tasks[$task_index]}"
    candidate="${task%%:*}"
    fold="${task##*:}"
    echo "Launching candidate $candidate fold $fold on GPU $gpu_id"
    GPU_ID="$gpu_id" CANDIDATE="$candidate" FOLD="$fold" \
      CONDA_ENV="$CONDA_ENV" FORCE="$FORCE" \
      bash "$SCRIPT_DIR/run_base_strict_cv_fold.sh" &
    pids+=("$!")
    names+=("$task")
    task_index=$((task_index + 1))
  done

  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failed+=("${names[$index]}")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        echo "Stopping after failed CV task ${names[$index]}" >&2
        exit 1
      fi
    fi
  done
done

conda run -n "$CONDA_ENV" python scripts/summarize_base_strict_cv.py
if (( ${#failed[@]} > 0 )); then
  printf '%s\n' "${failed[@]}" > logs/base_strict_cv_failed.txt
  echo "CV completed with failures: ${failed[*]}" >&2
  exit 1
fi
