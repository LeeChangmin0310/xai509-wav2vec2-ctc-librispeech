#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
CANDIDATES_TEXT="${CANDIDATES:-H}"
FOLDS_TEXT="${FOLDS:-1 2 3 4}"
GPU_IDS_TEXT="${GPU_IDS:-4 5}"
FORCE="${FORCE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

read -r -a candidates <<< "$CANDIDATES_TEXT"
read -r -a folds <<< "$FOLDS_TEXT"
read -r -a gpu_ids <<< "$GPU_IDS_TEXT"

if (( ${#candidates[@]} == 0 || ${#folds[@]} == 0 || ${#gpu_ids[@]} == 0 )); then
  echo "CANDIDATES, FOLDS, and GPU_IDS must each contain at least one value." >&2
  exit 2
fi
for candidate in "${candidates[@]}"; do
  if [[ "$candidate" != "H" && "$candidate" != "F" ]]; then
    echo "Unsupported candidate $candidate; expected H or F." >&2
    exit 2
  fi
done
for fold in "${folds[@]}"; do
  if [[ ! "$fold" =~ ^[0-4]$ ]]; then
    echo "Invalid fold $fold; expected 0 through 4." >&2
    exit 2
  fi
done

mkdir -p "$PROJECT_ROOT/logs/base_strict_cv"
queue_log="$PROJECT_ROOT/logs/base_strict_cv/hf_cv_gpu45_queue.log"
tasks=()
for candidate in "${candidates[@]}"; do
  for fold in "${folds[@]}"; do
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
    command_text="GPU_ID=$gpu_id CANDIDATE=$candidate FOLD=$fold CONDA_ENV=$CONDA_ENV FORCE=$FORCE RUN_SUFFIX=$RUN_SUFFIX bash scripts/run_base_strict_strong_cv_fold.sh"
    printf '[%s] launch %s\n' "$(date -Is)" "$command_text" | tee -a "$queue_log"
    GPU_ID="$gpu_id" CANDIDATE="$candidate" FOLD="$fold" \
      CONDA_ENV="$CONDA_ENV" FORCE="$FORCE" RUN_SUFFIX="$RUN_SUFFIX" \
      bash "$SCRIPT_DIR/run_base_strict_strong_cv_fold.sh" &
    pids+=("$!")
    names+=("$task")
    task_index=$((task_index + 1))
  done

  batch_failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf '[%s] success task=%s\n' \
        "$(date -Is)" "${names[$index]}" | tee -a "$queue_log"
    else
      failed+=("${names[$index]}")
      batch_failed=1
      printf '[%s] failure task=%s\n' \
        "$(date -Is)" "${names[$index]}" | tee -a "$queue_log" >&2
    fi
  done
  if (( batch_failed != 0 )); then
    printf 'CV queue stopped after failures: %s\n' "${failed[*]}" | tee -a "$queue_log" >&2
    exit 1
  fi
done

printf '[%s] queue_complete tasks=%s\n' \
  "$(date -Is)" "${tasks[*]}" | tee -a "$queue_log"
