#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

echo "== Git =="
git -C "$PROJECT_ROOT" branch --show-current
git -C "$PROJECT_ROOT" log -1 --oneline --decorate

echo
echo "== Data shards =="
for split in train test-clean test-other; do
  split_dir="$DATA_DIR/$split"
  if [[ -d "$split_dir" ]]; then
    count="$(find "$split_dir" -maxdepth 1 -type f -name '*.tar' | wc -l | tr -d '[:space:]')"
    echo "$split: $count"
  else
    echo "$split: missing ($split_dir)"
  fi
done

echo
echo "== Outputs =="
if [[ -d "$PROJECT_ROOT/outputs" ]]; then
  find "$PROJECT_ROOT/outputs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
else
  echo "(none)"
fi

echo
echo "== Results =="
if [[ -d "$PROJECT_ROOT/results" ]]; then
  find "$PROJECT_ROOT/results" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
else
  echo "(none)"
fi

echo
echo "== ASR-init outputs =="
if [[ -d "$PROJECT_ROOT/outputs" ]]; then
  asrinit_outputs="$(
    find "$PROJECT_ROOT/outputs" -mindepth 1 -maxdepth 1 -type d \
      \( -name 'asrinit_*' -o -name 'asr_pretrained_*' \) -printf '%f\n' | sort
  )"
  printf '%s\n' "${asrinit_outputs:-(none)}"
else
  echo "(none)"
fi

echo
echo "== ASR-init results =="
if [[ -d "$PROJECT_ROOT/results" ]]; then
  asrinit_results="$(
    find "$PROJECT_ROOT/results" -mindepth 1 -maxdepth 1 -type d \
      \( -name 'asrinit_*' -o -name 'asr_pretrained_*' \) -printf '%f\n' | sort
  )"
  printf '%s\n' "${asrinit_results:-(none)}"
else
  echo "(none)"
fi

echo
echo "== Experiment configs =="
if [[ -d "$PROJECT_ROOT/configs" ]]; then
  find "$PROJECT_ROOT/configs" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' | sort
else
  echo "(none)"
fi

echo
echo "== WER summary =="
if [[ -f "$PROJECT_ROOT/results/wer_summary.csv" ]]; then
  cat "$PROJECT_ROOT/results/wer_summary.csv"
else
  echo "(not created yet)"
fi

echo
echo "== ASR-init WER summary =="
if [[ -f "$PROJECT_ROOT/results/wer_summary_asrinit.csv" ]]; then
  cat "$PROJECT_ROOT/results/wer_summary_asrinit.csv"
else
  echo "(not created yet)"
fi

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi could not query the GPU."
else
  echo "nvidia-smi is not available."
fi
