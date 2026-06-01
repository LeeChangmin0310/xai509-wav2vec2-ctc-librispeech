#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

print_manual_instructions() {
  cat <<EOF
Automatic download was not completed.

Download the course-provided final WebDataset shards manually and place them at:
  $DATA_DIR/train/*.tar
  $DATA_DIR/test-clean/*.tar
  $DATA_DIR/test-other/*.tar

Keep each .tar file intact. Do not extract the final WebDataset shards.
Then run:
  bash scripts/check_data.sh
EOF
}

download_split() {
  local split="$1"
  local url="$2"
  local output_dir="$DATA_DIR/$split"
  mkdir -p "$output_dir"

  if find "$output_dir" -maxdepth 1 -type f -name '*.tar' -print -quit |
      grep -q .; then
    echo "Skipping $split: tar shards already exist."
    return 0
  fi
  if [[ -z "$url" ]]; then
    echo "Missing Google Drive folder URL for $split."
    return 1
  fi
  if ! gdown --folder "$url" -O "$output_dir"; then
    echo "gdown failed for $split."
    return 1
  fi
  if ! find "$output_dir" -maxdepth 1 -type f -name '*.tar' -print -quit |
      grep -q .; then
    echo "No .tar shards were downloaded for $split."
    return 1
  fi
}

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is not installed. Install it with: python -m pip install gdown"
  print_manual_instructions
  exit 1
fi

failed=0
download_split "train" "${TRAIN_GDRIVE_URL:-}" || failed=1
download_split "test-clean" "${TEST_CLEAN_GDRIVE_URL:-}" || failed=1
download_split "test-other" "${TEST_OTHER_GDRIVE_URL:-}" || failed=1

if [[ "$failed" == "1" ]]; then
  print_manual_instructions
  exit 1
fi

bash "$SCRIPT_DIR/check_data.sh"
