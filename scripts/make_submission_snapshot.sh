#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SNAPSHOT_DIR="$PROJECT_ROOT/submission_snapshot"

copy_tree_without_weights() {
  local source_dir="$1"
  local destination_dir="$2"
  [[ -d "$source_dir" ]] || return 0
  while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$source_dir"/}"
    destination_file="$destination_dir/$relative_path"
    mkdir -p "$(dirname "$destination_file")"
    cp "$source_file" "$destination_file"
  done < <(
    find "$source_dir" -type f \
      ! -name '*.tar' \
      ! -name '*.pt' \
      ! -name '*.pth' \
      ! -name '*.ckpt' \
      ! -name '*.bin' \
      ! -name '*.safetensors' \
      ! -path '*/__pycache__/*' \
      -print0
  )
}

rm -rf "$SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"

cp "$PROJECT_ROOT/README.md" "$SNAPSHOT_DIR/"
cp "$PROJECT_ROOT/requirements.txt" "$SNAPSHOT_DIR/"
cp "$PROJECT_ROOT/environment.yml" "$SNAPSHOT_DIR/"
find "$PROJECT_ROOT" -maxdepth 1 -type f -name '*.py' -exec cp {} "$SNAPSHOT_DIR/" \;

copy_tree_without_weights "$PROJECT_ROOT/scripts" "$SNAPSHOT_DIR/scripts"
copy_tree_without_weights "$PROJECT_ROOT/configs" "$SNAPSHOT_DIR/configs"
copy_tree_without_weights "$PROJECT_ROOT/reports" "$SNAPSHOT_DIR/reports"

mkdir -p "$SNAPSHOT_DIR/results"
for result_file in wer_summary.csv wer_summary.md; do
  if [[ -f "$PROJECT_ROOT/results/$result_file" ]]; then
    cp "$PROJECT_ROOT/results/$result_file" "$SNAPSHOT_DIR/results/"
  fi
done
copy_tree_without_weights \
  "$PROJECT_ROOT/results/figures" \
  "$SNAPSHOT_DIR/results/figures"

echo "Created submission snapshot: $SNAPSHOT_DIR"
