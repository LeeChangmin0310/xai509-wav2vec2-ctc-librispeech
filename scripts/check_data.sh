#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
failed=0

for split in train test-clean test-other; do
  split_dir="$DATA_DIR/$split"
  if [[ ! -d "$split_dir" ]]; then
    echo "$split: missing directory $split_dir"
    failed=1
    continue
  fi
  shard_count="$(
    find "$split_dir" -maxdepth 1 -type f -name '*.tar' |
      wc -l |
      tr -d '[:space:]'
  )"
  split_size="$(du -sh "$split_dir" | cut -f1)"
  echo "$split: $shard_count tar shard(s), $split_size"
  if [[ "$shard_count" == "0" ]]; then
    failed=1
  fi
done

if [[ "$failed" == "1" ]]; then
  echo "Data check failed. Keep final WebDataset .tar shards intact."
  exit 1
fi
