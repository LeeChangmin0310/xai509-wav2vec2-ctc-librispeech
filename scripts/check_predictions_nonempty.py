#!/usr/bin/env python3
"""Report empty hypotheses and sample REF/HYP pairs from an inference result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse the result file path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", type=Path)
    return parser.parse_args()


def read_pairs(result_file: Path) -> List[Tuple[str, str]]:
    """Read the REF/HYP format written by wav2vec_inference.py."""
    lines = [
        line.strip()
        for line in result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected REF/HYP pairs in {result_file}")

    pairs = []
    for index in range(0, len(lines), 2):
        ref_line = lines[index]
        hyp_line = lines[index + 1]
        if not ref_line.startswith("REF:") or not hyp_line.startswith("HYP:"):
            raise ValueError(f"Malformed REF/HYP pair near line {index + 1}")
        pairs.append(
            (
                ref_line[len("REF:") :].strip(),
                hyp_line[len("HYP:") :].strip(),
            )
        )
    return pairs


def main() -> None:
    """Print empty-hypothesis diagnostics."""
    args = parse_args()
    pairs = read_pairs(args.result_file)
    empty_count = sum(not hypothesis for _, hypothesis in pairs)
    empty_fraction = empty_count / len(pairs) if pairs else 0.0

    print(f"result_file: {args.result_file}")
    print(f"samples: {len(pairs)}")
    print(f"empty hypotheses: {empty_count}")
    print(f"empty hypothesis fraction: {empty_fraction:.6f}")
    print("first 5 REF/HYP pairs:")
    for index, (reference, hypothesis) in enumerate(pairs[:5], start=1):
        print(f"[{index}] REF: {reference}")
        print(f"[{index}] HYP: {hypothesis or '<EMPTY>'}")


if __name__ == "__main__":
    main()
