#!/usr/bin/env python3
"""Summarize a validation-only exploratory acoustic refinement."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_wer import read_ref_hyp_pairs
from text_normalization import normalize_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    references, hypotheses = read_ref_hyp_pairs(args.predictions)
    references = [normalize_transcript(text) for text in references]
    hypotheses = [normalize_transcript(text) for text in hypotheses]
    row = {
        "variant": args.variant,
        "training_best_greedy_wer": metadata.get("best_metric", ""),
        "fixed_decoder_validation_wer": wer(references, hypotheses),
        "decoder_type": "beam_lm",
        "beam_width": 50,
        "alpha": 0.3,
        "beta": 1.5,
        "validation_shard": "data/train/shard-000004.tar",
        "test_splits_evaluated": False,
        "checkpoint_path": metadata["saved_model_path"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"{args.variant} validation WER: {row['fixed_decoder_validation_wer']:.9f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
