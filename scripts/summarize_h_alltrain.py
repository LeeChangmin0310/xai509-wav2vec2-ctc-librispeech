#!/usr/bin/env python3
"""Summarize the fixed-recipe exploratory H all-train test evaluation."""

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


MAIN_TEST_CLEAN_WER = 0.24638618381010347
MAIN_TEST_OTHER_WER = 0.3292894942972317
EXPECTED_TRAIN_SHARDS = {
    str((PROJECT_ROOT / f"data/train/shard-{index:06d}.tar").resolve())
    for index in range(5)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default="results/base_strict_exploratory/h_alltrain/run_metadata.json",
    )
    parser.add_argument(
        "--decoder",
        default="results/base_strict_final/selected_decoder_config.json",
    )
    parser.add_argument(
        "--test-clean",
        default=(
            "results/base_strict_exploratory/"
            "h_alltrain_test_clean_predictions.txt"
        ),
    )
    parser.add_argument(
        "--test-other",
        default=(
            "results/base_strict_exploratory/"
            "h_alltrain_test_other_predictions.txt"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/base_strict_exploratory/"
            "h_alltrain_wer_summary.csv"
        ),
    )
    return parser.parse_args()


def prediction_wer(path: Path) -> float:
    references, hypotheses = read_ref_hyp_pairs(str(path))
    references = [normalize_transcript(text) for text in references]
    hypotheses = [normalize_transcript(text) for text in hypotheses]
    return wer(references, hypotheses)


def main() -> None:
    args = parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    decoder = json.loads(Path(args.decoder).read_text(encoding="utf-8"))
    train_shards = {str(Path(path).resolve()) for path in metadata["train_shards"]}
    if train_shards != EXPECTED_TRAIN_SHARDS:
        raise ValueError("H all-train metadata does not contain exactly five shards.")
    if metadata.get("main_source_checkpoint") != "facebook/wav2vec2-base":
        raise ValueError("H all-train model has invalid source provenance.")
    if metadata.get("test_splits_used_during_training") is not False:
        raise ValueError("H all-train metadata indicates test use during training.")
    expected_decoder = {
        "decoding_method": "beam_lm",
        "beam_width": 50,
        "alpha": 0.3,
        "beta": 1.5,
    }
    for key, expected_value in expected_decoder.items():
        if decoder.get(key) != expected_value:
            raise ValueError(f"Preserved decoder mismatch for {key}.")

    test_clean_wer = prediction_wer(Path(args.test_clean))
    test_other_wer = prediction_wer(Path(args.test_other))
    row = {
        "candidate": "H_alltrain_fixed_recipe",
        "main_source_checkpoint": "facebook/wav2vec2-base",
        "train_shard_count": 5,
        "stage1_head_only_epochs": 10,
        "stage2_encoder_epochs": 40,
        "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
        "decoder_type": decoder["decoding_method"],
        "beam_width": decoder["beam_width"],
        "alpha": decoder["alpha"],
        "beta": decoder["beta"],
        "language_model_path": decoder["language_model_path"],
        "test_clean_wer": test_clean_wer,
        "test_other_wer": test_other_wer,
        "main_test_clean_wer": MAIN_TEST_CLEAN_WER,
        "main_test_other_wer": MAIN_TEST_OTHER_WER,
        "test_clean_change_vs_main": test_clean_wer - MAIN_TEST_CLEAN_WER,
        "test_other_change_vs_main": test_other_wer - MAIN_TEST_OTHER_WER,
        "test_clean_improved_over_main": test_clean_wer < MAIN_TEST_CLEAN_WER,
        "test_other_improved_over_main": test_other_wer < MAIN_TEST_OTHER_WER,
        "checkpoint_path": "outputs/base_strict_exploratory/h_alltrain/best_model",
        "test_splits_used_for_tuning": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"test-clean WER: {test_clean_wer:.9f}")
    print(f"test-other WER: {test_other_wer:.9f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
