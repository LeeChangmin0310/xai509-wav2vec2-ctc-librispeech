"""Compute WER from REF/HYP result files and optionally update a summary CSV."""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

from jiwer import wer

from text_normalization import normalize_transcript


SUMMARY_FIELDS = [
    "experiment",
    "train_setting",
    "decoding_method",
    "learning_rate",
    "freeze_setting",
    "layerwise_lr_decay",
    "beam_width",
    "test_clean_wer",
    "test_other_wer",
    "checkpoint_path",
]


def parse_args():
    """Parse WER evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_files", nargs="+", help="REF/HYP result files")
    parser.add_argument("--experiment_name", help="Experiment name for summary CSV")
    parser.add_argument("--summary_csv", help="CSV file to create or update")
    parser.add_argument("--metadata_json", help="Optional inference metadata JSON")
    parser.add_argument("--train_setting")
    parser.add_argument("--decoding_method", choices=("greedy", "beam"))
    parser.add_argument("--learning_rate")
    parser.add_argument("--freeze_setting")
    parser.add_argument("--layerwise_lr_decay")
    parser.add_argument("--beam_width")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--normalize_text", action="store_true")
    return parser.parse_args()


def read_ref_hyp_pairs(file_path: str) -> Tuple[List[str], List[str]]:
    """Read strict alternating REF/HYP lines from a result file."""
    with open(file_path, "r", encoding="utf-8") as result_file:
        lines = [line.strip() for line in result_file if line.strip()]

    if len(lines) % 2:
        raise ValueError(f"Expected alternating REF/HYP lines in {file_path}")

    references = []
    hypotheses = []
    for index in range(0, len(lines), 2):
        ref_line, hyp_line = lines[index : index + 2]
        if not ref_line.startswith("REF:") or not hyp_line.startswith("HYP:"):
            raise ValueError(f"Malformed REF/HYP pair near line {index + 1}: {file_path}")
        references.append(ref_line[len("REF:") :].strip())
        hypotheses.append(hyp_line[len("HYP:") :].strip())
    return references, hypotheses


def split_name(file_path: str) -> str:
    """Map a standard result filename to its CSV column prefix."""
    filename = os.path.basename(file_path).replace("-", "_")
    if "test_clean" in filename:
        return "test_clean"
    if "test_other" in filename:
        return "test_other"
    raise ValueError(f"Cannot infer test split from filename: {file_path}")


def read_metadata(metadata_json: Optional[str], first_result_file: str) -> Dict:
    """Read inference metadata when available, including older result folders."""
    if metadata_json is None:
        metadata_json = os.path.join(os.path.dirname(first_result_file), "metadata.json")
    if not os.path.exists(metadata_json):
        return {}
    with open(metadata_json, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def update_summary(
    summary_csv: str,
    experiment: str,
    scores: Dict[str, float],
    metadata: Dict,
):
    """Upsert one experiment row in the WER summary CSV."""
    rows = []
    if os.path.exists(summary_csv):
        with open(summary_csv, "r", encoding="utf-8", newline="") as summary_file:
            rows = list(csv.DictReader(summary_file))

    row = next((item for item in rows if item["experiment"] == experiment), None)
    if row is None:
        row = {"experiment": experiment}
        rows.append(row)
    for field in SUMMARY_FIELDS:
        if field in metadata and metadata[field] is not None:
            row[field] = str(metadata[field])
    for split, score in scores.items():
        row[f"{split}_wer"] = f"{score:.6f}"

    parent_dir = os.path.dirname(summary_csv)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(summary_csv, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({field: item.get(field, "") for field in SUMMARY_FIELDS})


def main():
    """Compute WER values and optionally save them."""
    args = parse_args()
    scores = {}
    for result_file in args.result_files:
        references, hypotheses = read_ref_hyp_pairs(result_file)
        if args.normalize_text:
            references = [normalize_transcript(text) for text in references]
            hypotheses = [normalize_transcript(text) for text in hypotheses]
        split = split_name(result_file)
        scores[split] = wer(references, hypotheses)
        print(f"{split} WER: {scores[split]:.4f}")

    if args.summary_csv:
        if not args.experiment_name:
            raise ValueError("--experiment_name is required with --summary_csv")
        metadata = read_metadata(args.metadata_json, args.result_files[0])
        for field in SUMMARY_FIELDS:
            value = getattr(args, field, None)
            if value is not None:
                metadata[field] = value
        update_summary(args.summary_csv, args.experiment_name, scores, metadata)
        print(f"Updated summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
