#!/usr/bin/env python3
"""Generate the strict base text-normalization audit from validation data."""

import argparse
import os
import sys
import tarfile
from pathlib import Path

from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from evaluate_wer import read_ref_hyp_pairs
from text_normalization import normalize_transcript


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation_shards",
        default="data/train/shard-000004.tar",
    )
    parser.add_argument(
        "--validation_predictions",
        default="results/base_strict_final/validation_best_predictions.txt",
    )
    parser.add_argument(
        "--output_report",
        default="reports/text_normalization_audit.md",
    )
    parser.add_argument(
        "--processor",
        default="facebook/wav2vec2-base",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def read_raw_transcripts(shard_spec):
    transcripts = []
    for shard_path in sample_util.find_shards(shard_spec):
        with tarfile.open(shard_path, "r:*") as shard:
            for member in shard.getmembers():
                if member.isfile() and member.name.endswith(".text"):
                    transcripts.append(
                        shard.extractfile(member).read().decode("utf-8").strip()
                    )
    return transcripts


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.processor,
        local_files_only=args.local_files_only,
    )
    tokenizer = processor.tokenizer
    raw_transcripts = read_raw_transcripts(args.validation_shards)
    normalized = [normalize_transcript(text) for text in raw_transcripts]
    characters = sorted(set("".join(raw_transcripts)))

    lines = [
        "# Text Normalization Audit",
        "",
        "## Data Scope",
        "",
        f"- Validation shards: `{args.validation_shards}`",
        "- Test-clean and test-other are not used for this audit or tuning.",
        f"- Validation transcript count: `{len(raw_transcripts)}`",
        "",
        "## Tokenizer and CTC Conventions",
        "",
        f"- Processor: `{args.processor}`",
        f"- Transcript casing: uppercase",
        f"- Observed raw characters: `{''.join(characters)}`",
        "- Punctuation: removed except ASCII apostrophe.",
        "- Curly apostrophes/backticks: converted to ASCII apostrophe.",
        f"- Word delimiter token: `{tokenizer.word_delimiter_token}`",
        f"- Blank/pad token: `{tokenizer.pad_token}` with ID `{tokenizer.pad_token_id}`",
        "- Label padding: `-100` in the CTC data collator.",
        "- REF/HYP WER normalization: uppercase, allowed characters `A-Z`, "
        "apostrophe, and spaces; repeated whitespace collapsed.",
        "",
        "## Validation Transcript Examples",
        "",
        "| # | Raw | Normalized |",
        "| ---: | --- | --- |",
    ]
    for index, (raw, clean) in enumerate(
        list(zip(raw_transcripts, normalized))[:10],
        start=1,
    ):
        lines.append(f"| {index} | {raw} | {clean} |")

    lines.extend(["", "## Validation REF/HYP Examples Before WER", ""])
    if os.path.exists(args.validation_predictions):
        references, hypotheses = read_ref_hyp_pairs(args.validation_predictions)
        lines.extend(
            [
                "| # | REF | HYP |",
                "| ---: | --- | --- |",
            ]
        )
        for index, (reference, hypothesis) in enumerate(
            list(zip(references, hypotheses))[:10],
            start=1,
        ):
            lines.append(
                f"| {index} | {normalize_transcript(reference)} | "
                f"{normalize_transcript(hypothesis)} |"
            )
    else:
        lines.append(
            f"Pending decoder tuning: `{args.validation_predictions}` does not exist."
        )

    lines.extend(
        [
            "",
            "All decoder hyperparameters are selected from these validation "
            "references and hypotheses only. Test references are excluded from "
            "normalization decisions and tuning.",
            "",
        ]
    )
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
