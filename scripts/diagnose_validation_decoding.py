#!/usr/bin/env python3
"""Measure greedy CTC blank-collapse diagnostics on validation shards only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from jiwer import wer
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from experiment_guard import (
    is_supervised_librispeech_checkpoint,
    validate_checkpoint_role,
)
from text_normalization import normalize_transcript
from wav2vec_inference import build_collate_fn


FIELDNAMES = [
    "checkpoint",
    "validation_shards",
    "sample_count",
    "validation_wer",
    "average_hypothesis_character_length",
    "average_hypothesis_word_count",
    "average_hypothesis_word_length",
    "empty_hypothesis_rate",
    "argmax_token_count",
    "nonblank_token_rate",
    "blank_token_rate",
    "blank_token_id",
    "top_10_predicted_token_ids",
    "top_10_tokenizer_tokens",
    "top_10_decoded_tokens",
    "top_10_token_counts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Local Wav2Vec2 CTC checkpoint. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--validation_shards", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--examples_output", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--use_attention_mask_for_forward",
        action="store_true",
        help=(
            "Pass the padding mask to the model. Leave disabled to reproduce "
            "strict base training with --no_attention_mask_for_loss."
        ),
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def reject_disallowed_inputs(checkpoints: List[str], validation_shards: str) -> None:
    """Keep this diagnostic strictly on base-derived checkpoints and train shards."""
    if (
        "test-clean" in validation_shards.lower()
        or "test-other" in validation_shards.lower()
    ):
        raise ValueError("Validation diagnostics must use train shards only.")
    for checkpoint in checkpoints:
        if is_supervised_librispeech_checkpoint(checkpoint):
            raise ValueError(
                f"Supervised/960h checkpoint is forbidden for this diagnostic: {checkpoint}"
            )
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        if not (checkpoint_path / "config.json").is_file():
            raise FileNotFoundError(f"Missing checkpoint config.json: {checkpoint}")

        validate_checkpoint_role(checkpoint, "main")


def token_text(processor, token_id: int) -> str:
    """Decode one token without CTC repeat grouping."""
    return processor.decode(
        [token_id],
        group_tokens=False,
        skip_special_tokens=False,
    )


def summarize_checkpoint(
    checkpoint: str,
    validation_shards: str,
    batch_size: int,
    max_samples: int | None,
    device: torch.device,
    use_attention_mask_for_forward: bool,
    local_files_only: bool,
) -> tuple[Dict, List[tuple[str, str]]]:
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    model = AutoModelForCTC.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    blank_token_id = processor.tokenizer.pad_token_id
    if blank_token_id is None:
        raise ValueError(f"Checkpoint has no pad/CTC blank token ID: {checkpoint}")
    if model.config.pad_token_id != blank_token_id:
        raise ValueError(
            "Model and tokenizer disagree about the CTC blank token: "
            f"model={model.config.pad_token_id}, tokenizer={blank_token_id}"
        )

    dataset = sample_util.make_dataset(
        validation_shards,
        processor,
        do_tokenization=False,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=build_collate_fn(processor),
    )

    references: List[str] = []
    hypotheses: List[str] = []
    token_counts: Counter[int] = Counter()

    with torch.inference_mode():
        for batch in loader:
            batch_references = [
                normalize_transcript(text) for text in batch.pop("references")
            ]
            attention_mask = batch["attention_mask"]
            raw_input_lengths = attention_mask.sum(dim=-1).to(torch.long)
            model_inputs = {"input_values": batch["input_values"].to(device)}
            if use_attention_mask_for_forward:
                model_inputs["attention_mask"] = attention_mask.to(device)

            logits = model(**model_inputs).logits
            output_lengths = model._get_feat_extract_output_lengths(
                raw_input_lengths.to(device)
            )
            predicted_ids = logits.argmax(dim=-1).cpu()
            output_lengths = output_lengths.cpu().tolist()

            batch_hypotheses = []
            for sample_ids, output_length in zip(predicted_ids, output_lengths):
                valid_ids = sample_ids[:output_length].tolist()
                token_counts.update(valid_ids)
                batch_hypotheses.append(
                    normalize_transcript(processor.decode(valid_ids))
                )

            references.extend(batch_references)
            hypotheses.extend(batch_hypotheses)

    if not references:
        raise RuntimeError(f"No validation samples loaded from {validation_shards}")

    character_lengths = [len(text) for text in hypotheses]
    word_counts = [len(text.split()) for text in hypotheses]
    hypothesis_words = [word for text in hypotheses for word in text.split()]
    empty_count = sum(not text for text in hypotheses)
    total_argmax_tokens = sum(token_counts.values())
    blank_count = token_counts[blank_token_id]
    nonblank_count = total_argmax_tokens - blank_count
    top_tokens = token_counts.most_common(10)
    top_ids = [token_id for token_id, _ in top_tokens]

    def mean(values: List[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    row = {
        "checkpoint": checkpoint,
        "validation_shards": ",".join(sample_util.find_shards(validation_shards)),
        "sample_count": len(references),
        "validation_wer": wer(references, hypotheses),
        "average_hypothesis_character_length": mean(character_lengths),
        "average_hypothesis_word_count": mean(word_counts),
        "average_hypothesis_word_length": mean(
            [len(word) for word in hypothesis_words]
        ),
        "empty_hypothesis_rate": empty_count / len(hypotheses),
        "argmax_token_count": total_argmax_tokens,
        "nonblank_token_rate": (
            nonblank_count / total_argmax_tokens if total_argmax_tokens else 0.0
        ),
        "blank_token_rate": (
            blank_count / total_argmax_tokens if total_argmax_tokens else 0.0
        ),
        "blank_token_id": blank_token_id,
        "top_10_predicted_token_ids": json.dumps(top_ids),
        "top_10_tokenizer_tokens": json.dumps(
            processor.tokenizer.convert_ids_to_tokens(top_ids),
            ensure_ascii=False,
        ),
        "top_10_decoded_tokens": json.dumps(
            [token_text(processor, token_id) for token_id in top_ids],
            ensure_ascii=False,
        ),
        "top_10_token_counts": json.dumps(
            [count for _, count in top_tokens]
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, list(zip(references, hypotheses))


def write_outputs(
    rows: List[Dict],
    examples_by_checkpoint: List[tuple[str, List[tuple[str, str]]]],
    output_csv: str,
    examples_output: str,
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(examples_output) or ".", exist_ok=True)
    with open(examples_output, "w", encoding="utf-8") as output_file:
        for checkpoint_index, (checkpoint, pairs) in enumerate(
            examples_by_checkpoint
        ):
            if checkpoint_index:
                output_file.write("\n")
            output_file.write(f"CHECKPOINT: {checkpoint}\n")
            for example_index, (reference, hypothesis) in enumerate(
                pairs[:10],
                start=1,
            ):
                output_file.write(f"[{example_index}] REF: {reference}\n")
                output_file.write(
                    f"[{example_index}] HYP: {hypothesis or '<EMPTY>'}\n"
                )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be greater than zero")
    reject_disallowed_inputs(args.checkpoint, args.validation_shards)
    validation_paths = sample_util.find_shards(args.validation_shards)

    if args.dry_run:
        print(f"checkpoints: {args.checkpoint}")
        print(f"validation_shards: {validation_paths}")
        print(f"device: {args.device}")
        print(
            "attention_mask_for_forward: "
            f"{args.use_attention_mask_for_forward}"
        )
        print(f"output_csv: {args.output_csv}")
        print(f"examples_output: {args.examples_output}")
        return

    device = torch.device(args.device)
    rows = []
    examples_by_checkpoint = []
    for checkpoint in args.checkpoint:
        print(f"Diagnosing validation decoding: {checkpoint}", flush=True)
        row, pairs = summarize_checkpoint(
            checkpoint=checkpoint,
            validation_shards=args.validation_shards,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=device,
            use_attention_mask_for_forward=args.use_attention_mask_for_forward,
            local_files_only=args.local_files_only,
        )
        rows.append(row)
        examples_by_checkpoint.append((checkpoint, pairs))
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    write_outputs(
        rows,
        examples_by_checkpoint,
        args.output_csv,
        args.examples_output,
    )
    print(f"Saved diagnostics CSV: {args.output_csv}")
    print(f"Saved REF/HYP examples: {args.examples_output}")


if __name__ == "__main__":
    main()
