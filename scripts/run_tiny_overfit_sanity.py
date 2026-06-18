#!/usr/bin/env python3
"""Overfit a tiny train-only subset to validate the CTC training/decoding path."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from jiwer import wer
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from experiment_guard import ALLOWED_MAIN_SOURCE, validate_checkpoint_role
from text_normalization import normalize_transcript
from wav2vec_finetuning import (
    DataCollatorCTCWithPadding,
    configure_spec_augment,
)


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "tiny_subset_wer",
    "empty_hypothesis_rate",
    "blank_token_rate",
    "nonblank_token_rate",
    "average_hypothesis_character_length",
    "average_hypothesis_word_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_shards", default="data/train/shard-000001.tar")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--eval_every_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_spec_augment", action="store_true")
    parser.add_argument("--mask_time_prob", type=float, default=0.05)
    parser.add_argument("--mask_time_length", type=int, default=10)
    parser.add_argument("--mask_time_min_masks", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--stop_wer",
        type=float,
        default=0.05,
        help="Stop after an evaluation reaches this WER or lower.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    normalized_shards = args.train_shards.lower()
    if "test-clean" in normalized_shards or "test-other" in normalized_shards:
        raise ValueError("Tiny overfit must use train shards only.")
    validate_checkpoint_role(
        ALLOWED_MAIN_SOURCE,
        "diagnostic",
        require_base_source=False,
    )
    for name in (
        "num_samples",
        "num_epochs",
        "batch_size",
        "eval_every_epochs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be greater than zero")
    if args.encoder_learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("Learning rates must be greater than zero.")
    if args.head_learning_rate < args.encoder_learning_rate:
        raise ValueError("Head LR must be at least the encoder LR.")
    if not 0.0 <= args.mask_time_prob < 1.0:
        raise ValueError("--mask_time_prob must be in [0, 1).")
    if args.mask_time_length <= 0 or args.mask_time_min_masks < 0:
        raise ValueError("SpecAugment mask lengths/counts must be non-negative.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_tiny_subset(
    model,
    processor,
    samples: List[Dict],
    collator: DataCollatorCTCWithPadding,
    device: torch.device,
    batch_size: int,
) -> tuple[Dict[str, float], List[tuple[str, str]]]:
    loader = DataLoader(samples, batch_size=batch_size, collate_fn=collator)
    references: List[str] = []
    hypotheses: List[str] = []
    token_counts: Counter[int] = Counter()
    blank_token_id = processor.tokenizer.pad_token_id

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            attention_mask = batch["attention_mask"]
            raw_input_lengths = attention_mask.sum(dim=-1).to(torch.long)
            logits = model(input_values=batch["input_values"].to(device)).logits
            output_lengths = model._get_feat_extract_output_lengths(
                raw_input_lengths.to(device)
            )
            predicted_ids = logits.argmax(dim=-1).cpu()

            for label_ids, sample_ids, output_length in zip(
                labels,
                predicted_ids,
                output_lengths.cpu().tolist(),
            ):
                valid_label_ids = label_ids[label_ids.ne(-100)].tolist()
                valid_prediction_ids = sample_ids[:output_length].tolist()
                token_counts.update(valid_prediction_ids)
                references.append(
                    normalize_transcript(
                        processor.decode(
                            valid_label_ids,
                            group_tokens=False,
                        )
                    )
                )
                hypotheses.append(
                    normalize_transcript(
                        processor.decode(valid_prediction_ids)
                    )
                )

    total_tokens = sum(token_counts.values())
    blank_count = token_counts[blank_token_id]
    character_lengths = [len(text) for text in hypotheses]
    word_counts = [len(text.split()) for text in hypotheses]
    metrics = {
        "tiny_subset_wer": wer(references, hypotheses),
        "empty_hypothesis_rate": (
            sum(not text for text in hypotheses) / len(hypotheses)
        ),
        "blank_token_rate": blank_count / total_tokens if total_tokens else 0.0,
        "nonblank_token_rate": (
            (total_tokens - blank_count) / total_tokens if total_tokens else 0.0
        ),
        "average_hypothesis_character_length": (
            sum(character_lengths) / len(character_lengths)
        ),
        "average_hypothesis_word_count": (
            sum(word_counts) / len(word_counts)
        ),
    }
    return metrics, list(zip(references, hypotheses))


def append_examples(
    output_path: Path,
    epoch: int,
    pairs: List[tuple[str, str]],
) -> None:
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(f"EPOCH: {epoch}\n")
        for index, (reference, hypothesis) in enumerate(pairs, start=1):
            output_file.write(f"[{index}] REF: {reference}\n")
            output_file.write(
                f"[{index}] HYP: {hypothesis or '<EMPTY>'}\n"
            )
        output_file.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    examples_path = output_dir / "ref_hyp_examples.txt"
    metadata_path = output_dir / "metadata.json"
    model_dir = output_dir / "final_model"
    examples_path.write_text("", encoding="utf-8")

    processor = AutoProcessor.from_pretrained(
        ALLOWED_MAIN_SOURCE,
        local_files_only=args.local_files_only,
    )
    source_dataset = sample_util.make_dataset(
        args.train_shards,
        processor,
        shuffle=False,
    )
    samples = list(itertools.islice(iter(source_dataset), args.num_samples))
    if len(samples) != args.num_samples:
        raise RuntimeError(
            f"Requested {args.num_samples} samples but loaded {len(samples)}."
        )

    model = AutoModelForCTC.from_pretrained(
        ALLOWED_MAIN_SOURCE,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        local_files_only=args.local_files_only,
    )
    model.config.ctc_zero_infinity = True
    configure_spec_augment(model, args.enable_spec_augment)
    for config in (model.config, model.wav2vec2.config):
        config.mask_time_prob = args.mask_time_prob
        config.mask_time_length = args.mask_time_length
        config.mask_time_min_masks = args.mask_time_min_masks
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()

    encoder_parameters = [
        parameter
        for parameter in model.wav2vec2.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter for parameter in model.lm_head.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
                "group_name": "wav2vec2_encoder",
            },
            {
                "params": head_parameters,
                "lr": args.head_learning_rate,
                "group_name": "ctc_head",
            },
        ],
        weight_decay=0.0,
    )
    device = torch.device(args.device)
    model.to(device)
    collator = DataCollatorCTCWithPadding(processor=processor)

    rows = []
    initial_metrics, initial_pairs = evaluate_tiny_subset(
        model,
        processor,
        samples,
        collator,
        device,
        args.batch_size,
    )
    rows.append({"epoch": 0, "train_loss": "", **initial_metrics})
    append_examples(examples_path, 0, initial_pairs)
    print({"epoch": 0, **initial_metrics}, flush=True)

    completed_epoch = 0
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    for epoch in range(1, args.num_epochs + 1):
        loader = DataLoader(
            samples,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        )
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_values=batch["input_values"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss).all().item():
                raise RuntimeError(
                    f"Non-finite tiny-overfit loss at epoch {epoch}: {loss}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().item())

        completed_epoch = epoch
        if epoch % args.eval_every_epochs != 0 and epoch != args.num_epochs:
            continue
        metrics, pairs = evaluate_tiny_subset(
            model,
            processor,
            samples,
            collator,
            device,
            args.batch_size,
        )
        row = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            **metrics,
        }
        rows.append(row)
        append_examples(examples_path, epoch, pairs)
        print(row, flush=True)
        if metrics["tiny_subset_wer"] <= args.stop_wer:
            break

    with history_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    metadata = {
        "experiment_role": "diagnostic",
        "main_source_checkpoint": ALLOWED_MAIN_SOURCE,
        "train_shards": sample_util.find_shards(args.train_shards),
        "test_splits_used": False,
        "same_samples_used_for_train_and_eval": True,
        "num_samples": args.num_samples,
        "requested_epochs": args.num_epochs,
        "completed_epoch": completed_epoch,
        "batch_size": args.batch_size,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "apply_spec_augment": args.enable_spec_augment,
        "mask_time_prob": args.mask_time_prob,
        "mask_time_length": args.mask_time_length,
        "mask_time_min_masks": args.mask_time_min_masks,
        "ctc_zero_infinity": True,
        "attention_mask_passed_to_model": False,
        "history_csv": str(history_path),
        "examples_file": str(examples_path),
        "final_metrics": rows[-1],
        "seed": args.seed,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved tiny-overfit diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
