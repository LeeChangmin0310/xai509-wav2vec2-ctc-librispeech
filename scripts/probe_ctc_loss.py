#!/usr/bin/env python3
"""Probe Wav2Vec2 CTC loss variants on a small training batch."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCTC, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from wav2vec_finetuning import DataCollatorCTCWithPadding, configure_spec_augment


def reset_torch_seed() -> None:
    """Reset stochastic training layers before each comparable forward pass."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def parse_args() -> argparse.Namespace:
    """Parse probe options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_shards", default="data/train")
    parser.add_argument(
        "--model_name_or_path",
        default="facebook/wav2vec2-base-960h",
    )
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num_samples must be greater than zero")
    return args


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Print shape, dtype, and basic floating-point statistics."""
    float_tensor = tensor.detach().float()
    print(f"{name} shape: {tuple(tensor.shape)}")
    print(f"{name} dtype: {tensor.dtype}")
    print(f"{name} min: {float_tensor.min().item()}")
    print(f"{name} max: {float_tensor.max().item()}")
    print(f"{name} mean: {float_tensor.mean().item()}")
    print(f"{name} std: {float_tensor.std().item()}")


def print_forward_variant(
    model,
    batch,
    name: str,
    training: bool,
    apply_spec_augment: bool,
    attention_mask: Optional[torch.Tensor],
    ctc_zero_infinity: bool,
) -> None:
    """Print logits and loss for one independently configured forward variant."""
    try:
        model.train(training)
        configure_spec_augment(model, apply_spec_augment)
        model.config.ctc_zero_infinity = ctc_zero_infinity
        reset_torch_seed()
        with torch.no_grad():
            outputs = model(
                input_values=batch["input_values"],
                attention_mask=attention_mask,
                labels=batch["labels"],
            )
        logits = outputs.logits
        print(f"\n=== {name} ===")
        print(f"model.training: {model.training}")
        print(f"model.config.apply_spec_augment: {model.config.apply_spec_augment}")
        print(f"model.config.ctc_zero_infinity: {model.config.ctc_zero_infinity}")
        print(f"attention_mask passed: {attention_mask is not None}")
        print_tensor_stats("logits", logits)
        print(f"logits contain nan: {torch.isnan(logits).any().item()}")
        print(f"logits contain inf: {torch.isinf(logits).any().item()}")
        print(f"loss: {outputs.loss.detach().cpu().item()}")
    except Exception as error:  # noqa: BLE001 - diagnostics must continue
        print(f"\n=== {name} ===")
        print(f"raised {type(error).__name__}: {error}")


def main() -> None:
    """Load a small training batch and probe CTC loss behavior."""
    args = parse_args()
    print(f"device: {args.device}")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"train_shards: {args.train_shards}")
    print(f"num_samples: {args.num_samples}")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    ).to(args.device)
    dataset = sample_util.make_dataset(
        args.train_shards,
        processor,
        max_samples=args.num_samples,
    )
    samples = list(itertools.islice(iter(dataset), args.num_samples))
    if not samples:
        raise RuntimeError("No training samples were loaded")

    collator = DataCollatorCTCWithPadding(processor=processor)
    batch = {
        key: value.to(args.device)
        for key, value in collator(samples).items()
    }
    input_values = batch["input_values"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]
    valid_labels = labels[labels.ne(-100)]

    print(
        "processor.feature_extractor.return_attention_mask: "
        f"{getattr(processor.feature_extractor, 'return_attention_mask', None)}"
    )
    print(f"model.config.feat_extract_norm: {getattr(model.config, 'feat_extract_norm', None)}")
    print(f"model.config.vocab_size: {model.config.vocab_size}")
    print(f"model.config.pad_token_id: {model.config.pad_token_id}")
    print(
        "model.config.ctc_loss_reduction: "
        f"{getattr(model.config, 'ctc_loss_reduction', None)}"
    )
    print(
        "model.config.ctc_zero_infinity: "
        f"{getattr(model.config, 'ctc_zero_infinity', None)}"
    )
    print(f"model.training: {model.training}")
    print_tensor_stats("input_values", input_values)
    print(f"input_values contain nan: {torch.isnan(input_values).any().item()}")
    print(f"input_values contain inf: {torch.isinf(input_values).any().item()}")
    if attention_mask is None:
        print("attention_mask: None")
    else:
        print(f"attention_mask shape: {tuple(attention_mask.shape)}")
        print(f"attention_mask sum: {attention_mask.sum(dim=-1).detach().cpu().tolist()}")
    print(f"labels shape: {tuple(labels.shape)}")
    print(f"labels dtype: {labels.dtype}")
    print(f"labels min: {labels.min().item()}")
    print(f"labels max: {labels.max().item()}")
    print(f"valid label lengths: {labels.ne(-100).sum(dim=-1).detach().cpu().tolist()}")
    print(f"valid labels min: {valid_labels.min().item()}")
    print(f"valid labels max: {valid_labels.max().item()}")
    print(
        "out-of-range valid label count: "
        f"{valid_labels.ge(model.config.vocab_size).sum().item()}"
    )
    first_labels = labels[0][labels[0].ne(-100)].detach().cpu().tolist()
    print(f"decoded first label: {processor.decode(first_labels, group_tokens=False)!r}")

    raw_input_lengths = (
        attention_mask.sum(dim=-1)
        if attention_mask is not None
        else torch.full(
            (input_values.shape[0],),
            input_values.shape[1],
            device=input_values.device,
            dtype=torch.long,
        )
    )
    logits_lengths = model._get_feat_extract_output_lengths(raw_input_lengths)
    print(f"raw input lengths: {raw_input_lengths.detach().cpu().tolist()}")
    print(f"derived logits lengths: {logits_lengths.detach().cpu().tolist()}")

    original_training = model.training
    original_spec_augment = getattr(model.config, "apply_spec_augment", True)
    original_zero_infinity = getattr(model.config, "ctc_zero_infinity", False)
    try:
        print_forward_variant(
            model,
            batch,
            "eval mode",
            training=False,
            apply_spec_augment=True,
            attention_mask=None,
            ctc_zero_infinity=original_zero_infinity,
        )
        print_forward_variant(
            model,
            batch,
            "train mode with apply_spec_augment=True",
            training=True,
            apply_spec_augment=True,
            attention_mask=None,
            ctc_zero_infinity=original_zero_infinity,
        )
        print_forward_variant(
            model,
            batch,
            "train mode with apply_spec_augment=False",
            training=True,
            apply_spec_augment=False,
            attention_mask=None,
            ctc_zero_infinity=original_zero_infinity,
        )
        print_forward_variant(
            model,
            batch,
            "stable train mode with attention_mask",
            training=True,
            apply_spec_augment=False,
            attention_mask=attention_mask,
            ctc_zero_infinity=original_zero_infinity,
        )
        print_forward_variant(
            model,
            batch,
            "stable train mode with ctc_zero_infinity=True",
            training=True,
            apply_spec_augment=False,
            attention_mask=None,
            ctc_zero_infinity=True,
        )
    finally:
        model.train(original_training)
        configure_spec_augment(model, original_spec_augment)
        model.config.ctc_zero_infinity = original_zero_infinity


if __name__ == "__main__":
    main()
