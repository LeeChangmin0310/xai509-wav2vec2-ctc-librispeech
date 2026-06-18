#!/usr/bin/env python3
"""Verify a train-mode SpecAugment CTC batch is finite before long training."""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCTC, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from experiment_guard import ALLOWED_MAIN_SOURCE, validate_checkpoint_role
from wav2vec_finetuning import (
    DataCollatorCTCWithPadding,
    configure_spec_augment,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_shards", default="data/train/shard-000000.tar")
    parser.add_argument("--model_name_or_path", default=ALLOWED_MAIN_SOURCE)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_report",
        default="reports/specaugment_finite_loss_check.md",
    )
    parser.add_argument(
        "--output_json",
        default="results/base_strict_cv/specaugment_finite_loss_check.json",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def tensor_finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def run_variant(model, batch, name, mask_time_prob, mask_time_length):
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = mask_time_prob
    model.config.mask_time_length = mask_time_length
    model.config.ctc_zero_infinity = True
    if hasattr(model, "wav2vec2"):
        model.wav2vec2.config.apply_spec_augment = True
        model.wav2vec2.config.mask_time_prob = mask_time_prob
        model.wav2vec2.config.mask_time_length = mask_time_length
    model.train()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        outputs = model(
            input_values=batch["input_values"],
            attention_mask=None,
            labels=batch["labels"],
        )
    return {
        "name": name,
        "mask_time_prob": mask_time_prob,
        "mask_time_length": mask_time_length,
        "logits_finite": tensor_finite(outputs.logits),
        "loss_finite": outputs.loss is not None and tensor_finite(outputs.loss),
        "loss": None if outputs.loss is None else outputs.loss.item(),
    }


def main():
    args = parse_args()
    validate_checkpoint_role(
        args.model_name_or_path,
        "main",
        require_base_source=True,
    )
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        local_files_only=args.local_files_only,
    ).to(device)
    configure_spec_augment(model, True)
    model.config.ctc_zero_infinity = True

    dataset = sample_util.make_dataset(
        args.train_shards,
        processor,
        max_samples=args.num_samples,
    )
    samples = list(itertools.islice(iter(dataset), args.num_samples))
    batch = DataCollatorCTCWithPadding(processor)(samples)
    batch = {key: value.to(device) for key, value in batch.items()}

    labels = batch["labels"]
    attention_mask = batch.get("attention_mask")
    valid_label_mask = labels.ne(-100)
    raw_lengths = torch.full(
        (batch["input_values"].shape[0],),
        batch["input_values"].shape[1],
        device=device,
        dtype=torch.long,
    )
    logit_lengths = model._get_feat_extract_output_lengths(raw_lengths)
    target_lengths = valid_label_mask.sum(dim=-1)

    checks = {
        "checkpoint": args.model_name_or_path,
        "checkpoint_role": "main",
        "device": str(device),
        "input_shape": list(batch["input_values"].shape),
        "input_values_finite": tensor_finite(batch["input_values"]),
        "labels_shape": list(labels.shape),
        "labels_valid": bool(((labels >= 0) | labels.eq(-100)).all().item()),
        "label_padding_uses_minus_100": bool(labels.eq(-100).any().item()),
        "target_lengths": target_lengths.cpu().tolist(),
        "logit_lengths": logit_lengths.cpu().tolist(),
        "targets_fit_inputs": bool((target_lengths <= logit_lengths).all().item()),
        "processor_return_attention_mask": getattr(
            processor.feature_extractor, "return_attention_mask", None
        ),
        "attention_mask_present": attention_mask is not None,
        "attention_mask_passed_for_loss": False,
        "ctc_zero_infinity": bool(model.config.ctc_zero_infinity),
        "apply_spec_augment": bool(model.config.apply_spec_augment),
    }
    variants = [
        run_variant(model, batch, "default SpecAugment", 0.05, 10),
    ]
    if not (
        variants[0]["logits_finite"] and variants[0]["loss_finite"]
    ):
        variants.append(
            run_variant(model, batch, "lighter SpecAugment", 0.03, 5)
        )

    selected = next(
        (
            variant
            for variant in variants
            if variant["logits_finite"] and variant["loss_finite"]
        ),
        None,
    )
    checks["variants"] = variants
    checks["selected_variant"] = None if selected is None else selected["name"]
    checks["status"] = "PASS" if selected is not None else "FAIL"

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(checks, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    lines = [
        "# SpecAugment Finite-Loss Check",
        "",
        f"- Status: **{checks['status']}**",
        f"- Checkpoint: `{args.model_name_or_path}`",
        "- Experiment role: `main`",
        f"- Device: `{device}`",
        f"- Input shape: `{tuple(checks['input_shape'])}`",
        f"- Labels shape: `{tuple(checks['labels_shape'])}`",
        f"- Input values finite: `{checks['input_values_finite']}`",
        f"- Label IDs/padding valid: `{checks['labels_valid']}`",
        f"- Label padding includes `-100`: `{checks['label_padding_uses_minus_100']}`",
        f"- Target lengths: `{checks['target_lengths']}`",
        f"- CTC input lengths: `{checks['logit_lengths']}`",
        f"- Targets fit CTC inputs: `{checks['targets_fit_inputs']}`",
        f"- Processor return_attention_mask: `{checks['processor_return_attention_mask']}`",
        "- Attention mask passed to loss forward: `False`",
        "- `ctc_zero_infinity`: `True`",
        "- SpecAugment remained enabled for every attempted variant.",
        "",
        "## Forward Variants",
        "",
        "| Variant | mask_time_prob | mask_time_length | Logits finite | Loss finite | Loss |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for variant in variants:
        lines.append(
            f"| {variant['name']} | {variant['mask_time_prob']} | "
            f"{variant['mask_time_length']} | {variant['logits_finite']} | "
            f"{variant['loss_finite']} | {variant['loss']} |"
        )
    lines.extend(
        [
            "",
            "Full training is permitted only when this check passes. If the default "
            "variant fails, the script tries lighter masking while keeping "
            "SpecAugment enabled; it never silently disables SpecAugment.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))
    print(json.dumps(checks, indent=2, sort_keys=True))
    if selected is None:
        raise RuntimeError("No finite SpecAugment-on forward variant was found")


if __name__ == "__main__":
    main()
