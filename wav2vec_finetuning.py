"""Fine-tune Wav2Vec 2.0 with Hugging Face or project CTC loss."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import csv
import itertools
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
import torch
from jiwer import wer
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import ctc_loss
import sample_util
from experiment_guard import (
    ALLOWED_MAIN_SOURCE,
    validate_checkpoint_role,
    write_checkpoint_provenance,
)


def parse_args():
    """Parse fine-tuning arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_shards", default="data/train")
    parser.add_argument(
        "--eval_shards",
        help="Optional validation shards for Trainer evaluation/checkpoint selection.",
    )
    parser.add_argument("--test_clean_shards", default="data/test-clean")
    parser.add_argument("--test_other_shards", default="data/test-other")
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-base")
    parser.add_argument("--output_dir", default="outputs/baseline_lr1e-4")
    parser.add_argument(
        "--experiment_role",
        choices=("main", "positive_control_only", "diagnostic"),
        default="main",
        help="Main runs enforce unsupervised-base provenance and strict data isolation.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps. Values <= 0 use epochs.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        help="Limit validation samples for smoke tests.",
    )
    parser.add_argument("--freeze_feature_encoder", action="store_true")
    parser.add_argument(
        "--freeze_wav2vec2",
        action="store_true",
        help="Freeze the complete Wav2Vec2 backbone and train only the CTC head.",
    )
    parser.add_argument("--freeze_n_layers", type=int, default=0)
    parser.add_argument(
        "--layerwise_lr_decay",
        action="store_true",
        help="Use lower learning rates for lower Wav2Vec2 encoder layers.",
    )
    parser.add_argument(
        "--layerwise_lr_decay_rate",
        type=float,
        default=0.9,
        help="Multiplier applied while moving from upper to lower encoder layers.",
    )
    parser.add_argument(
        "--head_learning_rate",
        type=float,
        help="Learning rate for the CTC head in layer-wise decay mode.",
    )
    parser.add_argument(
        "--feature_extractor_learning_rate",
        type=float,
        help="Optional feature extractor LR. Omit it to freeze the extractor.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument(
        "--early_stopping_start_epoch",
        type=float,
        default=0.0,
        help="Do not count early-stopping patience before this epoch.",
    )
    parser.add_argument(
        "--eval_delay",
        type=float,
        default=0.0,
        help="Delay evaluation/early-stopping checks by this many epochs or steps.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    load_best_group = parser.add_mutually_exclusive_group()
    load_best_group.add_argument(
        "--load_best_model_at_end",
        dest="load_best_model_at_end",
        action="store_true",
    )
    load_best_group.add_argument(
        "--no_load_best_model_at_end",
        dest="load_best_model_at_end",
        action="store_false",
        help="Keep the last model, used for a fixed-length head-warmup stage.",
    )
    parser.set_defaults(load_best_model_at_end=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Checkpoint path to resume, or 'latest' to use Trainer discovery.",
    )
    parser.add_argument("--final_model_subdir", default="final_model")
    parser.add_argument("--training_log_csv")
    parser.add_argument("--validation_history_csv")
    parser.add_argument("--run_metadata_json")
    parser.add_argument(
        "--finite_loss_check_samples",
        type=int,
        default=2,
        help="Number of samples in the pre-training finite-loss gate.",
    )
    parser.add_argument(
        "--loss_impl",
        choices=("hf", "custom"),
        default="hf",
        help="CTC loss implementation. Hugging Face model loss is the default.",
    )
    parser.add_argument(
        "--debug_first_batch",
        action="store_true",
        help="Print shapes, label lengths, and decoded text for the first batch.",
    )
    parser.add_argument(
        "--ctc_zero_infinity",
        action="store_true",
        help="Replace infinite Hugging Face CTC losses with zero.",
    )
    spec_augment_group = parser.add_mutually_exclusive_group()
    spec_augment_group.add_argument(
        "--disable_spec_augment",
        dest="apply_spec_augment",
        action="store_false",
        help="Disable SpecAugment during diagnostic or positive-control runs.",
    )
    spec_augment_group.add_argument(
        "--enable_spec_augment",
        dest="apply_spec_augment",
        action="store_true",
        help="Enable SpecAugment during fine-tuning.",
    )
    attention_mask_group = parser.add_mutually_exclusive_group()
    attention_mask_group.add_argument(
        "--use_attention_mask_for_loss",
        dest="use_attention_mask_for_loss",
        action="store_true",
        help="Always pass the collator attention mask during the loss forward pass.",
    )
    attention_mask_group.add_argument(
        "--no_attention_mask_for_loss",
        dest="use_attention_mask_for_loss",
        action="store_false",
        help="Never pass an attention mask during the loss forward pass.",
    )
    parser.set_defaults(apply_spec_augment=None, use_attention_mask_for_loss=None)
    parser.add_argument("--mask_time_prob", type=float)
    parser.add_argument("--mask_time_length", type=int)
    parser.add_argument("--mask_time_min_masks", type=int)
    parser.add_argument("--mask_feature_prob", type=float)
    parser.add_argument("--mask_feature_length", type=int)
    parser.add_argument(
        "--blank_bias_init",
        type=float,
        help="Diagnostic initialization for the CTC blank-token output bias.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load cached model and processor files without network access.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate shard paths and print the plan without loading a model.",
    )
    return parser.parse_args()


def resolve_spec_augment(args) -> bool:
    """Use SpecAugment by default for main runs and preserve diagnostic control."""
    if args.apply_spec_augment is not None:
        return args.apply_spec_augment
    return args.experiment_role == "main"


def validate_strict_main_args(args) -> None:
    """Enforce checkpoint provenance and train-only hyperparameter selection."""
    validate_checkpoint_role(
        args.model_name_or_path,
        args.experiment_role,
        require_base_source=False,
    )
    if args.experiment_role != "main":
        return

    if args.eval_shards is None:
        raise ValueError("--eval_shards is required for a main experiment.")
    for name, shard_spec in (
        ("train_shards", args.train_shards),
        ("eval_shards", args.eval_shards),
    ):
        normalized = shard_spec.lower()
        if "test-clean" in normalized or "test-other" in normalized:
            raise ValueError(
                f"--{name} must contain train shards only for a main experiment."
            )
    if args.loss_impl != "hf":
        raise ValueError("Main experiments require --loss_impl hf.")
    if not args.ctc_zero_infinity:
        raise ValueError("Main experiments require --ctc_zero_infinity.")
    if not args.apply_spec_augment:
        raise ValueError(
            "Main experiments require SpecAugment ON. Use --enable_spec_augment."
        )
    if args.fp16:
        raise ValueError("Strict main training starts in fp32; do not pass --fp16.")
    if args.finite_loss_check_samples <= 0:
        raise ValueError("Main experiments require a finite-loss preflight batch.")
    if args.load_best_model_at_end and args.early_stopping_patience <= 0:
        raise ValueError("Main experiments require positive early-stopping patience.")
    if not args.load_best_model_at_end and args.early_stopping_patience > 0:
        raise ValueError(
            "Early stopping requires --load_best_model_at_end. Use patience 0 "
            "for a fixed-length warmup stage."
        )
    if args.early_stopping_start_epoch < 0:
        raise ValueError("--early_stopping_start_epoch must be non-negative.")
    if args.eval_delay < 0:
        raise ValueError("--eval_delay must be non-negative.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in the interval [0, 1).")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max_grad_norm must be greater than zero.")


def configure_spec_augment_parameters(model, args) -> None:
    """Apply optional lighter SpecAugment settings without disabling masking."""
    for name in (
        "mask_time_prob",
        "mask_time_length",
        "mask_time_min_masks",
        "mask_feature_prob",
        "mask_feature_length",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(model.config, name, value)
            if hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "config"):
                setattr(model.wav2vec2.config, name, value)


def write_log_history_csv(log_history: List[Dict], output_path: Optional[str]) -> None:
    """Write heterogeneous Trainer log records to a CSV file."""
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = sorted({key for record in log_history for key in record})
    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in log_history:
            writer.writerow(record)


def write_validation_history_csv(
    log_history: List[Dict], output_path: Optional[str]
) -> None:
    """Write only validation records, including WER, loss, epoch, and step."""
    if not output_path:
        return
    validation_records = [
        record for record in log_history if "eval_wer" in record
    ]
    write_log_history_csv(validation_records, output_path)


def run_finite_loss_preflight(
    model,
    processor,
    train_dataset,
    args,
    use_attention_mask_for_loss: bool,
) -> Dict:
    """Run one train-mode SpecAugment forward and reject non-finite CTC inputs."""
    samples = list(
        itertools.islice(iter(train_dataset), args.finite_loss_check_samples)
    )
    if not samples:
        raise RuntimeError("Finite-loss preflight could not load a training sample.")

    collator = DataCollatorCTCWithPadding(processor=processor)
    batch = collator(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = batch["labels"]
    attention_mask = (
        batch.get("attention_mask") if use_attention_mask_for_loss else None
    )

    if not torch.isfinite(batch["input_values"]).all().item():
        raise RuntimeError("Finite-loss preflight found non-finite input_values.")
    valid_label_mask = labels.ne(-100)
    if not ((labels >= 0) | labels.eq(-100)).all().item():
        raise RuntimeError("Label padding must use -100 exclusively.")
    valid_labels = labels[valid_label_mask]
    if valid_labels.numel() == 0:
        raise RuntimeError("Finite-loss preflight found an empty target batch.")
    if valid_labels.max().item() >= model.config.vocab_size:
        raise RuntimeError("Finite-loss preflight found an out-of-vocabulary label ID.")

    raw_input_lengths = (
        attention_mask.sum(dim=-1)
        if attention_mask is not None
        else torch.full(
            (batch["input_values"].shape[0],),
            batch["input_values"].shape[1],
            device=device,
            dtype=torch.long,
        )
    )
    logit_lengths = model._get_feat_extract_output_lengths(raw_input_lengths)
    target_lengths = valid_label_mask.sum(dim=-1)
    if torch.any(target_lengths > logit_lengths).item():
        raise RuntimeError(
            "Finite-loss preflight found a target longer than its CTC input."
        )

    model.train()
    with torch.no_grad():
        outputs = model(
            input_values=batch["input_values"],
            attention_mask=attention_mask,
            labels=labels,
        )
    logits_finite = torch.isfinite(outputs.logits).all().item()
    loss_finite = outputs.loss is not None and torch.isfinite(outputs.loss).all().item()
    details = {
        "status": "pass" if logits_finite and loss_finite else "fail",
        "device": str(device),
        "model_name_or_path": args.model_name_or_path,
        "experiment_role": args.experiment_role,
        "apply_spec_augment": bool(model.config.apply_spec_augment),
        "mask_time_prob": getattr(model.config, "mask_time_prob", None),
        "mask_time_length": getattr(model.config, "mask_time_length", None),
        "mask_feature_prob": getattr(model.config, "mask_feature_prob", None),
        "mask_feature_length": getattr(model.config, "mask_feature_length", None),
        "ctc_zero_infinity": bool(model.config.ctc_zero_infinity),
        "attention_mask_passed": attention_mask is not None,
        "input_shape": list(batch["input_values"].shape),
        "label_shape": list(labels.shape),
        "target_lengths": target_lengths.detach().cpu().tolist(),
        "logit_lengths": logit_lengths.detach().cpu().tolist(),
        "label_padding_uses_minus_100": True,
        "inputs_finite": True,
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "loss": None if outputs.loss is None else outputs.loss.detach().cpu().item(),
    }
    report_path = os.path.join(args.output_dir, "finite_loss_check.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(details, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    if not logits_finite or not loss_finite:
        raise RuntimeError(
            "SpecAugment-on finite-loss preflight failed. Full training was stopped; "
            f"see {report_path}."
        )
    print(f"SpecAugment finite-loss preflight passed: {details}")
    model.zero_grad(set_to_none=True)
    return details


def build_compute_metrics(processor):
    """Create WER and blank-collapse metrics for each validation evaluation."""

    def compute_metrics(pred) -> Dict[str, float]:
        if isinstance(pred.predictions, (tuple, list)):
            logits, logit_lengths = pred.predictions
            logit_lengths = np.asarray(logit_lengths).reshape(-1).astype(int)
        else:
            logits = pred.predictions
            logit_lengths = np.full(logits.shape[0], logits.shape[1], dtype=int)

        pred_ids = np.argmax(logits, axis=-1)
        references = []
        predictions = []
        token_counts: Counter[int] = Counter()
        for sample_ids, sample_length, sample_labels in zip(
            pred_ids,
            logit_lengths,
            pred.label_ids,
        ):
            valid_pred_ids = sample_ids[:sample_length].tolist()
            valid_label_ids = sample_labels[sample_labels != -100].tolist()
            token_counts.update(valid_pred_ids)
            predictions.append(processor.decode(valid_pred_ids))
            references.append(
                processor.decode(valid_label_ids, group_tokens=False)
            )

        blank_token_id = processor.tokenizer.pad_token_id
        total_tokens = sum(token_counts.values())
        blank_count = token_counts[blank_token_id]
        character_lengths = [len(text) for text in predictions]
        word_counts = [len(text.split()) for text in predictions]
        return {
            "wer": wer(references, predictions),
            "empty_hypothesis_rate": (
                sum(not text.strip() for text in predictions) / len(predictions)
            ),
            "blank_token_rate": (
                blank_count / total_tokens if total_tokens else 0.0
            ),
            "nonblank_token_rate": (
                (total_tokens - blank_count) / total_tokens
                if total_tokens
                else 0.0
            ),
            "average_hypothesis_character_length": (
                sum(character_lengths) / len(character_lengths)
            ),
            "average_hypothesis_word_count": (
                sum(word_counts) / len(word_counts)
            ),
        }

    return compute_metrics


@dataclass
class DataCollatorCTCWithPadding:
    """Dynamically pad audio inputs and target token sequences."""

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return batch


class MyCtcTrainer(Trainer):
    """Trainer with Hugging Face CTC loss and an optional custom path."""

    def __init__(
        self,
        *args,
        loss_impl="hf",
        debug_first_batch=False,
        debug_processor=None,
        use_attention_mask_for_loss=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_impl = loss_impl
        self.debug_first_batch = debug_first_batch
        self.debug_processor = debug_processor
        self.use_attention_mask_for_loss = use_attention_mask_for_loss
        self._logged_first_batch = False

    def _log_first_batch(self, inputs, labels, outputs):
        if not self.debug_first_batch or self._logged_first_batch:
            return

        attention_mask = inputs.get("attention_mask")
        label_lengths = labels.ne(-100).sum(dim=-1).detach().cpu().tolist()
        print("Debug first batch:")
        print(f"  input_values shape: {tuple(inputs['input_values'].shape)}")
        if attention_mask is None:
            print("  attention_mask shape: None")
        else:
            print(f"  attention_mask shape: {tuple(attention_mask.shape)}")
        print(
            "  attention_mask passed during loss forward: "
            f"{self.use_attention_mask_for_loss}"
        )
        print(f"  labels shape: {tuple(labels.shape)}")
        print(f"  valid label tokens per sample: {label_lengths}")

        if self.debug_processor is not None and labels.shape[0] > 0:
            first_labels = labels[0][labels[0].ne(-100)].detach().cpu().tolist()
            first_prediction = outputs.logits[0].argmax(dim=-1).detach().cpu().tolist()
            print(
                "  first decoded label text: "
                f"{self.debug_processor.decode(first_labels, group_tokens=False)!r}"
            )
            print(
                "  first prediction text before training: "
                f"{self.debug_processor.decode(first_prediction)!r}"
            )
        self._logged_first_batch = True

    def _compute_custom_loss(self, model, inputs, labels, outputs, attention_mask):
        if attention_mask is None:
            input_lengths = torch.full(
                (inputs["input_values"].shape[0],),
                inputs["input_values"].shape[1],
                device=inputs["input_values"].device,
                dtype=torch.long,
            )
        else:
            input_lengths = attention_mask.sum(dim=-1).to(torch.long)

        actual_model = model.module if hasattr(model, "module") else model
        logits_lengths = actual_model._get_feat_extract_output_lengths(input_lengths)
        logits = outputs.logits.float()
        target_lengths = labels.ge(0).sum(dim=-1).to(torch.long)
        return ctc_loss.CtcLoss.apply(
            labels,
            target_lengths,
            logits.log_softmax(dim=-1),
            logits_lengths,
        ).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        attention_mask = (
            inputs.get("attention_mask") if self.use_attention_mask_for_loss else None
        )

        if self.loss_impl == "hf":
            outputs = model(
                input_values=inputs["input_values"],
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        else:
            outputs = model(
                input_values=inputs["input_values"],
                attention_mask=attention_mask,
            )
            loss = self._compute_custom_loss(model, inputs, labels, outputs, attention_mask)

        self._log_first_batch(inputs, labels, outputs)
        if loss is None or not torch.isfinite(loss).all().item():
            loss_value = None if loss is None else loss.detach().cpu().tolist()
            raise RuntimeError(
                f"Non-finite {self.loss_impl} CTC loss detected: {loss_value}. "
                "Check audio lengths, label lengths, and the selected loss implementation."
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Return valid logit lengths with logits for padding-safe diagnostics."""
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raw_input_lengths = torch.full(
                (inputs["input_values"].shape[0],),
                inputs["input_values"].shape[1],
                dtype=torch.long,
                device=inputs["input_values"].device,
            )
        else:
            raw_input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        actual_model = model.module if hasattr(model, "module") else model
        logit_lengths = actual_model._get_feat_extract_output_lengths(
            raw_input_lengths
        )
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        if logits is None:
            return loss, logits, labels
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logit_lengths = logit_lengths.to(logits.device)
        return loss, (logits, logit_lengths.unsqueeze(-1)), labels


def resolve_use_attention_mask_for_loss(processor, model, requested: Optional[bool]):
    """Resolve whether training forwards the collator attention mask to the model."""
    if requested is not None:
        print(f"Attention mask for loss forward explicitly set to {requested}")
        return requested

    return_attention_mask = getattr(
        processor.feature_extractor, "return_attention_mask", None
    )
    feat_extract_norm = getattr(model.config, "feat_extract_norm", None)
    use_attention_mask = not (
        return_attention_mask is False or feat_extract_norm == "group"
    )
    print(
        "Attention mask for loss forward auto-selected: "
        f"{use_attention_mask} "
        f"(return_attention_mask={return_attention_mask}, "
        f"feat_extract_norm={feat_extract_norm})"
    )
    return use_attention_mask


def configure_spec_augment(model, apply_spec_augment: bool):
    """Set SpecAugment consistently on the CTC wrapper and Wav2Vec2 module."""
    model.config.apply_spec_augment = apply_spec_augment
    if hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "config"):
        wav2vec2_config = model.wav2vec2.config
        if hasattr(wav2vec2_config, "apply_spec_augment"):
            wav2vec2_config.apply_spec_augment = apply_spec_augment


def freeze_model_layers(
    model,
    freeze_feature_encoder: bool,
    freeze_wav2vec2: bool,
    freeze_n_layers: int,
):
    """Apply the requested feature encoder and transformer layer freezing."""
    if freeze_wav2vec2:
        for parameter in model.wav2vec2.parameters():
            parameter.requires_grad = False
        return

    if freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        else:
            model.freeze_feature_extractor()

    layers = model.wav2vec2.encoder.layers
    if freeze_n_layers < 0 or freeze_n_layers > len(layers):
        raise ValueError(
            f"--freeze_n_layers must be between 0 and {len(layers)}, "
            f"got {freeze_n_layers}"
        )
    for layer in layers[:freeze_n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def validate_layerwise_args(args):
    """Validate optional layer-wise learning-rate settings."""
    optional_lrs = (args.head_learning_rate, args.feature_extractor_learning_rate)
    if not args.layerwise_lr_decay and any(lr is not None for lr in optional_lrs):
        raise ValueError(
            "--head_learning_rate and --feature_extractor_learning_rate require "
            "--layerwise_lr_decay"
        )
    if args.freeze_feature_encoder and args.feature_extractor_learning_rate is not None:
        raise ValueError(
            "--freeze_feature_encoder cannot be combined with "
            "--feature_extractor_learning_rate"
        )
    if args.freeze_wav2vec2 and args.layerwise_lr_decay:
        raise ValueError(
            "--freeze_wav2vec2 cannot be combined with --layerwise_lr_decay"
        )
    if not 0.0 < args.layerwise_lr_decay_rate <= 1.0:
        raise ValueError("--layerwise_lr_decay_rate must be in the interval (0, 1]")
    if args.layerwise_lr_decay:
        head_lr = args.head_learning_rate or args.learning_rate
        if head_lr <= 0.0:
            raise ValueError("--head_learning_rate must be greater than zero")
        if head_lr < args.learning_rate:
            raise ValueError("--head_learning_rate must be at least --learning_rate")
        if (
            args.feature_extractor_learning_rate is not None
            and args.feature_extractor_learning_rate > head_lr
        ):
            raise ValueError(
                "--feature_extractor_learning_rate cannot exceed the CTC head LR"
            )
        if (
            args.feature_extractor_learning_rate is not None
            and args.feature_extractor_learning_rate <= 0.0
        ):
            raise ValueError(
                "--feature_extractor_learning_rate must be greater than zero"
            )


def build_layerwise_optimizer(model, args) -> torch.optim.Optimizer:
    """Build AdamW groups with progressively larger LRs toward the CTC head."""
    head_lr = args.head_learning_rate or args.learning_rate
    encoder_layers = model.wav2vec2.encoder.layers
    decay_rate = args.layerwise_lr_decay_rate
    assigned: Set[int] = set()
    groups = []

    def add_group(name: str, parameters, learning_rate: float):
        trainable = [
            parameter
            for parameter in parameters
            if parameter.requires_grad and id(parameter) not in assigned
        ]
        if not trainable:
            return
        assigned.update(id(parameter) for parameter in trainable)
        groups.append({"params": trainable, "lr": learning_rate, "group_name": name})
        print(f"Optimizer group {name}: {len(trainable)} tensors, lr={learning_rate:g}")

    # The CTC projection head receives the largest learning rate.
    add_group("ctc_head", model.lm_head.parameters(), head_lr)

    # Layer indices increase from the acoustic input toward the CTC head.
    # Applying more decay to earlier layers keeps lower representations stable.
    for layer_index, layer in enumerate(encoder_layers):
        exponent = len(encoder_layers) - 1 - layer_index
        layer_lr = args.learning_rate * (decay_rate ** exponent)
        add_group(f"encoder_layer_{layer_index}", layer.parameters(), layer_lr)

    # Shared Wav2Vec2 modules sit below the Transformer stack and use a
    # conservatively decayed learning rate.
    shared_lr = args.learning_rate * (decay_rate ** len(encoder_layers))
    feature_extractor = model.wav2vec2.feature_extractor
    if args.feature_extractor_learning_rate is None:
        for parameter in feature_extractor.parameters():
            parameter.requires_grad = False
        print("Feature extractor frozen for layer-wise LR decay")
    else:
        add_group(
            "feature_extractor",
            feature_extractor.parameters(),
            args.feature_extractor_learning_rate,
        )

    add_group(
        "wav2vec2_shared",
        (
            parameter
            for parameter in model.wav2vec2.parameters()
            if id(parameter) not in assigned
        ),
        shared_lr,
    )
    add_group(
        "other_trainable",
        (
            parameter
            for parameter in model.parameters()
            if id(parameter) not in assigned
        ),
        shared_lr,
    )
    return torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=0.0)


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    """Count patience only after a configured epoch."""

    def __init__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        start_epoch: float,
    ):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch is None or state.epoch < self.start_epoch:
            self.early_stopping_patience_counter = 0
            return control
        return super().on_evaluate(
            args,
            state,
            control,
            metrics,
            **kwargs,
        )


def main():
    """Load data, train the model, and save a stable final model directory."""
    args = parse_args()
    args.apply_spec_augment = resolve_spec_augment(args)
    validate_strict_main_args(args)
    validate_layerwise_args(args)
    set_seed(args.seed)

    if args.dry_run:
        shard_sets = {"train": args.train_shards}
        if args.eval_shards is not None:
            shard_sets["eval"] = args.eval_shards
        if args.experiment_role != "main":
            shard_sets["test-clean"] = args.test_clean_shards
            shard_sets["test-other"] = args.test_other_shards
        for split, shards in shard_sets.items():
            paths = sample_util.find_shards(shards)
            print(f"{split}: {len(paths)} shard(s)")
        print(
            "Dry run complete: "
            f"model={args.model_name_or_path}, output_dir={args.output_dir}, "
            f"experiment_role={args.experiment_role}, "
            f"max_train_steps={args.max_train_steps}, "
            f"max_eval_samples={args.max_eval_samples}, "
            f"eval_shards={args.eval_shards or args.test_clean_shards}, "
            f"layerwise_lr_decay={args.layerwise_lr_decay}, "
            f"loss_impl={args.loss_impl}, "
            f"ctc_zero_infinity={args.ctc_zero_infinity}, "
            f"apply_spec_augment={args.apply_spec_augment}, "
            f"use_attention_mask_for_loss={args.use_attention_mask_for_loss}"
        )
        return

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    train_dataset = sample_util.make_dataset(
        args.train_shards, processor, shuffle=True
    )
    eval_shards = args.eval_shards or args.test_clean_shards
    eval_dataset = sample_util.make_dataset(
        eval_shards,
        processor,
        max_samples=args.max_eval_samples,
    )

    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        local_files_only=args.local_files_only,
    )
    if args.ctc_zero_infinity:
        model.config.ctc_zero_infinity = True
    configure_spec_augment(model, args.apply_spec_augment)
    configure_spec_augment_parameters(model, args)
    if args.blank_bias_init is not None:
        if model.lm_head.bias is None:
            raise ValueError("--blank_bias_init requires an lm_head bias.")
        with torch.no_grad():
            model.lm_head.bias[processor.tokenizer.pad_token_id] = (
                args.blank_bias_init
            )
        print(
            "Initialized CTC blank-token bias "
            f"to {args.blank_bias_init:g}"
        )
    use_attention_mask_for_loss = resolve_use_attention_mask_for_loss(
        processor, model, args.use_attention_mask_for_loss
    )
    print(f"model.config.apply_spec_augment={model.config.apply_spec_augment}")
    print(f"experiment_role={args.experiment_role}")
    print(f"model.training={model.training}")
    print(f"loss_impl={args.loss_impl}")
    print(f"trainer_eval_shards={eval_shards}")
    print(f"use_attention_mask_for_loss={use_attention_mask_for_loss}")
    print(f"ctc_zero_infinity={getattr(model.config, 'ctc_zero_infinity', False)}")
    freeze_model_layers(
        model,
        args.freeze_feature_encoder,
        args.freeze_wav2vec2,
        args.freeze_n_layers,
    )
    optimizer: Optional[torch.optim.Optimizer] = None
    if args.layerwise_lr_decay:
        optimizer = build_layerwise_optimizer(model, args)
    finite_loss_details = run_finite_loss_preflight(
        model,
        processor,
        train_dataset,
        args,
        use_attention_mask_for_loss,
    )

    bounded_run = args.max_train_steps > 0
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_train_steps,
        fp16=args.fp16,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="steps" if bounded_run else "epoch",
        eval_delay=args.eval_delay,
        save_strategy="steps" if bounded_run else "epoch",
        eval_steps=args.max_train_steps if bounded_run else None,
        save_steps=args.max_train_steps if bounded_run else 500,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=False,
    )
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            DelayedEarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
                start_epoch=args.early_stopping_start_epoch,
            )
        )
    trainer = MyCtcTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
        loss_impl=args.loss_impl,
        debug_first_batch=args.debug_first_batch,
        debug_processor=processor,
        use_attention_mask_for_loss=use_attention_mask_for_loss,
        callbacks=callbacks,
    )
    started_at = time.time()
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "latest":
        resume_from_checkpoint = True
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    runtime_seconds = time.time() - started_at

    final_model_dir = os.path.join(args.output_dir, args.final_model_subdir)
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    provenance = {
        "experiment_role": args.experiment_role,
        "main_source_checkpoint": (
            ALLOWED_MAIN_SOURCE if args.experiment_role == "main" else None
        ),
        "model_name_or_path": args.model_name_or_path,
        "train_shards": sample_util.find_shards(args.train_shards),
        "eval_shards": sample_util.find_shards(eval_shards),
        "test_splits_used_during_training": False,
        "loss_impl": args.loss_impl,
        "apply_spec_augment": args.apply_spec_augment,
        "ctc_zero_infinity": args.ctc_zero_infinity,
        "use_attention_mask_for_loss": use_attention_mask_for_loss,
        "learning_rate": args.learning_rate,
        "freeze_feature_encoder": args.freeze_feature_encoder,
        "freeze_wav2vec2": args.freeze_wav2vec2,
        "freeze_n_layers": args.freeze_n_layers,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_start_epoch": args.early_stopping_start_epoch,
        "eval_delay": args.eval_delay,
        "load_best_model_at_end": args.load_best_model_at_end,
        "blank_bias_init": args.blank_bias_init,
        "mask_time_prob": getattr(model.config, "mask_time_prob", None),
        "mask_time_length": getattr(model.config, "mask_time_length", None),
        "mask_time_min_masks": getattr(model.config, "mask_time_min_masks", None),
        "seed": args.seed,
    }
    write_checkpoint_provenance(final_model_dir, provenance)

    log_history = trainer.state.log_history
    training_log_csv = args.training_log_csv or os.path.join(
        args.output_dir, "training_log.csv"
    )
    validation_history_csv = args.validation_history_csv or os.path.join(
        args.output_dir, "validation_wer_history.csv"
    )
    write_log_history_csv(log_history, training_log_csv)
    write_validation_history_csv(log_history, validation_history_csv)

    metadata = {
        **provenance,
        "output_dir": args.output_dir,
        "saved_model_path": final_model_dir,
        "best_checkpoint_path": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "runtime_seconds": runtime_seconds,
        "train_metrics": train_result.metrics,
        "finite_loss_check": finite_loss_details,
        "training_log_csv": training_log_csv,
        "validation_history_csv": validation_history_csv,
    }
    metadata_path = args.run_metadata_json or os.path.join(
        args.output_dir, "run_metadata.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    print(f"Saved run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
