"""Fine-tune Wav2Vec 2.0 with Hugging Face or project CTC loss."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
import torch
from jiwer import wer
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

import ctc_loss
import sample_util


def parse_args():
    """Parse fine-tuning arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_shards", default="data/train")
    parser.add_argument("--test_clean_shards", default="data/test-clean")
    parser.add_argument("--test_other_shards", default="data/test-other")
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-base")
    parser.add_argument("--output_dir", default="outputs/baseline_lr1e-4")
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
        help="Disable SpecAugment during fine-tuning. This is the stable default.",
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
    parser.set_defaults(apply_spec_augment=False, use_attention_mask_for_loss=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate shard paths and print the plan without loading a model.",
    )
    return parser.parse_args()


def build_compute_metrics(processor):
    """Create a WER metric callback bound to the selected processor."""

    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(pred_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer(references, predictions)}

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


def freeze_model_layers(model, freeze_feature_encoder: bool, freeze_n_layers: int):
    """Apply the requested feature encoder and transformer layer freezing."""
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


def main():
    """Load data, train the model, and save a stable final model directory."""
    args = parse_args()
    validate_layerwise_args(args)
    set_seed(args.seed)

    if args.dry_run:
        shard_sets = {
            "train": args.train_shards,
            "test-clean": args.test_clean_shards,
            "test-other": args.test_other_shards,
        }
        for split, shards in shard_sets.items():
            paths = sample_util.find_shards(shards)
            print(f"{split}: {len(paths)} shard(s)")
        print(
            "Dry run complete: "
            f"model={args.model_name_or_path}, output_dir={args.output_dir}, "
            f"max_train_steps={args.max_train_steps}, "
            f"max_eval_samples={args.max_eval_samples}, "
            f"layerwise_lr_decay={args.layerwise_lr_decay}, "
            f"loss_impl={args.loss_impl}, "
            f"ctc_zero_infinity={args.ctc_zero_infinity}, "
            f"apply_spec_augment={args.apply_spec_augment}, "
            f"use_attention_mask_for_loss={args.use_attention_mask_for_loss}"
        )
        return

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    train_dataset = sample_util.make_dataset(
        args.train_shards, processor, shuffle=True
    )
    test_clean_dataset = sample_util.make_dataset(
        args.test_clean_shards,
        processor,
        max_samples=args.max_eval_samples,
    )

    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    if args.ctc_zero_infinity:
        model.config.ctc_zero_infinity = True
    configure_spec_augment(model, args.apply_spec_augment)
    use_attention_mask_for_loss = resolve_use_attention_mask_for_loss(
        processor, model, args.use_attention_mask_for_loss
    )
    print(f"model.config.apply_spec_augment={model.config.apply_spec_augment}")
    print(f"model.training={model.training}")
    print(f"loss_impl={args.loss_impl}")
    print(f"use_attention_mask_for_loss={use_attention_mask_for_loss}")
    print(f"ctc_zero_infinity={getattr(model.config, 'ctc_zero_infinity', False)}")
    freeze_model_layers(model, args.freeze_feature_encoder, args.freeze_n_layers)
    optimizer: Optional[torch.optim.Optimizer] = None
    if args.layerwise_lr_decay:
        optimizer = build_layerwise_optimizer(model, args)

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
        eval_strategy="steps" if bounded_run else "epoch",
        save_strategy="steps" if bounded_run else "epoch",
        eval_steps=args.max_train_steps if bounded_run else None,
        save_steps=args.max_train_steps if bounded_run else 500,
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=False,
    )
    trainer = MyCtcTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_clean_dataset,
        processing_class=processor,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
        loss_impl=args.loss_impl,
        debug_first_batch=args.debug_first_batch,
        debug_processor=processor,
        use_attention_mask_for_loss=use_attention_mask_for_loss,
    )
    trainer.train()

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)


if __name__ == "__main__":
    main()
