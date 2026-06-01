"""Fine-tune Wav2Vec 2.0 with the provided custom CTC loss."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Union

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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
    """Trainer that uses the project custom CTC loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        attention_mask = inputs.get("attention_mask")
        outputs = model(**inputs)

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
        loss = ctc_loss.CtcLoss.apply(
            labels,
            target_lengths,
            logits.log_softmax(dim=-1),
            logits_lengths,
        ).mean()

        return (loss, outputs) if return_outputs else loss


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


def main():
    """Load data, train the model, and save a stable final model directory."""
    args = parse_args()
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
            f"max_eval_samples={args.max_eval_samples}"
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
    freeze_model_layers(model, args.freeze_feature_encoder, args.freeze_n_layers)

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
        evaluation_strategy="steps" if bounded_run else "epoch",
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
        tokenizer=processor,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
    )
    trainer.train()

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)


if __name__ == "__main__":
    main()
