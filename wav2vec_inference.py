"""Run batched Wav2Vec 2.0 inference on test-clean and test-other."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor, set_seed

import sample_util


def parse_args():
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_clean_shards", default="data/test-clean")
    parser.add_argument("--test_other_shards", default="data/test-other")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--decoding_method",
        choices=("greedy", "beam"),
        default="greedy",
        help="CTC decoding strategy. Beam decoding requires pyctcdecode.",
    )
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="Limit samples per test split for smoke tests.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate shard paths and print the plan without loading a model.",
    )
    return parser.parse_args()


def build_collate_fn(processor):
    """Create an inference collator that retains reference transcripts."""

    def collate_fn(features: List[Dict]):
        batch = processor.pad(
            [{"input_values": feature["input_values"]} for feature in features],
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch["references"] = [feature["labels"] for feature in features]
        return batch

    return collate_fn


def build_beam_decoder(processor):
    """Build a pyctcdecode decoder without requiring an external LM."""
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as error:
        raise RuntimeError(
            "Beam decoding requires pyctcdecode. Install it with: "
            "python -m pip install pyctcdecode"
        ) from error

    vocab = processor.tokenizer.get_vocab()
    labels = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        labels[token_id] = token
    labels[processor.tokenizer.pad_token_id] = ""
    delimiter = processor.tokenizer.word_delimiter_token
    if delimiter in vocab:
        labels[vocab[delimiter]] = " "
    return build_ctcdecoder(labels=labels)


def decode_logits(logits, processor, decoding_method, decoder, beam_width):
    """Decode model logits with greedy or optional CTC beam search."""
    if decoding_method == "greedy":
        return processor.batch_decode(torch.argmax(logits, dim=-1))
    logit_batches = logits.detach().cpu().numpy()
    return [decoder.decode(logit, beam_width=beam_width) for logit in logit_batches]


def write_results(
    dataset,
    model,
    processor,
    device,
    batch_size,
    fp16,
    output_file,
    decoding_method,
    decoder=None,
    beam_width=100,
):
    """Write REF/HYP pairs for one evaluation split."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=build_collate_fn(processor),
    )
    use_fp16 = fp16 and device.type == "cuda"
    with open(output_file, "w", encoding="utf-8") as output:
        for batch in loader:
            references = batch.pop("references")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    logits = model(**batch).logits
            hypotheses = decode_logits(
                logits, processor, decoding_method, decoder, beam_width
            )
            for reference, hypothesis in zip(references, hypotheses):
                output.write(f"REF: {reference}\n")
                output.write(f"HYP: {hypothesis}\n\n")


def main():
    """Transcribe both evaluation splits."""
    args = parse_args()
    if args.beam_width <= 0:
        raise ValueError("--beam_width must be greater than zero")
    set_seed(args.seed)

    if args.dry_run:
        shard_sets = {
            "test-clean": args.test_clean_shards,
            "test-other": args.test_other_shards,
        }
        for split, shards in shard_sets.items():
            paths = sample_util.find_shards(shards)
            print(f"{split}: {len(paths)} shard(s)")
        print(
            "Dry run complete: "
            f"model={args.model_name_or_path}, output_dir={args.output_dir}, "
            f"max_test_samples={args.max_test_samples}, "
            f"decoding_method={args.decoding_method}, beam_width={args.beam_width}"
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForCTC.from_pretrained(args.model_name_or_path)
    decoder: Optional[object] = None
    if args.decoding_method == "beam":
        decoder = build_beam_decoder(processor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    datasets = {
        "test_clean_result.txt": sample_util.make_dataset(
            args.test_clean_shards,
            processor,
            do_tokenization=False,
            max_samples=args.max_test_samples,
        ),
        "test_other_result.txt": sample_util.make_dataset(
            args.test_other_shards,
            processor,
            do_tokenization=False,
            max_samples=args.max_test_samples,
        ),
    }
    for filename, dataset in datasets.items():
        write_results(
            dataset,
            model,
            processor,
            device,
            args.per_device_eval_batch_size,
            args.fp16,
            os.path.join(args.output_dir, filename),
            args.decoding_method,
            decoder=decoder,
            beam_width=args.beam_width,
        )

    metadata = {
        "checkpoint_path": args.model_name_or_path,
        "decoding_method": args.decoding_method,
        "beam_width": args.beam_width if args.decoding_method == "beam" else "",
        "max_test_samples": args.max_test_samples,
        "fp16": args.fp16,
        "seed": args.seed,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
