"""Run batched Wav2Vec 2.0 inference on test-clean and test-other."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from typing import Dict, List

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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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


def write_results(dataset, model, processor, device, batch_size, fp16, output_file):
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
            hypotheses = processor.batch_decode(torch.argmax(logits, dim=-1))
            for reference, hypothesis in zip(references, hypotheses):
                output.write(f"REF: {reference}\n")
                output.write(f"HYP: {hypothesis}\n\n")


def main():
    """Transcribe both evaluation splits."""
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForCTC.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    datasets = {
        "test_clean_result.txt": sample_util.make_dataset(
            args.test_clean_shards, processor, do_tokenization=False
        ),
        "test_other_result.txt": sample_util.make_dataset(
            args.test_other_shards, processor, do_tokenization=False
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
        )


if __name__ == "__main__":
    main()
