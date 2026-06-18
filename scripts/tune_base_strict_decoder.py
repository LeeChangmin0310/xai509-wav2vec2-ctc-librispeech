#!/usr/bin/env python3
"""Tune greedy, beam, and train-text n-gram shallow fusion on validation only."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from jiwer import wer
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from experiment_guard import validate_checkpoint_role
from text_normalization import normalize_transcript
from wav2vec_inference import (
    build_beam_decoder,
    build_collate_fn,
    decode_logits,
)


def comma_values(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path",
        default="outputs/base_strict_final/best_model",
    )
    parser.add_argument(
        "--validation_shards",
        default="data/train/shard-000004.tar",
    )
    parser.add_argument(
        "--lm_path",
        default="results/base_strict_final/train_text_trigram_lm.json",
    )
    parser.add_argument(
        "--output_csv",
        default="results/base_strict_final/validation_decoding_sweep.csv",
    )
    parser.add_argument(
        "--best_decoder_json",
        default="results/base_strict_final/selected_decoder_config.json",
    )
    parser.add_argument(
        "--greedy_output_csv",
        default="results/base_strict_final/validation_greedy_wer.csv",
    )
    parser.add_argument(
        "--best_predictions",
        default="results/base_strict_final/validation_ref_hyp_examples.txt",
    )
    parser.add_argument("--beam_widths", default="50,100,200,300")
    parser.add_argument("--alphas", default="0.0,0.3,0.5,0.7,1.0,1.5")
    parser.add_argument("--betas", default="-1.0,0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--no_attention_mask_for_forward",
        action="store_true",
        help="Match strict base evaluation by omitting the padding attention mask.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def write_predictions(path, references, hypotheses):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for reference, hypothesis in zip(references, hypotheses):
            output_file.write(f"REF: {reference}\n")
            output_file.write(f"HYP: {hypothesis}\n\n")


def main():
    args = parse_args()
    validate_checkpoint_role(args.model_name_or_path, "main")
    beam_widths = comma_values(args.beam_widths, int)
    alphas = comma_values(args.alphas, float)
    betas = comma_values(args.betas, float)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = sample_util.make_dataset(
        args.validation_shards,
        processor,
        do_tokenization=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=build_collate_fn(processor),
    )

    references = []
    logits_batches = []
    use_fp16 = args.fp16 and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            references.extend(
                normalize_transcript(text) for text in batch.pop("references")
            )
            if args.no_attention_mask_for_forward:
                batch.pop("attention_mask", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits_batches.append(model(**batch).logits.cpu())

    rows = []
    best = None
    best_hypotheses = None

    def evaluate_setting(method, beam_width=None, alpha=None, beta=None, lm_path=None):
        nonlocal best, best_hypotheses
        decoder = None
        if method != "greedy":
            decoder = build_beam_decoder(
                processor,
                language_model_path=lm_path,
                alpha=0.5 if alpha is None else alpha,
                beta=1.5 if beta is None else beta,
            )
        hypotheses = []
        for logits in logits_batches:
            hypotheses.extend(
                normalize_transcript(text)
                for text in decode_logits(
                    logits,
                    processor,
                    "greedy" if method == "greedy" else "beam",
                    decoder,
                    100 if beam_width is None else beam_width,
                )
            )
        validation_wer = wer(references, hypotheses)
        row = {
            "decoding_method": method,
            "beam_width": "" if beam_width is None else beam_width,
            "alpha": "" if alpha is None else alpha,
            "beta": "" if beta is None else beta,
            "language_model_path": "" if lm_path is None else lm_path,
            "validation_wer": validation_wer,
        }
        rows.append(row)
        if best is None or validation_wer < best["validation_wer"]:
            best = row.copy()
            best_hypotheses = hypotheses
        print(row, flush=True)

    evaluate_setting("greedy")
    for beam_width in beam_widths:
        evaluate_setting("beam", beam_width=beam_width)
    for beam_width in beam_widths:
        for alpha in alphas:
            for beta in betas:
                evaluate_setting(
                    "beam_lm",
                    beam_width=beam_width,
                    alpha=alpha,
                    beta=beta,
                    lm_path=args.lm_path,
                )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    greedy_row = next(row for row in rows if row["decoding_method"] == "greedy")
    with open(
        args.greedy_output_csv, "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(greedy_row))
        writer.writeheader()
        writer.writerow(greedy_row)
    best.update(
        {
            "acoustic_checkpoint": args.model_name_or_path,
            "validation_shards": args.validation_shards,
            "selection_source": "validation_only",
            "test_splits_used_for_selection": False,
            "attention_mask_passed": not args.no_attention_mask_for_forward,
        }
    )
    with open(args.best_decoder_json, "w", encoding="utf-8") as output_file:
        json.dump(best, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    write_predictions(args.best_predictions, references, best_hypotheses)
    print(f"Saved greedy validation WER: {args.greedy_output_csv}")
    print(f"Selected decoder: {best}")


if __name__ == "__main__":
    main()
