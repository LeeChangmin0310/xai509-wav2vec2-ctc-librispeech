#!/usr/bin/env python3
"""Evaluate strict H-fold logit averaging and ROVER-style word voting."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

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
from wav2vec_inference import build_beam_decoder, build_collate_fn, decode_logits


DEFAULT_MODELS = ",".join(
    f"outputs/base_strict_cv/two_stage_head_warmup/fold_{fold}/best_model"
    for fold in range(5)
)
MAIN_TEST_CLEAN_WER = 0.24638618381010347
MAIN_TEST_OTHER_WER = 0.3292894942972317


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--model_paths", default=DEFAULT_MODELS)
    parser.add_argument(
        "--validation_shards", default="data/train/shard-000004.tar"
    )
    parser.add_argument("--test_clean_shards", default="data/test-clean")
    parser.add_argument("--test_other_shards", default="data/test-other")
    parser.add_argument(
        "--language_model_path",
        default="results/base_strict_final/train_text_trigram_lm.json",
    )
    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_attention_mask_for_forward", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--result_dir", default="results/base_strict_exploratory"
    )
    return parser.parse_args()


def write_predictions(
    path: Path, references: Iterable[str], hypotheses: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for reference, hypothesis in zip(references, hypotheses):
            output_file.write(f"REF: {reference}\n")
            output_file.write(f"HYP: {hypothesis}\n\n")


def representative(column: list[str]) -> str:
    """Choose a stable non-null representative for progressive alignment."""
    nonempty = [token for token in column if token]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    best_count = max(counts.values())
    for token in nonempty:
        if counts[token] == best_count:
            return token
    raise AssertionError("unreachable")


def align_words(left: list[str], right: list[str]) -> list[tuple[str, int, int]]:
    """Return deterministic Levenshtein alignment operations."""
    rows = len(left) + 1
    columns = len(right) + 1
    costs = [[0] * columns for _ in range(rows)]
    backtrace: list[list[str | None]] = [
        [None] * columns for _ in range(rows)
    ]
    for left_index in range(1, rows):
        costs[left_index][0] = left_index
        backtrace[left_index][0] = "delete"
    for right_index in range(1, columns):
        costs[0][right_index] = right_index
        backtrace[0][right_index] = "insert"

    for left_index in range(1, rows):
        for right_index in range(1, columns):
            diagonal_cost = costs[left_index - 1][right_index - 1] + (
                left[left_index - 1] != right[right_index - 1]
            )
            delete_cost = costs[left_index - 1][right_index] + 1
            insert_cost = costs[left_index][right_index - 1] + 1
            best_cost = min(diagonal_cost, delete_cost, insert_cost)
            costs[left_index][right_index] = best_cost
            if diagonal_cost == best_cost:
                backtrace[left_index][right_index] = "diagonal"
            elif delete_cost == best_cost:
                backtrace[left_index][right_index] = "delete"
            else:
                backtrace[left_index][right_index] = "insert"

    operations = []
    left_index = len(left)
    right_index = len(right)
    while left_index or right_index:
        operation = backtrace[left_index][right_index]
        if operation == "diagonal":
            operations.append(("diagonal", left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
        elif operation == "delete":
            operations.append(("delete", left_index - 1, right_index))
            left_index -= 1
        elif operation == "insert":
            operations.append(("insert", left_index, right_index - 1))
            right_index -= 1
        else:
            raise RuntimeError("Invalid word-alignment backtrace")
    operations.reverse()
    return operations


def rover_vote(hypotheses: list[str]) -> str:
    """Combine transcripts with progressive alignment and null-aware voting."""
    token_sequences = [hypothesis.split() for hypothesis in hypotheses]
    columns = [[token] for token in token_sequences[0]]
    systems_seen = 1
    for tokens in token_sequences[1:]:
        base = [representative(column) for column in columns]
        aligned_columns: list[list[str]] = []
        for operation, left_index, right_index in align_words(base, tokens):
            if operation == "diagonal":
                column = list(columns[left_index])
                column.append(tokens[right_index])
                aligned_columns.append(column)
            elif operation == "delete":
                column = list(columns[left_index])
                column.append("")
                aligned_columns.append(column)
            else:
                aligned_columns.append([""] * systems_seen + [tokens[right_index]])
        columns = aligned_columns
        systems_seen += 1

    voted_words = []
    for column in columns:
        counts = Counter(column)
        best_count = max(counts.values())
        winners = {token for token, count in counts.items() if count == best_count}
        preferred = representative(column)
        winner = preferred if preferred in winners else next(
            token for token in column if token in winners
        )
        if winner:
            voted_words.append(winner)
    return " ".join(voted_words)


def load_models(
    model_paths: list[str],
    device: torch.device,
    local_files_only: bool,
) -> tuple[object, list[torch.nn.Module]]:
    for model_path in model_paths:
        validate_checkpoint_role(model_path, "main")
    processor = AutoProcessor.from_pretrained(
        model_paths[0], local_files_only=local_files_only
    )
    reference_vocab = processor.tokenizer.get_vocab()
    models = []
    for model_path in model_paths:
        candidate_processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        if candidate_processor.tokenizer.get_vocab() != reference_vocab:
            raise ValueError(f"Tokenizer mismatch for ensemble model: {model_path}")
        model = AutoModelForCTC.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        model.to(device)
        model.eval()
        models.append(model)
    return processor, models


def evaluate_split(
    shard_spec: str,
    processor,
    models: list[torch.nn.Module],
    decoder,
    device: torch.device,
    batch_size: int,
    beam_width: int,
    fp16: bool,
    no_attention_mask_for_forward: bool,
    methods: set[str],
) -> tuple[list[str], dict[str, list[str]], list[list[str]]]:
    dataset = sample_util.make_dataset(
        shard_spec, processor, do_tokenization=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=build_collate_fn(processor),
    )
    references: list[str] = []
    method_hypotheses = {method: [] for method in methods}
    fold_hypotheses: list[list[str]] = [[] for _ in models]
    use_fp16 = fp16 and device.type == "cuda"

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch_references = [
                normalize_transcript(text) for text in batch.pop("references")
            ]
            references.extend(batch_references)
            if no_attention_mask_for_forward:
                batch.pop("attention_mask", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits_sum = None
            batch_fold_hypotheses: list[list[str]] = []
            for model_index, model in enumerate(models):
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    logits = model(**batch).logits
                logits_float = logits.float()
                logits_sum = (
                    logits_float
                    if logits_sum is None
                    else logits_sum + logits_float
                )
                if "rover" in methods:
                    decoded = [
                        normalize_transcript(text)
                        for text in decode_logits(
                            logits_float,
                            processor,
                            "beam",
                            decoder,
                            beam_width,
                        )
                    ]
                    fold_hypotheses[model_index].extend(decoded)
                    batch_fold_hypotheses.append(decoded)

            if "logit_average" in methods:
                averaged_logits = logits_sum / len(models)
                method_hypotheses["logit_average"].extend(
                    normalize_transcript(text)
                    for text in decode_logits(
                        averaged_logits,
                        processor,
                        "beam",
                        decoder,
                        beam_width,
                    )
                )
            if "rover" in methods:
                for sample_index in range(len(batch_references)):
                    method_hypotheses["rover"].append(
                        rover_vote(
                            [
                                hypotheses[sample_index]
                                for hypotheses in batch_fold_hypotheses
                            ]
                        )
                    )
            if batch_index % 10 == 0:
                print(
                    f"Processed {len(references)} utterances from {shard_spec}",
                    flush=True,
                )
    return references, method_hypotheses, fold_hypotheses


def run_validation(args, model_paths, processor, models, decoder, device) -> None:
    result_dir = Path(args.result_dir)
    references, ensemble_hypotheses, fold_hypotheses = evaluate_split(
        args.validation_shards,
        processor,
        models,
        decoder,
        device,
        args.batch_size,
        args.beam_width,
        args.fp16,
        args.no_attention_mask_for_forward,
        {"logit_average", "rover"},
    )
    rows = []
    for fold_index, hypotheses in enumerate(fold_hypotheses):
        rows.append(
            {
                "method": f"fold_{fold_index}",
                "method_type": "single_model_diagnostic",
                "validation_wer": wer(references, hypotheses),
                "beam_width": args.beam_width,
                "alpha": args.alpha,
                "beta": args.beta,
                "language_model_path": args.language_model_path,
                "model_count": 1,
            }
        )
        write_predictions(
            result_dir / f"validation_fold_{fold_index}_predictions.txt",
            references,
            hypotheses,
        )
    for method in ("logit_average", "rover"):
        hypotheses = ensemble_hypotheses[method]
        rows.append(
            {
                "method": method,
                "method_type": "ensemble_candidate",
                "validation_wer": wer(references, hypotheses),
                "beam_width": args.beam_width,
                "alpha": args.alpha,
                "beta": args.beta,
                "language_model_path": args.language_model_path,
                "model_count": len(models),
            }
        )
        write_predictions(
            result_dir / f"validation_{method}_predictions.txt",
            references,
            hypotheses,
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    validation_csv = result_dir / "h_fold_ensemble_validation.csv"
    with validation_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = [
        row for row in rows if row["method_type"] == "ensemble_candidate"
    ]
    selected = min(candidates, key=lambda row: float(row["validation_wer"]))
    selection = {
        **selected,
        "model_paths": model_paths,
        "validation_shards": args.validation_shards,
        "selection_source": "validation_shard_000004_only",
        "test_splits_used_for_selection": False,
        "decoder_config_source": (
            "results/base_strict_final/selected_decoder_config.json"
        ),
        "attention_mask_passed": not args.no_attention_mask_for_forward,
        "validation_leakage_caveat": (
            "H folds 0-3 included shard 000004 in acoustic training; only fold 4 "
            "held it out. Ensemble validation is therefore optimistic."
        ),
    }
    selection_path = result_dir / "selected_h_fold_ensemble.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {validation_csv}")
    print(f"Selected ensemble method: {selection['method']}")
    print(f"Selected validation WER: {selection['validation_wer']}")


def run_test(args, model_paths, processor, models, decoder, device) -> None:
    result_dir = Path(args.result_dir)
    selection_path = result_dir / "selected_h_fold_ensemble.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "Validation selection is missing; run --phase validation before test."
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_splits_used_for_selection") is not False:
        raise ValueError("Invalid ensemble selection provenance.")
    method = selection["method"]
    if method not in {"logit_average", "rover"}:
        raise ValueError(f"Unsupported selected ensemble method: {method}")

    test_rows = []
    for split, shard_spec, main_wer in (
        ("test-clean", args.test_clean_shards, MAIN_TEST_CLEAN_WER),
        ("test-other", args.test_other_shards, MAIN_TEST_OTHER_WER),
    ):
        references, method_hypotheses, _ = evaluate_split(
            shard_spec,
            processor,
            models,
            decoder,
            device,
            args.batch_size,
            args.beam_width,
            args.fp16,
            args.no_attention_mask_for_forward,
            {method},
        )
        hypotheses = method_hypotheses[method]
        split_wer = wer(references, hypotheses)
        prediction_path = (
            result_dir
            / f"h_fold_ensemble_{split.replace('-', '_')}_predictions.txt"
        )
        write_predictions(prediction_path, references, hypotheses)
        test_rows.append(
            {
                "split": split,
                "selected_method": method,
                "wer": split_wer,
                "main_final_wer": main_wer,
                "absolute_wer_change_vs_main": split_wer - main_wer,
                "improved_over_main": split_wer < main_wer,
                "beam_width": args.beam_width,
                "alpha": args.alpha,
                "beta": args.beta,
                "model_count": len(models),
                "prediction_path": str(prediction_path),
            }
        )
        print(f"{split} WER: {split_wer:.9f}", flush=True)

    summary_path = result_dir / "h_fold_ensemble_test_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"Wrote {summary_path}")


def main() -> None:
    args = parse_args()
    model_paths = [
        item.strip() for item in args.model_paths.split(",") if item.strip()
    ]
    if len(model_paths) != 5:
        raise ValueError("Exactly five H fold models are required.")
    if args.beam_width != 50 or args.alpha != 0.3 or args.beta != 1.5:
        raise ValueError("The preserved validation-selected decoder must remain fixed.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, models = load_models(
        model_paths, device, args.local_files_only
    )
    decoder = build_beam_decoder(
        processor,
        language_model_path=args.language_model_path,
        alpha=args.alpha,
        beta=args.beta,
    )
    if args.phase == "validation":
        run_validation(
            args, model_paths, processor, models, decoder, device
        )
    else:
        run_test(args, model_paths, processor, models, decoder, device)


if __name__ == "__main__":
    main()
