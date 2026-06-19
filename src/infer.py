"""Run batched Wav2Vec 2.0 inference on test-clean and test-other."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import torch
from jiwer import wer
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor, set_seed

from . import data
from .guard import validate_checkpoint_role
from .normalization import normalize_transcript


def parse_args():
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_clean_shards", default="data/test-clean")
    parser.add_argument("--test_other_shards", default="data/test-other")
    parser.add_argument(
        "--input_shards",
        help="Optional single validation or evaluation shard specification.",
    )
    parser.add_argument(
        "--result_file",
        help="Output REF/HYP file used with --input_shards.",
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--experiment_role",
        choices=("main", "positive_control_only", "diagnostic"),
        default="main",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--decoding_method",
        choices=("greedy", "beam"),
        default="greedy",
        help="CTC decoding strategy. Beam decoding requires pyctcdecode.",
    )
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--language_model_path")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="Limit samples per test split for smoke tests.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--no_attention_mask_for_forward",
        action="store_true",
        help="Omit the padding attention mask to match strict base evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--tune_decoder",
        action="store_true",
        help="Tune greedy, beam, and train-text LM decoding on validation only.",
    )
    parser.add_argument(
        "--validation_shards",
        default="data/train/shard-000004.tar",
    )
    parser.add_argument(
        "--decoder_config_output",
        default="results/selected_decoder_config.json",
    )
    parser.add_argument(
        "--decoder_sweep_output",
        default="outputs/strict_final/validation_decoding_sweep.csv",
    )
    parser.add_argument("--beam_widths", default="50,100,200,300")
    parser.add_argument("--alphas", default="0.0,0.3,0.5,0.7,1.0,1.5")
    parser.add_argument("--betas", default="-1.0,0.0,0.5,1.0,1.5,2.0")
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


def build_beam_decoder(processor, language_model_path=None, alpha=0.5, beta=1.5):
    """Build a pyctcdecode decoder with an optional KenLM model."""
    try:
        from pyctcdecode import BeamSearchDecoderCTC, build_ctcdecoder
        from pyctcdecode.alphabet import Alphabet
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
    if language_model_path and language_model_path.endswith(".json"):
        from .lm import SimpleWordNGramLanguageModel

        language_model = SimpleWordNGramLanguageModel.load(language_model_path)
        language_model.reset_params(alpha=alpha, beta=beta)
        return BeamSearchDecoderCTC(
            Alphabet.build_alphabet(labels),
            language_model=language_model,
        )
    return build_ctcdecoder(
        labels=labels,
        kenlm_model_path=language_model_path,
        alpha=alpha,
        beta=beta,
    )


def decode_logits(logits, processor, decoding_method, decoder, beam_width):
    """Decode model logits with greedy or optional CTC beam search."""
    if decoding_method == "greedy":
        return processor.batch_decode(torch.argmax(logits, dim=-1))
    logit_batches = logits.detach().cpu().numpy()
    return [decoder.decode(logit, beam_width=beam_width) for logit in logit_batches]


def comma_values(value, cast):
    """Parse a comma-separated hyperparameter list."""
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def tune_decoder(args) -> None:
    """Select decoding hyperparameters using the validation shard only."""
    if not args.language_model_path:
        raise ValueError("--language_model_path is required with --tune_decoder")

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
    dataset = data.make_dataset(
        args.validation_shards,
        processor,
        do_tokenization=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
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

    def evaluate_setting(
        method,
        beam_width=None,
        alpha=None,
        beta=None,
        language_model_path=None,
    ):
        decoder = None
        decoding_method = "greedy"
        if method != "greedy":
            decoding_method = "beam"
            decoder = build_beam_decoder(
                processor,
                language_model_path=language_model_path,
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
                    decoding_method,
                    decoder,
                    100 if beam_width is None else beam_width,
                )
            )
        row = {
            "decoding_method": method,
            "beam_width": "" if beam_width is None else beam_width,
            "alpha": "" if alpha is None else alpha,
            "beta": "" if beta is None else beta,
            "language_model_path": (
                "" if language_model_path is None else language_model_path
            ),
            "validation_wer": wer(references, hypotheses),
        }
        rows.append(row)
        print(row, flush=True)

    evaluate_setting("greedy")
    for beam_width in comma_values(args.beam_widths, int):
        evaluate_setting("beam", beam_width=beam_width)
    for beam_width in comma_values(args.beam_widths, int):
        for alpha in comma_values(args.alphas, float):
            for beta in comma_values(args.betas, float):
                evaluate_setting(
                    "beam_lm",
                    beam_width=beam_width,
                    alpha=alpha,
                    beta=beta,
                    language_model_path=args.language_model_path,
                )

    best = min(rows, key=lambda row: float(row["validation_wer"])).copy()
    best.update(
        {
            "acoustic_checkpoint": args.model_name_or_path,
            "display_name": (
                "Staged CTC fine-tuned Wav2Vec2 + beam/trigram LM"
            ),
            "internal_id": "H",
            "role": "main_validation_selected_decoder",
            "validation_shards": args.validation_shards,
            "selection_source": "validation_only",
            "test_splits_used_for_selection": False,
            "attention_mask_passed": not args.no_attention_mask_for_forward,
        }
    )
    os.makedirs(os.path.dirname(args.decoder_sweep_output) or ".", exist_ok=True)
    with open(
        args.decoder_sweep_output, "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(os.path.dirname(args.decoder_config_output) or ".", exist_ok=True)
    with open(args.decoder_config_output, "w", encoding="utf-8") as output_file:
        json.dump(best, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Selected decoder: {best}")


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
    normalize_text=False,
    no_attention_mask_for_forward=False,
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
            if no_attention_mask_for_forward:
                batch.pop("attention_mask", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    logits = model(**batch).logits
            hypotheses = decode_logits(
                logits, processor, decoding_method, decoder, beam_width
            )
            for reference, hypothesis in zip(references, hypotheses):
                if normalize_text:
                    reference = normalize_transcript(reference)
                    hypothesis = normalize_transcript(hypothesis)
                output.write(f"REF: {reference}\n")
                output.write(f"HYP: {hypothesis}\n\n")


def main():
    """Transcribe both evaluation splits."""
    args = parse_args()
    if args.beam_width <= 0:
        raise ValueError("--beam_width must be greater than zero")
    if bool(args.input_shards) != bool(args.result_file):
        raise ValueError("--input_shards and --result_file must be provided together")
    validate_checkpoint_role(args.model_name_or_path, args.experiment_role)
    set_seed(args.seed)

    if args.tune_decoder:
        tune_decoder(args)
        return

    if args.dry_run:
        shard_sets = (
            {"input": args.input_shards}
            if args.input_shards
            else {
                "test-clean": args.test_clean_shards,
                "test-other": args.test_other_shards,
            }
        )
        for split, shards in shard_sets.items():
            paths = data.find_shards(shards)
            print(f"{split}: {len(paths)} shard(s)")
        print(
            "Dry run complete: "
            f"model={args.model_name_or_path}, output_dir={args.output_dir}, "
            f"experiment_role={args.experiment_role}, "
            f"max_test_samples={args.max_test_samples}, "
            f"decoding_method={args.decoding_method}, beam_width={args.beam_width}"
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    decoder: Optional[object] = None
    if args.decoding_method == "beam":
        decoder = build_beam_decoder(
            processor,
            language_model_path=args.language_model_path,
            alpha=args.alpha,
            beta=args.beta,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.input_shards:
        datasets = {
            args.result_file: data.make_dataset(
                args.input_shards,
                processor,
                do_tokenization=False,
                max_samples=args.max_test_samples,
            )
        }
    else:
        datasets = {
            "test_clean_result.txt": data.make_dataset(
                args.test_clean_shards,
                processor,
                do_tokenization=False,
                max_samples=args.max_test_samples,
            ),
            "test_other_result.txt": data.make_dataset(
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
            (
                filename
                if os.path.isabs(filename)
                else os.path.join(args.output_dir, filename)
            ),
            args.decoding_method,
            decoder=decoder,
            beam_width=args.beam_width,
            normalize_text=args.normalize_text,
            no_attention_mask_for_forward=args.no_attention_mask_for_forward,
        )

    metadata = {
        "checkpoint_path": args.model_name_or_path,
        "experiment_role": args.experiment_role,
        "decoding_method": args.decoding_method,
        "beam_width": args.beam_width if args.decoding_method == "beam" else "",
        "language_model_path": args.language_model_path,
        "alpha": args.alpha if args.decoding_method == "beam" else "",
        "beta": args.beta if args.decoding_method == "beam" else "",
        "input_shards": args.input_shards,
        "normalize_text": args.normalize_text,
        "max_test_samples": args.max_test_samples,
        "fp16": args.fp16,
        "attention_mask_passed": not args.no_attention_mask_for_forward,
        "seed": args.seed,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
