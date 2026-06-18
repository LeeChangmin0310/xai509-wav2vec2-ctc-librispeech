#!/usr/bin/env python3
"""Build the final WER summary and strict-base experiment report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_wer import read_ref_hyp_pairs
from text_normalization import normalize_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-summary",
        default="results/base_strict_cv/strict_cv_summary.csv",
    )
    parser.add_argument(
        "--selected-acoustic",
        default="results/base_strict_final/selected_acoustic_config.json",
    )
    parser.add_argument(
        "--decoding-sweep",
        default="results/base_strict_final/validation_decoding_sweep.csv",
    )
    parser.add_argument(
        "--selected-decoder",
        default="results/base_strict_final/selected_decoder_config.json",
    )
    parser.add_argument(
        "--test-clean",
        default="results/base_strict_final/test_clean_predictions.txt",
    )
    parser.add_argument(
        "--test-other",
        default="results/base_strict_final/test_other_predictions.txt",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/base_strict_final/wer_summary.csv",
    )
    parser.add_argument(
        "--report-path",
        default="reports/final_strict_base_report.md",
    )
    return parser.parse_args()


def prediction_wer(path: Path) -> float:
    references, hypotheses = read_ref_hyp_pairs(str(path))
    references = [normalize_transcript(text) for text in references]
    hypotheses = [normalize_transcript(text) for text in hypotheses]
    return wer(references, hypotheses)


def best_method(rows: list[dict[str, str]], method: str) -> dict[str, str] | None:
    candidates = [row for row in rows if row["decoding_method"] == method]
    return min(candidates, key=lambda row: float(row["validation_wer"])) if candidates else None


def value(row: dict[str, str] | None, key: str) -> str:
    return "" if row is None else row.get(key, "")


def main() -> None:
    args = parse_args()
    acoustic = json.loads(Path(args.selected_acoustic).read_text(encoding="utf-8"))
    decoder = json.loads(Path(args.selected_decoder).read_text(encoding="utf-8"))
    with Path(args.cv_summary).open(encoding="utf-8", newline="") as input_file:
        cv_rows = list(csv.DictReader(input_file))
    with Path(args.decoding_sweep).open(
        encoding="utf-8", newline=""
    ) as input_file:
        decoding_rows = list(csv.DictReader(input_file))

    greedy = best_method(decoding_rows, "greedy")
    beam = best_method(decoding_rows, "beam")
    beam_lm = best_method(decoding_rows, "beam_lm")
    test_clean_wer = prediction_wer(Path(args.test_clean))
    test_other_wer = prediction_wer(Path(args.test_other))

    summary = {
        "selected_acoustic_candidate": acoustic["candidate"],
        "selected_acoustic_id": acoustic["candidate_id"],
        "cv_mean_wer": acoustic["mean"],
        "cv_std_wer": acoustic["std"],
        "validation_greedy_wer": value(greedy, "validation_wer"),
        "validation_best_beam_wer": value(beam, "validation_wer"),
        "validation_best_lm_fusion_wer": value(beam_lm, "validation_wer"),
        "test_clean_wer": test_clean_wer,
        "test_other_wer": test_other_wer,
        "decoder_type": decoder["decoding_method"],
        "beam_width": decoder.get("beam_width", ""),
        "alpha": decoder.get("alpha", ""),
        "beta": decoder.get("beta", ""),
        "checkpoint_path": "outputs/base_strict_final/best_model",
    }
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    candidate_names = {
        "H": "two_stage_head_warmup",
        "F": "lr2e-4_freeze_feature",
    }
    aggregates = {}
    for candidate_id, candidate in candidate_names.items():
        scores = [
            float(row["best_wer"])
            for row in cv_rows
            if row["candidate_id"] == candidate_id
        ]
        if scores:
            aggregates[candidate_id] = {
                "candidate": candidate,
                "count": len(scores),
                "mean": sum(scores) / len(scores),
                "std": statistics.pstdev(scores),
                "min": min(scores),
                "max": max(scores),
            }
    scores_by_fold = {
        candidate_id: {
            int(row["fold"]): float(row["best_wer"])
            for row in cv_rows
            if row["candidate_id"] == candidate_id
        }
        for candidate_id in candidate_names
    }
    paired_folds = sorted(
        set(scores_by_fold["H"]).intersection(scores_by_fold["F"])
    )
    h_pair_wins = sum(
        scores_by_fold["H"][fold] < scores_by_fold["F"][fold]
        for fold in paired_folds
    )

    lines = [
        "# Final Strict Wav2Vec2-base ASR Report",
        "",
        "## Checkpoint provenance",
        "",
        "- Main initialization: `facebook/wav2vec2-base`.",
        "- `facebook/wav2vec2-base-960h` was not used in main results.",
        "- No supervised LibriSpeech ASR-fine-tuned checkpoint was used.",
        "",
        "## Strict data usage",
        "",
        "- Final acoustic training: train shards `000000`–`000003`.",
        "- Checkpoint selection and decoder tuning: validation shard `000004` only.",
        "- Test-clean and test-other were used only after decoder selection.",
        "",
        "## SpecAugment",
        "",
        "- Default SpecAugment caused blank collapse in tiny-overfit diagnostics.",
        "- Weak SpecAugment succeeded in tiny-overfit and was retained.",
        "- Final setting: SpecAugment ON, `mask_time_prob=0.01`, "
        "`mask_time_length=5`, `min_masks=1`.",
        "",
        "## Five-fold acoustic CV",
        "",
        "| ID | Candidate | Fold | Best WER | Blank | Nonblank | Empty |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        cv_rows, key=lambda item: (item["candidate_id"], int(item["fold"]))
    ):
        lines.append(
            f"| {row['candidate_id']} | `{row['candidate']}` | {row['fold']} | "
            f"{float(row['best_wer']):.6f} | "
            f"{float(row['blank_rate_at_best']):.6f} | "
            f"{float(row['nonblank_rate_at_best']):.6f} | "
            f"{float(row['empty_hypothesis_rate_at_best']):.6f} |"
        )
    lines.extend(
        [
            "",
            "| ID | Count | Mean WER | Std WER | Min WER | Max WER |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_id in ("H", "F"):
        aggregate = aggregates[candidate_id]
        lines.append(
            f"| {candidate_id} | {aggregate['count']} | "
            f"{aggregate['mean']:.6f} | {aggregate['std']:.6f} | "
            f"{aggregate['min']:.6f} | {aggregate['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### H vs F paired-fold comparison",
            "",
            "| Fold | H WER | F WER | Winner | Absolute difference |",
            "| ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for fold in paired_folds:
        h_wer = scores_by_fold["H"][fold]
        f_wer = scores_by_fold["F"][fold]
        winner = "H" if h_wer < f_wer else "F" if f_wer < h_wer else "Tie"
        lines.append(
            f"| {fold} | {h_wer:.6f} | {f_wer:.6f} | {winner} | "
            f"{abs(h_wer - f_wer):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected acoustic configuration: **{acoustic['candidate_id']}** "
            f"(`{acoustic['candidate']}`), based on its lower five-fold mean "
            f"WER and {h_pair_wins}/{len(paired_folds)} paired-fold wins over F.",
            "",
            "## Final acoustic checkpoint",
            "",
            "- `outputs/base_strict_final/best_model`",
            "",
            "## Validation decoding",
            "",
            "| Decoder | Beam width | Alpha | Beta | Validation WER |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Greedy |  |  |  | {float(value(greedy, 'validation_wer')):.6f} |",
            (
                f"| Beam | {value(beam, 'beam_width')} |  |  | "
                f"{float(value(beam, 'validation_wer')):.6f} |"
            ),
            (
                f"| Train-text trigram LM fusion | {value(beam_lm, 'beam_width')} | "
                f"{value(beam_lm, 'alpha')} | {value(beam_lm, 'beta')} | "
                f"{float(value(beam_lm, 'validation_wer')):.6f} |"
            ),
            "",
            "Selected decoder:",
            "",
            "```json",
            json.dumps(decoder, indent=2, sort_keys=True),
            "```",
            "",
            "## Final test results",
            "",
            "| Split | WER |",
            "| --- | ---: |",
            f"| test-clean | {test_clean_wer:.6f} |",
            f"| test-other | {test_other_wer:.6f} |",
            "",
            "## Limitations",
            "",
            "- The acoustic training set is small.",
            "- No supervised LibriSpeech ASR checkpoint was used.",
            "- The language model was trained only on final train-shard transcripts.",
            "- Decoder hyperparameters were tuned only on validation shard `000004`.",
            "- Results are not directly comparable to supervised LibriSpeech-960h checkpoints.",
            "",
        ]
    )
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"test-clean WER: {test_clean_wer:.6f}")
    print(f"test-other WER: {test_other_wer:.6f}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
