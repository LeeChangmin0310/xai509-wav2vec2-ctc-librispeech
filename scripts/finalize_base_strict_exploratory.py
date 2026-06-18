#!/usr/bin/env python3
"""Build the post-final strict-compatible exploratory ASR report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-summary",
        default="results/base_strict_final/wer_summary.csv",
    )
    parser.add_argument(
        "--ensemble-validation",
        default=(
            "results/base_strict_exploratory/"
            "h_fold_ensemble_validation.csv"
        ),
    )
    parser.add_argument(
        "--ensemble-test",
        default=(
            "results/base_strict_exploratory/"
            "h_fold_ensemble_test_summary.csv"
        ),
    )
    parser.add_argument(
        "--alltrain-summary",
        default=(
            "results/base_strict_exploratory/"
            "h_alltrain_wer_summary.csv"
        ),
    )
    parser.add_argument(
        "--h2-summary",
        default=(
            "results/base_strict_exploratory/"
            "head_refinement_h2/validation_summary.csv"
        ),
    )
    parser.add_argument(
        "--h3-summary",
        default=(
            "results/base_strict_exploratory/"
            "head_refinement_h3/validation_summary.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default="reports/base_strict_exploratory_report.md",
    )
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def fmt(value: str | float) -> str:
    return f"{float(value):.6f}"


def yes_no(value: str | bool) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "yes" if value.lower() == "true" else "no"


def main() -> None:
    args = parse_args()
    main_row = read_rows(args.main_summary)[0]
    validation_rows = read_rows(args.ensemble_validation)
    ensemble_test_rows = read_rows(args.ensemble_test)
    alltrain = read_rows(args.alltrain_summary)[0]
    diagnostics = []
    for path in (args.h2_summary, args.h3_summary):
        if Path(path).is_file():
            diagnostics.extend(read_rows(path))

    ensemble_candidates = [
        row
        for row in validation_rows
        if row["method_type"] == "ensemble_candidate"
    ]
    selected_ensemble = min(
        ensemble_candidates, key=lambda row: float(row["validation_wer"])
    )
    ensemble_by_split = {row["split"]: row for row in ensemble_test_rows}
    ensemble_improved = all(
        row["improved_over_main"].lower() == "true"
        for row in ensemble_test_rows
    )
    alltrain_improved = (
        alltrain["test_clean_improved_over_main"].lower() == "true"
        and alltrain["test_other_improved_over_main"].lower() == "true"
    )

    lines = [
        "# Strict Wav2Vec2-base Post-final Exploratory Report",
        "",
        "These are post-final exploratory experiments. The preserved main result "
        "remains the strict final result; none of its result files or decoder "
        "selection were replaced.",
        "",
        "## 1. Preserved strict final result",
        "",
        "| Acoustic configuration | Decoder | Validation WER | test-clean WER | test-other WER |",
        "| --- | --- | ---: | ---: | ---: |",
        (
            f"| H / `{main_row['selected_acoustic_candidate']}` | "
            f"beam + train-text trigram LM "
            f"(beam {main_row['beam_width']}, alpha {main_row['alpha']}, "
            f"beta {main_row['beta']}) | "
            f"{fmt(main_row['validation_best_lm_fusion_wer'])} | "
            f"{fmt(main_row['test_clean_wer'])} | "
            f"{fmt(main_row['test_other_wer'])} |"
        ),
        "",
        "- Main initialization remained `facebook/wav2vec2-base`.",
        "- `facebook/wav2vec2-base-960h` and every supervised LibriSpeech "
        "ASR checkpoint remained forbidden.",
        "- No official LibriSpeech LM or pretrained neural LM was used.",
        "- The fixed decoder and LM were reused read-only from the preserved main run.",
        "",
        "## 2. Exploratory H-fold ensemble",
        "",
        "Five existing H fold models were combined using raw-logit averaging and "
        "a deterministic progressive word-alignment/voting fallback.",
        "",
        "### Validation diagnostics on shard 000004",
        "",
        "| Method | Type | Validation WER |",
        "| --- | --- | ---: |",
    ]
    for row in validation_rows:
        lines.append(
            f"| `{row['method']}` | {row['method_type']} | "
            f"{fmt(row['validation_wer'])} |"
        )
    lines.extend(
        [
            "",
            f"Selected exploratory ensemble: **{selected_ensemble['method']}** "
            f"with validation WER {fmt(selected_ensemble['validation_wer'])}.",
            "",
            "> Important leakage caveat: H folds 0–3 trained on shard 000004; "
            "only fold 4 held it out. The ensemble validation number is therefore "
            "optimistic and is not an unbiased held-out estimate.",
            "",
            "### Frozen-method test evaluation",
            "",
            "| Split | Ensemble WER | Preserved main WER | Change | Improved? |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for split in ("test-clean", "test-other"):
        row = ensemble_by_split[split]
        lines.append(
            f"| {split} | {fmt(row['wer'])} | {fmt(row['main_final_wer'])} | "
            f"{float(row['absolute_wer_change_vs_main']):+.6f} | "
            f"{yes_no(row['improved_over_main'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Exploratory H all-train model",
            "",
            "The model used all five train shards with the fixed H schedule: "
            "10 head-only epochs followed by 40 encoder-training epochs. "
            "The final epoch was retained without validation-based checkpoint "
            "selection. Evaluation used the preserved decoder and original "
            "train-shards-000000–000003 trigram LM unchanged.",
            "",
            "| Split | All-train WER | Preserved main WER | Change | Improved? |",
            "| --- | ---: | ---: | ---: | --- |",
            (
                f"| test-clean | {fmt(alltrain['test_clean_wer'])} | "
                f"{fmt(alltrain['main_test_clean_wer'])} | "
                f"{float(alltrain['test_clean_change_vs_main']):+.6f} | "
                f"{yes_no(alltrain['test_clean_improved_over_main'])} |"
            ),
            (
                f"| test-other | {fmt(alltrain['test_other_wer'])} | "
                f"{fmt(alltrain['main_test_other_wer'])} | "
                f"{float(alltrain['test_other_change_vs_main']):+.6f} | "
                f"{yes_no(alltrain['test_other_improved_over_main'])} |"
            ),
            "",
            "- Checkpoint: `outputs/base_strict_exploratory/h_alltrain/best_model`.",
            "- The directory name `best_model` satisfies the requested namespace; "
            "this artifact is the fixed final-epoch model, not a newly tuned "
            "validation-best checkpoint.",
            "",
            "## 4. Optional head-refinement validation diagnostics",
            "",
        ]
    )
    if diagnostics:
        lines.extend(
            [
                "| Variant | Training-selection greedy WER | Fixed-decoder validation WER | Test evaluated? |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in diagnostics:
            lines.append(
                f"| {row['variant']} | "
                f"{fmt(row['training_best_greedy_wer'])} | "
                f"{fmt(row['fixed_decoder_validation_wer'])} | "
                f"{yes_no(row['test_splits_evaluated'])} |"
            )
    else:
        lines.append("No optional refinement run completed.")
    lines.extend(
        [
            "",
            "H4 checkpoint averaging was skipped: the retained fold4 checkpoints "
            "were epochs 22, 29, and 30, so there was no nearby window around "
            "the best epoch-22 checkpoint suitable for local averaging.",
            "",
            "## Interpretation and limitations",
            "",
            "- These are post-final exploratory results, not replacements for the "
            "preserved strict final result.",
            "- No 960h or supervised LibriSpeech ASR checkpoint was used.",
            "- No external official LibriSpeech LM or pretrained neural LM was used.",
            "- The ensemble method was selected only from shard 000004, but that "
            "selection is affected by the fold-membership leakage described above.",
            "- Test-clean and test-other had already been observed in the preserved "
            "main run. Any later exploratory test comparison should therefore be "
            "interpreted cautiously even though no exploratory hyperparameters "
            "were tuned on test.",
            "- The all-train model has no untouched in-domain validation split.",
            "",
            "## Output paths",
            "",
            "- Ensemble validation: `results/base_strict_exploratory/h_fold_ensemble_validation.csv`",
            "- Ensemble test: `results/base_strict_exploratory/h_fold_ensemble_test_summary.csv`",
            "- All-train summary: `results/base_strict_exploratory/h_alltrain_wer_summary.csv`",
            "- All-train checkpoint: `outputs/base_strict_exploratory/h_alltrain/best_model`",
            "- This report: `reports/base_strict_exploratory_report.md`",
            "",
        ]
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Exploratory validation results:")
    for row in ensemble_candidates:
        print(f"  {row['method']}: {fmt(row['validation_wer'])}")
    for row in diagnostics:
        print(
            f"  {row['variant']}: "
            f"{fmt(row['fixed_decoder_validation_wer'])}"
        )
    print("Exploratory test results:")
    for row in ensemble_test_rows:
        print(f"  ensemble {row['split']}: {fmt(row['wer'])}")
    print(f"  all-train test-clean: {fmt(alltrain['test_clean_wer'])}")
    print(f"  all-train test-other: {fmt(alltrain['test_other_wer'])}")
    print(f"Ensemble improved over main on both splits: {ensemble_improved}")
    print(f"All-train improved over main on both splits: {alltrain_improved}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
