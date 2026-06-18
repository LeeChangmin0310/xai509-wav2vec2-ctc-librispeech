#!/usr/bin/env python3
"""Aggregate available strict F/H shard-CV validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


CANDIDATES = (
    ("H", "two_stage_head_warmup"),
    ("F", "lr2e-4_freeze_feature"),
)
METRIC_COLUMNS = (
    "eval_loss",
    "eval_blank_token_rate",
    "eval_nonblank_token_rate",
    "eval_empty_hypothesis_rate",
    "eval_average_hypothesis_character_length",
    "eval_average_hypothesis_word_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/base_strict_cv")
    parser.add_argument(
        "--csv-path",
        default="results/base_strict_cv/strict_cv_summary.csv",
    )
    parser.add_argument(
        "--report-path",
        default="reports/strict_cv_summary.md",
    )
    parser.add_argument(
        "--selected-config-path",
        default="results/base_strict_cv/selected_acoustic_config.json",
    )
    return parser.parse_args()


def optional_float(value: str | None) -> float | str:
    if value in (None, ""):
        return ""
    number = float(value)
    return number if math.isfinite(number) else ""


def read_best_row(history_path: Path) -> dict[str, str]:
    with history_path.open(encoding="utf-8", newline="") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row.get("eval_wer") not in (None, "")
        ]
    if not rows:
        raise ValueError(f"No eval_wer rows found in {history_path}")
    return min(rows, key=lambda row: float(row["eval_wer"]))


def format_number(value: object, digits: int = 6) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    rows: list[dict[str, object]] = []

    for candidate_id, candidate in CANDIDATES:
        candidate_root = results_root / candidate
        for history_path in sorted(candidate_root.glob("fold_*/validation_wer_history.csv")):
            fold_text = history_path.parent.name.removeprefix("fold_")
            if not fold_text.isdigit():
                continue
            best = read_best_row(history_path)
            metadata_path = history_path.parent / "run_metadata.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "candidate": candidate,
                "fold": int(fold_text),
                "best_wer": float(best["eval_wer"]),
                "best_epoch": optional_float(best.get("epoch")),
                "best_step": optional_float(best.get("step")),
                "eval_loss_at_best": optional_float(best.get("eval_loss")),
                "blank_rate_at_best": optional_float(
                    best.get("eval_blank_token_rate")
                ),
                "nonblank_rate_at_best": optional_float(
                    best.get("eval_nonblank_token_rate")
                ),
                "empty_hypothesis_rate_at_best": optional_float(
                    best.get("eval_empty_hypothesis_rate")
                ),
                "average_hypothesis_character_length_at_best": optional_float(
                    best.get("eval_average_hypothesis_character_length")
                ),
                "average_hypothesis_word_count_at_best": optional_float(
                    best.get("eval_average_hypothesis_word_count")
                ),
                "best_checkpoint_path": metadata.get("best_checkpoint_path", ""),
                "exported_model_path": metadata.get("saved_model_path", ""),
            }
            rows.append(row)

    rows.sort(key=lambda row: (str(row["candidate_id"]), int(row["fold"])))
    if not rows:
        raise SystemExit("No F/H validation history files were found.")

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for candidate_id, candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        scores = [float(row["best_wer"]) for row in candidate_rows]
        if not scores:
            continue
        aggregates.append(
            {
                "candidate_id": candidate_id,
                "candidate": candidate,
                "count": len(scores),
                "mean": statistics.mean(scores),
                "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
                "blank_fold_count": sum(
                    float(row["nonblank_rate_at_best"] or 0.0) == 0.0
                    for row in candidate_rows
                ),
                "empty_issue_fold_count": sum(
                    float(row["empty_hypothesis_rate_at_best"] or 0.0) > 0.0
                    for row in candidate_rows
                ),
            }
        )

    rows_by_candidate_fold = {
        (str(row["candidate_id"]), int(row["fold"])): row for row in rows
    }
    paired_rows = []
    for fold in range(5):
        h_row = rows_by_candidate_fold.get(("H", fold))
        f_row = rows_by_candidate_fold.get(("F", fold))
        if h_row is None or f_row is None:
            continue
        h_wer = float(h_row["best_wer"])
        f_wer = float(f_row["best_wer"])
        paired_rows.append(
            {
                "fold": fold,
                "h_wer": h_wer,
                "f_wer": f_wer,
                "winner": "H" if h_wer < f_wer else "F" if f_wer < h_wer else "tie",
                "absolute_difference": abs(h_wer - f_wer),
            }
        )

    complete_aggregates = {
        aggregate["candidate_id"]: aggregate
        for aggregate in aggregates
        if aggregate["count"] == 5
    }
    selected = None
    selection_reason = "Both H and F must complete all five folds."
    if {"H", "F"} <= complete_aggregates.keys():
        h_aggregate = complete_aggregates["H"]
        f_aggregate = complete_aggregates["F"]
        mean_gap = abs(h_aggregate["mean"] - f_aggregate["mean"])
        if mean_gap <= 0.001:
            ordered = sorted(
                (h_aggregate, f_aggregate),
                key=lambda item: (
                    item["std"],
                    item["blank_fold_count"] + item["empty_issue_fold_count"],
                    item["mean"],
                ),
            )
            selected = ordered[0]
            selection_reason = (
                "Mean WERs were within 0.001, so lower standard deviation and "
                "fewer blank/empty issues were used as tie-breakers."
            )
        else:
            selected = min(
                (h_aggregate, f_aggregate),
                key=lambda item: item["mean"],
            )
            selection_reason = "Selected by lower five-fold mean validation WER."

    selected_payload = None
    if selected is not None:
        selected_payload = {
            **selected,
            "selection_reason": selection_reason,
            "selected_fold4_model_path": str(
                Path("outputs/base_strict_cv")
                / selected["candidate"]
                / "fold_4"
                / "best_model"
            ),
            "final_train_shards": [
                f"data/train/shard-{index:06d}.tar" for index in range(4)
            ],
            "final_validation_shard": "data/train/shard-000004.tar",
            "main_source_checkpoint": "facebook/wav2vec2-base",
            "test_splits_used_for_selection": False,
        }
    selected_config_path = Path(args.selected_config_path)
    selected_config_path.parent.mkdir(parents=True, exist_ok=True)
    selected_config_path.write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Strict Wav2Vec2-base CV Summary",
        "",
        "`strict_cv_summary.csv` did not previously exist and is now generated.",
        "Fold 0 already existed before this CV expansion.",
        "",
        "## Per-fold results",
        "",
        "| ID | Candidate | Fold | Best WER | Epoch | Step | Eval loss | Blank | Nonblank | Empty |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_id']} | `{row['candidate']}` | {row['fold']} | "
            f"{format_number(row['best_wer'])} | "
            f"{format_number(row['best_epoch'], 1)} | "
            f"{format_number(row['best_step'], 0)} | "
            f"{format_number(row['eval_loss_at_best'])} | "
            f"{format_number(row['blank_rate_at_best'])} | "
            f"{format_number(row['nonblank_rate_at_best'])} | "
            f"{format_number(row['empty_hypothesis_rate_at_best'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate results",
            "",
        "| ID | Candidate | Count | Mean WER | Std WER | Min WER | Max WER | Blank folds | Empty-issue folds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for aggregate in aggregates:
        lines.append(
            f"| {aggregate['candidate_id']} | `{aggregate['candidate']}` | "
            f"{aggregate['count']} | {aggregate['mean']:.6f} | "
            f"{aggregate['std']:.6f} | {aggregate['min']:.6f} | "
            f"{aggregate['max']:.6f} | {aggregate['blank_fold_count']} | "
            f"{aggregate['empty_issue_fold_count']} |"
        )
    lines.extend(
        [
            "",
            "## Paired fold comparison",
            "",
            "| Fold | H WER | F WER | Winner | Absolute difference |",
            "| ---: | ---: | ---: | --- | ---: |",
        ]
    )
    if paired_rows:
        for pair in paired_rows:
            lines.append(
                f"| {pair['fold']} | {pair['h_wer']:.6f} | "
                f"{pair['f_wer']:.6f} | {pair['winner']} | "
                f"{pair['absolute_difference']:.6f} |"
            )
    else:
        lines.append("|  |  |  | Paired results incomplete |  |")
    lines.extend(
        [
            "",
            "## Selected acoustic configuration",
            "",
            (
                "Selection pending until both candidates complete five folds."
                if selected is None
                else f"Selected **{selected['candidate_id']}** "
                f"(`{selected['candidate']}`): {selection_reason} "
                f"Mean WER `{selected['mean']:.6f}`, std `{selected['std']:.6f}`."
            ),
            "",
            "Only the five train-shard folds were used. "
            "Test-clean and test-other were not used.",
            "`facebook/wav2vec2-base-960h` and other supervised LibriSpeech "
            "ASR checkpoints were not used.",
            "",
        ]
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {selected_config_path}")


if __name__ == "__main__":
    main()
