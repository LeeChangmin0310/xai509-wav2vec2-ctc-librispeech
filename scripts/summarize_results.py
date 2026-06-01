#!/usr/bin/env python
"""Create Markdown and optional plot summaries from the WER CSV."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional


EXPERIMENT_ORDER = [
    "baseline_lr1e-4",
    "lr1e-5",
    "lr5e-5",
    "freeze_feature_lr1e-4",
    "freeze3_lr1e-4",
    "freeze6_lr1e-4",
]


def parse_args():
    """Parse summary input and output paths."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=project_root / "results" / "wer_summary.csv",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=project_root / "results" / "wer_summary.md",
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=project_root / "results" / "figures" / "wer_barplot.png",
    )
    return parser.parse_args()


def parse_score(value: str) -> Optional[float]:
    """Parse a CSV score, preserving empty or malformed cells as missing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sorted_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Sort known experiments first, followed by any additional rows."""
    order = {name: index for index, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            order.get(row.get("experiment", ""), len(order)),
            row.get("experiment", ""),
        ),
    )


def best_score(rows: List[Dict[str, str]], field: str) -> Optional[float]:
    """Return the lowest available WER for one result column."""
    scores = [parse_score(row.get(field, "")) for row in rows]
    valid_scores = [score for score in scores if score is not None]
    return min(valid_scores) if valid_scores else None


def format_score(score: Optional[float], best: Optional[float]) -> str:
    """Format a score and bold the best value."""
    if score is None:
        return ""
    value = f"{score:.6f}"
    if best is not None and math.isclose(score, best, rel_tol=0.0, abs_tol=1e-12):
        return f"**{value}**"
    return value


def write_markdown(rows: List[Dict[str, str]], output_md: Path):
    """Write a fixed-order Markdown WER table."""
    best_clean = best_score(rows, "test_clean_wer")
    best_other = best_score(rows, "test_other_wer")
    lines = [
        "# WER Summary",
        "",
        "| Experiment | test-clean WER | test-other WER |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        clean = parse_score(row.get("test_clean_wer", ""))
        other = parse_score(row.get("test_other_wer", ""))
        lines.append(
            f"| `{row.get('experiment', '')}` | "
            f"{format_score(clean, best_clean)} | "
            f"{format_score(other, best_other)} |"
        )
    lines.extend(["", "Best values are shown in bold.", ""])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def write_plot(rows: List[Dict[str, str]], output_plot: Path):
    """Write a grouped WER bar plot when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping WER plot.")
        return False

    labels = [row.get("experiment", "") for row in rows]
    clean_scores = [parse_score(row.get("test_clean_wer", "")) for row in rows]
    other_scores = [parse_score(row.get("test_other_wer", "")) for row in rows]
    clean_scores = [
        score if score is not None else float("nan") for score in clean_scores
    ]
    other_scores = [
        score if score is not None else float("nan") for score in other_scores
    ]
    positions = list(range(len(rows)))
    width = 0.4

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        [position - width / 2 for position in positions],
        clean_scores,
        width,
        label="test-clean",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        other_scores,
        width,
        label="test-other",
    )
    axis.set_ylabel("WER")
    axis.set_title("Wav2Vec2 CTC experiment WER")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.legend()
    figure.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_plot, dpi=150)
    plt.close(figure)
    return True


def main():
    """Load the CSV and generate available summary artifacts."""
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"WER summary not found: {args.input_csv}")

    with args.input_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = sorted_rows(list(csv.DictReader(csv_file)))
    if not rows:
        raise ValueError(f"WER summary has no experiment rows: {args.input_csv}")

    write_markdown(rows, args.output_md)
    print(f"Wrote Markdown summary: {args.output_md}")
    if write_plot(rows, args.output_plot):
        print(f"Wrote WER plot: {args.output_plot}")


if __name__ == "__main__":
    main()
