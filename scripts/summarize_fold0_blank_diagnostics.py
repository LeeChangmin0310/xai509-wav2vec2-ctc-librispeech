#!/usr/bin/env python3
"""Combine candidate A/B/C fold-0 blank diagnostics into one CSV."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "base_strict_cv"
CANDIDATES = (
    ("A", "lr3e-5_freeze_feature"),
    ("B", "lr5e-5_freeze_feature"),
    ("C", "lr1e-4_freeze_feature"),
)


def main() -> None:
    rows = []
    source_fields = None
    for candidate_id, candidate_name in CANDIDATES:
        input_path = RESULT_ROOT / candidate_name / "fold_0" / "blank_diagnostics.csv"
        with input_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            source_fields = reader.fieldnames
            for row in reader:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        **row,
                    }
                )

    if not rows or source_fields is None:
        raise RuntimeError("No fold-0 blank diagnostic rows found.")
    output_path = RESULT_ROOT / "fold0_blank_diagnostics_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["candidate_id", "candidate_name", *source_fields],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
