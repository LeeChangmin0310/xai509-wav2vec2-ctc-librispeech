#!/usr/bin/env python3
"""Create the final acoustic namespace from the selected fold-4 CV model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-config",
        default="results/base_strict_cv/selected_acoustic_config.json",
    )
    parser.add_argument("--final-output-root", default="outputs/base_strict_final")
    parser.add_argument("--final-results-root", default="results/base_strict_final")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    selected_path = Path(args.selected_config)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not selected:
        raise RuntimeError("No acoustic candidate has been selected.")
    if selected.get("main_source_checkpoint") != "facebook/wav2vec2-base":
        raise ValueError("Selected config is not rooted in facebook/wav2vec2-base.")

    source_model = Path(selected["selected_fold4_model_path"]).resolve()
    if not (source_model / "config.json").is_file():
        raise FileNotFoundError(f"Missing selected fold4 model: {source_model}")

    output_root = Path(args.final_output_root)
    destination_model = output_root / "best_model"
    output_root.mkdir(parents=True, exist_ok=True)
    if destination_model.is_symlink():
        if destination_model.resolve() != source_model:
            if not args.force:
                raise FileExistsError(
                    f"{destination_model} points to a different model."
                )
            destination_model.unlink()
    elif destination_model.exists():
        if not args.force:
            raise FileExistsError(
                f"{destination_model} already exists and is not a symlink."
            )
        if destination_model.is_dir():
            shutil.rmtree(destination_model)
        else:
            destination_model.unlink()
    if not destination_model.exists():
        relative_source = os.path.relpath(source_model, destination_model.parent)
        destination_model.symlink_to(relative_source, target_is_directory=True)

    candidate = selected["candidate"]
    source_result = Path("results/base_strict_cv") / candidate / "fold_4"
    final_results = Path(args.final_results_root)
    final_results.mkdir(parents=True, exist_ok=True)
    copy_file(
        source_result / "validation_wer_history.csv",
        final_results / "validation_wer_history.csv",
    )
    copy_file(
        source_result / "run_metadata.json",
        final_results / "acoustic_run_metadata.json",
    )
    copy_file(
        Path("results/base_strict_cv/strict_cv_summary.csv"),
        final_results / "strict_cv_summary.csv",
    )
    copy_file(
        selected_path,
        final_results / "selected_acoustic_config.json",
    )
    print(f"Selected acoustic candidate: {selected['candidate_id']} ({candidate})")
    print(f"Final acoustic checkpoint: {destination_model}")


if __name__ == "__main__":
    main()
