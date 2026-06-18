"""Checkpoint provenance guards for strict and diagnostic experiments."""

import json
import os
from typing import Dict


ALLOWED_MAIN_SOURCE = "facebook/wav2vec2-base"
SUPERVISED_LIBRISPEECH_MARKERS = (
    "960h",
    "asr_pretrained",
    "asrinit",
)
PROVENANCE_FILENAME = "checkpoint_provenance.json"


def is_supervised_librispeech_checkpoint(model_name_or_path: str) -> bool:
    """Return whether a checkpoint path looks LibriSpeech-ASR supervised."""
    normalized = model_name_or_path.lower()
    return any(marker in normalized for marker in SUPERVISED_LIBRISPEECH_MARKERS)


def read_checkpoint_provenance(model_name_or_path: str) -> Dict:
    """Read local checkpoint provenance metadata when present."""
    provenance_path = os.path.join(model_name_or_path, PROVENANCE_FILENAME)
    if not os.path.isfile(provenance_path):
        return {}
    with open(provenance_path, "r", encoding="utf-8") as provenance_file:
        return json.load(provenance_file)


def validate_checkpoint_role(
    model_name_or_path: str,
    experiment_role: str,
    *,
    require_base_source: bool = False,
) -> Dict:
    """Reject supervised LibriSpeech checkpoints outside positive controls."""
    if (
        is_supervised_librispeech_checkpoint(model_name_or_path)
        and experiment_role != "positive_control_only"
    ):
        raise ValueError(
            "Checkpoint provenance guard rejected "
            f"{model_name_or_path!r}. Checkpoints containing '960h', "
            "'asr_pretrained', or 'asrinit' are supervised LibriSpeech ASR "
            "artifacts and are allowed only with "
            "--experiment_role positive_control_only."
        )

    provenance = read_checkpoint_provenance(model_name_or_path)
    if experiment_role != "main":
        return provenance

    if require_base_source:
        if model_name_or_path != ALLOWED_MAIN_SOURCE:
            raise ValueError(
                "Main training must initialize exactly from "
                f"{ALLOWED_MAIN_SOURCE!r}; got {model_name_or_path!r}."
            )
        return provenance

    if model_name_or_path == ALLOWED_MAIN_SOURCE:
        return provenance
    if provenance.get("main_source_checkpoint") != ALLOWED_MAIN_SOURCE:
        raise ValueError(
            "Main inference requires either the exact unsupervised base source "
            f"{ALLOWED_MAIN_SOURCE!r} or a local checkpoint containing "
            f"{PROVENANCE_FILENAME} with matching main_source_checkpoint."
        )
    if provenance.get("experiment_role") != "main":
        raise ValueError("Local checkpoint provenance is not marked as a main run.")
    return provenance


def write_checkpoint_provenance(output_dir: str, metadata: Dict) -> str:
    """Write checkpoint provenance beside model and processor files."""
    os.makedirs(output_dir, exist_ok=True)
    provenance_path = os.path.join(output_dir, PROVENANCE_FILENAME)
    with open(provenance_path, "w", encoding="utf-8") as provenance_file:
        json.dump(metadata, provenance_file, indent=2, sort_keys=True)
        provenance_file.write("\n")
    return provenance_path
