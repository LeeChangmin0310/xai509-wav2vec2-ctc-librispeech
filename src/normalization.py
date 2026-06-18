"""Shared transcript normalization for training, decoding, LM text, and WER."""

import re


NON_TRANSCRIPT_CHARACTERS = re.compile(r"[^A-Z' ]+")
MULTIPLE_SPACES = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """Normalize LibriSpeech text to the Wav2Vec2 CTC tokenizer alphabet."""
    normalized = text.upper().replace("’", "'").replace("`", "'")
    normalized = NON_TRANSCRIPT_CHARACTERS.sub(" ", normalized)
    return MULTIPLE_SPACES.sub(" ", normalized).strip()
