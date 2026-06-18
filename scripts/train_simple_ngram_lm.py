#!/usr/bin/env python3
"""Train a small pyctcdecode-compatible word n-gram LM from train shards."""

import argparse
import os
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sample_util
from simple_ngram_lm import SimpleWordNGramLanguageModel
from text_normalization import normalize_transcript


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_shards", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_output_path")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--add_k", type=float, default=0.1)
    return parser.parse_args()


def read_transcripts(shard_spec):
    transcripts = []
    for shard_path in sample_util.find_shards(shard_spec):
        with tarfile.open(shard_path, "r:*") as shard:
            for member in shard.getmembers():
                if member.isfile() and member.name.endswith(".text"):
                    transcript = shard.extractfile(member).read().decode("utf-8")
                    transcripts.append(normalize_transcript(transcript))
    return transcripts


def main():
    args = parse_args()
    if args.order < 1:
        raise ValueError("--order must be at least 1")
    transcripts = read_transcripts(args.train_shards)
    if not transcripts:
        raise RuntimeError("No transcripts were found in the LM training shards")
    language_model = SimpleWordNGramLanguageModel.fit(
        transcripts,
        order=args.order,
        add_k=args.add_k,
    )
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    language_model.save(args.output_path)
    if args.text_output_path:
        os.makedirs(os.path.dirname(args.text_output_path) or ".", exist_ok=True)
        with open(args.text_output_path, "w", encoding="utf-8") as output_file:
            for transcript in transcripts:
                output_file.write(f"{transcript}\n")
    print(f"Trained {args.order}-gram LM on {len(transcripts)} train transcripts")
    print(f"Vocabulary size: {len(language_model.vocabulary)}")
    print(f"Saved LM: {args.output_path}")


if __name__ == "__main__":
    main()
