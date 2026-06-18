"""Utilities for loading LibriSpeech WebDataset shards."""

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import io
import itertools
import os
import tarfile
from typing import Dict, Iterable, List, Optional, Union

import torch
import torchaudio
import webdataset as wds

from text_normalization import normalize_transcript


ShardInput = Union[str, Iterable[str]]


class SizedWebDataset(torch.utils.data.IterableDataset):
    """Wrap a WebDataset pipeline with a sample count for Trainer."""

    def __init__(
        self,
        dataset: wds.WebDataset,
        length: int,
        max_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.length = length
        self.max_samples = max_samples

    def __iter__(self):
        iterator = iter(self.dataset)
        if self.max_samples is None:
            return iterator
        return itertools.islice(iterator, self.max_samples)

    def __len__(self):
        return self.length


def _expand_shard_spec(shard_spec: str) -> List[str]:
    """Expand one shard spec as a directory, tar file, or glob pattern."""
    shard_spec = shard_spec.strip()
    if not shard_spec:
        return []
    if os.path.isdir(shard_spec):
        return glob.glob(os.path.join(shard_spec, "*.tar"))
    if os.path.isfile(shard_spec):
        return [shard_spec]
    return glob.glob(shard_spec)


def find_shards(shards: ShardInput) -> List[str]:
    """Resolve shard directories, globs, tar paths, comma lists, or iterables."""
    if isinstance(shards, str):
        shard_paths = list(
            itertools.chain.from_iterable(
                _expand_shard_spec(shard_spec)
                for shard_spec in shards.split(",")
            )
        )
    else:
        shard_paths = list(shards)

    shard_paths = sorted(
        dict.fromkeys(path for path in shard_paths if path.endswith(".tar"))
    )
    if not shard_paths:
        raise FileNotFoundError(f"No .tar WebDataset shards found for: {shards}")
    return shard_paths


def count_samples(
    shard_paths: Iterable[str], max_samples: Optional[int] = None
) -> int:
    """Count WebDataset sample keys by reading tar headers without extraction."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")

    sample_count = 0
    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r:*") as shard:
            keys = {
                member.name.rsplit(".", 1)[0]
                for member in shard.getmembers()
                if member.isfile() and "." in member.name
            }
        sample_count += len(keys)
        if max_samples is not None and sample_count >= max_samples:
            return max_samples
    return sample_count


def preprocess_sample(sample: Dict, processor, do_tokenization: bool = True) -> Dict:
    """Convert one raw WebDataset sample into model inputs and labels."""
    audio = sample["audio"]
    if isinstance(audio, (bytes, bytearray)):
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio))
    elif isinstance(audio, (tuple, list)) and len(audio) == 2:
        waveform, sample_rate = audio
    else:
        raise TypeError(f"Unsupported audio value type: {type(audio)!r}")

    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    input_values = processor.feature_extractor(
        waveform, sampling_rate=sample_rate
    ).input_values[0]

    text = sample["text"]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    text = normalize_transcript(text)

    labels = processor.tokenizer(text).input_ids if do_tokenization else text
    return {"input_values": input_values, "labels": labels}


def make_dataset(
    shards: ShardInput,
    processor,
    do_tokenization: bool = True,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    max_samples: Optional[int] = None,
) -> SizedWebDataset:
    """Create a sized WebDataset pipeline from final tar shards."""
    shard_paths = find_shards(shards)
    dataset = wds.WebDataset(shard_paths)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = (
        dataset.to_tuple("audio", "text", "meta")
        .map(
            lambda sample: {
                "audio": sample[0],
                "text": sample[1],
                "meta": sample[2],
            }
        )
        .map(lambda sample: preprocess_sample(sample, processor, do_tokenization))
    )
    return SizedWebDataset(
        dataset,
        count_samples(shard_paths, max_samples=max_samples),
        max_samples=max_samples,
    )
