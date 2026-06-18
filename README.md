# Strict Wav2Vec2-base ASR

This repository fine-tunes the unsupervised `facebook/wav2vec2-base` checkpoint
for LibriSpeech-style CTC recognition. The final submission keeps only the
reproducible training/evaluation code, compact result summaries, and reports.

## Strict checkpoint protocol

- Main runs initialize from `facebook/wav2vec2-base`.
- Training uses `data/train/shard-000000.tar` through
  `data/train/shard-000003.tar`.
- `data/train/shard-000004.tar` is reserved for acoustic checkpoint selection
  and decoder tuning.
- `test-clean` and `test-other` are evaluated only after the acoustic checkpoint
  and decoder are frozen.
- The selected H recipe trains the CTC head for 10 epochs, then trains the
  encoder for up to 40 epochs with weak SpecAugment and validation-based early
  stopping.
- Every saved main checkpoint includes `checkpoint_provenance.json`.
  `src/guard.py` rejects main evaluation of local checkpoints without matching
  provenance.
- `facebook/wav2vec2-base-960h` and other supervised LibriSpeech ASR
  checkpoints are not used for main training or evaluation.

## Setup

```bash
python -m pip install -r requirements.txt
```

GPU selection is controlled with `GPU_ID`, and `PYTHON` can override the Python
executable. Set `LOCAL_FILES_ONLY=1` to require an already cached base model.

## Data layout

Dataset archives are local-only and ignored by Git:

```text
data/
  train/
    shard-000000.tar
    shard-000001.tar
    shard-000002.tar
    shard-000003.tar
    shard-000004.tar
  test-clean/
    *.tar
  test-other/
    *.tar
```

See `data/README.md` for details.

## Training

```bash
GPU_ID=0 bash scripts/run_train.sh
```

This reproduces H / `two_stage_head_warmup` and writes generated checkpoints
under ignored `outputs/strict_final/`.

## Validation decoding

```bash
GPU_ID=0 bash scripts/run_validation_decode.sh
```

The script trains a trigram LM from train-shard transcripts only, sweeps decoder
settings on shard `000004`, and updates
`results/selected_decoder_config.json`.

## Final test evaluation

```bash
GPU_ID=0 bash scripts/run_test_eval.sh
```

Raw predictions remain under ignored `outputs/`; the compact WER row is written
to `results/strict_final_summary.csv`.

## Strict final result

| Acoustic configuration | Decoder | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| H / `two_stage_head_warmup` | beam + train-text trigram LM | 0.228487 | 0.246386 | 0.329289 |

## Exploratory results

```bash
GPU_ID=0 bash scripts/run_exploratory.sh
```

| Experiment | test-clean WER | test-other WER | Interpretation |
| --- | ---: | ---: | --- |
| H all-train | 0.216867 | 0.303307 | Fixed H recipe on all five train shards |
| H-fold ROVER ensemble | 0.197314 | 0.271383 | Best observed exploratory result; report only |

The ROVER ensemble has validation leakage: folds 0–3 trained on shard `000004`,
so its ensemble selection was not based on an unbiased held-out validation set.
It is therefore reported only as the best observed exploratory result and does
not replace the strict final result.

## Repository layout

```text
src/       training, inference, evaluation, guards, LM, and data utilities
scripts/   four reproducible shell entrypoints
data/      local dataset instructions only
results/   compact CSV/JSON artifacts only
reports/   strict and exploratory reports
```
