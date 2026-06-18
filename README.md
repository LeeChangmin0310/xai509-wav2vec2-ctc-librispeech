# XAI509 Automatic Speech Recognition Semester Project

**Wav2Vec2-CTC Fine-tuning on the Provided LibriSpeech WebDataset**

## Overview

This repository contains the final semester project for the **XAI509 Automatic
Speech Recognition** course. It implements an end-to-end experimental pipeline
for fine-tuning Wav2Vec2 with CTC, decoding speech, and evaluating word error
rate (WER) on the provided LibriSpeech WebDataset shards.

The main model starts from `facebook/wav2vec2-base`. The supervised
`facebook/wav2vec2-base-960h` checkpoint is not used for the main results, and
the code records and validates checkpoint provenance to enforce that policy.

## Repository structure

```text
README.md                    Project overview and reproduction guide
requirements.txt             Python dependencies
src/                         Training, inference, WER, data, LM, and guard code
scripts/                     Reproducible shell entrypoints
data/README.md               Expected local WebDataset layout
results/                     Compact final CSV and JSON artifacts
reports/                     Concise final and exploratory reports
docs/EXPERIMENT_SUMMARY.md   Full experimental process and interpretation
```

Generated checkpoints, predictions, decoder sweeps, logs, and dataset archives
are intentionally excluded from Git.

## Installation

Python 3.10 and a CUDA-capable PyTorch environment are recommended.

```bash
python -m pip install -r requirements.txt
```

Set `PYTHON` to choose another Python executable, `GPU_ID` to select a GPU, and
`LOCAL_FILES_ONLY=1` to require locally cached Hugging Face files.

## Data layout

Place the provided WebDataset archives at:

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

The main data split protocol trains on shards `000000`–`000003` and reserves
shard `000004` for checkpoint selection and decoder tuning. `test-clean` and
`test-other` are used only for final evaluation. See
[data/README.md](data/README.md) for the local-data contract.

## Quick start

Run the main pipeline in order:

```bash
GPU_ID=0 bash scripts/run_train.sh
GPU_ID=0 bash scripts/run_validation_decode.sh
GPU_ID=0 bash scripts/run_test_eval.sh
```

Run the optional all-train experiment with:

```bash
GPU_ID=0 bash scripts/run_exploratory.sh
```

## Main pipeline

1. `run_train.sh` reproduces configuration H: a 10-epoch CTC-head warmup
   followed by encoder fine-tuning with weak SpecAugment.
2. `run_validation_decode.sh` trains a word trigram LM only from the main
   training transcripts and selects decoder settings on shard `000004`.
3. `run_test_eval.sh` freezes that selection, evaluates `test-clean` and
   `test-other`, and writes the compact WER summary.

Main experiments initialize from `facebook/wav2vec2-base`. Checkpoints marked
as `960h`, `asrinit`, or `asr_pretrained` are rejected from main evaluation by
the checkpoint policy in `src/guard.py`.

## Key results

| Setting | Role | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| H single model + beam LM | Main reproducible result | 0.228487 | 0.246386 | 0.329289 |
| H all-train | Exploratory all-train | — | 0.216867 | 0.303307 |
| H-fold ROVER ensemble | Best observed exploratory | — | 0.197314 | 0.271383 |

The selected main decoder uses beam width 50 with the train-text trigram LM,
alpha 0.3, and beta 1.5.

## Exploratory results

The H all-train model applies the fixed H recipe to all five train shards and
retains the final epoch without validation-based checkpoint selection.

The H-fold ROVER ensemble achieved the best observed test WER, but it is
reported separately as exploratory because its validation selection is
affected by fold-membership leakage: folds 0–3 trained on shard `000004`.
ROVER is therefore not presented as the clean main validation-selected model.

## Detailed experiment summary

The complete experimental narrative—including blank-collapse failures,
SpecAugment diagnosis, H/F cross-validation, decoder tuning, final results, and
limitations—is in
[docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md).
