# XAI 509 Wav2Vec2 CTC Project

This project fine-tunes Wav2Vec 2.0 with the provided custom CTC loss, runs
inference on LibriSpeech `test-clean` and `test-other`, and records WER for
seven training experiments plus optional beam decoding.

Final WebDataset `.tar` shards are read directly. Do not extract them.

## Current Status

- The project skeleton is implemented.
- Data shards are present under `data/`.
- Python and shell syntax checks pass.
- Full baseline training has not been started.
- Smoke and unattended single-GPU queue scripts are available.

## Environment Setup

The target GPU environment uses PyTorch `2.5.1+cu121` and torchaudio
`2.5.1+cu121`.

Using conda:

```bash
conda env create -f environment.yml
conda activate xai509-wav2vec2-ctc
```

Using an existing Python environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pyctcdecode` is optional and only needed for beam decoding. The included beam
path works without KenLM:

```bash
python -m pip install pyctcdecode
```

## Data Layout

The expected layout is:

```text
data/
  train/*.tar
  test-clean/*.tar
  test-other/*.tar
```

Verify shard counts and directory sizes:

```bash
bash scripts/check_data.sh
```

For automatic download, set the Google Drive folder URLs supplied by the
course:

```bash
export TRAIN_GDRIVE_URL="COURSE_TRAIN_FOLDER_URL"
export TEST_CLEAN_GDRIVE_URL="COURSE_TEST_CLEAN_FOLDER_URL"
export TEST_OTHER_GDRIVE_URL="COURSE_TEST_OTHER_FOLDER_URL"
bash scripts/download_data.sh
```

If `gdown` is unavailable or a Drive download fails, the script prints manual
setup instructions. Keep final `.tar` shards intact.

## Dry Runs

Dry runs validate shard paths and print the plan without loading a model,
training, or running inference:

```bash
python wav2vec_finetuning.py --dry_run
python wav2vec_inference.py \
  --model_name_or_path facebook/wav2vec2-base \
  --output_dir results/dry_run \
  --dry_run
```

## Smoke Tests

When a GPU is available, run a two-step training smoke test and bounded
inference/WER evaluation:

```bash
GPU_ID=0 bash scripts/run_smoke_train.sh
GPU_ID=0 bash scripts/run_smoke_inference_eval.sh
```

Defaults are two optimizer steps, four validation samples, and four inference
samples per test split. Override them when needed:

```bash
GPU_ID=0 \
SMOKE_TRAIN_STEPS=3 \
SMOKE_EVAL_SAMPLES=8 \
SMOKE_TEST_SAMPLES=8 \
bash scripts/run_smoke_train.sh
```

Smoke artifacts use:

```text
outputs/smoke/
results/smoke/
logs/smoke_train.log
logs/smoke_inference_eval.log
```

## Single-GPU RTX 3090 Queue

After smoke tests pass, launch all seven experiments sequentially on one GPU:

```bash
GPU_ID=0 FP16=1 CONTINUE_ON_ERROR=1 bash scripts/run_queue_3090.sh
```

The queue trains, transcribes both test splits, evaluates WER, and updates
`results/wer_summary.csv` for each experiment. It records stage timestamps and
writes failures to `logs/failed_experiments.txt`.

Completed checkpoints and inference result files are skipped by default. To
rerun completed work:

```bash
GPU_ID=0 FP16=1 FORCE=1 CONTINUE_ON_ERROR=1 bash scripts/run_queue_3090.sh
```

The required queue order is:

| Experiment | Learning rate | Freezing |
| --- | ---: | --- |
| `baseline_lr1e-4` | `1e-4` | None |
| `lr1e-5` | `1e-5` | None |
| `lr5e-5` | `5e-5` | None |
| `freeze_feature_lr1e-4` | `1e-4` | Feature encoder |
| `freeze3_lr1e-4` | `1e-4` | First 3 encoder layers |
| `freeze6_lr1e-4` | `1e-4` | First 6 encoder layers |
| `layerwise_lr_decay` | encoder top: `5e-5`, head: `1e-4` | Feature extractor |

The layer-wise experiment applies a `0.9` LR multiplier while moving from
upper to lower Transformer layers. Its feature extractor is frozen because no
`--feature_extractor_learning_rate` is supplied.

## Individual Runs

The original individual scripts remain available:

```bash
FP16=1 bash scripts/run_baseline.sh
FP16=1 bash scripts/run_lr_sweep.sh
FP16=1 bash scripts/run_freeze_sweep.sh
FP16=1 bash scripts/run_layerwise_lr.sh
FP16=1 bash scripts/run_inference.sh baseline_lr1e-4
bash scripts/run_eval.sh baseline_lr1e-4
```

Optional beam decoding is intentionally separate from the queue. It defaults
to the baseline checkpoint and saves a distinct `<experiment>_beam` result:

```bash
python -m pip install pyctcdecode
BEST_EXPERIMENT=baseline_lr1e-4 GPU_ID=0 bash scripts/run_beam_decode.sh
```

Override the beam width when needed:

```bash
BEST_EXPERIMENT=layerwise_lr_decay BEAM_WIDTH=50 GPU_ID=0 \
bash scripts/run_beam_decode.sh
```

Training supports:

```text
--train_shards
--test_clean_shards
--test_other_shards
--model_name_or_path
--output_dir
--learning_rate
--num_train_epochs
--per_device_train_batch_size
--per_device_eval_batch_size
--gradient_accumulation_steps
--max_train_steps
--max_eval_samples
--freeze_feature_encoder
--freeze_n_layers
--layerwise_lr_decay
--layerwise_lr_decay_rate
--head_learning_rate
--feature_extractor_learning_rate
--fp16
--seed
--dry_run
```

Inference additionally supports `--max_test_samples`,
`--decoding_method greedy|beam`, `--beam_width`, and `--dry_run`. Greedy
decoding remains the default.

The documentation-only experiment definitions are stored under `configs/`.

## Outputs

Generated artifacts use this structure:

```text
outputs/<experiment_name>/
logs/<experiment_name>.log
results/<experiment_name>/
results/<experiment_name>/metadata.json
results/wer_summary.csv
results/wer_summary.md
results/figures/wer_barplot.png
```

## Monitoring

Print a project and GPU status snapshot:

```bash
bash scripts/status.sh
```

Monitor the GPU and a running experiment:

```bash
watch -n 2 nvidia-smi
tail -f logs/baseline_lr1e-4.log
cat logs/failed_experiments.txt
```

## Result Summary

Generate a fixed-order Markdown table and a bar plot:

```bash
python scripts/summarize_results.py
```

The CSV and Markdown summary support training settings, decoding method,
learning rate, freezing, layer-wise decay, beam width, WER values, and
checkpoint paths. The best `test-clean` and `test-other` WER values are bolded
in `results/wer_summary.md`. The plot is skipped gracefully if matplotlib is
not installed.

## Submission Snapshot

Create a source-and-report snapshot without data, downloads, model weights,
outputs, checkpoints, or logs:

```bash
bash scripts/make_submission_snapshot.sh
```

The generated `submission_snapshot/` directory contains the README,
environment files, Python source, scripts, reports, and available summary
artifacts.

## Troubleshooting

- Missing shards: run `bash scripts/check_data.sh` and restore the expected
  `data/train`, `data/test-clean`, and `data/test-other` `.tar` files.
- CUDA out of memory: lower `PER_DEVICE_TRAIN_BATCH_SIZE` or
  `PER_DEVICE_EVAL_BATCH_SIZE`, then increase `GRADIENT_ACCUMULATION_STEPS` if
  needed.
- Wrong GPU: set `GPU_ID`, for example
  `GPU_ID=1 bash scripts/run_smoke_train.sh`.
- Interrupted queue: rerun `scripts/run_queue_3090.sh`. Existing checkpoints
  and completed inference files are skipped unless `FORCE=1`.
- Model download errors: make sure the Hugging Face cache is writable and the
  initial `facebook/wav2vec2-base` download can reach the network.
- Beam dependency errors: install `pyctcdecode`. KenLM is not required for the
  provided beam decoding script.
- Queue failures: inspect `logs/failed_experiments.txt` and the corresponding
  `logs/<experiment_name>.log`.

## Lightweight Checks

Run syntax checks without starting training:

```bash
python -m py_compile *.py
bash -n scripts/*.sh
```
