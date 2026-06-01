# XAI 509 Wav2Vec2 CTC Project

This directory fine-tunes Wav2Vec 2.0 with the provided custom CTC loss,
runs inference on LibriSpeech `test-clean` and `test-other`, and records WER
for the required experiments.

Final WebDataset `.tar` shards are read directly. Do not extract them.

## Environment Setup

Create an environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio transformers accelerate webdataset jiwer gdown
```

Install the PyTorch build appropriate for your CUDA environment when using a
GPU. The default pretrained model is `facebook/wav2vec2-base`.

## Data Setup

The expected layout is:

```text
data/
  train/*.tar
  test-clean/*.tar
  test-other/*.tar
```

For automatic download, set the Google Drive folder URLs supplied by the
course and run:

```bash
export TRAIN_GDRIVE_URL="COURSE_TRAIN_FOLDER_URL"
export TEST_CLEAN_GDRIVE_URL="COURSE_TEST_CLEAN_FOLDER_URL"
export TEST_OTHER_GDRIVE_URL="COURSE_TEST_OTHER_FOLDER_URL"
bash scripts/download_data.sh
```

If `gdown` is unavailable or a Drive download fails, the script prints manual
setup instructions and exits without extracting any files. For manual setup,
place the final `.tar` shard files directly in the three directories above.

Verify the shard counts and directory sizes:

```bash
bash scripts/check_data.sh
```

## Baseline Training

Run the baseline experiment:

```bash
FP16=1 bash scripts/run_baseline.sh
```

The scripts use these optional environment overrides:

```bash
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
SEED=42
FP16=0
FORCE=0
```

Training skips an existing checkpoint or `final_model` directory. Set
`FORCE=1` to rerun an experiment.

The equivalent direct Python command is:

```bash
python wav2vec_finetuning.py \
  --train_shards data/train \
  --test_clean_shards data/test-clean \
  --test_other_shards data/test-other \
  --model_name_or_path facebook/wav2vec2-base \
  --output_dir outputs/baseline_lr1e-4 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --seed 42 \
  --fp16
```

Optional freezing arguments are `--freeze_feature_encoder` and
`--freeze_n_layers N`.

## Inference And WER

Run inference and WER evaluation for the baseline:

```bash
FP16=1 bash scripts/run_inference.sh baseline_lr1e-4
bash scripts/run_eval.sh baseline_lr1e-4
```

Inference writes:

```text
results/baseline_lr1e-4/test_clean_result.txt
results/baseline_lr1e-4/test_other_result.txt
```

WER evaluation updates `results/wer_summary.csv`. Inference skips an
experiment when both result files already exist unless `FORCE=1`.

The direct WER command is:

```bash
python evaluate_wer.py \
  --experiment_name baseline_lr1e-4 \
  --summary_csv results/wer_summary.csv \
  results/baseline_lr1e-4/test_clean_result.txt \
  results/baseline_lr1e-4/test_other_result.txt
```

## Experiment Sweeps

Run the learning-rate and freezing sweeps:

```bash
FP16=1 bash scripts/run_lr_sweep.sh
FP16=1 bash scripts/run_freeze_sweep.sh
```

Run all six training experiments:

```bash
FP16=1 bash scripts/run_all_experiments.sh
```

After training is complete, run inference and WER evaluation for all six:

```bash
FP16=1 bash scripts/run_all_inference_eval.sh
```

The required experiments are:

| Experiment | Learning rate | Freezing |
| --- | ---: | --- |
| `baseline_lr1e-4` | `1e-4` | None |
| `lr1e-5` | `1e-5` | None |
| `lr5e-5` | `5e-5` | None |
| `freeze_feature_lr1e-4` | `1e-4` | Feature encoder |
| `freeze3_lr1e-4` | `1e-4` | First 3 encoder layers |
| `freeze6_lr1e-4` | `1e-4` | First 6 encoder layers |

## Outputs

Generated artifacts use this structure:

```text
outputs/<experiment_name>/
logs/<experiment_name>.log
results/<experiment_name>/
results/wer_summary.csv
```

The final summary table has one row per experiment:

| experiment | test_clean_wer | test_other_wer |
| --- | ---: | ---: |
| `baseline_lr1e-4` | measured value | measured value |
| `lr1e-5` | measured value | measured value |
| `lr5e-5` | measured value | measured value |
| `freeze_feature_lr1e-4` | measured value | measured value |
| `freeze3_lr1e-4` | measured value | measured value |
| `freeze6_lr1e-4` | measured value | measured value |

## Lightweight Checks

Run syntax checks without starting training:

```bash
python -m py_compile *.py
bash -n scripts/*.sh
```
