# XAI 509 Wav2Vec2 CTC Project

This project fine-tunes Wav2Vec 2.0 with CTC loss, runs inference on LibriSpeech
`test-clean` and `test-other`, and records WER for ASR-initialized final
experiments plus diagnostic base-model runs. Hugging Face model loss is the
default; the provided custom CTC implementation remains available explicitly.

Final WebDataset `.tar` shards are read directly. Do not extract them.

## Current Status

- The project skeleton is implemented.
- Data shards are present under `data/`.
- Python and shell syntax checks pass.
- Smoke and unattended single-GPU queue scripts are available.
- The pretrained `facebook/wav2vec2-base-960h` inference control works.
- Train-mode SpecAugment was identified as the source of NaN logits. Fine-tuning
  now disables SpecAugment by default.
- ASR-init smoke training has finite train and evaluation loss. A 20-sample
  bounded check produced `0.1049` test-clean WER and `0.1301` test-other WER.

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

Use first-batch diagnostics or override the model and output directory:

```bash
GPU_ID=0 DEBUG_FIRST_BATCH=1 \
MODEL_NAME_OR_PATH=facebook/wav2vec2-base-960h \
OUTPUT_DIR=outputs/smoke_asrinit \
bash scripts/run_smoke_train.sh
```

Smoke artifacts use:

```text
outputs/smoke/
results/smoke/
logs/smoke_train.log
logs/smoke_inference_eval.log
```

Probe CTC inputs and loss variants before retrying training:

```bash
bash scripts/run_probe_ctc_loss.sh
```

For the dedicated two-step fp32 ASR-initialized diagnostic, the script disables
SpecAugment, enables `ctc_zero_infinity`, and omits the loss-forward attention
mask:

```bash
GPU_ID=0 DEBUG_FIRST_BATCH=1 bash scripts/run_smoke_asrinit_train.sh
CUDA_VISIBLE_DEVICES=0 python wav2vec_inference.py \
  --test_clean_shards data/test-clean \
  --test_other_shards data/test-other \
  --model_name_or_path outputs/smoke_asrinit/final_model \
  --output_dir results/smoke_asrinit \
  --per_device_eval_batch_size 1 \
  --max_test_samples 4
python evaluate_wer.py \
  --experiment_name smoke_asrinit \
  --summary_csv results/smoke_asrinit/wer_summary.csv \
  results/smoke_asrinit/test_clean_result.txt \
  results/smoke_asrinit/test_other_result.txt
python scripts/check_predictions_nonempty.py \
  results/smoke_asrinit/test_clean_result.txt
python scripts/check_predictions_nonempty.py \
  results/smoke_asrinit/test_other_result.txt
```

## ASR-Initialized Final Experiments

The final result path starts from `facebook/wav2vec2-base-960h`, trains in fp32,
disables SpecAugment, uses Hugging Face CTC loss, and enables
`ctc_zero_infinity`. Inference uses fp16 by default when CUDA is available.

Launch the complete resumable queue on one GPU:

```bash
GPU_ID=0 CONTINUE_ON_ERROR=1 bash scripts/run_asrinit_queue_3090.sh
```

Set `INFERENCE_FP16=0` to disable fp16 inference. Set `FORCE=1` to rerun
completed checkpoints and transcripts. Failures are appended to
`logs/failed_asrinit_experiments.txt`.

The final queue order is:

| Experiment | Learning rate | Freezing | Notes |
| --- | ---: | --- | --- |
| `asr_pretrained_960h_full` | - | None | Inference-only pretrained control |
| `asrinit_lr1e-6_fp32` | `1e-6` | None | Full fp32 fine-tuning |
| `asrinit_lr3e-6_fp32` | `3e-6` | None | Full fp32 fine-tuning |
| `asrinit_lr1e-5_fp32_fixed` | `1e-5` | None | New fixed output path |
| `asrinit_freeze_feature_lr3e-6_fp32` | `3e-6` | Feature encoder | fp32 |
| `asrinit_freeze3_lr3e-6_fp32` | `3e-6` | First 3 encoder layers | fp32 |
| `asrinit_layerwise_lr_decay_fixed` | encoder top: `3e-6`, head: `1e-5` | Feature extractor | Decay rate `0.9` |

Results are written to `results/wer_summary_asrinit.csv`. Generate the separate
Markdown summary and plot with:

```bash
python scripts/summarize_results.py \
  --input_csv results/wer_summary_asrinit.csv
```

## Original Base Queue

The original `facebook/wav2vec2-base` experiments are retained as
failure-analysis history. They collapsed to blank predictions and are not the
recommended final-results path.

To reproduce that diagnostic queue:

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

The original queue order is:

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
--loss_impl hf|custom
--debug_first_batch
--ctc_zero_infinity
--disable_spec_augment
--enable_spec_augment
--use_attention_mask_for_loss
--no_attention_mask_for_loss
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
results/wer_summary_asrinit.csv
results/wer_summary_asrinit.md
results/figures/wer_barplot.png
results/figures/wer_barplot_asrinit.png
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
tail -f logs/asrinit_lr3e-6_fp32.log
cat logs/failed_asrinit_experiments.txt
```

## Result Summary

Generate a fixed-order Markdown table and a bar plot:

```bash
python scripts/summarize_results.py
python scripts/summarize_results.py \
  --input_csv results/wer_summary_asrinit.csv
```

The CSV and Markdown summary support training settings, decoding method,
learning rate, freezing, layer-wise decay, beam width, WER values, and
checkpoint paths. The best `test-clean` and `test-other` WER values are bolded
in each generated Markdown table. The plot is skipped gracefully if matplotlib
is not installed.

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
- Interrupted ASR-init queue: rerun `scripts/run_asrinit_queue_3090.sh`.
  Existing checkpoints, transcripts, and summary rows are reused unless
  `FORCE=1`.
- Model download errors: make sure the Hugging Face cache is writable and the
  initial `facebook/wav2vec2-base` download can reach the network.
- Beam dependency errors: install `pyctcdecode`. KenLM is not required for the
  provided beam decoding script.
- Queue failures: inspect `logs/failed_experiments.txt` and the corresponding
  `logs/<experiment_name>.log`.
- Loss-path diagnosis: pretrained `facebook/wav2vec2-base-960h` inference
  controls produced `0.186112` test-clean WER and `0.245802` test-other WER.
  Earlier base fine-tuning collapsed to blank predictions, and an ASR-init run
  showed training loss `0`, `grad_norm=nan`, `eval_loss=nan`, and `eval_wer=1`.
  A bounded ASR-init retry with Hugging Face loss then produced `nan` on its
  first batch. The probe showed that train-mode SpecAugment caused NaN logits.
- SpecAugment NaNs: fine-tuning disables SpecAugment by default for stable
  low-resource runs. Pass `--disable_spec_augment` explicitly in reproducible
  scripts. Use `--enable_spec_augment` only for intentional experiments.
- Verify the diagnosis: run `bash scripts/run_probe_ctc_loss.sh`. It compares
  eval mode, train mode with `apply_spec_augment=True`, and train mode with
  `apply_spec_augment=False`.
- Attention-mask safeguards: group-normalized Wav2Vec2 models omit the
  loss-forward attention mask by default unless `--use_attention_mask_for_loss`
  is explicitly set. The ASR-init smoke script also enables
  `--ctc_zero_infinity`.
- Evaluation control: use pretrained `facebook/wav2vec2-base-960h` inference to
  verify the data reader and WER path independently of fine-tuning.
- Blank predictions: run
  `python scripts/check_predictions_nonempty.py results/<experiment>/test_clean_result.txt`
  and inspect `reports/blank_collapse_debug.md`.

## Lightweight Checks

Run syntax checks without starting training:

```bash
python -m py_compile *.py scripts/*.py
bash -n scripts/*.sh
```
