# XAI 509 Wav2Vec2 CTC Project

## TL;DR

The final strict experiment uses `facebook/wav2vec2-base-960h`, trains on four
LibriSpeech train WebDataset shards, selects checkpoints on one held-out train
shard, and reserves `test-clean`/`test-other` for final evaluation only.

Best final model:

```text
asrinit_lr1e-5_fp32_properval_beam
test-clean WER = 0.152598
test-other WER = 0.212235
```

Compared with the pretrained ASR control
`asr_pretrained_960h_full_properval`, the final beam result improves WER by
`0.033514` absolute on test-clean and `0.033567` absolute on test-other. These
are relative reductions of `18.01%` and `13.66%`.

## Repository Structure

```text
.
  wav2vec_finetuning.py          # Wav2Vec2 CTC fine-tuning
  wav2vec_inference.py           # Greedy or optional beam inference
  evaluate_wer.py                # REF/HYP WER evaluation and CSV summaries
  sample_util.py                 # WebDataset shard loading helpers
  ctc_loss*.py                   # Course custom CTC implementation and tests
  scripts/                       # Data checks, queues, smoke tests, summaries
  configs/                       # Documentation configs for each experiment
  reports/                       # Report and presentation outlines
  results/                       # WER summaries and figures, ignored by git
  outputs/                       # Checkpoints, ignored by git
  logs/                          # Queue logs, ignored by git
  data/                          # WebDataset tar shards, ignored by git
```

Final WebDataset `.tar` shards are read directly. Do not extract them.

## Environment

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

`pyctcdecode` is optional and only needed for beam decoding. KenLM is not
required for the included beam path:

```bash
python -m pip install pyctcdecode
```

## Data Layout

Expected shard layout:

```text
data/
  train/*.tar
  test-clean/*.tar
  test-other/*.tar
```

Verify data without extracting final shards:

```bash
bash scripts/check_data.sh
```

If course Google Drive links are available:

```bash
export TRAIN_GDRIVE_URL="COURSE_TRAIN_FOLDER_URL"
export TEST_CLEAN_GDRIVE_URL="COURSE_TEST_CLEAN_FOLDER_URL"
export TEST_OTHER_GDRIVE_URL="COURSE_TEST_OTHER_FOLDER_URL"
bash scripts/download_data.sh
```

If `gdown` is unavailable or download fails, the script prints manual setup
instructions.

## Strict Proper Validation Split

The final comparison uses this split:

```text
train_shards = data/train/shard-000000.tar,data/train/shard-000001.tar,data/train/shard-000002.tar,data/train/shard-000003.tar
eval_shards  = data/train/shard-000004.tar
final tests  = data/test-clean and data/test-other
```

`eval_shards` is the Trainer validation/checkpoint-selection split.
`test-clean` and `test-other` are used only after training for final inference
and WER evaluation.

## Final Results

Main final table:

```text
results/wer_summary_properval_final.csv
results/wer_summary_properval_final.md
results/figures/wer_barplot_properval_final.png
```

| Experiment | Decoding | test-clean WER | test-other WER |
| --- | --- | ---: | ---: |
| `asr_pretrained_960h_full_properval` | Greedy | `0.186112` | `0.245802` |
| `asrinit_lr1e-6_fp32_properval` | Greedy | `0.180558` | `0.240567` |
| `asrinit_lr3e-6_fp32_properval` | Greedy | `0.169678` | `0.229620` |
| `asrinit_lr1e-5_fp32_properval` | Greedy | `0.154709` | `0.214737` |
| `asrinit_freeze_feature_lr3e-6_fp32_properval` | Greedy | `0.181566` | `0.241809` |
| `asrinit_freeze3_lr3e-6_fp32_properval` | Greedy | `0.176126` | `0.236612` |
| `asrinit_layerwise_lr_decay_properval` | Greedy | `0.184742` | `0.244292` |
| `asrinit_lr1e-5_fp32_properval_beam` | Beam | **`0.152598`** | **`0.212235`** |

Conservative interpretation:

- LR `1e-5` was the best tested greedy fine-tuning setting.
- Freezing did not help as much as unfrozen fine-tuning.
- The tested layer-wise LR decay setup did not improve WER.
- Beam decoding gave a small extra gain over the best greedy checkpoint:
  `0.002111` absolute WER on test-clean and `0.002502` on test-other.

## Reproduction Commands

Run the final proper-validation queue on one GPU:

```bash
GPU_ID=0 CONTINUE_ON_ERROR=1 bash scripts/run_asrinit_properval_queue_3090.sh
```

Rerun completed stages if needed:

```bash
GPU_ID=0 FORCE=1 CONTINUE_ON_ERROR=1 bash scripts/run_asrinit_properval_queue_3090.sh
```

Summarize final proper-validation results:

```bash
python scripts/summarize_results.py \
  --input_csv results/wer_summary_properval_final.csv
```

Run beam decoding for the selected strict checkpoint and update the final CSV:

```bash
CUDA_VISIBLE_DEVICES=0 python wav2vec_inference.py \
  --test_clean_shards data/test-clean \
  --test_other_shards data/test-other \
  --model_name_or_path outputs/asrinit_lr1e-5_fp32_properval/final_model \
  --output_dir results/asrinit_lr1e-5_fp32_properval_beam \
  --decoding_method beam \
  --beam_width 100 \
  --per_device_eval_batch_size 8 \
  --fp16

python evaluate_wer.py \
  --experiment_name asrinit_lr1e-5_fp32_properval_beam \
  --summary_csv results/wer_summary_properval_final.csv \
  --train_setting asr_initialized_properval \
  --decoding_method beam \
  --learning_rate 1e-5 \
  --freeze_setting none \
  --layerwise_lr_decay false \
  --beam_width 100 \
  --checkpoint_path outputs/asrinit_lr1e-5_fp32_properval/final_model \
  results/asrinit_lr1e-5_fp32_properval_beam/test_clean_result.txt \
  results/asrinit_lr1e-5_fp32_properval_beam/test_other_result.txt
```

Monitor a running queue:

```bash
watch -n 2 nvidia-smi
tail -f logs/asrinit_lr1e-5_fp32_properval.log
cat logs/failed_properval_experiments.txt
bash scripts/status.sh
```

## Smoke Tests

Run a short training smoke test:

```bash
GPU_ID=0 bash scripts/run_smoke_train.sh
GPU_ID=0 bash scripts/run_smoke_inference_eval.sh
```

Run the ASR-initialized two-step fp32 diagnostic:

```bash
GPU_ID=0 DEBUG_FIRST_BATCH=1 bash scripts/run_smoke_asrinit_train.sh
```

Probe CTC inputs and SpecAugment loss behavior:

```bash
bash scripts/run_probe_ctc_loss.sh
```

Dry runs validate paths without loading a model:

```bash
python wav2vec_finetuning.py --dry_run
python wav2vec_inference.py \
  --model_name_or_path facebook/wav2vec2-base \
  --output_dir results/dry_run \
  --dry_run
```

## Diagnostic Story

The original `facebook/wav2vec2-base` fine-tuning runs collapsed to blank
predictions. A later ASR-initialized run also failed before the stability fix,
with loss-path diagnostics showing zero or non-finite training values and WER
near `1.0`.

The pretrained ASR inference-only control worked throughout:

```text
facebook/wav2vec2-base-960h
test-clean WER = 0.186112
test-other WER = 0.245802
```

This isolated the data reader, inference code, and WER evaluation as working.
The CTC probe then showed that train-mode SpecAugment with
`apply_spec_augment=True` produced NaN logits. Disabling SpecAugment
(`apply_spec_augment=False`) restored finite logits and finite CTC loss.

Final training therefore used:

```text
--loss_impl hf
--disable_spec_augment
--ctc_zero_infinity
--no_attention_mask_for_loss
fp32 training
```

The earlier ASR-init results in `results/wer_summary_asrinit_final.csv` are kept
as useful preliminary diagnostics, but they are not the main final comparison
because `test-clean` was used as Trainer evaluation data. Use
`results/wer_summary_properval_final.csv` for final reporting.

## Outputs

Generated artifacts follow this pattern:

```text
outputs/<experiment_name>/
logs/<experiment_name>.log
results/<experiment_name>/
results/<experiment_name>/metadata.json
results/wer_summary_properval_final.csv
results/wer_summary_properval_final.md
results/figures/wer_barplot_properval_final.png
```

Large artifacts, data, checkpoints, logs, model weights, and full transcripts
are ignored by git and should not be committed.

## Useful Commands

Evaluate an existing transcript pair:

```bash
python evaluate_wer.py \
  --experiment_name asrinit_lr1e-5_fp32_properval \
  --summary_csv results/wer_summary_properval.csv \
  results/asrinit_lr1e-5_fp32_properval/test_clean_result.txt \
  results/asrinit_lr1e-5_fp32_properval/test_other_result.txt
```

Check whether predictions are blank:

```bash
python scripts/check_predictions_nonempty.py \
  results/asrinit_lr1e-5_fp32_properval/test_clean_result.txt
```

Create a source/report submission snapshot:

```bash
bash scripts/make_submission_snapshot.sh
```

## Troubleshooting

- Missing shards: run `bash scripts/check_data.sh`.
- CUDA out of memory: lower `PER_DEVICE_TRAIN_BATCH_SIZE` or
  `PER_DEVICE_EVAL_BATCH_SIZE`, then use gradient accumulation if needed.
- Wrong GPU: set `GPU_ID`, for example
  `GPU_ID=1 bash scripts/run_smoke_train.sh`.
- Interrupted queue: rerun the same queue. Existing checkpoints, transcripts,
  and summary rows are reused unless `FORCE=1`.
- Beam dependency errors: install `pyctcdecode`.
- Blank predictions: run `scripts/check_predictions_nonempty.py` and inspect
  `reports/blank_collapse_debug.md`.
- NaN loss or logits: run `bash scripts/run_probe_ctc_loss.sh` and confirm
  SpecAugment is disabled for fine-tuning.

## Lightweight Checks

Run syntax checks without starting training:

```bash
python -m py_compile *.py scripts/*.py
bash -n scripts/*.sh
```
