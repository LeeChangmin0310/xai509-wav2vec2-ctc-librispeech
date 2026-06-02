# Blank Prediction Collapse Debug Notes

## Observed Evidence

- The inference-only control using `facebook/wav2vec2-base-960h` works:
  - test-clean WER: `0.186112`
  - test-other WER: `0.245802`
- Fine-tuning from `facebook/wav2vec2-base` collapsed to blank predictions.
- Fine-tuning from `facebook/wav2vec2-base-960h` also collapsed.
- The ASR-initialized training log reported training loss `0`,
  `grad_norm=nan`, `eval_loss=nan`, and `eval_wer=1`.
- A bounded ASR-init retry using Hugging Face model loss failed on its first
  batch with `RuntimeError: Non-finite hf CTC loss detected: nan`.
- The failing batch had input shape `(1, 159600)`, attention-mask shape
  `(1, 159600)`, label shape `(1, 162)`, and `162` valid label tokens. Its
  decoded first label text was valid.
- A manual train-mode probe showed that `apply_spec_augment=True` produced NaN
  logits. Setting `model.config.apply_spec_augment=False` produced finite
  logits and finite loss `187.5894`.

These results isolate the failure to the fine-tuning path rather than the
WebDataset inference reader or greedy decoder.

## Final Diagnosis

The inference-only ASR control worked because inference runs in evaluation mode.
Fine-tuning entered train mode, where SpecAugment and `masked_spec_embed`
behavior produced NaN logits before CTC loss was calculated.

Fine-tuning now disables SpecAugment by default:

```text
--disable_spec_augment
```

Use `--enable_spec_augment` only for intentional follow-up experiments.

## Loss-Path Safeguards

`wav2vec_finetuning.py` now defaults to Hugging Face model loss:

```text
--loss_impl hf
```

This passes `input_values` and `labels` to the `AutoModelForCTC` forward method
and uses `outputs.loss`. For group-normalized Wav2Vec2 models, the default
training policy omits the loss-forward attention mask unless it is explicitly
enabled. The provided custom CTC implementation remains available only when
explicitly selected:

```text
--loss_impl custom
```

Both paths raise a clear `RuntimeError` when the computed loss is non-finite.
The ASR-init smoke script also enables `--ctc_zero_infinity`.

## Next Probe

The probe checks attention-mask handling, CTC input lengths, label IDs, logits,
and separate Hugging Face forward variants:

```bash
bash scripts/run_probe_ctc_loss.sh
```

It reports eval mode, train mode with `apply_spec_augment=True`, and train mode
with `apply_spec_augment=False`.

## Next Bounded Check

After inspecting the probe output, rerun a two-step fp32 smoke test from the
pretrained ASR checkpoint:

```bash
GPU_ID=0 DEBUG_FIRST_BATCH=1 bash scripts/run_smoke_asrinit_train.sh
```

Then run bounded inference and inspect whether hypotheses remain non-empty:

```bash
CUDA_VISIBLE_DEVICES=0 python wav2vec_inference.py \
  --test_clean_shards data/test-clean \
  --test_other_shards data/test-other \
  --model_name_or_path outputs/smoke_asrinit/final_model \
  --output_dir results/smoke_asrinit \
  --per_device_eval_batch_size 1 \
  --max_test_samples 4
python scripts/check_predictions_nonempty.py \
  results/smoke_asrinit/test_clean_result.txt
python scripts/check_predictions_nonempty.py \
  results/smoke_asrinit/test_other_result.txt
```

Record the smoke loss values, gradient norm, WER, and empty-hypothesis counts
before resuming any full experiment queue.
