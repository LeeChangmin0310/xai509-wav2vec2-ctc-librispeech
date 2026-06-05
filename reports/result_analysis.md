# Final Result Analysis

## Scope

The completed ASR-initialized results in `results/wer_summary_asrinit_final.csv`
are useful for diagnosis and preliminary comparison, but they are not the
strictest final comparison because `test-clean` was used as the Trainer
evaluation/checkpoint-selection split. The old `asrinit_lr1e-5_fp32` row with
WER `1.0` is excluded because it was produced before the SpecAugment stability
fix.

The stricter follow-up uses `results/wer_summary_properval.csv` after running
`scripts/run_asrinit_properval_queue_3090.sh`: four train shards are used for
training, `data/train/shard-000004.tar` is used for Trainer validation, and
`test-clean`/`test-other` are reserved for final inference and WER only.

## Proper-Validation Results (Pending)

| Experiment | Trainer validation split | test-clean WER | test-other WER | Notes |
| --- | --- | ---: | ---: | --- |
| `asr_pretrained_960h_full_properval` | None | TBD | TBD | Inference-only ASR control |
| `asrinit_lr1e-6_fp32_properval` | Held-out train shard | TBD | TBD | LR sweep |
| `asrinit_lr3e-6_fp32_properval` | Held-out train shard | TBD | TBD | LR sweep |
| `asrinit_lr1e-5_fp32_properval` | Held-out train shard | TBD | TBD | LR sweep |
| `asrinit_freeze_feature_lr3e-6_fp32_properval` | Held-out train shard | TBD | TBD | Freeze feature encoder |
| `asrinit_freeze3_lr3e-6_fp32_properval` | Held-out train shard | TBD | TBD | Freeze first 3 encoder layers |
| `asrinit_layerwise_lr_decay_properval` | Held-out train shard | TBD | TBD | Layer-wise LR decay |

## Completed Diagnostic ASR-Init Results

| Experiment | Decoding | test-clean WER | test-other WER |
| --- | --- | ---: | ---: |
| `asr_pretrained_960h_full` | Greedy | `0.186112` | `0.245802` |
| `asrinit_lr1e-6_fp32` | Greedy | `0.181223` | `0.241274` |
| `asrinit_lr3e-6_fp32` | Greedy | `0.164029` | `0.224137` |
| `asrinit_lr1e-5_fp32_fixed` | Greedy | `0.140958` | `0.199759` |
| `asrinit_freeze_feature_lr3e-6_fp32` | Greedy | `0.180044` | `0.240682` |
| `asrinit_freeze3_lr3e-6_fp32` | Greedy | `0.172208` | `0.232639` |
| `asrinit_layerwise_lr_decay_fixed` | Greedy | `0.184438` | `0.244312` |
| `asrinit_lr1e-5_fp32_fixed_beam` | Beam | **`0.139227`** | **`0.196760`** |

## Main Findings

For the completed diagnostic run family, `1e-5` helped most among the tested
learning rates. With greedy decoding,
`asrinit_lr1e-5_fp32_fixed` reduced WER relative to the pretrained ASR control
by `0.045154` on test-clean and `0.046043` on test-other. These correspond to
relative reductions of `24.26%` and `18.73%`.

Freezing did not help as much as unfrozen fine-tuning. At learning rate `3e-6`,
the unfrozen run reached `0.164029` and `0.224137`, while feature-encoder
freezing reached `0.180044` and `0.240682`, and freezing the first three encoder
layers reached `0.172208` and `0.232639`.

The tested layer-wise LR decay setup did not improve WER. Its scores,
`0.184438` and `0.244312`, remained close to the pretrained control.

Beam decoding gave a small additional improvement over greedy decoding for the
selected `1e-5` checkpoint: `0.001731` absolute WER on test-clean and `0.002999`
on test-other. The final beam result reduced WER relative to the pretrained
control by `25.19%` and `19.95%`.

## Failure Diagnosis

The original `facebook/wav2vec2-base` experiments collapsed to blank
predictions. An early ASR-initialized run also produced WER `1.0`, zero or
non-finite training diagnostics, and blank output. The probe isolated the
numerical failure to train-mode SpecAugment: `apply_spec_augment=True` produced
NaN logits before CTC loss was calculated. Setting
`model.config.apply_spec_augment=False` restored finite logits and finite loss.

Final fine-tuning therefore used fp32 training, Hugging Face CTC loss,
`ctc_zero_infinity`, no loss-forward attention mask for the group-normalized
Wav2Vec2 model, and SpecAugment disabled.

## Conservative Interpretation

The completed diagnostic results support three narrow conclusions for the tested
configurations:

1. ASR-initialized fine-tuning at learning rate `1e-5` improved both test splits
   most among the evaluated learning rates.
2. The evaluated freezing and layer-wise LR decay settings did not outperform
   standard unfrozen fine-tuning.
3. Beam decoding added a small improvement after selecting the best greedy
   checkpoint.

These results do not establish that freezing or layer-wise decay cannot help
under other schedules, seeds, data regimes, or the stricter proper-validation
selection setup.
