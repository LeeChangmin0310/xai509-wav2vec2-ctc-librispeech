# Final Result Analysis

## Scope

The main final comparison is `results/wer_summary_properval_final.csv`.
Checkpoint selection used a held-out train shard:
`data/train/shard-000004.tar`. The LibriSpeech `test-clean` and `test-other`
splits were reserved for final inference and WER evaluation only.

Earlier ASR-initialized runs in `results/wer_summary_asrinit_final.csv` are
useful preliminary diagnostics, but they are not the main final comparison
because `test-clean` was used as the Trainer evaluation/checkpoint-selection
split.

## Final Proper-Validation Results

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

## Main Findings

The best final model is `asrinit_lr1e-5_fp32_properval_beam`. Relative to the
pretrained ASR control, it reduced WER by `0.033514` absolute on test-clean and
`0.033567` absolute on test-other. These are relative reductions of `18.01%`
and `13.66%`.

Among the tested greedy fine-tuning runs, learning rate `1e-5` performed best:
`0.154709` on test-clean and `0.214737` on test-other. This was better than
`1e-6` and `3e-6`, but the conclusion is limited to the tested schedule, data
split, and seed.

Freezing did not help as much as unfrozen fine-tuning in this setup. At
learning rate `3e-6`, unfrozen training reached `0.169678` and `0.229620`,
while feature-encoder freezing reached `0.181566` and `0.241809`, and freezing
the first three encoder layers reached `0.176126` and `0.236612`.

The tested layer-wise LR decay setup did not improve WER. Its scores,
`0.184742` and `0.244292`, were close to the pretrained ASR control.

Beam decoding added a small additional improvement over the selected greedy
checkpoint: `0.002111` absolute WER on test-clean and `0.002502` on test-other.

## Preliminary Non-Strict ASR-Init Results

The earlier ASR-init run family produced stronger-looking WER for the same
`1e-5` setting, but it used `test-clean` as Trainer evaluation data. Those
numbers are retained for debugging history and should not be mixed with the
strict final comparison.

| Experiment | Decoding | test-clean WER | test-other WER |
| --- | --- | ---: | ---: |
| `asr_pretrained_960h_full` | Greedy | `0.186112` | `0.245802` |
| `asrinit_lr1e-6_fp32` | Greedy | `0.181223` | `0.241274` |
| `asrinit_lr3e-6_fp32` | Greedy | `0.164029` | `0.224137` |
| `asrinit_lr1e-5_fp32_fixed` | Greedy | `0.140958` | `0.199759` |
| `asrinit_freeze_feature_lr3e-6_fp32` | Greedy | `0.180044` | `0.240682` |
| `asrinit_freeze3_lr3e-6_fp32` | Greedy | `0.172208` | `0.232639` |
| `asrinit_layerwise_lr_decay_fixed` | Greedy | `0.184438` | `0.244312` |
| `asrinit_lr1e-5_fp32_fixed_beam` | Beam | `0.139227` | `0.196760` |

## Failure Diagnosis

The original `facebook/wav2vec2-base` fine-tuning experiments collapsed to blank
predictions. An early ASR-initialized run also produced WER `1.0`, zero or
non-finite training diagnostics, and blank output.

The pretrained ASR inference-only control worked, which confirmed that the data
reader, inference code, and WER evaluation path were usable. The CTC probe then
isolated the numerical failure to train-mode SpecAugment:
`apply_spec_augment=True` produced NaN logits before CTC loss. Setting
`apply_spec_augment=False` restored finite logits and finite loss.

Final fine-tuning therefore used fp32 training, Hugging Face CTC loss,
`ctc_zero_infinity`, no loss-forward attention mask for the group-normalized
Wav2Vec2 model, and SpecAugment disabled.

## Conservative Interpretation

The final results support these narrow conclusions for the tested
configurations:

1. ASR-initialized fine-tuning at learning rate `1e-5` was the best tested
   greedy setting under the strict proper-validation split.
2. The evaluated freezing and layer-wise LR decay settings did not outperform
   standard unfrozen fine-tuning.
3. Beam decoding provided a small extra gain after selecting the best greedy
   checkpoint.

These results do not prove that other freezing schedules, layer-wise decay
rates, seeds, or decoding settings cannot help.
