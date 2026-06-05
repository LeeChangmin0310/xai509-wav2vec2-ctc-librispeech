# Final Report Outline

## Abstract

Fine-tuning `facebook/wav2vec2-base-960h` with stable fp32 Hugging Face CTC loss
improved WER in the completed diagnostic runs. However, those runs used
`test-clean` as the Trainer evaluation/checkpoint-selection split. The stricter
final setup trains on four train shards, validates on one held-out train shard,
and reserves `test-clean`/`test-other` for final inference and WER only.

## 1. Introduction

- Motivation for automatic speech recognition with Wav2Vec 2.0
- Role of CTC loss in sequence alignment
- Scope of the LibriSpeech experiments

## 2. Implementation

- WebDataset shard loading without extraction
- Wav2Vec2 CTC fine-tuning pipeline
- Hugging Face CTC loss integration with an optional custom diagnostic path
- Stable fp32 ASR-initialized training with SpecAugment disabled
- Batched inference and WER evaluation
- Optional layer-wise LR decay optimizer groups
- Optional CTC beam decoding without a required language model

## 3. Experimental Setup

- Hardware: single RTX 3090
- Final initialization: `facebook/wav2vec2-base-960h`
- Training split: `data/train/shard-000000.tar` through
  `data/train/shard-000003.tar`
- Trainer validation split: `data/train/shard-000004.tar`
- Final evaluation splits: `data/test-clean` and `data/test-other`
- Shared hyperparameters: fp32 training, Hugging Face CTC loss,
  `ctc_zero_infinity`, and SpecAugment disabled

## 4. Results

### 4.1 Proper-Validation Results (Pending)

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asr_pretrained_960h_full_properval` | TBD | TBD |
| `asrinit_lr1e-6_fp32_properval` | TBD | TBD |
| `asrinit_lr3e-6_fp32_properval` | TBD | TBD |
| `asrinit_lr1e-5_fp32_properval` | TBD | TBD |
| `asrinit_freeze_feature_lr3e-6_fp32_properval` | TBD | TBD |
| `asrinit_freeze3_lr3e-6_fp32_properval` | TBD | TBD |
| `asrinit_layerwise_lr_decay_properval` | TBD | TBD |

Use `results/wer_summary_properval.csv` and
`results/wer_summary_properval.md` once the strict queue finishes.

### 4.2 Diagnostic Pretrained ASR Control

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asr_pretrained_960h_full` | `0.186112` | `0.245802` |

### 4.3 Diagnostic ASR-Initialized Learning-Rate Sweep

These completed results are useful but not strict because `test-clean` was used
for Trainer evaluation.

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_lr1e-6_fp32` | `0.181223` | `0.241274` |
| `asrinit_lr3e-6_fp32` | `0.164029` | `0.224137` |
| `asrinit_lr1e-5_fp32_fixed` | **`0.140958`** | **`0.199759`** |

### 4.4 Diagnostic ASR-Initialized Freezing Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_freeze_feature_lr3e-6_fp32` | `0.180044` | `0.240682` |
| `asrinit_freeze3_lr3e-6_fp32` | `0.172208` | `0.232639` |

### 4.5 Test-Clean Versus Test-Other

Test-other WER remained higher than test-clean WER for every configuration. The
gap was `0.058801` for the best greedy model and `0.057533` after beam decoding.

### 4.6 Diagnostic ASR-Initialized Layer-Wise Learning-Rate Decay

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_layerwise_lr_decay_fixed` | `0.184438` | `0.244312` |

Layer-wise LR decay did not improve over the pretrained control or the standard
ASR-initialized fine-tuning runs.

### 4.7 Original Base-Model Failure Analysis

- Original initialization: `facebook/wav2vec2-base`
- Observed failure: blank prediction collapse
- Diagnostic finding: train-mode SpecAugment produced NaN logits
- Forward/loss probe after `apply_spec_augment=False`: finite loss `187.5894`
- ASR-init 20-sample smoke WER: test-clean `0.1049`, test-other `0.1301`
- Excluded debug row: pre-fix `asrinit_lr1e-5_fp32` with WER `1.0`

### 4.8 Optional Beam Decoding

- Selected checkpoint: `asrinit_lr1e-5_fp32_fixed`
- Beam width: `100`
- Greedy WER: test-clean `0.140958`, test-other `0.199759`
- Beam WER: test-clean `0.139227`, test-other `0.196760`

## 5. Discussion

- Effect of learning rate: among tested values, `1e-5` helped most.
- Effect of freezing: freezing did not help as much as unfrozen `3e-6` training.
- Effect of layer-wise LR decay: the tested decay setup did not improve WER.
- Effect of optional beam decoding: beam search gave a small additional gain.
- Diagnostic best configuration: `asrinit_lr1e-5_fp32_fixed` with beam decoding.
- Proper-validation results should be used for the strict final comparison once
  available.
- Limitations: only a small set of learning rates, freezing choices, and one
  layer-wise decay schedule were evaluated.

## 6. Conclusion

Stable ASR-initialized fine-tuning improved both LibriSpeech test splits in the
completed diagnostic runs. The stricter conclusion should be based on the
proper-validation queue, where model selection uses a held-out train shard and
`test-clean`/`test-other` are used only for final evaluation.
