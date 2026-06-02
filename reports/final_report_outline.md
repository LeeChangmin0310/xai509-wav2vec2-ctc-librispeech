# Final Report Outline

## Abstract

`[Summarize the task, methods, and best WER result.]`

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
- Training and evaluation splits
- Shared hyperparameters: fp32 training, Hugging Face CTC loss,
  `ctc_zero_infinity`, and SpecAugment disabled

## 4. Results

### 4.1 Pretrained ASR Control

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asr_pretrained_960h_full` | `0.186112` | `0.245802` |

### 4.2 ASR-Initialized Learning-Rate Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_lr1e-6_fp32` | `[FILL IN]` | `[FILL IN]` |
| `asrinit_lr3e-6_fp32` | `[FILL IN]` | `[FILL IN]` |
| `asrinit_lr1e-5_fp32_fixed` | `[FILL IN]` | `[FILL IN]` |

### 4.3 ASR-Initialized Freezing Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_freeze_feature_lr3e-6_fp32` | `[FILL IN]` | `[FILL IN]` |
| `asrinit_freeze3_lr3e-6_fp32` | `[FILL IN]` | `[FILL IN]` |

### 4.4 Test-Clean Versus Test-Other

`[Discuss the WER gap and any consistent trends.]`

### 4.5 ASR-Initialized Layer-Wise Learning-Rate Decay

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_layerwise_lr_decay_fixed` | `[FILL IN]` | `[FILL IN]` |

`[Discuss whether smaller lower-layer learning rates improved robustness.]`

### 4.6 Original Base-Model Failure Analysis

- Original initialization: `facebook/wav2vec2-base`
- Observed failure: blank prediction collapse
- Diagnostic finding: train-mode SpecAugment produced NaN logits
- Forward/loss probe after `apply_spec_augment=False`: finite loss `187.5894`
- ASR-init 20-sample smoke WER: test-clean `0.1049`, test-other `0.1301`

### 4.7 Optional Beam Decoding

- Selected checkpoint: `[FILL IN]`
- Beam width: `[FILL IN]`
- Greedy WER: `[FILL IN]`
- Beam WER: `[FILL IN]`

## 5. Discussion

- Effect of learning rate: `[FILL IN]`
- Effect of freezing: `[FILL IN]`
- Effect of layer-wise LR decay: `[FILL IN]`
- Effect of optional beam decoding: `[FILL IN]`
- Best overall configuration: `[FILL IN]`
- Limitations: `[FILL IN]`

## 6. Conclusion

`[State the final interpretation and the most important experimental result.]`
