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
- Batched inference and WER evaluation
- Optional layer-wise LR decay optimizer groups
- Optional CTC beam decoding without a required language model

## 3. Experimental Setup

- Hardware: single RTX 3090
- Base model: `facebook/wav2vec2-base`
- Training and evaluation splits
- Shared hyperparameters

## 4. Results

### 4.1 Baseline WER

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `baseline_lr1e-4` | `[FILL IN]` | `[FILL IN]` |

### 4.2 Learning-Rate Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `lr1e-5` | `[FILL IN]` | `[FILL IN]` |
| `lr5e-5` | `[FILL IN]` | `[FILL IN]` |

### 4.3 Freezing Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `freeze_feature_lr1e-4` | `[FILL IN]` | `[FILL IN]` |
| `freeze3_lr1e-4` | `[FILL IN]` | `[FILL IN]` |
| `freeze6_lr1e-4` | `[FILL IN]` | `[FILL IN]` |

### 4.4 Test-Clean Versus Test-Other

`[Discuss the WER gap and any consistent trends.]`

### 4.5 Layer-Wise Learning-Rate Decay

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `layerwise_lr_decay` | `[FILL IN]` | `[FILL IN]` |

`[Discuss whether smaller lower-layer learning rates improved robustness.]`

### 4.6 Optional Beam Decoding

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
