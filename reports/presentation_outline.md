# Presentation Outline

## 1. Project Goal

- Fine-tune Wav2Vec 2.0 with the Hugging Face CTC loss path.
- Compare ASR-initialized learning-rate, freezing, and layer-wise LR decay strategies.
- Evaluate both `test-clean` and `test-other`.

## 2. Experimental Setup

- GPU: single RTX 3090
- Final initialization: `facebook/wav2vec2-base-960h`
- Training data: provided LibriSpeech WebDataset shards
- Evaluation metric: word error rate (WER)
- Stability settings: fp32 training, SpecAugment disabled, `ctc_zero_infinity`

## 3. Pretrained ASR Control

- Control configuration: `asr_pretrained_960h_full`
- `test-clean` WER: `0.186112`
- `test-other` WER: `0.245802`

## 4. ASR-Initialized Learning-Rate Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_lr1e-6_fp32` | `[FILL IN]` | `[FILL IN]` |
| `asrinit_lr3e-6_fp32` | `[FILL IN]` | `[FILL IN]` |
| `asrinit_lr1e-5_fp32_fixed` | `[FILL IN]` | `[FILL IN]` |

## 5. ASR-Initialized Freezing Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_freeze_feature_lr3e-6_fp32` | `[FILL IN]` | `[FILL IN]` |
| `asrinit_freeze3_lr3e-6_fp32` | `[FILL IN]` | `[FILL IN]` |

## 6. Test-Clean Versus Test-Other

- Observed performance gap: `[FILL IN]`
- Likely reason for the gap: `[FILL IN]`

## 7. Layer-Wise LR Decay

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_layerwise_lr_decay_fixed` | `[FILL IN]` | `[FILL IN]` |

- Comparison against baseline: `[FILL IN]`

## 8. Original Base-Model Failure Analysis

- `facebook/wav2vec2-base` fine-tuning collapsed to blank predictions.
- Train-mode SpecAugment produced NaN logits.
- Setting `apply_spec_augment=False` restored finite forward loss.
- ASR-init 20-sample smoke WER: clean `0.1049`, other `0.1301`.

## 9. Optional Beam Decoding

- Selected checkpoint: `[FILL IN]`
- Beam width: `[FILL IN]`
- Greedy versus beam WER comparison: `[FILL IN]`

## 10. Final Interpretation

- Best experiment: `[FILL IN]`
- Main result: `[FILL IN]`
- Trade-offs and next steps: `[FILL IN]`
