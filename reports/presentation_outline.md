# Presentation Outline

## 1. Project Goal

- Fine-tune Wav2Vec 2.0 with the Hugging Face CTC loss path.
- Compare learning-rate, freezing, and layer-wise LR decay strategies.
- Evaluate both `test-clean` and `test-other`.

## 2. Experimental Setup

- GPU: single RTX 3090
- Base model: `facebook/wav2vec2-base`
- Training data: provided LibriSpeech WebDataset shards
- Evaluation metric: word error rate (WER)

## 3. Baseline

- Baseline configuration: `baseline_lr1e-4`
- `test-clean` WER: `[FILL IN]`
- `test-other` WER: `[FILL IN]`

## 4. Learning-Rate Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `baseline_lr1e-4` | `[FILL IN]` | `[FILL IN]` |
| `lr1e-5` | `[FILL IN]` | `[FILL IN]` |
| `lr5e-5` | `[FILL IN]` | `[FILL IN]` |

## 5. Freezing Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `freeze_feature_lr1e-4` | `[FILL IN]` | `[FILL IN]` |
| `freeze3_lr1e-4` | `[FILL IN]` | `[FILL IN]` |
| `freeze6_lr1e-4` | `[FILL IN]` | `[FILL IN]` |

## 6. Test-Clean Versus Test-Other

- Observed performance gap: `[FILL IN]`
- Likely reason for the gap: `[FILL IN]`

## 7. Layer-Wise LR Decay

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `layerwise_lr_decay` | `[FILL IN]` | `[FILL IN]` |

- Comparison against baseline: `[FILL IN]`

## 8. Optional Beam Decoding

- Selected checkpoint: `[FILL IN]`
- Beam width: `[FILL IN]`
- Greedy versus beam WER comparison: `[FILL IN]`

## 9. Final Interpretation

- Best experiment: `[FILL IN]`
- Main result: `[FILL IN]`
- Trade-offs and next steps: `[FILL IN]`
