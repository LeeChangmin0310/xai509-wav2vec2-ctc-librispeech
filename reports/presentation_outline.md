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
| `asrinit_lr1e-6_fp32` | `0.181223` | `0.241274` |
| `asrinit_lr3e-6_fp32` | `0.164029` | `0.224137` |
| `asrinit_lr1e-5_fp32_fixed` | **`0.140958`** | **`0.199759`** |

## 5. ASR-Initialized Freezing Sweep

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_freeze_feature_lr3e-6_fp32` | `0.180044` | `0.240682` |
| `asrinit_freeze3_lr3e-6_fp32` | `0.172208` | `0.232639` |

## 6. Test-Clean Versus Test-Other

- Every configuration had higher WER on test-other.
- Best greedy gap: `0.058801`; best beam gap: `0.057533`.
- Interpretation: test-other remains the more challenging split.

## 7. Layer-Wise LR Decay

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| `asrinit_layerwise_lr_decay_fixed` | `0.184438` | `0.244312` |

- The tested layer-wise schedule did not improve WER.

## 8. Original Base-Model Failure Analysis

- `facebook/wav2vec2-base` fine-tuning collapsed to blank predictions.
- Train-mode SpecAugment produced NaN logits.
- Setting `apply_spec_augment=False` restored finite forward loss.
- ASR-init 20-sample smoke WER: clean `0.1049`, other `0.1301`.
- Exclude the pre-fix `asrinit_lr1e-5_fp32` WER `1.0` debug row.

## 9. Optional Beam Decoding

- Selected checkpoint: `asrinit_lr1e-5_fp32_fixed`
- Beam width: `100`
- Greedy: clean `0.140958`, other `0.199759`
- Beam: clean `0.139227`, other `0.196760`
- Beam search gave a small additional improvement.

## 10. Final Interpretation

- Best experiment: `asrinit_lr1e-5_fp32_fixed` with beam decoding.
- LR `1e-5` helped most among the tested learning rates.
- Freezing did not help as much; layer-wise decay did not improve WER.
- Relative to the pretrained control, final beam WER fell by `25.19%` on
  test-clean and `19.95%` on test-other.
