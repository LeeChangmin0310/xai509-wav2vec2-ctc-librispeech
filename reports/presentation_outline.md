# Presentation Outline

## Slide 1: Goal

- Fine-tune Wav2Vec2 CTC for LibriSpeech ASR.
- Use strict model selection: train-shard validation, final test-only WER.
- Main metric: WER on `test-clean` and `test-other`.

## Slide 2: Strict Data Split

- Train: `data/train/shard-000000.tar` through `shard-000003.tar`.
- Validation/checkpoint selection: `data/train/shard-000004.tar`.
- Final evaluation only: `data/test-clean` and `data/test-other`.
- Final WebDataset tar shards were not extracted.

## Slide 3: Stable Training Setup

- Initialization: `facebook/wav2vec2-base-960h`.
- Loss: Hugging Face CTC loss.
- Training: fp32, `ctc_zero_infinity`, no loss-forward attention mask.
- SpecAugment disabled because train-mode SpecAugment produced NaN logits.

## Slide 4: Final Result Table

| Experiment | Decoding | test-clean | test-other |
| --- | --- | ---: | ---: |
| ASR pretrained control | Greedy | `0.186112` | `0.245802` |
| LR `1e-6` | Greedy | `0.180558` | `0.240567` |
| LR `3e-6` | Greedy | `0.169678` | `0.229620` |
| LR `1e-5` | Greedy | `0.154709` | `0.214737` |
| Freeze feature encoder | Greedy | `0.181566` | `0.241809` |
| Freeze first 3 layers | Greedy | `0.176126` | `0.236612` |
| Layer-wise LR decay | Greedy | `0.184742` | `0.244292` |
| LR `1e-5` | Beam | **`0.152598`** | **`0.212235`** |

## Slide 5: Best Model

- Best final model: `asrinit_lr1e-5_fp32_properval_beam`.
- Absolute WER improvement over ASR control:
  `0.033514` test-clean, `0.033567` test-other.
- Relative WER reduction:
  `18.01%` test-clean, `13.66%` test-other.
- Beam decoding added a small gain over greedy decoding.

## Slide 6: Experiment Trends

- LR `1e-5` helped most among tested greedy runs.
- Freezing feature encoder or early encoder layers did not beat unfrozen tuning.
- The tested layer-wise LR decay setup stayed close to the pretrained control.
- Test-other remained harder than test-clean for every configuration.

## Slide 7: Debugging Story

- Base Wav2Vec2 fine-tuning collapsed to blank predictions.
- Pretrained ASR inference control worked, so data/inference/WER were sound.
- ASR-init training exposed NaN loss/logits.
- Probe showed train-mode SpecAugment caused NaN logits.
- Disabling SpecAugment made training numerically stable.

## Slide 8: Takeaways

- Strict proper-validation results are the final comparison.
- ASR-initialized `1e-5` fine-tuning was the best tested greedy setup.
- Beam decoding gave the best final WER.
- Claims are conservative: other seeds, schedules, or decoding settings could
  change the outcome.
