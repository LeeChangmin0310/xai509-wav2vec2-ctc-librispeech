# Final Report Outline

## Abstract

This project fine-tuned `facebook/wav2vec2-base-960h` with CTC loss on the
provided LibriSpeech WebDataset shards. The strict final setup trained on four
train shards, selected checkpoints on one held-out train shard, and reserved
`test-clean`/`test-other` for final evaluation only. The best final model,
`asrinit_lr1e-5_fp32_properval_beam`, reached `0.152598` test-clean WER and
`0.212235` test-other WER.

## 1. Introduction

- Motivation for automatic speech recognition with Wav2Vec 2.0
- CTC loss for unsegmented speech-to-text alignment
- Course constraint: use final WebDataset tar shards without extraction

## 2. Implementation

- WebDataset shard loading from directories, tar files, globs, and comma lists
- Hugging Face `AutoProcessor` and `AutoModelForCTC` / `Wav2Vec2ForCTC`
- Hugging Face CTC loss as the stable default
- Optional custom CTC loss retained for diagnostics
- Stable ASR-initialized fp32 training with SpecAugment disabled
- Greedy inference plus optional `pyctcdecode` beam decoding
- WER evaluation and CSV/Markdown/plot summaries

## 3. Experimental Setup

- Hardware: single RTX 3090
- Initialization: `facebook/wav2vec2-base-960h`
- Train shards: `data/train/shard-000000.tar` through
  `data/train/shard-000003.tar`
- Trainer validation shard: `data/train/shard-000004.tar`
- Final evaluation: `data/test-clean` and `data/test-other`
- Stability flags: `--loss_impl hf`, `--disable_spec_augment`,
  `--ctc_zero_infinity`, `--no_attention_mask_for_loss`
- Training precision: fp32

## 4. Final Results

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

## 5. Interpretation

- Best final model: `asrinit_lr1e-5_fp32_properval_beam`
- Improvement over pretrained ASR control: `18.01%` relative WER reduction on
  test-clean and `13.66%` on test-other
- Best greedy setting: unfrozen ASR-init fine-tuning with learning rate `1e-5`
- Freezing did not help as much as unfrozen fine-tuning in these runs
- The tested layer-wise LR decay configuration did not improve WER
- Beam decoding gave a small additional improvement over greedy decoding

## 6. Failure and Debugging Analysis

- Original `facebook/wav2vec2-base` runs collapsed to blank predictions
- Pretrained ASR inference-only control worked, so data/inference/WER were valid
- ASR-init training initially produced non-finite loss/logits
- CTC probe identified train-mode SpecAugment as the NaN source
- Setting `apply_spec_augment=False` restored finite forward/loss behavior

## 7. Preliminary Non-Strict Runs

Earlier ASR-init results used `test-clean` as Trainer evaluation data. They are
useful for debugging history, but the final report should use
`results/wer_summary_properval_final.csv` as the main comparison table.

## 8. Conclusion

The strict proper-validation results show that ASR-initialized fine-tuning can
improve over the pretrained ASR control under this small-shard setup. The best
tested configuration was unfrozen `1e-5` fine-tuning followed by beam decoding.
The claims should remain limited to the tested learning rates, freezing choices,
layer-wise decay setup, seed, and data split.
