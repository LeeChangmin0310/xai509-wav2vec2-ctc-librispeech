# Final Strict Wav2Vec2-base ASR Report

## Checkpoint provenance

- Main initialization: `facebook/wav2vec2-base`.
- `facebook/wav2vec2-base-960h` was not used in main results.
- No supervised LibriSpeech ASR-fine-tuned checkpoint was used.

## Strict data usage

- Final acoustic training: train shards `000000`–`000003`.
- Checkpoint selection and decoder tuning: validation shard `000004` only.
- Test-clean and test-other were used only after decoder selection.

## SpecAugment

- Default SpecAugment caused blank collapse in tiny-overfit diagnostics.
- Weak SpecAugment succeeded in tiny-overfit and was retained.
- Final setting: SpecAugment ON, `mask_time_prob=0.01`, `mask_time_length=5`, `min_masks=1`.

## Five-fold acoustic CV

| ID | Candidate | Fold | Best WER | Blank | Nonblank | Empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F | `lr2e-4_freeze_feature` | 0 | 0.239875 | 0.421801 | 0.578199 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 1 | 0.263260 | 0.368046 | 0.631954 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 2 | 0.219080 | 0.358743 | 0.641257 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 3 | 0.269305 | 0.403066 | 0.596934 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 4 | 0.241840 | 0.419275 | 0.580725 | 0.000000 |
| H | `two_stage_head_warmup` | 0 | 0.216511 | 0.403518 | 0.596482 | 0.000000 |
| H | `two_stage_head_warmup` | 1 | 0.227251 | 0.375605 | 0.624395 | 0.000000 |
| H | `two_stage_head_warmup` | 2 | 0.208054 | 0.374010 | 0.625990 | 0.000000 |
| H | `two_stage_head_warmup` | 3 | 0.244208 | 0.398134 | 0.601866 | 0.000000 |
| H | `two_stage_head_warmup` | 4 | 0.232938 | 0.383733 | 0.616267 | 0.000000 |

| ID | Count | Mean WER | Std WER | Min WER | Max WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| H | 5 | 0.225792 | 0.012595 | 0.208054 | 0.244208 |
| F | 5 | 0.246672 | 0.017991 | 0.219080 | 0.269305 |

### H vs F paired-fold comparison

| Fold | H WER | F WER | Winner | Absolute difference |
| ---: | ---: | ---: | --- | ---: |
| 0 | 0.216511 | 0.239875 | H | 0.023364 |
| 1 | 0.227251 | 0.263260 | H | 0.036010 |
| 2 | 0.208054 | 0.219080 | H | 0.011026 |
| 3 | 0.244208 | 0.269305 | H | 0.025097 |
| 4 | 0.232938 | 0.241840 | H | 0.008902 |

Selected acoustic configuration: **H** (`two_stage_head_warmup`), based on its lower five-fold mean WER and 5/5 paired-fold wins over F.

## Final acoustic checkpoint

- `outputs/base_strict_final/best_model`

## Validation decoding

| Decoder | Beam width | Alpha | Beta | Validation WER |
| --- | ---: | ---: | ---: | ---: |
| Greedy |  |  |  | 0.232938 |
| Beam | 50 |  |  | 0.232938 |
| Train-text trigram LM fusion | 50 | 0.3 | 1.5 | 0.228487 |

Selected decoder:

```json
{
  "acoustic_checkpoint": "/home/user/disk4/LCM/assignments/SR/outputs/base_strict_final/best_model",
  "alpha": 0.3,
  "attention_mask_passed": false,
  "beam_width": 50,
  "beta": 1.5,
  "decoding_method": "beam_lm",
  "language_model_path": "/home/user/disk4/LCM/assignments/SR/results/base_strict_final/train_text_trigram_lm.json",
  "selection_source": "validation_only",
  "test_splits_used_for_selection": false,
  "validation_shards": "/home/user/disk4/LCM/assignments/SR/data/train/shard-000004.tar",
  "validation_wer": 0.228486646884273
}
```

## Final test results

| Split | WER |
| --- | ---: |
| test-clean | 0.246386 |
| test-other | 0.329289 |

## Limitations

- The acoustic training set is small.
- No supervised LibriSpeech ASR checkpoint was used.
- The language model was trained only on final train-shard transcripts.
- Decoder hyperparameters were tuned only on validation shard `000004`.
- Results are not directly comparable to supervised LibriSpeech-960h checkpoints.
