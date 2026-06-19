# Main Wav2Vec2-CTC Result

## Data and model setup

- Initialization: `facebook/wav2vec2-base`.
- Acoustic training: train shards `000000`–`000003`.
- Checkpoint selection and decoder tuning: train shard `000004`.
- Final evaluation: `test-clean` and `test-other`.
- Model provenance is recorded by the training code for reproducibility.

## Selected acoustic method

**Staged CTC Fine-tuning** (internal ID: H) first trains the newly initialized
CTC head for 10 epochs while the Wav2Vec2 encoder is frozen. It then fine-tunes
the encoder with the convolutional feature extractor frozen and weak
SpecAugment enabled (`mask_time_prob=0.01`, `mask_time_length=5`,
`min_masks=1`).

The selected decoder uses beam width 50 and the train-text trigram language
model with alpha 0.3 and beta 1.5.

| Setting | Validation WER | test-clean WER | test-other WER |
| --- | ---: | ---: | ---: |
| Staged CTC fine-tuned Wav2Vec2 + beam/trigram LM | 0.228487 | 0.246386 | 0.329289 |

## Limitations

- The acoustic training set is small.
- The trigram LM uses only transcripts from train shards `000000`–`000003`.
- Performance is sensitive to CTC-head initialization, augmentation strength,
  and the fine-tuning schedule.
