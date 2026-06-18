# Final Strict Wav2Vec2-base ASR Report

## Protocol

- Initialization: `facebook/wav2vec2-base`.
- Acoustic training: train shards `000000`–`000003`.
- Checkpoint selection and decoder tuning: train shard `000004` only.
- Final tests: accessed after acoustic and decoder selection.
- Supervised ASR checkpoints, including `facebook/wav2vec2-base-960h`, were not
  used.
- Checkpoint provenance is written by `src/train.py` and enforced by
  `src/guard.py`.

## Selected system

The selected acoustic configuration was H / `two_stage_head_warmup`: 10
head-only epochs followed by encoder training with weak SpecAugment
(`mask_time_prob=0.01`, `mask_time_length=5`, one minimum mask).

The selected validation-only decoder used beam width 50 and the train-text
trigram LM with alpha 0.3 and beta 1.5.

| Acoustic configuration | Decoder | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| H / `two_stage_head_warmup` | beam + train-text trigram LM | 0.228487 | 0.246386 | 0.329289 |

## Limitations

- The acoustic training set is small.
- The trigram LM uses only transcripts from train shards `000000`–`000003`.
- These results are not directly comparable to systems initialized from a
  supervised LibriSpeech ASR checkpoint.
