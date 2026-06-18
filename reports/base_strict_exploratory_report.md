# Post-final Wav2Vec2-base Exploratory Results

These post-final experiments do not replace the main reproducible result. They
use the same unsupervised `facebook/wav2vec2-base` source and the frozen main
decoder; no supervised LibriSpeech ASR checkpoint is used.

## H all-train

The fixed H schedule was trained on all five train shards: 10 head-only epochs
followed by 40 encoder-training epochs. Because every train shard was used,
there is no untouched in-domain validation split and the fixed final epoch was
retained.

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| H all-train | 0.216867 | 0.303307 |

## H-fold ROVER ensemble

Five H fold models were combined with deterministic word-alignment voting.

| Experiment | test-clean WER | test-other WER |
| --- | ---: | ---: |
| H-fold ROVER ensemble | 0.197314 | 0.271383 |

The ROVER ensemble is the best observed exploratory result, but it is report
only. Folds 0–3 trained on shard `000004`; only fold 4 held it out. The
validation WER used to choose the ensemble method is therefore leaked and
optimistic rather than an unbiased held-out estimate.
