# Post-final Wav2Vec2-CTC Exploratory Results

These experiments extend the selected Staged CTC Fine-tuning recipe after the
main reproducible result was finalized.

## Staged CTC All-Train Model

The fixed staged schedule was trained on all five train shards: 10 head-only
epochs followed by 40 encoder fine-tuning epochs. Because every train shard was
used, there is no untouched in-domain validation split and the fixed final
epoch was retained.

| Setting | test-clean WER | test-other WER |
| --- | ---: | ---: |
| Staged CTC All-Train Model | 0.216867 | 0.303307 |

## Staged CTC Fold Ensemble with ROVER Voting

Five staged fine-tuning fold models were combined with deterministic
word-alignment voting.

| Setting | test-clean WER | test-other WER |
| --- | ---: | ---: |
| Staged CTC Fold Ensemble with ROVER Voting | 0.197314 | 0.271383 |

The ROVER system gives the best observed test WER, but it remains exploratory.
Folds 0–3 trained on shard `000004`, while only fold 4 held it out, so the
ensemble’s validation selection is affected by fold-membership leakage.
