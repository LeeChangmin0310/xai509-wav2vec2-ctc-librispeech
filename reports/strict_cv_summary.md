# Strict Wav2Vec2-base CV Summary

`strict_cv_summary.csv` did not previously exist and is now generated.
Fold 0 already existed before this CV expansion.

## Per-fold results

| ID | Candidate | Fold | Best WER | Epoch | Step | Eval loss | Blank | Nonblank | Empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | `lr2e-4_freeze_feature` | 0 | 0.239875 | 49.0 | 1421 | 0.710217 | 0.421801 | 0.578199 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 1 | 0.263260 | 29.0 | 841 | 0.606155 | 0.368046 | 0.631954 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 2 | 0.219080 | 38.0 | 1102 | 0.544856 | 0.358743 | 0.641257 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 3 | 0.269305 | 43.0 | 1204 | 0.792801 | 0.403066 | 0.596934 | 0.000000 |
| F | `lr2e-4_freeze_feature` | 4 | 0.241840 | 49.0 | 1421 | 0.644391 | 0.419275 | 0.580725 | 0.000000 |
| H | `two_stage_head_warmup` | 0 | 0.216511 | 37.0 | 1073 | 0.591116 | 0.403518 | 0.596482 | 0.000000 |
| H | `two_stage_head_warmup` | 1 | 0.227251 | 31.0 | 899 | 0.537694 | 0.375605 | 0.624395 | 0.000000 |
| H | `two_stage_head_warmup` | 2 | 0.208054 | 35.0 | 1015 | 0.465839 | 0.374010 | 0.625990 | 0.000000 |
| H | `two_stage_head_warmup` | 3 | 0.244208 | 40.0 | 1120 | 0.719905 | 0.398134 | 0.601866 | 0.000000 |
| H | `two_stage_head_warmup` | 4 | 0.232938 | 22.0 | 638 | 0.517387 | 0.383733 | 0.616267 | 0.000000 |

## Aggregate results

| ID | Candidate | Count | Mean WER | Std WER | Min WER | Max WER | Blank folds | Empty-issue folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H | `two_stage_head_warmup` | 5 | 0.225792 | 0.012595 | 0.208054 | 0.244208 | 0 | 0 |
| F | `lr2e-4_freeze_feature` | 5 | 0.246672 | 0.017991 | 0.219080 | 0.269305 | 0 | 0 |

## Paired fold comparison

| Fold | H WER | F WER | Winner | Absolute difference |
| ---: | ---: | ---: | --- | ---: |
| 0 | 0.216511 | 0.239875 | H | 0.023364 |
| 1 | 0.227251 | 0.263260 | H | 0.036010 |
| 2 | 0.208054 | 0.219080 | H | 0.011026 |
| 3 | 0.244208 | 0.269305 | H | 0.025097 |
| 4 | 0.232938 | 0.241840 | H | 0.008902 |

## Selected acoustic configuration

Selected **H** (`two_stage_head_warmup`): Selected by lower five-fold mean validation WER. Mean WER `0.225792`, std `0.012595`.

Only the five train-shard folds were used. Test-clean and test-other were not used.
`facebook/wav2vec2-base-960h` and other supervised LibriSpeech ASR checkpoints were not used.
