# Strict Wav2Vec2-base Post-final Exploratory Report

These are post-final exploratory experiments. The preserved main result remains the strict final result; none of its result files or decoder selection were replaced.

## 1. Preserved strict final result

| Acoustic configuration | Decoder | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| H / `two_stage_head_warmup` | beam + train-text trigram LM (beam 50, alpha 0.3, beta 1.5) | 0.228487 | 0.246386 | 0.329289 |

- Main initialization remained `facebook/wav2vec2-base`.
- `facebook/wav2vec2-base-960h` and every supervised LibriSpeech ASR checkpoint remained forbidden.
- No official LibriSpeech LM or pretrained neural LM was used.
- The fixed decoder and LM were reused read-only from the preserved main run.

## 2. Exploratory H-fold ensemble

Five existing H fold models were combined using raw-logit averaging and a deterministic progressive word-alignment/voting fallback.

### Validation diagnostics on shard 000004

| Method | Type | Validation WER |
| --- | --- | ---: |
| `fold_0` | single_model_diagnostic | 0.000000 |
| `fold_1` | single_model_diagnostic | 0.000000 |
| `fold_2` | single_model_diagnostic | 0.000000 |
| `fold_3` | single_model_diagnostic | 0.000000 |
| `fold_4` | single_model_diagnostic | 0.228487 |
| `logit_average` | ensemble_candidate | 0.108803 |
| `rover` | ensemble_candidate | 0.000000 |

Selected exploratory ensemble: **rover** with validation WER 0.000000.

> Important leakage caveat: H folds 0–3 trained on shard 000004; only fold 4 held it out. The ensemble validation number is therefore optimistic and is not an unbiased held-out estimate.

### Frozen-method test evaluation

| Split | Ensemble WER | Preserved main WER | Change | Improved? |
| --- | ---: | ---: | ---: | --- |
| test-clean | 0.197314 | 0.246386 | -0.049072 | yes |
| test-other | 0.271383 | 0.329289 | -0.057907 | yes |

## 3. Exploratory H all-train model

The model used all five train shards with the fixed H schedule: 10 head-only epochs followed by 40 encoder-training epochs. The final epoch was retained without validation-based checkpoint selection. Evaluation used the preserved decoder and original train-shards-000000–000003 trigram LM unchanged.

| Split | All-train WER | Preserved main WER | Change | Improved? |
| --- | ---: | ---: | ---: | --- |
| test-clean | 0.216867 | 0.246386 | -0.029519 | yes |
| test-other | 0.303307 | 0.329289 | -0.025982 | yes |

- Checkpoint: `outputs/base_strict_exploratory/h_alltrain/best_model`.
- The directory name `best_model` satisfies the requested namespace; this artifact is the fixed final-epoch model, not a newly tuned validation-best checkpoint.

## 4. Optional head-refinement validation diagnostics

| Variant | Training-selection greedy WER | Fixed-decoder validation WER | Test evaluated? |
| --- | ---: | ---: | --- |
| H2 | 0.216123 | 0.216123 | no |
| H3 | 0.235410 | 0.239367 | no |

H4 checkpoint averaging was skipped: the retained fold4 checkpoints were epochs 22, 29, and 30, so there was no nearby window around the best epoch-22 checkpoint suitable for local averaging.

## Interpretation and limitations

- These are post-final exploratory results, not replacements for the preserved strict final result.
- No 960h or supervised LibriSpeech ASR checkpoint was used.
- No external official LibriSpeech LM or pretrained neural LM was used.
- The ensemble method was selected only from shard 000004, but that selection is affected by the fold-membership leakage described above.
- Test-clean and test-other had already been observed in the preserved main run. Any later exploratory test comparison should therefore be interpreted cautiously even though no exploratory hyperparameters were tuned on test.
- The all-train model has no untouched in-domain validation split.

## Output paths

- Ensemble validation: `results/base_strict_exploratory/h_fold_ensemble_validation.csv`
- Ensemble test: `results/base_strict_exploratory/h_fold_ensemble_test_summary.csv`
- All-train summary: `results/base_strict_exploratory/h_alltrain_wer_summary.csv`
- All-train checkpoint: `outputs/base_strict_exploratory/h_alltrain/best_model`
- This report: `reports/base_strict_exploratory_report.md`
