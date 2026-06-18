# XAI509 Automatic Speech Recognition Experiment Summary

## Project goal

This XAI509 Automatic Speech Recognition semester project investigates how to
adapt the self-supervised `facebook/wav2vec2-base` encoder to a small,
course-provided LibriSpeech WebDataset. The implementation covers WebDataset
loading, transcript normalization, Wav2Vec2-CTC fine-tuning, validation-based
checkpoint and decoder selection, language-model-assisted decoding, and WER
evaluation.

The central experimental question was not simply whether Wav2Vec2 could fit the
data, but how to train a newly initialized CTC output head without relying on a
supervised LibriSpeech ASR checkpoint. The final main result therefore uses the
unsupervised base checkpoint and a reproducible validation protocol.

## Dataset and data split protocol

The provided data contains five training archives and separate test splits:

```text
data/train/shard-000000.tar ... shard-000004.tar
data/test-clean/*.tar
data/test-other/*.tar
```

The five train shards served two related purposes:

- During acoustic model comparison, five-fold cross-validation held out one
  train shard per fold and trained on the other four.
- For the final main run, shards `000000`–`000003` were the acoustic and
  language-model training data. Shard `000004` was the only validation shard
  used for checkpoint selection and decoder tuning.

The `test-clean` and `test-other` archives were excluded from training,
checkpoint selection, and decoder tuning. They were decoded only after the
main acoustic model and decoder configuration had been selected.

## Checkpoint provenance policy

Main experiments use `facebook/wav2vec2-base`, which is a self-supervised
acoustic representation model rather than a LibriSpeech ASR-fine-tuned model.
The following checkpoint names are banned from main results:

- `facebook/wav2vec2-base-960h`
- paths containing `asrinit`
- paths containing `asr_pretrained`

No supervised LibriSpeech ASR-fine-tuned checkpoint is used in the reported
main or exploratory results. Saved main checkpoints include
`checkpoint_provenance.json`, and `src/guard.py` verifies both the source
checkpoint and experiment role before main evaluation. This prevents an
accidental supervised checkpoint from entering the main pipeline under a local
directory name.

## Wav2Vec2-base + CTC pipeline

Audio and transcripts are read directly from tar-based WebDataset shards.
Waveforms are converted to mono when necessary and passed through the
Wav2Vec2 feature extractor. Transcripts are uppercased and normalized to the
CTC tokenizer alphabet.

`AutoModelForCTC` supplies a CTC projection layer above the Wav2Vec2 encoder.
Training uses Hugging Face model loss for the main experiments, CTC
`zero_infinity`, dynamic batch padding, finite-loss preflight checks, gradient
clipping, and WER-based validation. Each exported model includes the processor
and provenance metadata needed for evaluation.

Inference supports greedy decoding and CTC beam search. WER is computed from
normalized reference/hypothesis pairs. Raw checkpoints and predictions are
generated under ignored output directories; only compact result summaries are
kept in Git.

## Initial failures and blank collapse

`facebook/wav2vec2-base` has no supervised ASR CTC head. When loaded for CTC,
the output projection is randomly initialized, so the early optimization
problem is substantially harder than fine-tuning an existing ASR model.

The initial lower-learning-rate configurations A, B, and C all collapsed to
the CTC blank token. Fold-0 diagnostics showed validation WER `1.0`, empty
hypotheses for every sample, a blank-token rate of `1.0`, and a nonblank-token
rate of `0.0`. Increasing training time at those settings did not fix the
underlying optimization behavior.

This failure prompted targeted diagnostics of logits, labels, input lengths,
loss finiteness, and augmentation behavior. The solution was not to switch to
`facebook/wav2vec2-base-960h`; doing so would have introduced a supervised ASR
checkpoint and invalidated the intended base-only comparison.

## SpecAugment diagnosis

The diagnostic runs isolated a second instability in the small-data,
random-head setting. Default SpecAugment was too aggressive and could produce
non-finite behavior or blank collapse during train-mode forward passes, even
when evaluation-mode inference was healthy.

Completely removing augmentation was useful diagnostically, but the selected
training policy retained a weaker form of SpecAugment:

```text
mask_time_prob = 0.01
mask_time_length = 5
min_masks = 1
```

This setting passed the finite-loss preflight and avoided the collapse observed
with the default masking configuration. It was used for the stable F and H
acoustic configurations.

## Training configurations tried

| ID / setting | Main idea | Outcome |
| --- | --- | --- |
| A/B/C | Early base-only, lower-LR variants with a randomly initialized CTC head | Blank collapse; validation WER around 1.0 |
| F / `lr2e-4_freeze_feature` | Freeze the feature encoder and train with LR `2e-4` plus weak SpecAugment | Stable; five-fold mean WER 0.246672 |
| H / `two_stage_head_warmup` | 10 head-only epochs, then encoder fine-tuning with weak SpecAugment | Stable; five-fold mean WER 0.225792; selected |
| H2 | Post-final head-refinement diagnostic | Validation WER 0.216123; diagnostic only |
| H3 | Alternative post-final head-refinement diagnostic | Validation WER 0.239367; diagnostic only |
| H all-train | Fixed H schedule using all five train shards | test-clean 0.216867; test-other 0.303307; exploratory |
| H-fold ROVER | Word-alignment voting across five H fold models | test-clean 0.197314; test-other 0.271383; best observed exploratory |

Configuration H uses two stages. Stage 1 freezes the Wav2Vec2 backbone and
trains the CTC head for 10 epochs at learning rate `1e-3`. Stage 2 starts from
that checkpoint, freezes the convolutional feature encoder, and trains the
Transformer encoder at learning rate `1e-4` with a `1e-3` head learning rate.
Validation WER controls checkpoint selection during the main run.

## H/F 5-fold acoustic CV

The two stable candidates were compared on the same five folds.

| Fold | H WER | F WER | Winner |
| ---: | ---: | ---: | --- |
| 0 | 0.216511 | 0.239875 | H |
| 1 | 0.227251 | 0.263260 | H |
| 2 | 0.208054 | 0.219080 | H |
| 3 | 0.244208 | 0.269305 | H |
| 4 | 0.232938 | 0.241840 | H |

| Configuration | Mean WER | Standard deviation | Paired-fold wins |
| --- | ---: | ---: | ---: |
| H / `two_stage_head_warmup` | 0.225792 | 0.012595 | 5 / 5 |
| F / `lr2e-4_freeze_feature` | 0.246672 | 0.017991 | 0 / 5 |

H had both the lower mean WER and lower fold-to-fold variation, and it won all
five direct paired comparisons. It was therefore selected as the main acoustic
training recipe.

## Decoder tuning and train-text trigram LM fusion

Decoder selection used only validation shard `000004`. The language model was
a small word trigram model trained solely from transcripts in training shards
`000000`–`000003`; no official LibriSpeech language model or pretrained neural
language model was used.

| Decoder | Beam width | Alpha | Beta | Validation WER |
| --- | ---: | ---: | ---: | ---: |
| Greedy | — | — | — | 0.232938 |
| Beam | 50 | — | — | 0.232938 |
| Beam + train-text trigram LM | 50 | 0.3 | 1.5 | 0.228487 |

Plain beam search did not improve over greedy decoding. Shallow fusion with the
train-text trigram LM produced a modest but consistent validation improvement,
so beam width 50, alpha 0.3, and beta 1.5 were frozen before test decoding.

## Final main result

The main result is the validation-selected H single model decoded with the
selected train-text trigram LM configuration.

| Setting | Role | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| H single model + beam LM | Main reproducible result | 0.228487 | 0.246386 | 0.329289 |

This is the clean result to use when describing the project’s primary
performance: its acoustic checkpoint and decoder were selected without using
the test splits.

## Post-final exploratory improvements

Two experiments were run after the main protocol had been completed.

| Setting | Role | test-clean WER | test-other WER |
| --- | --- | ---: | ---: |
| H all-train | Exploratory all-train | 0.216867 | 0.303307 |
| H-fold ROVER ensemble | Best observed exploratory | 0.197314 | 0.271383 |

The H all-train model used the fixed H schedule on all five train shards and
kept the final epoch. It improved both test splits, but it has no untouched
in-domain validation set.

The ROVER system combined the five H fold hypotheses using deterministic
progressive word alignment and voting. It achieved the best observed test WER,
but it must remain an exploratory result. Folds 0–3 had trained on shard
`000004`, while only fold 4 held that shard out. Consequently, the validation
selection used for the ensemble is affected by fold-membership leakage and is
not an unbiased held-out estimate. ROVER must not be presented as the clean
main validation-selected model.

## Interpretation and limitations

The experiments show that the main difficulty was optimization of a random CTC
head on limited data, not merely decoding. Conservative augmentation and a
head-warmup stage materially improved stability, and H’s five-fold advantage
over F suggests that the gain was robust across the available train shards.
The train-text trigram LM provided a smaller improvement than the acoustic
training changes, but it improved the selected validation model without using
external text.

The main limitations are the small acoustic dataset, sensitivity to
augmentation and training schedule, and the absence of an untouched validation
split for the all-train experiment. The exploratory test results were produced
after the main test had already been observed and should therefore be
interpreted cautiously. Most importantly, the ROVER validation result is
leaked by fold membership; its lower test WER is an interesting upper-bound
observation, not a replacement for the reproducible main result.
