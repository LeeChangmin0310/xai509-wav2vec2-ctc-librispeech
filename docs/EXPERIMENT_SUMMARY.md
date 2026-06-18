# XAI509 Automatic Speech Recognition Experiment Summary

## Project goal

This semester project adapts `facebook/wav2vec2-base` to the
course-provided LibriSpeech WebDataset using a CTC objective. The implementation
covers WebDataset loading, transcript normalization, Wav2Vec2-CTC fine-tuning,
validation-based acoustic and decoder selection, train-text language-model
fusion, and WER evaluation on `test-clean` and `test-other`.

## Dataset and split protocol

The provided data contains five training archives and separate test splits:

```text
data/train/shard-000000.tar ... shard-000004.tar
data/test-clean/*.tar
data/test-other/*.tar
```

Five-fold acoustic comparison holds out one train shard per fold and trains on
the other four. For the final main run, shards `000000`–`000003` provide the
acoustic and language-model training data, while shard `000004` is reserved for
checkpoint selection and decoder tuning. The two test splits are decoded only
after the acoustic model and decoder have been selected.

## Model initialization

All main experiments initialize from `facebook/wav2vec2-base`. The CTC output
layer is initialized for the project tokenizer and trained on the provided
transcripts. This keeps the acoustic adaptation process centered on the
self-supervised Wav2Vec2 representation rather than an existing ASR output
head. Model provenance is recorded by the code for reproducibility.

## Wav2Vec2-CTC pipeline

Audio and transcripts are read directly from tar-based WebDataset shards.
Waveforms are converted to mono when necessary and processed by the Wav2Vec2
feature extractor. Transcripts are uppercased and normalized to the tokenizer
alphabet.

`AutoModelForCTC` adds a CTC projection layer above the Wav2Vec2 encoder. Main
training uses Hugging Face model loss, CTC `zero_infinity`, dynamic padding,
finite-loss preflight checks, gradient clipping, weak SpecAugment, and
WER-based validation. Inference supports greedy decoding and CTC beam search.
WER is computed from normalized reference/hypothesis pairs.

## Staged CTC Fine-tuning

The selected acoustic method is **Staged CTC Fine-tuning** (internal ID: H).
Its two stages separate adaptation of the newly initialized classification
layer from broader acoustic fine-tuning:

1. Freeze the Wav2Vec2 backbone and train the CTC head for 10 epochs at
   learning rate `1e-3`.
2. Initialize from the warmed-up model, freeze the convolutional feature
   extractor, and fine-tune the Transformer encoder at learning rate `1e-4`
   with a `1e-3` head learning rate.

The second stage uses validation WER for checkpoint selection in the main
experiment. Warming up the head first makes useful nonblank alignments more
likely before encoder parameters begin moving.

## Initial blank-collapse diagnosis

`facebook/wav2vec2-base` does not provide a CTC classification layer trained
for the project vocabulary. Initial learning-rate sweep therefore
started with a random CTC head and converged to blank-dominant outputs.

Fold-0 diagnostics for the early variants showed validation WER around `1.0`,
empty hypotheses for every sample, a blank-token rate of `1.0`, and a
nonblank-token rate of `0.0`. These results motivated direct inspection of
logits, labels, CTC lengths, loss finiteness, and train/evaluation mode rather
than treating the failure as a decoding problem.

## SpecAugment and stability diagnosis

Default SpecAugment was too aggressive for the small-data, random-head setting
and could trigger non-finite behavior or blank collapse in train mode. Turning
augmentation off helped isolate the cause, after which a weaker configuration
was introduced:

```text
mask_time_prob = 0.01
mask_time_length = 5
min_masks = 1
```

This setting passed the finite-loss preflight and retained modest acoustic
augmentation without recreating the original instability. It was used for the
stable acoustic configurations.

## Training configurations

| Configuration | Main idea | Outcome |
| --- | --- | --- |
| Early base-only LR variants | Lower-LR fine-tuning with a randomly initialized CTC head | Blank collapse; validation WER around 1.0 |
| Frozen-feature LR-2e-4 Baseline (internal ID: F) | Freeze the feature encoder and train at LR `2e-4` with weak SpecAugment | Stable; five-fold mean WER 0.246672 |
| Staged CTC Fine-tuning (internal ID: H) | Warm up the CTC head, then fine-tune the encoder | Stable; five-fold mean WER 0.225792; selected |
| Staged CTC refinement run 1 | Post-hoc refinement experiment | Validation WER 0.216123; diagnostic only |
| Staged CTC refinement run 2 | Alternative post-final refinement diagnostic | Validation WER 0.239367; diagnostic only |
| Staged CTC All-Train Model | Apply the fixed staged schedule to all five train shards | test-clean 0.216867; test-other 0.303307; exploratory |
| Staged CTC Fold Ensemble with ROVER Voting | Word-alignment voting across five staged fine-tuning fold models | test-clean 0.197314; test-other 0.271383; best observed exploratory |

## Acoustic model comparison

Staged CTC Fine-tuning and the Frozen-feature LR-2e-4 Baseline were compared on
the same five train-shard folds.

| Fold | Staged CTC Fine-tuning WER | Frozen-feature baseline WER |
| ---: | ---: | ---: |
| 0 | 0.216511 | 0.239875 |
| 1 | 0.227251 | 0.263260 |
| 2 | 0.208054 | 0.219080 |
| 3 | 0.244208 | 0.269305 |
| 4 | 0.232938 | 0.241840 |

| Configuration | Mean WER | Standard deviation | Paired-fold wins |
| --- | ---: | ---: | ---: |
| Staged CTC Fine-tuning | 0.225792 | 0.012595 | 5 / 5 |
| Frozen-feature LR-2e-4 Baseline | 0.246672 | 0.017991 | 0 / 5 |

The staged method achieved lower WER on every paired fold, as well as a lower
mean and standard deviation, so it was selected for the main model.

## Decoder tuning and train-text trigram LM fusion

Decoder selection used only validation shard `000004`. A small word trigram
model was trained solely from transcripts in training shards
`000000`–`000003`; no external text corpus or pretrained neural language model
was used.

| Decoder | Beam width | Alpha | Beta | Validation WER |
| --- | ---: | ---: | ---: | ---: |
| Greedy | — | — | — | 0.232938 |
| Beam | 50 | — | — | 0.232938 |
| Beam + train-text trigram LM | 50 | 0.3 | 1.5 | 0.228487 |

Plain beam search matched greedy decoding. Shallow fusion with the train-text
trigram model improved validation WER, so beam width 50, alpha 0.3, and beta
1.5 were frozen before test decoding.

## Main result

| Setting | Role | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| Staged CTC fine-tuned Wav2Vec2 + beam/trigram LM | Main reproducible result | 0.228487 | 0.246386 | 0.329289 |

This is the primary project result. Both the acoustic checkpoint and decoder
were selected without using the test splits.

## Exploratory improvements

| Setting | Role | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| Staged CTC All-Train Model | Exploratory all-train | — | 0.216867 | 0.303307 |
| Staged CTC Fold Ensemble with ROVER Voting | Best observed exploratory | — | 0.197314 | 0.271383 |

The all-train model applies the fixed staged schedule to all five train shards
and retains the final epoch, so it has no untouched in-domain validation split.

The ROVER system combines fold hypotheses using deterministic progressive word
alignment and voting. It gives the best observed test WER, but it is reported
as exploratory because folds 0–3 trained on shard `000004`; validation
selection is therefore affected by fold-membership leakage. It is not the
clean validation-selected main model.

## Interpretation and limitations

The experiments indicate that stable optimization of the newly initialized CTC
head was the central challenge. Weak augmentation and staged optimization
produced a robust improvement over the frozen-feature baseline across all five
folds. Train-text trigram fusion contributed a smaller additional gain without
using external text.

Limitations include the small acoustic dataset, sensitivity to augmentation
and schedule choices, and the absence of an untouched validation split for the
all-train model. Exploratory test comparisons were also conducted after the
main test result was known. Finally, the ensemble’s validation selection is
leaked by fold membership, so its lower test WER should be interpreted as an
exploratory upper-bound observation rather than a replacement for the main
result.
