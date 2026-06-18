# Tiny-Overfit Sanity Check

## Scope

This is a diagnostic-only experiment. It uses the same eight utterances from
`data/train/shard-000001.tar` for training and evaluation. It does not use
`test-clean`, `test-other`, `facebook/wav2vec2-base-960h`, or any supervised
ASR checkpoint.

Both successful settings initialize from `facebook/wav2vec2-base`, freeze the
convolutional feature extractor, use the Hugging Face CTC loss with
`ctc_zero_infinity=True`, and train the encoder at `1e-4` and the random CTC
head at `1e-3`.

## Results

| Setting | SpecAugment | Final epoch | Train loss | Tiny WER | Empty rate | Blank rate | Nonblank rate | Avg. hypothesis characters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No augmentation | OFF | 100 | 0.1022 | 0.0709 | 0.0000 | 0.3540 | 0.6460 | 179.4 |
| Lighter augmentation | ON (`p=0.01`, length `5`, min masks `1`) | 100 | 0.1112 | 0.0676 | 0.0000 | 0.3929 | 0.6071 | 180.8 |
| Original augmentation diagnostic | ON (`p=0.05`, length `10`, min masks `2`) | 100 | 2.8568 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0 |

The no-augmentation run first produced nonblank validation text at epoch 40.
The lighter SpecAugment run first produced nonblank text at epoch 45. By epoch
100, both settings closely transcribed all eight memorized utterances.

## Interpretation

The tokenizer, `-100` label padding, Hugging Face CTC loss, optimizer path,
model loading, and greedy CTC decoding can learn and emit nonblank text.
Therefore the broad-run blank collapse is not evidence of a broken CTC or
decoding implementation.

The original SpecAugment strength prevented this tiny diagnostic from learning,
while a weaker but still enabled setting succeeded. Stronger fold-0 candidates
should keep SpecAugment ON using the validated lighter setting before any
five-fold expansion.

## Artifacts

- No augmentation: `results/base_strict_debug/tiny_overfit_noaug/`
- Lighter SpecAugment: `results/base_strict_debug/tiny_overfit_specaug/`
- Original SpecAugment failure: `results/base_strict_debug/tiny_overfit_specaug_default/`
