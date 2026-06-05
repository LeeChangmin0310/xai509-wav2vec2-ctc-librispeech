# Fine-Tuning Wav2Vec2-CTC on LibriSpeech WebDataset Shards

## 1. Abstract

This project fine-tuned Wav2Vec2 with CTC loss for automatic speech recognition
on the provided LibriSpeech WebDataset shards. The final experimental setup used
a strict validation protocol: train shards 0-3 were used for fine-tuning, train
shard 4 was used for validation and checkpoint selection, and the LibriSpeech
`test-clean` and `test-other` splits were reserved for final evaluation only.

The final model path started from the pretrained ASR checkpoint
`facebook/wav2vec2-base-960h`, used the Hugging Face CTC loss implementation,
trained in fp32, and disabled SpecAugment after diagnosing train-mode
SpecAugment as the source of NaN logits. The pretrained ASR control achieved
WER `0.186112` on test-clean and `0.245802` on test-other. The best greedy
fine-tuned model, `asrinit_lr1e-5_fp32_properval`, improved those scores to
`0.154709` and `0.214737`. Applying beam decoding to the same checkpoint gave
the best final result, `0.152598` on test-clean and `0.212235` on test-other.

The final beam result reduced WER relative to the pretrained ASR control by
`0.033514` absolute on test-clean and `0.033567` absolute on test-other. These
correspond to relative reductions of `18.01%` and `13.66%`. The conclusions are
kept conservative: within the tested settings, learning rate `1e-5` worked best,
freezing did not improve over full fine-tuning, the tested layer-wise learning
rate decay did not help, and beam decoding provided a small additional gain.

## 2. Introduction

Automatic speech recognition maps an audio sequence to a text sequence. This is
challenging because audio frames and output characters or tokens are not aligned
one-to-one. Connectionist Temporal Classification (CTC) is a standard solution
for this setting because it marginalizes over possible alignments between the
input acoustic frames and the output transcription.

Wav2Vec2 is a self-supervised speech representation model that can be adapted to
ASR by adding a CTC projection head and fine-tuning on labeled speech. The course
project skeleton included utilities and TODOs for Wav2Vec2 fine-tuning,
inference, WER evaluation, and a custom CTC loss implementation. The goal of this
project was to complete that skeleton, run controlled ASR experiments on the
provided LibriSpeech WebDataset shards, and report final WER on both
`test-clean` and `test-other`.

The final experimental design emphasizes separation between validation and test
data. Earlier diagnostic runs used `test-clean` as the Trainer evaluation split,
which made it act as validation and checkpoint-selection data. The final results
therefore use a stricter setup: one held-out training shard is used for
validation, and the two official test splits are used only after training for
final inference and WER computation.

## 3. Dataset and Strict Validation Split

The data was provided as final WebDataset `.tar` shards. These shards were read
directly and were not extracted. The expected local layout was:

```text
data/
  train/*.tar
  test-clean/*.tar
  test-other/*.tar
```

The strict final split was:

```text
Fine-tuning train shards:
  data/train/shard-000000.tar
  data/train/shard-000001.tar
  data/train/shard-000002.tar
  data/train/shard-000003.tar

Validation and checkpoint selection shard:
  data/train/shard-000004.tar

Final evaluation only:
  data/test-clean/*.tar
  data/test-other/*.tar
```

This means train shards 0-3 were used to update model parameters, train shard 4
was used by the Hugging Face Trainer for validation and best-checkpoint
selection, and `test-clean` and `test-other` were reserved for final inference
and WER evaluation. This split avoids using either test set as validation data.

The evaluation metric was word error rate (WER). WER was computed from result
files containing alternating reference and hypothesis lines for each decoded
sample.

## 4. Method

The implementation used Hugging Face Wav2Vec2 CTC models and processors. Audio
and text examples were loaded from WebDataset shards, converted to model inputs,
dynamically padded by a CTC data collator, and passed to the Trainer.

The final fine-tuning path used:

```text
model_name_or_path = facebook/wav2vec2-base-960h
loss_impl = hf
disable_spec_augment = true
ctc_zero_infinity = true
no_attention_mask_for_loss = true
training precision = fp32
```

The Hugging Face model loss was used as the default CTC loss path:

```python
outputs = model(
    input_values=input_values,
    attention_mask=attention_mask_or_none,
    labels=labels,
)
loss = outputs.loss
```

The project also retained the custom CTC implementation for diagnostic use, but
it was not used as the default for final experiments.

The experiments compared:

- pretrained ASR inference without fine-tuning,
- ASR-initialized full fine-tuning at several learning rates,
- feature-encoder freezing,
- freezing the first three encoder layers,
- layer-wise learning rate decay,
- greedy versus beam decoding for the selected best checkpoint.

Greedy decoding was the default inference method. Beam decoding used
`pyctcdecode` without requiring KenLM.

## 5. Stability Diagnosis

The initial `facebook/wav2vec2-base` fine-tuning runs collapsed to blank
predictions. This was not simply an evaluation-script issue: the pretrained ASR
control using `facebook/wav2vec2-base-960h` produced valid transcriptions and
reasonable WER:

```text
test-clean WER = 0.186112
test-other WER = 0.245802
```

This control validated the data reader, inference path, and WER computation.
Because the control worked, the failure was likely in the fine-tuning path rather
than in the dataset or evaluation pipeline.

An ASR-initialized training attempt also failed before the fix. It produced
unstable diagnostics, including non-finite values and WER near `1.0`. A dedicated
CTC loss probe then isolated the numerical failure: in train mode,
`apply_spec_augment=True` produced NaN logits. When SpecAugment was disabled
with `apply_spec_augment=False`, logits and CTC loss became finite.

The final training scripts therefore disabled SpecAugment by default and
explicitly used:

```text
--disable_spec_augment
--loss_impl hf
--ctc_zero_infinity
--no_attention_mask_for_loss
```

This made the ASR-initialized smoke test and the final proper-validation queue
numerically stable.

## 6. Experiments

The final strict proper-validation experiment set consisted of:

| Experiment | Initialization | Training setting | Decoding |
| --- | --- | --- | --- |
| `asr_pretrained_960h_full_properval` | `facebook/wav2vec2-base-960h` | No fine-tuning | Greedy |
| `asrinit_lr1e-6_fp32_properval` | ASR pretrained | Full fine-tuning, LR `1e-6` | Greedy |
| `asrinit_lr3e-6_fp32_properval` | ASR pretrained | Full fine-tuning, LR `3e-6` | Greedy |
| `asrinit_lr1e-5_fp32_properval` | ASR pretrained | Full fine-tuning, LR `1e-5` | Greedy |
| `asrinit_freeze_feature_lr3e-6_fp32_properval` | ASR pretrained | Feature encoder frozen, LR `3e-6` | Greedy |
| `asrinit_freeze3_lr3e-6_fp32_properval` | ASR pretrained | First 3 encoder layers frozen, LR `3e-6` | Greedy |
| `asrinit_layerwise_lr_decay_properval` | ASR pretrained | Layer-wise LR decay | Greedy |
| `asrinit_lr1e-5_fp32_properval_beam` | Best greedy checkpoint | Beam decoding, width `100` | Beam |

All fine-tuning runs used fp32 training, Hugging Face CTC loss, SpecAugment
disabled, and `ctc_zero_infinity` enabled.

## 7. Results

The final proper-validation results are:

| Experiment | Decoding | test-clean WER | test-other WER |
| --- | --- | ---: | ---: |
| `asr_pretrained_960h_full_properval` | Greedy | `0.186112` | `0.245802` |
| `asrinit_lr1e-6_fp32_properval` | Greedy | `0.180558` | `0.240567` |
| `asrinit_lr3e-6_fp32_properval` | Greedy | `0.169678` | `0.229620` |
| `asrinit_lr1e-5_fp32_properval` | Greedy | `0.154709` | `0.214737` |
| `asrinit_freeze_feature_lr3e-6_fp32_properval` | Greedy | `0.181566` | `0.241809` |
| `asrinit_freeze3_lr3e-6_fp32_properval` | Greedy | `0.176126` | `0.236612` |
| `asrinit_layerwise_lr_decay_properval` | Greedy | `0.184742` | `0.244292` |
| `asrinit_lr1e-5_fp32_properval_beam` | Beam | **`0.152598`** | **`0.212235`** |

The pretrained ASR control achieved WER `0.186112` on test-clean and `0.245802`
on test-other. The best greedy fine-tuned model was
`asrinit_lr1e-5_fp32_properval`, with WER `0.154709` and `0.214737`. Compared
with the control, this was an absolute improvement of `0.031403` on test-clean
and `0.031065` on test-other, or relative reductions of `16.87%` and `12.64%`.

The best final model was `asrinit_lr1e-5_fp32_properval_beam`. It reached
`0.152598` on test-clean and `0.212235` on test-other. Compared with the
pretrained ASR control, this improved WER by `0.033514` absolute on test-clean
and `0.033567` absolute on test-other, corresponding to relative reductions of
`18.01%` and `13.66%`.

Beam decoding improved over the best greedy checkpoint by `0.002111` absolute
WER on test-clean and `0.002502` on test-other. This was a small but consistent
additional gain.

## 8. Discussion

The results show that ASR-initialized fine-tuning improved over the pretrained
ASR control under the strict validation split. Among the tested learning rates,
`1e-5` performed best for greedy decoding. The lower learning rates improved
less: `1e-6` stayed close to the pretrained control, while `3e-6` produced a
moderate improvement but did not match `1e-5`.

Freezing did not help as much as unfrozen fine-tuning in these experiments.
With learning rate `3e-6`, the unfrozen run reached `0.169678` on test-clean and
`0.229620` on test-other. Freezing the feature encoder produced `0.181566` and
`0.241809`, while freezing the first three encoder layers produced `0.176126`
and `0.236612`. These freezing results improved little or moderately over the
control, but did not outperform unfrozen fine-tuning.

The tested layer-wise learning rate decay configuration also did not improve the
model. Its WER values, `0.184742` and `0.244292`, were close to the pretrained
ASR control. This does not prove that layer-wise decay is ineffective in
general; it only shows that the tested configuration was not helpful in this
specific setup.

The diagnostic process was important. Original base-model fine-tuning collapsed
to blank predictions, and ASR-initialized training initially produced NaN
behavior. The pretrained ASR control confirmed that data loading, inference, and
WER evaluation worked. The CTC probe identified train-mode SpecAugment as the
source of NaN logits. Disabling SpecAugment turned the project from a broken
fine-tuning path into a stable training pipeline.

The claims should remain conservative for several reasons. The experiment used a
small set of train shards, one validation shard, one random seed, a small number
of learning rates, and a limited set of freezing and layer-wise decay settings.
Different seeds, schedules, regularization choices, or decoding settings could
change the ranking. The strongest conclusion is therefore narrow: for this
strict split and tested configuration set, ASR-initialized full fine-tuning with
learning rate `1e-5`, followed by beam decoding, produced the best WER.

## 9. Conclusion

This project completed an end-to-end Wav2Vec2-CTC ASR pipeline for the provided
LibriSpeech WebDataset shards. The final evaluation used a strict split in which
train shards 0-3 were used for fine-tuning, train shard 4 was used for
validation and checkpoint selection, and `test-clean` and `test-other` were
reserved for final evaluation.

The final best model was `asrinit_lr1e-5_fp32_properval_beam`, with WER
`0.152598` on test-clean and `0.212235` on test-other. This improved over the
pretrained ASR control by `18.01%` relative on test-clean and `13.66%` relative
on test-other. The project also documented an important stability issue:
train-mode SpecAugment caused NaN logits during fine-tuning, and disabling
SpecAugment made training stable.

Within the tested settings, learning rate `1e-5` helped most, freezing did not
help as much as full fine-tuning, layer-wise learning rate decay did not improve
WER, and beam decoding gave a small additional gain. These findings are useful
for the project setting, but should not be overgeneralized beyond the tested
split, seed, and hyperparameter choices.
