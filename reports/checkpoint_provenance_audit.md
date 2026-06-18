# Checkpoint Provenance Audit

## Final Rule

All main training, model selection, decoder tuning, and final result reporting
must use the unsupervised checkpoint:

```text
facebook/wav2vec2-base
```

The following checkpoint is banned from main experiments:

```text
facebook/wav2vec2-base-960h
```

`facebook/wav2vec2-base-960h` is already supervised ASR-fine-tuned on
LibriSpeech 960 hours. Using it as the main initialization would import
supervised LibriSpeech ASR training into a project whose goal is to fine-tune
the unsupervised base model on the provided course data.

The `960h` checkpoint may be used only when a command explicitly declares:

```text
--experiment_role positive_control_only
```

It must not appear in a main training run, main checkpoint, hyperparameter
selection result, main WER table, or final conclusion.

## Repository Findings

The audit searched for:

- `facebook/wav2vec2-base-960h`
- `960h`
- `asr_pretrained`
- `asrinit`
- `pretrained ASR control`

Historical references were found in:

- the previous README and final-report materials,
- ASR-initialized experiment configs,
- old `run_asrinit*.sh` queues,
- the previous CTC probe and ASR-init smoke script,
- result summarization order and status output.

These files describe or reproduce earlier diagnostic work. They are not part of
the new `base_strict_*` main result path. Historical results may remain in the
repository for transparency, but they must be labeled as supervised positive
controls or invalid-for-main comparisons.

## Code-Level Enforcement

`experiment_guard.py` enforces checkpoint provenance:

1. Any model path containing `960h`, `asr_pretrained`, or `asrinit` is rejected
   unless the role is exactly `positive_control_only`.
2. Main training must initialize exactly from
   `facebook/wav2vec2-base`.
3. Main inference accepts only the exact base source or a local checkpoint with
   `checkpoint_provenance.json` stating that its main source was
   `facebook/wav2vec2-base`.
4. Strict main training requires train-only validation shards and rejects
   `test-clean` or `test-other` in train/eval shard specifications.

## Main Output Namespace

The provenance-clean experiment family uses:

```text
outputs/base_strict_cv/
results/base_strict_cv/
outputs/base_strict_final/
results/base_strict_final/
reports/final_strict_base_report.md
```

Old `asrinit*`, `asr_pretrained*`, and `*960h*` artifacts are excluded from the
new main result selection and final table.
