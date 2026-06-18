# SpecAugment Finite-Loss Check

- Status: **PASS**
- Checkpoint: `facebook/wav2vec2-base`
- Experiment role: `main`
- Device: `cuda`
- Input shape: `(2, 248480)`
- Labels shape: `(2, 248)`
- Input values finite: `True`
- Label IDs/padding valid: `True`
- Label padding includes `-100`: `True`
- Target lengths: `[248, 170]`
- CTC input lengths: `[776, 776]`
- Targets fit CTC inputs: `True`
- Processor return_attention_mask: `False`
- Attention mask passed to loss forward: `False`
- `ctc_zero_infinity`: `True`
- SpecAugment remained enabled for every attempted variant.

## Forward Variants

| Variant | mask_time_prob | mask_time_length | Logits finite | Loss finite | Loss |
| --- | ---: | ---: | --- | --- | ---: |
| default SpecAugment | 0.05 | 10 | True | True | 9.894462585449219 |

Full training is permitted only when this check passes. If the default variant fails, the script tries lighter masking while keeping SpecAugment enabled; it never silently disables SpecAugment.
