# WER Summary

| Experiment | Train setting | Decoding | LR | Freeze setting | Layer-wise decay | Beam width | test-clean WER | test-other WER | Checkpoint |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `asr_pretrained_960h_full` | pretrained_asr_control | greedy |  | none | false |  | 0.186112 | 0.245802 | `facebook/wav2vec2-base-960h` |
| `asrinit_lr1e-6_fp32` | asr_initialized_finetuning | greedy | 1e-6 | none | false |  | 0.181223 | 0.241274 | `outputs/asrinit_lr1e-6_fp32/final_model` |
| `asrinit_lr3e-6_fp32` | asr_initialized_finetuning | greedy | 3e-6 | none | false |  | 0.164029 | 0.224137 | `outputs/asrinit_lr3e-6_fp32/final_model` |
| `asrinit_lr1e-5_fp32_fixed` | asr_initialized_finetuning | greedy | 1e-5 | none | false |  | 0.140958 | 0.199759 | `outputs/asrinit_lr1e-5_fp32_fixed/final_model` |
| `asrinit_lr1e-5_fp32_fixed_beam` | asr_initialized_finetuning | beam | 1e-5 | none | false | 100 | **0.139227** | **0.196760** | `outputs/asrinit_lr1e-5_fp32_fixed/final_model` |
| `asrinit_freeze_feature_lr3e-6_fp32` | asr_initialized_finetuning | greedy | 3e-6 | feature_encoder | false |  | 0.180044 | 0.240682 | `outputs/asrinit_freeze_feature_lr3e-6_fp32/final_model` |
| `asrinit_freeze3_lr3e-6_fp32` | asr_initialized_finetuning | greedy | 3e-6 | first_3_encoder_layers | false |  | 0.172208 | 0.232639 | `outputs/asrinit_freeze3_lr3e-6_fp32/final_model` |
| `asrinit_layerwise_lr_decay_fixed` | asr_initialized_finetuning | greedy | 3e-6 | feature_extractor | 0.9 |  | 0.184438 | 0.244312 | `outputs/asrinit_layerwise_lr_decay_fixed/final_model` |

Best values are shown in bold.
