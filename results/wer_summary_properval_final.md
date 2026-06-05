# WER Summary

| Experiment | Train setting | Decoding | LR | Freeze setting | Layer-wise decay | Beam width | test-clean WER | test-other WER | Checkpoint |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `asr_pretrained_960h_full_properval` | pretrained_asr_control | greedy |  | none | false |  | 0.186112 | 0.245802 | `facebook/wav2vec2-base-960h` |
| `asrinit_lr1e-6_fp32_properval` | asr_initialized_properval | greedy | 1e-06 | none | false |  | 0.180558 | 0.240567 | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_lr1e-6_fp32_properval/final_model` |
| `asrinit_lr3e-6_fp32_properval` | asr_initialized_properval | greedy | 3e-06 | none | false |  | 0.169678 | 0.229620 | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_lr3e-6_fp32_properval/final_model` |
| `asrinit_lr1e-5_fp32_properval` | asr_initialized_properval | greedy | 1e-05 | none | false |  | 0.154709 | 0.214737 | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_lr1e-5_fp32_properval/final_model` |
| `asrinit_lr1e-5_fp32_properval_beam` | asr_initialized_properval | beam | 1e-05 | none | false | 100 | **0.152598** | **0.212235** | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_lr1e-5_fp32_properval/final_model` |
| `asrinit_freeze_feature_lr3e-6_fp32_properval` | asr_initialized_properval | greedy | 3e-06 | feature_encoder | false |  | 0.181566 | 0.241809 | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_freeze_feature_lr3e-6_fp32_properval/final_model` |
| `asrinit_freeze3_lr3e-6_fp32_properval` | asr_initialized_properval | greedy | 3e-06 | first_3_encoder_layers | false |  | 0.176126 | 0.236612 | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_freeze3_lr3e-6_fp32_properval/final_model` |
| `asrinit_layerwise_lr_decay_properval` | asr_initialized_properval | greedy | 3e-06 | feature_extractor | 0.9 |  | 0.184742 | 0.244292 | `/home/user/disk4/LCM/assignments/SR/outputs/asrinit_layerwise_lr_decay_properval/final_model` |

Best values are shown in bold.
