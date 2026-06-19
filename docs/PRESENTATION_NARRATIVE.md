# XAI509 ASR Final Presentation Narrative

## Presentation framing

The presentation should distinguish two outcomes:

- **Main reproducible single-model pipeline:** Staged CTC Fine-tuning followed
  by beam decoding with the train-text trigram language model.
- **Best observed ensemble extension:** fold-specific Staged CTC models, each
  decoded with the selected beam/trigram-LM configuration, followed by ROVER
  word-level voting.

“Best observed” and “main reproducible” should not be conflated. The
single-model result is the clean validation-selected project result. The ROVER
extension has lower observed test WER, but its validation selection is affected
by fold-membership leakage and is therefore exploratory.

## Nine-slide outline

### 1. Title

- XAI509 Automatic Speech Recognition Semester Project
- Wav2Vec2-CTC Fine-tuning on the Provided LibriSpeech WebDataset
- Evaluation on `test-clean` and `test-other`

### 2. Roadmap

- Task and data protocol
- Fine-tuning challenge
- Search-space coverage
- Selected single-model pipeline
- Ensemble extension
- Results and future work

### 3. Task and experimental setup

- Initialize from `facebook/wav2vec2-base`
- Fine-tune with CTC on the five provided train shards
- Use train shard `000004` for main checkpoint and decoder selection
- Evaluate WER on `test-clean` and `test-other`

### 4. Fine-tuning challenge: newly initialized CTC head

- The project tokenizer requires a newly initialized CTC classification layer
- Early lower-LR runs produced blank-dominant hypotheses and WER around 1.0
- Diagnostics isolated optimization and augmentation stability issues
- Weak SpecAugment and staged optimization restored stable learning

### 5. Fine-tuning search-space coverage

- Learning rate: lower-LR variants and separated encoder/head rates
- Freezing: full tuning, frozen feature extractor, frozen encoder for warmup
- SpecAugment: default, off for diagnosis, and weak augmentation
- Schedule: one-stage, staged warmup, and refinement variants
- Decoding: greedy, beam, and beam + train-text trigram LM
- Coverage: 5 folds, 2 paired acoustic recipes, and 3 decoder settings

### 6. Selected single-model pipeline: Staged CTC + LM fusion

- Stage 1: train the CTC head with the Wav2Vec2 encoder frozen
- Stage 2: fine-tune the encoder with the feature extractor frozen
- Select the main acoustic checkpoint using the reserved validation shard
- Decode with beam width 50, alpha 0.3, beta 1.5, and the train-text trigram LM
- Main WER: 0.246386 on `test-clean`, 0.329289 on `test-other`

### 7. Ensemble extension: ROVER over LM-fused fold hypotheses

- Decode each of the five fold-specific acoustic models with the selected
  beam/trigram-LM configuration
- Apply ROVER only after decoding
- Align the five word hypotheses and use deterministic voting
- ROVER complements LM fusion; it does not replace it

### 8. Results: main reproducible vs best observed ensemble

| Setting | Role | Validation WER | test-clean WER | test-other WER |
| --- | --- | ---: | ---: | ---: |
| Staged CTC fine-tuned Wav2Vec2 + beam/trigram LM | Main reproducible result | 0.228487 | 0.246386 | 0.329289 |
| Staged CTC All-Train Model | Exploratory all-train | — | 0.216867 | 0.303307 |
| Staged CTC Fold Ensemble with ROVER Voting | Best observed ensemble extension | — | 0.197314 | 0.271383 |

- The ensemble has the lowest observed WER
- It remains exploratory because ensemble selection is affected by
  fold-membership leakage

### 9. Conclusion and future work beyond scope

- Staged optimization stabilized a newly initialized CTC head
- Weak SpecAugment and separated encoder/head training improved robustness
- Train-text LM fusion improved each model’s decoded hypothesis
- ROVER produced an additional post-decoding ensemble gain
- Future work: stronger LMs, cleaner ensemble validation, larger ASR model
  families, and ensemble-diversity analysis

## Core presentation message

> We stabilized Wav2Vec2-CTC fine-tuning by searching over LR, freezing,
> SpecAugment, and training schedules, selected Staged CTC with beam/trigram-LM
> decoding, and obtained the best observed WER by applying ROVER voting over
> fold-specific LM-fused transcripts.
