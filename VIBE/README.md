# VIBE - Valence-Informed Benchmark of Emotion

Note: The code and benchmark would be released here after review.

## Status

Pre-publication placeholder. This track intentionally contains only this
README until the review gate.

## Scope

VIBE studies the context-conditioned but reproducible affective profile of
LLM responses. The working measurement space is Valence-Arousal, with an
optional Dominance extension.

The project does not claim that models experience emotions. It measures
affective properties of generated text under a fixed measurement contract.

## Object of Study

The basic unit is one model response to one standardized situational prompt.
The model-level affective profile is an aggregated VA distribution over a
neutral core item bank.

## Measurement Shape

- neutral core protocol first
- drift layer after the core profile is fixed
- drift factors: language, format, role, context, and inference parameters
- item-based prompts rather than ad hoc examples
- external or explicitly specified VA/VAD scoring

## Passport Output

The intended output is an affective passport: a reproducible VA distribution
for each model, plus drift summaries across controlled protocol factors.

## Literature Anchors

- Buechel and Hahn, 2017 - EmoBank
- Mendes and Martins, 2023 - Quantifying Valence and Arousal
- Cho et al., 2025 - Language-based Valence and Arousal
- Ishikawa and Yoshino, 2025 - AI with Emotions

## Future Release Layout

```text
VIBE/
  README.md
  docs/
  configs/
  item_banks/
  scripts/
  results/
```

The directories above are not included in the pre-publication scaffold.
