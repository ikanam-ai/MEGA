# Annotation Contract

## Annotator

`Value4AI/ValueLlama-3-8B` — the sole annotator model. Returns continuous Schwartz-10 scores for each input text.

## Input contract

Each item sent to ValueLlama must have:
- `item_id` — unique string identifier
- `text` — the Russian text to annotate
- `source_dataset` — which dataset this came from (`tape`, `sensitive_topics`)
- `source_subset` — which subset/split

## Output contract

ValueLlama returns a JSON object with keys for each Schwartz value (10 floats, typically in [0,1]).
The parser must handle:
- Malformed JSON → `parse_ok: false`, `scores: null`
- Missing value keys → filled with `null`
- Scores outside [0,1] → clamped and flagged

## Scoring

From raw scores, downstream:
1. `top1_value` = argmax of scores
2. `top3_values` = top-3 by score
3. `score_entropy` = entropy of score distribution (measures certainty)

## Parser policy

`strict_json_with_schwartz10_score_normalization` — same philosophy as STONIC H0 scorer,
adapted for continuous scores rather than closed label classification.
