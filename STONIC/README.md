# STONIC - Schwartz-Theory-Oriented Normative Integrity Check

Note: The code and benchmark would be released here after review.

## Status

Pre-publication placeholder. This track intentionally contains only this
README until the review gate.

## Scope

STONIC studies the context-conditioned but reproducible value profile of LLMs
in the Schwartz basic values space.

## Object of Study

The object is a model-level value profile: a vector over the 10 Schwartz
basic values, estimated from item-based responses under a fixed measurement
contract.

## Measurement Shape

- neutral core protocol first
- item-based measurement inspired by the PVQ line
- explicit construct vs operationalization separation
- drift layer for language, format, role, context, and inference parameters
- coherence and stability checks across protocol factors

## Passport Output

The intended output is a value passport: a vector over Schwartz values for
each model, plus reliability, coherence, and drift summaries.

## Literature Anchors

- Schwartz, 2012 - An Overview of the Schwartz Theory of Basic Values
- Schwartz et al., 2001 - Extending the Cross-Cultural Validity
- Jacobs and Wallach, 2021 - Measurement and Fairness
- Han et al., 2025 - Value Portrait
- Yao et al., 2024 - Value FULCRA
- Kovac et al., 2024 - Stick to your role!
- Rozen et al., 2025 - Do LLMs have Consistent Values?

## Future Release Layout

```text
STONIC/
  README.md
  docs/
  configs/
  item_banks/
  scripts/
  results/
  paper/
```

The directories above are not included in the pre-publication scaffold.
