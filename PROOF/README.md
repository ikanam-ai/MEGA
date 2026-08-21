# PROOF - Profiling Reliability Of Object-level Facts

Note: The code and benchmark would be released here after review.

## Status

Pre-publication placeholder. This track intentionally contains only this
README until the review gate.

## Scope

PROOF studies large language models as carriers of parametric factual
knowledge. The core question is which object-level facts can be extracted
from a model without external sources, and how reliable that extraction is.

## Object of Study

The object is parametric factual coverage of LLMs: the ability to recover
atomic facts from a fixed world of truth under a specified extraction
protocol.

## Measurement Shape

- fixed truth world
- atomic fact extraction protocol
- prompt variation as a measurement factor
- temporal correctness and cut-off sensitivity
- unknown-boundary behavior
- cross-lingual consistency checks

## Passport Output

The intended output is a factual passport: coverage, temporal reliability,
prompt robustness, language consistency, and unknown-boundary behavior for
each model.

## Literature Anchors

- Park et al., 2025 - CHROKNOWLEDGE
- Polo et al., 2024 - Efficient multi-prompt evaluation of LLMs
- Qi et al., 2023 - Cross-Lingual Consistency of Factual Knowledge

## Future Release Layout

```text
PROOF/
  README.md
  docs/
  configs/
  item_banks/
  scripts/
  results/
```

The directories above are not included in the pre-publication scaffold.
