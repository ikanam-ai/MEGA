# Data layout

The repository tracks the 500-target bank. Download the row-level artifacts
from [OSF](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a)
and keep either of these equivalent layouts:

```text
REGARD-data/                       REGARD-19model-extension/
  data/generations/...              generations/...
  data/scores/...                   scores/...
  data/targets/...                  targets/...
```

Required filenames are `generations_raw.jsonl`, `judge_scores_raw.jsonl`, and
`aist_cis_targets_final.csv`. The analysis checks for exactly 19 generator
IDs and complete primary-judge model coverage before producing results.

For human annotation, place `annotation_items.jsonl` and `assignments.csv` in
`vad_annotation_package/data/`. Never publish `users.csv`; it contains
operational credentials rather than research data.
