# Data layout

The repository tracks only `targets/aist_cis_targets_final.csv`. Download the
row-level artifacts from the
[OSF archive](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a)
and preserve this layout when running the publication analysis:

```text
data/
  targets/aist_cis_targets_final.csv
  generations/generations_raw.jsonl
  scores/judge_scores_raw.jsonl
  processed/merged_dataset.parquet
vad_annotation_package/
  data/annotation_items.jsonl
  data/assignments.csv
  annotations/annXX/*.json
```

The analysis uses the 900 assignments listed in `assignments.csv` (three
ratings for each of 300 items). It ignores annotation files outside that set.
Do not publish `users.csv`; it is an operational credential file, not research
data.
