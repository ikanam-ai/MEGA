# Publication analysis

`reproduce_publication_stats.py` rebuilds the statistics and figures used in the manuscript. It distinguishes:

- target-bank uncertainty, conditional on the eight evaluated systems;
- exact model-label permutation inference over the eight model means;
- prompt-specific and pooled contrasts;
- complete-case and quality-flag sensitivity analyses;
- judge-specific effects;
- item-level human agreement and the human group-contrast check.
- human-minus-judge residual comparisons by model group, with Holm correction;
- retained response counts for every missingness and quality-filter sensitivity.

Example:

```bash
python reproduce_publication_stats.py \
  --data-root /path/to/REGARD-data \
  --output-dir output \
  --copy-figures-to ../paper
```

The cleaned human analysis uses only the 900 assignments listed in `assignments.csv`: three ratings for each of 300 items. Files outside that assignment set are ignored.
