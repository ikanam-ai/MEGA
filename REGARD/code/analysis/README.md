# Publication analysis

`reproduce_publication_stats.py` validates the fixed 19-model panel and
rebuilds the current paper's primary Qwen-judge results:

- per-model VAD and quality profiles;
- Ward-linkage clustering on mean V, A, D, and generic-answer rate;
- cross-model correlations and bootstrap intervals;
- category- and country-level cluster summaries;
- paired Qwen/GPT agreement on the available original eight-model subset;
- the five full-panel publication figures.

```bash
python reproduce_publication_stats.py \
  --data-root /path/to/REGARD-data \
  --output-dir output
```

Human-rating inputs are not required for this full-panel runner. They and the
interface used to collect them are archived separately through OSF.
