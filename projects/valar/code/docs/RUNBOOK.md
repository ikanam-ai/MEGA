# VALAR Runbook

## Prerequisites

1. Python 3.12, Poetry installed.
2. ValueLlama-3-8B served locally via vLLM on port 8000 (OpenAI-compatible).
3. Raw datasets in `../datasets/` (populated by `pull_raw_datasets.py`).

## Setup

```bash
cd VALAR
cp .env.example .env        # set VALAR_VALUELLAMA_API_BASE_URL
poetry install --with annotation
```

## Step 1 — Build item banks from raw datasets

```bash
python scripts/data/build_annotation_banks.py \
  --experiment-config configs/experiments/valar_tape_annotation.yaml \
  --output-dir data/item_banks/valar/
```

## Step 2 — Smoke test (20 items)

```bash
make annotate-smoke DATASET=tape
```

## Step 3 — Full annotation run

```bash
python scripts/annotate/run_valuellama_annotation.py \
  --experiment-config configs/experiments/valar_tape_annotation.yaml \
  --run-config configs/runs/annotation_run.yaml \
  --output-dir results/tape_annotation_v1
```

## Step 4 — Analyse results

```bash
python scripts/analyze/analyze_annotation_results.py \
  --results-dir results/tape_annotation_v1
```

## Outputs

Each annotation run produces under `results/<run_id>/`:
```
generation/
  annotation_rows.jsonl      <- raw ValueLlama outputs per item
  annotation_summary.json
scoring/
  value_scores.csv           <- parsed scores per item x value
  top1_value_per_item.csv
  value_distribution.csv
analysis/
  value_distribution_by_subset.csv
  value_shift_table.csv
```
