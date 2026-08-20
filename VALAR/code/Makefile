PYTHON ?= python
DATASET ?= tape
SMOKE_LIMIT ?= 20
RESULTS_ROOT ?= results

.PHONY: compile test annotate-smoke pull-datasets clean

compile:
	$(PYTHON) -m compileall -q valar scripts

test:
	pytest -q

pull-datasets:
	$(PYTHON) scripts/data/pull_raw_datasets.py

annotate-smoke:
	$(PYTHON) scripts/annotate/run_valuellama_annotation.py \
	  --dataset $(DATASET) \
	  --run-config configs/runs/annotation_run.yaml \
	  --limit-items $(SMOKE_LIMIT) \
	  --output-dir $(RESULTS_ROOT)/_smoke_$(DATASET) \
	  --dry-run

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache
