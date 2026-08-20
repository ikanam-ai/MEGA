PYTHON ?= python

.PHONY: check test

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q
	$(PYTHON) run_experiment.py --help >/dev/null
