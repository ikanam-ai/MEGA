# VALAR experiment code

Minimal, complete runner for the experiments in **Which Values Do LLMs
Confuse? A Schwartz-Based Recognition Study** (AIST 2026, LNCS).

The repository includes the released 1,000-item Russian gold bank, the
100-item ancillary L0 bank, exact prompts, response parsing, scoring, retry,
and resume logic. Complete model outputs, human annotations, and statistical
artifacts are archived on [OSF](https://osf.io/u56kq/overview?view_only=1c3bc242d37247de83e92113d7837be3).

## Install

Python 3.11 or 3.12 is required.

```bash
poetry install --with dev
cp .env.example .env
```

Alternatively: `python -m pip install httpx pytest ruff`.

## Run

The runner accepts any OpenAI-compatible chat-completions endpoint:

```bash
poetry run python run_experiment.py \
  --model MODEL_NAME \
  --api-base http://127.0.0.1:8000/v1 \
  --smoke
```

Remove `--smoke` for all 1,000 gold items and 100 L0 items. The protocol uses
temperature 0 and a ranked `top1`/`top2`/`top3` JSON response. Results are
written under `results/<model>_<UTC timestamp>/`. To continue an interrupted
run, pass that directory with `--run-dir`.

Useful endpoint options:

- `--api-key` or `VALAR_API_KEY` for an authenticated endpoint;
- `--thinking never` for servers that reject `chat_template_kwargs`;
- `--no-system-role` for chat templates without a system role;
- `--parallelism N` to control concurrent requests.

## Verify

```bash
poetry run pytest -q
poetry run ruff check .
poetry run python run_experiment.py --help
```

## Layout

```text
run_experiment.py   generation, parsing, scoring, retry, and resume
data/               the two evaluation banks and their manifests
tests/              offline unit and end-to-end tests
```

The paper's candidate-construction intermediates are intentionally not
duplicated here: they are archival provenance rather than inputs to the final
evaluation. They remain available through OSF and Git history.

Source code is MIT-licensed. Dataset and model-output terms are documented in
`DATA_LICENSE.md`; citation metadata are in [`../CITATION.cff`](../CITATION.cff).
