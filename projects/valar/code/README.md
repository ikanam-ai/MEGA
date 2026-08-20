# VALAR

Official code and benchmark-construction pipeline for **Which Values Do LLMs
Confuse? A Schwartz-Based Recognition Study**, accepted for the AIST 2026
Springer LNCS proceedings.

VALAR studies controlled top-1 recognition over Schwartz's ten basic values.
The evaluation set contains 1,000 unique Russian situational texts, balanced
across the ten labels and independently reviewed by two human annotators per
item. The paper evaluates 21 instruction-tuned runs; 20 reliable runs form the
semantic panel used for directed-confusion analysis.

## Artifacts

- Complete model-run data and human-validation artifacts:
  [OSF archive](https://osf.io/u56kq/overview?view_only=1c3bc242d37247de83e92113d7837be3)
- Preprint: [arXiv:2607.20270](https://arxiv.org/abs/2607.20270)
- Code: this repository

The repository contains the released gold bank, model-assisted construction
artifacts, prompts, and collection/evaluation scripts. The OSF archive is the
authoritative source for the complete 21-run panel, row-level predictions,
human labels, case codes, and statistical outputs used in the paper.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[evaluation,dev]"
cp .env.example .env
```

For local ValueLlama annotation, additionally install the `annotation` extra.
Never commit `.env`.

## Run Schwartz-10 evaluation

The evaluation client accepts any OpenAI-compatible endpoint. Start with a
five-item smoke run:

```bash
python scripts/eval/run_schwartz_eval.py \
  --model MODEL_NAME \
  --api-base http://127.0.0.1:8000/v1 \
  --smoke \
  --no-clearml
```

Remove `--smoke` for the complete 1,000-item Russian bank and the ancillary
100-item L0 bank. The main paper uses temperature `0`, a ranked JSON response
with `top1`, `top2`, and `top3`, and top-1 as the primary decision.

## Rebuild the evaluation bank

The construction flow is:

1. pull the documented TAPE and inappropriate/sensitive-topic source slices;
2. convert them to a common item-bank schema;
3. annotate candidates with ValueLlama and an independent GPT classifier;
4. retain exact model agreement, balance 100 items per value, and attach the
   independent human labels distributed through OSF.

Commands and schemas are documented in [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md),
[`docs/RUNBOOK.md`](docs/RUNBOOK.md), and
[`docs/ANNOTATION_CONTRACT.md`](docs/ANNOTATION_CONTRACT.md).

## Tests

```bash
pytest -q
python -m compileall -q valar scripts
```

## Repository layout

```text
configs/             model, prompt, experiment, and run configurations
data/item_banks/     released gold bank and construction artifacts
docs/                data contracts and runbooks
scripts/annotate/    model-assisted annotation clients
scripts/data/        source-to-item-bank builders
scripts/eval/        ranked Schwartz-10 evaluation client
tests/               parsing and value-space unit tests
valar/               reusable Python package
```

## License and citation

Source code is released under the MIT License. The license does not
automatically extend to source datasets, derived text collections, or model
outputs; see [`DATA_LICENSE.md`](DATA_LICENSE.md) and the OSF record. Citation
metadata are provided in [`CITATION.cff`](CITATION.cff).
