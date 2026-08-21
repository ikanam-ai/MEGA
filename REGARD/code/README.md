# REGARD experiment code

Minimal, complete pipeline for **Regional Affective Differences in Large
Language Models** (AIST 2026, LNCS): 19 generators, 500 targets, three Russian
prompts, Qwen primary judging, the original eight-model GPT check, Ward
clustering, result tables, figures, and the human-rating interface.

Row-level generations, judge scores, and human annotations are distributed
through the [OSF record](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a).
The preprint is [arXiv:2607.20722](https://arxiv.org/abs/2607.20722).

## Install

Python 3.11 or 3.12 is required.

```bash
poetry install --with analysis,dev
cp .env.example .env
```

Set `VLLM_ENDPOINT` to an OpenAI-compatible server. `OPENAI_API_KEY` is
required only for the GPT-4o-mini subset check.

## Generate the 28,500 responses

Each command is resumable and appends only missing model-target-prompt rows:

```bash
poetry run python -m scripts.stage02_generate
poetry run python -m scripts.stage02_generate --prompt-id neutral_descriptive
poetry run python -m scripts.stage02_generate --prompt-id evaluative_paraphrase
```

The fixed protocol is temperature `0.7`, no system prompt, and at most 512
output tokens. Use `--models model_a,model_b` for a partial run or `--workers N`
to control concurrency.

## Score

Qwen3.6-35B-A3B is the primary judge for all 19 models and is the safe default:

```bash
poetry run python -m scripts.stage03_score
```

GPT-4o-mini was run only on the original eight-model subset:

```bash
poetry run python -m scripts.stage03_score \
  --judges gpt4o_mini \
  --models yandexgpt,gigachat,tpro,avibe,qwen25_14b,glm,gemma2_27b,ministral_14b
```

Both runners resume from their JSONL outputs by stable identifiers.

## Reproduce the 19-model analysis

Download the OSF artifacts in the layout described in `data/README.md`, then:

```bash
poetry run python analysis/reproduce_publication_stats.py \
  --data-root /path/to/REGARD-data \
  --output-dir analysis/output
```

The script uses only Qwen scores with `target_coverage >= 0.5` for the primary
profile and clustering. GPT scores enter only the paired judge-agreement table.
It verifies the exact 19-model panel before writing tables and figures.

## Human annotation and verification

The retained Streamlit interface is documented in
`vad_annotation_package/README.md`. To test the code:

```bash
poetry run pytest -q
poetry run ruff check .
poetry run python -m scripts.stage02_generate --help
poetry run python -m scripts.stage03_score --help
```

## Layout

```text
analysis/                 current 19-model publication analysis
config/                   exact 19-model registry and prompt/judge contracts
data/targets/             REGARD-500 target bank
scripts/stage02_generate  resumable generation
scripts/stage03_score     resumable primary/subset judging
vad_annotation_package/  human-rating interface
tests/                    offline tests
```

Source code is MIT-licensed. See `DATA_LICENSE.md` for data and model-output
terms and [`../CITATION.cff`](../CITATION.cff) for citation metadata.
