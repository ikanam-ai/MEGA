# REGARD

Official code for **Regional Affective Differences in LLMs**, accepted for the
AIST 2026 Springer LNCS proceedings.

The current REGARD paper compares Russian-language affective framing by 19
instruction-tuned LLMs on 500 post-Soviet targets. All 19 models are scored on
continuous valence, arousal, and dominance (VAD) dimensions by the primary
Qwen3.6-35B-A3B judge. The original eight-model subset is additionally scored
by GPT-4o-mini and checked against a human-rated subset.

This directory contains the general collection and scoring pipeline and the
exact original eight-model analysis. The 19-model paper source and scope notes
are in the sibling `paper/` directory and the project README. Full-panel clustering is conditional on
the primary Qwen judge; cross-judge and human comparisons are subset-based.

## Artifacts

- Data, generated responses, judge scores, and human annotations (the OSF
  record currently exposes the original eight-model release; the 19-model
  extension is to be added as a new version):
  [OSF archive](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a)
- Preprint: [arXiv:2607.20722](https://arxiv.org/abs/2607.20722)
- Code: this repository

The repository intentionally excludes credentials and the large row-level data
files stored on OSF. It includes the 500-target bank, generation and judging
pipeline, exact camera-ready analysis, machine-readable result tables, and the
human-annotation interface.

## Installation

Python 3.11 is required. With Poetry:

```bash
poetry install --with analysis,dev
cp .env.example .env
```

Add API credentials and OpenAI-compatible endpoints to `.env`. Never commit
that file.

## Reproduce the original eight-model analysis

Download the OSF archive and arrange it as described in
[`data/README.md`](data/README.md), then run:

```bash
poetry run python analysis/reproduce_publication_stats.py \
  --data-root /path/to/REGARD-data \
  --output-dir analysis/output
```

This analysis uses the fixed random seed `20260713` and reports
target-level bootstrap uncertainty separately from exact model-label
permutation inference.

## Run the collection pipeline

Generate one response for every selected model, target, and prompt:

```bash
poetry run python -m scripts.stage02_generate \
  --models yandexgpt,gigachat,tpro,avibe,glm,gemma2_27b,qwen25_14b,ministral_14b

poetry run python -m scripts.stage02_generate --prompt-id neutral_descriptive \
  --models yandexgpt,gigachat,tpro,avibe,glm,gemma2_27b,qwen25_14b,ministral_14b

poetry run python -m scripts.stage02_generate --prompt-id evaluative_paraphrase \
  --models yandexgpt,gigachat,tpro,avibe,glm,gemma2_27b,qwen25_14b,ministral_14b
```

Score the generations with both judges and prepare the merged tables:

```bash
poetry run python -m scripts.stage03_score --judges qwen36_35b,gpt4o_mini
poetry run python -m scripts.stage05_merge_and_analyze
```

Stages 2 and 3 append JSONL records and skip completed identifiers, so an
interrupted run can be resumed.

## Add a new model

1. Add a generator entry under `generators` in `config/models.yaml`. Give it a
   stable `model_id`, exact served checkpoint name, group label, endpoint,
   message format, authentication rule, and response paths.
2. Add any required environment variable to `.env.example`, leaving its value
   empty.
3. Run `stage02_generate` with `--models <model_id>` once for each of the three
   prompt IDs. Keep temperature `0.7` and `max_tokens=400` to match the paper.
4. Run `stage03_score` for both judges. Keep judge temperature `0.0` and the
   scoring contract unchanged.
5. Download or combine the existing OSF artifacts with the new rows, run the
   publication analysis, and inspect per-model coverage and quality flags
   before making a group comparison.

Adding a checkpoint expands the observed panel; it does not by itself create a
representative regional sample. Treat model-family, capability, language
proficiency, and deployment-policy differences as possible confounders.

## Tests

```bash
poetry run pytest -q
```

## Repository layout

```text
analysis/                 exact publication analysis and result tables
config/                   model registry and versioned prompt contracts
data/targets/             curated REGARD-500 target bank
scripts/                  generation, judging, merge, and provider code
tests/                    HTTP-client unit tests
vad_annotation_package/  human-rating interface and preparation scripts
```

## License and citation

Source code is released under the MIT License. The license does not
automatically extend to generated model outputs, third-party metadata, or other
data artifacts; see [`DATA_LICENSE.md`](DATA_LICENSE.md) and the OSF record.
Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
