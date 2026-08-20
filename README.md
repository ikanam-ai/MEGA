<div align="center">

<img src="assets/mega-logo.svg" width="720" alt="MEGA - Make Evaluation Great Again" />

<br />

**An umbrella research repository for LLM worldview passports.**

One repository, multiple self-contained research projects

</div>

## Overview

MEGA (**Make Evaluation Great Again**) is a shared research umbrella for
studying large language models through complementary model-passport protocols.
Every MEGA project has its own top-level directory containing its paper,
executable code, data-access instructions, licenses, and citation metadata.

## Tracks

| Track | Full name | Model-passport layer |
| --- | --- | --- |
| [`VIBE`](VIBE/) | Valence-Informed Benchmark of Emotion | Affective profile of generated responses in Valence-Arousal, with optional Dominance extension. |
| [`STONIC`](STONIC/) | Schwartz-Theory-Oriented Normative Integrity Check | Value profile in the Schwartz basic values space under a fixed measurement contract. |
| [`PROOF`](PROOF/) | Profiling Reliability Of Object-level Facts | Parametric factual coverage and reliable extraction of object-level facts. |

## Published Projects

| Project | Paper | Code | Data |
| --- | --- | --- | --- |
| [`VALAR`](VALAR/) | *Which Values Do LLMs Confuse? A Schwartz-Based Recognition Study* ([arXiv:2607.20270](https://arxiv.org/abs/2607.20270)) | [source code](VALAR/code/) | [OSF](https://osf.io/u56kq/overview?view_only=1c3bc242d37247de83e92113d7837be3) |
| [`REGARD`](REGARD/) | *Regional Affective Differences in LLMs* ([arXiv:2607.20722](https://arxiv.org/abs/2607.20722)) | [source code](REGARD/code/) | [OSF](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a) |

Both papers were accepted for publication in the Springer Lecture Notes in
Computer Science proceedings of AIST 2026.

## Layout

```text
MEGA/
  README.md
  LICENSE
  .gitignore
  assets/
  docs/
  VIBE/
  STONIC/
  PROOF/
  VALAR/
    README.md
    paper/
    code/
  REGARD/
    README.md
    paper/
    code/
```

## Measurement Principle

Each track starts with a neutral core profile and only then measures drift
under controlled variation of language, role, prompt format, context, time,
or inference parameters.

The shared rule is to separate the target construct from the operational
measurement protocol. A model passport is a reproducible measurement artifact,
not a claim that the model has human-like inner states.

## Project Policy

Pre-publication projects contain templates and protocol descriptions.
Benchmarks, executable code, item banks, model outputs, and paper artifacts are
added to the corresponding project directory after the review gate.

When a track is not yet accepted, its README must include:

> Note: The code and benchmark would be released here after review.

## Data distribution

Large row-level research data are distributed through the project-specific OSF
records rather than duplicated in Git. See each project's `DATA_LICENSE.md`
before redistributing generated outputs or source-derived text.

## Licenses

The root license applies to shared MEGA materials. Code licensing is also
stated inside each published project's `code/` directory. Paper source, third-party
material, model outputs, and derived datasets are not automatically covered by
the code licenses.

## Branding

The MEGA mark and track colors are documented in [`docs/BRANDING.md`](docs/BRANDING.md).
