<div align="center">

<img src="assets/mega-logo.svg" width="720" alt="MEGA - Make Evaluation Great Again" />

<br />

**An umbrella research repository for LLM worldview passports.**

Pre-publication scaffold - measurement contracts first - track-local releases

</div>

## Overview

MEGA (**Make Evaluation Great Again**) is a shared research umbrella for
studying large language models through complementary model-passport protocols.
The root stays intentionally small. Before publication, each track is a
lightweight placeholder with its own README only.

## Tracks

| Track | Full name | Model-passport layer |
| --- | --- | --- |
| [`VIBE`](VIBE/) | Valence-Informed Benchmark of Emotion | Affective profile of generated responses in Valence-Arousal, with optional Dominance extension. |
| [`STONIC`](STONIC/) | Schwartz-Theory-Oriented Normative Integrity Check | Value profile in the Schwartz basic values space under a fixed measurement contract. |
| [`PROOF`](PROOF/) | Profiling Reliability Of Object-level Facts | Parametric factual coverage and reliable extraction of object-level facts. |

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
```

Current track layout:

```text
TRACK/
  README.md
```

After the corresponding review gate, a track may expand into a full
paper-specific release:

```text
TRACK/
  README.md
  docs/
  configs/
  item_banks/
  scripts/
  results/
  paper/
```

## Measurement Principle

Each track starts with a neutral core profile and only then measures drift
under controlled variation of language, role, prompt format, context, time,
or inference parameters.

The shared rule is to separate the target construct from the operational
measurement protocol. A model passport is a reproducible measurement artifact,
not a claim that the model has human-like inner states.

## Release Policy

Pre-publication tracks contain templates and protocol descriptions only.
Benchmarks, executable code, item banks, model outputs, and paper artifacts
are added only after the corresponding review gate.

When a track is not yet accepted, its README must include:

> Note: The code and benchmark would be released here after review.

## Current Status

This repository is a pre-publication scaffold. It is not yet a benchmark
release and should not be treated as a source of final measurements.

## Branding

The MEGA mark and track colors are documented in [`docs/BRANDING.md`](docs/BRANDING.md).
