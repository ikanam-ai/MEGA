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
    README.md
  STONIC/
    README.md
  PROOF/
    README.md
```

## Measurement Principle

Each track starts with a neutral core profile and measures drift only after
the core protocol is fixed.

## Release Policy

Pre-publication tracks contain placeholders only. Benchmarks, executable code,
item banks, model outputs, and paper artifacts are added only after review.

When a track is not yet accepted, its README must include:

> Note: The code and benchmark would be released here after review.

## Development

- `main` stores the public repository state.
- `develop` is the integration branch.
- Changes should be made in short-lived branches and merged through pull requests.

## Current Status

This repository is a pre-publication scaffold. It is not yet a benchmark
release and should not be treated as a source of final measurements.

## Branding

The MEGA mark and track colors are documented in [`docs/BRANDING.md`](docs/BRANDING.md).
