# MEGA

Official monorepository for research artifacts maintained by the Ikanam AI
team. Each project keeps its paper source, runnable code, data-access
instructions, license information, and citation metadata in one place.

## Projects

| Project | Paper | Code | Data |
|---|---|---|---|
| [VALAR](projects/valar/) | *Which Values Do LLMs Confuse? A Schwartz-Based Recognition Study* ([arXiv:2607.20270](https://arxiv.org/abs/2607.20270)) | [source code](projects/valar/code/) | [OSF](https://osf.io/u56kq/overview?view_only=1c3bc242d37247de83e92113d7837be3) |
| [REGARD](projects/regard/) | *Regional Affective Differences in LLMs* ([arXiv:2607.20722](https://arxiv.org/abs/2607.20722)) | [source code](projects/regard/code/) | [OSF](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a) |

Both papers were accepted for publication in the Springer Lecture Notes in
Computer Science proceedings of AIST 2026.

## Repository layout

```text
projects/
  valar/
    paper/   LaTeX source for the current paper version
    code/    benchmark construction and evaluation code
  regard/
    paper/   LaTeX source for the current paper version
    code/    generation, scoring, and analysis code
```

Large row-level research data are distributed through the project-specific OSF
records rather than duplicated in Git. See each project's `DATA_LICENSE.md`
before redistributing generated outputs or source-derived text.

## Licenses

Code licensing is defined separately inside each project's `code/` directory.
Paper source, third-party material, model outputs, and derived datasets are not
automatically covered by those code licenses.
