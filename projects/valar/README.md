# VALAR

Artifacts for **Which Values Do LLMs Confuse? A Schwartz-Based Recognition
Study**, accepted to AIST 2026 for publication in Springer LNCS.

- [Current LaTeX paper source](paper/)
- [Code and released benchmark materials](code/)
- [Complete data and human-validation artifacts on OSF](https://osf.io/u56kq/overview?view_only=1c3bc242d37247de83e92113d7837be3)
- [arXiv:2607.20270](https://arxiv.org/abs/2607.20270)

The paper evaluates 21 instruction-tuned runs on 1,000 Russian situational
texts labelled with Schwartz's ten basic values. The semantic analysis uses 20
runs with reliable ranked outputs.

Build the paper from `paper/` with two direct `pdflatex` passes; the supplied
`main.bbl` makes a separate BibTeX run unnecessary.
