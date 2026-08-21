# VALAR

Artifacts for **Which Values Do LLMs Confuse? A Schwartz-Based Recognition
Study**, accepted to AIST 2026 for publication in Springer LNCS.

- [Code and released benchmark materials](code/)
- [Complete data and human-validation artifacts on OSF](https://osf.io/u56kq/overview?view_only=1c3bc242d37247de83e92113d7837be3)
- [arXiv:2607.20270](https://arxiv.org/abs/2607.20270)
- [Citation metadata](CITATION.cff)

The paper evaluates 21 instruction-tuned runs on 1,000 Russian situational
texts labelled with Schwartz's ten basic values. The semantic analysis uses 20
runs with reliable ranked outputs.

## Study overview

![VALAR task and evaluation overview](assets/task-overview.png)

## Dataset construction

![VALAR dataset construction pipeline](assets/dataset-construction.png)

## Citation

Until the final LNCS bibliographic record is available, cite:

```bibtex
@article{chetvergov2026values,
  title   = {Which Values Do LLMs Confuse? A Schwartz-Based Recognition Study},
  author  = {Andrei Chetvergov and Stepan Ukolov and Timofei Sivoraksha and Alexander Evseev and Mikhail Solovev and Valeriia Kuschenko and Maria Chistyakova and Sergey Bolovtsov},
  journal = {arXiv preprint arXiv:2607.20270},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.20270}
}
```

## Licenses

Code is released under the [MIT License](code/LICENSE). Original author-created
figures are released under the repository's [CC BY 4.0
notice](../LICENSE-CONTENT.md). Dataset-specific terms are described in
[`code/DATA_LICENSE.md`](code/DATA_LICENSE.md) and on OSF.
