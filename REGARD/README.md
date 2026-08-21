# REGARD

Artifacts for **Regional Affective Differences in LLMs**, accepted to AIST
2026 for publication in Springer LNCS.

- [Generation, scoring, analysis, and annotation code](code/)
- [Data artifacts on OSF](https://osf.io/dwcr6/overview?view_only=0e731877e6c64892b8fca563278e631a)
- [arXiv:2607.20722](https://arxiv.org/abs/2607.20722)
- [Citation metadata](CITATION.cff)

The current paper analyses 28,500 responses from 19 generator models. All 19
models are evaluated with the primary Qwen3.6-35B-A3B VAD judge. The original
eight-model subset is additionally evaluated with GPT-4o-mini and contains the
300-item human-validation sample. The full-panel clustering therefore uses the
primary Qwen judge, while cross-judge and human comparisons are explicitly
subset-based.

The linked OSF record currently contains the original eight-model release. The
19-model generation and primary-score extension should be added as a new OSF
version before the record is described as the complete current-paper dataset.

## Study overview

![REGARD study overview](assets/study-overview.png)

## Citation

Until the final LNCS bibliographic record is available, cite:

```bibtex
@article{chetvergov2026regard,
  title   = {Regional Affective Differences in LLMs},
  author  = {Chetvergov, Andrei and Evseev, Alexander and Solovev, Mikhail and Sivoraksha, Timofei and Ukolov, Stepan and Kuschenko, Valeriia and Chistyakova, Maria and Bolovtsov, Sergey},
  journal = {arXiv preprint arXiv:2607.20722},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.20722}
}
```

## Licenses

Code is released under the [MIT License](code/LICENSE). Original author-created
figures are released under the repository's [CC BY 4.0
notice](../LICENSE-CONTENT.md). Dataset-specific terms are described in
[`code/DATA_LICENSE.md`](code/DATA_LICENSE.md) and on OSF.
