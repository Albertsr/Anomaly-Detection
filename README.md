# Anomaly Detection

[English](README.md) | [简体中文](README.zh-CN.md)

An educational collection of anomaly-detection implementations, paper notes, and reproducible experiments originally written and maintained by MaXiao.

This repository is being revived incrementally in 2026. The goal is to improve documentation, reproducibility, and compatibility without replacing the original handwritten work with a wholesale rewrite. Historical scripts and explanations remain available while tested interfaces are introduced in small, reviewable steps.

> **Maintenance status:** revival in progress. Most current scripts are historical learning or research implementations, not a production-ready library. See the [roadmap](ROADMAP.md) before relying on an API.

## What is included

The current repository focuses mainly on point anomalies in tabular data. It covers several complementary ideas:

- isolation-based detection;
- global and local distance or density methods;
- PCA and Kernel PCA reconstruction error;
- principal-component deviation methods;
- learning with partially observed anomalies;
- positive-unlabeled learning;
- KADOA, an experimental variation proposed by the repository author.

Time-series, image, graph, and streaming anomaly detection are outside the first revival scope.

## Algorithm map

| Learning setting | Method | Current form | Status |
| --- | --- | --- | --- |
| Unsupervised | [Isolation Forest](UnSupervised-Isolation%20Forest/ReadMe.md) | Paper notes and a scikit-learn example | Legacy |
| Unsupervised | [PCA / Kernel PCA reconstruction error](UnSupervised-Based%20on%20PCA/ReadMe.md) | Handwritten implementations and experiments | Legacy |
| Unsupervised | [RobustPCC](UnSupervised-Based%20on%20PCA/ReadMe.md) | Paper-inspired implementation | Experimental |
| Unsupervised | [Mahalanobis distance](UnSupervised-Mahalanobis%20Distance/ReadMe.md) | Handwritten implementation and equivalence experiment | Legacy |
| Unsupervised | [Local Outlier Factor](UnSupervised-Local%20Outlier%20Factor/ReadMe.md) | Paper notes and a scikit-learn example | Legacy |
| Semi-supervised | [ADOA](SemiSupervised-ADOA/ReadMe.md) | Paper-inspired implementation | Experimental |
| Semi-supervised | [KADOA](SemiSupervised-KADOA-Original/ReadMe.md) | Author-proposed experimental variation | Experimental |
| Semi-supervised | [Positive-Unlabeled Learning](SemiSupervised-PU%20Learning/ReadMe.md) | Research implementations and notes | Experimental |

**Legacy** means that the material is preserved for learning and historical reference but is not yet covered by the planned compatibility and continuous-integration matrix.

**Experimental** means that the implementation or empirical conclusion is preliminary. It should not be treated as a generally validated performance claim.

## Minimal historical example

The repository does not yet provide a locked environment or an installable package. To run the Mahalanobis equivalence example from a fresh clone:

```bash
git clone https://github.com/Albertsr/Anomaly-Detection.git
cd Anomaly-Detection
python -m pip install numpy pandas scikit-learn
python "UnSupervised-Mahalanobis Distance/verify_mahal_equivalence.py"
```

Dependencies vary by module. Some semi-supervised comparison scripts also require optional libraries such as LightGBM or XGBoost. A reproducible dependency definition and tested package interface are planned for the next maintenance phase.

## How to read the historical results

The existing figures and comparisons are educational experiments, not universal rankings:

- when ground-truth labels are available, metrics such as F1 or ROC AUC can evaluate predictions under the stated experimental setup;
- when labels are unavailable, agreement with Isolation Forest is only detector agreement, not proof of correctness;
- results from a small set of synthetic datasets should not be generalized to all anomaly types or data domains;
- thresholds, contamination assumptions, random seeds, and data construction can materially change the outcome.

The benchmark revival will make these boundaries explicit and add public labeled datasets, repeated seeds, runtime measurements, and uncertainty reporting.

## About KADOA

KADOA is presented in this repository as an original experimental idea by MaXiao. It adapts the ADOA workflow by replacing its Isolation Forest component with Kernel PCA reconstruction error while retaining the broader clustering-and-weighting idea.

KADOA has not been described here as a peer-reviewed standalone method. The published repository experiment is preliminary and does not establish universal superiority over ADOA. If you use or discuss KADOA, please describe it as an experimental repository method, acknowledge its relationship to ADOA, and cite this repository using [`CITATION.cff`](CITATION.cff).

## Revival principles

- Preserve the original Git history, explanations, and recognizable implementation style.
- Add characterization tests before changing algorithm behavior.
- Keep historical paths available while a stable package API is introduced.
- Separate documentation, compatibility fixes, API extraction, benchmarks, and new algorithms into reviewable changes.
- Prefer scoped empirical claims over broad statements of superiority.
- Add a small number of representative methods with explanations and tests, rather than maximizing the algorithm count.

The first planned additions are One-Class SVM and ECOD. Autoencoder, Deep SVDD, and Deep SAD are later candidates after the classical benchmark is stable.

## Contributing and questions

Bug reports, reproducibility notes, documentation corrections, and carefully scoped algorithm contributions are welcome through [GitHub Issues](https://github.com/Albertsr/Anomaly-Detection/issues).

Please include the Python and dependency versions, the script or method used, a minimal input example, and the observed traceback or result.

## Citation

Use the repository citation metadata in [`CITATION.cff`](CITATION.cff). A DOI, release version, and publication reference have not been claimed.

## License

This project is licensed under the [MIT License](LICENSE).
