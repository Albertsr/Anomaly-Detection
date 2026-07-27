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
| Unsupervised | [PCA reconstruction error](UnSupervised-Based%20on%20PCA/ReadMe.md) | Alpha API plus handwritten source | Alpha API |
| Unsupervised | [Kernel PCA reconstruction error](UnSupervised-Based%20on%20PCA/ReadMe.md) | Handwritten implementation and experiments | Legacy |
| Unsupervised | [RobustPCC](UnSupervised-Based%20on%20PCA/ReadMe.md) | Paper-inspired implementation | Experimental |
| Unsupervised | [Mahalanobis distance](UnSupervised-Mahalanobis%20Distance/ReadMe.md) | Alpha API plus handwritten source | Alpha API |
| Unsupervised | [Local Outlier Factor](UnSupervised-Local%20Outlier%20Factor/ReadMe.md) | Paper notes and a scikit-learn example | Legacy |
| Semi-supervised | [ADOA](SemiSupervised-ADOA/ReadMe.md) | Paper-inspired implementation | Experimental |
| Semi-supervised | [KADOA](SemiSupervised-KADOA-Original/ReadMe.md) | Author-proposed experimental variation | Experimental |
| Semi-supervised | [Positive-Unlabeled Learning](SemiSupervised-PU%20Learning/ReadMe.md) | Research implementations and notes | Experimental |

**Legacy** means that the material is preserved for learning and historical reference but is not yet covered by the planned compatibility and continuous-integration matrix.

**Experimental** means that the implementation or empirical conclusion is preliminary. It should not be treated as a generally validated performance claim.

**Alpha API** means that a small public interface is documented and covered by
contract and legacy-equivalence tests, but the project has not made a
production-readiness or semantic-versioning guarantee.

## Minimal stable API (alpha)

The first tested package surface deliberately contains only two detectors:
`MahalanobisDetector` and `PCAReconstructionDetector`. Both provide
`fit(X)`, `score_samples(X)`, and `predict(X)`; larger scores mean more
anomalous, and predictions use `1` for anomaly and `0` for normal.

```bash
python -m pip install -e .
```

```python
from anomaly_detection import MahalanobisDetector

detector = MahalanobisDetector(contamination=0.05).fit(X_train)
scores = detector.score_samples(X_test)
labels = detector.predict(X_test)
```

The new package is additive: the original scripts remain in place, and
characterization tests compare the extracted calculations with the handwritten
Mahalanobis and PCA implementations. See the complete
[API contract and migration notes](docs/stable-api.md).

RobustPCC and Kernel PCA remain experimental legacy implementations for now.
Their current threshold semantics and compatibility risks need separate tests
before they can be exposed through the same stable surface.

## Minimal historical example

To run the original Mahalanobis equivalence example from a fresh clone:

```bash
git clone https://github.com/Albertsr/Anomaly-Detection.git
cd Anomaly-Detection
python -m pip install numpy pandas scikit-learn
python "UnSupervised-Mahalanobis Distance/verify_mahal_equivalence.py"
```

Dependencies still vary across historical modules. Some semi-supervised
comparison scripts also require optional libraries such as LightGBM or XGBoost;
they are not installed by the minimal package.

## How to read the historical results

The existing figures and comparisons are educational experiments, not universal rankings:

- when ground-truth labels are available, metrics such as F1 or ROC AUC can evaluate predictions under the stated experimental setup;
- when labels are unavailable, agreement with Isolation Forest is only detector agreement, not proof of correctness;
- results from a small set of synthetic datasets should not be generalized to all anomaly types or data domains;
- thresholds, contamination assumptions, random seeds, and data construction can materially change the outcome.

The benchmark revival will make these boundaries explicit and add public labeled datasets, repeated seeds, runtime measurements, and uncertainty reporting.

### Historical experiment snapshots

The following figures preserve the original comparison outputs for ten synthetic datasets. The data-generation settings, dependency versions, hardware, and complete evaluation protocol have not yet been fully reconstructed, so treat them as historical snapshots—not current benchmarks or universal algorithm rankings.

![Historical F1 scores for nine anomaly-detection configurations across ten synthetic datasets](Algo%20Contrast/Pics/f1_contrast_v1.png)

*Historical F1 snapshot across ten synthetic datasets. Variation across datasets is the intended reading; individual levels should not be treated as a general ranking of the methods.*

<details>
<summary><strong>View the corresponding historical runtime snapshot</strong></summary>

![Historical runtime in seconds for nine anomaly-detection configurations across ten synthetic datasets](Algo%20Contrast/Pics/timecost_contrast_v1.png)

*Runtime depends on the original hardware, software environment, parameters, and implementation details. These values should not be used as current performance estimates.*

</details>

The historical figures include linear- and RBF-kernel One-Class SVM configurations, but the code revision that generated those two series has not yet been reconstructed. A dedicated method page, tested baseline, and stable repository interface for One-Class SVM remain planned.

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
