# Anomaly Detection

[English](README.md) | [简体中文](README.zh-CN.md)

An educational, code-first collection of handwritten Python implementations, visual method notes, and historical experiments by **Ma Xiao (MaXiao)**.

This repository records how I explored several early tabular anomaly-detection methods with Python—especially **PCA / Kernel PCA reconstruction error** and **RobustPCC**. The historically preserved repository materials—code, formula notes, explanations, figures, and experiment records—are the main content of the project.

> **Historical origin (around 2018):** this project began as my personal anomaly-detection study notebook—handwritten Python implementations, formula notes, paper walkthroughs, and exploratory experiments accumulated while I was learning the field.

> **Historical code policy:** version-specific APIs, naming, comments, and implementation style are part of that record. They are documented rather than silently rewritten. Readers who want to run a historical script on a modern stack can adapt a copy—manually or with modern coding tools—while keeping the original file as the reference.

The repository is now being revived incrementally. The goal is to add tests, compatibility work, clearer English documentation, and carefully scoped maintained interfaces around the historical work—not to replace it with a wholesale rewrite.

> **Authorship boundary:** for methods originating in cited papers, my contribution is the Python implementation, explanatory notes, and exploratory experiments, not the invention of the underlying algorithm. KADOA is separately labeled as an author-proposed experimental variation of ADOA.

## Featured handwritten studies

### 1. PCA and Kernel PCA reconstruction error

The PCA study is one of the most complete threads in this repository. It connects paper reading, matrix reconstruction, anomaly scoring, a NumPy/SVD implementation, Kernel PCA experiments, and historical comparisons.

| Implementation | Python entry point | What it explores | Current status |
|---|---|---|---|
| Linear PCA reconstruction | [`recon_error_pca.py`](UnSupervised-Based%20on%20PCA/recon_error_pca.py) | Weighted reconstruction errors across an increasing sequence of principal components | Historical implementation · Maintained candidate |
| NumPy/SVD reconstruction | [`recon_error_pca_svd.py`](UnSupervised-Based%20on%20PCA/recon_error_pca_svd.py) | The same reconstruction idea implemented directly with NumPy and SVD rather than scikit-learn PCA | Historical implementation · Characterization pending |
| Kernel PCA reconstruction | [`recon_error_kpca.py`](UnSupervised-Based%20on%20PCA/recon_error_kpca.py) | Reconstruction error in nonlinear feature spaces using different kernels | Historical implementation · 2018-era dependency API |

The detailed derivation and original learning path remain in the [PCA walkthrough](UnSupervised-Based%20on%20PCA/ReadMe.md).

#### Reconstruction and anomaly score

For each component count `k`, the historical implementation reconstructs the standardized data with the first `k` principal components. It then weights the corresponding reconstruction error by the cumulative explained-variance ratio and adds the weighted errors into one anomaly score.

![Historical PCA reconstruction formula](UnSupervised-Based%20on%20PCA/Pics/recon_matrix.jpg)

*Historical reconstruction formula preserved in the original PCA study notes. `R^k` is the matrix reconstructed with the first `k` principal components.*

![Historical weighted PCA reconstruction-error score](UnSupervised-Based%20on%20PCA/Pics/outlierscore.jpg)

*Historical score formula preserved in the original notes. The Chinese word “其中” means “where”; `ev(k)` is the cumulative explained-variance ratio used as the weight for reconstruction `k`.*

The linear implementation is being considered for a future maintained API. The Kernel PCA script remains visible in its historical form; in the audited scikit-learn 1.9 environment, its former `lambdas_` attribute is no longer available. This version boundary is documented instead of being silently rewritten into the original file.

### 2. RobustPCC: major and minor principal-component deviations

[`robustpcc.py`](UnSupervised-Based%20on%20PCA/robustpcc.py) is my paper-guided Python implementation of a principal-component classifier for anomaly detection.

The implementation keeps the method's main steps visible:

1. standardize the training data;
2. use a Mahalanobis-style score to trim a small proportion of potential extremes;
3. fit PCA on the remaining observations;
4. identify major components that explain roughly the first 50% of variance;
5. identify minor components whose eigenvalues fall below the historical threshold;
6. calculate normalized deviations on both component groups;
7. classify a sample as anomalous when either deviation exceeds its quantile threshold.

In the interpretation used by the cited RobustPCC method, these component groups provide two complementary anomaly views:

- **major-component deviation:** observations with extreme values in the original variables;
- **minor-component deviation:** observations that are unusual because their correlation structure differs from the training data.

The [PCA and RobustPCC notes](UnSupervised-Based%20on%20PCA/ReadMe.md) preserve the original formulas, terminology, implementation choices, and experiment discussion.

> **Current evidence status:** RobustPCC is an important historical implementation in this repository, not a production-ready detector. Its threshold semantics and numerical edge cases are still being characterized before any maintained API decision.

### 3. An eigenvalue-sensitivity experiment

[`max_ev_decrease.py`](UnSupervised-Based%20on%20PCA/max_ev_decrease.py) records an additional experiment from the original PCA study. It compares the PCA eigenvalues before and after removing observations detected as anomalies.

![Historical PCA eigenvalue-decrease experiment](UnSupervised-Based%20on%20PCA/Pics/max_ev_decrease.jpg)

*Historical output from ten synthetic datasets. In these recorded runs, the three largest relative eigenvalue decreases always included index 0 or 19. This is an experiment snapshot, not a claim that the pattern holds for every dataset.*

## Historical experiment gallery

The following figure is preserved because it shows the range of algorithms explored in the original Python experiments: Isolation Forest, LOF, Mahalanobis distance, PCA/KPCA reconstruction error, RobustPCC, and two One-Class SVM configurations.

![Historical F1 scores for nine anomaly-detection configurations across ten synthetic datasets](Algo%20Contrast/Pics/f1_contrast_v1.png)

*Historical F1 snapshot from ten synthetic datasets. The original environment and complete nine-series generation code have not yet been reconstructed, so the figure should be read as part of the repository's experiment history—not as a current benchmark or a general ranking of the methods.*

<details>
<summary><strong>View the corresponding historical runtime snapshot</strong></summary>

![Historical runtime in seconds for nine anomaly-detection configurations across ten synthetic datasets](Algo%20Contrast/Pics/timecost_contrast_v1.png)

*Runtime depends on the original hardware, software environment, parameters, and implementation details. These values should not be used as current performance estimates.*

</details>

The historical chart contains linear- and RBF-kernel One-Class SVM series, but their generating code has not been recovered from the reachable repository history. A future sklearn baseline must therefore be presented as new maintenance work, not reconstructed historical code.

## Algorithm collection at a glance

The repository contains both unsupervised and partially supervised study implementations.

| Learning setting | Method | Repository material | Evidence status |
|---|---|---|---|
| Unsupervised | **PCA reconstruction error** | [Linear PCA](UnSupervised-Based%20on%20PCA/recon_error_pca.py), [NumPy/SVD](UnSupervised-Based%20on%20PCA/recon_error_pca_svd.py), and [method notes](UnSupervised-Based%20on%20PCA/ReadMe.md) | Historical implementation · Linear version is a maintained candidate |
| Unsupervised | **Kernel PCA reconstruction error** | [Kernel PCA code](UnSupervised-Based%20on%20PCA/recon_error_kpca.py) and nonlinear-kernel experiments | Historical implementation · 2018-era dependency API |
| Unsupervised | **RobustPCC** | [Python implementation](UnSupervised-Based%20on%20PCA/robustpcc.py) and major/minor-component explanation | Historical implementation · Experimental validation |
| Unsupervised | [Mahalanobis distance](UnSupervised-Mahalanobis%20Distance/ReadMe.md) | Direct covariance-inverse form, PCA-space variant, and an equivalence experiment | Historical implementation · Maintained candidate |
| Unsupervised | [Isolation Forest](UnSupervised-Isolation%20Forest/ReadMe.md) | Paper notes and a scikit-learn example | Historical example · 2018-era dependency API |
| Unsupervised | [Local Outlier Factor](UnSupervised-Local%20Outlier%20Factor/ReadMe.md) | Visual method notes and train/novelty examples | Historical example |
| Partially supervised | [ADOA](SemiSupervised-ADOA/ReadMe.md) | Paper-guided implementation, clustering helper, and visual explanation | Historical implementation · Characterization pending |
| Partially supervised | [KADOA](SemiSupervised-KADOA-Original/ReadMe.md) | Author-proposed variation using Kernel PCA reconstruction error in the ADOA workflow | Author-proposed · Experimental |
| Positive-unlabeled | [PU Learning](SemiSupervised-PU%20Learning/ReadMe.md) | Spy/two-step strategy, Biased SVM, weighted logistic regression, papers, and notes | Historical code and notes · End-to-end execution not yet verified |

Status terms in this README describe the repository evidence, not the importance of a method:

- **Historical implementation:** original code is preserved and readable, but is not yet covered by a supported package and version matrix.
- **Maintained candidate:** a future maintained interface is under review; it is not yet a stable API on `master`.
- **2018-era dependency API:** the code preserves the library interface used when it was written; modern adaptations should be documented separately rather than silently replacing the historical source.
- **Experimental:** behavior or empirical conclusions remain preliminary and should not be generalized.

> **Known characterization findings:** the historical RobustPCC code selects all components when no eigenvalue meets its minor-component threshold, and the shared ADOA/KADOA clustering helper can break coordinate pairings by sorting cluster centers column by column. Their end-to-end impact has not yet been quantified.

## A minimal historical PCA example

The current `master` branch does not yet expose an installable stable package. This small example uses the historical class directly and scores the same matrix supplied at construction time:

```python
import sys

import numpy as np

sys.path.insert(0, "UnSupervised-Based on PCA")
from recon_error_pca import PCA_Recon_Error

rng = np.random.RandomState(2018)
X = rng.normal(size=(100, 4))

detector = PCA_Recon_Error(X, contamination=0.05)
scores = detector.get_anomaly_score()
labels = detector.predict()  # anomaly = 1, normal = 0

print(scores.shape)
print(labels.sum())
```

This is an ordinary-input historical smoke example, not a supported-version guarantee. It demonstrates the historical interface only. A future maintained detector will require an explicit, tested `fit` / `score_samples` / `predict` contract before it is presented as stable.

## Suggested reading paths

- **Start with the featured work:** [PCA reconstruction and RobustPCC](UnSupervised-Based%20on%20PCA/ReadMe.md).
- **Follow the unsupervised collection:** [Isolation Forest](UnSupervised-Isolation%20Forest/ReadMe.md) → [Mahalanobis distance](UnSupervised-Mahalanobis%20Distance/ReadMe.md) → [Local Outlier Factor](UnSupervised-Local%20Outlier%20Factor/ReadMe.md).
- **Explore partially observed anomalies:** [ADOA](SemiSupervised-ADOA/ReadMe.md) → [KADOA](SemiSupervised-KADOA-Original/ReadMe.md).
- **Explore positive-unlabeled learning:** [PU Learning notes and implementations](SemiSupervised-PU%20Learning/ReadMe.md).
- **Read the full original Chinese presentation:** [README.zh-CN.md](README.zh-CN.md).

## Revival boundary

The current `master` branch is still a historical learning collection rather than a stable installable package. Maintenance is deliberately incremental: preserve original paths, characterize behavior before changing it, and keep compatibility notes or maintained interfaces separate from the historical source.

RobustPCC and Kernel PCA remain central historical implementations even though they are not in the first maintained API candidate set. See the [revival roadmap](ROADMAP.md) for the current evidence boundary.

## About KADOA

KADOA is presented here as an experimental variation proposed by Ma Xiao. It retains the broader clustering-and-weighting structure of ADOA while replacing the Isolation Forest component with Kernel PCA reconstruction error.

KADOA has not been presented as a peer-reviewed standalone method, and the historical repository experiment does not establish universal superiority over ADOA. If you use or discuss it, describe it as an experimental repository method, acknowledge its relationship to ADOA, and cite this repository.

## Contributing and questions

Bug reports, compatibility notes, documentation corrections, and carefully scoped algorithm contributions are welcome through [GitHub Issues](https://github.com/Albertsr/Anomaly-Detection/issues).

For reproducibility reports, please include the Python and dependency versions, the exact script, a minimal input, and the observed traceback or output.

## Citation

Use the software citation metadata in [`CITATION.cff`](CITATION.cff). The project does not currently claim a DOI, formal release version, or independent KADOA publication.

## License and third-party materials

The author-owned code and documentation in this repository are licensed under the [MIT License](LICENSE).

Committed papers, publisher-formatted excerpts, and other third-party materials may have separate licenses. The repository's MIT License does not relicense those materials.
