# Minimal stable API

The repository now exposes a deliberately small, alpha-stage package interface
while keeping every historical script in its original path.

## Contract

Both public detectors provide:

- `fit(X)`, which returns the fitted detector;
- `score_samples(X)`, where a larger value always means more anomalous;
- `predict(X)`, which returns `1` for an anomaly and `0` for a normal sample.

`contamination` must be a finite float in `[0, 0.5]`. During `fit`, the API
learns `threshold_` from the highest training scores implied by this fraction.
With `contamination=0`, the threshold is infinite and all predictions are
normal, while scores remain available. `predict` applies the fitted threshold
to later samples. For positive contamination, the selected training count is
`ceil(n_samples * contamination)`. This can exceed the exact fraction for small
or odd sample counts. When several samples tie exactly at the threshold, the
inclusive `>=` boundary can increase the count further; this is intentional and
avoids order-dependent tie breaking.

Inputs must be finite, numeric, two-dimensional arrays. A fitted detector
accepts one or more samples but requires the same feature count used in `fit`.
Training requires at least two samples. PCA reconstruction also requires at
least one feature with non-zero variance.

## Included detectors

### `MahalanobisDetector`

This is a tested extraction of the handwritten Mahalanobis-distance script.
It preserves the original centred covariance calculation and score direction.
The package version uses a pseudo-inverse, which agrees with the original
inverse on nonsingular inputs and also defines behaviour for singular
covariance matrices.

### `PCAReconstructionDetector`

This preserves the repository's distinctive PCA reconstruction aggregation:
reconstructions from one through the available component count are evaluated,
each reconstruction error is weighted by cumulative explained variance, and
the weighted errors are summed. The implementation now fits PCA once and
reuses its ordered components with the deterministic full SVD solver. On the
characterization inputs where the historical `auto` solver selects a
non-randomized decomposition, tests verify that this optimization preserves the
historical scores. The stable API does not promise bitwise equivalence to an old
run where scikit-learn selected a randomized solver. It is intentionally not
replaced with a generic single-reconstruction PCA detector.

`random_state=2018` is retained as an explicit migration parameter. The current
full-SVD implementation is deterministic and does not normally consume it; the
parameter prevents a future solver change from introducing an implicit seed.

## Installation and example

From a clone of this repository:

```bash
python -m pip install -e .
```

```python
import numpy as np

from anomaly_detection import MahalanobisDetector

rng = np.random.default_rng(2018)
X_train = rng.normal(size=(100, 4))
X_test = np.vstack([rng.normal(size=(5, 4)), np.full((1, 4), 8.0)])

detector = MahalanobisDetector(contamination=0.05).fit(X_train)
scores = detector.score_samples(X_test)
labels = detector.predict(X_test)
```

The interface is alpha-stage: the three methods, score direction, and label
meaning are the supported core. Additional estimators and convenience methods
will be added only through separately tested changes.

## Historical compatibility

The package is additive. It does not delete, rename, or silently redirect the
historical files. Tests compare the new training scores with the original
Mahalanobis and PCA implementations on deterministic nonsingular data, so the
author's handwritten algorithmic choices remain visible and reviewable.
