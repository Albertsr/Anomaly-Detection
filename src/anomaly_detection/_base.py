"""Shared behavior for the small public detector API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted


class BaseAnomalyDetector(BaseEstimator, ABC):
    """Base class for detectors whose larger score means more anomalous.

    The repository API intentionally uses ``1`` for an anomaly and ``0`` for
    a normal observation. This differs from the ``-1`` / ``1`` convention
    used by several scikit-learn outlier estimators.
    """

    def __init__(self, contamination: float = 0.01) -> None:
        self.contamination = contamination

    def fit(self, X, y=None):
        """Fit the detector and learn a threshold from the training scores."""
        del y
        self._validate_contamination()
        X_checked = self._validate_X(X, reset=True)
        self._fit_model(X_checked)

        train_scores = self._score_samples(X_checked)
        self._validate_scores(train_scores, len(X_checked))
        self.train_scores_ = train_scores.copy()
        self.threshold_ = self._threshold_from_scores(train_scores)
        return self

    def score_samples(self, X) -> np.ndarray:
        """Return anomaly scores, where a larger value is more anomalous."""
        check_is_fitted(self, attributes=["threshold_", "n_features_in_"])
        X_checked = self._validate_X(X, reset=False)
        scores = np.asarray(self._score_samples(X_checked), dtype=float)
        self._validate_scores(scores, len(X_checked))
        return scores

    def predict(self, X) -> np.ndarray:
        """Return ``1`` for anomalies and ``0`` for normal observations."""
        scores = self.score_samples(X)
        return (scores >= self.threshold_).astype(np.int8)

    def _validate_contamination(self) -> None:
        if (
            not isinstance(self.contamination, Real)
            or isinstance(self.contamination, bool)
            or not np.isfinite(self.contamination)
            or not 0.0 <= float(self.contamination) <= 0.5
        ):
            raise ValueError("contamination must be a finite float in [0, 0.5].")

    def _validate_X(self, X, *, reset: bool) -> np.ndarray:
        X_checked = check_array(
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype=float,
            ensure_min_samples=2 if reset else 1,
        )

        if reset:
            self.n_features_in_ = X_checked.shape[1]
        elif X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X_checked.shape[1]} features, but this detector was fitted "
                f"with {self.n_features_in_} features."
            )
        return X_checked

    def _threshold_from_scores(self, scores: np.ndarray) -> float:
        if float(self.contamination) == 0.0:
            return float("inf")
        anomaly_count = max(1, int(np.ceil(len(scores) * float(self.contamination))))
        threshold_index = len(scores) - anomaly_count
        return float(np.partition(scores, threshold_index)[threshold_index])

    @staticmethod
    def _validate_scores(scores: np.ndarray, sample_count: int) -> None:
        if scores.shape != (sample_count,):
            raise RuntimeError("A detector must return one score per input sample.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("The detector produced non-finite anomaly scores.")

    @abstractmethod
    def _fit_model(self, X: np.ndarray) -> None:
        """Fit algorithm-specific state."""

    @abstractmethod
    def _score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score validated input without checking fitted state."""
