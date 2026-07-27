"""Mahalanobis detector extracted from the historical handwritten script."""

from __future__ import annotations

import numpy as np

from ._base import BaseAnomalyDetector


class MahalanobisDetector(BaseAnomalyDetector):
    """Score observations by their Mahalanobis distance from the fitted centre.

    The calculation follows ``UnSupervised-Mahalanobis Distance/mahal_dist.py``.
    A pseudo-inverse is used so that singular covariance matrices have defined,
    testable behaviour instead of failing with a linear-algebra exception.
    """

    def __init__(self, contamination: float = 0.01) -> None:
        super().__init__(contamination=contamination)

    def _fit_model(self, X: np.ndarray) -> None:
        self.location_ = np.mean(X, axis=0)
        covariance = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))

        self.covariance_ = covariance
        self.precision_ = np.linalg.pinv(covariance, hermitian=True)

    def _score_samples(self, X: np.ndarray) -> np.ndarray:
        delta = X - self.location_
        squared_distance = np.einsum(
            "ij,jk,ik->i",
            delta,
            self.precision_,
            delta,
            optimize=True,
        )
        return np.sqrt(np.maximum(squared_distance, 0.0))
