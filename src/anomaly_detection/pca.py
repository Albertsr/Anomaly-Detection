"""PCA reconstruction detector extracted from the historical implementation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ._base import BaseAnomalyDetector


class PCAReconstructionDetector(BaseAnomalyDetector):
    """Aggregate reconstruction errors across the PCA component sequence.

    This preserves the distinctive idea in the historical
    ``PCA_Recon_Error`` class: fit reconstruction models from one component
    through the available component count, weight each reconstruction error by
    cumulative explained variance, and add the weighted errors.
    """

    def __init__(
        self,
        contamination: float = 0.01,
        random_state: int | None = 2018,
    ) -> None:
        super().__init__(contamination=contamination)
        self.random_state = random_state

    def _fit_model(self, X: np.ndarray) -> None:
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)
        if not np.any(np.var(X_scaled, axis=0) > 0.0):
            raise ValueError("PCA reconstruction requires data with non-zero variance.")

        self.pca_ = PCA(
            n_components=None,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca_.fit(X_scaled)
        eigenvalues = np.asarray(self.pca_.explained_variance_, dtype=float)
        total_eigenvalue = float(np.sum(eigenvalues))
        if not np.isfinite(total_eigenvalue) or total_eigenvalue <= 0.0:
            raise ValueError("PCA reconstruction requires data with non-zero variance.")

        self.component_weights_ = np.cumsum(eigenvalues) / total_eigenvalue

    def _score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        reduced = self.pca_.transform(X_scaled)
        scores = np.zeros(len(X_scaled), dtype=float)
        reconstructed = np.broadcast_to(
            self.pca_.mean_,
            X_scaled.shape,
        ).copy()

        for component_index, weight in enumerate(self.component_weights_):
            reconstructed += (
                reduced[:, component_index, np.newaxis]
                * self.pca_.components_[component_index]
            )
            reconstruction_error = np.linalg.norm(
                X_scaled - reconstructed,
                axis=1,
            )
            scores += reconstruction_error * weight
        return scores
