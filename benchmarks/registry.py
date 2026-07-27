"""Detector registry with one anomaly-score direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from anomaly_detection import MahalanobisDetector, PCAReconstructionDetector


@dataclass(frozen=True)
class DetectorSpec:
    """Factory and score orientation for one benchmark detector."""

    name: str
    factory: Callable[[int], object]
    score_multiplier: float

    def create(self, seed: int):
        return self.factory(seed)

    def score_samples(self, detector, X) -> np.ndarray:
        scores = np.asarray(detector.score_samples(X), dtype=float)
        return scores * self.score_multiplier


DETECTORS = {
    "mahalanobis": DetectorSpec(
        name="mahalanobis",
        factory=lambda seed: MahalanobisDetector(contamination=0.0),
        score_multiplier=1.0,
    ),
    "pca_reconstruction": DetectorSpec(
        name="pca_reconstruction",
        factory=lambda seed: PCAReconstructionDetector(
            contamination=0.0,
            random_state=seed,
        ),
        score_multiplier=1.0,
    ),
    "isolation_forest": DetectorSpec(
        name="isolation_forest",
        factory=lambda seed: IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        ),
        score_multiplier=-1.0,
    ),
    "local_outlier_factor": DetectorSpec(
        name="local_outlier_factor",
        factory=lambda seed: LocalOutlierFactor(
            n_neighbors=20,
            contamination="auto",
            novelty=True,
            n_jobs=1,
        ),
        score_multiplier=-1.0,
    ),
    "one_class_svm": DetectorSpec(
        name="one_class_svm",
        factory=lambda seed: OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.1,
        ),
        score_multiplier=-1.0,
    ),
}


def available_detectors() -> tuple[str, ...]:
    return tuple(DETECTORS)


def get_detector_spec(name: str) -> DetectorSpec:
    try:
        return DETECTORS[name]
    except KeyError as exc:
        choices = ", ".join(available_detectors())
        raise ValueError(f"Unknown detector {name!r}. Available: {choices}.") from exc
