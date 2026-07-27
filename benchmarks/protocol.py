"""Leakage-resistant train/validation/test benchmark protocol."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .registry import get_detector_spec


METRIC_FIELDS = (
    "average_precision",
    "roc_auc",
    "f1",
    "precision",
    "recall",
)

TIMING_FIELDS = (
    "preprocessing_fit_seconds",
    "preprocessing_transform_seconds",
    "fit_seconds",
    "validation_score_seconds",
    "test_score_seconds",
)

RESULT_FIELDS = (
    "dataset",
    "dataset_sha256",
    "source",
    "trial_seed",
    "detector",
    "parameters",
    "n_train",
    "n_validation",
    "n_test",
    "n_features",
    "threshold_strategy",
    "threshold",
    *METRIC_FIELDS,
    *TIMING_FIELDS,
    "timing_repeats",
    "status",
    "error",
)

_SPLIT_FOLD_COUNTS = (3, 1, 1)
_N_SPLIT_FOLDS = sum(_SPLIT_FOLD_COUNTS)


@dataclass(frozen=True)
class DataSplit:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def split_dataset(
    X,
    y,
    *,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> DataSplit:
    """Create one deterministic stratified, duplicate-grouped split.

    Exact duplicate feature rows are kept in one partition. Labels are used
    only to preserve class proportions across partitions and are never passed
    to a detector.
    """
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction],
        dtype=float,
    )
    if not np.all(fractions > 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("train, validation, and test fractions must be positive and sum to 1.")
    expected_fractions = np.asarray(_SPLIT_FOLD_COUNTS, dtype=float) / _N_SPLIT_FOLDS
    if not np.allclose(fractions, expected_fractions):
        raise ValueError(
            "The grouped benchmark currently requires a 0.6 / 0.2 / 0.2 split."
        )

    X_array = np.asarray(X, dtype=float)
    raw_y = np.asarray(y)
    if X_array.ndim != 2 or raw_y.ndim != 1 or len(X_array) != len(raw_y):
        raise ValueError("X must be 2D and y must be a matching 1D array.")
    if not np.all(np.isin(raw_y, (0, 1))):
        raise ValueError("Benchmark labels must use only 0 and 1.")
    y_array = np.asarray(raw_y, dtype=np.int8)
    if set(np.unique(y_array).tolist()) != {0, 1}:
        raise ValueError("Benchmark labels must contain both 0 and 1.")
    if not 0 <= int(seed) <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be between 0 and 2**32 - 1.")
    if np.isinf(X_array).any():
        raise ValueError("X may contain finite values or NaN, but not infinity.")

    groups = _feature_row_groups(X_array)
    if len(np.unique(groups)) < _N_SPLIT_FOLDS:
        raise ValueError(
            f"At least {_N_SPLIT_FOLDS} distinct feature-row groups are required."
        )
    _validate_group_labels(groups, y_array)

    fold_ids = np.full(len(X_array), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(
        n_splits=_N_SPLIT_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    for fold_id, (_, held_out_indices) in enumerate(
        splitter.split(X_array, y_array, groups)
    ):
        fold_ids[held_out_indices] = fold_id
    if np.any(fold_ids < 0):
        raise RuntimeError("Grouped split did not assign every sample to a fold.")

    train_stop = _SPLIT_FOLD_COUNTS[0]
    validation_stop = train_stop + _SPLIT_FOLD_COUNTS[1]
    train_indices = np.flatnonzero(fold_ids < train_stop)
    validation_indices = np.flatnonzero(
        (fold_ids >= train_stop) & (fold_ids < validation_stop)
    )
    test_indices = np.flatnonzero(fold_ids >= validation_stop)
    train_indices = _shuffle_partition_indices(train_indices, int(seed), 1)
    validation_indices = _shuffle_partition_indices(
        validation_indices,
        int(seed),
        2,
    )
    test_indices = _shuffle_partition_indices(test_indices, int(seed), 3)

    for partition_name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        if set(np.unique(y_array[indices]).tolist()) != {0, 1}:
            raise ValueError(
                f"Grouped {partition_name} split does not contain both label classes."
            )

    return DataSplit(
        X_train=X_array[train_indices],
        X_validation=X_array[validation_indices],
        X_test=X_array[test_indices],
        y_train=y_array[train_indices],
        y_validation=y_array[validation_indices],
        y_test=y_array[test_indices],
    )


def _feature_row_groups(X: np.ndarray) -> np.ndarray:
    """Assign the same group id to bitwise-equivalent normalized rows."""

    canonical = np.array(X, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    bits = canonical.view("<u8")
    bits[np.isnan(canonical)] = np.uint64(0x7FF8000000000000)
    _, groups = np.unique(bits, axis=0, return_inverse=True)
    return groups


def _validate_group_labels(groups: np.ndarray, y: np.ndarray) -> None:
    """Reject identical feature rows whose proxy labels disagree."""

    group_count = int(np.max(groups)) + 1
    minimum = np.ones(group_count, dtype=np.int8)
    maximum = np.zeros(group_count, dtype=np.int8)
    np.minimum.at(minimum, groups, y)
    np.maximum.at(maximum, groups, y)
    conflicting = np.flatnonzero(minimum != maximum)
    if len(conflicting):
        raise ValueError(
            "Exact duplicate feature rows have conflicting labels in "
            f"{len(conflicting)} group(s)."
        )


def _shuffle_partition_indices(
    indices: np.ndarray,
    seed: int,
    partition_id: int,
) -> np.ndarray:
    """Shuffle one partition reproducibly after grouped fold assignment."""

    partition_seed = (seed + 0x9E3779B9 * partition_id) % (2**32)
    rng = np.random.RandomState(partition_seed)
    return indices[rng.permutation(len(indices))]


def make_preprocessor() -> Pipeline:
    """Return the shared train-only preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("drop_constant", VarianceThreshold()),
            ("scale", StandardScaler()),
        ]
    )


def preprocess_split(split: DataSplit):
    preprocessor = make_preprocessor()

    fit_start = time.perf_counter_ns()
    X_train = preprocessor.fit_transform(split.X_train)
    preprocessing_fit_seconds = (time.perf_counter_ns() - fit_start) / 1e9

    transform_start = time.perf_counter_ns()
    X_validation = preprocessor.transform(split.X_validation)
    X_test = preprocessor.transform(split.X_test)
    preprocessing_transform_seconds = (time.perf_counter_ns() - transform_start) / 1e9

    transformed = DataSplit(
        X_train=np.asarray(X_train, dtype=float),
        X_validation=np.asarray(X_validation, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_train=split.y_train,
        y_validation=split.y_validation,
        y_test=split.y_test,
    )
    return transformed, preprocessing_fit_seconds, preprocessing_transform_seconds


def select_validation_threshold(y_validation, scores_validation) -> float:
    """Maximize validation F1 and choose the higher threshold on ties."""
    y_validation = np.asarray(y_validation, dtype=np.int8)
    scores_validation = np.asarray(scores_validation, dtype=float)
    if y_validation.shape != scores_validation.shape:
        raise ValueError("Validation labels and scores must have the same shape.")
    if not np.all(np.isfinite(scores_validation)):
        raise ValueError("Validation scores must be finite.")

    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        scores_validation,
    )
    if len(thresholds) == 0:
        raise ValueError("Validation scores do not define a usable threshold.")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    best_f1 = float(np.max(f1_values))
    best_indices = np.flatnonzero(
        np.isclose(f1_values, best_f1, rtol=1e-12, atol=1e-15)
    )
    return float(np.max(thresholds[best_indices]))


def compute_test_metrics(y_test, scores_test, threshold: float) -> dict[str, float]:
    y_test = np.asarray(y_test, dtype=np.int8)
    scores_test = np.asarray(scores_test, dtype=float)
    prediction = (scores_test >= threshold).astype(np.int8)
    return {
        "average_precision": float(average_precision_score(y_test, scores_test)),
        "roc_auc": float(roc_auc_score(y_test, scores_test)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
    }


def _validate_scores(scores, expected_length: int, detector_name: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (expected_length,):
        raise RuntimeError(
            f"{detector_name} returned scores with shape {scores.shape}; "
            f"expected {(expected_length,)}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{detector_name} produced non-finite scores.")
    return scores


def _json_parameters(detector) -> str:
    if not hasattr(detector, "get_params"):
        return "{}"
    return json.dumps(
        detector.get_params(deep=False),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def evaluate_detector(
    *,
    dataset_name: str,
    dataset_sha256: str,
    source: str,
    detector_name: str,
    split: DataSplit,
    trial_seed: int,
    preprocessing_fit_seconds: float,
    preprocessing_transform_seconds: float,
    timing_repeats: int = 1,
    warmup: bool = False,
) -> dict:
    """Fit and evaluate one detector without exposing labels to the estimator."""
    if timing_repeats < 1:
        raise ValueError("timing_repeats must be at least 1.")

    spec = get_detector_spec(detector_name)
    detector_for_parameters = spec.create(trial_seed)
    base_row = {
        "dataset": dataset_name,
        "dataset_sha256": dataset_sha256,
        "source": source,
        "trial_seed": trial_seed,
        "detector": detector_name,
        "parameters": _json_parameters(detector_for_parameters),
        "n_train": len(split.X_train),
        "n_validation": len(split.X_validation),
        "n_test": len(split.X_test),
        "n_features": split.X_train.shape[1],
        "threshold_strategy": "validation_max_f1",
        "threshold": None,
        **{field: None for field in METRIC_FIELDS},
        "preprocessing_fit_seconds": preprocessing_fit_seconds,
        "preprocessing_transform_seconds": preprocessing_transform_seconds,
        "fit_seconds": None,
        "validation_score_seconds": None,
        "test_score_seconds": None,
        "timing_repeats": timing_repeats,
        "status": "error",
        "error": None,
    }

    try:
        with threadpool_limits(limits=1):
            if warmup:
                warmup_detector = spec.create(trial_seed)
                warmup_detector.fit(split.X_train[: min(64, len(split.X_train))])
                spec.score_samples(
                    warmup_detector,
                    split.X_validation[: min(32, len(split.X_validation))],
                )

            fit_times = []
            validation_score_times = []
            test_score_times = []
            validation_scores = None
            test_scores = None

            for _ in range(timing_repeats):
                detector = spec.create(trial_seed)

                start = time.perf_counter_ns()
                detector.fit(split.X_train)
                fit_times.append((time.perf_counter_ns() - start) / 1e9)

                start = time.perf_counter_ns()
                current_validation_scores = spec.score_samples(
                    detector,
                    split.X_validation,
                )
                validation_score_times.append((time.perf_counter_ns() - start) / 1e9)

                start = time.perf_counter_ns()
                current_test_scores = spec.score_samples(detector, split.X_test)
                test_score_times.append((time.perf_counter_ns() - start) / 1e9)

                validation_scores = _validate_scores(
                    current_validation_scores,
                    len(split.X_validation),
                    detector_name,
                )
                test_scores = _validate_scores(
                    current_test_scores,
                    len(split.X_test),
                    detector_name,
                )

        threshold = select_validation_threshold(
            split.y_validation,
            validation_scores,
        )
        metrics = compute_test_metrics(split.y_test, test_scores, threshold)
        return {
            **base_row,
            "threshold": threshold,
            **metrics,
            "fit_seconds": statistics.median(fit_times),
            "validation_score_seconds": statistics.median(validation_score_times),
            "test_score_seconds": statistics.median(test_score_times),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            **base_row,
            "error": f"{type(exc).__name__}: {exc}",
        }
