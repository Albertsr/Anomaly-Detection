"""Reproducible datasets for the tabular anomaly-detection benchmark.

The public UCI datasets in this module are classification datasets converted
into anomaly-detection proxies. Their labels are useful for evaluation, but do
not make them universal ground truth for real-world anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import time
from typing import Any, Mapping
from urllib.error import URLError

import numpy as np


_DEFAULT_MANIFEST = Path(__file__).resolve().with_name("data_manifest.json")
_HASH_PREFIX = b"anomaly-detection-canonical-dataset-v1\0"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_UCI_FETCH_ATTEMPTS = 3


@dataclass(frozen=True)
class DatasetBundle:
    """Normalized benchmark data and the provenance needed to reproduce it."""

    name: str
    X: np.ndarray
    y: np.ndarray
    source: str
    sha256: str
    metadata: dict


class DatasetIntegrityError(ValueError):
    """Raised when cached or downloaded data do not match their declared hash."""


def load_manifest(path: str | os.PathLike[str] | None = None) -> dict:
    """Load and minimally validate the benchmark dataset manifest."""

    manifest_path = Path(path) if path is not None else _DEFAULT_MANIFEST
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported dataset manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"Manifest {manifest_path} must define a non-empty datasets object.")

    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {name!r} must be an object.")
        expected = entry.get("expected_sha256", "")
        if expected and not _SHA256_PATTERN.fullmatch(str(expected)):
            raise ValueError(f"Manifest entry {name!r} has an invalid expected_sha256.")

    return manifest


def load_dataset(
    name: str,
    *,
    seed: int = 2018,
    synthetic_config: Mapping[str, Any] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
) -> DatasetBundle:
    """Load one normalized benchmark dataset.

    Parameters
    ----------
    name:
        A dataset key from ``data_manifest.json``.
    seed:
        Random seed for the shuffled synthetic dataset.
    synthetic_config:
        Optional explicit overrides for the synthetic generator. Dataset size,
        dimensionality, and contamination never vary implicitly with ``seed``.
    cache_dir:
        Optional directory for canonical ``.npz`` files. Network access is only
        attempted when a requested UCI dataset is absent from this cache.
    manifest_path:
        Optional alternate manifest, mainly for controlled tests.
    """

    manifest = load_manifest(manifest_path)
    try:
        entry = manifest["datasets"][name]
    except KeyError as exc:
        available = ", ".join(sorted(manifest["datasets"]))
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}") from exc

    kind = entry.get("kind")
    if kind == "synthetic":
        config = _resolve_synthetic_config(entry, synthetic_config)
        cache_key = _cache_key(name, {"seed": int(seed), "config": config})
        cached = _load_cached_bundle(cache_dir, cache_key, name, entry)
        if cached is not None:
            return cached
        bundle = _make_historical_synthetic(name, int(seed), config, entry)
    elif kind == "uci_proxy":
        if synthetic_config is not None:
            raise ValueError("synthetic_config is only valid for synthetic_historical.")
        cache_key = _cache_key(name, entry)
        cached = _load_cached_bundle(cache_dir, cache_key, name, entry)
        if cached is not None:
            return cached
        bundle = _load_uci_proxy(name, entry)
    else:
        raise ValueError(f"Dataset {name!r} has unsupported kind {kind!r}.")

    _verify_expected_hash(name, bundle.sha256, entry)
    _write_cached_bundle(cache_dir, cache_key, bundle)
    return bundle


def _resolve_synthetic_config(
    entry: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    defaults = entry.get("generator")
    if not isinstance(defaults, dict):
        raise ValueError("synthetic_historical must define a generator object.")

    config = dict(defaults)
    if overrides is not None:
        unknown = set(overrides).difference(config)
        if unknown:
            raise ValueError(f"Unknown synthetic_config keys: {sorted(unknown)}")
        config.update(overrides)

    n_samples = int(config["n_samples"])
    n_features = int(config["n_features"])
    contamination = float(config["contamination"])
    gamma_shape = float(config["gamma_shape"])
    gamma_scale = float(config["gamma_scale"])
    exponential_scale = float(config["exponential_scale"])

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be in the open interval (0, 0.5).")
    if min(gamma_shape, gamma_scale, exponential_scale) <= 0.0:
        raise ValueError("Synthetic distribution parameters must be positive.")
    if int(n_samples * contamination) < 1:
        raise ValueError("Synthetic configuration must produce at least one anomaly.")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "contamination": contamination,
        "gamma_shape": gamma_shape,
        "gamma_scale": gamma_scale,
        "exponential_scale": exponential_scale,
    }


def _make_historical_synthetic(
    name: str,
    seed: int,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> DatasetBundle:
    """Generate the historical Normal versus Gamma/Exponential experiment.

    Unlike the legacy notebook, ``seed`` changes only the sampled values and
    final row order. Shape and contamination come from an explicit config.
    """

    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be between 0 and 2**32 - 1.")

    rng = np.random.RandomState(seed)
    n_samples = int(config["n_samples"])
    n_features = int(config["n_features"])
    n_anomalies = int(n_samples * float(config["contamination"]))
    n_normal = n_samples - n_anomalies
    n_gamma = n_anomalies // 2
    n_exponential = n_anomalies - n_gamma

    normal = rng.standard_normal(size=(n_normal, n_features))
    gamma = rng.gamma(
        shape=float(config["gamma_shape"]),
        scale=float(config["gamma_scale"]),
        size=(n_gamma, n_features),
    )
    exponential = rng.exponential(
        scale=float(config["exponential_scale"]),
        size=(n_exponential, n_features),
    )

    X = np.concatenate((normal, gamma, exponential), axis=0)
    y = np.concatenate(
        (
            np.zeros(n_normal, dtype=np.int8),
            np.ones(n_anomalies, dtype=np.int8),
        )
    )
    order = rng.permutation(n_samples)
    X, y = _normalize_arrays(X[order], y[order])
    digest = _canonical_sha256(X, y)

    metadata = {
        "kind": "synthetic",
        "seed": seed,
        "generator": dict(config),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_anomalies": int(y.sum()),
        "actual_contamination": float(y.mean()),
        "shuffled": True,
        "proxy": False,
        "cache_hit": False,
        "hash_schema": 1,
    }
    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        source=str(entry["source_url"]),
        sha256=digest,
        metadata=metadata,
    )


def _load_uci_proxy(name: str, entry: Mapping[str, Any]) -> DatasetBundle:
    """Fetch a UCI classification dataset and apply its declared proxy labels."""

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "UCI benchmark datasets require the optional dependency "
            "'ucimlrepo==0.0.7'. Install the benchmark extras or provide cache_dir."
        ) from exc

    uci_id = int(entry["uci_id"])
    repository_dataset = _fetch_uci_dataset(fetch_ucirepo, uci_id)
    (
        raw_X,
        target_values,
        selected_column,
        excluded_columns,
        data_layout,
    ) = _extract_uci_arrays(repository_dataset, entry)
    y = _map_proxy_labels(target_values, entry["target_mapping"], name)
    X, y = _normalize_arrays(raw_X, y)
    digest = _canonical_sha256(X, y)

    metadata = {
        "kind": "uci_proxy",
        "uci_id": uci_id,
        "doi": entry["doi"],
        "license": entry["license"],
        "proxy": True,
        "proxy_boundary": entry["proxy_boundary"],
        "target_column": selected_column,
        "excluded_columns": excluded_columns,
        "uci_data_layout": data_layout,
        "target_mapping": dict(entry["target_mapping"]),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_anomalies": int(y.sum()),
        "actual_contamination": float(y.mean()),
        "missing_values": int(np.isnan(X).sum()),
        "cache_hit": False,
        "hash_schema": 1,
    }
    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        source=str(entry["source_url"]),
        sha256=digest,
        metadata=metadata,
    )


def _fetch_uci_dataset(fetcher: Any, uci_id: int) -> Any:
    """Retry transient UCI transport failures without caching partial data."""

    retryable_errors = (ConnectionError, IncompleteRead, TimeoutError, URLError)
    for attempt in range(1, _UCI_FETCH_ATTEMPTS + 1):
        try:
            return fetcher(id=uci_id)
        except retryable_errors as exc:
            if attempt == _UCI_FETCH_ATTEMPTS:
                raise RuntimeError(
                    f"UCI dataset {uci_id} download failed after "
                    f"{_UCI_FETCH_ATTEMPTS} attempts; no cache was written."
                ) from exc
            time.sleep(0.5 * attempt)

    raise AssertionError("Unreachable UCI retry state.")


def _extract_uci_arrays(
    repository_dataset: Any,
    entry: Mapping[str, Any],
) -> tuple[Any, np.ndarray, str, list[str], str]:
    """Use curated UCI roles when available, otherwise split ``data.original``."""

    data = repository_dataset.data
    raw_X = getattr(data, "features", None)
    raw_targets = getattr(data, "targets", None)
    requested_target = str(entry["target_mapping"]["column"])

    if raw_X is not None and raw_targets is not None:
        target_values, selected_column = _select_target(raw_targets, requested_target)
        return raw_X, target_values, selected_column, [], "features_targets"

    original = getattr(data, "original", None)
    if original is None or not hasattr(original, "columns") or not hasattr(original, "drop"):
        raise ValueError(
            f"UCI dataset {entry['uci_id']} provides neither features/targets "
            "nor a tabular data.original fallback."
        )

    target_column = _resolve_column(original.columns, requested_target, "target")
    excluded_columns = [
        _resolve_column(original.columns, str(column), "excluded")
        for column in entry.get("excluded_columns", [])
    ]
    feature_exclusions = list(dict.fromkeys([target_column, *excluded_columns]))
    raw_X = original.drop(columns=feature_exclusions)
    if raw_X.shape[1] == 0:
        raise ValueError(f"UCI dataset {entry['uci_id']} has no feature columns after exclusion.")
    target_values = np.asarray(original[target_column]).reshape(-1)
    return (
        raw_X,
        target_values,
        str(target_column),
        [str(column) for column in excluded_columns],
        "original_fallback",
    )


def _resolve_column(columns: Any, requested: str, role: str) -> Any:
    available = list(columns)
    exact = [column for column in available if str(column) == requested]
    matches = exact or [
        column
        for column in available
        if str(column).strip().casefold() == requested.strip().casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected {role} column {requested!r}; available columns: "
            f"{[str(column) for column in available]}"
        )
    return matches[0]


def _select_target(raw_targets: Any, requested_column: str) -> tuple[np.ndarray, str]:
    if hasattr(raw_targets, "columns"):
        selected = _resolve_column(raw_targets.columns, requested_column, "target")
        return np.asarray(raw_targets[selected]).reshape(-1), str(selected)

    if hasattr(raw_targets, "name") and getattr(raw_targets, "name") is not None:
        selected_name = str(raw_targets.name)
        if selected_name.strip().casefold() != requested_column.strip().casefold():
            raise ValueError(
                f"Expected target column {requested_column!r}, got {selected_name!r}."
            )
        return np.asarray(raw_targets).reshape(-1), selected_name

    target_array = np.asarray(raw_targets)
    if target_array.ndim == 1:
        return target_array, requested_column
    if target_array.ndim == 2 and target_array.shape[1] == 1:
        return target_array[:, 0], requested_column
    raise ValueError(
        f"Cannot select target column {requested_column!r} from an unnamed "
        f"array with shape {target_array.shape}."
    )


def _map_proxy_labels(
    raw_target: np.ndarray,
    mapping: Mapping[str, Any],
    dataset_name: str,
) -> np.ndarray:
    anomaly_values = {_canonical_label(value) for value in mapping["anomaly_values"]}
    normal_values = {_canonical_label(value) for value in mapping["normal_values"]}
    overlap = anomaly_values.intersection(normal_values)
    if overlap:
        raise ValueError(f"Target mapping for {dataset_name!r} overlaps: {sorted(overlap)}")

    y = np.full(len(raw_target), -1, dtype=np.int8)
    unknown: list[Any] = []
    for index, raw_value in enumerate(raw_target):
        value = _canonical_label(raw_value)
        if value in anomaly_values:
            y[index] = 1
        elif value in normal_values:
            y[index] = 0
        elif len(unknown) < 10:
            unknown.append(raw_value)

    if np.any(y < 0):
        raise ValueError(
            f"Dataset {dataset_name!r} contains labels outside its manifest mapping. "
            f"Examples: {unknown!r}"
        )
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError(f"Dataset {dataset_name!r} must contain both anomaly classes.")
    return y


def _canonical_label(value: Any) -> tuple[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ("missing", None)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ("text", "")
        try:
            number = float(stripped)
        except ValueError:
            return ("text", stripped.casefold())
        return _canonical_numeric_label(number)

    if isinstance(value, (bool, int, float)):
        return _canonical_numeric_label(float(value))
    return ("text", str(value).strip().casefold())


def _canonical_numeric_label(value: float) -> tuple[str, Any]:
    if not np.isfinite(value):
        return ("missing", None)
    if value.is_integer():
        return ("number", int(value))
    return ("number", float(value).hex())


def _normalize_arrays(raw_X: Any, raw_y: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        if hasattr(raw_X, "to_numpy"):
            X = raw_X.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(raw_X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Benchmark features must be numeric or parseable as float64.") from exc

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(np.asarray(raw_y).reshape(-1), dtype=np.int8)

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2D matrix, got shape {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}.")
    if np.isinf(X).any():
        raise ValueError("X may contain finite values or NaN, but not infinity.")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("y must use 1 for anomaly and 0 for normal.")
    return X, y


def _canonical_sha256(X: np.ndarray, y: np.ndarray) -> str:
    """Hash canonical little-endian arrays, normalizing NaN payloads and -0.0."""

    canonical_X = np.array(X, dtype="<f8", order="C", copy=True)
    canonical_X[canonical_X == 0.0] = 0.0
    bits = canonical_X.view("<u8")
    bits[np.isnan(canonical_X)] = np.uint64(0x7FF8000000000000)
    canonical_y = np.asarray(y, dtype=np.int8, order="C")

    digest = hashlib.sha256()
    digest.update(_HASH_PREFIX)
    digest.update(
        struct.pack(
            "<QQQ",
            int(canonical_X.shape[0]),
            int(canonical_X.shape[1]),
            int(canonical_y.shape[0]),
        )
    )
    digest.update(canonical_X.tobytes(order="C"))
    digest.update(canonical_y.tobytes(order="C"))
    return digest.hexdigest()


def _verify_expected_hash(
    name: str,
    observed: str,
    entry: Mapping[str, Any],
) -> None:
    expected = str(entry.get("expected_sha256", "")).strip().casefold()
    if expected and observed.casefold() != expected:
        raise DatasetIntegrityError(
            f"Canonical SHA-256 mismatch for {name!r}: "
            f"expected {expected}, observed {observed}."
        )


def _cache_key(name: str, identity: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    suffix = hashlib.sha256(serialized).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return f"{safe_name}-{suffix}"


def _cache_path(
    cache_dir: str | os.PathLike[str] | None,
    cache_key: str,
) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir).expanduser().resolve() / f"{cache_key}.npz"


def _load_cached_bundle(
    cache_dir: str | os.PathLike[str] | None,
    cache_key: str,
    expected_name: str,
    entry: Mapping[str, Any],
) -> DatasetBundle | None:
    path = _cache_path(cache_dir, cache_key)
    if path is None or not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as cached:
            name = str(cached["name"].item())
            source = str(cached["source"].item())
            stored_hash = str(cached["sha256"].item())
            metadata = json.loads(str(cached["metadata_json"].item()))
            X, y = _normalize_arrays(cached["X"], cached["y"])
    except Exception as exc:
        raise DatasetIntegrityError(f"Cannot read dataset cache {path}: {exc}") from exc

    if name != expected_name:
        raise DatasetIntegrityError(
            f"Cache {path} contains dataset {name!r}, expected {expected_name!r}."
        )
    observed_hash = _canonical_sha256(X, y)
    if observed_hash.casefold() != stored_hash.casefold():
        raise DatasetIntegrityError(
            f"Cached SHA-256 mismatch for {expected_name!r}: "
            f"stored {stored_hash}, observed {observed_hash}."
        )
    _verify_expected_hash(expected_name, observed_hash, entry)

    metadata = dict(metadata)
    metadata["cache_hit"] = True
    metadata["cache_path"] = str(path)
    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        source=source,
        sha256=observed_hash,
        metadata=metadata,
    )


def _write_cached_bundle(
    cache_dir: str | os.PathLike[str] | None,
    cache_key: str,
    bundle: DatasetBundle,
) -> None:
    path = _cache_path(cache_dir, cache_key)
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        bundle.metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.stem}-",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        np.savez_compressed(
            temporary_path,
            X=bundle.X,
            y=bundle.y,
            name=np.asarray(bundle.name),
            source=np.asarray(bundle.source),
            sha256=np.asarray(bundle.sha256),
            metadata_json=np.asarray(metadata_json),
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
