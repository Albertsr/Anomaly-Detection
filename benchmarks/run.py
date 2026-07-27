"""Command-line runner for the reproducible benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .datasets import load_dataset
from .protocol import (
    METRIC_FIELDS,
    RESULT_FIELDS,
    TIMING_FIELDS,
    evaluate_detector,
    preprocess_split,
    split_dataset,
)
from .registry import available_detectors
from .summarize import render_markdown_report, summarize_results


def load_config(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {"datasets", "detectors", "seeds", "split", "timing"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}.")

    unknown_detectors = set(config["detectors"]) - set(available_detectors())
    if unknown_detectors:
        raise ValueError(f"Unknown detectors in config: {sorted(unknown_detectors)}.")
    if not config["datasets"] or not config["detectors"] or not config["seeds"]:
        raise ValueError("datasets, detectors, and seeds must be non-empty.")
    return config


def _git_value(arguments, default="unknown"):
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or default
    except (OSError, subprocess.CalledProcessError):
        return default


def collect_run_metadata(config: dict) -> dict:
    package_names = (
        "numpy",
        "scikit-learn",
        "pandas",
        "threadpoolctl",
        "ucimlrepo",
    )
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"], default="")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "thread_limit": 1,
        "dependency_versions": versions,
        "config_schema_version": config.get("schema_version", 1),
    }


def _failed_rows_for_dataset(
    *,
    dataset_name,
    detector_names,
    seed,
    error,
    timing_repeats,
):
    rows = []
    for detector_name in detector_names:
        row = {field: None for field in RESULT_FIELDS}
        row.update(
            {
                "dataset": dataset_name,
                "dataset_sha256": "",
                "source": "",
                "trial_seed": seed,
                "detector": detector_name,
                "parameters": "",
                "threshold_strategy": "validation_max_f1",
                "timing_repeats": timing_repeats,
                "status": "error",
                "error": error,
            }
        )
        rows.append(row)
    return rows


def run_benchmark(config: dict, *, data_cache=None) -> list[dict]:
    rows = []
    split_config = config["split"]
    timing_config = config["timing"]
    timing_repeats = int(timing_config.get("repeats", 1))
    warmup = bool(timing_config.get("warmup", False))

    for dataset_name in config["datasets"]:
        for seed in config["seeds"]:
            try:
                bundle = load_dataset(
                    dataset_name,
                    seed=int(seed),
                    synthetic_config=config.get("synthetic"),
                    cache_dir=data_cache,
                )
                raw_split = split_dataset(
                    bundle.X,
                    bundle.y,
                    seed=int(seed),
                    train_fraction=float(split_config["train"]),
                    validation_fraction=float(split_config["validation"]),
                    test_fraction=float(split_config["test"]),
                )
                split, preprocessing_fit, preprocessing_transform = preprocess_split(
                    raw_split
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rows.extend(
                    _failed_rows_for_dataset(
                        dataset_name=dataset_name,
                        detector_names=config["detectors"],
                        seed=seed,
                        error=error,
                        timing_repeats=timing_repeats,
                    )
                )
                continue

            for detector_name in config["detectors"]:
                rows.append(
                    evaluate_detector(
                        dataset_name=bundle.name,
                        dataset_sha256=bundle.sha256,
                        source=bundle.source,
                        detector_name=detector_name,
                        split=split,
                        trial_seed=int(seed),
                        preprocessing_fit_seconds=preprocessing_fit,
                        preprocessing_transform_seconds=preprocessing_transform,
                        timing_repeats=timing_repeats,
                        warmup=warmup,
                    )
                )

    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            str(row["detector"]),
            int(row["trial_seed"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, config: dict, rows: list[dict]) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = collect_run_metadata(config)
    summaries = summarize_results(rows)

    with (output_path / "config.resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_path / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    metric_fields = [
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
        "status",
        "error",
    ]
    timing_fields = [
        "dataset",
        "trial_seed",
        "detector",
        *TIMING_FIELDS,
        "timing_repeats",
        "status",
        "error",
    ]
    _write_csv(output_path / "metrics.csv", rows, metric_fields)
    _write_csv(output_path / "timings.csv", rows, timing_fields)
    if summaries:
        _write_csv(
            output_path / "summary.csv",
            summaries,
            list(summaries[0]),
        )
    (output_path / "report.md").write_text(
        render_markdown_report(summaries, metadata),
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the leakage-resistant anomaly-detection benchmark."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--data-cache",
        default=None,
        help="Optional cache directory for normalized public datasets.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    rows = run_benchmark(config, data_cache=args.data_cache)
    write_outputs(args.output, config, rows)

    failed = sum(row["status"] != "ok" for row in rows)
    print(f"Completed {len(rows)} benchmark rows; failures: {failed}.")
    print(f"Outputs: {Path(args.output).resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
