"""Stable summaries without selecting an overall winner."""

from __future__ import annotations

import statistics
from collections import defaultdict

from .protocol import METRIC_FIELDS, TIMING_FIELDS


def summarize_results(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["detector"])].append(row)

    summaries = []
    for (dataset, detector), group_rows in sorted(groups.items()):
        successful = [row for row in group_rows if row["status"] == "ok"]
        summary = {
            "dataset": dataset,
            "detector": detector,
            "successful_trials": len(successful),
            "failed_trials": len(group_rows) - len(successful),
        }
        for field in (*METRIC_FIELDS, *TIMING_FIELDS):
            values = [float(row[field]) for row in successful]
            summary[f"{field}_mean"] = (
                statistics.fmean(values) if values else None
            )
            summary[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        summaries.append(summary)
    return summaries


def render_markdown_report(summaries: list[dict], metadata: dict) -> str:
    lines = [
        "# Benchmark report",
        "",
        "This report keeps results separated by dataset and does not select an "
        "overall winner. Values are means across successful trial seeds; "
        "uncertainty remains available in `summary.csv`.",
        "",
        "Interpretation boundary: public UCI rows are supervised "
        "classification-to-anomaly proxies, while synthetic rows are controlled "
        "stress tests. Neither establishes a universal algorithm ranking.",
        "",
        f"- Git commit: `{metadata.get('git_commit', 'unknown')}`",
        f"- Python: `{metadata.get('python', 'unknown')}`",
        f"- Platform: `{metadata.get('platform', 'unknown')}`",
        "",
        "| Dataset | Detector | Successful | Failed | Average precision | ROC AUC | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        def display(field):
            value = row.get(field)
            return "" if value is None else f"{value:.4f}"

        lines.append(
            "| {dataset} | {detector} | {successful} | {failed} | {ap} | {auc} | {f1} |".format(
                dataset=row["dataset"],
                detector=row["detector"],
                successful=row["successful_trials"],
                failed=row["failed_trials"],
                ap=display("average_precision_mean"),
                auc=display("roc_auc_mean"),
                f1=display("f1_mean"),
            )
        )
    lines.append("")
    return "\n".join(lines)
