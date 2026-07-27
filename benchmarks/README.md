# Fair benchmark protocol

This benchmark replaces the historical comparison script with a new,
score-first protocol. The old script and figures remain unchanged as historical
evidence; new outputs never overwrite them.

## What “fair” means here

- Every trial uses an approximately `60% / 20% / 20%` stratified
  train/validation/test split.
- Exact duplicate feature rows are grouped before splitting, so one row pattern
  cannot appear across train, validation, and test partitions. Conflicting
  labels within an identical-row group fail explicitly. Each completed
  partition is then deterministically shuffled.
- Labels preserve split proportions, calibrate one validation threshold, and
  compute final metrics. No detector receives labels in `fit`.
- Median imputation, constant-feature removal, and standardization are fitted
  on training features only.
- The detector registry converts every method to one convention: larger scores
  mean more anomalous.
- Average precision and ROC AUC use raw test scores. F1, precision, and recall
  use the threshold that maximizes validation F1; ties choose the higher
  threshold. `y_test` cannot affect that threshold.
- Trial seeds are reported separately. Summaries contain per-dataset mean and
  sample standard deviation and do not create a cross-dataset “best
  algorithm.”
- Failed methods remain explicit result rows.
- Model timing uses one thread, `perf_counter_ns`, an optional warm-up, and the
  median of configured repeats. Shared preprocessing time is reported
  separately. Timing is descriptive, never a CI speed gate.

This is a supervised evaluation protocol for unsupervised scores: validation
labels calibrate the thresholded metrics. Ranking metrics remain independent of
that threshold.

## Detectors

The benchmark includes the two repository-owned alpha APIs plus three locked
scikit-learn baselines:

- Mahalanobis distance;
- handwritten PCA reconstruction aggregation;
- Isolation Forest;
- Local Outlier Factor in novelty mode;
- RBF One-Class SVM.

Model parameters are emitted in every result row. Repository detectors use
`contamination=0` because the benchmark consumes their scores rather than their
internal labels. Isolation Forest and LOF use `contamination="auto"`;
One-Class SVM uses the predeclared `nu=0.1`. No value is inferred from
`y_test`.

## Dataset modes

`smoke.json` is fully offline. It fixes sample count, feature count, and
contamination while retaining the historical normal-versus-Gamma/Exponential
construction. Samples are shuffled before splitting.

`core.json` downloads three UCI classification datasets through the optional
`ucimlrepo` dependency:

- [Cardiotocography (UCI 193)](https://archive.ics.uci.edu/dataset/193/cardiotocography):
  pathologic fetal state is mapped to anomaly;
- [SECOM (UCI 179)](https://archive.ics.uci.edu/dataset/179/secom):
  manufacturing failure is mapped to anomaly;
- [Ionosphere (UCI 52)](https://archive.ics.uci.edu/dataset/52/ionosphere):
  a bad radar return is mapped to anomaly.

All three are CC BY 4.0 datasets according to their UCI records. They are
classification-to-anomaly **proxies**, not a complete representation of
real-world anomaly detection. Dataset files are not committed to this
repository. The normalized cache records a canonical SHA-256 and the loader
fails if a locked manifest hash changes.

## Run

The benchmark runner is a repository workflow rather than part of the alpha
wheel. Run the following commands from the repository root.

Install benchmark-only dependencies:

```bash
python -m pip install -e ".[benchmark]"
```

Offline smoke:

```bash
python -m benchmarks.run \
  --config benchmarks/configs/smoke.json \
  --output benchmark_outputs/smoke
```

Public proxy suite:

```bash
python -m benchmarks.run \
  --config benchmarks/configs/core.json \
  --output benchmark_outputs/core \
  --data-cache .benchmark_data
```

Each run writes:

- `config.resolved.json`;
- `run_metadata.json`;
- `metrics.csv`;
- `timings.csv`;
- `summary.csv`;
- `report.md`.

Raw CSV values retain full precision. The Markdown report rounds for reading
only and never labels an overall winner.
