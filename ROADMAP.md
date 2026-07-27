# Revival Roadmap

This roadmap keeps the original handwritten project recognizable while making future changes reproducible and reviewable. Each phase is intended to be a separate pull request or a small series of focused pull requests.

## Principles

- Preserve the existing Git history and historical learning material.
- Do not mass-format or rewrite legacy code.
- Lock current behavior with characterization tests before changing it.
- Distinguish learning examples, experimental research, and supported interfaces.
- Treat benchmark conclusions as conditional on data, labels, thresholds, and evaluation design.
- Keep deep-learning dependencies optional.

## Phase 1 — Restore the project front door

- Make the English README the default international entry.
- Preserve the complete Chinese README as `README.zh-CN.md`.
- Add language navigation, a maintenance notice, an algorithm-status map, and cautious evidence wording.
- Add machine-readable repository citation metadata.
- Add the maintainer-approved MIT License.

## Phase 2 — Characterize historical behavior

- Add a reproducible Python dependency definition.
- Port the existing smoke tests for Mahalanobis, PCA reconstruction error, Kernel PCA reconstruction error, RobustPCC, and evaluation helpers.
- Reproduce and fix the ambiguous `coverage` import reported in issue 2.
- Add tests for labels, contamination, thresholds, random state, invalid input, and singular matrices.

## Phase 3 — Introduce a stable API incrementally

- Add `src/anomaly_detection/` without deleting the historical scripts.
- Standardize `fit`, `score_samples`, and `predict` where the algorithm permits it.
- Use `1` for anomaly and `0` for normal in the public project API.
- Document compatibility wrappers or migration paths for historical entry points.

Initial scope: expose only the handwritten Mahalanobis and linear-PCA
reconstruction calculations. RobustPCC and Kernel PCA remain outside the stable
surface until their multi-threshold and compatibility behavior is characterized.

## Phase 4 — Rebuild the benchmark

- Separate supervised evaluation from unsupervised detector agreement.
- Add a small curated set of public labeled datasets and explicit anomaly types.
- Report ROC AUC, PR AUC, F1, recall, runtime, and memory where applicable.
- Use repeated seeds and report variability.
- Prevent threshold or algorithm selection on the final test set.

## Phase 5 — Add representative modern methods

- Add One-Class SVM as a missing boundary-based baseline.
- Add ECOD as an interpretable empirical-tail method.
- Require English and Chinese explanations, tests, examples, and benchmark entries for every new method.

## Phase 6 — Add optional deep and semi-supervised methods

- Add an Autoencoder baseline and Deep SVDD behind an optional deep-learning dependency.
- Evaluate Deep SAD as a modern continuation of the semi-supervised track.
- Consider Deep Isolation Forest only after the benchmark and maintenance cost are understood.

## First release criteria

- A new reader can find the correct language entry and run at least one documented example.
- Historical files remain traceable and are not replaced by a bulk rewrite.
- Core behavior is covered by tests on the supported Python versions.
- Benchmark claims are reproducible and appropriately scoped.
- KADOA's experimental status, lineage, citation, and evidence limits are explicit.
- A maintainer-approved license is present.
