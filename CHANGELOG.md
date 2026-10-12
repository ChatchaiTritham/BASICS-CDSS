# Changelog

All notable changes to this repository are documented here.

## Unreleased

## v1.5.0 (2026-10-12)

Round-2 revision. No existing result value changed; every previously committed result file
regenerates byte for byte, and the changes below only add rows, columns or files.

- `stat_tests.py`: Friedman test and Holm-adjusted Wilcoxon tests on mean within-disease static
  AUROC (new rows in `stat_friedman.csv` and `stat_pairwise.csv`).
- `simulator_checks.py`: severity course for every antibiotic-delay arm (1-18 h), plus the share of
  twins whose cumulative damage grew since the previous checked hour (`severity_course.csv`).
- `counterfactual_segments.py`: paired difference of each delay arm from the 1-h arm with
  twin-bootstrap intervals (`counterfactual_paired_diff.csv`).
- `run_all.py`: regret quartiles, share of regret > 1 and maximum regret in
  `counterfactual_alignment.csv`.
- Figures: the counterfactual figure now plots the paired difference from 1 h; outcome labels say
  "with outcome" instead of "deaths"; risk-coverage legend says Clean/Degraded.
- `requirements-lock.txt` pins SciPy 1.18.1. README describes the disease models as explicit-Euler
  difference equations with additive noise. Package version set to 1.5.0 to match the release tag.

## v1.4.0 (2026-10-12)

Revision answering a simulated ten-reviewer PeerJ CS review. All results were regenerated on the
pinned stack (CPython 3.13.9); seed-42 and 20-seed numbers changed because of the imputation fix.

- **Imputation leakage fixed.** Disease-disjoint features are imputed with training-split medians
  only (`split_indices`, `impute_table`, `impute_tensor`, `feature_matrix`); the degraded regime
  uses training-split medians and SDs (`ref=` in `_degrade`, `_degrade_sequence`, `_add_noise`).
  `cohort.csv` now keeps NaN for disease-disjoint features.
- **Counterfactual alignment uses common random numbers** (one noise seed per twin for every
  action); reports the share of twins for which each action is optimal.
- New metrics: calibration slope/intercept (`calibration_slope_intercept`), exact AURC with
  softmax-response confidence (`aurc_confidence`, replaces the threshold-grid AURC in the driver),
  within-disease AUROC, oracle metrics (simulator's own risk), Platt/isotonic recalibration on the
  calibration split, conformal coverage on clean and degraded data, overall and among deaths,
  marginal and Mondrian.
- `multiseed.py` writes oracle, recalibration and conformal files; `stat_tests.py` drops the
  Nemenyi test, adds effect sizes (median difference with bootstrap CI, rank-biserial r), a
  permutation test of AUROC/ECE rank agreement, ratio-of-drops CIs and oracle gaps.
- `bootstrap_ci.py` adds net benefit over p_t = 0.10-0.40 with paired CIs and % flagged.
- New: `sensitivity.py` (one-at-a-time +/-50% parameter sweep), `simulator_checks.py` (step-size
  convergence, untreated severity course), `run_revision.sh` (run order with timings).
- Removed: NNT and synthetic-group fairness outputs (`nnt.csv`, `fairness.csv`, `synthetic_group`).
- README rewritten from the current results; run table with run times; tests for the new metrics.


### Revision 2026-10-11 (manuscript revision analyses)

- `run_all.py`: emits per-twin antibiotic-delay sweep (`counterfactual_delay_per_twin.csv`);
  BEWS computation factored into `_bews_from_groups`; `_initial_state` accepts
  `force_worsening` (default unchanged). Existing outputs are byte-identical.
- New: input-redundancy control (`redundancy_control.py`), mid-horizon-onset
  sensitivity (`onset_sensitivity.py`), counterfactual segment slopes
  (`counterfactual_segments.py`), noise-realisation averaging (`noise_realisations.py`),
  across-seed rank tests (`stat_tests.py`), label diagnostics (`label_diagnostics.py`),
  shared helper `revision_common.py`.
- Figures: counterfactual figure shows segment fits; robustness figure averages 30
  noise draws; new `fig_input_control`.

- Standardized root metadata, authorship, reference, testing, and container files.
- Verified local tests and cleaned generated cache or coverage artifacts before publishing.
