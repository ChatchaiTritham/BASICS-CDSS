# Reproducibility — BASICS-CDSS

This repository is the executable companion to the *Neural Computing and
Applications* manuscript. All reported data are synthetic. The code and results
support an in-distribution methodological evaluation, not clinical validation or
deployment.

## Reproduce the study

```bash
pip install -e .
python scripts/run_all.py
python scripts/generate_results_figures.py
python -m pytest -q
```

### Revision analyses (2026-10-11)

Run after `run_all.py` and `multiseed.py`, from the repository root:

```bash
python scripts/counterfactual_segments.py   # 1-9 h / 9-18 h slopes, paired 18 h - 1 h effect (bootstrap CI)
python scripts/noise_realisations.py        # noise sweep averaged over 30 noise draws per point
python scripts/redundancy_control.py        # tabular on 12-h window, sequence on t=0; 20 seeds
python scripts/onset_sensitivity.py         # mid-horizon deterioration onset cohort; 20 seeds
python scripts/stat_tests.py                # Friedman + Nemenyi, Wilcoxon (Holm) over seeds
python scripts/label_diagnostics.py         # per-disease distribution of simulated p(death)
python scripts/generate_results_figures.py
```

`run_all.py` additionally writes `results/counterfactual_delay_per_twin.csv`; all
previously released `results/` files are unchanged byte-for-byte at seed 42.

The canonical run uses seed 42 and a cohort of 1,000 digital twins: sepsis 400,
ARDS 350, and ACS/cardiac 250. A seeded 60/20/20 permutation produces 600
training, 200 calibration, and 200 held-out test cases. The driver pins numerical
thread counts to one before importing the scientific stack and writes its
machine-readable evidence to `results/`.

## Outcome and model scope

The binary outcome is synthetic mortality, sampled from a literature-anchored
logistic mapping of each twin's terminal cumulative-damage state. Both classes
are present in every disease cohort.

Six model families are trained and scored:

- logistic regression, random forest, gradient boosting, and XGBoost on the
  initial-state feature table;
- LSTM and TCN on each twin's complete 25-step trajectory.

The temporal regime applies the same 20% MCAR missingness and two-times-noise
definition per time step. PyTorch models and XGBoost are seeded, and the driver
uses deterministic execution settings.

## Evidence map

- `results/model_metrics.csv`: static and temporally perturbed AUROC and accuracy.
- `results/calibration.csv`: ECE and Brier scores.
- `results/decision_curve.csv`: decision-curve net benefit.
- `results/conformal.csv`: split-conformal coverage (marginal and Mondrian, clean and
  degraded test data, overall and among deaths) and set size.
- `results/recalibration.csv` and `results/oracle_metrics.csv`: Platt/isotonic
  recalibration on the calibration split, and the simulator's own risk scored as a predictor.
- `results/dbrs.csv`, `results/tcb.csv`, `results/temporal_consistency.csv`, and
  `results/temporal_metrics.csv`: the manuscript's digital-twin robustness
  measures.
- `results/nns.csv`: alert burden (PPV, number needed to alert, % flagged) at p_t = 0.30.
- `results/noise_sensitivity.csv` and `results/masking_sweep.csv`: robustness
  sweeps.
- `results/counterfactual_alignment.csv`: counterfactual alignment and regret
  (common random numbers across actions since v1.4.0).

The NNT table and the fairness demonstration over a fabricated `synthetic_group`
attribute were removed in v1.4.0; they were never reported in the manuscript.
- `results/run_metadata.json` and `results/summary.json`: cohort, split, seed,
  and headline run metadata.

The manuscript tables and narrative have been reconciled to these committed
artifacts. Historical numbers from an earlier external pipeline are not treated
as evidence.

## Interpretation boundaries

- The cohort is entirely simulated and has no real demographic attribute.
  Fairness results demonstrate metric execution only; they are not evidence of
  real-world equity or bias.
- The seeded driver establishes deterministic evidence for the declared
  environment. It does not establish external validity across hospitals,
  devices, populations, or software stacks.
- The experiment is an in-distribution finite-sample evaluation. It does not
  authorize clinical use.

## Data and code availability

The source, synthetic outputs, tests, and reproduction drivers are public in
this repository. An anonymized snapshot should be supplied to reviewers if the
journal applies double-blind review, and a permanent archival DOI should be
minted for the accepted version.

## Revision of 2026-10-04: observation window and masking sweep

Two errors were corrected and every result was regenerated.

1. **Label leakage.** The sequence models had been trained on the full 24-hour trajectory, whose terminal
   state determines the outcome label. `build_trajectory_tensor` now returns only the first
   `OBSERVATION_WINDOW_HOURS = 12` hours (13 hourly rows), with imputation medians computed over that window.
2. **Masking sweep.** `compute_masking_sweep` scored the tabular models on the terminal-timestep row although
   they were trained on the presentation (t = 0) row. Because the gap never touches t = 0, the sweep is now
   reported for the sequence models only.

The regenerated `results/` were produced with Python 3.13.9, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.9.1,
torch 2.9.0+cpu and xgboost 3.1.1 (`requirements-lock.txt`). The earlier environment comparison below refers
to the pre-revision configuration and is kept for the record; the tabular families remain insensitive to the
software stack, while the torch sequence models remain environment-specific.

## Environment sensitivity (added 2026-09-28)

The committed `results/` were produced by the environment recorded in
`results/run_metadata.json` (Python 3.13.14, numpy 2.2.6, pandas 2.3.3, torch 2.9.0+cpu,
xgboost 3.1.1), pinned in `requirements-lock.txt`.

Re-running `scripts/run_all.py` at the same seed under a different stack (Python 3.11.15,
numpy 2.4.6, pandas 3.0.5, torch 2.13.0+cpu, xgboost 3.2.0) reproduces the four tabular
families to the reported precision, but **not** the two torch sequence models:

| quantity (seed 42) | pinned environment | other stack |
| --- | --- | --- |
| LSTM AUROC (static) | 0.873 | 0.897 |
| LSTM ECE (static) | 0.222 | 0.066 |
| LSTM net benefit @0.30 | 0.151 | 0.316 |
| TCN AUROC (static) | 0.920 | 0.919 |
| TCB `delta_min` | 0.00084 | 0.0071 |

The seed fixes the cohort, the split and the initialisations; it does not fix the numerical
path a given torch version takes through training. Any claim about a specific sequence model's
calibration or net benefit is therefore a property of one training run in one environment, not
of the architecture. `scripts/multiseed.py` re-runs the cohort and all six model fits across
many seeds so that between-family differences can be read against that variability, and its
output (`results/multiseed_*.csv`) is what the manuscript relies on for ordering claims.
