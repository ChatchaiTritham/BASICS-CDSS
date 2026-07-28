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
- `results/conformal.csv`: split-conformal coverage and set size.
- `results/dbrs.csv`, `results/tcb.csv`, `results/temporal_consistency.csv`, and
  `results/temporal_metrics.csv`: the manuscript's digital-twin robustness
  measures.
- `results/nnt.csv` and `results/nns.csv`: clinical-impact summaries.
- `results/noise_sensitivity.csv` and `results/masking_sweep.csv`: robustness
  sweeps.
- `results/counterfactual_alignment.csv`: counterfactual alignment and regret.
- `results/fairness.csv`: a methodology demonstration using a fabricated
  `synthetic_group` attribute.
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
