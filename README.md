# Digital Twin Temporal Evaluation of Clinical Decision Support (BASICS-CDSS)

> A seeded, CPU-only pipeline that stress-tests clinical risk models on simulated patient trajectories and shows where accuracy hides temporal failure that calibration, selective prediction, and decision-curve analysis expose.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Reproducible](https://img.shields.io/badge/reproducible-seed--42-success)

## Overview

A model that scores well on a frozen retrospective split often disappoints at the bedside. The usual culprit is the evaluation itself: a single-timepoint test feeds the model complete, simultaneously available inputs, which is not how clinical data actually arrive, and it never asks whether a recommendation would have helped more had it been issued earlier. This repository takes a different route. It builds patient digital twins from mechanistic ODE/SDE disease models for sepsis, ARDS, and acute coronary syndrome, then evaluates ordinary scikit-learn classifiers on those trajectories under both clean and degraded data conditions.

The point of the simulator is mechanistic grounding rather than realism for its own sake. Because each twin's latent physiological state is known at every step, we can read off a ground-truth risk trajectory, run exact counterfactuals on intervention timing, and ablate one biomarker at a time without recruiting a single patient. That access is what lets the code measure things a static benchmark cannot — a trajectory-level calibration gap, the stability of a recommendation as data go missing, and the reliability of individual markers against a known answer.

What the code reports, it computes. Three classifier families are trained and scored end to end; nothing is typed in to match a target. Deep sequence models and gradient-boosting libraries outside the declared dependency set are deliberately absent, and the parts of the manuscript story the committed code cannot regenerate are documented rather than faked (see `REPRODUCIBILITY.md`).

## Key results

All figures below are read from `results/` after `python scripts/run_all.py` at seed 42 (synthetic cohort, in-distribution, n = 100 held-out test split):

- **Accuracy is a poor robustness signal.** Under the degraded ("temporal") regime, held-out accuracy fell 4 points for logistic regression and 3 for gradient boosting but *rose* 3 points for random forest (90→86, 86→83, 87→90), so an accuracy-only readout would have misjudged at least one model.
- **Discrimination and calibration degrade everywhere.** AUROC dropped by 0.060 (LR), 0.034 (RF), and 0.044 (GB); expected calibration error rose from a 0.058–0.107 static range to 0.110–0.135 temporal.
- **A leaderboard reversal.** Random forest led on static AUROC (0.936) yet its ECE more than doubled under perturbation (0.058→0.122), while logistic regression — weakest on AUROC — gave the lowest risk–coverage area (AURC 0.110) and the highest decision-curve net benefit at the 0.30 threshold (0.229).
- **Selective prediction stays valid.** Split-conformal coverage matched its 95% target (0.94 LR, 0.95 RF/GB) with average set size 1.13–1.23; the Temporal Coverage Bound held in 100% of configurations (δ_min = 0.013 at ρ = 0.20, Brier units).
- **DBRS recovers disease-specific structure.** Across 48 disease×biomarker×model configurations (mean 1.58, range 0.93–28.0) only two markers cleared the critical threshold of 1.15, both cardiac: ST-elevation under LR (28.0, on n = 24 cardiac twins) and heart rate under GB (1.69).
- **Counterfactual direction reproduces; magnitude is shallow.** The seeded antibiotic-delay sweep is monotone (mortality probability 0.354→0.390 across 1–18 h; slope 0.22 pp/h, R² = 0.81). This is far below the ~7.6%/h clinical estimate because the simulator's organ-damage state saturates within 24 h — reported honestly, not tuned.
- **BEWS is degenerate here (caveat).** Failure status is fixed largely at presentation, so the early-warning lead time saturates at the full 24 h horizon for every marker; we report it as exploratory rather than a graded lead-time claim.

## Repository structure

```
src/basics_cdss/      package: temporal (digital twin, disease models, counterfactual),
                      metrics, clinical_metrics, causal, xai, visualization, governance
scripts/              run_all.py (driver) + generate_results_figures.py (figures from results/)
results/              CSV/JSON written by run_all.py (the source of every reported number)
figures/results/      figures rendered directly from results/*.csv (the only figures used for claims)
figures/legacy/       archived illustrative artifacts (not used for headline claims)
tests/                unit, smoke, and metric checks
examples/             runnable demonstrations
data/, evaluation/    placeholders; the cohort is generated, not loaded
```

## Installation

```bash
git clone https://github.com/ChatchaiTritham/BASICS-CDSS.git
cd BASICS-CDSS
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Reproducing the results

```bash
python scripts/run_all.py                  # seed 42; runs on a commodity CPU in minutes
python scripts/generate_results_figures.py # renders figures/results/* from results/*.csv
python -m pytest -q                        # 115 passed, 4 skipped
```

`run_all.py` writes the cohort (`cohort.csv`, `cohort_summary.json`), the static-vs-temporal model table (`model_metrics.csv`, `calibration.csv`, `coverage_risk.csv`, `harm.csv`), the conformal and decision-curve outputs (`conformal.csv`, `decision_curve.csv`), the counterfactual sweep (`counterfactual_delay.csv`), the temporal constructs (`dbrs.csv`, `tcb.csv`, `temporal_consistency.csv`, `temporal_metrics.csv`, `bews.csv`), and a `summary.json` headline roll-up. The driver pins numerical-library thread counts to one before importing the scientific stack; with that in place, two runs from a clean checkout — even on different hosts — produce byte-identical files. The one stochastic path in the simulator (its unseeded `stochastic=False` branch) is avoided by the driver, which drives every twin on the seeded `stochastic=True` path.

## Results and figures

The honest, results-driven figure set lives in `figures/results/` and is regenerated by `generate_results_figures.py`, which reads only `results/*.csv` (provenance in `figures/results/figure_provenance.csv`):

- `figures/results/fig_auroc_static_vs_temporal.png` — paired AUROC bars per model. Every bar drops from static to temporal; the gap is largest for LR and smallest for RF, the visual form of the discrimination decay.
- `figures/results/fig_calibration_ece.png` — ECE by model and regime. All three rise under perturbation; RF's near-doubling against its top static AUROC is the leaderboard reversal at a glance.
- `figures/results/fig_decision_curve.png` — net benefit at the 0.30 threshold vs. each model's maximum. LR leads at 0.30 despite the weakest AUROC, separating clinical utility from discrimination.
- `figures/results/fig_conformal_coverage.png` — empirical vs. target conformal coverage. Markers sit on the 0.90/0.95 reference lines, confirming coverage validity under both regimes.
- `figures/results/fig_counterfactual_delay.png` — terminal harm vs. antibiotic delay over 120 seeded sepsis twins. The curve rises monotonically then flattens, the saturation that makes the fitted slope shallow.

The earlier hardcoded figure scripts (`generate_baseline_metrics.py`, `generate_paper1_figures*.py`) and the curated `figures/manuscript/` set they fed have been removed: they baked in display literals (including XGBoost/LSTM/TCN AUROC values of 0.873/0.891/0.887 for models with no implementation in this package) over runtime-synthesised inputs. Only `figures/results/`, rendered from `results/*.csv`, is used for any claim.

## Data

No human subjects and no real patient records are used, so no IRB approval is required. The cohort is generated: 500 digital twins (sepsis 200, ARDS 175, ACS 125) evolved over a 24-hour horizon by the committed ODE/SDE disease models. The binary mortality label comes from each twin's terminal cumulative-damage state, mapped through a literature-anchored logistic link and realised as a seeded Bernoulli draw, which keeps the task learnable but not perfectly separable. The realised prevalence (reported, not targeted) is 31.5% sepsis, 53.1% ARDS, 21.6% cardiac.

## Citation

```bibtex
@article{tritham_basics_cdss,
  title   = {Digital Twin Simulation for Temporal Evaluation of Clinical Decision Support Systems},
  author  = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  journal = {Neural Computing and Applications},
  note    = {to appear},
  year    = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).

## Contact

**Chatchai Tritham** — Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand. Email: chatchait66@nu.ac.th · ORCID: 0000-0001-7899-228X
**Chakkrit Snae Namahoot** — same affiliation. Email: chakkrits@nu.ac.th · ORCID: 0000-0003-4660-4590

## Portfolio relationship

| Repository | Role |
|---|---|
| BASICS-CDSS | Digital-twin temporal evaluation of CDSS (this repository) |
| TRI-X | Framework-level package |
| ORASR | Routing and safety-action component |
| DRAS-5 | Dynamic risk-state component |
| SAFE-Gate | Safety-gated ensemble framework |
| SynDX | Synthetic validation and explainability evidence |
| SURgul | SRGL/governance reproducibility component |
| Selective-CDSS | Risk-controlled selective-prediction (abstention) component |
| Causal-CDSS | Causal-inference evaluation component |
| Beyond-Accuracy | Simulation-based safety/calibration evaluation framework |
