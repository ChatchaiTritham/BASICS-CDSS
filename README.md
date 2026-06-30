# Digital Twin Temporal Evaluation of Clinical Decision Support (BASICS-CDSS)

> A seeded, CPU-only pipeline that stress-tests clinical risk models on simulated patient trajectories and shows where accuracy hides temporal failure that calibration, selective prediction, and decision-curve analysis expose.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Reproducible](https://img.shields.io/badge/reproducible-seed--42-success)

## Overview

A model that scores well on a frozen retrospective split often disappoints at the bedside. The usual culprit is the evaluation itself: a single-timepoint test feeds the model complete, simultaneously available inputs, which is not how clinical data actually arrive, and it never asks whether a recommendation would have helped more had it been issued earlier. This repository takes a different route. It builds patient digital twins from mechanistic ODE/SDE disease models for sepsis, ARDS, and acute coronary syndrome, then evaluates ordinary scikit-learn classifiers on those trajectories under both clean and degraded data conditions.

The point of the simulator is mechanistic grounding rather than realism for its own sake. Because each twin's latent physiological state is known at every step, we can read off a ground-truth risk trajectory, run exact counterfactuals on intervention timing, and ablate one biomarker at a time without recruiting a single patient. That access is what lets the code measure things a static benchmark cannot — a trajectory-level calibration gap, the stability of a recommendation as data go missing, and the reliability of individual markers against a known answer.

What the code reports, it computes. All six model families the manuscript headlines are trained and scored end to end — four tabular (logistic regression, random forest, gradient boosting, XGBoost) on the initial-state feature table, and two sequence models (a torch LSTM and a dilated-Conv1d TCN) on the per-twin 24-hour trajectory — all seeded to 42; nothing is typed in to match a target. The parts of the manuscript story the committed code still cannot regenerate (e.g. the calibrated mortality percentages) are documented rather than faked (see `REPRODUCIBILITY.md`).

## Key results

All figures below are read from `results/` after `python scripts/run_all.py` at seed 42 (synthetic cohort N = 1,000: sepsis 400 / ARDS 350 / ACS 250; seeded 60/20/20 split → train 600 / calibration 200 / **test 200**). A complete claim-by-claim reconciliation against the manuscript lives in `RECONCILIATION_TABLE.md`.

- **All six model families are computed (seed 42).** Static AUROC: LR 0.916, RF 0.918, GB 0.917, XGBoost 0.905, LSTM 0.873, TCN 0.920; temporal AUROC: 0.902 / 0.875 / 0.845 / 0.822 / 0.842 / 0.907. At N = 1,000 the tabular families cluster tightly (0.905–0.918); **TCN has the highest static AUROC and LSTM the lowest** — the opposite of the earlier literals, reported as-is.
- **Accuracy is a poor robustness signal.** Under the degraded ("temporal") regime, held-out accuracy fell 8.5 pp (GB) and 8.0 pp (XGB) but *rose* for the sequence models (LSTM 77→81, TCN flat), so an accuracy-only readout misjudges the tree/NN ordering.
- **Calibration, not discrimination, separates the models.** Logistic regression is best-calibrated (static ECE 0.050, the only model under 0.05) despite mid-pack AUROC; the LSTM has strong-ish AUROC but the **worst ECE (0.222)** — a discrimination-vs-calibration dissociation.
- **Selective prediction stays valid.** Split-conformal coverage meets its 95% target for every model (0.945–0.98) with average set size 1.30–1.64; the Temporal Coverage Bound held in 100% of configurations (δ_min = 0.0008 at ρ = 0.20, Brier units).
- **Tree models are NOT the noise-robust ones here.** Under 2× Gaussian noise the gradient-boosting/XGBoost AUROC degrades most (GB −0.13) while the LSTM/TCN sequence models are nearly flat (|Δ| ≤ 0.015) — refuting the intuition that trees tolerate noise better than neural nets on this trajectory task.
- **Clinical-impact metrics are computed end to end.** NNT (with bootstrap 95% CI), number-needed-to-screen @ 0.30, decision-curve net benefit, fairness across a clearly-labelled *synthetic* group attribute, and counterfactual alignment/regret are all written to `results/*.csv`; see `RECONCILIATION_TABLE.md` for the verbatim numbers and where they refute the manuscript narrative.
- **Counterfactual direction reproduces; magnitude is shallow.** The seeded antibiotic-delay sweep is monotone (mortality probability 0.354→0.390 across 1–18 h; slope 0.22 pp/h, R² = 0.81), far below the ~7.6%/h clinical estimate because the simulator's organ-damage state saturates within 24 h — reported honestly, not tuned.
- **BEWS lead time saturates (caveat).** Failure status is fixed largely at presentation, so the early-warning lead time saturates near the 24 h horizon for most markers (precision range 0.59–0.90); we report it as exploratory rather than a graded lead-time claim.

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

`run_all.py` writes the cohort (`cohort.csv`, `cohort_summary.json`), the static-vs-temporal model table (`model_metrics.csv`, `calibration.csv`, `coverage_risk.csv`, `harm.csv`), the conformal and decision-curve outputs (`conformal.csv`, `decision_curve.csv`), the counterfactual sweep (`counterfactual_delay.csv`), the temporal constructs (`dbrs.csv`, `tcb.csv`, `temporal_consistency.csv`, `temporal_metrics.csv`, `bews.csv`), the clinical-impact + robustness analyses (`nnt.csv`, `nns.csv`, `fairness.csv`, `noise_sensitivity.csv`, `masking_sweep.csv`, `counterfactual_alignment.csv`), and a `summary.json` headline roll-up. The driver pins numerical-library thread counts to one before importing the scientific stack; with that in place, two runs from a clean checkout — even on different hosts — produce byte-identical files. The one stochastic path in the simulator (its unseeded `stochastic=False` branch) is avoided by the driver, which drives every twin on the seeded `stochastic=True` path.

## Results and figures

The honest, results-driven figure set lives in `figures/results/` and is regenerated by `generate_results_figures.py`, which reads only `results/*.csv` (provenance in `figures/results/figure_provenance.csv`):

- `figures/results/fig_auroc_static_vs_temporal.png` — paired AUROC bars per model. Every bar drops from static to temporal; the gap is largest for LR and smallest for RF, the visual form of the discrimination decay.
- `figures/results/fig_calibration_ece.png` — ECE by model and regime. All three rise under perturbation; RF's near-doubling against its top static AUROC is the leaderboard reversal at a glance.
- `figures/results/fig_decision_curve.png` — net benefit at the 0.30 threshold vs. each model's maximum. LR leads at 0.30 despite the weakest AUROC, separating clinical utility from discrimination.
- `figures/results/fig_conformal_coverage.png` — empirical vs. target conformal coverage. Markers sit on the 0.90/0.95 reference lines, confirming coverage validity under both regimes.
- `figures/results/fig_counterfactual_delay.png` — terminal harm vs. antibiotic delay over 120 seeded sepsis twins. The curve rises monotonically then flattens, the saturation that makes the fitted slope shallow.

The earlier hardcoded figure scripts (`generate_baseline_metrics.py`, `generate_paper1_figures*.py`) and the curated `figures/manuscript/` set they fed have been removed: they baked in display literals (including XGBoost/LSTM/TCN AUROC values of 0.873/0.891/0.887) over runtime-synthesised inputs. Those three models are now genuinely trained by `run_all.py` (XGBoost on the tabular features; LSTM and TCN as torch sequence models over the trajectory tensor), so their rows are computed rather than typed in — and the computed values differ from those literals (at the canonical N = 1,000, TCN has the highest static AUROC and LSTM the lowest; see `REPRODUCIBILITY.md` and `RECONCILIATION_TABLE.md`). Only `figures/results/`, rendered from `results/*.csv`, is used for any claim.

## Data

No human subjects and no real patient records are used, so no IRB approval is required. The cohort is generated: 1,000 digital twins (sepsis 400, ARDS 350, ACS 250) evolved over a 24-hour horizon by the committed ODE/SDE disease models, split 60/20/20 (train 600 / calibration 200 / test 200) at seed 42. The binary mortality label comes from each twin's terminal cumulative-damage state, mapped through a literature-anchored logistic link and realised as a seeded Bernoulli draw, which keeps the task learnable but not perfectly separable. The realised prevalence (reported, not targeted) is 29.8% sepsis, 56.6% ARDS, 27.2% cardiac (overall 38.5%). The cohort also carries a clearly-labelled **synthetic** `synthetic_group` attribute used only to demonstrate the fairness metrics — it is fabricated and carries no clinical meaning (see `REPRODUCIBILITY.md`).

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
