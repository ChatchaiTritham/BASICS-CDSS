# Digital Twin Temporal Evaluation of Clinical Decision Support (BASICS-CDSS)

> A seeded, CPU-only stress-test harness that takes candidate clinical risk models through a mechanistic three-disease simulator and reports discrimination against an oracle, calibration before and after recalibration, selective-prediction risk, conformal coverage under degradation and net benefit, over repeated seeds and with input and onset controls.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.13.9%20(pinned)-blue) ![Reproducible](https://img.shields.io/badge/reproducible-seeded-success)

## Overview

Clinical prediction models are usually compared on one discrimination figure computed on complete data. This repository simulates patients with three reduced mechanistic disease models (sepsis, ARDS, acute coronary syndrome), trains six model families on them and evaluates the families under clean and degraded data. Because the simulator knows every patient's true outcome probability, each family can be compared with an oracle, and perturbations and counterfactual treatment schedules can be applied to otherwise identical cohorts.

The simulated patients are **untreated**: the label is a *simulated 24-hour adverse outcome* drawn from each patient's latent damage, not clinical mortality. The simulator is not identified on patient data. Its results describe evaluation methods under controlled conditions; they are not clinical findings.

Every number is computed by the scripts in `scripts/` and written to `results/`; nothing is typed in.

## Key results (release v1.4.0)

Cohort N = 1,000 (sepsis 400 / ARDS 350 / ACS 250), split 60/20/20 (train 600 / calibration 200 / test 200), 20 seeds (42–61).

- **Discrimination vs the oracle** (`results/oracle_gap.csv`). Mean pooled static AUROC 0.919 (LR) to 0.942 (LSTM); oracle 0.954. Within disease the spread is 0.912–0.921 and logistic regression ranks first.
- **Discrimination does not predict calibration** (`results/rank_agreement.csv`). The AUROC leader is the best-calibrated family in 2/20 seeds; chance expectation 3.3 (permutation test). After Platt recalibration on the calibration split: 3/20; isotonic: 1/20. Recalibration halves the ECE spread across families (0.066 → 0.036).
- **Degradation** (20% MCAR + noise; `results/multiseed_summary.csv`). Mean AUROC drop: GB 0.080, XGB 0.065, RF 0.045, LR 0.029, TCN 0.011, LSTM 0.005.
- **Input-redundancy control** (`results/redundancy_ratio.csv`). On the 12-hour window the tree ensembles' loss falls by 38–63%; a TCN given only the presentation state loses 0.048.
- **Conformal coverage under degradation** (`results/multiseed_conformal_summary.csv`). 95% sets calibrated on clean data keep 0.934–0.964 marginal coverage on degraded data, but coverage among deaths is below 0.95 in most seeds for five of six families; Mondrian (class-conditional) calibration reduces the shortfall.
- **Net benefit** over p_t = 0.10–0.40 (`results/decision_curve_range.csv`): every family beats treat-all/treat-none from 0.20 to 0.40; families are indistinguishable from one another; 46–55% of patients flagged at 0.30.
- **Counterfactual timing** (`results/counterfactual_segments.csv`). Paired 18 h − 1 h effect 3.6 percentage points [3.1, 4.2]; slope 0.40 pp/h over 1–9 h, 0.03 over 9–18 h. Far below, and not comparable with, the clinical estimate.
- **Simulator checks** (`results/dt_convergence.csv`, `results/severity_course.csv`, `results/sensitivity_summary.csv`). Step size: with noise off, 99.3% of labels unchanged between Δt = 1 h and 0.25 h. One-at-a-time ±50% changes of six parameters: the AUROC leader is the best-calibrated family in 0–4 of 10 seeds in 12 of 13 settings (6/10 in one); the LSTM loses ≤ 0.014 under degradation in every setting. At 9 h, 74% of untreated sepsis twins still accrue damage; in the delay sweep all 120 do at every checked hour.

## Repository structure

```
src/basics_cdss/      package: temporal (simulator, disease models, counterfactual), metrics,
                      clinical_metrics, causal, xai, visualization, governance
scripts/              run_all.py (seed-42 driver) and the analysis scripts listed below
results/              CSV/JSON written by the scripts (the source of every reported number)
figures/results/      figures rendered from results/*.csv
figures/legacy/       archived illustrative artifacts (not used for any claim)
tests/                unit, smoke and metric tests
examples/             runnable demonstrations
data/, evaluation/    placeholders; the cohort is generated, not loaded
```

## Installation

```bash
git clone https://github.com/ChatchaiTritham/BASICS-CDSS.git
cd BASICS-CDSS
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt                # pinned stack (Python 3.13.9)
pip install -e .
```

## Reproducing the results

Run the scripts in this order (or `bash scripts/run_revision.sh`, which runs them all and prints the wall time of each). Times are wall-clock on one core of a commodity Windows 11 x86-64 CPU, Python 3.13.9, pinned stack.

| Step | Script | Writes | Time |
|---|---|---|---|
| 1 | `run_all.py` | seed-42 cohort, model metrics, calibration, recalibration, oracle, AURC, conformal, decision curve, harm, NNA, noise/gap sweeps, delay sweep, alignment, TCS/TCE/DBRS/TCB/BEWS | 2 min |
| 2 | `multiseed.py` | `multiseed_*.csv` (20 seeds: metrics, oracle, recalibration, conformal) | 6 min |
| 3 | `redundancy_control.py` | `redundancy_control_*.csv` | 7.5 min |
| 4 | `onset_sensitivity.py` | `onset_sensitivity_*.csv` | 6 min |
| 5 | `bootstrap_ci.py` | `bootstrap_ci.csv`, `decision_curve_range.csv` | 2 min |
| 6 | `noise_realisations.py` | `noise_sensitivity_realisations.csv` | 0.5 min |
| 7 | `counterfactual_segments.py` | `counterfactual_segments.csv` (needs step 1) | < 0.1 min |
| 8 | `label_diagnostics.py` | `label_diagnostics.csv` (needs step 1) | < 0.1 min |
| 9 | `simulator_checks.py` | `dt_convergence.csv`, `severity_course.csv` | 0.5 min |
| 10 | `stat_tests.py` | `stat_friedman.csv`, `stat_pairwise.csv`, `rank_agreement.csv`, `oracle_gap.csv`, `redundancy_ratio.csv`, seed summaries (needs 2–4) | 5 min |
| 11 | `sensitivity.py` | `sensitivity_*.csv` | 30 min |
| 12 | `generate_results_figures.py`, `make_extra_figs.py` | `figures/results/*` | < 0.5 min |

```bash
python -m pytest -q    # 130 passed, 4 skipped (Python 3.13.9, pinned stack)
```

Numerical-library threads are pinned to one. On the pinned stack every result file is reproduced byte for byte. On other builds the tabular families reproduce exactly; the two PyTorch models can differ in the fourth decimal.

## Results and figures

Figures are rendered only from `results/*.csv` (provenance in `figures/results/figure_provenance.csv`). The manuscript figures are `fig_multiseed`, `fig_robustness`, `fig_input_control`, `fig_riskcoverage_conformal`, `fig_decision_curve` and `fig_counterfactual_delay`.

Earlier versions of this repository wrote an NNT table and fairness metrics over a fabricated synthetic group attribute. Neither was reported in the manuscript; both were removed in v1.4.0 so that every output file is described in the paper.

## Data

No human subjects and no real patient records are used, so no IRB approval is required. The cohort is generated: 1,000 digital twins (sepsis 400, ARDS 350, ACS 250) evolved over a 24-hour horizon by the committed ODE/SDE disease models, split 60/20/20 (train 600 / calibration 200 / test 200) at seed 42. The twins receive no treatment. The binary label (a simulated 24-hour adverse outcome, not clinical mortality) comes from each twin's terminal cumulative-damage state, mapped through a logistic link and realised as a seeded Bernoulli draw; the twin's probability is kept as the oracle risk. The realised prevalence (reported, not targeted) is 29.8% sepsis, 56.6% ARDS, 27.2% ACS (overall 38.5%).

## Dataset information

No external or clinical dataset is distributed or required. All data are produced by the digital-twin
simulator at seed 42: 1,000 twins over a 24-hour horizon at one-hour resolution, distributed across
sepsis (n = 400), acute respiratory distress syndrome (n = 350) and acute coronary syndrome (n = 250),
each generated by reduced first-order difference-equation models with additive noise (explicit Euler step, Δt = 1 h). Two splits are
produced from the same twins: a clean split and a degraded split with per-timestep
missing-completely-at-random masking and additive noise. The committed outputs of a seeded run are the
tables in `results/` (including `summary.json`, `cohort_summary.json` and `run_metadata.json`, which
records the exact interpreter and library versions). `data/` and `models/` are placeholders.

## Code information

- `src/basics_cdss/` — the package: disease models, twin simulator, temporal metrics, conformal
  prediction, decision-curve and risk-coverage diagnostics, sequence models.
- `scripts/run_all.py` — the deterministic driver (seed 42) that trains the six model families and
  writes every result table.
- `evaluation/`, `experiments/`, `visualizations/` — analysis entry points and figure generators.
- `tests/` — unit tests for the simulator, metrics and model wrappers.

## Requirements

Python 3.10 or newer, no GPU required (results pinned to CPython 3.13.9). Runtime dependencies: NumPy, pandas, SciPy, scikit-learn,
XGBoost (≥ 2.0), PyTorch (≥ 2.0, CPU build is sufficient), Matplotlib, Seaborn, PyYAML, tqdm and
Pydantic (≥ 2); see `requirements.txt`. The reported results were produced with CPython 3.13.9,
NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.9.1, SciPy 1.18.1, XGBoost 3.1.1 and PyTorch 2.9.0+cpu (pinned in `requirements-lock.txt`), as recorded in
`results/run_metadata.json`. `Dockerfile` / `docker-compose.yml` reproduce that stack.

## Methodology

1. Simulate the seeded twin cohort from the mechanistic disease models.
2. Build two evaluation splits from the same twins: clean, and degraded by per-timestep
   missing-completely-at-random masking plus additive noise.
3. Preprocess: split 60/20/20 by a seeded permutation (not stratified); impute disease-disjoint
   missing features with medians of the training split only; standardise features inside the
   logistic-regression pipeline; feed sequence models the first 12 hours (13 steps) with the same
   imputation. No feature selection or resampling is performed.
4. Train six model families (logistic regression, random forest, gradient boosting, XGBoost, LSTM,
   TCN) with fixed, untuned settings.
5. Compute AUROC (pooled and within disease), ECE (10 equal-width bins), calibration slope and
   intercept, Platt/isotonic recalibration on the calibration split, AURC (confidence max(p, 1-p),
   0-1 loss), split-conformal coverage (marginal and Mondrian; clean and degraded test data; overall
   and among deaths), net benefit over p_t = 0.10-0.40, the oracle, DBRS, TCS, and the counterfactual
   delay sweep and action alignment (common random numbers).
6. Write all tables to `results/` and redraw the figures from those tables.

## Usage instructions

```bash
pip install -e .            # install the package
bash scripts/run_revision.sh  # every script in run order (see table above)
pytest -q                   # run the test suite
```

## Citation

```bibtex
@article{tritham_basics_cdss,
  title   = {Digital Twin Simulation for Temporal Evaluation of Clinical Decision Support Systems},
  author  = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  journal = {PeerJ Computer Science},
  note    = {under review; code release v1.4.0},
  year    = {2026}
}
```

## License and contribution guidelines

Released under the MIT License (see `LICENSE`). Contributions are welcome: please read `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` before opening an issue or a pull request.

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

## Relationship to our other repositories

The package under `src/basics_cdss/` is shared infrastructure: the same modules appear in
several of our repositories covering complementary aspects of clinical decision support
evaluation. Sharing it keeps one implementation of the common metric and simulation code
rather than four divergent copies.

What belongs to **this** study is the driver, the computed results and the figures:

- `scripts/run_all.py` — seeded driver
- `results/` — the computed metric tables
- `RECONCILIATION_TABLE.md` — historical audit of the first manuscript version (superseded)

Anything under `src/basics_cdss/` that this study does not exercise is inherited from the
shared package and is not part of the results reported here.
