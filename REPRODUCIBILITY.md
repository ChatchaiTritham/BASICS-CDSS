# Reproducibility Report — BASICS-CDSS

This document records, honestly, what the committed code in this repository can
and cannot reproduce of the manuscript's headline numbers. Nothing here is tuned
to match the manuscript. All values below come from the seeded driver
`scripts/run_all.py` (seed = 42), which builds a digital-twin cohort with the
committed disease models and evaluates it with the committed metric modules.

## How to reproduce

```bash
pip install -e .
python scripts/run_all.py             # writes results/*.csv|json (deterministic, seed=42)
python scripts/generate_results_figures.py   # figures/results/*.png|pdf from results/
python -m pytest -q                   # 115 passed, 4 skipped
```

Determinism was verified by running the driver twice from a clean environment:
every file in `results/` is byte-identical across runs. The driver pins
numerical-library thread counts to one (`OMP_NUM_THREADS` etc.) before importing
the scientific stack — without this, gradient-boosting's float reductions are
order-sensitive and its ECE/Brier jitter at the third decimal across hosts.
With the pin in place the whole pipeline is byte-for-byte reproducible at seed 42
regardless of machine.

### Cardiac single-class fix (pre-requisite)

An earlier label (`int(instantaneous_severity >= 0.5)`) collapsed the cardiac
cohort to a single outcome class, making AUROC and DBRS undefined for cardiac.
The label is now the **terminal cumulative-damage → calibrated-mortality →
seeded Bernoulli** outcome described below, which yields both classes for all
three diseases (realised prevalence: sepsis 31.5%, ARDS 53.1%, cardiac 21.6%).
Per-disease DBRS is therefore computable for cardiac, and the manuscript reports
per-disease reliability without scoping to sepsis only.

## Cohort label: calibrated mortality from terminal cumulative damage

The binary cohort outcome is **mortality**, derived from each twin's terminal
cumulative irreversible-damage state (Component A): sepsis `_organ_damage`, ARDS
`_lung_damage`, cardiac `_infarct`. Each terminal damage is mapped to a
probability of death through a literature-anchored logistic link
(`damage_to_mortality` / `lung_damage_to_mortality` / `infarct_to_mortality` in
`temporal/counterfactual.py`) and the label is a **seeded Bernoulli draw** from
that probability (`outcome ~ Bernoulli(p_death)`), so the task is learnable but
not perfectly separable. No parameter is tuned to a target prevalence; the
emergent per-disease mortality mix is **reported** in `cohort_summary.json`
(`prevalence_by_disease`).

This replaces an earlier degenerate label, `int(instantaneous_severity >= 0.5)`,
which had two integrity defects:
1. **Cardiac was single-class (all 125 twins outcome = 1).** `CardiacEventModel`
   re-derived ischemia from its own downstream markers each step
   (`ischemia <- troponin/ST`, whose targets were ~2.5x ischemia), a
   self-amplifying loop that drove every cardiac twin -- regardless of
   presentation -- to full occlusion within a few hours. AUROC needs both
   classes, so cardiac DBRS was undefined and cardiac TCS collapsed to a
   degenerate 1.0. Fixed by making ischemia a **carried latent state** (seeded
   from presentation at t=0, then evolving by its own slow natural history and by
   reperfusion); troponin/ST are downstream readouts only.
2. **ARDS was near-degenerate (~5.7% positive)** because the instantaneous lung-
   injury signal relaxes toward resolution; the cumulative `_lung_damage` state
   integrates the injury window instead.

Emergent prevalence (seed 42, N = 1,000, reported not targeted): sepsis 0.298,
ARDS 0.566, cardiac 0.272 (overall 0.385). Both classes are present in every
disease cohort; cardiac DBRS is defined and cardiac TCS is non-degenerate. The
sepsis antibiotic-delay counterfactual is unchanged by this fix (it uses
`damage_to_mortality` on sepsis organ damage, untouched).

## Cohort design and split (canonical experiment)

The released run **is** the manuscript's stated experiment. The cohort is
**N = 1,000** digital twins (sepsis 400, ARDS 350, ACS/cardiac 250). The seeded
60/20/20 permutation split (seed 42) is **train = 600, calibration = 200,
test = 200**; all AUROC / accuracy / ECE / NNT / fairness / sweep numbers are
computed on the 200-case held-out test set (see `run_metadata.json` ->
`cohort_design`, `split`). The pipeline scales linearly if a different cohort
size is desired.

## Synthetic demographic attribute (fairness demonstration ONLY)

The cohort carries a `synthetic_group` column with values `A`/`B`. **It is
fabricated for fairness-methodology demonstration only** — the digital-twin
simulator has no real demographics, so this attribute carries no clinical meaning
and is not a real protected attribute. It is drawn on a dedicated seeded RNG
(`SEED + 104729`) as `Bernoulli(p)` where `p` tilts mildly with the twin's latent
mortality risk, so the two groups differ in base rate and the fairness metrics
have a non-trivial (but entirely synthetic) disparity to surface. The fairness
numbers in `fairness.csv` therefore demonstrate that the committed fairness
module runs end to end; they are **not** a claim about real-world bias.

## What the committed code genuinely supports

- A digital-twin cohort generator: the ODE-style `SepsisModel`,
  `RespiratoryDistressModel`, and `CardiacEventModel` in
  `src/basics_cdss/temporal/disease_models.py`, driven by `PatientDigitalTwin`.
- Real metric implementations used as-is by the driver:
  performance/AUROC (`metrics/performance.py`), calibration/ECE/Brier
  (`metrics/calibration.py`), coverage-risk/AURC (`metrics/coverage_risk.py`),
  harm-aware metrics (`metrics/harm.py`), decision-curve net benefit
  (`clinical_metrics/utility_metrics.py`), split-conformal coverage
  (`clinical_metrics/conformal_prediction.py`).
- All six manuscript model families, now trained and scored end to end by the
  driver:
  - **Tabular** (initial-state feature table): logistic regression, random
    forest, gradient boosting (scikit-learn) and **XGBoost**
    (`xgboost.XGBClassifier`, seed 42, `tree_method="hist"`, single-threaded).
  - **Sequence** (per-twin 24-hour trajectory tensor, shape `n × T × d`):
    **LSTM** (1 layer, hidden 64) and **TCN** (3-block dilated causal Conv1d
    stack, 64 channels), both implemented in `torch` and seeded to 42
    (`basics_cdss/temporal/sequence_models.py`). They consume the full 25-step
    trajectory, not the single initial-state row; the degraded ("temporal")
    regime applies the same 20% MCAR + 2× noise definition *per timestep*.
- **Downstream clinical-impact + robustness analyses**, all computed on the
  held-out test set with the committed metric code (`results/` CSV in brackets):
  - **NNT** per model, overall + per disease + per risk stratum, with a seeded
    1000-replicate bootstrap 95% CI (`calculate_nnt`; `nnt.csv`).
  - **Number-needed-to-screen at threshold 0.30** per model
    (`clinical_impact_analysis`; `nns.csv`).
  - **Fairness** across the synthetic group attribute: demographic-parity ratio,
    equalized-odds (TPR/FPR diffs), calibration fairness (`fairness_metrics`;
    `fairness.csv`).
  - **Noise-sensitivity sweep**: per-column Gaussian noise at 1×/2×/3× the
    baseline plus variance-matched Student-t(ν=3) heavy-tailed noise
    (`noise_sensitivity.csv`).
  - **Temporal-masking sweep**: contiguous LOCF-imputed gaps of 1 h / 2 h / 4 h
    (`masking_sweep.csv`).
  - **Counterfactual alignment + regret** per model on the held-out sepsis test
    twins, using the committed `CounterfactualEvaluator` harm function over a
    discrete antibiotic-timing action set on the seeded stochastic path
    (`counterfactual_alignment.csv`).

## What the committed code does NOT support

- **No calibrated mortality model.** The counterfactual harm function is a
  severity-based score, not a probability of death, so the 12.3 / 27.1 / 41.2 %
  mortality trajectory cannot be reproduced as percentages.
- **No driver previously existed.** Before this work, `data/ evaluation/
  experiments/ models/` held only `.gitkeep`; nothing regenerated the headline
  tables, and the figure scripts plotted hardcoded literals over runtime-
  synthesised data.

## Manuscript headline claim vs computed result

`ms` = value in the manuscript; `code` = value computed by `run_all.py`.
"Not implemented" = no committed code path produces the quantity.

The manuscript (`Journals/NCA/sn-article.tex`) has been reconciled to the values
below; the table now records the computed value and the manuscript's matching
claim. All `code value` entries are read from `results/*.csv` produced by the
current `run_all.py`.

All six model families (LR / RF / GB / XGBoost / LSTM / TCN) are trained and
scored by the driver on the **canonical N = 1,000 cohort, 200-case test set**, so
every model row is computed, not assumed. The earlier manuscript headline values
(static AUROC LR 0.812 / RF 0.856 / XGB 0.873 / LSTM 0.891 / TCN 0.887) are
**superseded** by the computed values below; the `.tex` should be reconciled to
these. The complete claim-by-claim reconciliation table is maintained separately
in `RECONCILIATION_TABLE.md`.

Order in cells: LR / RF / GB / XGBoost / LSTM / TCN.

| Metric | code value (results file) | manuscript | status |
|---|---|---|---|
| Static AUROC | 0.916 / 0.918 / 0.917 / 0.905 / 0.873 / 0.920 (`model_metrics.csv`) | Tables 1–2 | computed; **TCN best, LSTM worst** at N=1000 |
| Temporal AUROC | 0.902 / 0.875 / 0.845 / 0.822 / 0.842 / 0.907 (`model_metrics.csv`) | Tables 1–2 | computed; **TCN/LR most robust** |
| ΔAUROC static→temporal | −0.014 / −0.043 / −0.071 / −0.083 / −0.032 / −0.014 | Results | all degrade; trees degrade MOST, not least |
| Static accuracy | 0.85 / 0.835 / 0.825 / 0.815 / 0.77 / 0.835 (`model_metrics.csv`) | Table 1 | computed |
| Temporal accuracy | 0.825 / 0.79 / 0.74 / 0.735 / 0.81 / 0.835 (`model_metrics.csv`) | Table 1 | computed; LSTM/TCN accuracy *rises* |
| ECE static range | 0.050 (LR) – 0.222 (LSTM) (`calibration.csv`) | Table 1 / Results | computed; only LR clears <0.05 |
| ECE temporal range | 0.052 (LR) – 0.209 (LSTM) (`calibration.csv`) | Table 1 / Results | computed; LR best, LSTM worst |
| Counterfactual delay→mortality | 0.354→0.390 prob; slope 0.216 pp/hr, R²=0.806 (`counterfactual_delay.csv`) | Results / Fig 3 | monotone direction reproduces; magnitude shallow |
| DCA net benefit @ 0.30 | 0.311 / 0.305 / 0.307 / 0.298 / 0.151 / 0.301 (`decision_curve.csv`) | Results / Fig 4 | reported as-is |
| Conformal coverage @95% | 0.98 / 0.95 / 0.96 / 0.96 / 0.945 / 0.97 (`conformal.csv`) | Results | coverage valid (≥0.95) |
| Avg prediction-set size @95% | 1.64 / 1.295 / 1.32 / 1.37 / 1.635 / 1.36 (`conformal.csv`) | Results | reported as-is |
| DBRS | 64 evaluated (LR/RF/GB/XGB), 3 critical; mean 1.02 (`dbrs.csv`, `summary.json`) | Results | reported as-is |
| TCB δ_min @ ρ=0.20 | 0.0008, bound holds 100% (`tcb.csv`) | Prop. 1 / Results | computed across all 6 models |
| TCS mean by missingness (0.2/0.4/0.6) | 0.808 / 0.755 / 0.772 (`temporal_consistency.csv`) | Results | reported as-is |
| TCE overall mean | 0.334 (`temporal_metrics.csv`) | Results | reported as-is |
| NNT (overall, per model) | LR 1.45, RF 1.56, GB 1.54, XGB 1.57, LSTM 2.48, TCN 1.50 (`nnt.csv`) | Results | computed w/ bootstrap CI |
| NNS @ 0.30 (per model) | 1.30 / 1.44 / 1.41 / 1.38 / 2.48 / 1.31 (`nns.csv`) | Results | computed |
| Fairness (synthetic group) | parity-ratio 0.41–1.00, eq-odds-ratio 0.81–0.99, calib-fair 0.69–0.81 (`fairness.csv`) | Results | SYNTHETIC attribute — demo only |
| Counterfactual alignment / regret | alignment 0.18–0.23; regret 18.3–27.3 (`counterfactual_alignment.csv`) | Results | computed; ranking differs from ms |
| Seed = 42 | enforced in driver (numpy + torch) | Methods | reproduces |

### Reproduces (computed honestly, deterministic at seed 42)
- Full static-vs-temporal AUROC / accuracy / ECE / AURC table for **all six**
  model families on the canonical N=1,000 / test=200 split.
- The accuracy↔calibration dissociation (LR best-calibrated despite mid-pack
  AUROC; LSTM strong-ish AUROC but worst ECE).
- Monotone antibiotic delay→mortality direction; conformal coverage validity;
  decision-curve net benefit; DBRS reliability structure; TCB; TCS; TCE; NNT; NNS;
  fairness (synthetic); noise and masking sweeps; counterfactual alignment/regret.

### Honest divergences from the manuscript narrative (real numbers, N=1,000)
- **"XGBoost highest static / LSTM best temporal"** — REFUTED. At N=1,000 the
  tabular models cluster tightly (0.905–0.918) and **TCN** has the highest static
  AUROC (0.920); **LSTM is the *lowest* (0.873) and worst-calibrated** (ECE 0.222).
- **"Tree models more noise-robust than NNs"** — REFUTED. In `noise_sensitivity.csv`
  GB/XGB degrade most under Gaussian noise (GB AUROC −0.13 at 2×) while the NN
  sequence models (LSTM/TCN) are nearly noise-flat (|Δ|≤0.015); the sequence models
  are the noise-robust ones here.
- **"t(3) ≈ 2× worse than Gaussian"** — partial. Variance-matched t(3) is on the
  whole *milder* than equal-scale Gaussian for most models (heavier tails but the
  same variance), so the manuscript's 2× claim is NOT reproduced.
- **"masking 1 h −1.2%, 2 h −4.7%, 4 h −12.3%, LSTM best gap tolerance"** — the
  LSTM-best-tolerance direction reproduces (LSTM/TCN ≈ 0% degradation), but the
  graded 1/2/4 h ladder does NOT: degradation is flat across gap lengths and large
  for tabular models (see `masking_sweep.csv` caveat below).

### Not claimed (removed from the manuscript rather than fabricated)
| item | why |
|---|---|
| antibiotic-delay 12.3 → 27.1 → 41.2 % / 4.8 pp·hr⁻¹ / R²=0.89 | fabricated; replaced with the computed 0.216 pp·hr⁻¹, R²=0.806 sweep |
| DBRS lactate=1.34 / PF=1.28 / troponin=1.31 | fabricated; replaced with the computed per-configuration DBRS |
| BEWS 2.1–4.2 h lead / precision 0.86–0.91 | lead time saturates at the 23–24 h horizon (presentation-driven label); precision range 0.59–0.90 is computed, lead-time claim not reproduced |
| DIR / fairness real-world bias numbers | only a SYNTHETIC group attribute exists; fairness is a methodology demo, not a real-bias claim |

## Internal manuscript inconsistencies (confirmed from the .tex)

These are inconsistencies **within the manuscript sources** — the code does not
adjudicate them, it only makes the gap visible. The `.tex` was not edited.

1. **Accuracy 87.3 % → 72.1 % (abstract) vs 83.6 % → 74.1 % (results table).**
   `ABSTRACT_HUMANIZED.tex` reports mean accuracy falling 87.3 % → 72.1 %.
   `main.tex` Table 1 (XGBoost row) and body report 83.6 % static → 74.1 %
   temporal (−9.5 pp). The abstract's "87.3" equals the XGBoost static **AUROC**
   (0.873) read as a percentage; it appears nowhere as an accuracy in the body.
   `main.tex`'s own embedded abstract uses the consistent "6.6–9.5 pp" framing,
   so `ABSTRACT_HUMANIZED.tex` disagrees with both the body and the embedded
   abstract.
2. **DCA net benefit 0.28 (Fig 4 caption) vs 0.31 (body).** The body states
   "maximum net benefit 0.31 at threshold 0.35"; the Fig 4 caption states
   "net benefit = 0.28 at threshold 0.30". These sit one paragraph apart with
   overlapping thresholds and read as inconsistent.
3. Additional divergences in `ABSTRACT_HUMANIZED.tex` only (stale abstract):
   optimal antibiotic window "3–6 h" vs body "1–3 h"; "+34 % delayed-intervention
   risk", "+15.2 % accuracy", and "temporal consistency 0.68–0.92" appear in the
   abstract but match no value in the body.

## Root cause

The repository shipped a genuine evaluation **library** (working metrics + a
working ODE digital-twin simulator) but no **experiment driver** and no committed
data or model predictions. The headline tables in the manuscript were produced
by an external, uncommitted pipeline (including deep models that are not part of
this package), so the numbers could not be regenerated from the repository. The
figure scripts then hardcoded those external numbers as display literals and
synthesised plausible-looking inputs at run time, which hid the gap.

`scripts/run_all.py` closes the gap for everything the committed code can
actually compute, and this report documents the parts it cannot — chiefly the
deep-model rows and the calibrated mortality trajectory — without fabricating
them.

A secondary defect: `disease_models.*.evolve()` injects **unseeded** noise when
called with `rng=None` (the path `CounterfactualEvaluator` uses for its
`stochastic=False` "deterministic" comparison), making that path non-reproducible.
The driver avoids it by driving twins on their seeded `stochastic=True` path.

## Recommended manuscript tempers

- Either add the deep-model / XGBoost training code (and dependencies) to the
  repository, or reframe the package as the **evaluation methodology** and
  present the LSTM/TCN/XGBoost numbers as produced by an external modelling
  pipeline that is out of scope for this artifact.
- Reconcile the abstract's 87.3 % → 72.1 % with the body's 83.6 % → 74.1 %;
  retire or rewrite `ABSTRACT_HUMANIZED.tex` so it matches `main.tex`.
- Reconcile the DCA 0.28 (caption) vs 0.31 (body), or state explicitly that 0.28
  is the value at threshold 0.30 and 0.31 is the maximum at threshold 0.35.
- Soften the counterfactual antibiotic-delay claim, or strengthen the committed
  `SepsisModel` so terminal harm depends on intervention timing; as committed,
  the simulator does not produce a delay→harm gradient over a 24 h horizon.
