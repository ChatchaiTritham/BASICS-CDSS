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

Emergent prevalence (seed 42, reported not targeted): sepsis 0.315, ARDS 0.531,
cardiac 0.216. Both classes are present in every disease cohort; cardiac DBRS is
defined and cardiac TCS is non-degenerate (0.84-0.94, std > 0). The sepsis
antibiotic-delay counterfactual is unchanged by this fix (it uses
`damage_to_mortality` on sepsis organ damage, untouched).

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
- Model families the declared dependency set (scikit-learn only) can train:
  logistic regression, random forest, gradient boosting.

## What the committed code does NOT support

- **No deep sequence models.** The manuscript's headline rows (LSTM, TCN) and
  XGBoost have no committed training code, and neither `torch`, `tensorflow`,
  nor `xgboost` is in `requirements.txt` / `pyproject.toml`. The driver therefore
  cannot reproduce the LSTM 0.891 / TCN 0.887 / XGBoost 0.873 AUROC values; it
  reports the model families the package actually ships.
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

| Metric | code value (results file) | manuscript | status |
|---|---|---|---|
| Static AUROC (LR / RF / GB) | 0.875 / 0.936 / 0.905 (`model_metrics.csv`) | Tables 1–2 | reported as-is |
| Temporal AUROC (LR / RF / GB) | 0.815 / 0.902 / 0.861 (`model_metrics.csv`) | Tables 1–2 | reported as-is |
| AUROC drop static→temporal | −0.060 / −0.034 / −0.044 | Results | every model degrades; reported as-is |
| Accuracy static→temporal | LR 90→86, RF 87→90, GB 86→83 | Table 1 | reported as-is; **accuracy is non-monotone** (RF rises) — this is the headline |
| ECE static range | 0.058–0.107 (`calibration.csv`) | Table 1 / Results | reported as-is |
| ECE temporal range | 0.110–0.135 (`calibration.csv`) | Table 1 / Results | reported as-is |
| Counterfactual delay→mortality | 0.354→0.390 prob; slope 0.22 pp/hr, R²=0.81 (`counterfactual_delay.csv`) | Results / Fig 3 | monotone direction reproduces; magnitude shallow (damage saturates in 24 h) |
| DCA net benefit @ 0.30 | LR 0.229, RF 0.188, GB 0.175 (`decision_curve.csv`) | Results / Fig 4 | reported as-is |
| Conformal coverage @95% | LR 0.94, RF 0.95, GB 0.95 (`conformal.csv`) | Results | reproduces coverage validity |
| Avg prediction-set size @95% | 1.13 / 1.23 / 1.23 (`conformal.csv`) | Results | reported as-is |
| DBRS | 48 evaluated, 2 critical (cardiac ST-elevation, cardiac HR); mean 1.58 (`dbrs.csv`, `summary.json`) | Results | reported as-is |
| TCB δ_min @ ρ=0.20 | 0.013, bound holds 100% (`tcb.csv`) | Prop. 1 / Results | reported as-is |
| TCS mean by missingness | 0.82 / 0.79 / 0.83 (`temporal_consistency.csv`) | Results | reported as-is |
| TCE overall mean | 0.258 (`temporal_metrics.csv`) | Results | reported as-is |
| Seed = 42 | enforced in driver | Methods | reproduces |

### Reproduces (computed honestly, byte-identical at seed 42)
- Full static-vs-temporal AUROC / accuracy / ECE / AURC table for LR, RF, GB.
- The accuracy↔calibration dissociation and the discrimination-vs-calibration
  leaderboard reversal (the manuscript's central claim).
- Monotone antibiotic delay→mortality direction; conformal coverage validity;
  decision-curve net benefit; DBRS reliability structure; TCB; TCS; TCE.

### Not claimed (removed from the manuscript rather than fabricated)
| item | why |
|---|---|
| LSTM / TCN / XGBoost rows | no deep-learning or XGBoost code or dependency in the committed package — removed from the manuscript |
| antibiotic-delay 12.3 → 27.1 → 41.2 % / 4.8 pp·hr⁻¹ / R²=0.89 | fabricated; replaced with the computed 0.22 pp·hr⁻¹, R²=0.81 sweep |
| DBRS lactate=1.34 / PF=1.28 / troponin=1.31 | fabricated; replaced with the computed per-configuration DBRS |
| BEWS 2.1–4.2 h lead / precision 0.86–0.91 | degenerate under the presentation-driven label (lead time saturates at the 24 h horizon); reported honestly as exploratory, lead-time claim removed |
| DIR / Counterfactual Regret / fairness numbers | no generating code path — moved to future work |

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
