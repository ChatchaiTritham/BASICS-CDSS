# BASICS-CDSS — Complete Manuscript ↔ Code Reconciliation Table

Canonical run: **seed 42**, cohort **N = 1,000** (sepsis 400 / ARDS 350 / ACS 250),
seeded 60/20/20 split = **train 600 / calibration 200 / test 200**. Every value in
the "Computed value" column is read **verbatim** from `results/*.csv` produced by
`python scripts/run_all.py`. No metric is hardcoded. Numbers reported honestly —
where the computed value refutes the manuscript narrative, it is marked REFUTED.

Match legend: **Y** = matches; **close** = same direction/order of magnitude;
**REFUTED** = computed value contradicts the claim; **NOT-PRODUCED** = no code path
yields the quantity.

Model-cell order everywhere: **LR / RF / GB / XGBoost / LSTM / TCN**.

## 1. Main performance table

| Quantity | Manuscript value | Computed value (verbatim from CSV) | results CSV | Match? |
|---|---|---|---|---|
| Static AUROC (6 models) | LR 0.812 / RF 0.856 / XGB 0.873 / LSTM 0.891 / TCN 0.887 | 0.9159375 / 0.9176041667 / 0.9167708333 / 0.9046875 / 0.8732291667 / 0.9202083333 | `model_metrics.csv` | REFUTED (values + ranking: TCN best, LSTM worst) |
| Temporal AUROC (6 models) | (not separately tabulated, implied lower) | 0.901875 / 0.875 / 0.84546875 / 0.8219791667 / 0.8415625 / 0.9065625 | `model_metrics.csv` | NOT-PRODUCED in ms / computed here |
| ΔAUROC static→temporal | "temporal models ~30% less degradation" | LR −0.0141 / RF −0.0426 / GB −0.0713 / XGB −0.0827 / LSTM −0.0317 / TCN −0.0136 | `model_metrics.csv` | REFUTED (trees degrade MORE; TCN/LR least) |
| Static accuracy (6 models) | XGBoost 83.6% (headline); abstract mean 87.3% | 0.85 / 0.835 / 0.825 / 0.815 / 0.77 / 0.835 | `model_metrics.csv` | close (XGB 0.815 vs 0.836) |
| Temporal accuracy (6 models) | XGBoost 74.1% | 0.825 / 0.79 / 0.74 / 0.735 / 0.81 / 0.835 | `model_metrics.csv` | close (XGB 0.735 vs 0.741) |
| Δaccuracy (pp, static−temporal) | 6.6–9.5 pp fall | LR 2.5 / RF 4.5 / GB 8.5 / XGB 8.0 / LSTM −4.0 / TCN 0.0 | `model_metrics.csv` | close (GB/XGB 8.0–8.5 in band; LSTM/TCN rise) |

## 2. Abstract claims

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Accuracy fall | 6.6–9.5 pp | tree models GB 8.5 pp / XGB 8.0 pp (in band); LR 2.5 pp; LSTM −4.0 pp, TCN 0.0 pp | `model_metrics.csv` | close (tree models only) |
| Temporal (NN) models degrade ~30% less | sequence models more robust | TCN ΔAUROC −0.0136, LSTM −0.0317 vs tree mean −0.0655 → TCN 79% less, LSTM 52% less | `model_metrics.csv` | close (direction Y; magnitude >30%) |
| ECE static <0.05 → 0.067–0.118 | <0.05 baseline, rising to 0.067–0.118 | static ECE: 0.0499 / 0.0791 / 0.0837 / 0.1148 / 0.2220 / 0.1346 | `calibration.csv` | REFUTED (only LR <0.05; LSTM 0.222) |
| LSTM best ECE | LSTM lowest ECE | LSTM static ECE 0.2220 = **worst**; LR 0.0499 = best | `calibration.csv` | REFUTED |
| Antibiotic delay→mortality slope | 4.8 pp/hr, R²=0.89 | slope 0.2161 pp/hr, R²=0.8058 | `counterfactual_delay.csv` | REFUTED (slope shallow; R² close) |
| BEWS lead time | 2.1–4.2 h | lead-time range 23.0–24.0 h (saturates at horizon) | `bews.csv`, `summary.json` | REFUTED (saturates) |
| BEWS precision | 0.86–0.91 | precision range 0.586–0.904 | `bews.csv` | close (upper end overlaps) |
| TCB "holds in X% of configs" | (holds) | δ_min 0.000838, bound holds 100% of 6 configs | `tcb.csv`, `summary.json` | Y |

## 3. Counterfactual

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Alignment per model | 62% (LR) – 78% (LSTM) | LR 0.2025 / RF 0.1899 / GB 0.1899 / XGB 0.1772 / LSTM 0.2278 / TCN 0.2152 | `counterfactual_alignment.csv` | REFUTED (all ~0.18–0.23) |
| Regret per model | LSTM 2.1 / XGB 3.8 / RF 5.7 / LR 6.2 (LSTM best) | mean regret LR 19.65 / RF 19.79 / GB 19.77 / XGB 19.85 / LSTM 27.26 / TCN 18.26 (harm units) | `counterfactual_alignment.csv` | REFUTED (LSTM worst, not best; different units) |
| Median regret per model | — | LR 1.11 / RF 1.38 / GB 1.18 / XGB 1.47 / LSTM 1.58 / TCN 1.11 | `counterfactual_alignment.csv` | computed |
| ARDS counterfactual −14.3 pp | −14.3 pp | NOT-PRODUCED (ARDS counterfactual sweep not implemented; only sepsis antibiotic timing) | — | NOT-PRODUCED |
| STEMI 2.8 pp/hr | 2.8 pp/hr | NOT-PRODUCED (no cardiac/STEMI delay sweep) | — | NOT-PRODUCED |

## 4. Decision curve

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Net benefit @ 0.30 | 0.28 (caption) / 0.31 (body) | LR 0.3106 / RF 0.3053 / GB 0.3066 / XGB 0.2979 / LSTM 0.1507 / TCN 0.3014 | `decision_curve.csv` | close (tabular ~0.30–0.31) |
| Max net benefit + threshold | 0.31 at threshold 0.35 | max NB LR 0.3940 / RF 0.3866 / GB 0.3940 / XGB 0.3827 / LSTM 0.3939 / TCN 0.3635; **threshold = 0.01 for all** | `decision_curve.csv` | REFUTED (peak at low-threshold edge, not 0.35) |
| "28 more per 100" | +28 net true positives per 100 | net_benefit_model − net_benefit_all @0.30 (×100): see note; LR NB@0.30 = 0.311 → 31.1 per 100 vs treat-all | `decision_curve.csv` | close (LR ≈ 31/100) |

## 5. NNT (number needed to treat), threshold 0.30, bootstrap 95% CI

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| NNT overall (per model) | — | LR 1.45 [1.27–1.72] / RF 1.56 [1.35–1.85] / GB 1.54 [1.34–1.83] / XGB 1.57 [1.36–1.90] / LSTM 2.48 [2.11–2.96] / TCN 1.50 [1.31–1.79] | `nnt.csv` | computed |
| LSTM sepsis NNT | 6.2 [5.1–7.5] | LSTM sepsis 3.35 [2.45–4.89] | `nnt.csv` | REFUTED |
| LSTM ARDS NNT | 8.4 | LSTM ARDS = inf (ARR 0; LSTM flags all ARDS positive) | `nnt.csv` | REFUTED |
| LSTM ACS NNT | 5.8 | LSTM cardiac = inf (ARR 0) | `nnt.csv` | REFUTED |
| High-risk NNT | 3.2 | LR 1.61 / RF 1.61 / GB 1.69 / XGB 1.76 / LSTM 2.39 / TCN 1.88 (risk:high) | `nnt.csv` | REFUTED (lower NNT than claimed) |
| Low-risk NNT | 12.5 | LR 1.10 / RF 1.93 / GB 1.77 / XGB 1.70 / LSTM 4.14 / TCN 1.34 (risk:low) | `nnt.csv` | REFUTED |

## 6. Number-needed-to-screen @ 0.30

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| NNS @ 0.30 per model | — | LR 1.296 / RF 1.44 / GB 1.405 / XGB 1.380 / LSTM 2.475 / TCN 1.314 | `nns.csv` | computed |
| PPV @ 0.30 per model | — | LR 0.772 / RF 0.694 / GB 0.712 / XGB 0.724 / LSTM 0.404 / TCN 0.761 | `nns.csv` | computed |

## 7. Fairness (across SYNTHETIC `synthetic_group` attribute — methodology demo only)

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Demographic parity ratio | 0.89–0.94 | LR 0.408 / RF 0.501 / GB 0.469 / XGB 0.515 / LSTM 0.999 / TCN 0.408 | `fairness.csv` | REFUTED (synthetic disparity larger; LSTM ~1.0) |
| Equalized-odds ratio | 0.88–0.95 | LR 0.819 / RF 0.839 / GB 0.820 / XGB 0.892 / LSTM 0.994 / TCN 0.807 | `fairness.csv` | close (overlaps upper band) |
| Calibration fairness | 0.94–0.97 | LR 0.805 / RF 0.787 / GB 0.779 / XGB 0.693 / LSTM 0.733 / TCN 0.693 | `fairness.csv` | REFUTED (lower; synthetic attribute) |

NOTE: `synthetic_group` is fabricated for fairness-methodology demonstration only
(documented in code + REPRODUCIBILITY.md); these are not real-world bias claims.

## 8. Conformal prediction

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Coverage @ 95% per model | valid ≈ 0.95 | LR 0.98 / RF 0.95 / GB 0.96 / XGB 0.96 / LSTM 0.945 / TCN 0.97 | `conformal.csv` | Y (all ≥ target) |
| Avg set size @ 95% | 1.34 | LR 1.64 / RF 1.295 / GB 1.32 / XGB 1.37 / LSTM 1.635 / TCN 1.36 | `conformal.csv` | close (XGB 1.37, TCN 1.36 ≈ 1.34) |
| Singleton fraction | 78% | @95%: LR 0.36 / RF 0.705 / GB 0.68 / XGB 0.63 / LSTM 0.365 / TCN 0.64; @90%: LR 0.825 / RF 0.815 / GB 0.765 / XGB 0.815 / LSTM 0.57 / TCN 0.79 | `conformal.csv` | close (≈0.78 at 90% for trees) |

## 9. Noise-sensitivity sweep (AUROC degradation vs clean)

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Tree models more noise-robust than NN | trees > NN robustness | Gaussian 2×: GB −0.130, XGB −0.078, RF −0.065 vs LSTM +0.004, TCN +0.011 | `noise_sensitivity.csv` | REFUTED (NN sequence models more robust) |
| t(3) ≈ 2× worse than Gaussian | t3 degradation ≈ 2× Gaussian | t3 generally milder/comparable to equal-scale Gaussian (variance-matched) | `noise_sensitivity.csv` | REFUTED |
| Gaussian 1×/2×/3× degradation (GB) | — | GB −0.092 / −0.130 / −0.111 | `noise_sensitivity.csv` | computed |

## 10. Temporal-masking sweep (accuracy degradation, contiguous gaps)

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| 1 h gap accuracy drop | −1.2% (p=.12) | LR −17.0% / RF −32.5% / GB −41.0% / XGB −36.5% / LSTM 0.0% / TCN −0.5% | `masking_sweep.csv` | REFUTED (tabular much larger; NN ~0) |
| 2 h gap | −4.7% | LR −17.0% / RF −32.5% / GB −41.0% / XGB −36.5% / LSTM 0.0% / TCN 0.0% | `masking_sweep.csv` | REFUTED (flat across gaps) |
| 4 h gap | −12.3% | LR −17.0% / RF −32.5% / GB −41.0% / XGB −36.5% / LSTM 0.0% / TCN 0.0% | `masking_sweep.csv` | REFUTED |
| LSTM best gap tolerance | LSTM most tolerant | LSTM/TCN ≈ 0% degradation across all gaps | `masking_sweep.csv` | Y (direction reproduces) |

CAVEAT: tabular models score the (LOCF-masked) terminal-timestep row, so their
degradation is driven by whether the masked block reaches the terminal step and is
not graded by gap length; the graded 1/2/4 h ladder is therefore NOT reproduced.
The qualitative result (sequence models tolerate contiguous gaps; per-row tabular
models do not) is robust and reported honestly.

## 11. Supplementary — temporal metrics

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| TCS by missingness (0.2/0.4/0.6) | — | mean 0.808 / 0.755 / 0.772 (min 0.6175, max 0.914) | `temporal_consistency.csv`, `summary.json` | computed |
| TCE overall mean (L2) | — | 0.3337 (per-model: LR 0.339, RF 0.314, GB 0.344, XGB 0.339) | `temporal_metrics.csv`, `summary.json` | computed |
| DBRS | lactate 1.34 / PF 1.28 / troponin 1.31 | 64 configs, mean 1.016, range 0.959–1.405; 3 critical (ARDS O2-sat RF/XGB, ARDS PF-ratio GB) | `dbrs.csv`, `summary.json` | REFUTED (different markers/values) |

## 12. Cohort / design / determinism

| Quantity | Manuscript value | Computed value (verbatim) | results CSV | Match? |
|---|---|---|---|---|
| Cohort N | 1,000 | 1000 (sepsis 400 / ARDS 350 / cardiac 250) | `cohort_summary.json` | Y |
| Split | train 600 / cal 200 / test 200 | train 600 / calibration 200 / test 200 | `run_metadata.json` | Y |
| Outcome prevalence | — | overall 0.385 (sepsis 0.298, ARDS 0.566, cardiac 0.272) | `cohort_summary.json` | computed |
| Seed | 42 | 42 (numpy + torch); deterministic across two runs | `run_metadata.json` | Y |

## Items the pipeline still does NOT produce (marked NOT-PRODUCED above)
- ARDS counterfactual −14.3 pp and STEMI 2.8 pp/hr: only the sepsis antibiotic-
  timing counterfactual is implemented; no ARDS/cardiac intervention sweep exists.
- Any real-world (non-synthetic) demographic fairness claim: only a synthetic
  attribute is available.
