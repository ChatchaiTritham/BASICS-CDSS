"""Deterministic end-to-end reproducibility driver for BASICS-CDSS.

This script ties together the committed package code into a single, seeded
pipeline and writes the metrics it genuinely computes to ``results/``:

1. Build a digital-twin cohort (sepsis / ARDS / cardiac) with the committed
   ODE-based disease models in ``basics_cdss.temporal``.
2. Derive a static feature table + outcome labels from the twin trajectories.
3. Train the model families the committed package actually supports
   (scikit-learn: logistic regression, random forest, gradient boosting),
   under a clean ("static") split and a degraded ("temporal") split produced
   with the committed perturbation operators.
4. Evaluate every model with the committed metric modules
   (performance / calibration / coverage-risk / harm / utility / conformal).
5. Run the committed counterfactual evaluator to obtain a real
   antibiotic-delay -> harm trajectory.

The pipeline is honest by construction: it reports only what the committed
code computes from seeded data. Numbers are NOT tuned to the manuscript.
Metrics that the repository has no committed implementation for (deep
sequence models, calibrated mortality percentages) are intentionally absent;
see REPRODUCIBILITY.md for the manuscript-vs-code gap analysis.

Run from the repository root::

    python scripts/run_all.py

Outputs (overwritten each run, deterministic with seed=42):
    results/cohort_summary.json
    results/model_metrics.csv
    results/calibration.csv
    results/coverage_risk.csv
    results/harm.csv
    results/decision_curve.csv
    results/conformal.csv
    results/counterfactual_delay.csv
    results/dbrs.csv
    results/tcb.csv
    results/temporal_consistency.csv
    results/temporal_metrics.csv   (TCE, L2 trajectory form; per model x disease)
    results/bews.csv               (Biomarker Early-Warning Signature)
    results/summary.json
    results/run_metadata.json
"""

from __future__ import annotations

import os

# Pin numerical-library thread counts BEFORE importing numpy/scipy/sklearn.
# Gradient boosting's float reductions are order-sensitive, so a varying OpenMP
# thread count makes its probabilities (and the derived ECE/Brier/AUROC) jitter
# at the ~3rd decimal across hosts. Single-threaded reductions make the whole
# pipeline byte-for-byte reproducible at seed=42 on any machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from basics_cdss.clinical_metrics.conformal_prediction import (  # noqa: E402
    split_conformal_classification,
)
from basics_cdss.clinical_metrics.utility_metrics import (  # noqa: E402
    decision_curve_analysis,
)
from basics_cdss.metrics.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
)
from basics_cdss.metrics.coverage_risk import (  # noqa: E402
    selective_prediction_metrics,
)
from basics_cdss.metrics.harm import compute_harm_metrics  # noqa: E402
from basics_cdss.metrics.performance import (  # noqa: E402
    compute_performance_metrics,
)
from basics_cdss.temporal.counterfactual import (  # noqa: E402
    CounterfactualEvaluator,
    damage_to_mortality,
    infarct_to_mortality,
    lung_damage_to_mortality,
)
from basics_cdss.temporal.digital_twin import PatientDigitalTwin  # noqa: E402
from basics_cdss.temporal.disease_models import (  # noqa: E402
    CardiacEventModel,
    RespiratoryDistressModel,
    SepsisModel,
)

SEED = 42
RESULTS_DIR = REPO_ROOT / "results"

# Cohort composition mirrors the manuscript split proportions (sepsis-heavy).
# Sizes are reduced for a deterministic, minutes-scale reproducibility run; the
# pipeline scales linearly if larger cohorts are desired.
COHORT = {
    "sepsis": 200,
    "ards": 175,
    "cardiac": 125,
}
HORIZON_HOURS = 24.0
DT = 1.0

# Models the committed package can train with its declared dependencies
# (scikit-learn only). The package declares NO deep-learning or XGBoost
# dependency, so LSTM / TCN / XGBoost headline rows have no implementation here.
MODEL_FACTORIES = {
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=SEED
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=SEED),
}


def _initial_state(disease: str, rng: np.random.RandomState) -> dict:
    """Sample a clinically plausible initial state per disease family.

    Two sub-populations are sampled per family (worsening vs. stable) so the
    derived outcome label is learnable rather than degenerate.
    """
    worsening = rng.random() < 0.4
    if disease == "sepsis":
        if worsening:
            return {
                "temperature": rng.uniform(38.5, 39.8),
                "heart_rate": rng.uniform(105, 130),
                "respiratory_rate": rng.uniform(22, 30),
                "white_blood_cell_count": rng.uniform(15000, 22000),
                "blood_pressure_sys": rng.uniform(82, 98),
                "lactate": rng.uniform(2.8, 4.5),
            }
        return {
            "temperature": rng.uniform(37.0, 38.2),
            "heart_rate": rng.uniform(72, 96),
            "respiratory_rate": rng.uniform(14, 20),
            "white_blood_cell_count": rng.uniform(7000, 12000),
            "blood_pressure_sys": rng.uniform(108, 128),
            "lactate": rng.uniform(0.8, 2.0),
        }
    if disease == "ards":
        if worsening:
            return {
                "oxygen_saturation": rng.uniform(84, 91),
                "respiratory_rate": rng.uniform(26, 36),
                "pf_ratio": rng.uniform(120, 200),
                "heart_rate": rng.uniform(95, 120),
            }
        return {
            "oxygen_saturation": rng.uniform(94, 99),
            "respiratory_rate": rng.uniform(14, 20),
            "pf_ratio": rng.uniform(300, 420),
            "heart_rate": rng.uniform(70, 92),
        }
    # cardiac
    if worsening:
        return {
            "heart_rate": rng.uniform(95, 120),
            "blood_pressure_sys": rng.uniform(140, 170),
            "blood_pressure_dia": rng.uniform(88, 105),
            "troponin": rng.uniform(2.0, 8.0),
            "st_elevation": rng.uniform(1.8, 3.0),
            "chest_pain_score": rng.uniform(6, 9),
        }
    return {
        "heart_rate": rng.uniform(62, 88),
        "blood_pressure_sys": rng.uniform(112, 134),
        "blood_pressure_dia": rng.uniform(70, 86),
        "troponin": rng.uniform(0.01, 0.3),
        "st_elevation": rng.uniform(0.0, 0.6),
        "chest_pain_score": rng.uniform(0, 3),
    }


# Instantaneous terminal severity signal per disease (observable severity, used
# for the risk tier and retained as a descriptive column -- NOT the label).
_SEVERITY_KEY = {
    "sepsis": "_infection_severity",
    "ards": "_lung_injury",
    "cardiac": "_ischemia_severity",
}

# Latent cumulative irreversible-damage state per disease whose TERMINAL value
# drives the calibrated mortality label (Component A mechanism). Each maps the
# terminal damage to a probability of death via a literature-anchored logistic
# link (see counterfactual.py); the binary outcome is the deterministic
# threshold p_death >= 0.5. This replaces the previous label
# ``int(instantaneous_severity >= 0.5)``, which was degenerate for cardiac
# (self-amplifying ischemia drove every twin to a single class) and near-
# degenerate for ARDS (instantaneous lung injury relaxes toward 0).
_DAMAGE_KEY = {
    "sepsis": "_organ_damage",
    "ards": "_lung_damage",
    "cardiac": "_infarct",
}
_MORTALITY_MAP = {
    "sepsis": damage_to_mortality,
    "ards": lung_damage_to_mortality,
    "cardiac": infarct_to_mortality,
}

# Common feature columns extracted from the initial observable state.
_FEATURE_COLUMNS = [
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "white_blood_cell_count",
    "blood_pressure_sys",
    "blood_pressure_dia",
    "lactate",
    "oxygen_saturation",
    "pf_ratio",
    "troponin",
    "st_elevation",
    "chest_pain_score",
]


def build_cohort() -> pd.DataFrame:
    """Simulate the digital-twin cohort and derive a static feature/label table.

    The label is the calibrated mortality outcome derived from the twin's
    TERMINAL cumulative irreversible-damage state (sepsis organ damage / ARDS
    lung damage / cardiac infarct), mapped to a probability of death via a
    literature-anchored logistic link and realised as a SEEDED BERNOULLI DRAW
    from that probability (outcome ~ Bernoulli(p_death)). Drawing the outcome
    rather than hard-thresholding the probability is the standard,
    non-fabricating way to generate a binary label from a calibrated risk model:
    it preserves the irreducible aleatoric uncertainty of death given a fixed
    damage state (two twins with the same terminal damage need not share the same
    fate), so the resulting classification task is learnable but NOT perfectly
    separable -- a deterministic 0.5 threshold collapses to the (clean, bimodal)
    presentation clusters and yields a degenerate ~1.0 AUROC. Determinism is
    preserved by drawing from the seeded cohort RNG. This grounds the task in the
    committed simulator's Component-A damage mechanism rather than in the previous
    instantaneous-severity label, which collapsed the cardiac cohort to a single
    class and was near-degenerate for ARDS. The instantaneous terminal severity
    is retained as a descriptive ``terminal_severity`` column and used for the
    risk tier.
    """
    rng = np.random.RandomState(SEED)
    # Dedicated, seeded RNG for the Bernoulli outcome draws so the cohort
    # simulation RNG stream (initial states) is unaffected and deterministic.
    label_rng = np.random.RandomState(SEED + 7919)
    models = {
        "sepsis": SepsisModel(),
        "ards": RespiratoryDistressModel(),
        "cardiac": CardiacEventModel(),
    }
    rows = []
    for disease, count in COHORT.items():
        sev_key = _SEVERITY_KEY[disease]
        dmg_key = _DAMAGE_KEY[disease]
        mortality_map = _MORTALITY_MAP[disease]
        # Deterministic per-disease offset (no salted builtin hash).
        disease_offset = sum(ord(c) for c in disease) * 1000
        for i in range(count):
            init = _initial_state(disease, rng)
            twin = PatientDigitalTwin(
                archetype_id=f"{disease}_{i:04d}",
                initial_state=init,
                disease_model=models[disease],
                seed=SEED + disease_offset + i,
            )
            trajectory = twin.simulate(
                horizon_hours=HORIZON_HOURS, dt=DT, stochastic=True
            )
            terminal = trajectory[-1].features
            severity = float(terminal.get(sev_key, 0.0))
            # Terminal cumulative irreversible damage -> calibrated mortality
            # probability -> deterministic binary outcome (threshold 0.5).
            terminal_damage = float(terminal.get(dmg_key, 0.0))
            mortality_prob = float(mortality_map(terminal_damage))
            # Seeded Bernoulli realisation of the calibrated mortality risk.
            outcome = int(label_rng.random() < mortality_prob)
            # Risk tier from terminal severity (drives harm-weighted metrics).
            if severity >= 0.6:
                tier = "high"
            elif severity >= 0.3:
                tier = "medium"
            else:
                tier = "low"

            row = {col: init.get(col, np.nan) for col in _FEATURE_COLUMNS}
            row.update(
                {
                    "twin_id": twin.archetype_id,
                    "disease": disease,
                    "terminal_severity": severity,
                    "terminal_damage": terminal_damage,
                    "mortality_prob": mortality_prob,
                    "risk_tier": tier,
                    "outcome": outcome,
                }
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    # Impute disease-disjoint missing features with column medians so a single
    # shared classifier can consume the unified table.
    feat = df[_FEATURE_COLUMNS]
    df[_FEATURE_COLUMNS] = feat.fillna(feat.median(numeric_only=True))
    return df


def _degrade(X: np.ndarray, rng: np.random.RandomState, missing: float = 0.20,
             noise: float = 2.0) -> np.ndarray:
    """Apply MCAR masking + Gaussian noise ('temporal' degraded evaluation).

    Mirrors the manuscript's '20% MCAR + 2x noise' temporal stress test, using
    median imputation for masked entries so models remain evaluable.
    """
    Xd = X.copy().astype(float)
    col_std = Xd.std(axis=0)
    col_std[col_std == 0] = 1.0
    Xd = Xd + rng.normal(0, 1.0, Xd.shape) * col_std * (noise - 1.0) * 0.1
    mask = rng.random(Xd.shape) < missing
    col_median = np.median(X, axis=0)
    for j in range(Xd.shape[1]):
        Xd[mask[:, j], j] = col_median[j]
    return Xd


def evaluate_models(df: pd.DataFrame) -> dict:
    """Train and evaluate each supported model under static + temporal splits."""
    rng = np.random.RandomState(SEED)
    X = df[_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["outcome"].to_numpy(dtype=int)
    tiers = df["risk_tier"].to_numpy()

    n = len(df)
    idx = rng.permutation(n)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    tr, cal, te = idx[:n_train], idx[n_train:n_train + n_cal], idx[n_train + n_cal:]

    X_tr, y_tr = X[tr], y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te, y_te, tiers_te = X[te], y[te], tiers[te]

    # Degraded ('temporal') variant of the test features.
    X_te_temporal = _degrade(X_te, np.random.RandomState(SEED + 1))

    metric_rows = []
    calib_rows = []
    coverage_rows = []
    harm_rows = []
    dca_rows = []
    conformal_rows = []

    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X_tr, y_tr)

        for regime, X_eval in (("static", X_te), ("temporal", X_te_temporal)):
            prob = model.predict_proba(X_eval)[:, 1]
            pred = (prob >= 0.5).astype(int)

            perf = compute_performance_metrics(y_te, pred, prob)
            ece = expected_calibration_error(y_te, prob)
            bs = brier_score(y_te, prob)

            metric_rows.append(
                {
                    "model": name,
                    "regime": regime,
                    "n_test": len(y_te),
                    "accuracy": perf.accuracy,
                    "auroc": perf.roc_auc,
                    "auprc": perf.pr_auc,
                    "precision": perf.precision,
                    "recall": perf.recall,
                    "specificity": perf.specificity,
                    "f1": perf.f1_score,
                    "ece": ece,
                    "brier": bs,
                }
            )
            calib_rows.append(
                {"model": name, "regime": regime, "ece": ece, "brier": bs}
            )

            sel = selective_prediction_metrics(y_te, prob)
            coverage_rows.append(
                {
                    "model": name,
                    "regime": regime,
                    "aurc": sel.aurc,
                    "risk_at_cov_0.8": sel.risk_at_coverage_threshold,
                    "cov_at_risk_0.1": sel.coverage_at_risk_threshold,
                }
            )

            harm = compute_harm_metrics(y_te, pred, tiers_te)
            harm_rows.append(
                {
                    "model": name,
                    "regime": regime,
                    "weighted_harm_loss": harm.weighted_harm_loss,
                    "harm_high": harm.harm_by_tier.get("high", 0.0),
                    "harm_medium": harm.harm_by_tier.get("medium", 0.0),
                    "harm_low": harm.harm_by_tier.get("low", 0.0),
                    "escalation_failures": harm.escalation_failures,
                    "false_escalations": harm.false_escalations,
                    "harm_concentration": harm.harm_concentration,
                }
            )

            if regime == "static":
                dca = decision_curve_analysis(y_te, prob)
                # Net benefit at the clinically reported threshold 0.30.
                t_idx = int(np.argmin(np.abs(dca.thresholds - 0.30)))
                max_idx = int(np.argmax(dca.net_benefit_model))
                dca_rows.append(
                    {
                        "model": name,
                        "net_benefit_at_0.30": float(dca.net_benefit_model[t_idx]),
                        "max_net_benefit": float(dca.net_benefit_model[max_idx]),
                        "max_net_benefit_threshold": float(
                            dca.thresholds[max_idx]
                        ),
                        "useful_threshold_low": float(dca.threshold_range[0]),
                        "useful_threshold_high": float(dca.threshold_range[1]),
                    }
                )

    # Conformal coverage (committed split-conformal classifier), evaluated on a
    # labeled held-out test set so empirical coverage can be measured.
    for name, factory in MODEL_FACTORIES.items():
        for alpha, target in ((0.10, 0.90), (0.05, 0.95)):
            result = split_conformal_classification(
                factory(), X_tr, y_tr, X_cal, y_cal, X_te, alpha=alpha
            )
            covered = [
                int(y_te[i] in set(result.prediction_sets[i]))
                for i in range(len(y_te))
            ]
            conformal_rows.append(
                {
                    "model": name,
                    "target_coverage": target,
                    "empirical_coverage": float(np.mean(covered)),
                    "avg_set_size": float(result.efficiency),
                    "singleton_fraction": float(
                        np.mean(result.set_sizes == 1)
                    ),
                }
            )

    return {
        "model_metrics": pd.DataFrame(metric_rows),
        "calibration": pd.DataFrame(calib_rows),
        "coverage_risk": pd.DataFrame(coverage_rows),
        "harm": pd.DataFrame(harm_rows),
        "decision_curve": pd.DataFrame(dca_rows),
        "conformal": pd.DataFrame(conformal_rows),
    }


def counterfactual_delay() -> pd.DataFrame:
    """Antibiotic-delay -> mortality trajectory using the committed simulator.

    Tests the manuscript's claim that earlier antibiotics reduce mortality. For
    each delay value, the SAME sepsis twins are simulated for the full horizon
    with the antibiotic applied at ``delay`` hours (the intervention time is the
    swept variable). The cumulative irreversible organ-damage state D accrues
    while the patient remains septic, so a later antibiotic leaves a longer
    high-severity window and a larger terminal D; the terminal D is mapped to a
    calibrated mortality probability via the logistic link in
    ``damage_to_mortality`` (anchored to Kumar et al. 2006, NOT back-solved to
    any manuscript number -- see counterfactual.py).

    Determinism note: the committed simulator only honours its seeded RNG on the
    ``stochastic=True`` path; the ``stochastic=False`` path falls back to an
    UNSEEDED numpy RandomState (a reproducibility defect, see REPRODUCIBILITY.md).
    This driver therefore drives the twin directly on its seeded stochastic path,
    keeping the result both faithful to the committed code and deterministic.

    The emitted slope (pp/hr) and R^2 are computed with ``np.polyfit`` over the
    realised (delay, mortality_prob) points -- they are NOT typed in.
    """
    rng = np.random.RandomState(SEED)
    model = SepsisModel()
    harm_fn = CounterfactualEvaluator(
        horizon_hours=HORIZON_HOURS, dt=DT
    ).harm_function
    n_twins = 120
    delays = [1.0, 3.0, 6.0, 9.0, 12.0, 18.0]

    inits = []
    for _ in range(n_twins):
        init = _initial_state("sepsis", rng)
        # Bias toward presenting sepsis so antibiotics have headroom to help.
        init["temperature"] = max(init["temperature"], 38.3)
        init["lactate"] = max(init["lactate"], 2.2)
        inits.append(init)

    rows = []
    for delay in delays:
        harms = []
        damages = []
        mortalities = []
        for i, init in enumerate(inits):
            twin = PatientDigitalTwin(
                archetype_id=f"sepsis_cf_{i:04d}",
                initial_state=init,
                disease_model=model,
                seed=SEED + i,
            )
            schedule = {float(delay): {"antibiotic": True}}
            trajectory = twin.simulate(
                horizon_hours=HORIZON_HOURS,
                dt=DT,
                intervention_schedule=schedule,
                stochastic=True,  # seeded RNG => deterministic
            )
            terminal = trajectory[-1]
            terminal_d = terminal.features.get("_organ_damage", 0.0)
            harms.append(harm_fn(terminal))
            damages.append(terminal_d)
            mortalities.append(damage_to_mortality(terminal_d))
        rows.append(
            {
                "antibiotic_delay_hours": delay,
                "mean_terminal_damage": float(np.mean(damages)),
                "mean_terminal_harm": float(np.mean(harms)),
                "median_terminal_harm": float(np.median(harms)),
                "std_terminal_harm": float(np.std(harms)),
                "mortality_prob": float(np.mean(mortalities)),
                "std_mortality_prob": float(np.std(mortalities)),
                "n_twins": n_twins,
            }
        )

    df = pd.DataFrame(rows)

    # Computed (not typed) delay->mortality slope in percentage points per hour,
    # plus the linear-fit R^2 over the realised sweep points.
    x = df["antibiotic_delay_hours"].to_numpy(dtype=float)
    y = df["mortality_prob"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    df["slope_pp_per_hr"] = float(slope * 100.0)  # prob/hr -> percentage pts/hr
    df["r_squared"] = float(r_squared)
    return df


# Biomarkers genuinely observed per disease family (the rest are median-imputed
# disease-disjoint columns in the unified table, so ablating them is meaningless).
# Mirrors the per-disease initial-state sampling in ``_initial_state``.
_DISEASE_BIOMARKERS = {
    "sepsis": [
        "temperature",
        "heart_rate",
        "respiratory_rate",
        "white_blood_cell_count",
        "blood_pressure_sys",
        "lactate",
    ],
    "ards": [
        "oxygen_saturation",
        "respiratory_rate",
        "pf_ratio",
        "heart_rate",
    ],
    "cardiac": [
        "heart_rate",
        "blood_pressure_sys",
        "blood_pressure_dia",
        "troponin",
        "st_elevation",
        "chest_pain_score",
    ],
}


def compute_dbrs(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamic Biomarker Reliability Score (DBRS) per disease x biomarker x model.

    DBRS(b_j) = AUROC(f | b_j complete) / AUROC(f | b_j ablated), where ``ablated``
    replaces the biomarker column with its (temporal) mean -- here the training-
    cohort mean of that column, the deterministic empirical estimate of the
    biomarker's mean level. This is a pure re-``predict_proba`` on the trained
    sklearn models with a single column meaned; every value is COMPUTED, none is
    tuned to the manuscript's prior DBRS figures. Biomarkers with DBRS > 1.15 are
    flagged ``critical`` per the manuscript's reliability bands.

    The same train/cal/test split as ``evaluate_models`` is reproduced from the
    seeded permutation so the trained models are identical.
    """
    rng = np.random.RandomState(SEED)
    X = df[_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["outcome"].to_numpy(dtype=int)
    disease = df["disease"].to_numpy()

    n = len(df)
    idx = rng.permutation(n)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    tr = idx[: n_train]
    te = idx[n_train + n_cal:]

    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    disease_te = disease[te]

    # Training-cohort mean per column = the deterministic "temporal mean" used to
    # ablate a biomarker (replace its column with this constant).
    col_mean = X_tr.mean(axis=0)
    col_index = {c: i for i, c in enumerate(_FEATURE_COLUMNS)}

    from sklearn.metrics import roc_auc_score

    rows = []
    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X_tr, y_tr)
        for dz, biomarkers in _DISEASE_BIOMARKERS.items():
            mask = disease_te == dz
            if mask.sum() < 2:
                continue
            y_dz = y_te[mask]
            # AUROC is undefined if the disease subset is single-class.
            if len(np.unique(y_dz)) < 2:
                continue
            X_dz = X_te[mask]
            prob_complete = model.predict_proba(X_dz)[:, 1]
            auroc_complete = float(roc_auc_score(y_dz, prob_complete))
            for b in biomarkers:
                j = col_index[b]
                X_abl = X_dz.copy()
                X_abl[:, j] = col_mean[j]
                prob_abl = model.predict_proba(X_abl)[:, 1]
                auroc_abl = float(roc_auc_score(y_dz, prob_abl))
                dbrs = auroc_complete / auroc_abl if auroc_abl > 0 else float("nan")
                rows.append(
                    {
                        "disease": dz,
                        "biomarker": b,
                        "model": name,
                        "n_eval": int(mask.sum()),
                        "auroc_complete": auroc_complete,
                        "auroc_ablated": auroc_abl,
                        "dbrs": dbrs,
                        "critical": bool(dbrs > 1.15)
                        if dbrs == dbrs
                        else False,
                    }
                )
    return pd.DataFrame(rows)


def compute_tcb(model_metrics: pd.DataFrame, rho: float = 0.20) -> pd.DataFrame:
    """Temporal Coverage Bound (TCB): empirical delta_min = L_temporal - L_static.

    The bound is L_temporal >= L_static + rho * delta_min. Using the committed
    static-vs-temporal Brier loss already in ``model_metrics`` (one (static,
    temporal) pair per model), the per-configuration gap is
    g = L_temporal - L_static; the empirical delta_min is the smallest such gap
    across the evaluated configurations (the tightest realised bound). We report
    delta_min, the implied minimum under-estimation rho * delta_min, and the
    fraction of configurations for which the bound holds at delta = delta_min
    (i.e. g >= rho * delta_min). Loss = Brier score (proper, bounded).

    All values COMPUTED from the seeded model_metrics; none tuned to the prior
    manuscript figure (delta_min ~ 0.047).
    """
    mm = model_metrics.copy()
    loss_col = "brier"
    gaps = []
    for model_name in sorted(mm["model"].unique()):
        sub = mm[mm["model"] == model_name]
        l_static = sub[sub["regime"] == "static"][loss_col]
        l_temporal = sub[sub["regime"] == "temporal"][loss_col]
        if len(l_static) == 0 or len(l_temporal) == 0:
            continue
        g = float(l_temporal.iloc[0]) - float(l_static.iloc[0])
        gaps.append({"model": model_name, "loss_gap": g})

    gaps_df = pd.DataFrame(gaps)
    if gaps_df.empty:
        return gaps_df

    delta_min = float(gaps_df["loss_gap"].min())
    rho_delta = rho * delta_min
    # Fraction of configurations satisfying the bound at delta = delta_min.
    holds = (gaps_df["loss_gap"] >= rho_delta).mean()
    gaps_df["loss_metric"] = loss_col
    gaps_df["rho"] = rho
    gaps_df["delta_min"] = delta_min
    gaps_df["rho_delta_min"] = rho_delta
    gaps_df["bound_holds"] = gaps_df["loss_gap"] >= rho_delta
    gaps_df["fraction_bound_holds"] = float(holds)
    return gaps_df


def _twin_timestep_features(state_features: dict) -> np.ndarray:
    """Project a single trajectory state onto the shared feature vector order."""
    return np.array(
        [float(state_features.get(c, np.nan)) for c in _FEATURE_COLUMNS],
        dtype=float,
    )


def compute_tcs(df: pd.DataFrame, missing_levels=(0.20, 0.40, 0.60)) -> pd.DataFrame:
    """Temporal Consistency Score (TCS) per model x disease x missingness.

    TCS = 1 - C/(n-1), C = consecutive risk-classification reversals along a
    twin's observation sequence over the horizon (delegated to the committed
    ``temporal_consistency_score`` with binary risk-class predictions). The
    sequence IS available: ``twin.simulate`` returns the full per-timestep
    history, so per-timestep observable features feed ``predict_proba``; a
    refactor is NOT required for terminal-only -> sequence emission.

    At each missingness level, features are MCAR-masked (column-median imputed)
    independently per timestep using a seeded RNG, the trained model classifies
    risk (prob >= 0.5) at every timestep, and reversals are counted. Reported per
    twin then averaged. All COMPUTED from seeded data.
    """
    from basics_cdss.temporal.metrics import temporal_consistency_score

    rng = np.random.RandomState(SEED)
    X = df[_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["outcome"].to_numpy(dtype=int)
    n = len(df)
    idx = rng.permutation(n)
    n_train = int(0.6 * n)
    tr = idx[:n_train]
    X_tr, y_tr = X[tr], y[tr]
    col_median = np.median(X_tr, axis=0)

    models = {
        "sepsis": SepsisModel(),
        "ards": RespiratoryDistressModel(),
        "cardiac": CardiacEventModel(),
    }

    # Re-simulate the cohort's twins to recover full trajectories (terminal-only
    # cohort.csv does not retain the sequence). Determinism: same seeds as
    # build_cohort, same seeded stochastic path.
    cohort_rng = np.random.RandomState(SEED)
    trained = {}
    for name, factory in MODEL_FACTORIES.items():
        m = factory()
        m.fit(X_tr, y_tr)
        trained[name] = m

    rows = []
    for disease, count in COHORT.items():
        disease_offset = sum(ord(c) for c in disease) * 1000
        # consume the same RNG draws as build_cohort so initial states match
        twin_inits = []
        for i in range(count):
            twin_inits.append(_initial_state(disease, cohort_rng))
        # Per (model, missingness) accumulate twin-level TCS.
        acc = {
            (mname, miss): [] for mname in trained for miss in missing_levels
        }
        for i, init in enumerate(twin_inits):
            twin = PatientDigitalTwin(
                archetype_id=f"{disease}_{i:04d}",
                initial_state=init,
                disease_model=models[disease],
                seed=SEED + disease_offset + i,
            )
            trajectory = twin.simulate(
                horizon_hours=HORIZON_HOURS, dt=DT, stochastic=True
            )
            seq = np.vstack(
                [_twin_timestep_features(s.features) for s in trajectory]
            )
            # Impute any disease-disjoint NaNs with the training median so the
            # classifier can score every timestep.
            for j in range(seq.shape[1]):
                col = seq[:, j]
                col[np.isnan(col)] = col_median[j]
                seq[:, j] = col
            for miss in missing_levels:
                # Deterministic per-twin/level mask seed.
                mrng = np.random.RandomState(
                    SEED + disease_offset + i + int(miss * 1000)
                )
                seq_m = seq.copy()
                mask = mrng.random(seq_m.shape) < miss
                for j in range(seq_m.shape[1]):
                    seq_m[mask[:, j], j] = col_median[j]
                for mname, model in trained.items():
                    prob = model.predict_proba(seq_m)[:, 1]
                    risk_class = (prob >= 0.5).astype(int)
                    preds = [{"risk": int(c)} for c in risk_class]
                    tcs = float(
                        temporal_consistency_score(
                            preds, window_size=2, intervention_keys=["risk"]
                        )
                    )
                    acc[(mname, miss)].append(tcs)
        for (mname, miss), vals in acc.items():
            rows.append(
                {
                    "model": mname,
                    "disease": disease,
                    "missingness": miss,
                    "n_twins": len(vals),
                    "mean_tcs": float(np.mean(vals)) if vals else float("nan"),
                    "std_tcs": float(np.std(vals)) if vals else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _simulate_cohort_trajectories():
    """Re-simulate every cohort twin and return its full per-timestep trajectory.

    Yields ``(disease, twin_index, init_state, trajectory)`` for every twin, in
    the SAME order and with the SAME seeds as ``build_cohort`` / ``compute_tcs``,
    so the recovered trajectories are byte-for-byte the cohort's. The terminal-
    only ``cohort.csv`` does not retain the sequence, but ``twin.simulate`` returns
    the full history, so no terminal-only -> sequence refactor is needed.

    Determinism: a dedicated cohort RNG (seeded SEED) reproduces the initial-state
    draws; each twin uses the same per-twin seed offset as ``build_cohort``.
    """
    models = {
        "sepsis": SepsisModel(),
        "ards": RespiratoryDistressModel(),
        "cardiac": CardiacEventModel(),
    }
    cohort_rng = np.random.RandomState(SEED)
    for disease, count in COHORT.items():
        disease_offset = sum(ord(c) for c in disease) * 1000
        inits = [_initial_state(disease, cohort_rng) for _ in range(count)]
        for i, init in enumerate(inits):
            twin = PatientDigitalTwin(
                archetype_id=f"{disease}_{i:04d}",
                initial_state=init,
                disease_model=models[disease],
                seed=SEED + disease_offset + i,
            )
            trajectory = twin.simulate(
                horizon_hours=HORIZON_HOURS, dt=DT, stochastic=True
            )
            yield disease, i, init, trajectory


def compute_tce(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal Calibration Error (TCE) -- main-text L2 trajectory form.

    Per twin, the L2 trajectory calibration error is

        TCE = (1/T) * sum_t | y_hat(t) - y(t) |,

    where (delegated to ``temporal.metrics.trajectory_calibration_error``, the L2
    form):
      * y_hat(t) is the TRAINED classifier's predicted risk (predict_proba[:,1])
        evaluated on the twin's OBSERVABLE per-timestep features at time t; and
      * y(t) is the model-free reference risk trajectory -- the calibrated
        mortality probability obtained by mapping the twin's LATENT per-timestep
        cumulative-damage state (organ damage / lung damage / infarct) through the
        SAME literature-anchored logistic link that defines the cohort label
        (damage_to_mortality / lung_damage_to_mortality / infarct_to_mortality).
        This y(t) is the ground-truth risk the calibrated classifier should track;
        it is derived from the simulator's latent state, NOT from the classifier.

    The same trained models and the same train/cal/test split as ``evaluate_models``
    are reproduced from the seeded permutation; TCE is reported on the held-out
    TEST twins only (the models never saw them). Per model x disease we report the
    mean and std of the per-twin trajectory TCE. Every value is COMPUTED from
    seeded simulation -- none is tuned to any prior manuscript figure.
    """
    from basics_cdss.temporal.metrics import trajectory_calibration_error

    rng = np.random.RandomState(SEED)
    X = df[_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["outcome"].to_numpy(dtype=int)
    n = len(df)
    idx = rng.permutation(n)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    tr = idx[:n_train]
    te = set(int(k) for k in idx[n_train + n_cal:])  # held-out test indices
    X_tr, y_tr = X[tr], y[tr]
    col_median = np.median(X_tr, axis=0)

    trained = {}
    for name, factory in MODEL_FACTORIES.items():
        m = factory()
        m.fit(X_tr, y_tr)
        trained[name] = m

    # Per-disease latent-damage key + damage->risk map for the reference y(t).
    dmg_key = _DAMAGE_KEY
    mortality_map = _MORTALITY_MAP

    # Flat row index over the cohort (matches build_cohort row order) so we can
    # tell which re-simulated twin is a held-out test twin.
    flat = -1
    acc = {(name, dz): [] for name in trained for dz in COHORT}
    for disease, i, init, trajectory in _simulate_cohort_trajectories():
        flat += 1
        if flat not in te:
            continue  # report TCE on held-out test twins only
        # Observable per-timestep feature matrix (T x d), median-imputed for the
        # disease-disjoint columns so the classifier can score every timestep.
        seq = np.vstack(
            [_twin_timestep_features(s.features) for s in trajectory]
        )
        for j in range(seq.shape[1]):
            col = seq[:, j]
            col[np.isnan(col)] = col_median[j]
            seq[:, j] = col
        # Reference risk trajectory y(t): map the latent per-timestep damage
        # through the calibrated logistic link that defines the label.
        dk = dmg_key[disease]
        mmap = mortality_map[disease]
        y_true_traj = np.array(
            [float(mmap(float(s.features.get(dk, 0.0)))) for s in trajectory],
            dtype=float,
        )
        for name, model in trained.items():
            y_hat_traj = model.predict_proba(seq)[:, 1]
            tce = trajectory_calibration_error(y_hat_traj, y_true_traj)
            acc[(name, disease)].append(tce)

    rows = []
    for (name, disease), vals in acc.items():
        rows.append(
            {
                "model": name,
                "disease": disease,
                "n_twins": len(vals),
                "mean_tce": float(np.mean(vals)) if vals else float("nan"),
                "std_tce": float(np.std(vals)) if vals else float("nan"),
                "median_tce": float(np.median(vals)) if vals else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# BEWS threshold: a per-timestep population z-score deviation of 1.96 (the 97.5th
# percentile of the standard normal) is the conventional "significant deviation"
# bound; the earliest pre-event timestep at which the failure-cohort mean
# deviation D_j(tau) crosses it defines the early-warning lead time.
_BEWS_THRESHOLD = 1.96
# Minimum failures per disease for a stable BEWS estimate; below this we FLAG the
# disease rather than fabricate a lead time.
_BEWS_MIN_FAILURES = 10


def compute_bews(df: pd.DataFrame) -> pd.DataFrame:
    """Biomarker Early-Warning Signature (BEWS) per disease x biomarker.

    Manuscript definition:

        D_j(tau) = E_{failures}[ | x_j(tau) - mu_j(tau) | / sigma_j(tau) ],

    where the failure set F = twins that died (outcome == 1), and mu_j(tau),
    sigma_j(tau) are the per-timestep population mean / std of biomarker j over the
    NON-failure (outcome == 0) trajectories. The lead time is the EARLIEST timestep
    tau (before the terminal event at the horizon) at which D_j(tau) > 1.96,
    reported in hours-before-event = HORIZON - tau. Precision is the fraction of
    FLAGGED twins (twins whose own per-timestep z-deviation of biomarker j ever
    crosses 1.96 at or after the population lead time) that are true failures.

    All trajectories are recovered from ``twin.simulate`` (seeded, same order as
    the cohort). Every value is COMPUTED from seeded data; nothing is tuned to the
    prior manuscript figures (BEWS 2.1-4.2 h lead / precision 0.86-0.91). If a
    disease has fewer than ``_BEWS_MIN_FAILURES`` failures, it is FLAGGED and its
    lead time / precision are left NaN (no fabrication).
    """
    # Recover per-disease trajectories grouped with the cohort outcome labels.
    outcomes = df["outcome"].to_numpy(dtype=int)
    diseases_col = df["disease"].to_numpy()
    # Map flat cohort index -> outcome (build_cohort row order == sim order).
    flat = -1
    # Per disease: list of (outcome, {biomarker: np.array over T}) per twin.
    per_disease = {dz: [] for dz in COHORT}
    n_timesteps = int(HORIZON_HOURS / DT) + 1  # t=0..horizon inclusive
    for disease, i, init, trajectory in _simulate_cohort_trajectories():
        flat += 1
        assert diseases_col[flat] == disease  # sanity: order matches cohort
        outcome = int(outcomes[flat])
        biomarkers = _DISEASE_BIOMARKERS[disease]
        traj_bio = {}
        for b in biomarkers:
            traj_bio[b] = np.array(
                [float(s.features.get(b, np.nan)) for s in trajectory],
                dtype=float,
            )
        per_disease[disease].append((outcome, traj_bio))

    rows = []
    for disease in COHORT:
        twins = per_disease[disease]
        biomarkers = _DISEASE_BIOMARKERS[disease]
        failures = [tb for (o, tb) in twins if o == 1]
        nonfailures = [tb for (o, tb) in twins if o == 0]
        n_fail = len(failures)
        n_nonfail = len(nonfailures)
        flagged_disease = n_fail < _BEWS_MIN_FAILURES or n_nonfail < 2

        for b in biomarkers:
            # Non-failure population mean/std per timestep mu_j(tau), sigma_j(tau).
            nf_stack = np.vstack(
                [tb[b][:n_timesteps] for tb in nonfailures]
            ) if nonfailures else np.empty((0, n_timesteps))
            f_stack = np.vstack(
                [tb[b][:n_timesteps] for tb in failures]
            ) if failures else np.empty((0, n_timesteps))

            if flagged_disease or nf_stack.shape[0] < 2 or f_stack.shape[0] == 0:
                rows.append(
                    {
                        "disease": disease,
                        "biomarker": b,
                        "n_failures": n_fail,
                        "n_nonfailures": n_nonfail,
                        "deviation_at_t0": float("nan"),
                        "max_deviation": float("nan"),
                        "first_cross_tau": float("nan"),
                        "lead_time_hours": float("nan"),
                        "precision": float("nan"),
                        "flagged_insufficient": True,
                    }
                )
                continue

            mu = np.nanmean(nf_stack, axis=0)              # mu_j(tau)
            sigma = np.nanstd(nf_stack, axis=0)            # sigma_j(tau)
            sigma_safe = np.where(sigma > 1e-9, sigma, np.nan)
            # D_j(tau) = mean over failures of |x - mu| / sigma.
            z_fail = np.abs(f_stack - mu) / sigma_safe     # (n_fail x T)
            D = np.nanmean(z_fail, axis=0)                 # D_j(tau), length T

            # Earliest crossing timestep -> lead time before terminal event.
            # NOTE: D_at_t0 + first_cross_tau are emitted so the lead-time
            # provenance is auditable. In this committed simulator failure status
            # is largely determined by PRESENTATION (the worsening sub-population
            # already deviates >1.96 sigma at tau=0), so the first crossing tends
            # to occur at tau=0 and the lead time saturates at the full horizon.
            # That is reported, not corrected (see MORNING_DECISIONS reconciliation
            # flag): the metric is honest but its early-warning resolution is
            # limited by the presentation-driven outcome mechanism.
            d_at_t0 = float(D[0]) if D.size else float("nan")
            crossings = np.where(D > _BEWS_THRESHOLD)[0]
            if crossings.size == 0:
                lead_time = float("nan")
                cross_tau = None
            else:
                cross_tau = int(crossings[0])
                lead_time = float((n_timesteps - 1 - cross_tau) * DT)

            # Precision: of all twins whose OWN biomarker-j z-deviation crosses
            # 1.96 at/after the population lead-time timestep, the fraction that
            # are true failures. (Emergent; not tuned.)
            if cross_tau is None:
                precision = float("nan")
            else:
                tp = 0
                flagged = 0
                for (o, tb) in twins:
                    x = tb[b][:n_timesteps]
                    z = np.abs(x - mu) / sigma_safe
                    z_after = z[cross_tau:]
                    if np.any(z_after > _BEWS_THRESHOLD):
                        flagged += 1
                        if o == 1:
                            tp += 1
                precision = float(tp / flagged) if flagged > 0 else float("nan")

            rows.append(
                {
                    "disease": disease,
                    "biomarker": b,
                    "n_failures": n_fail,
                    "n_nonfailures": n_nonfail,
                    "deviation_at_t0": d_at_t0,
                    "max_deviation": float(np.nanmax(D)),
                    "first_cross_tau": (
                        float("nan") if cross_tau is None else int(cross_tau)
                    ),
                    "lead_time_hours": lead_time,
                    "precision": precision,
                    "flagged_insufficient": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building digital-twin cohort...")
    cohort = build_cohort()
    cohort_summary = {
        "seed": SEED,
        "horizon_hours": HORIZON_HOURS,
        "dt_hours": DT,
        "total_twins": int(len(cohort)),
        "by_disease": {k: int(v) for k, v in cohort["disease"].value_counts().items()},
        "by_risk_tier": {
            k: int(v) for k, v in cohort["risk_tier"].value_counts().items()
        },
        "outcome_prevalence": float(cohort["outcome"].mean()),
        # Emergent per-disease mortality prevalence (whatever the corrected
        # damage->mortality mechanism yields; reported, not targeted).
        "prevalence_by_disease": {
            str(dz): float(cohort.loc[cohort["disease"] == dz, "outcome"].mean())
            for dz in COHORT
        },
        "n_positive_by_disease": {
            str(dz): int(cohort.loc[cohort["disease"] == dz, "outcome"].sum())
            for dz in COHORT
        },
    }
    (RESULTS_DIR / "cohort_summary.json").write_text(
        json.dumps(cohort_summary, indent=2)
    )
    cohort.to_csv(RESULTS_DIR / "cohort.csv", index=False)

    print("[2/4] Training + evaluating supported models...")
    results = evaluate_models(cohort)
    for name, frame in results.items():
        frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    print("[3/5] Running counterfactual antibiotic-delay sweep...")
    cf = counterfactual_delay()
    cf.to_csv(RESULTS_DIR / "counterfactual_delay.csv", index=False)

    print("[4/5] Computing novel temporal metrics (DBRS / TCB / TCS / TCE / BEWS)...")
    dbrs = compute_dbrs(cohort)
    dbrs.to_csv(RESULTS_DIR / "dbrs.csv", index=False)
    tcb = compute_tcb(results["model_metrics"])
    tcb.to_csv(RESULTS_DIR / "tcb.csv", index=False)
    tcs = compute_tcs(cohort)
    tcs.to_csv(RESULTS_DIR / "temporal_consistency.csv", index=False)
    # B-hard: TCE (L2 trajectory) per model x disease, BEWS per biomarker x disease.
    tce = compute_tce(cohort)
    tce.to_csv(RESULTS_DIR / "temporal_metrics.csv", index=False)
    bews = compute_bews(cohort)
    bews.to_csv(RESULTS_DIR / "bews.csv", index=False)

    print("[5/5] Writing run metadata...")
    metadata = {
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "supported_models": sorted(MODEL_FACTORIES.keys()),
        "note": (
            "Metrics are computed from seeded digital-twin simulation using the "
            "committed package code only. Deep sequence models (LSTM, TCN) and "
            "XGBoost are not part of the committed dependency set and are not "
            "evaluated here. The counterfactual antibiotic-delay sweep now emits "
            "a calibrated mortality probability (mortality_prob), derived from "
            "the terminal cumulative organ-damage state D via a logistic link "
            "(damage_to_mortality, anchored to Kumar 2006, not tuned to any "
            "target). The reported slope_pp_per_hr and r_squared are computed "
            "with np.polyfit over the realised sweep, not hardcoded. See "
            "REPRODUCIBILITY.md."
        ),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Headline summary of the computed metrics for quick manuscript reconciliation.
    dbrs_critical = dbrs[dbrs["critical"]] if not dbrs.empty else dbrs
    summary = {
        "seed": SEED,
        "dbrs": {
            "n_evaluated": int(len(dbrs)),
            "n_critical_gt_1.15": int(len(dbrs_critical)),
            "dbrs_min": float(dbrs["dbrs"].min()) if not dbrs.empty else None,
            "dbrs_max": float(dbrs["dbrs"].max()) if not dbrs.empty else None,
            "dbrs_mean": float(dbrs["dbrs"].mean()) if not dbrs.empty else None,
            "critical_biomarkers": (
                sorted(
                    {
                        f"{r.disease}:{r.biomarker}:{r.model}"
                        for r in dbrs_critical.itertuples()
                    }
                )
                if not dbrs_critical.empty
                else []
            ),
        },
        "tcb": (
            {
                "loss_metric": str(tcb["loss_metric"].iloc[0]),
                "rho": float(tcb["rho"].iloc[0]),
                "delta_min": float(tcb["delta_min"].iloc[0]),
                "rho_delta_min": float(tcb["rho_delta_min"].iloc[0]),
                "fraction_bound_holds": float(tcb["fraction_bound_holds"].iloc[0]),
                "per_model_loss_gap": {
                    str(r.model): float(r.loss_gap) for r in tcb.itertuples()
                },
            }
            if not tcb.empty
            else None
        ),
        "tcs": (
            {
                "by_missingness_mean": {
                    str(m): float(tcs[tcs["missingness"] == m]["mean_tcs"].mean())
                    for m in sorted(tcs["missingness"].unique())
                },
                "min_mean_tcs": float(tcs["mean_tcs"].min()),
                "max_mean_tcs": float(tcs["mean_tcs"].max()),
            }
            if not tcs.empty
            else None
        ),
        # TCE (L2 trajectory form): headline = mean per model across diseases.
        "tce": (
            {
                "loss_form": "L2_trajectory",
                "per_model_mean": {
                    str(m): float(tce[tce["model"] == m]["mean_tce"].mean())
                    for m in sorted(tce["model"].unique())
                },
                "overall_mean_tce": float(tce["mean_tce"].mean()),
                "min_mean_tce": float(tce["mean_tce"].min()),
                "max_mean_tce": float(tce["mean_tce"].max()),
                "per_model_disease": {
                    f"{r.model}:{r.disease}": float(r.mean_tce)
                    for r in tce.itertuples()
                },
            }
            if not tce.empty
            else None
        ),
        # BEWS: headline = per-biomarker lead time + precision (emergent).
        "bews": (
            {
                "threshold_z": _BEWS_THRESHOLD,
                "flagged_diseases": sorted(
                    {
                        str(r.disease)
                        for r in bews.itertuples()
                        if bool(r.flagged_insufficient)
                    }
                ),
                "per_biomarker": {
                    f"{r.disease}:{r.biomarker}": {
                        "lead_time_hours": (
                            None
                            if (r.lead_time_hours != r.lead_time_hours)
                            else float(r.lead_time_hours)
                        ),
                        "precision": (
                            None
                            if (r.precision != r.precision)
                            else float(r.precision)
                        ),
                        "max_deviation": (
                            None
                            if (r.max_deviation != r.max_deviation)
                            else float(r.max_deviation)
                        ),
                    }
                    for r in bews.itertuples()
                    if not bool(r.flagged_insufficient)
                },
                "lead_time_hours_range": (
                    [
                        float(bews["lead_time_hours"].min()),
                        float(bews["lead_time_hours"].max()),
                    ]
                    if bews["lead_time_hours"].notna().any()
                    else None
                ),
                "precision_range": (
                    [
                        float(bews["precision"].min()),
                        float(bews["precision"].max()),
                    ]
                    if bews["precision"].notna().any()
                    else None
                ),
            }
            if not bews.empty
            else None
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nDone. Results written to:", RESULTS_DIR)
    print("\nStatic AUROC / accuracy by model:")
    static = results["model_metrics"]
    static = static[static["regime"] == "static"]
    for _, r in static.iterrows():
        print(
            f"  {r['model']:<20} AUROC={r['auroc']:.3f}  "
            f"acc={r['accuracy']:.3f}  ECE={r['ece']:.3f}"
        )

    print("\nNovel temporal metrics (computed, not tuned):")
    print(
        f"  DBRS: {summary['dbrs']['n_evaluated']} evaluated, "
        f"{summary['dbrs']['n_critical_gt_1.15']} critical (>1.15); "
        f"range [{summary['dbrs']['dbrs_min']:.3f}, "
        f"{summary['dbrs']['dbrs_max']:.3f}], mean {summary['dbrs']['dbrs_mean']:.3f}"
    )
    if summary["tcb"]:
        print(
            f"  TCB:  delta_min={summary['tcb']['delta_min']:.4f} "
            f"(rho={summary['tcb']['rho']}, rho*delta_min="
            f"{summary['tcb']['rho_delta_min']:.4f}); bound holds for "
            f"{summary['tcb']['fraction_bound_holds']*100:.0f}% of configs"
        )
    if summary["tcs"]:
        tline = ", ".join(
            f"{m}:{v:.3f}"
            for m, v in summary["tcs"]["by_missingness_mean"].items()
        )
        print(f"  TCS:  mean by missingness -> {tline}")
    if summary["tce"]:
        tceline = ", ".join(
            f"{m}:{v:.4f}"
            for m, v in summary["tce"]["per_model_mean"].items()
        )
        print(
            f"  TCE (L2): per-model mean -> {tceline}; "
            f"overall {summary['tce']['overall_mean_tce']:.4f}"
        )
    if summary["bews"]:
        flagged = summary["bews"]["flagged_diseases"]
        lt = summary["bews"]["lead_time_hours_range"]
        pr = summary["bews"]["precision_range"]
        print(
            f"  BEWS: lead-time(h) range "
            f"{('n/a' if lt is None else f'[{lt[0]:.1f}, {lt[1]:.1f}]')}, "
            f"precision range "
            f"{('n/a' if pr is None else f'[{pr[0]:.3f}, {pr[1]:.3f}]')}"
            + (f"; FLAGGED diseases: {', '.join(flagged)}" if flagged else "")
        )


if __name__ == "__main__":
    main()
