"""Shared helpers for the 2026-10 revision analyses.

Used by ``redundancy_control.py`` (input-redundancy control) and
``onset_sensitivity.py`` (mid-horizon deterioration onset). Both evaluate the
same model families on several *input representations* under the canonical
static/temporal regimes of ``run_all.evaluate_models``:

  tabular families (LR / RF / GB / XGBoost)
    presentation    the t=0 feature row (the canonical tabular input)
    window_flat     the 13 x 12 observation window flattened to 156 columns
    window_summary  per-channel window mean, last value and last-minus-first (36 columns)
  sequence families (LSTM / TCN)
    window          the 13-step observation window (the canonical sequence input)
    presentation    the t=0 step only (a length-1 sequence)

Degradation is the canonical 20% MCAR + additive Gaussian noise (SD = 0.1 x feature
SD) with median imputation. Window inputs are degraded per timestep with
``run_all._degrade_sequence`` (the exact tensor the sequence models see), so the
window_flat / window_summary tabular controls receive the same redundant,
independently masked observations as the sequence models. The presentation inputs
are degraded with ``run_all._degrade`` (tabular, canonical) and
``run_all._degrade_sequence`` on the t=0 slice (sequence), i.e. one masked vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402
from run_all import brier_score, expected_calibration_error  # noqa: E402


def _flat(Xs: np.ndarray) -> np.ndarray:
    return Xs.reshape(Xs.shape[0], -1)


def _summary(Xs: np.ndarray) -> np.ndarray:
    return np.hstack([Xs.mean(axis=1), Xs[:, -1, :], Xs[:, -1, :] - Xs[:, 0, :]])


def evaluate_inputs(X: np.ndarray, X_seq: np.ndarray, y: np.ndarray, seed: int,
                    groups: np.ndarray | None = None) -> list[dict]:
    """Fit every family on every input representation; return one row per
    (model, input, regime). Split and degradation seeds follow ``evaluate_models``
    exactly, so the canonical (tabular, presentation) and (sequence, window) rows
    equal ``results/multiseed_metrics.csv`` for the same seed.

    If ``groups`` (e.g. disease) is given, each row also carries the mean of the
    within-group AUROCs, which removes discrimination that comes only from
    between-group differences in base rate."""
    run_all.SEED = seed  # factories read the module global at call time
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.6 * n)], idx[int(0.6 * n) + int(0.2 * n):]
    y_tr, y_te = y[tr], y[te]
    g_te = None if groups is None else np.asarray(groups)[te]

    X_te_t = run_all._degrade(X[te], np.random.RandomState(seed + 1))
    Xs_te_t = run_all._degrade_sequence(X_seq[te], np.random.RandomState(seed + 2))
    Xs0_te_t = run_all._degrade_sequence(X_seq[te][:, :1, :], np.random.RandomState(seed + 2))

    conds = []
    for name, factory in run_all.MODEL_FACTORIES.items():
        conds += [
            (name, "tabular", "presentation", factory, X[tr], X[te], X_te_t),
            (name, "tabular", "window_flat", factory,
             _flat(X_seq[tr]), _flat(X_seq[te]), _flat(Xs_te_t)),
            (name, "tabular", "window_summary", factory,
             _summary(X_seq[tr]), _summary(X_seq[te]), _summary(Xs_te_t)),
        ]
    for name, factory in run_all.SEQUENCE_MODEL_FACTORIES.items():
        conds += [
            (name, "sequence", "window", factory, X_seq[tr], X_seq[te], Xs_te_t),
            (name, "sequence", "presentation", factory,
             X_seq[tr][:, :1, :], X_seq[te][:, :1, :], Xs0_te_t),
        ]

    rows = []
    for name, kind, inp, factory, Xtr, Xte, Xte_t in conds:
        model = factory()
        model.fit(Xtr, y_tr)
        for regime, Xe in (("static", Xte), ("temporal", Xte_t)):
            p = model.predict_proba(Xe)[:, 1]
            extra = {}
            if g_te is not None:
                within = [roc_auc_score(y_te[g_te == g], p[g_te == g]) for g in np.unique(g_te)
                          if len(np.unique(y_te[g_te == g])) == 2]
                extra["auroc_within_group_mean"] = float(np.mean(within))
            rows.append({**extra,
                "seed": seed, "model": name, "kind": kind, "input": inp, "regime": regime,
                "auroc": float(roc_auc_score(y_te, p)),
                "brier": float(brier_score(y_te, p)),
                "ece": float(expected_calibration_error(y_te, p)),
            })
    return rows


def summarise_drops(rows) -> "object":
    """Per (model, input): mean/sd static AUROC, mean/sd AUROC drop, seeds with a drop."""
    import pandas as pd

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["seed", "model", "kind", "input"], columns="regime",
                          values="auroc").reset_index()
    wide["drop"] = wide["static"] - wide["temporal"]
    out = (wide.groupby(["model", "kind", "input"])
           .agg(n_seeds=("seed", "nunique"),
                static_auroc_mean=("static", "mean"), static_auroc_sd=("static", "std"),
                temporal_auroc_mean=("temporal", "mean"),
                drop_mean=("drop", "mean"), drop_sd=("drop", "std"),
                n_seeds_drop_gt0=("drop", lambda v: int((v > 0).sum())))
           .reset_index())
    return wide, out
