"""Bootstrap 95% intervals for AUROC (static and temporal) and net benefit at p_t = 0.30.

Reuses the seeded split and trained models of run_all.py, and the same degradation seeds as
``evaluate_models``, so the point estimates equal those in results/model_metrics.csv and
results/decision_curve.csv. Test twins are resampled with replacement (2,000 resamples).

It also writes the decision curve over the pre-specified threshold range 0.10-0.40 with
paired bootstrap intervals for the net-benefit differences and the share of twins flagged.

These intervals resample the 200 test twins with the fitted models held fixed, so they
do not include training-set variability; the 20-seed standard deviations
(results/multiseed_summary.csv) carry that component.

    python scripts/bootstrap_ci.py   -> results/bootstrap_ci.csv, results/decision_curve_range.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

N_BOOT = 2000
THRESHOLD = 0.30
DCA_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def net_benefit(y, p, pt=THRESHOLD):
    treat = p >= pt
    n = len(y)
    tp = np.sum(treat & (y == 1))
    fp = np.sum(treat & (y == 0))
    return tp / n - fp / n * pt / (1 - pt)


def main():
    df = run_all.build_cohort()
    split = run_all._trained_split(df)
    te, y = split["te"], split["y"][split["te"]]
    tr = split["tr"]
    X_te_temp = run_all._degrade(split["X"][te], np.random.RandomState(run_all.SEED + 1),
                                 ref=split["X"][tr])
    Xs_te_temp = run_all._degrade_sequence(split["X_seq"][te], np.random.RandomState(run_all.SEED + 2),
                                           ref=split["X_seq"][tr])

    rng = np.random.RandomState(run_all.SEED)
    boots = [rng.randint(0, len(y), len(y)) for _ in range(N_BOOT)]
    boots = [b for b in boots if 0 < y[b].sum() < len(b)]

    rows = []
    for name, (kind, model) in split["models"].items():
        p_static = split["test_prob"][name]
        p_temp = model.predict_proba(X_te_temp if kind == "tabular" else Xs_te_temp)[:, 1]
        for metric, fn, p in (("auroc_static", roc_auc_score, p_static),
                              ("auroc_temporal", roc_auc_score, p_temp),
                              ("net_benefit_0.30", net_benefit, p_static)):
            est = fn(y, p)
            dist = np.array([fn(y[b], p[b]) for b in boots])
            rows.append({"model": name, "metric": metric, "estimate": est,
                         "ci_lo": np.percentile(dist, 2.5), "ci_hi": np.percentile(dist, 97.5)})
        drop = np.array([roc_auc_score(y[b], p_static[b]) - roc_auc_score(y[b], p_temp[b]) for b in boots])
        rows.append({"model": name, "metric": "auroc_drop",
                     "estimate": roc_auc_score(y, p_static) - roc_auc_score(y, p_temp),
                     "ci_lo": np.percentile(drop, 2.5), "ci_hi": np.percentile(drop, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(run_all.RESULTS_DIR / "bootstrap_ci.csv", index=False)
    print(out.round(3).to_string())

    # Decision curve over the pre-specified range p_t = 0.10-0.40 (step 0.05).
    # For each threshold: net benefit of every family with a 95% interval; the
    # difference from the best default strategy (max of treat-all and treat-none,
    # computed within each resample) and from the best family at that threshold
    # (identified on the full test fold), both with paired bootstrap intervals;
    # and the share of twins flagged (p >= p_t).
    probs = {name: split["test_prob"][name] for name in split["models"]}
    dca_rows = []
    for pt in DCA_THRESHOLDS:
        nb_full = {m: net_benefit(y, p, pt) for m, p in probs.items()}
        best = max(nb_full, key=nb_full.get)
        nb_all = net_benefit(y, np.ones_like(y, dtype=float), pt)
        for m, p in probs.items():
            nb_b = np.array([net_benefit(y[b], p[b], pt) for b in boots])
            dflt_b = np.array([max(net_benefit(y[b], np.ones(len(b)), pt), 0.0) for b in boots])
            best_b = np.array([net_benefit(y[b], probs[best][b], pt) for b in boots])
            d_def, d_best = nb_b - dflt_b, nb_b - best_b
            dca_rows.append({
                "model": m, "threshold": pt, "net_benefit": nb_full[m],
                "nb_ci_lo": np.percentile(nb_b, 2.5), "nb_ci_hi": np.percentile(nb_b, 97.5),
                "nb_treat_all": nb_all, "best_default_nb": max(nb_all, 0.0),
                "delta_nb_vs_default": nb_full[m] - max(nb_all, 0.0),
                "delta_vs_default_ci_lo": np.percentile(d_def, 2.5),
                "delta_vs_default_ci_hi": np.percentile(d_def, 97.5),
                "best_family": best, "delta_nb_vs_best_family": nb_full[m] - nb_full[best],
                "delta_vs_best_ci_lo": np.percentile(d_best, 2.5),
                "delta_vs_best_ci_hi": np.percentile(d_best, 97.5),
                "percent_flagged": float(np.mean(p >= pt) * 100.0),
                "prevalence": float(y.mean()),
            })
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(run_all.RESULTS_DIR / "decision_curve_range.csv", index=False)
    print(dca.round(3).to_string())


if __name__ == "__main__":
    main()
