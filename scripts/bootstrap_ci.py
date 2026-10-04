"""Bootstrap 95% intervals for AUROC (static and temporal) and net benefit at p_t = 0.30.

Reuses the seeded split and trained models of run_all.py, and the same degradation seeds as
``evaluate_models``, so the point estimates equal those in results/model_metrics.csv and
results/decision_curve.csv. Test twins are resampled with replacement (2,000 resamples).

    python scripts/bootstrap_ci.py   -> results/bootstrap_ci.csv
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
    X_te_temp = run_all._degrade(split["X"][te], np.random.RandomState(run_all.SEED + 1))
    Xs_te_temp = run_all._degrade_sequence(split["X_seq"][te], np.random.RandomState(run_all.SEED + 2))

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


if __name__ == "__main__":
    main()
