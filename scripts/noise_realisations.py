"""Noise-sensitivity sweep averaged over independent noise realisations.

``run_all.compute_noise_sensitivity`` scores each (distribution, scale) point on a single
noise draw. This script re-scores the seed-42 trained models (``run_all._trained_split``)
on R = 30 independent draws per point (same noise operator ``run_all._add_noise``) and
reports the mean, SD and 2.5/97.5 percentiles of AUROC and of the AUROC loss.

    python scripts/noise_realisations.py -> results/noise_sensitivity_realisations.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

R = 30


def main() -> None:
    np.random.seed(run_all.SEED)
    df = run_all.build_cohort()
    split = run_all._trained_split(df)
    te = split["te"]
    y = split["y"][te]
    X_te, Xs_te = split["X"][te], split["X_seq"][te]
    rows = []
    for dist in ("gaussian", "t3"):
        for scale in (1.0, 2.0, 3.0):
            for name, (kind, model) in split["models"].items():
                clean = float(roc_auc_score(y, split["test_prob"][name]))
                aucs = []
                for r in range(R):
                    rng = np.random.RandomState(run_all.SEED + 100_000 + 1_000 * r
                                                + int(scale * 10) + (0 if dist == "gaussian" else 500))
                    if kind == "tabular":
                        Xp = run_all._add_noise(X_te, rng, scale, dist)
                    else:
                        n, T, d = Xs_te.shape
                        Xp = run_all._add_noise(Xs_te.reshape(-1, d), rng, scale, dist).reshape(n, T, d)
                    aucs.append(float(roc_auc_score(y, model.predict_proba(Xp)[:, 1])))
                a = np.array(aucs)
                rows.append({"model": name, "kind": kind, "noise_dist": dist, "noise_scale": scale,
                             "n_realisations": R, "auroc_clean": clean,
                             "auroc_mean": a.mean(), "auroc_sd": a.std(ddof=1),
                             "auroc_p2_5": np.percentile(a, 2.5), "auroc_p97_5": np.percentile(a, 97.5),
                             "auroc_loss_mean": clean - a.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(run_all.RESULTS_DIR / "noise_sensitivity_realisations.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
