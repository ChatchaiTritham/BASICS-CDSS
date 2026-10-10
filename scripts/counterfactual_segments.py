"""Segment slopes and paired total effect for the antibiotic-delay sweep.

The full-sweep linear fit in results/counterfactual_delay.csv averages over a plateau
(mortality flattens after ~12 h). From the per-twin sweep
(results/counterfactual_delay_per_twin.csv, written by run_all.py) this script reports:

  * the least-squares slope over the 1-9 h segment and over the 9-18 h segment
    (fitted to the cohort-mean mortality at each delay), in percentage points per hour;
  * the paired total effect, mean over twins of p(18 h) - p(1 h), in percentage points;
  * the full-sweep slope (reproduces counterfactual_delay.csv).

95% intervals: 2,000 bootstrap resamples of the 120 twins (paired across delays).

    python scripts/counterfactual_segments.py -> results/counterfactual_segments.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

N_BOOT = 2000


def _slope(delays, means, lo, hi):
    m = (delays >= lo) & (delays <= hi)
    return float(np.polyfit(delays[m], means[m], 1)[0] * 100.0)


def main() -> None:
    long = pd.read_csv(run_all.RESULTS_DIR / "counterfactual_delay_per_twin.csv")
    wide = long.pivot(index="twin", columns="antibiotic_delay_hours", values="mortality_prob")
    delays = wide.columns.to_numpy(dtype=float)
    P = wide.to_numpy()  # twins x delays

    def stats(Pm):
        mu = Pm.mean(axis=0)
        return {
            "slope_1_9h_pp_per_hr": _slope(delays, mu, 1, 9),
            "slope_9_18h_pp_per_hr": _slope(delays, mu, 9, 18),
            "slope_full_pp_per_hr": _slope(delays, mu, 1, 18),
            "total_effect_18h_minus_1h_pp": float((Pm[:, -1] - Pm[:, 0]).mean() * 100.0),
        }

    est = stats(P)
    rng = np.random.RandomState(run_all.SEED)
    boots = [stats(P[rng.randint(0, len(P), len(P))]) for _ in range(N_BOOT)]
    rows = []
    for k, v in est.items():
        dist = np.array([b[k] for b in boots])
        rows.append({"quantity": k, "estimate": v, "ci_lo": float(np.percentile(dist, 2.5)),
                     "ci_hi": float(np.percentile(dist, 97.5)), "n_twins": len(P),
                     "n_boot": N_BOOT})
    frac_harmed = float(((P[:, -1] - P[:, 0]) > 0).mean())
    rows.append({"quantity": "frac_twins_mortality_higher_at_18h", "estimate": frac_harmed,
                 "ci_lo": np.nan, "ci_hi": np.nan, "n_twins": len(P), "n_boot": 0})
    out = pd.DataFrame(rows)
    out.to_csv(run_all.RESULTS_DIR / "counterfactual_segments.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
