"""Input-redundancy control for the tabular-vs-sequence fragility comparison.

In the canonical run the tabular families read one presentation vector while the
sequence families read 13 hourly observations, and the degraded regime masks 20% of
entries independently. A sequence model can therefore recover a masked value from
neighbouring timesteps; a tabular model cannot. This script separates architecture
from input redundancy by crossing family x input (see ``revision_common``):
tabular families on the flattened / summarised 12-h window, and sequence families
restricted to t=0. Repeated over the same 20 seeds as ``multiseed.py``.

Outputs
  results/redundancy_control_metrics.csv   seed x model x input x regime
  results/redundancy_control_summary.csv   per (model, input): static AUROC, AUROC drop

Usage:  python scripts/redundancy_control.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import revision_common as rc
from revision_common import run_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for k in range(args.seeds):
        seed = args.start + k
        print(f"[seed {seed}] ({k + 1}/{args.seeds})", flush=True)
        run_all.SEED = seed
        np.random.seed(seed)
        cohort = run_all.build_cohort()
        X = cohort[run_all._FEATURE_COLUMNS].to_numpy(dtype=float)
        y = cohort["outcome"].to_numpy(dtype=int)
        X_seq = run_all.build_trajectory_tensor(cohort)
        rows += rc.evaluate_inputs(X, X_seq, y, seed, groups=cohort["disease"].to_numpy())

    res = run_all.RESULTS_DIR
    pd.DataFrame(rows).to_csv(res / "redundancy_control_metrics.csv", index=False)
    _, summary = rc.summarise_drops(rows)
    summary.to_csv(res / "redundancy_control_summary.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
