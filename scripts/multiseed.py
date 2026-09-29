"""Repeat the cohort + model evaluation over many seeds.

Every headline number in the manuscript comes from a single seeded run (seed 42). This script
re-runs the parts that the seed actually controls -- cohort generation, the 60/20/20 split and
all six model fits -- once per seed, so the between-family differences can be read against the
sampling variability of the simulator itself.

Only ``run_all.SEED`` is changed; the module's factories and split helpers read that global at
call time, so each seed reproduces the pipeline exactly as ``run_all.main`` would.

Outputs
  results/multiseed_metrics.csv   one row per (seed, model, regime)
  results/multiseed_summary.csv   mean / sd / 2.5-97.5 percentile per (model, regime, metric)
  results/multiseed_ranks.csv     per-seed rank of each model, per metric

Usage:  python scripts/multiseed.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_all  # noqa: E402

METRICS = ["auroc", "auprc", "accuracy", "ece", "brier"]


def run_seed(seed: int) -> pd.DataFrame:
    run_all.SEED = seed
    np.random.seed(seed)
    cohort = run_all.build_cohort()
    out = run_all.evaluate_models(cohort)["model_metrics"].copy()
    out.insert(0, "seed", seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="multi-seed repetition of the BASICS-CDSS pipeline")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    args = ap.parse_args()

    frames = []
    for k in range(args.seeds):
        seed = args.start + k
        print(f"[seed {seed}] ({k + 1}/{args.seeds})", flush=True)
        frames.append(run_seed(seed))
    rows = pd.concat(frames, ignore_index=True)

    res = REPO / "results"
    rows.to_csv(res / "multiseed_metrics.csv", index=False)

    recs = []
    for (model, regime), g in rows.groupby(["model", "regime"]):
        for m in METRICS:
            if m not in g:
                continue
            v = g[m].astype(float)
            recs.append({
                "model": model, "regime": regime, "metric": m, "n_seeds": len(v),
                "mean": v.mean(), "sd": v.std(ddof=1),
                "p2_5": v.quantile(0.025), "p97_5": v.quantile(0.975),
                "min": v.min(), "max": v.max(),
                "seed42": float(g.loc[g.seed == args.start, m].iloc[0]) if (g.seed == args.start).any() else float("nan"),
            })
    summary = pd.DataFrame(recs)
    summary.to_csv(res / "multiseed_summary.csv", index=False)

    rank_rows = []
    for (seed, regime), g in rows.groupby(["seed", "regime"]):
        for m in METRICS:
            if m not in g:
                continue
            asc = m in ("ece", "brier")  # lower is better
            order = g.sort_values(m, ascending=asc)["model"].tolist()
            for pos, model in enumerate(order, start=1):
                rank_rows.append({"seed": seed, "regime": regime, "metric": m,
                                  "model": model, "rank": pos})
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(res / "multiseed_ranks.csv", index=False)

    print("\nAUROC (static) over seeds:")
    s = summary[(summary.metric == "auroc") & (summary.regime == "static")]
    for _, r in s.sort_values("mean", ascending=False).iterrows():
        print(f"  {r['model']:20} mean {r['mean']:.3f}  sd {r['sd']:.3f}  "
              f"[{r['p2_5']:.3f}, {r['p97_5']:.3f}]  seed42 {r['seed42']:.3f}")

    print("\nHow often each model takes rank 1 (static AUROC):")
    top = ranks[(ranks.metric == "auroc") & (ranks.regime == "static") & (ranks["rank"] == 1)]
    for model, c in top["model"].value_counts().items():
        print(f"  {model:20} {c}/{rows.seed.nunique()}")
    print("\nwrote results/multiseed_{metrics,summary,ranks}.csv")


if __name__ == "__main__":
    main()
