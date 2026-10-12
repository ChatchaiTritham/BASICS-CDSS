"""One-at-a-time parameter sensitivity of the two headline findings.

Each disease-model parameter below is multiplied by 0.5 and by 1.5 (one at a time,
all others at their committed defaults); for every setting the full cohort and all
six families are rebuilt and evaluated with ``run_all.evaluate_models`` over the
same seeds. The baseline setting with seeds 42.. reproduces ``multiseed.py``.

Parameters (``run_all.MODEL_PARAMS``):
  sepsis   infection_growth_rate (r_S), k_damage (k_D), theta_damage (theta_D)
  ards     k_baro (lung-damage gain)
  cardiac  k_salvage (infarct gain), salvage_lambda (salvage-decay rate)

Reported per setting: prevalence, oracle AUROC (the simulator's own p(death)),
and the two headline findings -- (1) how often the static-AUROC leader is also the
best-calibrated family and the mean per-seed Spearman correlation of the AUROC and
ECE rankings; (2) the mean static-to-temporal AUROC drop per family.

Outputs
  results/sensitivity_metrics.csv   setting x seed x model x regime
  results/sensitivity_cohort.csv    setting x seed: prevalence, oracle AUROC
  results/sensitivity_summary.csv   one row per setting

Usage:  python scripts/sensitivity.py [--seeds 10] [--start 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

DEFAULTS = {
    ("sepsis", "infection_growth_rate"): 0.15,
    ("sepsis", "k_damage"): 1.0,
    ("sepsis", "theta_damage"): 0.2,
    ("ards", "k_baro"): 0.3,
    ("cardiac", "k_salvage"): 1.0,
    ("cardiac", "salvage_lambda"): 0.12,
}
FACTORS = (0.5, 1.5)
FAMILIES = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lstm", "tcn"]


def settings():
    yield "baseline", None, None, 1.0
    for (dz, par), val in DEFAULTS.items():
        for f in FACTORS:
            yield f"{dz}.{par}x{f}", dz, par, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start", type=int, default=42)
    args = ap.parse_args()

    metric_rows, cohort_rows = [], []
    for label, dz, par, f in settings():
        run_all.MODEL_PARAMS = {"sepsis": {}, "ards": {}, "cardiac": {}}
        if dz is not None:
            run_all.MODEL_PARAMS[dz][par] = DEFAULTS[(dz, par)] * f
        for k in range(args.seeds):
            seed = args.start + k
            print(f"[{label}] seed {seed}", flush=True)
            run_all.SEED = seed
            np.random.seed(seed)
            cohort = run_all.build_cohort()
            res = run_all.evaluate_models(cohort)
            mm = res["model_metrics"]
            for r in mm.itertuples():
                metric_rows.append({"setting": label, "disease": dz, "parameter": par,
                                    "factor": f, "seed": seed, "model": r.model,
                                    "regime": r.regime, "auroc": r.auroc, "ece": r.ece})
            orc = res["oracle_metrics"].iloc[0]
            cohort_rows.append({
                "setting": label, "disease": dz, "parameter": par, "factor": f, "seed": seed,
                "prevalence": float(cohort["outcome"].mean()),
                **{f"prevalence_{d}": float(cohort.loc[cohort.disease == d, "outcome"].mean())
                   for d in run_all.COHORT},
                "oracle_auroc": float(orc["auroc"]),
                "oracle_auroc_within_disease": float(orc["auroc_within_disease"]),
            })
    run_all.MODEL_PARAMS = {"sepsis": {}, "ards": {}, "cardiac": {}}

    met = pd.DataFrame(metric_rows)
    coh = pd.DataFrame(cohort_rows)
    res_dir = run_all.RESULTS_DIR
    met.to_csv(res_dir / "sensitivity_metrics.csv", index=False)
    coh.to_csv(res_dir / "sensitivity_cohort.csv", index=False)

    summ = []
    for label, g in met.groupby("setting", sort=False):
        st = g[g.regime == "static"]
        A = st.pivot(index="seed", columns="model", values="auroc")[FAMILIES]
        E = st.pivot(index="seed", columns="model", values="ece")[FAMILIES]
        match = int(np.sum(A.to_numpy().argmax(1) == E.to_numpy().argmin(1)))
        rho = float(np.mean([spearmanr(-A.loc[s], E.loc[s]).statistic for s in A.index]))
        w = g.pivot_table(index=["seed", "model"], columns="regime", values="auroc").reset_index()
        w["drop"] = w["static"] - w["temporal"]
        drops = w.groupby("model")["drop"].mean()
        D = w.pivot(index="seed", columns="model", values="drop")
        trees_gt_lstm = float(np.mean([
            (D.loc[s, ["random_forest", "gradient_boosting", "xgboost"]] > D.loc[s, "lstm"]).all()
            for s in D.index]))
        c = coh[coh.setting == label]
        summ.append({
            "setting": label, "disease": g["disease"].iloc[0], "parameter": g["parameter"].iloc[0],
            "factor": g["factor"].iloc[0], "n_seeds": A.shape[0],
            "prevalence_mean": c["prevalence"].mean(),
            **{f"prevalence_{d}_mean": c[f"prevalence_{d}"].mean() for d in run_all.COHORT},
            "oracle_auroc_mean": c["oracle_auroc"].mean(),
            "n_leader_best_calibrated": match, "mean_spearman_auroc_ece": rho,
            **{f"drop_{m}": float(drops[m]) for m in FAMILIES},
            "frac_seeds_all_trees_drop_more_than_lstm": trees_gt_lstm,
        })
    pd.DataFrame(summ).to_csv(res_dir / "sensitivity_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string())


if __name__ == "__main__":
    main()
