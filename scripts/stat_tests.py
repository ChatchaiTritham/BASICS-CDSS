"""Across-seed tests and summaries for the family comparisons (20 seeds, 42-61).

All inputs are written by ``multiseed.py``, ``redundancy_control.py`` and
``onset_sensitivity.py``; nothing is recomputed from the simulator here.

1. Friedman omnibus test over the seeds (blocks) for the six families, on static
   AUROC and on the static-to-temporal AUROC drop. Post-hoc comparisons use
   Holm-adjusted Wilcoxon signed-rank tests only (Benavoli et al., 2016, advise
   against the mean-rank Nemenyi test), each reported with an effect size: the
   median paired difference with a seed-bootstrap 95% interval and the
   matched-pairs rank-biserial correlation.
2. Rank agreement between discrimination and calibration (raw ECE and ECE after
   Platt / isotonic recalibration): number of seeds in which the static-AUROC
   leader is also the best-calibrated family, and the mean per-seed Spearman
   correlation of the two rankings, against a permutation null in which the ECE
   values are shuffled across families within each seed (10,000 permutations).
3. Input-redundancy control and onset sensitivity: per family, Wilcoxon test of
   the AUROC drop under the alternative input versus the canonical input, and the
   ratio of mean drops (window / presentation) with a seed-bootstrap interval.
4. Oracle gap: the simulator's own p(death) scored as a predictor, and each
   family's AUROC shortfall from it.
5. Seed summaries of recalibration and conformal coverage.

    python scripts/stat_tests.py -> results/stat_friedman.csv, stat_pairwise.csv,
        rank_agreement.csv, redundancy_ratio.csv, oracle_gap.csv,
        multiseed_recalibration_summary.csv, multiseed_conformal_summary.csv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

RES = run_all.RESULTS_DIR
N_BOOT = 2000
N_PERM = 10000
FAMILIES = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lstm", "tcn"]


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of paired differences d (zeros dropped)."""
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _boot_median_ci(d: np.ndarray, rng: np.random.RandomState) -> tuple:
    n = len(d)
    meds = np.array([np.median(d[rng.randint(0, n, n)]) for _ in range(N_BOOT)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _drops(df, keys):
    w = df.pivot_table(index=keys, columns="regime", values="auroc").reset_index()
    w["drop"] = w["static"] - w["temporal"]
    return w


def family_tests():
    ms = pd.read_csv(RES / "multiseed_metrics.csv")
    w = _drops(ms, ["seed", "model"])
    rng = np.random.RandomState(run_all.SEED)
    fr_rows, pw_rows = [], []
    for metric, higher_better in (("static", True), ("drop", False)):
        M = w.pivot(index="seed", columns="model", values=metric)[FAMILIES]
        k, n = M.shape[1], M.shape[0]
        stat, p = friedmanchisquare(*[M[m].to_numpy() for m in FAMILIES])
        ranks = (-M if higher_better else M).rank(axis=1).mean()
        name = "static_auroc" if metric == "static" else "auroc_drop"
        fr_rows.append({"metric": name, "n_seeds": n, "k": k, "friedman_chi2": stat,
                        "p_value": p, **{f"mean_rank_{m}": ranks[m] for m in FAMILIES}})
        pairs = list(itertools.combinations(FAMILIES, 2))
        ps = [wilcoxon(M[a], M[b]).pvalue for a, b in pairs]
        for (a, b), pv, ph in zip(pairs, ps, _holm(ps)):
            d = (M[a] - M[b]).to_numpy()
            lo, hi = _boot_median_ci(d, rng)
            pw_rows.append({"analysis": "multiseed", "metric": name, "a": a, "b": b,
                            "median_diff_a_minus_b": float(np.median(d)),
                            "median_diff_ci_lo": lo, "median_diff_ci_hi": hi,
                            "rank_biserial": _rank_biserial(d),
                            "wilcoxon_p": pv, "holm_p": ph,
                            "holm_significant_0.05": bool(ph < 0.05)})
    return fr_rows, pw_rows


def input_tests(fname, analysis):
    path = RES / fname
    if not path.exists():
        return [], []
    w = _drops(pd.read_csv(path), ["seed", "model", "kind", "input"])
    rows, ratio_rows = [], []
    rng = np.random.RandomState(run_all.SEED + 1)
    for model, g in w.groupby("model"):
        kind = g["kind"].iloc[0]
        canon = "presentation" if kind == "tabular" else "window"
        base = g[g["input"] == canon].set_index("seed")["drop"]
        for inp in sorted(set(g["input"]) - {canon}):
            alt = g[g["input"] == inp].set_index("seed")["drop"].reindex(base.index)
            d = (alt - base).to_numpy()
            lo, hi = _boot_median_ci(d, rng)
            rows.append({"analysis": analysis, "metric": "auroc_drop", "a": f"{model}:{inp}",
                         "b": f"{model}:{canon}",
                         "median_diff_a_minus_b": float(np.median(d)),
                         "median_diff_ci_lo": lo, "median_diff_ci_hi": hi,
                         "rank_biserial": _rank_biserial(d),
                         "wilcoxon_p": float(wilcoxon(alt, base).pvalue),
                         "holm_p": np.nan, "holm_significant_0.05": np.nan})
            # Ratio of mean drops, window-type input over presentation input.
            num, den = (alt, base) if kind == "tabular" else (base, alt)
            num_lab, den_lab = (inp, canon) if kind == "tabular" else (canon, inp)
            a_, b_ = num.to_numpy(), den.to_numpy()
            n = len(a_)
            boots = []
            for _ in range(N_BOOT):
                ix = rng.randint(0, n, n)
                boots.append(a_[ix].mean() / b_[ix].mean())
            ratio_rows.append({"analysis": analysis, "model": model,
                               "numerator_input": num_lab, "denominator_input": den_lab,
                               "mean_drop_numerator": float(a_.mean()),
                               "mean_drop_denominator": float(b_.mean()),
                               "ratio": float(a_.mean() / b_.mean()),
                               "ratio_ci_lo": float(np.percentile(boots, 2.5)),
                               "ratio_ci_hi": float(np.percentile(boots, 97.5)),
                               "n_seeds": n})
    ph = _holm([r["wilcoxon_p"] for r in rows])
    for r, p in zip(rows, ph):
        r["holm_p"] = p
        r["holm_significant_0.05"] = bool(p < 0.05)
    return rows, ratio_rows


def _agreement_stats(A: np.ndarray, E: np.ndarray) -> tuple:
    """A, E: seeds x families (AUROC, ECE). Returns (n leader==best-calibrated, mean rho)."""
    match = int(np.sum(np.argmax(A, axis=1) == np.argmin(E, axis=1)))
    rhos = [spearmanr(-A[i], E[i]).statistic for i in range(A.shape[0])]
    return match, float(np.mean(rhos))


def rank_agreement():
    ms = pd.read_csv(RES / "multiseed_metrics.csv")
    st = ms[ms.regime == "static"]
    A = st.pivot(index="seed", columns="model", values="auroc")[FAMILIES]
    ece_sets = {"raw": st.pivot(index="seed", columns="model", values="ece")[FAMILIES]}
    rc_path = RES / "multiseed_recalibration.csv"
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        rc = rc[rc.regime == "static"]
        for method in sorted(rc["method"].unique()):
            ece_sets[method] = (rc[rc.method == method]
                                .pivot(index="seed", columns="model", values="ece")[FAMILIES])
    rng = np.random.RandomState(run_all.SEED + 2)
    out = []
    for label, E in ece_sets.items():
        Av, Ev = A.to_numpy(), E.loc[A.index].to_numpy()
        obs_match, obs_rho = _agreement_stats(Av, Ev)
        null_m, null_r = [], []
        for _ in range(N_PERM):
            Ep = np.array([row[rng.permutation(len(row))] for row in Ev])
            m_, r_ = _agreement_stats(Av, Ep)
            null_m.append(m_)
            null_r.append(r_)
        null_m, null_r = np.array(null_m), np.array(null_r)
        out.append({
            "ece_version": label, "n_seeds": len(Av), "n_families": Av.shape[1],
            "n_leader_best_calibrated": obs_match,
            "null_expected_count": float(null_m.mean()),
            "p_count_le_obs": float(np.mean(null_m <= obs_match)),
            "p_count_ge_obs": float(np.mean(null_m >= obs_match)),
            "mean_spearman_auroc_vs_ece_rank": obs_rho,
            "null_mean_spearman_p2_5": float(np.percentile(null_r, 2.5)),
            "null_mean_spearman_p97_5": float(np.percentile(null_r, 97.5)),
            "p_spearman_two_sided": float(np.mean(np.abs(null_r) >= abs(obs_rho))),
            "mean_ece_across_families": float(Ev.mean()),
            "mean_ece_spread_max_minus_min": float((Ev.max(axis=1) - Ev.min(axis=1)).mean()),
            "n_permutations": N_PERM,
        })
    return out


def oracle_gap():
    path = RES / "multiseed_oracle.csv"
    if not path.exists():
        return []
    orc = pd.read_csv(path).set_index("seed")
    ms = pd.read_csv(RES / "multiseed_metrics.csv")
    st = ms[ms.regime == "static"]
    rng = np.random.RandomState(run_all.SEED + 3)
    rows = [{"model": "oracle", "auroc_mean": orc["auroc"].mean(), "auroc_sd": orc["auroc"].std(),
             "auroc_within_disease_mean": orc["auroc_within_disease"].mean(),
             "ece_mean": orc["ece"].mean(), "cal_slope_mean": orc["cal_slope"].mean(),
             "gap_mean": 0.0, "gap_ci_lo": 0.0, "gap_ci_hi": 0.0,
             "gap_within_disease_mean": 0.0, "n_seeds_model_ge_oracle": np.nan}]
    for m in FAMILIES:
        g = st[st.model == m].set_index("seed").loc[orc.index]
        gap = (orc["auroc"] - g["auroc"]).to_numpy()
        gap_w = (orc["auroc_within_disease"] - g["auroc_within_disease"]).to_numpy()
        n = len(gap)
        bm = [gap[rng.randint(0, n, n)].mean() for _ in range(N_BOOT)]
        rows.append({"model": m, "auroc_mean": g["auroc"].mean(), "auroc_sd": g["auroc"].std(),
                     "auroc_within_disease_mean": g["auroc_within_disease"].mean(),
                     "ece_mean": g["ece"].mean(), "cal_slope_mean": g["cal_slope"].mean(),
                     "gap_mean": float(gap.mean()), "gap_ci_lo": float(np.percentile(bm, 2.5)),
                     "gap_ci_hi": float(np.percentile(bm, 97.5)),
                     "gap_within_disease_mean": float(gap_w.mean()),
                     "n_seeds_model_ge_oracle": int(np.sum(gap <= 0))})
    return rows


def seed_summaries():
    out = {}
    rc_path = RES / "multiseed_recalibration.csv"
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        ms = pd.read_csv(RES / "multiseed_metrics.csv")
        raw = ms.assign(method="raw")[["seed", "model", "regime", "method", "auroc", "ece",
                                       "brier", "cal_slope", "cal_intercept"]]
        allc = pd.concat([raw, rc[raw.columns]], ignore_index=True)
        out["multiseed_recalibration_summary.csv"] = (
            allc.groupby(["model", "regime", "method"])
            .agg(n_seeds=("seed", "nunique"), auroc_mean=("auroc", "mean"),
                 ece_mean=("ece", "mean"), ece_sd=("ece", "std"),
                 brier_mean=("brier", "mean"), cal_slope_mean=("cal_slope", "mean"),
                 cal_slope_sd=("cal_slope", "std"),
                 cal_intercept_mean=("cal_intercept", "mean"))
            .reset_index())
    cf_path = RES / "multiseed_conformal.csv"
    if cf_path.exists():
        cf = pd.read_csv(cf_path)
        out["multiseed_conformal_summary.csv"] = (
            cf.groupby(["model", "target_coverage", "method", "regime"])
            .agg(n_seeds=("seed", "nunique"),
                 coverage_mean=("empirical_coverage", "mean"),
                 coverage_min=("empirical_coverage", "min"),
                 coverage_deaths_mean=("coverage_deaths", "mean"),
                 coverage_deaths_min=("coverage_deaths", "min"),
                 coverage_survivors_mean=("coverage_survivors", "mean"),
                 set_size_mean=("avg_set_size", "mean"),
                 singleton_mean=("singleton_fraction", "mean"))
            .reset_index())
        s = out["multiseed_conformal_summary.csv"]
        s["n_seeds_below_target"] = [
            int((cf[(cf.model == r.model) & (cf.target_coverage == r.target_coverage)
                    & (cf.method == r.method) & (cf.regime == r.regime)]
                 ["empirical_coverage"] < r.target_coverage).sum())
            for r in s.itertuples()]
        s["n_seeds_deaths_below_target"] = [
            int((cf[(cf.model == r.model) & (cf.target_coverage == r.target_coverage)
                    & (cf.method == r.method) & (cf.regime == r.regime)]
                 ["coverage_deaths"] < r.target_coverage).sum())
            for r in s.itertuples()]
    return out


def main() -> None:
    fr, pw = family_tests()
    rows_r, ratio_r = input_tests("redundancy_control_metrics.csv", "redundancy_control")
    rows_o, ratio_o = input_tests("onset_sensitivity_metrics.csv", "onset_sensitivity")
    pw += rows_r + rows_o
    pd.DataFrame(fr).to_csv(RES / "stat_friedman.csv", index=False)
    pd.DataFrame(pw).to_csv(RES / "stat_pairwise.csv", index=False)
    pd.DataFrame(ratio_r + ratio_o).to_csv(RES / "redundancy_ratio.csv", index=False)
    ra = pd.DataFrame(rank_agreement())
    ra.to_csv(RES / "rank_agreement.csv", index=False)
    og = pd.DataFrame(oracle_gap())
    if not og.empty:
        og.to_csv(RES / "oracle_gap.csv", index=False)
    for fname, frame in seed_summaries().items():
        frame.to_csv(RES / fname, index=False)
    print(pd.DataFrame(fr).round(4).T.to_string())
    print(ra.round(4).T.to_string())
    print(og.round(4).to_string())
    print(pd.DataFrame(ratio_r).round(3).to_string())


if __name__ == "__main__":
    main()
