"""Across-seed significance tests for the family comparisons.

1. Friedman test over the 20 seeds (blocks) for six families, on static AUROC and on the
   static-to-temporal AUROC drop (results/multiseed_metrics.csv), with Nemenyi critical
   difference (Demsar 2006, q_0.05 for k = 6 is 2.850) and Holm-adjusted pairwise
   Wilcoxon signed-rank tests.
2. Input-redundancy control and onset sensitivity: per family, Wilcoxon signed-rank test
   of the AUROC drop under the alternative input versus the canonical input, across seeds
   (results/redundancy_control_metrics.csv, results/onset_sensitivity_metrics.csv).

    python scripts/stat_tests.py -> results/stat_friedman.csv, results/stat_pairwise.csv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

RES = run_all.RESULTS_DIR
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _drops(df, keys):
    w = df.pivot_table(index=keys, columns="regime", values="auroc").reset_index()
    w["drop"] = w["static"] - w["temporal"]
    return w


def family_tests():
    ms = pd.read_csv(RES / "multiseed_metrics.csv")
    w = _drops(ms, ["seed", "model"])
    fr_rows, pw_rows = [], []
    for metric, higher_better in (("static", True), ("drop", False)):
        M = w.pivot(index="seed", columns="model", values=metric)
        models = list(M.columns)
        k, n = len(models), len(M)
        stat, p = friedmanchisquare(*[M[m].to_numpy() for m in models])
        ranks = (-M if higher_better else M).rank(axis=1).mean()
        cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
        fr_rows.append({"metric": "static_auroc" if metric == "static" else "auroc_drop",
                        "n_seeds": n, "k": k, "friedman_chi2": stat, "p_value": p,
                        "nemenyi_cd": cd,
                        **{f"mean_rank_{m}": ranks[m] for m in models}})
        pairs = list(itertools.combinations(models, 2))
        ps = [wilcoxon(M[a], M[b]).pvalue for a, b in pairs]
        for (a, b), pv, ph in zip(pairs, ps, _holm(ps)):
            pw_rows.append({"analysis": "multiseed", "metric": fr_rows[-1]["metric"],
                            "a": a, "b": b, "median_diff_a_minus_b": float(np.median(M[a] - M[b])),
                            "wilcoxon_p": pv, "holm_p": ph,
                            "nemenyi_rank_diff": abs(ranks[a] - ranks[b]),
                            "nemenyi_significant": bool(abs(ranks[a] - ranks[b]) > cd)})
    return fr_rows, pw_rows


def input_tests(fname, analysis):
    path = RES / fname
    if not path.exists():
        return []
    w = _drops(pd.read_csv(path), ["seed", "model", "kind", "input"])
    rows = []
    for model, g in w.groupby("model"):
        kind = g["kind"].iloc[0]
        canon = "presentation" if kind == "tabular" else "window"
        base = g[g["input"] == canon].set_index("seed")["drop"]
        for inp in sorted(set(g["input"]) - {canon}):
            alt = g[g["input"] == inp].set_index("seed")["drop"].reindex(base.index)
            rows.append({"analysis": analysis, "metric": "auroc_drop", "a": f"{model}:{inp}",
                         "b": f"{model}:{canon}",
                         "median_diff_a_minus_b": float(np.median(alt - base)),
                         "wilcoxon_p": float(wilcoxon(alt, base).pvalue),
                         "holm_p": np.nan, "nemenyi_rank_diff": np.nan,
                         "nemenyi_significant": np.nan})
    ph = _holm([r["wilcoxon_p"] for r in rows])
    for r, p in zip(rows, ph):
        r["holm_p"] = p
    return rows


def main() -> None:
    fr, pw = family_tests()
    pw += input_tests("redundancy_control_metrics.csv", "redundancy_control")
    pw += input_tests("onset_sensitivity_metrics.csv", "onset_sensitivity")
    pd.DataFrame(fr).to_csv(RES / "stat_friedman.csv", index=False)
    pd.DataFrame(pw).to_csv(RES / "stat_pairwise.csv", index=False)
    print(pd.DataFrame(fr).round(4).T.to_string())
    print(pd.DataFrame(pw).round(4).to_string())


if __name__ == "__main__":
    main()
