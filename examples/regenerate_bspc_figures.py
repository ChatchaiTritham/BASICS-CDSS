"""Regenerate two BSPC BASICS-CDSS manuscript figures from released seed-42 results.

Reads ONLY from ../results/ CSVs (no hardcoded headline numbers) and writes
vector PDFs into the manuscript's elsarticle/figures/ directory.

Outputs:
  - figure3_sepsis_counterfactual.pdf  (mortality vs antibiotic delay, shallow gradient)
  - decision_curve.pdf                 (net benefit vs threshold, 5 models + refs)

Data anchors used:
  counterfactual_delay.csv : antibiotic_delay_hours, mortality_prob,
                             std_mortality_prob, slope_pp_per_hr, r_squared
  decision_curve.csv       : model, net_benefit_at_0.30, max_net_benefit,
                             max_net_benefit_threshold, useful_threshold_*
  cohort_summary.json      : outcome_prevalence (for treat-all reference)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
FIGDIR = Path(
    r"D:/PhD-NU/Manuscript/Manuscript/BSPC_BASICS-CDSS-Digital-Twin/elsarticle/figures"
)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "legend.frameon": False,
})


# ============================================================ FIGURE 1
def fig_counterfactual():
    df = pd.read_csv(RESULTS / "counterfactual_delay.csv")
    x = df["antibiotic_delay_hours"].to_numpy(dtype=float)
    y = df["mortality_prob"].to_numpy(dtype=float) * 100.0  # -> percent
    slope_pp = float(df["slope_pp_per_hr"].iloc[0])          # pp per hour
    r2 = float(df["r_squared"].iloc[0])

    # Least-squares line on the released points (percentage-point scale).
    coef = np.polyfit(x, y, 1)
    fit_slope, fit_intercept = coef[0], coef[1]

    # 95% CI band of the mean prediction (classic OLS interval).
    n = len(x)
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    yhat_pts = np.polyval(coef, x)
    dof = max(n - 2, 1)
    s_err = np.sqrt(np.sum((y - yhat_pts) ** 2) / dof)
    # two-sided t for small n (n=6 -> dof=4, t~2.776)
    tval = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(dof, 1.96)

    xg = np.linspace(x.min(), x.max(), 200)
    yfit = np.polyval(coef, xg)
    se_mean = s_err * np.sqrt(1.0 / n + (xg - xbar) ** 2 / sxx)
    ci = tval * se_mean

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.fill_between(xg, yfit - ci, yfit + ci, color="#c6dbef", alpha=0.7,
                    linewidth=0, label="95% CI")
    ax.plot(xg, yfit, color="#08519c", lw=1.8,
            label=f"Fitted trend ({slope_pp:.2f} pp/h, $R^2$={r2:.2f})")
    ax.plot(x, y, "o", color="#08306b", ms=5.5, mfc="white", mew=1.3,
            zorder=5, label="Simulated cohort")

    ax.set_xlabel("Antibiotic-administration delay (hours)")
    ax.set_ylabel("Simulated mortality (%)")
    ax.set_title("Counterfactual effect of antibiotic delay in the sepsis cohort",
                 fontsize=10.5, pad=8)
    ax.set_xlim(0, x.max() + 1)
    # Tight, honest y-window around the shallow band (do NOT imply 12->41%).
    lo = np.floor((y.min() - 1.5))
    hi = np.ceil((y.max() + 1.5))
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, ls=":", lw=0.5, color="#bbbbbb", alpha=0.7)
    ax.legend(loc="lower right", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "figure3_sepsis_counterfactual.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return dict(out=out, fit_slope=fit_slope, csv_slope=slope_pp, r2=r2,
                y_range=(float(y.min()), float(y.max())))


# ============================================================ FIGURE 2
def fig_decision_curve():
    # Full per-threshold series emitted by the pipeline (no interpolation).
    series = pd.read_csv(RESULTS / "decision_curve_series.csv")
    prev = json.loads((RESULTS / "cohort_summary.json").read_text())["outcome_prevalence"]

    # Reference lines on the same threshold grid as the series.
    thr = np.sort(series["threshold"].unique())
    odds = thr / (1.0 - thr)
    nb_all = prev - (1.0 - prev) * odds     # treat-all
    nb_none = np.zeros_like(thr)            # treat-none

    display = {
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "gradient_boosting": "Gradient boosting",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
        "tcn": "TCN",
    }
    colors = {
        "logistic_regression": "#1f77b4",
        "random_forest": "#2ca02c",
        "gradient_boosting": "#ff7f0e",
        "xgboost": "#9467bd",
        "lstm": "#d62728",
        "tcn": "#17becf",
    }
    # Five headline models the manuscript references: LR, RF, XGBoost, LSTM, TCN.
    order = ["logistic_regression", "random_forest", "xgboost", "lstm", "tcn"]

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    nb_at_030 = {}

    for m in order:
        sub = series[series["model"] == m].sort_values("threshold")
        mt = sub["threshold"].to_numpy(dtype=float)
        mnb = sub["net_benefit_model"].to_numpy(dtype=float)
        # Report the plotted NB nearest threshold 0.30.
        i030 = int(np.abs(mt - 0.30).argmin())
        nb_at_030[m] = float(mnb[i030])

        ax.plot(mt, mnb, lw=1.6, color=colors[m], label=display[m],
                zorder=4 if m == "lstm" else 3)

    ax.plot(thr, nb_all, "--", color="#555555", lw=1.1, label="Treat all")
    ax.plot(thr, nb_none, "-", color="#000000", lw=1.0, label="Treat none")

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis", fontsize=10.5, pad=8)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.02, 0.42)
    ax.axvline(0.30, color="#999999", lw=0.6, ls=":")
    ax.grid(True, ls=":", lw=0.5, color="#bbbbbb", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "decision_curve.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return dict(out=out, prevalence=prev, nb_at_030=nb_at_030)


if __name__ == "__main__":
    import sys
    # Fig 1 is already correct; by default regenerate ONLY the decision curve.
    # Pass "all" to also rebuild figure3_sepsis_counterfactual.pdf.
    if "all" in sys.argv:
        r1 = fig_counterfactual()
        print("== FIGURE 1 counterfactual ==")
        print(f"  csv slope_pp_per_hr = {r1['csv_slope']:.5f}")
        print(f"  refit slope (pp/h)  = {r1['fit_slope']:.5f}")
        print(f"  r_squared           = {r1['r2']:.5f}")
        print(f"  mortality range (%) = {r1['y_range'][0]:.3f} -> {r1['y_range'][1]:.3f}")
        print(f"  wrote {r1['out']}")
    r2 = fig_decision_curve()
    print("== FIGURE 2 decision curve ==")
    print(f"  prevalence (treat-all) = {r2['prevalence']}")
    for m, v in r2["nb_at_030"].items():
        print(f"  NB@0.30 {m:22s} = {v:.5f}")
    print(f"  wrote {r2['out']}")
