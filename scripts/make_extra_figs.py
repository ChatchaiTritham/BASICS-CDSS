"""Additional manuscript figures drawn from the released BASICS-CDSS result tables.

Reads results/coverage_risk.csv and results/multiseed_conformal.csv from the repository; no value is typed in
by hand. Output: figures/results/fig_riskcoverage_conformal.pdf (PeerJ text-block width).

Usage: python make_extra_figs.py [path/to/BASICS-CDSS/results]
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "figures" / "results"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#999999"

LABEL = {"logistic_regression": "LR", "random_forest": "RF", "gradient_boosting": "GB",
         "xgboost": "XGB", "lstm": "LSTM", "tcn": "TCN"}
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lstm", "tcn"]


def rows(name):
    with open(RES / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


cov = {(r["model"], r["regime"]): float(r["aurc"]) for r in rows("coverage_risk.csv")}
regimes = sorted({k[1] for k in cov})
print("regimes:", regimes)

fig, (a, b) = plt.subplots(2, 1, figsize=(5.75, 5.0), constrained_layout=True)
x = range(len(ORDER))
w = 0.38
for j, (reg, col) in enumerate(zip(regimes, (BLUE, ORANGE))):
    ys = [cov[(m, reg)] for m in ORDER]
    print("AURC", reg, dict(zip(ORDER, ys)))
    a.bar([p + (j - 0.5) * w for p in x], ys, width=w, color=col, edgecolor="black",
          linewidth=0.4, label={"static": "Clean", "temporal": "Degraded"}.get(reg, reg))
a.set_xticks(list(x), [LABEL[m] for m in ORDER])
a.set_ylabel("Area under risk\u2013coverage curve")
a.legend(frameon=False, loc="upper left", handlelength=1.1)
for j, reg in enumerate(regimes):
    for i, m in enumerate(ORDER):
        a.text(i + (j - 0.5) * w, cov[(m, reg)] + 0.004, f"{cov[(m, reg)]:.3f}",
               ha="center", va="bottom", fontsize=6.5)
a.set_ylim(0, max(cov.values()) * 1.18)
a.text(-0.02, 1.02, "(a)", transform=a.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=9)

# Panel (b): 95% split-conformal coverage over the 20 seeds (multiseed_conformal.csv).
# Sets are calibrated on the clean calibration split; coverage is measured on the
# clean and on the degraded test split, overall and among patients with the outcome, for the marginal
# and the class-conditional (Mondrian) procedure. Point = mean over seeds,
# bar = min-max over seeds.
mc = list(csv.DictReader(open(RES / "multiseed_conformal.csv", encoding="utf-8")))
series = [("marginal", "static", "empirical_coverage", "Marginal, clean: all", "o", BLUE),
          ("marginal", "temporal", "empirical_coverage", "Marginal, degraded: all", "s", ORANGE),
          ("marginal", "temporal", "coverage_deaths", "Marginal, degraded: with outcome", "v", "#D55E00"),
          ("mondrian", "temporal", "coverage_deaths", "Mondrian, degraded: with outcome", "^", "#009E73")]
target = 0.95
offs = [-0.27, -0.09, 0.09, 0.27]
lo_all = 1.0
for (meth, reg, col, lab, mk, colr), dx in zip(series, offs):
    means, los, his = [], [], []
    for m in ORDER:
        v = [float(r[col]) for r in mc if r["model"] == m and r["method"] == meth
             and r["regime"] == reg and abs(float(r["target_coverage"]) - target) < 1e-9]
        means.append(sum(v) / len(v)); los.append(min(v)); his.append(max(v))
    lo_all = min(lo_all, min(los))
    print("conformal", meth, reg, col, [round(x, 3) for x in means])
    b.errorbar([p + dx for p in x], means, yerr=[[a - c for a, c in zip(means, los)],
               [c - a for a, c in zip(means, his)]], fmt=mk, color=colr, ecolor=colr,
               elinewidth=0.8, capsize=1.8, markersize=4, markeredgecolor="black",
               markeredgewidth=0.3, label=lab)
b.axhline(target, color="#444444", ls="--", lw=0.8, zorder=1)
b.set_xticks(list(x), [LABEL[m] for m in ORDER])
b.set_ylim(max(0.0, lo_all - 0.03), 1.04)
b.set_ylabel("Coverage at 95% target")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=6.5, handlelength=1.0, columnspacing=0.8)
b.text(-0.02, 1.02, "(b)", transform=b.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=9)
n_seeds = len({r["seed"] for r in mc})
b.text(1.0, 1.02, f"mean and min–max over {n_seeds} seeds", transform=b.transAxes,
       ha="right", va="bottom", fontsize=6.8, color="#444444")

fig.savefig(OUT / "fig_riskcoverage_conformal.pdf")
print("wrote", OUT / "fig_riskcoverage_conformal.pdf")
