"""Additional manuscript figures drawn from the released BASICS-CDSS result tables.

Reads results/coverage_risk.csv and results/conformal.csv from the repository; no value is typed in
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
conf = {r["model"]: r for r in rows("conformal.csv")}

fig, (a, b) = plt.subplots(2, 1, figsize=(5.75, 4.6), constrained_layout=True)
x = range(len(ORDER))
w = 0.38
for j, (reg, col) in enumerate(zip(regimes, (BLUE, ORANGE))):
    ys = [cov[(m, reg)] for m in ORDER]
    print("AURC", reg, dict(zip(ORDER, ys)))
    a.bar([p + (j - 0.5) * w for p in x], ys, width=w, color=col, edgecolor="black",
          linewidth=0.4, label=reg.capitalize())
a.set_xticks(list(x), [LABEL[m] for m in ORDER])
a.set_ylabel("Area under risk\u2013coverage curve")
a.legend(frameon=False, loc="upper left", handlelength=1.1)
for j, reg in enumerate(regimes):
    for i, m in enumerate(ORDER):
        a.text(i + (j - 0.5) * w, cov[(m, reg)] + 0.004, f"{cov[(m, reg)]:.3f}",
               ha="center", va="bottom", fontsize=6.5)
a.set_ylim(0, max(cov.values()) * 1.18)
a.text(-0.02, 1.02, "(a)", transform=a.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=9)

emp = [float(conf[m]["empirical_coverage"]) for m in ORDER]
size = [float(conf[m]["avg_set_size"]) for m in ORDER]
target = float(conf[ORDER[0]]["target_coverage"])
print("conformal", dict(zip(ORDER, zip(emp, size))), "target", target)

# Coverage is a proportion over the held-out fold, so draw its Wilson 95% interval
# (n_test is read from the released metrics table, never typed in).
n_test = int(float(next(r for r in rows("model_metrics.csv") if r.get("n_test"))["n_test"]))


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


err = [[e - wilson(e, n_test)[0] for e in emp], [wilson(e, n_test)[1] - e for e in emp]]
b.axhspan(target, 1.045, color="#e8f3ec", zorder=0, label="at or above nominal target")
b.axhline(target, color="#444444", ls="--", lw=0.8, zorder=1, label=f"nominal target {target:.2f}")
b.bar(list(x), emp, width=0.6, color=BLUE, edgecolor="black", linewidth=0.4, zorder=2,
      yerr=err, error_kw={"elinewidth": 0.7, "capsize": 2.5, "ecolor": "#222222"})
for p, (e, sz) in enumerate(zip(emp, size)):
    b.text(p, 0.9025, f"$|C|$={sz:.2f}", ha="center", va="bottom", fontsize=6.8, color="white")
    b.text(p, e - 0.0035, f"{e:.3f}", ha="center", va="top", fontsize=6.8, color="white")
b.legend(loc="upper center", ncol=2, frameon=False, fontsize=6.8, handlelength=1.4)
b.set_xticks(list(x), [LABEL[m] for m in ORDER])
b.set_ylim(0.90, 1.045)
b.set_ylabel("Empirical coverage")
b.text(-0.02, 1.02, "(b)", transform=b.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=9)
b.text(1.0, 1.02, f"bars: Wilson 95% interval, $n$={n_test} held-out twins", transform=b.transAxes,
       ha="right", va="bottom", fontsize=6.8, color="#444444")

fig.savefig(OUT / "fig_riskcoverage_conformal.pdf")
print("wrote", OUT / "fig_riskcoverage_conformal.pdf")
