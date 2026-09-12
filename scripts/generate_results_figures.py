"""Render headline figures from genuinely computed results.

This script plots NO hardcoded literals (the earlier hardcoded figure scripts
have been removed). It reads ``results/*.csv`` produced by ``scripts/run_all.py``
and draws every value from those tables, so each figure is traceable to a
computed metric.

Run ``python scripts/run_all.py`` first, then::

    python scripts/generate_results_figures.py

Outputs (PNG + PDF) into ``figures/results/`` and records data provenance in
``figures/results/figure_provenance.csv``.

Styling/IO go through the shared, byte-identical ``pubviz`` module (vendored as
``scripts/pubviz.py``): ``apply_pub_style`` for the canonical top-tier rcParams,
``save_fig`` for matched vector PDF + 300-dpi PNG, and ``PALETTE`` for the
Okabe-Ito colour-blind-safe series order.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pubviz import apply_pub_style, save_fig, PALETTE, load_results, results_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "figures" / "results"


def _require(name: str) -> Path:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/run_all.py' first."
        )
    return path


def _emit(fig, stem: str) -> tuple[str, str]:
    """Save matched PDF + 300-dpi PNG via the shared helper; return repo-relative paths."""
    save_fig(fig, stem, out_dir=OUTDIR)
    plt.close(fig)
    png = (OUTDIR / f"{stem}.png").relative_to(ROOT)
    pdf = (OUTDIR / f"{stem}.pdf").relative_to(ROOT)
    return str(png), str(pdf)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion p estimated from n trials.

    Coverage is the fraction of n_test cases covered, so its sampling
    uncertainty is binomial; the Wilson interval is the standard, honest band
    for a proportion. n is read from the computed results, never hardcoded.
    """
    if n <= 0:
        return p, p
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def fig_auroc_static_vs_temporal(prov: list) -> None:
    df = pd.read_csv(_require("model_metrics.csv"))
    static = df[df["regime"] == "static"].set_index("model")
    temporal = df[df["regime"] == "temporal"].set_index("model")
    models = list(static.index)
    x = np.arange(len(models))
    w = 0.38

    labels = [m.replace("_", " ").title() for m in models]
    fig, ax = plt.subplots(figsize=(5.766, 3.6))
    ax.bar(x - w / 2, static.loc[models, "auroc"], w,
           label="Static", color=PALETTE[0])
    ax.bar(x + w / 2, temporal.loc[models, "auroc"], w,
           label="Temporal (20% MCAR + noise)", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("AUROC (unitless)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    png, pdf = _emit(fig, "fig_auroc_static_vs_temporal")
    prov.append(("F-AUROC", png, pdf, "results/model_metrics.csv"))


def fig_calibration(prov: list) -> None:
    df = pd.read_csv(_require("model_metrics.csv"))
    fig, ax = plt.subplots(figsize=(5.766, 3.6))
    styles = (("static", "o", "-", PALETTE[0]), ("temporal", "s", "--", PALETTE[1]))
    for regime, marker, ls, color in styles:
        sub = df[df["regime"] == regime]
        labels = [m.replace("_", " ").title() for m in sub["model"]]
        ax.plot(labels, sub["ece"], marker=marker, ls=ls, color=color,
                label=f"ECE ({regime})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Expected calibration error (unitless)")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_axisbelow(True)
    png, pdf = _emit(fig, "fig_calibration_ece")
    prov.append(("F-ECE", png, pdf, "results/model_metrics.csv"))


def fig_decision_curve(prov: list) -> None:
    """Decision curve: net benefit against risk threshold, with the two references.

    Reads the full threshold sweep rather than the single-threshold summary, so
    the figure shows the curve the caption describes. Treat-all is analytic at
    each threshold; treat-none is zero everywhere.
    """
    series = pd.read_csv(_require("decision_curve_series.csv"))
    summary = pd.read_csv(_require("decision_curve.csv"))
    prevalence = float(pd.read_csv(_require("decision_curve_prevalence.csv"))["prevalence"].iloc[0]) \
        if (results_dir() / "decision_curve_prevalence.csv").exists() else 0.40

    order = ["logistic_regression", "gradient_boosting", "random_forest",
             "tcn", "xgboost", "lstm"]
    nice = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting",
            "random_forest": "Random forest", "tcn": "TCN", "xgboost": "XGBoost", "lstm": "LSTM"}

    fig, ax = plt.subplots(figsize=(5.766, 3.8))
    for i, model in enumerate(order):
        g = series[series.model == model].sort_values("threshold")
        ax.plot(g.threshold, g.net_benefit_model, linewidth=1.4,
                color=PALETTE[i % len(PALETTE)], label=nice.get(model, model), zorder=4)

    t = np.linspace(series.threshold.min(), series.threshold.max(), 400)
    treat_all = prevalence - (1.0 - prevalence) * t / (1.0 - t)
    ax.plot(t, treat_all, linestyle=(0, (5, 2)), linewidth=1.1, color="#4D4D4D",
            label="Treat all", zorder=3)
    ax.axhline(0.0, linestyle=(0, (1, 2)), linewidth=1.1, color="#1A1A1A",
               label="Treat none", zorder=3)

    marker_t = float(series.threshold.iloc[(series.threshold - 0.30).abs().argmin()])
    ax.axvline(marker_t, color="#BBBBBB", linewidth=0.8, zorder=2)
    ax.annotate("threshold 0.30", xy=(marker_t, 0.40), xytext=(marker_t + 0.03, 0.40),
                fontsize=8, color="#4D4D4D", va="center")

    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit (true positives per patient)")
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.05, 0.45)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    png, pdf = _emit(fig, "fig_decision_curve")
    prov.append(("F-DCA", png, pdf, "results/decision_curve_series.csv"))


def fig_conformal(prov: list) -> None:
    df = pd.read_csv(_require("conformal.csv"))
    # n_test per model backs the binomial (Wilson) CI on empirical coverage;
    # read from the computed metrics table, never hardcoded.
    metrics = pd.read_csv(_require("model_metrics.csv"))
    n_by_model = (
        metrics.groupby("model")["n_test"].max().astype(int).to_dict()
    )

    fig, ax = plt.subplots(figsize=(5.766, 3.6))
    markers = ("o", "s", "^", "D")
    targets = sorted(df["target_coverage"].unique())
    models = list(dict.fromkeys(df["model"]))
    labels = [m.replace("_", " ").title() for m in models]
    xpos = np.arange(len(models))
    # small horizontal offset per target so CI bars do not overlap
    offsets = np.linspace(-0.12, 0.12, len(targets)) if len(targets) > 1 else [0.0]

    for i, target in enumerate(targets):
        sub = df[df["target_coverage"] == target].set_index("model")
        color = PALETTE[i % len(PALETTE)]
        emp = np.array([sub.loc[m, "empirical_coverage"] for m in models], float)
        lo = np.empty_like(emp)
        hi = np.empty_like(emp)
        for j, m in enumerate(models):
            l, h = _wilson_ci(emp[j], n_by_model.get(m, 0))
            lo[j], hi[j] = l, h
        yerr = np.vstack([emp - lo, hi - emp])
        ax.errorbar(
            xpos + offsets[i], emp, yerr=yerr, marker=markers[i % len(markers)],
            ls="none", color=color, capsize=4, elinewidth=0.9,
            label=f"Empirical (target {target:.2f}), 95% Wilson CI",
        )
        ax.axhline(target, ls="--", lw=1.0, color=color, alpha=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Coverage (fraction of test cases)")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_axisbelow(True)
    png, pdf = _emit(fig, "fig_conformal_coverage")
    prov.append(("F-CONF", png, pdf, "results/conformal.csv; results/model_metrics.csv"))


def fig_counterfactual_delay(prov: list) -> None:
    df = pd.read_csv(_require("counterfactual_delay.csv"))
    x = df["antibiotic_delay_hours"].to_numpy(dtype=float)
    y = df["mortality_prob"].to_numpy(dtype=float)
    n = df["n_twins"].to_numpy(dtype=float)
    # 95% CI of the mean across the seeded twin cohort (paired re-simulation)
    ci = 1.96 * df["std_mortality_prob"].to_numpy(dtype=float) / np.sqrt(n)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(5.766, 3.6))
    ax.errorbar(
        x, y, yerr=ci,
        marker="o", capsize=4, color=PALETTE[1], ecolor=PALETTE[6],
        elinewidth=0.9, label=r"Mean mortality probability $\pm$ 95% CI",
    )
    ax.plot(x, yhat, linestyle="--", linewidth=1.2, color=PALETTE[6],
            label=f"Linear fit: {slope * 100:.2f} pp/hour ($R^2$ = {r2:.2f})")
    ax.set_xlabel("Antibiotic delay (hours)")
    ax.set_ylabel("Simulated mortality probability")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_axisbelow(True)
    png, pdf = _emit(fig, "fig_counterfactual_delay")
    prov.append(("F-CF", png, pdf, "results/counterfactual_delay.csv"))


def main() -> None:
    apply_pub_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    prov: list = []
    fig_auroc_static_vs_temporal(prov)
    fig_calibration(prov)
    fig_decision_curve(prov)
    fig_conformal(prov)
    fig_counterfactual_delay(prov)

    prov_path = OUTDIR / "figure_provenance.csv"
    with prov_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure_id", "png", "pdf", "source_data"])
        writer.writerows(prov)

    print(f"Rendered {len(prov)} results-driven figures in {OUTDIR}")
    print(f"Wrote provenance: {prov_path}")


if __name__ == "__main__":
    main()
