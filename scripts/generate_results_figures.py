"""Render headline figures from genuinely computed results.

This script plots NO hardcoded literals (the earlier hardcoded figure scripts
have been removed). It reads ``results/*.csv`` produced by ``scripts/run_all.py``
and draws every value from those tables, so each figure is traceable to a
computed metric.

Run ``python scripts/run_all.py`` first, then::

    python scripts/generate_results_figures.py

Outputs (PNG + PDF) into ``figures/results/`` and records data provenance in
``figures/results/figure_provenance.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "figures" / "results"
DPI = 300


def _require(name: str) -> Path:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/run_all.py' first."
        )
    return path


def _save(fig, stem: str) -> tuple[str, str]:
    png = OUTDIR / f"{stem}.png"
    pdf = OUTDIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))


def fig_auroc_static_vs_temporal(prov: list) -> None:
    df = pd.read_csv(_require("model_metrics.csv"))
    static = df[df["regime"] == "static"].set_index("model")
    temporal = df[df["regime"] == "temporal"].set_index("model")
    models = list(static.index)
    x = np.arange(len(models))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, static.loc[models, "auroc"], w, label="static")
    ax.bar(x + w / 2, temporal.loc[models, "auroc"], w, label="temporal (20% MCAR + noise)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Static vs temporal AUROC (computed)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    png, pdf = _save(fig, "fig_auroc_static_vs_temporal")
    prov.append(("F-AUROC", png, pdf, "results/model_metrics.csv"))


def fig_calibration(prov: list) -> None:
    df = pd.read_csv(_require("model_metrics.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, marker in (("static", "o"), ("temporal", "s")):
        sub = df[df["regime"] == regime]
        ax.plot(sub["model"], sub["ece"], marker=marker, label=f"ECE ({regime})")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Calibration error by model and regime (computed)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    png, pdf = _save(fig, "fig_calibration_ece")
    prov.append(("F-ECE", png, pdf, "results/model_metrics.csv"))


def fig_decision_curve(prov: list) -> None:
    df = pd.read_csv(_require("decision_curve.csv"))
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, df["net_benefit_at_0.30"], 0.4, label="net benefit @ 0.30")
    ax.bar(x + 0.2, df["max_net_benefit"], 0.4, label="max net benefit")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve net benefit (computed)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    png, pdf = _save(fig, "fig_decision_curve")
    prov.append(("F-DCA", png, pdf, "results/decision_curve.csv"))


def fig_conformal(prov: list) -> None:
    df = pd.read_csv(_require("conformal.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in sorted(df["target_coverage"].unique()):
        sub = df[df["target_coverage"] == target]
        ax.plot(
            sub["model"], sub["empirical_coverage"], marker="o",
            label=f"empirical (target {target:.2f})",
        )
        ax.axhline(target, ls="--", alpha=0.4)
    ax.set_ylabel("Coverage")
    ax.set_title("Conformal empirical vs target coverage (computed)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    png, pdf = _save(fig, "fig_conformal_coverage")
    prov.append(("F-CONF", png, pdf, "results/conformal.csv"))


def fig_counterfactual_delay(prov: list) -> None:
    df = pd.read_csv(_require("counterfactual_delay.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        df["antibiotic_delay_hours"],
        df["mean_terminal_harm"],
        yerr=df["std_terminal_harm"],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Antibiotic delay (hours)")
    ax.set_ylabel("Terminal harm score (committed harm function)")
    ax.set_title("Antibiotic delay vs terminal harm (computed)")
    ax.grid(True, alpha=0.3)
    png, pdf = _save(fig, "fig_counterfactual_delay")
    prov.append(("F-CF", png, pdf, "results/counterfactual_delay.csv"))


def main() -> None:
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
