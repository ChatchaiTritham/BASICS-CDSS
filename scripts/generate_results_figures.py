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
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "figures" / "results"
DPI = 300

# Color-blind-safe (Okabe-Ito) — canonical shared palette, use in this order.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def apply_pub_style() -> None:
    """Apply the canonical top-tier publication style (shared across all repos)."""
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


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

    labels = [m.replace("_", " ").title() for m in models]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, static.loc[models, "auroc"], w,
           label="Static", color=PALETTE[0])
    ax.bar(x + w / 2, temporal.loc[models, "auroc"], w,
           label="Temporal (20% MCAR + noise)", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("AUROC (unitless)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Static vs temporal discrimination")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    png, pdf = _save(fig, "fig_auroc_static_vs_temporal")
    prov.append(("F-AUROC", png, pdf, "results/model_metrics.csv"))


def fig_calibration(prov: list) -> None:
    df = pd.read_csv(_require("model_metrics.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    styles = (("static", "o", "-", PALETTE[0]), ("temporal", "s", "--", PALETTE[1]))
    for regime, marker, ls, color in styles:
        sub = df[df["regime"] == regime]
        labels = [m.replace("_", " ").title() for m in sub["model"]]
        ax.plot(labels, sub["ece"], marker=marker, ls=ls, color=color,
                label=f"ECE ({regime})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Expected calibration error (unitless)")
    ax.set_ylim(bottom=0)
    ax.set_title("Calibration error by model and regime")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_axisbelow(True)
    png, pdf = _save(fig, "fig_calibration_ece")
    prov.append(("F-ECE", png, pdf, "results/model_metrics.csv"))


def fig_decision_curve(prov: list) -> None:
    df = pd.read_csv(_require("decision_curve.csv"))
    x = np.arange(len(df))
    labels = [m.replace("_", " ").title() for m in df["model"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - 0.2, df["net_benefit_at_0.30"], 0.4,
           label="Net benefit @ threshold 0.30", color=PALETTE[0])
    ax.bar(x + 0.2, df["max_net_benefit"], 0.4,
           label="Maximum net benefit", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Net benefit (true positives per patient)")
    ax.set_title("Decision-curve net benefit")
    ax.legend(loc="best")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    png, pdf = _save(fig, "fig_decision_curve")
    prov.append(("F-DCA", png, pdf, "results/decision_curve.csv"))


def fig_conformal(prov: list) -> None:
    df = pd.read_csv(_require("conformal.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    markers = ("o", "s", "^", "D")
    for i, target in enumerate(sorted(df["target_coverage"].unique())):
        sub = df[df["target_coverage"] == target]
        color = PALETTE[i % len(PALETTE)]
        labels = [m.replace("_", " ").title() for m in sub["model"]]
        ax.plot(
            labels, sub["empirical_coverage"], marker=markers[i % len(markers)],
            color=color, label=f"Empirical (target {target:.2f})",
        )
        ax.axhline(target, ls="--", lw=1.0, color=color, alpha=0.6)
    ax.set_xlabel("Model")
    ax.set_ylabel("Coverage (fraction of test cases)")
    ax.set_title("Conformal empirical vs target coverage")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_axisbelow(True)
    png, pdf = _save(fig, "fig_conformal_coverage")
    prov.append(("F-CONF", png, pdf, "results/conformal.csv"))


def fig_counterfactual_delay(prov: list) -> None:
    df = pd.read_csv(_require("counterfactual_delay.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(
        df["antibiotic_delay_hours"],
        df["mean_terminal_harm"],
        yerr=df["std_terminal_harm"],
        marker="o",
        capsize=4,
        color=PALETTE[1],
        ecolor=PALETTE[6],
        elinewidth=0.9,
        label="Mean $\\pm$ 1 SD",
    )
    ax.set_xlabel("Antibiotic delay (hours)")
    ax.set_ylabel("Terminal harm score (committed-harm units)")
    ax.set_title("Antibiotic delay vs terminal harm")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_axisbelow(True)
    png, pdf = _save(fig, "fig_counterfactual_delay")
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
