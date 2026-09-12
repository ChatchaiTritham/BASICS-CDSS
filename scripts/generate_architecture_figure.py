"""Render the framework architecture diagram at the PeerJ text-block width.

The earlier version of this diagram was authored for a two-column Elsevier page
(13 x 8.8 in) and then shrunk to fit, which dropped its smallest type to about
3.5 pt on the printed page. This version is authored at final size -- 6.02 in,
the PeerJ text block measured from wlpeerj.cls -- so nothing is rescaled and every label
renders at the size it is set in. Portrait orientation buys the vertical room
that the five-layer pipeline needs once the type is large enough to read.

House rules applied here (PeerJ artwork policy + general journal practice):
  * no title, subtitle or panel number inside the artwork; the caption carries them
  * white background, not off-white
  * 8 pt minimum type, two sizes and one family
  * typeset maths as real glyphs, never as source (no "x(t) in R^n", no "p_{miss}")
  * Okabe-Ito hues, which stay distinguishable in the common colour-vision deficits

    python scripts/generate_architecture_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"

# Okabe-Ito
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
RED = "#CC79A7"
SKY = "#56B4E9"
GREY = "#4D4D4D"
INK = "#1A1A1A"

TITLE_PT = 8.5
BODY_PT = 8.0

XMAX, YMAX = 100.0, 124.0        # layout units, kept square
W_IN = 5.766                      # 415.13 pt, the PeerJ text block
H_IN = W_IN * YMAX / XMAX

HEAD_H = 5.0                      # header bar height, layout units
LINE_H = 3.4                      # body line pitch


def body_h(nlines: int) -> float:
    return HEAD_H + 3.0 + nlines * LINE_H


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": BODY_PT,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def box(ax, x, y, w, h, header, lines, colour, tint):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=tint, edgecolor=colour, linewidth=0.9, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y + h - HEAD_H), w, HEAD_H,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=colour, edgecolor=colour, linewidth=0.9, zorder=4))
    ax.text(x + w / 2, y + h - HEAD_H / 2, header, ha="center", va="center",
            fontsize=TITLE_PT, fontweight="bold", color="white", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + 2.2, y + h - HEAD_H - 2.4 - i * LINE_H, ln, ha="left", va="top",
                fontsize=BODY_PT, color=INK, zorder=5)


def arrow(ax, x0, y0, x1, y1, colour):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.0, color=colour, zorder=2, shrinkA=0, shrinkB=0))


def note(ax, x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=BODY_PT, style="italic",
            color=GREY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"))


def band(ax, y, h, text):
    ax.text(-2.4, y + h / 2, text, ha="center", va="center", rotation=90,
            fontsize=BODY_PT, color=GREY, zorder=5)


def main() -> None:
    style()
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.set_xlim(-6, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Layer 1: the three disease models --------------------------------
    h1 = body_h(3)
    y1 = YMAX - h1 - 1.5
    w, gap = 31.0, 3.5
    box(ax, 0, y1, w, h1, "Sepsis",
        ["8 state variables", "ODE: pathogen, immune,", "organ damage"], BLUE, "#DCE9F5")
    box(ax, w + gap, y1, w, h1, "ARDS",
        ["6 state variables", "SDE: alveolar flooding", "and compliance"], GREEN, "#D9EFE8")
    box(ax, 2 * (w + gap), y1, w, h1, "ACS",
        ["8 state variables", "ODE: troponin, two", "compartments"], ORANGE, "#F7E2D5")
    band(ax, y1, h1, "Disease models")

    # ---- Layer 2: the twin core -------------------------------------------
    h2 = body_h(4)
    y2 = y1 - 8.0 - h2
    for dx, col in ((-31, BLUE), (0, GREEN), (31, ORANGE)):
        arrow(ax, XMAX / 2 + dx, y1, XMAX / 2 + dx, y2 + h2, col)
    note(ax, XMAX / 2, y1 - 4.0, "parameters θ and initial conditions")
    box(ax, 0, y2, XMAX, h2, "Patient digital twin core",
        ["State vector x(t), n dimensions  ·  horizon 24 h  ·  Δt = 1 h, 25 observations per twin",
         "Integration: explicit forward Euler for all three disease models",
         "Reproducibility: deterministic seeding (seed 42) across N = 1,000 twins",
         "Intervention response: a physiologically grounded Δx applied after integration"],
        BLUE, "#DCE9F5")
    band(ax, y2, h2, "Twin core")

    # ---- Layer 3: the measurement process ---------------------------------
    h3 = body_h(3)
    y3 = y2 - 8.0 - h3
    arrow(ax, XMAX / 2, y2, XMAX / 2, y3 + h3, GREEN)
    note(ax, XMAX / 2, y2 - 4.0, "true state x(t)")
    box(ax, 0, y3, XMAX, h3, "Measurement process simulator",
        ["Observation model: oᵢ(t) = sᵢ(t) + εᵢ(t) when measured, otherwise missing",
         "Noise: εᵢ ~ N(0, σᵢ²), drawn independently at each observation time",
         "Observation times: every integration step, hourly across the 24 h horizon"],
        GREEN, "#D9EFE8")
    band(ax, y3, h3, "Measurement")

    # ---- Layer 4: the perturbation operators ------------------------------
    h4 = body_h(3)
    y4 = y3 - 8.0 - h4
    pw = 22.5
    pgap = (XMAX - 4 * pw) / 3
    specs = [("Missing data", ["MCAR, 20%,", "used in results;", "MAR/MNAR ship only"], SKY, "#DFF0FB"),
             ("Noise", ["Gaussian, σ × 2,", "used in results;", "Laplacian, t(3) ship"], GREEN, "#D9EFE8"),
             ("Temporal mask", ["Gaps of 1–4 h;", "ships but is not", "used in any result"], ORANGE, "#F7E2D5"),
             ("Conflict", ["Discordant biomarker", "trends; ships but is", "not used in any result"], RED, "#F6E3EE")]
    for i, (hdr, lines, col, tint) in enumerate(specs):
        x = i * (pw + pgap)
        arrow(ax, XMAX / 2, y3, x + pw / 2, y4 + h4, ORANGE)
        box(ax, x, y4, pw, h4, hdr, lines, col, tint)
    note(ax, XMAX / 2, y3 - 4.0, "observations o(t)")
    band(ax, y4, h4, "Perturbation")

    # ---- Layer 5: the model families under evaluation ---------------------
    h5 = 13.0
    y5 = y4 - 8.0 - h5
    ax.add_patch(FancyBboxPatch((0, y5), XMAX, h5, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="#EDF2F7", edgecolor=BLUE, linewidth=0.9, zorder=3))
    models = ["Logistic", "Random", "Gradient", "XGBoost", "LSTM", "TCN"]
    subs = ["regression", "forest", "boosting", "", "", ""]
    mw = (XMAX - 4.0) / 6
    for i, m in enumerate(models):
        x = 2.0 + i * mw
        col = "#FBE3C4" if i >= 4 else "#CFE3F3"
        ax.add_patch(FancyBboxPatch((x + 0.7, y5 + 4.6), mw - 1.4, 7.2,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=col, edgecolor=BLUE, linewidth=0.7, zorder=4))
        dy = 1.5 if subs[i] else 0.0
        ax.text(x + mw / 2, y5 + 8.1 + dy, m, ha="center", va="center",
                fontsize=BODY_PT, color=INK, zorder=5)
        if subs[i]:
            ax.text(x + mw / 2, y5 + 8.1 - 2.3, subs[i], ha="center", va="center",
                    fontsize=BODY_PT, color=INK, zorder=5)
    ax.text(XMAX / 2, y5 + 2.2,
            "Metrics: AUROC · accuracy · TCS · TCE · DIR · counterfactual regret · DBRS · BEWS",
            ha="center", va="center", fontsize=BODY_PT, color=GREY, zorder=5)
    for i in range(4):
        arrow(ax, i * (pw + pgap) + pw / 2, y4, XMAX / 2, y5 + h5, GREY)
    note(ax, XMAX / 2, y4 - 4.0, "perturbed observations")
    band(ax, y5, h5, "Evaluation")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"figure1_architecture.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote %s  (%.2f x %.2f in)" % (OUTDIR / "figure1_architecture.pdf", W_IN, H_IN))


if __name__ == "__main__":
    main()
