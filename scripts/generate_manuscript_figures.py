"""Generate the curated BASICS-CDSS manuscript figure set.

The repository contains many baseline, tiered, demo, and legacy figure
artifacts. This script promotes a compact, reproducible manuscript subset into
``figures/manuscript/`` without deleting the broader archive.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "manuscript"
DEFAULT_MANIFEST = ROOT / "FIGURE_MANIFEST.csv"
DPI = 300

FIGURES = [
    {
        "figure_id": "BASICS-F1",
        "source": "examples/figures/baseline/fig01_reliability_diagram.png",
        "stem": "fig1_reliability_diagram",
        "caption": "Reliability diagram for calibration-aware CDSS evaluation.",
        "article_section": "Calibration and reliability",
    },
    {
        "figure_id": "BASICS-F2",
        "source": "examples/figures/baseline/fig04_coverage_risk.png",
        "stem": "fig2_coverage_risk",
        "caption": "Coverage-risk trade-off for selective prediction under safety constraints.",
        "article_section": "Selective prediction",
    },
    {
        "figure_id": "BASICS-F3",
        "source": "examples/figures/baseline/fig06_abstention_analysis.png",
        "stem": "fig3_abstention_analysis",
        "caption": "Abstention analysis showing when the evaluator defers unsafe predictions.",
        "article_section": "Abstention and uncertainty",
    },
    {
        "figure_id": "BASICS-F4",
        "source": "examples/figures/baseline/fig07_harm_by_tier.png",
        "stem": "fig4_harm_by_tier",
        "caption": "Harm distribution by safety tier for beyond-accuracy CDSS assessment.",
        "article_section": "Harm-aware evaluation",
    },
    {
        "figure_id": "BASICS-F5",
        "source": "examples/figures/baseline/fig10_metric_comparison.png",
        "stem": "fig5_metric_comparison",
        "caption": "Metric comparison across conventional and safety-aware evaluation criteria.",
        "article_section": "Metric comparison",
    },
    {
        "figure_id": "BASICS-F6",
        "source": "examples/figures/baseline/fig12_evaluation_dashboard.png",
        "stem": "fig6_evaluation_dashboard",
        "caption": "Compact evaluation dashboard retained as a supplementary overview panel.",
        "article_section": "Supplementary overview",
        "role": "supplementary",
    },
]


def copy_png_and_make_pdf(source: Path, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    shutil.copy2(source, png_path)

    with Image.open(source) as image:
        width, height = image.size
        fig_width = width / DPI
        fig_height = height / DPI
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
        ax.imshow(image.convert("RGB"))
        ax.axis("off")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    return png_path, pdf_path


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    fieldnames = [
        "figure_id",
        "role",
        "png",
        "pdf",
        "source_script",
        "source_data",
        "caption",
        "article_section",
        "generated_at",
        "dpi",
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "generated_at": generated_at, "dpi": str(DPI)})


def make_contact_sheet(output_dir: Path) -> Path:
    pngs = sorted(p for p in output_dir.glob("*.png") if not p.name.startswith("visual_qa"))
    thumbs = []
    for path in pngs:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            original = thumb.size
            thumb.thumbnail((460, 320), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (500, 390), "white")
            canvas.paste(thumb, ((500 - thumb.width) // 2, 42))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), path.name, fill="black")
            draw.text((8, 365), f"{original[0]}x{original[1]}", fill="black")
            thumbs.append(canvas)

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 500, rows * 390), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 500, (index // cols) * 390))

    sheet_path = output_dir / "visual_qa_contact_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curated BASICS-CDSS manuscript figures")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in FIGURES:
        source = ROOT / item["source"]
        if not source.exists():
            raise FileNotFoundError(source)
        png_path, pdf_path = copy_png_and_make_pdf(source, args.output_dir, item["stem"])
        rows.append(
            {
                "figure_id": item["figure_id"],
                "role": item.get("role", "manuscript"),
                "png": str(png_path.relative_to(ROOT)),
                "pdf": str(pdf_path.relative_to(ROOT)),
                "source_script": "scripts/generate_manuscript_figures.py",
                "source_data": item["source"],
                "caption": item["caption"],
                "article_section": item["article_section"],
            }
        )

    write_manifest(rows, args.manifest)
    sheet_path = make_contact_sheet(args.output_dir)
    print(f"Generated {len(rows)} curated figures in {args.output_dir}")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote visual QA contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
