"""
BASICS-CDSS Visualization Demo

Demonstrates all visualization functions with mock data.
Run this script to see examples of all available plots.
"""

import sys
import io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Create output directory
output_dir = Path("examples/output")
output_dir.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("BASICS-CDSS Visualization Demo")
print("=" * 60)
print()

# =====================================================
# 1. CALIBRATION PLOTS
# =====================================================
print("1. Generating Calibration Plots...")

from basics_cdss.visualization import (
    plot_reliability_diagram,
    plot_stratified_calibration,
    plot_calibration_comparison
)

# Mock calibration data
np.random.seed(42)
n_bins = 10
bin_confidences = np.linspace(0.1, 0.9, n_bins)
bin_accuracies = bin_confidences + np.random.normal(0, 0.05, n_bins)  # Slightly miscalibrated
bin_counts = np.random.randint(50, 200, n_bins)

# Plot 1.1: Single reliability diagram
fig, ax = plot_reliability_diagram(
    bin_confidences, bin_accuracies, bin_counts,
    title="Calibration Reliability Diagram (Demo)"
)
fig.savefig(output_dir / "01_reliability_diagram.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 01_reliability_diagram.png")
plt.close()

# Plot 1.2: Stratified calibration
calibration_by_tier = {
    "high": (
        np.linspace(0.2, 0.9, 5),
        np.linspace(0.15, 0.85, 5),  # Slightly underconfident
        np.array([30, 45, 60, 40, 25])
    ),
    "medium": (
        np.linspace(0.1, 0.8, 5),
        np.linspace(0.12, 0.82, 5),  # Well calibrated
        np.array([50, 70, 80, 60, 40])
    ),
    "low": (
        np.linspace(0.1, 0.7, 5),
        np.linspace(0.2, 0.65, 5),  # Overconfident
        np.array([100, 120, 110, 90, 60])
    )
}

fig, axes = plot_stratified_calibration(
    calibration_by_tier,
    title="Calibration Stratified by Risk Tier (Demo)"
)
fig.savefig(output_dir / "02_stratified_calibration.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 02_stratified_calibration.png")
plt.close()

# Plot 1.3: Comparison across models
models_cal = {
    "Baseline": (
        np.linspace(0.1, 0.9, 8),
        np.linspace(0.05, 0.85, 8)  # Underconfident
    ),
    "TRI-X": (
        np.linspace(0.1, 0.9, 8),
        np.linspace(0.12, 0.88, 8)  # Better calibrated
    ),
    "CDSS-A": (
        np.linspace(0.1, 0.9, 8),
        np.linspace(0.15, 0.95, 8)  # Overconfident
    )
}

fig, ax = plot_calibration_comparison(
    models_cal,
    title="Calibration Comparison: Multiple Models (Demo)"
)
fig.savefig(output_dir / "03_calibration_comparison.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 03_calibration_comparison.png")
plt.close()

# =====================================================
# 2. COVERAGE-RISK PLOTS
# =====================================================
print("\n2. Generating Coverage-Risk Plots...")

from basics_cdss.visualization import (
    plot_coverage_risk_curve,
    plot_selective_prediction_comparison,
    plot_abstention_analysis
)

# Mock coverage-risk data
coverages = np.linspace(0, 1, 50)
risks = 0.3 * (1 - coverages) ** 0.5 + np.random.normal(0, 0.02, 50)
risks = np.clip(risks, 0, None)

# Plot 2.1: Coverage-risk curve
fig, ax = plot_coverage_risk_curve(
    coverages, risks,
    title="Coverage-Risk Curve (Demo)",
    highlight_points=[(0.8, "Target 80% coverage"), (0.5, "50% coverage")]
)
fig.savefig(output_dir / "04_coverage_risk_curve.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 04_coverage_risk_curve.png")
plt.close()

# Plot 2.2: Selective prediction comparison
models_sp = {
    "Baseline": (
        coverages,
        0.4 * (1 - coverages) ** 0.3 + np.random.normal(0, 0.01, 50)
    ),
    "TRI-X": (
        coverages,
        0.25 * (1 - coverages) ** 0.5 + np.random.normal(0, 0.01, 50)
    )
}

fig, ax = plot_selective_prediction_comparison(
    models_sp,
    title="Selective Prediction Comparison (Demo)"
)
fig.savefig(output_dir / "05_selective_prediction_comparison.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 05_selective_prediction_comparison.png")
plt.close()

# Plot 2.3: Abstention analysis
y_prob = np.random.beta(2, 2, 500)
y_true = (np.random.random(500) < y_prob).astype(int)

fig, axes = plot_abstention_analysis(y_prob, y_true)
fig.savefig(output_dir / "06_abstention_analysis.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06_abstention_analysis.png")
plt.close()

# =====================================================
# 3. HARM-AWARE PLOTS
# =====================================================
print("\n3. Generating Harm-Aware Plots...")

from basics_cdss.visualization import (
    plot_harm_by_tier,
    plot_escalation_analysis,
    plot_harm_concentration
)

# Mock harm data
harm_by_tier = {
    "high": 3.5,
    "medium": 1.8,
    "low": 0.6
}

# Plot 3.1: Harm by tier
fig, ax = plot_harm_by_tier(
    harm_by_tier,
    title="Harm Distribution by Risk Tier (Demo)"
)
fig.savefig(output_dir / "07_harm_by_tier.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 07_harm_by_tier.png")
plt.close()

# Plot 3.2: Escalation analysis
fig, axes = plot_escalation_analysis(
    escalation_failures=12,
    false_escalations=25,
    high_risk_samples=80,
    low_risk_samples=200
)
fig.savefig(output_dir / "08_escalation_analysis.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 08_escalation_analysis.png")
plt.close()

# Plot 3.3: Harm concentration
fig, ax = plot_harm_concentration(
    harm_by_tier,
    concentration_index=0.75
)
fig.savefig(output_dir / "09_harm_concentration.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 09_harm_concentration.png")
plt.close()

# =====================================================
# 4. COMPARISON PLOTS
# =====================================================
print("\n4. Generating Comparison Plots...")

from basics_cdss.visualization import (
    plot_metric_comparison,
    plot_model_comparison_radar,
    create_evaluation_dashboard
)

# Mock model comparison data
models_metrics = {
    "Baseline": {
        "ECE": 0.15,
        "AURC": 0.28,
        "Weighted Harm": 2.8,
        "Accuracy": 0.82
    },
    "TRI-X": {
        "ECE": 0.08,
        "AURC": 0.18,
        "Weighted Harm": 1.5,
        "Accuracy": 0.88
    },
    "CDSS-A": {
        "ECE": 0.12,
        "AURC": 0.22,
        "Weighted Harm": 2.1,
        "Accuracy": 0.85
    }
}

# Plot 4.1: Metric comparison
fig, ax = plot_metric_comparison(
    models_metrics,
    title="Multi-Model Performance Comparison (Demo)"
)
fig.savefig(output_dir / "10_metric_comparison.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 10_metric_comparison.png")
plt.close()

# Plot 4.2: Radar chart
fig, ax = plot_model_comparison_radar(
    models_metrics,
    metrics=["ECE", "AURC", "Weighted Harm", "Accuracy"],
    title="Model Comparison Radar Chart (Demo)"
)
fig.savefig(output_dir / "11_radar_comparison.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 11_radar_comparison.png")
plt.close()

# Plot 4.3: Evaluation dashboard
dashboard_data_cal = {
    'confidences': bin_confidences,
    'accuracies': bin_accuracies,
    'counts': bin_counts
}

dashboard_data_cr = {
    'coverages': coverages,
    'risks': risks
}

dashboard_data_harm = {
    'harm_by_tier': harm_by_tier,
    'concentration_index': 0.75
}

fig = create_evaluation_dashboard(
    dashboard_data_cal,
    dashboard_data_cr,
    dashboard_data_harm,
    model_name="TRI-X Demo"
)
fig.savefig(output_dir / "12_evaluation_dashboard.png", dpi=300, bbox_inches='tight')
print("   ✓ Saved: 12_evaluation_dashboard.png")
plt.close()

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 60)
print("✅ All visualizations generated successfully!")
print("=" * 60)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
for i, filename in enumerate([
    "01_reliability_diagram.png",
    "02_stratified_calibration.png",
    "03_calibration_comparison.png",
    "04_coverage_risk_curve.png",
    "05_selective_prediction_comparison.png",
    "06_abstention_analysis.png",
    "07_harm_by_tier.png",
    "08_escalation_analysis.png",
    "09_harm_concentration.png",
    "10_metric_comparison.png",
    "11_radar_comparison.png",
    "12_evaluation_dashboard.png"
], 1):
    print(f"  {i:2d}. {filename}")

print("\n" + "=" * 60)
print("Run this script anytime to regenerate visualization examples")
print("=" * 60)
