"""
Performance Figures Generation Script

Master script to generate all performance visualization figures for BASICS-CDSS.

Generates publication-ready figures:
- Confusion matrices (binary & multi-class)
- ROC curves
- Precision-Recall curves
- Threshold analysis
- Multi-model comparisons
- 3D performance landscapes
- Stratified heatmaps

Usage:
    python generate_performance_figures.py --output-dir figures/performance
    python generate_performance_figures.py --demo-only  # Generate with synthetic data
    python generate_performance_figures.py --all        # Generate all figure types

Author: Chatchai Tritham
Affiliation: Naresuan University
Date: 2026-01-25
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from basics_cdss.metrics.performance import (
    confusion_matrix,
    compute_performance_metrics,
    compute_roc_curve,
    compute_pr_curve,
    sensitivity_specificity_analysis,
    multi_class_metrics,
)

from basics_cdss.visualization.performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_sensitivity_specificity_curve,
    plot_threshold_analysis,
    plot_multi_model_roc,
    plot_metrics_comparison_bar,
    plot_multi_class_confusion_matrix,
)

from basics_cdss.visualization.advanced_charts import (
    plot_3d_performance_surface,
    plot_contour_performance,
    plot_stratified_heatmap,
    plot_radar_chart,
    plot_multi_radar_comparison,
)


def generate_synthetic_data(n_samples: int = 200, seed: int = 42) -> dict:
    """Generate synthetic classification data for demonstration.

    Parameters:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with synthetic data
    """
    np.random.seed(seed)

    # Generate true labels (60% positive class)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # Generate predicted probabilities (with some noise)
    y_prob = np.zeros(n_samples)
    for i in range(n_samples):
        if y_true[i] == 1:
            y_prob[i] = np.random.beta(8, 2)  # Higher prob for positive class
        else:
            y_prob[i] = np.random.beta(2, 8)  # Lower prob for negative class

    # Generate predictions at threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # Generate risk tier stratification
    risk_tiers = np.random.choice(
        ['R1 (Low)', 'R2 (Medium)', 'R3 (High)', 'R4 (Critical)'],
        size=n_samples
    )

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'risk_tiers': risk_tiers,
    }


def generate_binary_classification_figures(data: dict, output_dir: Path):
    """Generate figures for binary classification performance.

    Parameters:
        data: Dictionary with y_true, y_pred, y_prob
        output_dir: Output directory for figures
    """
    print("[*] Generating binary classification figures...")

    y_true = data['y_true']
    y_pred = data['y_pred']
    y_prob = data['y_prob']

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    print("  - Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plot_confusion_matrix(
        cm.to_array(),
        title="CDSS Performance: Confusion Matrix",
        save_path=output_dir / "fig01_confusion_matrix.pdf",
        dpi=300
    )
    plt.close(fig)

    # 2. Normalized Confusion Matrix
    print("  - Generating normalized confusion matrix...")
    fig, ax = plot_confusion_matrix(
        cm.to_array(),
        normalize=True,
        title="CDSS Performance: Normalized Confusion Matrix",
        save_path=output_dir / "fig02_confusion_matrix_normalized.pdf",
        dpi=300
    )
    plt.close(fig)

    # 3. ROC Curve
    print("  - Generating ROC curve...")
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plot_roc_curve(
        fpr, tpr, roc_auc,
        title="CDSS Performance: ROC Curve",
        save_path=output_dir / "fig03_roc_curve.pdf",
        dpi=300
    )
    plt.close(fig)

    # 4. Precision-Recall Curve
    print("  - Generating PR curve...")
    precision, recall, _ = compute_pr_curve(y_true, y_prob)
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)

    fig, ax = plot_pr_curve(
        precision, recall, pr_auc,
        baseline_prevalence=prevalence,
        title="CDSS Performance: Precision-Recall Curve",
        save_path=output_dir / "fig04_pr_curve.pdf",
        dpi=300
    )
    plt.close(fig)

    # 5. Threshold Analysis
    print("  - Generating threshold analysis...")
    df_threshold = sensitivity_specificity_analysis(y_true, y_prob)

    fig, axes = plot_threshold_analysis(
        df_threshold,
        title="CDSS Performance: Threshold Analysis",
        save_path=output_dir / "fig05_threshold_analysis.pdf",
        dpi=300
    )
    plt.close(fig)

    # 6. Sensitivity-Specificity Curve
    print("  - Generating sensitivity-specificity curve...")
    optimal_idx = df_threshold['youdens_j'].idxmax()
    optimal_threshold = df_threshold.loc[optimal_idx, 'threshold']

    fig, ax = plot_sensitivity_specificity_curve(
        df_threshold['threshold'].values,
        df_threshold['sensitivity'].values,
        df_threshold['specificity'].values,
        optimal_threshold=optimal_threshold,
        title="CDSS Performance: Sensitivity-Specificity Tradeoff",
        save_path=output_dir / "fig06_sensitivity_specificity.pdf",
        dpi=300
    )
    plt.close(fig)

    print("[OK] Binary classification figures generated successfully!")


def generate_comparison_figures(output_dir: Path):
    """Generate multi-model comparison figures.

    Parameters:
        output_dir: Output directory for figures
    """
    print("[*] Generating comparison figures...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data for 3 models
    np.random.seed(42)
    n_samples = 200

    y_true = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    models_data = {}
    models_metrics = {}

    for model_name, (alpha, beta) in [
        ('Model A (Baseline)', (6, 3)),
        ('Model B (Improved)', (8, 2)),
        ('Model C (Advanced)', (7, 2.5))
    ]:
        # Generate probabilities
        y_prob = np.zeros(n_samples)
        for i in range(n_samples):
            if y_true[i] == 1:
                y_prob[i] = np.random.beta(alpha, beta)
            else:
                y_prob[i] = np.random.beta(beta, alpha)

        # Compute ROC
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_prob)

        models_data[model_name] = (fpr, tpr, roc_auc)

        # Compute metrics
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_performance_metrics(y_true, y_pred, y_prob)
        models_metrics[model_name] = metrics.to_dict()

    # 7. Multi-Model ROC Comparison
    print("  - Generating multi-model ROC comparison...")
    fig, ax = plot_multi_model_roc(
        models_data,
        title="CDSS Performance: Multi-Model ROC Comparison",
        save_path=output_dir / "fig07_multi_model_roc.pdf",
        dpi=300
    )
    plt.close(fig)

    # 8. Metrics Bar Comparison
    print("  - Generating metrics bar comparison...")
    fig, ax = plot_metrics_comparison_bar(
        models_metrics,
        title="CDSS Performance: Multi-Model Metrics Comparison",
        save_path=output_dir / "fig08_metrics_comparison.pdf",
        dpi=300
    )
    plt.close(fig)

    # 9. Radar Chart Comparison
    print("  - Generating radar chart comparison...")
    fig, ax = plot_multi_radar_comparison(
        models_metrics,
        title="CDSS Performance: Multi-Model Radar Comparison",
        save_path=output_dir / "fig09_radar_comparison.pdf",
        dpi=300
    )
    plt.close(fig)

    print("[OK] Comparison figures generated successfully!")


def generate_stratified_figures(data: dict, output_dir: Path):
    """Generate stratified performance figures.

    Parameters:
        data: Dictionary with performance data
        output_dir: Output directory for figures
    """
    print("[*] Generating stratified analysis figures...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 10. Stratified Performance Heatmap
    print("  - Generating stratified heatmap...")

    # Simulate performance across models and risk tiers
    metrics_matrix = np.array([
        [0.82, 0.85, 0.88, 0.91],  # Model A
        [0.80, 0.83, 0.86, 0.89],  # Model B
        [0.85, 0.88, 0.91, 0.94],  # Model C
    ])

    row_labels = ['Model A (Baseline)', 'Model B (Improved)', 'Model C (Advanced)']
    col_labels = ['R1 (Low)', 'R2 (Medium)', 'R3 (High)', 'R4 (Critical)']

    fig, ax = plot_stratified_heatmap(
        metrics_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title="CDSS Performance: Stratified F1-Score Heatmap",
        xlabel="Risk Tier",
        ylabel="Model",
        save_path=output_dir / "fig10_stratified_heatmap.pdf",
        dpi=300
    )
    plt.close(fig)

    print("[OK] Stratified analysis figures generated successfully!")


def generate_advanced_figures(output_dir: Path):
    """Generate advanced 3D and contour figures.

    Parameters:
        output_dir: Output directory for figures
    """
    print("[*] Generating advanced visualization figures...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 11. 3D Performance Surface
    print("  - Generating 3D performance surface...")

    # Create grid for threshold and regularization parameter
    thresholds = np.linspace(0.1, 0.9, 30)
    reg_params = np.linspace(0.001, 1.0, 30)

    # Simulate F1-score surface
    T, R = np.meshgrid(thresholds, reg_params)
    F1_surface = np.sin((T - 0.5) * np.pi * 2) * np.exp(-R) + 0.5

    fig, ax = plot_3d_performance_surface(
        thresholds, reg_params, F1_surface,
        xlabel="Classification Threshold",
        ylabel="Regularization Parameter",
        zlabel="F1-Score",
        title="CDSS Performance: 3D Performance Landscape",
        save_path=output_dir / "fig11_3d_surface.pdf",
        dpi=300
    )
    plt.close(fig)

    # 12. Contour Performance Map
    print("  - Generating contour performance map...")

    # Find optimal point
    optimal_idx = np.unravel_index(F1_surface.argmax(), F1_surface.shape)
    optimal_threshold = thresholds[optimal_idx[0]]
    optimal_reg = reg_params[optimal_idx[1]]

    fig, ax = plot_contour_performance(
        thresholds, reg_params, F1_surface,
        xlabel="Classification Threshold",
        ylabel="Regularization Parameter",
        title="CDSS Performance: F1-Score Contour Map",
        optimal_point=(optimal_threshold, optimal_reg),
        save_path=output_dir / "fig12_contour_map.pdf",
        dpi=300
    )
    plt.close(fig)

    print("[OK] Advanced visualization figures generated successfully!")


def generate_multi_class_figures(output_dir: Path):
    """Generate multi-class classification figures.

    Parameters:
        output_dir: Output directory for figures
    """
    print("[*] Generating multi-class classification figures...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic multi-class data
    np.random.seed(42)
    n_samples = 300
    n_classes = 4

    # True labels
    y_true = np.random.choice(n_classes, size=n_samples)

    # Predictions (with some confusion)
    y_pred = y_true.copy()
    confusion_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[confusion_indices] = np.random.choice(n_classes, size=len(confusion_indices))

    # 13. Multi-Class Confusion Matrix
    print("  - Generating multi-class confusion matrix...")

    from sklearn.metrics import confusion_matrix as sklearn_cm
    cm_multi = sklearn_cm(y_true, y_pred)
    class_names = ['R1 (Low)', 'R2 (Medium)', 'R3 (High)', 'R4 (Critical)']

    fig, ax = plot_multi_class_confusion_matrix(
        cm_multi,
        class_names=class_names,
        title="CDSS Performance: Multi-Class Confusion Matrix",
        save_path=output_dir / "fig13_multiclass_confusion.pdf",
        dpi=300
    )
    plt.close(fig)

    # 14. Normalized Multi-Class Confusion Matrix
    print("  - Generating normalized multi-class confusion matrix...")

    fig, ax = plot_multi_class_confusion_matrix(
        cm_multi,
        class_names=class_names,
        normalize=True,
        title="CDSS Performance: Normalized Multi-Class Confusion Matrix",
        save_path=output_dir / "fig14_multiclass_confusion_normalized.pdf",
        dpi=300
    )
    plt.close(fig)

    print("[OK] Multi-class classification figures generated successfully!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate performance visualization figures for BASICS-CDSS'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/performance',
        help='Output directory for figures (default: figures/performance)'
    )
    parser.add_argument(
        '--demo-only',
        action='store_true',
        help='Generate using synthetic demo data only'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all figure types (default if no flags specified)'
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Generate binary classification figures only'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Generate model comparison figures only'
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Generate stratified analysis figures only'
    )
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Generate advanced 3D/contour figures only'
    )
    parser.add_argument(
        '--multiclass',
        action='store_true',
        help='Generate multi-class figures only'
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("BASICS-CDSS Performance Figure Generation")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Determine which figures to generate
    generate_all = args.all or not (args.binary or args.comparison or args.stratified or args.advanced or args.multiclass)

    # Generate synthetic data
    print("[*] Generating synthetic demonstration data...")
    data = generate_synthetic_data(n_samples=200, seed=42)
    print("[OK] Synthetic data generated.")
    print()

    # Generate figures based on flags
    if generate_all or args.binary:
        generate_binary_classification_figures(data, output_dir / "binary")

    if generate_all or args.comparison:
        generate_comparison_figures(output_dir / "comparison")

    if generate_all or args.stratified:
        generate_stratified_figures(data, output_dir / "stratified")

    if generate_all or args.advanced:
        generate_advanced_figures(output_dir / "advanced")

    if generate_all or args.multiclass:
        generate_multi_class_figures(output_dir / "multiclass")

    print()
    print("=" * 70)
    print("[SUCCESS] All performance figures generated successfully!")
    print("=" * 70)
    print(f"Total figures generated: {len(list(output_dir.rglob('*.pdf')))}")
    print(f"Output location: {output_dir.absolute()}")
    print()

    # Print figure summary
    print("Figure Summary:")
    print("-" * 70)

    categories = {
        'binary': 'Binary Classification',
        'comparison': 'Multi-Model Comparison',
        'stratified': 'Stratified Analysis',
        'advanced': 'Advanced 3D/Contour',
        'multiclass': 'Multi-Class Classification'
    }

    for subdir, desc in categories.items():
        subpath = output_dir / subdir
        if subpath.exists():
            n_figs = len(list(subpath.glob('*.pdf')))
            print(f"  {desc:40s}: {n_figs:2d} figures")

    print("-" * 70)
    print()


if __name__ == '__main__':
    main()
