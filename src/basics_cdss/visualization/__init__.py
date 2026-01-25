"""
Visualization module for BASICS-CDSS evaluation results.

This module provides publication-ready plotting functions for:
- Calibration reliability diagrams
- Coverage-risk curves
- Harm concentration visualizations
- Uncertainty profile distributions
- Comparative evaluation plots
- Temporal trajectories and disease progression (Tier 1)
- Causal DAGs and intervention effects (Tier 2)
- Multi-agent interactions and workflow analysis (Tier 3)
"""

from .calibration_plots import (
    plot_reliability_diagram,
    plot_calibration_comparison,
    plot_stratified_calibration,
)

from .coverage_risk_plots import (
    plot_coverage_risk_curve,
    plot_selective_prediction_comparison,
    plot_abstention_analysis,
)

from .harm_plots import (
    plot_harm_by_tier,
    plot_escalation_analysis,
    plot_harm_concentration,
)

from .scenario_plots import (
    plot_uncertainty_distribution,
    plot_perturbation_effects,
    plot_scenario_summary,
)

from .comparison_plots import (
    plot_metric_comparison,
    plot_model_comparison_radar,
    create_evaluation_dashboard,
)

# Tier 1: Digital Twin / Temporal Analysis
from .temporal_plots import (
    plot_temporal_trajectory,
    plot_disease_progression,
    plot_counterfactual_analysis,
    plot_intervention_timing_analysis,
)

# Tier 2: Causal Simulation
from .causal_plots import (
    plot_causal_dag,
    plot_intervention_effects,
    plot_cate_heterogeneity,
    plot_confounding_analysis,
    plot_backdoor_adjustment,
)

# Tier 3: Multi-Agent Simulation
from .multiagent_plots import (
    plot_agent_interaction_network,
    plot_workflow_timeline,
    plot_alert_fatigue_dynamics,
    plot_override_rates_comparison,
    plot_system_resilience,
)

# Performance Plots
from .performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_sensitivity_specificity_curve,
    plot_threshold_analysis,
    plot_multi_model_roc,
    plot_metrics_comparison_bar,
    plot_multi_class_confusion_matrix,
)

# Advanced Charts
from .advanced_charts import (
    plot_3d_performance_surface,
    plot_contour_performance,
    plot_stratified_heatmap,
    plot_radar_chart,
    plot_multi_radar_comparison,
    plot_parallel_coordinates,
    plot_3d_scatter_performance,
)

__all__ = [
    # Calibration
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_stratified_calibration",
    # Coverage-Risk
    "plot_coverage_risk_curve",
    "plot_selective_prediction_comparison",
    "plot_abstention_analysis",
    # Harm-Aware
    "plot_harm_by_tier",
    "plot_escalation_analysis",
    "plot_harm_concentration",
    # Scenarios
    "plot_uncertainty_distribution",
    "plot_perturbation_effects",
    "plot_scenario_summary",
    # Comparison
    "plot_metric_comparison",
    "plot_model_comparison_radar",
    "create_evaluation_dashboard",
    # Tier 1: Temporal/Digital Twin
    "plot_temporal_trajectory",
    "plot_disease_progression",
    "plot_counterfactual_analysis",
    "plot_intervention_timing_analysis",
    # Tier 2: Causal
    "plot_causal_dag",
    "plot_intervention_effects",
    "plot_cate_heterogeneity",
    "plot_confounding_analysis",
    "plot_backdoor_adjustment",
    # Tier 3: Multi-Agent
    "plot_agent_interaction_network",
    "plot_workflow_timeline",
    "plot_alert_fatigue_dynamics",
    "plot_override_rates_comparison",
    "plot_system_resilience",
    # Performance Plots
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_sensitivity_specificity_curve",
    "plot_threshold_analysis",
    "plot_multi_model_roc",
    "plot_metrics_comparison_bar",
    "plot_multi_class_confusion_matrix",
    # Advanced Charts
    "plot_3d_performance_surface",
    "plot_contour_performance",
    "plot_stratified_heatmap",
    "plot_radar_chart",
    "plot_multi_radar_comparison",
    "plot_parallel_coordinates",
    "plot_3d_scatter_performance",
]
