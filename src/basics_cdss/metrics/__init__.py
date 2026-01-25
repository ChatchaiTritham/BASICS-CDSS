# Calibration metrics
from .calibration import (
    CalibrationMetrics,
    expected_calibration_error,
    brier_score,
    reliability_curve,
    stratified_calibration_metrics,
    calibration_summary,
)

# Coverage-risk metrics
from .coverage_risk import (
    SelectivePredictionMetrics,
    coverage_risk_curve,
    area_under_risk_coverage_curve,
    selective_prediction_metrics,
    abstention_rate,
    stratified_selective_metrics,
)

# Harm-aware metrics
from .harm import (
    HarmMetrics,
    DEFAULT_HARM_WEIGHTS,
    weighted_harm_loss,
    harm_by_risk_tier,
    escalation_failure_analysis,
    harm_concentration_index,
    compute_harm_metrics,
    asymmetric_cost_matrix,
)

# Performance metrics
from .performance import (
    PerformanceMetrics,
    ConfusionMatrixMetrics,
    confusion_matrix,
    compute_performance_metrics,
    stratified_performance_metrics,
    compute_roc_curve,
    compute_pr_curve,
    sensitivity_specificity_analysis,
    bootstrap_confidence_interval,
    mcnemar_test,
    multi_class_metrics,
    performance_summary,
)

__all__ = [
    # Calibration
    "CalibrationMetrics",
    "expected_calibration_error",
    "brier_score",
    "reliability_curve",
    "stratified_calibration_metrics",
    "calibration_summary",
    # Coverage-Risk
    "SelectivePredictionMetrics",
    "coverage_risk_curve",
    "area_under_risk_coverage_curve",
    "selective_prediction_metrics",
    "abstention_rate",
    "stratified_selective_metrics",
    # Harm-aware
    "HarmMetrics",
    "DEFAULT_HARM_WEIGHTS",
    "weighted_harm_loss",
    "harm_by_risk_tier",
    "escalation_failure_analysis",
    "harm_concentration_index",
    "compute_harm_metrics",
    "asymmetric_cost_matrix",
    # Performance
    "PerformanceMetrics",
    "ConfusionMatrixMetrics",
    "confusion_matrix",
    "compute_performance_metrics",
    "stratified_performance_metrics",
    "compute_roc_curve",
    "compute_pr_curve",
    "sensitivity_specificity_analysis",
    "bootstrap_confidence_interval",
    "mcnemar_test",
    "multi_class_metrics",
    "performance_summary",
]
