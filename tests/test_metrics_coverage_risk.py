"""
Tests for coverage-risk metrics module.
"""

import numpy as np
import pytest
from basics_cdss.metrics.coverage_risk import (SelectivePredictionMetrics,
                                               abstention_rate,
                                               area_under_risk_coverage_curve,
                                               coverage_risk_curve,
                                               selective_prediction_metrics,
                                               stratified_selective_metrics)


class TestCoverageRiskCurve:
    """Tests for coverage-risk curve computation."""

    def test_basic_curve(self):
        """Test basic coverage-risk curve generation."""
        y_true = np.array([1, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])

        coverages, risks, thresholds = coverage_risk_curve(
            y_true, y_prob, n_thresholds=10
        )

        # Should return arrays of same length
        assert len(coverages) == len(risks) == len(thresholds)

        # Coverage should be in [0, 1]
        assert np.all((coverages >= 0) & (coverages <= 1))

    def test_monotonic_coverage(self):
        """Coverage should decrease as threshold increases."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4])

        coverages, risks, thresholds = coverage_risk_curve(
            y_true, y_prob, n_thresholds=20
        )

        # Coverage should generally decrease (allowing for ties)
        for i in range(len(coverages) - 1):
            assert coverages[i] >= coverages[i + 1] - 1e-6

    def test_custom_risk_proxy(self):
        """Test with custom risk proxy."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        risk_proxy = np.array([0.1, 0.9, 0.2, 0.8])  # Custom risk scores

        coverages, risks, thresholds = coverage_risk_curve(
            y_true, y_prob, risk_proxy=risk_proxy, n_thresholds=5
        )

        assert len(coverages) > 0

    def test_empty_input(self):
        """Handle empty arrays."""
        y_true = np.array([])
        y_prob = np.array([])

        coverages, risks, thresholds = coverage_risk_curve(y_true, y_prob)

        assert len(coverages) == 0


class TestAreaUnderRiskCoverageCurve:
    """Tests for AURC computation."""

    def test_zero_risk(self):
        """Perfect predictions should have low AURC."""
        coverages = np.array([0.0, 0.5, 1.0])
        risks = np.array([0.0, 0.0, 0.0])  # No risk at any coverage

        aurc = area_under_risk_coverage_curve(coverages, risks)

        assert aurc == 0.0

    def test_high_risk(self):
        """High risk predictions should have high AURC."""
        coverages = np.array([0.0, 0.5, 1.0])
        risks = np.array([0.0, 0.5, 1.0])  # Linear increase in risk

        aurc = area_under_risk_coverage_curve(coverages, risks)

        assert aurc > 0.2  # Should be significantly positive

    def test_handles_nan(self):
        """Should handle NaN values gracefully."""
        coverages = np.array([0.0, 0.5, 1.0])
        risks = np.array([np.nan, 0.1, 0.2])

        aurc = area_under_risk_coverage_curve(coverages, risks)

        # Should compute with valid values only
        assert not np.isnan(aurc)


class TestSelectivePredictionMetrics:
    """Tests for comprehensive selective prediction metrics."""

    def test_basic_metrics(self):
        """Test basic selective prediction metrics."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4])

        metrics = selective_prediction_metrics(y_true, y_prob, n_thresholds=20)

        assert isinstance(metrics, SelectivePredictionMetrics)
        assert metrics.aurc >= 0
        assert metrics.coverage_curve is not None
        assert metrics.risk_curve is not None
        assert metrics.thresholds is not None

    def test_target_coverage(self):
        """Test risk at target coverage."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4, 0.75, 0.95])

        metrics = selective_prediction_metrics(
            y_true, y_prob, target_coverage=0.8, n_thresholds=50
        )

        # Should find risk at ~80% coverage
        if metrics.risk_at_coverage_threshold is not None:
            assert 0 <= metrics.risk_at_coverage_threshold <= 1

    def test_target_risk(self):
        """Test coverage at target risk."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4, 0.75, 0.95])

        metrics = selective_prediction_metrics(
            y_true, y_prob, target_risk=0.1, n_thresholds=50
        )

        # Should find maximum coverage while maintaining risk <= 0.1
        if metrics.coverage_at_risk_threshold is not None:
            assert 0 <= metrics.coverage_at_risk_threshold <= 1


class TestAbstentionRate:
    """Tests for abstention rate computation."""

    def test_no_abstention(self):
        """All confident predictions should have 0% abstention."""
        y_prob = np.array([0.9, 0.8, 0.7, 0.85, 0.95])
        rate = abstention_rate(y_prob, threshold=0.5)

        assert rate == 0.0

    def test_full_abstention(self):
        """All uncertain predictions should have 100% abstention."""
        y_prob = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        rate = abstention_rate(y_prob, threshold=0.5)

        assert rate == 1.0

    def test_partial_abstention(self):
        """Mixed confidence should have intermediate abstention rate."""
        y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7])  # 3 confident, 2 uncertain
        rate = abstention_rate(y_prob, threshold=0.5)

        assert 0.3 < rate < 0.5  # ~40% abstention

    def test_empty_input(self):
        """Handle empty array."""
        y_prob = np.array([])
        rate = abstention_rate(y_prob)

        assert rate == 0.0


class TestStratifiedSelectiveMetrics:
    """Tests for risk-tier stratified selective prediction."""

    def test_stratified_computation(self):
        """Test selective prediction stratified by risk tier."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.85, 0.25])
        risk_tiers = np.array(
            ["high", "low", "high", "low", "medium", "low", "high", "low"]
        )

        metrics = stratified_selective_metrics(
            y_true, y_prob, risk_tiers, n_thresholds=10
        )

        # Should have metrics for each tier
        assert "high" in metrics
        assert "low" in metrics
        assert "medium" in metrics

        # Each tier should have SelectivePredictionMetrics
        assert isinstance(metrics["high"], SelectivePredictionMetrics)

    def test_single_tier(self):
        """Handle case with single risk tier."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        risk_tiers = np.array(["high", "high", "high", "high"])

        metrics = stratified_selective_metrics(y_true, y_prob, risk_tiers)

        assert len(metrics) == 1
        assert "high" in metrics
