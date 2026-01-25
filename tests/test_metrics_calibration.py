"""
Tests for calibration metrics module.
"""

import pytest
import numpy as np
from basics_cdss.metrics.calibration import (
    expected_calibration_error,
    brier_score,
    reliability_curve,
    stratified_calibration_metrics,
    calibration_summary,
    CalibrationMetrics,
)


class TestExpectedCalibrationError:
    """Tests for ECE computation."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE=0."""
        # Predictions match true outcomes
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert ece < 0.01  # Should be very close to 0

    def test_overconfident_predictions(self):
        """Overconfident predictions should have positive ECE."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.9, 0.8, 0.9])  # High confidence but 40% error rate

        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert ece > 0.1  # Should detect miscalibration

    def test_empty_input(self):
        """Handle empty arrays gracefully."""
        y_true = np.array([])
        y_prob = np.array([])

        ece = expected_calibration_error(y_true, y_prob)
        assert ece == 0.0

    def test_deterministic(self):
        """Same inputs should produce same output."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        ece1 = expected_calibration_error(y_true, y_prob, n_bins=10)
        ece2 = expected_calibration_error(y_true, y_prob, n_bins=10)

        assert ece1 == ece2


class TestBrierScore:
    """Tests for Brier score computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score = 0."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])

        bs = brier_score(y_true, y_prob)
        assert bs == 0.0

    def test_worst_predictions(self):
        """Worst predictions should have Brier score = 1."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])  # Completely wrong

        bs = brier_score(y_true, y_prob)
        assert bs == 1.0

    def test_range(self):
        """Brier score should be in [0, 1]."""
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.random(50)

        bs = brier_score(y_true, y_prob)
        assert 0.0 <= bs <= 1.0

    def test_empty_input(self):
        """Handle empty arrays."""
        y_true = np.array([])
        y_prob = np.array([])

        bs = brier_score(y_true, y_prob)
        assert bs == 0.0


class TestReliabilityCurve:
    """Tests for reliability curve computation."""

    def test_uniform_binning(self):
        """Test uniform binning strategy."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4, 0.75, 0.95])

        confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=5, strategy="uniform")

        # Should return non-empty arrays
        assert len(confs) > 0
        assert len(accs) > 0
        assert len(counts) > 0

        # Lengths should match
        assert len(confs) == len(accs) == len(counts)

    def test_quantile_binning(self):
        """Test quantile binning strategy."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4, 0.75, 0.95])

        confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=5, strategy="quantile")

        assert len(confs) > 0

    def test_invalid_strategy(self):
        """Invalid strategy should raise error."""
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8])

        with pytest.raises(ValueError, match="Unknown strategy"):
            reliability_curve(y_true, y_prob, strategy="invalid")

    def test_empty_input(self):
        """Handle empty arrays."""
        y_true = np.array([])
        y_prob = np.array([])

        confs, accs, counts = reliability_curve(y_true, y_prob)

        assert len(confs) == 0
        assert len(accs) == 0
        assert len(counts) == 0


class TestStratifiedCalibrationMetrics:
    """Tests for risk-tier stratified calibration."""

    def test_stratified_computation(self):
        """Test calibration stratified by risk tier."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.1])
        risk_tiers = np.array(["high", "low", "high", "low", "medium", "low"])

        metrics = stratified_calibration_metrics(y_true, y_prob, risk_tiers, n_bins=5)

        # Should have metrics for each tier
        assert "high" in metrics
        assert "low" in metrics
        assert "medium" in metrics

        # Each tier should have CalibrationMetrics
        assert isinstance(metrics["high"], CalibrationMetrics)

    def test_single_tier(self):
        """Handle case with single risk tier."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        risk_tiers = np.array(["high", "high", "high", "high"])

        metrics = stratified_calibration_metrics(y_true, y_prob, risk_tiers)

        assert len(metrics) == 1
        assert "high" in metrics


class TestCalibrationSummary:
    """Tests for comprehensive calibration summary."""

    def test_overall_only(self):
        """Test summary without risk tier stratification."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])

        summary = calibration_summary(y_true, y_prob, risk_tiers=None)

        assert "overall" in summary
        assert "ece" in summary["overall"]
        assert "brier_score" in summary["overall"]
        assert "reliability_curve" in summary["overall"]

    def test_with_stratification(self):
        """Test summary with risk tier stratification."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.1])
        risk_tiers = np.array(["high", "low", "high", "low", "high", "low"])

        summary = calibration_summary(y_true, y_prob, risk_tiers=risk_tiers)

        assert "overall" in summary
        assert "by_risk_tier" in summary
        assert "high" in summary["by_risk_tier"]
        assert "low" in summary["by_risk_tier"]
