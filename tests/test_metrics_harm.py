"""
Tests for harm-aware evaluation metrics module.
"""

import numpy as np
import pytest
from basics_cdss.metrics.harm import (DEFAULT_HARM_WEIGHTS, HarmMetrics,
                                      asymmetric_cost_matrix,
                                      compute_harm_metrics,
                                      escalation_failure_analysis,
                                      harm_by_risk_tier,
                                      harm_concentration_index,
                                      weighted_harm_loss)


class TestWeightedHarmLoss:
    """Tests for weighted harm loss computation."""

    def test_equal_errors_different_tiers(self):
        """High-risk errors should be weighted more heavily."""
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])  # 2 errors
        risk_tiers = np.array(["high", "low"])

        loss = weighted_harm_loss(y_true, y_pred, risk_tiers)

        # Loss should be > 0 due to errors
        assert loss > 0

        # High-risk error contributes more than low-risk
        # With default weights: high=10, low=1
        # Expected: (10*1 + 1*1) / 2 = 5.5
        assert loss > 5.0

    def test_no_errors(self):
        """Perfect predictions should have zero harm."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        risk_tiers = np.array(["high", "low", "medium", "high"])

        loss = weighted_harm_loss(y_true, y_pred, risk_tiers)

        assert loss == 0.0

    def test_custom_weights(self):
        """Test with custom harm weights."""
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])  # 2 errors
        risk_tiers = np.array(["critical", "minor"])
        custom_weights = {"critical": 100.0, "minor": 1.0}

        loss = weighted_harm_loss(
            y_true, y_pred, risk_tiers, harm_weights=custom_weights
        )

        # Should use custom weights
        # Expected: (100*1 + 1*1) / 2 = 50.5
        assert loss > 50.0

    def test_empty_input(self):
        """Handle empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        risk_tiers = np.array([])

        loss = weighted_harm_loss(y_true, y_pred, risk_tiers)

        assert loss == 0.0


class TestHarmByRiskTier:
    """Tests for tier-specific harm computation."""

    def test_tier_separation(self):
        """Test harm computed separately for each tier."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1])  # Errors in high and low tiers
        risk_tiers = np.array(["high", "low", "high", "low", "high", "low"])

        harm = harm_by_risk_tier(y_true, y_pred, risk_tiers)

        assert "high" in harm
        assert "low" in harm

        # High tier errors should have higher weighted harm
        assert harm["high"] > harm["low"]

    def test_all_correct_tier(self):
        """Tier with no errors should have zero harm."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])  # Errors only in indices 2, 3
        risk_tiers = np.array(["high", "high", "low", "low"])

        harm = harm_by_risk_tier(y_true, y_pred, risk_tiers)

        # High tier has no errors -> zero harm
        assert harm["high"] == 0.0
        # Low tier has errors -> positive harm
        assert harm["low"] > 0.0


class TestEscalationFailureAnalysis:
    """Tests for escalation failure detection."""

    def test_escalation_failure(self):
        """Detect when high-risk case is not escalated."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0])  # Missed escalation for first case
        risk_tiers = np.array(["high", "high", "low", "low"])

        analysis = escalation_failure_analysis(y_true, y_pred, risk_tiers)

        assert analysis["escalation_failures"] == 1  # Missed 1 high-risk case
        assert analysis["false_escalations"] == 0

    def test_false_escalation(self):
        """Detect unnecessary escalation of low-risk case."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0])  # Over-escalated third case
        risk_tiers = np.array(["high", "high", "low", "low"])

        analysis = escalation_failure_analysis(y_true, y_pred, risk_tiers)

        assert analysis["escalation_failures"] == 0
        assert analysis["false_escalations"] == 1  # 1 unnecessary escalation

    def test_perfect_escalation(self):
        """No failures when escalations are correct."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        risk_tiers = np.array(["high", "high", "low", "low"])

        analysis = escalation_failure_analysis(y_true, y_pred, risk_tiers)

        assert analysis["escalation_failures"] == 0
        assert analysis["false_escalations"] == 0

    def test_custom_high_risk_labels(self):
        """Test with custom high-risk tier labels."""
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0, 1, 0])
        risk_tiers = np.array(["critical", "critical", "routine"])

        analysis = escalation_failure_analysis(
            y_true, y_pred, risk_tiers, high_risk_labels=["critical"]
        )

        assert analysis["escalation_failures"] == 1


class TestHarmConcentrationIndex:
    """Tests for harm concentration measurement."""

    def test_all_harm_in_high_tier(self):
        """All errors in high-risk tier should have concentration = 1."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])  # Errors only in high tier
        risk_tiers = np.array(["high", "high", "low", "low"])

        concentration = harm_concentration_index(y_true, y_pred, risk_tiers)

        # All harm is in high tier
        assert concentration > 0.9  # Should be close to 1.0

    def test_balanced_harm(self):
        """Errors distributed across tiers should have lower concentration."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0])  # Errors in both tiers
        risk_tiers = np.array(["high", "high", "low", "low"])

        concentration = harm_concentration_index(y_true, y_pred, risk_tiers)

        # Harm is distributed -> concentration < 1
        assert 0.0 < concentration < 1.0

    def test_no_errors(self):
        """No errors should have zero concentration."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        risk_tiers = np.array(["high", "low", "high", "low"])

        concentration = harm_concentration_index(y_true, y_pred, risk_tiers)

        assert concentration == 0.0


class TestComputeHarmMetrics:
    """Tests for comprehensive harm metrics computation."""

    def test_complete_metrics(self):
        """Test that all metrics are computed."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        risk_tiers = np.array(
            ["high", "high", "low", "medium", "low", "high", "low", "low"]
        )

        metrics = compute_harm_metrics(y_true, y_pred, risk_tiers)

        assert isinstance(metrics, HarmMetrics)
        assert metrics.weighted_harm_loss >= 0
        assert isinstance(metrics.harm_by_tier, dict)
        assert metrics.escalation_failures >= 0
        assert metrics.false_escalations >= 0
        assert 0 <= metrics.harm_concentration <= 1

    def test_custom_weights(self):
        """Test with custom harm weights."""
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        risk_tiers = np.array(["critical", "minor"])

        custom_weights = {"critical": 100.0, "minor": 1.0}
        metrics = compute_harm_metrics(
            y_true, y_pred, risk_tiers, harm_weights=custom_weights
        )

        assert metrics.weighted_harm_loss > 0


class TestAsymmetricCostMatrix:
    """Tests for asymmetric cost computation."""

    def test_false_negative_cost(self):
        """False negatives should incur high cost."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])  # 2 false negatives

        cost = asymmetric_cost_matrix(y_true, y_pred, cost_fn=10.0, cost_fp=1.0)

        # Cost = (2 * 10.0 + 2 * 0.0) / 4 = 5.0
        assert cost == 5.0

    def test_false_positive_cost(self):
        """False positives should incur lower cost."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])  # 2 false positives

        cost = asymmetric_cost_matrix(y_true, y_pred, cost_fn=10.0, cost_fp=1.0)

        # Cost = (2 * 0.0 + 2 * 1.0) / 4 = 0.5
        assert cost == 0.5

    def test_perfect_predictions(self):
        """Perfect predictions should have zero cost."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])

        cost = asymmetric_cost_matrix(y_true, y_pred, cost_fn=10.0, cost_fp=1.0)

        assert cost == 0.0

    def test_empty_input(self):
        """Handle empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        cost = asymmetric_cost_matrix(y_true, y_pred)

        assert cost == 0.0


class TestDefaultHarmWeights:
    """Tests for default harm weight constants."""

    def test_default_weights_exist(self):
        """Verify default weights are defined."""
        assert "high" in DEFAULT_HARM_WEIGHTS
        assert "medium" in DEFAULT_HARM_WEIGHTS
        assert "low" in DEFAULT_HARM_WEIGHTS

    def test_weight_ordering(self):
        """High-risk should have higher weight than low-risk."""
        assert DEFAULT_HARM_WEIGHTS["high"] > DEFAULT_HARM_WEIGHTS["medium"]
        assert DEFAULT_HARM_WEIGHTS["medium"] > DEFAULT_HARM_WEIGHTS["low"]
