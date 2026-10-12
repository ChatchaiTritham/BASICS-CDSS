"""Tests for the metrics added in v1.4.0 (calibration slope/intercept, exact AURC)."""
import numpy as np
import pytest

from basics_cdss.metrics import aurc_confidence, calibration_slope_intercept


def test_slope_intercept_perfectly_calibrated():
    rng = np.random.RandomState(0)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random_sample(p.size) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_slope_below_one_for_overconfident_predictions():
    rng = np.random.RandomState(1)
    lp = rng.normal(0, 1.5, 20000)
    y = (rng.random_sample(lp.size) < 1 / (1 + np.exp(-lp))).astype(int)
    p_over = 1 / (1 + np.exp(-2.0 * lp))  # logits doubled: too extreme
    slope, _ = calibration_slope_intercept(y, p_over)
    assert slope == pytest.approx(0.5, abs=0.05)


def test_intercept_positive_when_risk_underestimated():
    rng = np.random.RandomState(2)
    lp = rng.normal(0, 1, 20000)
    y = (rng.random_sample(lp.size) < 1 / (1 + np.exp(-lp))).astype(int)
    p_low = 1 / (1 + np.exp(-(lp - 1.0)))  # every logit shifted down by 1
    _, intercept = calibration_slope_intercept(y, p_low)
    assert intercept == pytest.approx(1.0, abs=0.06)


def test_aurc_matches_hand_computation():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.4, 0.6])
    # confidences 0.9, 0.8, 0.6, 0.6; errors 0, 0, 1, 1 (ties keep input order)
    # selective risks: 0/1, 0/2, 1/3, 2/4 -> mean = (0 + 0 + 1/3 + 1/2) / 4
    aurc, risk80 = aurc_confidence(y, p)
    assert aurc == pytest.approx((1 / 3 + 1 / 2) / 4)
    assert risk80 == pytest.approx(2 / 4)  # ceil(0.8 * 4) = 4 most confident


def test_aurc_zero_for_perfect_classifier():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.3, 0.8, 0.7])
    aurc, risk80 = aurc_confidence(y, p)
    assert aurc == 0.0 and risk80 == 0.0
