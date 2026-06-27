"""
Focused unit tests for BASICS-CDSS pure/deterministic functions.

Targets metric primitives (calibration, coverage-risk/selective prediction,
harm-aware loss, classification performance) with tiny hand-made inputs whose
expected values can be computed by hand. No network, no training, no datasets.
"""

import os
import sys

import numpy as np
import pytest

# Make the src/ layout importable without installing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from basics_cdss.metrics import calibration, coverage_risk, harm, performance


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_brier_score_perfect_and_worst():
    # Perfect predictions -> Brier 0.0
    y_true = np.array([1, 0, 1, 0])
    assert calibration.brier_score(y_true, y_true.astype(float)) == pytest.approx(0.0)
    # Worst predictions (fully confident & wrong) -> Brier 1.0
    y_prob = 1.0 - y_true.astype(float)
    assert calibration.brier_score(y_true, y_prob) == pytest.approx(1.0)


def test_ece_perfectly_calibrated_is_zero():
    # In each bin confidence equals empirical accuracy -> ECE 0.
    # Two probs at 1.0 with label 1, two at 0.0 with label 0.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    ece = calibration.expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    # Bounds: ECE always in [0, 1].
    assert 0.0 <= ece <= 1.0


def test_reliability_curve_shapes_and_bounds():
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    confs, accs, counts = calibration.reliability_curve(y_true, y_prob, n_bins=5)
    # Same length triple, no empty bins reported.
    assert len(confs) == len(accs) == len(counts)
    assert counts.sum() == len(y_true)
    # Accuracies are fractions, confidences are probabilities.
    assert np.all((accs >= 0.0) & (accs <= 1.0))
    assert np.all((confs >= 0.0) & (confs <= 1.0))


# --------------------------------------------------------------------------- #
# Coverage-risk / selective prediction
# --------------------------------------------------------------------------- #
def test_abstention_rate_threshold_logic():
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    # Below 0.5: {0.3, 0.2} -> 2/5
    assert coverage_risk.abstention_rate(y_prob, threshold=0.5) == pytest.approx(0.4)
    # Threshold 0 abstains nobody.
    assert coverage_risk.abstention_rate(y_prob, threshold=0.0) == pytest.approx(0.0)


def test_aurc_trapezoid_known_value():
    # Trapezoidal area under a straight line from (0,0) to (1,0.2) = 0.1
    coverages = np.array([0.0, 0.5, 1.0])
    risks = np.array([0.0, 0.1, 0.2])
    assert coverage_risk.area_under_risk_coverage_curve(
        coverages, risks
    ) == pytest.approx(0.1)


def test_coverage_risk_curve_monotone_coverage():
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    coverages, risks, thresholds = coverage_risk.coverage_risk_curve(
        y_true, y_prob, n_thresholds=11
    )
    assert len(coverages) == len(risks) == len(thresholds) == 11
    # As the acceptance threshold rises, coverage is non-increasing.
    assert np.all(np.diff(coverages) <= 1e-12)
    # Coverage at threshold 0 accepts everyone.
    assert coverages[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Harm-aware loss
# --------------------------------------------------------------------------- #
def test_weighted_harm_loss_hand_computed():
    # Errors at indices 1 (low, w=1) and 2 (high, w=10). N=4.
    # weighted loss = (10*1 + 1*1) / 4 = 2.75
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    risk_tiers = np.array(["high", "low", "high", "low"])
    loss = harm.weighted_harm_loss(y_true, y_pred, risk_tiers)
    assert loss == pytest.approx(2.75)


def test_weighted_harm_loss_zero_when_no_errors():
    y_true = np.array([1, 0, 1, 0])
    risk_tiers = np.array(["high", "low", "high", "low"])
    assert harm.weighted_harm_loss(y_true, y_true, risk_tiers) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def test_confusion_matrix_counts():
    # y_true vs y_pred -> tn,fp,fn,tp computed by hand.
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = performance.confusion_matrix(y_true, y_pred)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 2)
    assert cm.total == 5
    assert cm.prevalence == pytest.approx(3 / 5)


def test_compute_performance_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_pred = y_true.copy()
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = performance.compute_performance_metrics(y_true, y_pred, y_prob)
    assert m.accuracy == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.specificity == pytest.approx(1.0)
    assert m.f1_score == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    # All rates bounded in [0,1].
    for v in (m.accuracy, m.precision, m.recall, m.specificity, m.fpr, m.fnr):
        assert 0.0 <= v <= 1.0
