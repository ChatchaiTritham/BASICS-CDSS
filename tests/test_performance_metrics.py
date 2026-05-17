"""
Test performance metrics module with sample data
Tests all major functions in the performance metrics module
"""
import numpy as np
from basics_cdss.metrics import (
    confusion_matrix,
    compute_performance_metrics,
    compute_roc_curve,
    compute_pr_curve,
    bootstrap_confidence_interval,
    mcnemar_test,
    stratified_performance_metrics,
    sensitivity_specificity_analysis,
)

def test_confusion_matrix():
    """Test confusion matrix computation."""
    print("[TEST] confusion_matrix()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])

    result = confusion_matrix(y_true, y_pred)

    print(f"  True Positives:  {result.tp}")
    print(f"  True Negatives:  {result.tn}")
    print(f"  False Positives: {result.fp}")
    print(f"  False Negatives: {result.fn}")
    print(f"  Total:           {result.total}")
    print(f"  Prevalence:      {result.prevalence:.4f}")

    assert result.tp == 3, "TP should be 3"
    assert result.tn == 3, "TN should be 3"
    assert result.fp == 1, "FP should be 1"
    assert result.fn == 1, "FN should be 1"
    print("  [PASS]\n")

def test_performance_metrics():
    """Test comprehensive performance metrics."""
    print("[TEST] compute_performance_metrics()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.95, 0.6, 0.05])

    metrics = compute_performance_metrics(y_true, y_pred, y_prob)

    print(f"  Accuracy:    {metrics.accuracy:.4f}")
    print(f"  Precision:   {metrics.precision:.4f}")
    print(f"  Recall:      {metrics.recall:.4f}")
    print(f"  Specificity: {metrics.specificity:.4f}")
    print(f"  F1 Score:    {metrics.f1_score:.4f}")
    print(f"  ROC-AUC:     {metrics.roc_auc:.4f}")
    print(f"  PR-AUC:      {metrics.pr_auc:.4f}")
    print(f"  MCC:         {metrics.mcc:.4f}")
    print(f"  Kappa:       {metrics.kappa:.4f}")

    assert 0.0 <= metrics.accuracy <= 1.0, "Accuracy should be in [0,1]"
    assert 0.0 <= metrics.roc_auc <= 1.0, "ROC-AUC should be in [0,1]"
    assert 0.0 <= metrics.pr_auc <= 1.0, "PR-AUC should be in [0,1]"
    print("  [PASS]\n")

def test_roc_curve():
    """Test ROC curve computation."""
    print("[TEST] compute_roc_curve()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.95, 0.6, 0.05])

    fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)

    print(f"  FPR points:  {len(fpr)}")
    print(f"  TPR points:  {len(tpr)}")
    print(f"  Thresholds:  {len(thresholds)}")

    assert len(fpr) == len(tpr), "FPR and TPR should have same length"
    assert len(fpr) == len(thresholds), "FPR and thresholds should have same length"
    print("  [PASS]\n")

def test_pr_curve():
    """Test Precision-Recall curve computation."""
    print("[TEST] compute_pr_curve()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.95, 0.6, 0.05])

    precision, recall, thresholds = compute_pr_curve(y_true, y_prob)

    print(f"  Precision points: {len(precision)}")
    print(f"  Recall points:    {len(recall)}")
    print(f"  Thresholds:       {len(thresholds)}")

    assert len(precision) == len(recall), "Precision and Recall should have same length"
    print("  [PASS]\n")

def test_sensitivity_specificity():
    """Test sensitivity-specificity analysis."""
    print("[TEST] sensitivity_specificity_analysis()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.95, 0.6, 0.05])

    # Test with default thresholds
    result = sensitivity_specificity_analysis(y_true, y_prob)

    print(f"  Result type: {type(result).__name__}")
    print(f"  Rows: {len(result)}")
    print(f"  Columns: {list(result.columns)}")
    print(f"  Sample row:\n{result.iloc[0]}")

    assert len(result) > 0, "Should have at least one row"
    assert 'threshold' in result.columns, "Should have threshold column"
    assert 'sensitivity' in result.columns, "Should have sensitivity column"
    assert 'specificity' in result.columns, "Should have specificity column"
    print("  [PASS]\n")

def test_bootstrap_ci():
    """Test bootstrap confidence interval."""
    print("[TEST] bootstrap_confidence_interval()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0] * 10)  # Larger sample
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 10)

    point_est, ci_lower, ci_upper = bootstrap_confidence_interval(
        y_true, y_pred,
        metric='accuracy',
        n_bootstrap=100,  # Reduced for speed
        confidence_level=0.95,
        seed=42
    )

    print(f"  Point estimate: {point_est:.4f}")
    print(f"  CI lower:       {ci_lower:.4f}")
    print(f"  CI upper:       {ci_upper:.4f}")
    print(f"  CI width:       {ci_upper - ci_lower:.4f}")

    assert ci_lower <= point_est <= ci_upper, "Point estimate should be within CI"
    assert ci_lower < ci_upper, "CI lower should be less than upper"
    print("  [PASS]\n")

def test_mcnemar():
    """Test McNemar's test for model comparison."""
    print("[TEST] mcnemar_test()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0] * 5)  # Larger sample
    y_pred1 = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 5)
    y_pred2 = np.array([0, 1, 1, 1, 1, 1, 0, 0] * 5)

    statistic, p_value = mcnemar_test(y_true, y_pred1, y_pred2)

    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value:   {p_value:.4f}")
    print(f"  Significant at p<0.05: {p_value < 0.05}")

    assert p_value >= 0.0, "P-value should be non-negative"
    assert p_value <= 1.0, "P-value should be <= 1"
    print("  [PASS]\n")

def test_stratified_metrics():
    """Test stratified performance metrics."""
    print("[TEST] stratified_performance_metrics()")
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.95, 0.6, 0.05])
    risk_tiers = np.array(['low', 'low', 'high', 'high', 'medium', 'high', 'low', 'low'])

    result = stratified_performance_metrics(y_true, y_pred, y_prob, strata=risk_tiers)

    print(f"  Result type: {type(result).__name__}")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Overall Accuracy: {result['overall'].accuracy:.4f}")
    for tier in ['low', 'medium', 'high']:
        if tier in result:
            print(f"  {tier.capitalize()}: Accuracy={result[tier].accuracy:.4f}, F1={result[tier].f1_score:.4f}")

    assert 'overall' in result, "Should have overall metrics"
    assert isinstance(result['overall'], type(compute_performance_metrics(y_true, y_pred))), "Should return PerformanceMetrics"
    print("  [PASS]\n")

def main():
    print("=" * 80)
    print("Performance Metrics Functional Test")
    print("=" * 80)
    print()

    tests = [
        test_confusion_matrix,
        test_performance_metrics,
        test_roc_curve,
        test_pr_curve,
        test_sensitivity_specificity,
        test_bootstrap_ci,
        test_mcnemar,
        test_stratified_metrics,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {str(e)}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Tests run:    {passed + failed}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    print()

    if failed == 0:
        print("[SUCCESS] All performance metrics tests passed!")
        return 0
    else:
        print("[FAILURE] Some tests failed")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
