# Clinical Metrics Guide: Phase 1 - Medical AI Validation

**Version 2.1.0** | **Status:** Production-Ready

## Table of Contents

1. [Overview](#overview)
2. [Why These Metrics Matter](#why-these-metrics-matter)
3. [Clinical Utility Metrics](#clinical-utility-metrics)
4. [Fairness Metrics](#fairness-metrics)
5. [Conformal Prediction](#conformal-prediction)
6. [Quick Start Examples](#quick-start-examples)
7. [Visualization Gallery](#visualization-gallery)
8. [FDA Compliance Guidelines](#fda-compliance-guidelines)
9. [API Reference](#api-reference)
10. [Clinical Interpretation](#clinical-interpretation)

---

## Overview

The **Clinical Metrics** module provides comprehensive evaluation tools for medical AI systems, implementing **Phase 1: Critical Metrics** essential for:

- **FDA 510(k) submissions** and regulatory approval
- **Ethical AI compliance** (EU AI Act, WHO guidelines)
- **Health equity** assessment and bias mitigation
- **Clinical decision support** validation with uncertainty quantification

### Three Pillars of Phase 1

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 1: Critical Metrics               │
├─────────────────────────────────────────────────────────────┤
│  1. Clinical Utility      2. Fairness          3. Conformal │
│     └─ Net Benefit           └─ Demographic      └─ Prediction│
│     └─ NNT                      Parity            Sets     │
│     └─ Clinical Impact       └─ Equalized      └─ Coverage  │
│                                 Odds              Guarantees│
│                              └─ Calibration                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Why These Metrics Matter

### 1. Clinical Utility Metrics

**Problem:** Traditional ML metrics (accuracy, AUC) don't answer:
- *"Should I deploy this model in my hospital?"*
- *"How many patients benefit vs. how many are harmed?"*
- *"What's the cost-effectiveness?"*

**Solution:** Decision Curve Analysis and Net Benefit quantify **real clinical value**.

**Example:** A model with 95% accuracy might have **negative net benefit** if:
- False positives lead to unnecessary invasive procedures
- The harm from FPs outweighs benefits from TPs
- Alternative strategies (treat all, treat none) perform better

### 2. Fairness Metrics

**Problem:** AI models can perpetuate or amplify healthcare disparities:
- Lower sensitivity for minority groups → missed diagnoses
- Systematic over-prediction for certain demographics → unnecessary treatment
- Historical bias in training data → unfair outcomes

**Real Example:** In 2019, Obermeyer et al. found a widely-used healthcare algorithm exhibited **significant racial bias**, systematically under-predicting disease severity for Black patients ([Science, 2019](https://science.sciencemag.org/content/366/6464/447)).

**Solution:** Comprehensive fairness assessment across:
- **Demographic Parity:** Equal treatment rates across groups
- **Equalized Odds:** Equal TPR and FPR across groups
- **Calibration:** Predicted probabilities match observed frequencies

### 3. Conformal Prediction

**Problem:** Most ML models provide:
- Point predictions without uncertainty
- Unreliable confidence scores
- No guarantees about coverage

**Example:** A model predicts "80% probability of disease" for 100 patients.
- Without conformal prediction: Unknown if 80 actually have disease
- With conformal prediction: **Guaranteed** that ≥90% of prediction sets contain true diagnosis

**Solution:** **Distribution-free** uncertainty quantification with **mathematical guarantees**:
- P(Y ∈ C(X)) ≥ 1 - α (guaranteed coverage)
- Works for any model (no distributional assumptions)
- Provides prediction sets instead of point predictions

**Clinical Value:**
- **Safe AI deployment:** Model can say "I don't know" when uncertain
- **Regulatory compliance:** Demonstrate controlled false negative rates
- **Trust:** Clinicians get rigorous uncertainty estimates

---

## Clinical Utility Metrics

### 1. Net Benefit

**Definition:** Net Benefit quantifies clinical value by balancing true positives (benefits) against false positives (harms):

```
NB(pt) = (TP/N) - (FP/N) × [pt/(1-pt)]
```

where:
- **pt** = probability threshold
- **pt/(1-pt)** = harm-to-benefit ratio (odds at threshold)
- **TP/N** = true positive rate (benefit)
- **FP/N** = false positive rate (harm weighted by odds)

**Interpretation:**
- **NB > 0:** Model provides clinical benefit
- **NB < 0:** Model causes net harm
- **Higher NB:** Better clinical value

**Code Example:**

```python
from basics_cdss.clinical_metrics import calculate_net_benefit

# Calculate net benefit at 30% threshold
nb = calculate_net_benefit(y_true, y_pred_proba, threshold=0.3)

print(f"Net Benefit: {nb.net_benefit:.4f}")
print(f"True Positives: {nb.n_true_positives}/{nb.n_total}")
print(f"False Positives: {nb.n_false_positives}/{nb.n_total}")
print(f"Harm-to-Benefit Ratio: {nb.harm_to_benefit_ratio:.2f}")
```

### 2. Decision Curve Analysis (DCA)

**Purpose:** Compare model across **all** clinically reasonable thresholds vs.:
- **Treat All:** Everyone receives intervention
- **Treat None:** No one receives intervention

**The model is clinically useful when its curve is above both alternatives.**

**Code Example:**

```python
from basics_cdss.clinical_metrics import decision_curve_analysis
from basics_cdss.visualization import plot_decision_curve

# Perform DCA
dca = decision_curve_analysis(y_true, y_pred_proba)

# Plot
plot_decision_curve(dca, save_path='decision_curve.pdf')

# Find useful threshold range
print(f"Model useful for thresholds: {dca.threshold_range}")
# Output: Model useful for thresholds: (0.12, 0.68)
```

**Clinical Interpretation:**

| Threshold Range | Clinical Scenario | Recommended Action |
|----------------|-------------------|-------------------|
| 0.12 - 0.68 | Model outperforms alternatives | **Deploy model** |
| < 0.12 | Treat all performs better | Treat everyone |
| > 0.68 | Treat none performs better | Don't treat anyone |

### 3. Number Needed to Treat (NNT)

**Definition:** How many patients must be treated to prevent one adverse event.

```
NNT = 1 / ARR
ARR = |Control Event Rate - Treatment Event Rate|
```

**Clinical Interpretation:**

| NNT Value | Effectiveness | Clinical Decision |
|-----------|---------------|------------------|
| NNT < 10 | Excellent | Strongly recommend |
| NNT 10-20 | Moderate | Recommend if cost-effective |
| NNT > 20 | Limited | Reconsider deployment |

**Code Example:**

```python
from basics_cdss.clinical_metrics import calculate_nnt

nnt = calculate_nnt(y_true, y_pred, control_event_rate=0.30)

print(f"NNT: {nnt.nnt:.1f}")
print(f"ARR: {nnt.arr:.1f}%")
print(f"Interpretation: Treat {nnt.nnt:.0f} patients to prevent 1 event")

# With confidence interval
if nnt.confidence_interval:
    print(f"95% CI: [{nnt.confidence_interval[0]:.1f}, {nnt.confidence_interval[1]:.1f}]")
```

### 4. Clinical Impact

**Purpose:** Assess practical deployment implications:
- How many patients classified as high-risk?
- What proportion are true positives (PPV)?
- How many need to screen to find one case (NNS)?

**Code Example:**

```python
from basics_cdss.clinical_metrics import clinical_impact_analysis
from basics_cdss.visualization import plot_clinical_impact

impact = clinical_impact_analysis(y_true, y_pred_proba, threshold=0.3)

plot_clinical_impact(impact, save_path='clinical_impact.pdf')

print(f"Classify {impact.percent_high_risk:.1f}% as high-risk")
print(f"PPV: {impact.ppv:.3f} ({impact.n_true_positives}/{impact.n_high_risk} correct)")
print(f"NPV: {impact.npv:.3f}")
print(f"Screen {impact.number_needed_to_screen:.1f} to find 1 true case")
```

---

## Fairness Metrics

### 1. Demographic Parity

**Definition:** Positive prediction rate should be equal across protected groups:

```
P(Ŷ = 1 | A = a) = P(Ŷ = 1 | A = b)  for all groups a, b
```

**Fairness Criterion:** Max difference ≤ 0.1 (10%)

**Code Example:**

```python
from basics_cdss.clinical_metrics import demographic_parity
from basics_cdss.visualization import plot_demographic_parity

# Assess parity across race
dp = demographic_parity(y_pred, race, threshold=0.1)

plot_demographic_parity(dp, save_path='demographic_parity.pdf')

if not dp.is_fair:
    print(f"WARNING: Demographic parity violated!")
    print(f"Max difference: {dp.parity_difference:.3f}")
    print(f"Rates by group:")
    for group, rate in dp.group_positive_rates.items():
        print(f"  {group}: {rate:.3f}")
```

**Clinical Interpretation:**
- **Violated:** Model may systematically over- or under-predict for certain groups
- **Impact:** Unequal access to treatment or screening
- **Action:** Investigate causes, consider group-specific thresholds

### 2. Equalized Odds

**Definition:** Equal TPR and FPR across groups:

```
P(Ŷ = 1 | Y = y, A = a) = P(Ŷ = 1 | Y = y, A = b)  for y ∈ {0,1}
```

**Fairness Criterion:**
- TPR difference ≤ 0.1
- FPR difference ≤ 0.1

**Code Example:**

```python
from basics_cdss.clinical_metrics import equalized_odds
from basics_cdss.visualization import plot_equalized_odds

eo = equalized_odds(y_true, y_pred, race, threshold=0.1)

plot_equalized_odds(eo, save_path='equalized_odds.pdf')

print(f"TPR difference: {eo.tpr_difference:.3f}")
print(f"FPR difference: {eo.fpr_difference:.3f}")

if not eo.is_fair:
    print("WARNING: Equalized odds violated!")
    print("TPR by group:", eo.group_tpr)
    print("FPR by group:", eo.group_fpr)
```

**Clinical Interpretation:**
- **TPR violation:** Unequal sensitivity → some groups have more missed diagnoses
- **FPR violation:** Unequal specificity → some groups have more false alarms
- **Impact:** Differential quality of care across demographics

### 3. Disparate Impact

**Definition:** 80% rule from employment law - selection rate ratio should be ≥ 0.8:

```
DI = P(Ŷ = 1 | A = unprivileged) / P(Ŷ = 1 | A = privileged)
```

**Fairness Criterion:** 0.8 ≤ DI ≤ 1.25

**Code Example:**

```python
from basics_cdss.clinical_metrics import disparate_impact
from basics_cdss.visualization import plot_disparate_impact

# Compare minority groups to White
di_results = []
for group in ['Black', 'Asian', 'Hispanic']:
    di = disparate_impact(y_pred, race, privileged_group='White',
                          unprivileged_group=group)
    di_results.append(di)

    if not di.four_fifths_rule:
        print(f"WARNING: {group} fails 80% rule (DI = {di.disparate_impact_ratio:.3f})")

plot_disparate_impact(di_results, save_path='disparate_impact.pdf')
```

### 4. Calibration Fairness

**Definition:** Predicted probabilities match observed frequencies **for each group**:

```
P(Y = 1 | Score = s, A = a) = s  for all groups a
```

**Code Example:**

```python
from basics_cdss.clinical_metrics import calibration_by_group
from basics_cdss.visualization import plot_calibration_by_group

calib = calibration_by_group(y_true, y_pred_proba, race, threshold=0.1)

plot_calibration_by_group(calib, save_path='calibration.pdf')

print("Calibration Error by Group:")
for group, error in calib.calibration_error.items():
    print(f"  {group}: {error:.3f}")

if not calib.is_calibrated:
    print(f"WARNING: Poor calibration (max error: {calib.max_calibration_error:.3f})")
```

**Clinical Interpretation:**
- **Well-calibrated:** "70% risk" means actual risk is ~70% for all groups
- **Poorly calibrated:** Probabilities misleading for some groups → incorrect treatment decisions

### 5. Comprehensive Fairness Report

**Generate all metrics at once:**

```python
from basics_cdss.clinical_metrics import fairness_report
from basics_cdss.visualization import plot_fairness_radar

report = fairness_report(
    y_true, y_pred, y_pred_proba,
    protected_attribute=race,
    privileged_group='White',
    threshold=0.1
)

# Overall assessment
if report.overall_fair:
    print("✓ Model passes all fairness criteria")
else:
    print(f"✗ Failed criteria: {report.failed_criteria}")

# Detailed results
print(f"Demographic Parity: {'✓' if report.demographic_parity.is_fair else '✗'}")
print(f"Equalized Odds: {'✓' if report.equalized_odds.is_fair else '✗'}")
print(f"Equal Opportunity: {'✓' if report.equal_opportunity.is_fair else '✗'}")
print(f"Calibration: {'✓' if report.calibration.is_calibrated else '✗'}")

# Radar chart
plot_fairness_radar(report, save_path='fairness_radar.pdf')
```

---

## Conformal Prediction

### What is Conformal Prediction?

**Conformal prediction** provides **statistically rigorous** uncertainty quantification:
- **Distribution-free:** No assumptions about data distribution
- **Guaranteed coverage:** Mathematical guarantee that P(Y ∈ C(X)) ≥ 1 - α
- **Model-agnostic:** Works with any ML model

### Key Concepts

**Prediction Set:** Instead of predicting a single class, predict a **set** of possible classes:
- **Singleton set:** {Disease A} → High confidence
- **Large set:** {Disease A, Disease B, Disease C} → High uncertainty
- **Empty set:** (rare) Model very uncertain, defer to expert

**Coverage Guarantee:**
```
P(Y ∈ C(X)) ≥ 1 - α
```
- α = 0.1 → **90% coverage guarantee**
- Holds for **any** data distribution
- Finite-sample guarantee (not just asymptotic)

### 1. Split Conformal Classification

**Algorithm:**
1. Split data: Train / Calibration / Test
2. Train model on training set
3. Compute nonconformity scores on calibration set
4. Find (1-α) quantile of scores → **threshold**
5. For test samples, include all labels with score ≤ threshold

**Code Example:**

```python
from basics_cdss.clinical_metrics import split_conformal_classification
from basics_cdss.visualization import plot_prediction_set_sizes

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Conformal prediction with 90% coverage guarantee
conf_result = split_conformal_classification(
    model,
    X_train, y_train,
    X_cal, y_cal,
    X_test,
    alpha=0.1  # 90% coverage
)

# Analyze results
print(f"Target Coverage: {conf_result.target_coverage:.1%}")
print(f"Average Set Size: {conf_result.efficiency:.2f}")
print(f"Singleton Sets: {(conf_result.set_sizes == 1).sum()} / {len(conf_result.set_sizes)}")

# Visualize
plot_prediction_set_sizes(conf_result, save_path='prediction_sets.pdf')

# Interpret specific predictions
for i in range(5):
    pred_set = conf_result.prediction_sets[i]
    if len(pred_set) == 1:
        print(f"Sample {i}: High confidence → {pred_set}")
    else:
        print(f"Sample {i}: Uncertain → Consider: {pred_set}")
```

**Clinical Use Case:**
```python
# Example output
Sample 0: High confidence → ['No Disease']
Sample 1: High confidence → ['Pneumonia']
Sample 2: Uncertain → Consider: ['Pneumonia', 'COVID-19', 'Flu']
Sample 3: Uncertain → Consider: ['Heart Failure', 'Pulmonary Edema']
Sample 4: High confidence → ['No Disease']
```

**Clinical Action:**
- **Singleton set:** Proceed with diagnosis
- **Multiple classes:** Order additional tests to disambiguate
- **Large set:** Consult specialist or escalate to senior clinician

### 2. Conformal Regression

**For continuous outcomes** (e.g., survival time, biomarker levels):

```python
from basics_cdss.clinical_metrics import split_conformal_regression
from basics_cdss.visualization import plot_conformal_intervals

# Train regression model
model = RandomForestRegressor()

# Conformal prediction intervals
conf_interval = split_conformal_regression(
    model,
    X_train, y_train,
    X_cal, y_cal,
    X_test,
    alpha=0.1
)

# Visualize intervals
plot_conformal_intervals(
    conf_interval,
    y_true=y_test,  # Optional: show true values
    max_samples=50,
    save_path='conformal_intervals.pdf'
)

print(f"Average Interval Width: {conf_interval.average_width:.3f}")

# Use intervals for clinical decisions
for i in range(5):
    lower = conf_interval.lower_bounds[i]
    upper = conf_interval.upper_bounds[i]
    pred = conf_interval.point_predictions[i]

    print(f"Patient {i}: Predicted = {pred:.2f}, 90% CI = [{lower:.2f}, {upper:.2f}]")

    if upper - lower > 10:  # Wide interval
        print(f"  → High uncertainty, consider more tests")
```

### 3. Adaptive Conformal Prediction

**Idea:** Adjust prediction set size based on difficulty:
- **Easy samples:** Smaller sets (higher confidence)
- **Hard samples:** Larger sets (acknowledge uncertainty)

**Benefits:**
- **Better efficiency:** Smaller average set size
- **Better informativeness:** Uncertainty reflects sample difficulty

**Code Example:**

```python
from basics_cdss.clinical_metrics import adaptive_conformal_classification
from basics_cdss.visualization import plot_adaptive_efficiency_3d

adaptive_result = adaptive_conformal_classification(
    model,
    X_train, y_train,
    X_cal, y_cal,
    X_test,
    alpha=0.1
)

print(f"Efficiency Gain: {adaptive_result.efficiency_gain:.1%}")
print(f"Average Set Size: {adaptive_result.set_sizes.mean():.2f}")

# 3D visualization: difficulty vs set size
plot_adaptive_efficiency_3d(adaptive_result, save_path='adaptive_3d.pdf')
```

### 4. Risk Control (Learn Then Test)

**Problem:** Control a specific risk metric (e.g., false negative rate ≤ 5%)

**Solution:** Calibrate threshold to ensure risk ≤ target with high probability

**Code Example:**

```python
from basics_cdss.clinical_metrics import risk_control_conformal

# Define risk function (e.g., false negative rate)
def fnr_risk(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return fn / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0

# Calibrate threshold to control FNR ≤ 5%
result = risk_control_conformal(
    model, X_cal, y_cal, X_test,
    risk_function=fnr_risk,
    target_risk=0.05
)

print(f"Calibrated Threshold: {result.threshold:.3f}")
print(f"Empirical FNR: {result.empirical_risk:.3f}")
print(f"Risk Controlled: {result.risk_controlled}")
print(f"Rejection Rate: {result.rejection_rate:.1%}")
```

**Clinical Use Case:** Ensure no more than 5% of cancer cases are missed, even if it means higher false positive rate or some rejections (defer to expert).

---

## Quick Start Examples

### Example 1: Complete Clinical Utility Assessment

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from basics_cdss.clinical_metrics import (
    decision_curve_analysis,
    calculate_nnt,
    clinical_impact_analysis
)
from basics_cdss.visualization import (
    plot_decision_curve,
    plot_nnt_comparison,
    plot_clinical_impact
)

# Load your data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 1. Decision Curve Analysis
dca = decision_curve_analysis(y_test, y_pred_proba)
plot_decision_curve(dca, save_path='dca.pdf')

if dca.threshold_range[0] < dca.threshold_range[1]:
    print(f"✓ Model clinically useful for thresholds {dca.threshold_range}")
else:
    print("✗ Model not clinically useful")

# 2. NNT
nnt = calculate_nnt(y_test, y_pred)
if nnt.nnt < 10:
    print(f"✓ Excellent NNT: {nnt.nnt:.1f}")
elif nnt.nnt < 20:
    print(f"○ Moderate NNT: {nnt.nnt:.1f}")
else:
    print(f"✗ High NNT: {nnt.nnt:.1f} - reconsider deployment")

# 3. Clinical Impact
impact = clinical_impact_analysis(y_test, y_pred_proba, threshold=0.3)
plot_clinical_impact(impact, save_path='impact.pdf')

print(f"Deploy: Classify {impact.percent_high_risk:.1f}% as high-risk")
print(f"PPV: {impact.ppv:.3f} (precision of high-risk calls)")
```

### Example 2: Comprehensive Fairness Audit

```python
from basics_cdss.clinical_metrics import fairness_report
from basics_cdss.visualization import plot_fairness_radar

# Assuming you have protected attributes
race = df['race'].values  # e.g., ['White', 'Black', 'Asian', 'Hispanic']

# Generate comprehensive report
report = fairness_report(
    y_test, y_pred, y_pred_proba,
    protected_attribute=race,
    privileged_group='White',
    threshold=0.1
)

# Visualize
plot_fairness_radar(report, save_path='fairness_audit.pdf')

# Assess and report
if report.overall_fair:
    print("✓ Model passes all fairness criteria")
    print("Ready for ethical AI approval")
else:
    print(f"✗ Failed: {report.failed_criteria}")
    print("\nDetailed Results:")
    print(f"  Demographic Parity: {report.demographic_parity.parity_difference:.3f}")
    print(f"  Equalized Odds (TPR): {report.equalized_odds.tpr_difference:.3f}")
    print(f"  Equalized Odds (FPR): {report.equalized_odds.fpr_difference:.3f}")
    print(f"  Calibration Error: {report.calibration.max_calibration_error:.3f}")

    print("\nRecommended Actions:")
    if 'Demographic Parity' in report.failed_criteria:
        print("  - Investigate why positive rates differ across groups")
        print("  - Consider group-specific thresholds")
    if 'Equalized Odds' in report.failed_criteria:
        print("  - Model has differential performance across groups")
        print("  - Retrain with fairness constraints or balanced sampling")
    if 'Calibration' in report.failed_criteria:
        print("  - Recalibrate probabilities per group")
        print("  - Use isotonic regression or Platt scaling by group")
```

### Example 3: Safe Deployment with Conformal Prediction

```python
from basics_cdss.clinical_metrics import (
    split_conformal_classification,
    risk_control_conformal
)
from basics_cdss.visualization import (
    plot_prediction_set_sizes,
    plot_coverage_vs_alpha
)

# Split data: train, calibration, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Standard conformal prediction
conf_90 = split_conformal_classification(
    model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1
)

print(f"90% Coverage Guarantee")
print(f"  Average Set Size: {conf_90.efficiency:.2f}")
print(f"  Singleton Sets: {(conf_90.set_sizes == 1).sum()} / {len(conf_90.set_sizes)}")

plot_prediction_set_sizes(conf_90, save_path='prediction_sets_90.pdf')

# Control false negative rate
def fnr(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return fn / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0

fnr_control = risk_control_conformal(
    model, X_cal, y_cal, X_test,
    risk_function=fnr,
    target_risk=0.05  # Max 5% FNR
)

print(f"\nFalse Negative Rate Control (≤ 5%)")
print(f"  Calibrated Threshold: {fnr_control.threshold:.3f}")
print(f"  Empirical FNR: {fnr_control.empirical_risk:.3f}")
print(f"  Samples Rejected: {fnr_control.n_rejected} ({fnr_control.rejection_rate:.1%})")

if fnr_control.risk_controlled:
    print("  ✓ Safe to deploy with FNR guarantee")
else:
    print("  ✗ Cannot guarantee FNR ≤ 5% - increase model capacity or collect more data")
```

---

## Visualization Gallery

All visualization functions produce **publication-ready** figures (300 DPI, Times New Roman, colorblind-friendly):

### Clinical Utility

| Function | Description | Figure Type |
|----------|-------------|-------------|
| `plot_decision_curve()` | Net benefit vs threshold | 2D line plot |
| `plot_standardized_net_benefit()` | Net benefit per 100 patients | 2D bar chart |
| `plot_nnt_comparison()` | Compare NNT across models | 2D horizontal bar |
| `plot_clinical_impact()` | Classification breakdown + metrics | 2D dual subplot |
| `plot_clinical_impact_3d()` | Impact metrics vs threshold | 3D surface plot |

### Fairness

| Function | Description | Figure Type |
|----------|-------------|-------------|
| `plot_demographic_parity()` | Positive rates by group | 2D bar chart |
| `plot_equalized_odds()` | TPR/FPR by group | 2D grouped bar |
| `plot_disparate_impact()` | DI ratios with 80% rule | 2D horizontal bar |
| `plot_calibration_by_group()` | Calibration curves per group | 2D line plot |
| `plot_fairness_radar()` | All fairness metrics | 2D radar/spider chart |

### Conformal Prediction

| Function | Description | Figure Type |
|----------|-------------|-------------|
| `plot_prediction_set_sizes()` | Histogram of set sizes | 2D histogram |
| `plot_conformal_intervals()` | Prediction intervals for regression | 2D error bars |
| `plot_coverage_vs_alpha()` | Empirical vs target coverage | 2D line plot |
| `plot_adaptive_efficiency_3d()` | Difficulty vs set size | 3D scatter plot |

---

## FDA Compliance Guidelines

### Submitting Medical AI for FDA 510(k)

**Required Clinical Evidence:**

1. **Clinical Utility** (from this module):
   - ✅ Decision Curve Analysis showing net benefit
   - ✅ NNT calculation demonstrating effectiveness
   - ✅ Clinical impact assessment at proposed threshold

2. **Fairness/Equity** (from this module):
   - ✅ Performance metrics stratified by demographics
   - ✅ Equalized odds analysis
   - ✅ Calibration assessment per subgroup

3. **Uncertainty Quantification** (from this module):
   - ✅ Conformal prediction intervals
   - ✅ Risk control demonstration (e.g., FNR ≤ 5%)

**FDA Guidance References:**
- [Clinical Decision Support Software (2022)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
- [Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)

---

## API Reference

See module docstrings for complete API details:

```python
# Clinical Utility
help(calculate_net_benefit)
help(decision_curve_analysis)
help(calculate_nnt)
help(clinical_impact_analysis)

# Fairness
help(demographic_parity)
help(equalized_odds)
help(disparate_impact)
help(calibration_by_group)
help(fairness_report)

# Conformal Prediction
help(split_conformal_classification)
help(split_conformal_regression)
help(adaptive_conformal_classification)
help(risk_control_conformal)
```

---

## Clinical Interpretation

### Decision-Making Framework

```
┌─────────────────────────────────────────────────────────┐
│                Clinical Deployment Decision Tree         │
└─────────────────────────────────────────────────────────┘

1. Clinical Utility Assessment
   └─ Is Net Benefit > 0 for useful threshold range?
      ├─ YES: Proceed to Step 2
      └─ NO:  ✗ Do not deploy (model harmful)

2. Effectiveness Assessment
   └─ Is NNT acceptable (<20)?
      ├─ YES: Proceed to Step 3
      └─ NO:  ○ Deploy only if cost-effective

3. Fairness Audit
   └─ Does model pass all fairness criteria?
      ├─ YES: Proceed to Step 4
      └─ NO:  ✗ Address bias before deployment

4. Uncertainty Quantification
   └─ Can we guarantee acceptable risk (e.g., FNR ≤ 5%)?
      ├─ YES: ✓ Safe to deploy
      └─ NO:  ✗ Need better model or defer uncertain cases

```

### Threshold Selection Guide

| Clinical Context | Recommended Threshold | Rationale |
|-----------------|----------------------|-----------|
| **Screening** (low-risk population) | Low (0.1-0.3) | Maximize sensitivity, tolerate FPs |
| **Diagnosis** (symptomatic patients) | Medium (0.3-0.5) | Balance sensitivity/specificity |
| **High-risk intervention** (e.g., surgery) | High (0.5-0.8) | Minimize FPs, accept some FNs |

**Use Decision Curve Analysis to find optimal threshold for your clinical context.**

---

## References

### Clinical Utility
1. Vickers AJ, Elkin EB. (2006). Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
2. Vickers AJ, Van Calster B, Steyerberg EW. (2016). Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests. *BMJ*, 352:i6.
3. Laupacis A, Sackett DL, Roberts RS. (1988). An assessment of clinically useful measures of the consequences of treatment. *New England Journal of Medicine*, 318(26), 1728-1733.

### Fairness
4. Hardt M, Price E, Srebro N. (2016). Equality of opportunity in supervised learning. *NIPS*.
5. Chouldechova A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big data*, 5(2), 153-163.
6. Obermeyer Z, et al. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.

### Conformal Prediction
7. Vovk V, Gammerman A, Shafer G. (2005). *Algorithmic Learning in a Random World*. Springer.
8. Angelopoulos AN, Bates S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*.
9. Angelopoulos AN, et al. (2022). Learn then test: Calibrating predictive algorithms to achieve risk control. *arXiv:2110.01052*.

---

**For questions or support, see:** [BASICS-CDSS GitHub Issues](https://github.com/yourusername/BASICS-CDSS/issues)

**License:** MIT | **Version:** 2.1.0 | **Last Updated:** 2025-01-25
