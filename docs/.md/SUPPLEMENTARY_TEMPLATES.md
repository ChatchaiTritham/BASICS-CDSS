# Supplementary Materials Templates

**Ready-to-use templates for supplementary files**

Most high-impact journals require supplementary materials. This document provides templates for common supplementary content when using BASICS-CDSS Phase 1 metrics.

---

## Table of Contents

1. [Supplementary Methods](#supplementary-methods)
2. [Supplementary Tables](#supplementary-tables)
3. [Supplementary Figures](#supplementary-figures)
4. [Supplementary Results](#supplementary-results)
5. [Code/Data Availability Statements](#code-availability)
6. [Checklist for AI/ML Papers](#checklists)

---

## Supplementary Methods

### Supplementary Methods 1: Detailed Clinical Utility Calculations

```markdown
## SM1. Clinical Utility Metrics: Detailed Methodology

### SM1.1 Decision Curve Analysis

Decision curve analysis was performed as described by Vickers and Elkin [1]. For each
probability threshold pt ∈ {0.01, 0.02, ..., 0.99}, we computed net benefit as:

$$\text{NB}(p_t) = \frac{TP(p_t)}{N} - \frac{FP(p_t)}{N} \times \frac{p_t}{1-p_t}$$

where:
- TP(pt) = number of true positives at threshold pt
- FP(pt) = number of false positives at threshold pt
- N = total sample size
- pt/(1-pt) = harm-to-benefit ratio (odds at threshold)

**Baseline Strategies:**

Treat All: $\text{NB}_{\text{all}}(p_t) = \text{prevalence} - (1-\text{prevalence}) \times \frac{p_t}{1-p_t}$

Treat None: $\text{NB}_{\text{none}}(p_t) = 0$

**Standardized Net Benefit:**
We report net benefit per 100 patients: $\text{SNB}(p_t) = 100 \times \text{NB}(p_t)$

**Interventions Avoided:**
Relative to treat-all, the proportion of interventions avoided is:

$$\text{IA}(p_t) = \frac{N - N_{\text{treated}}(p_t)}{N}$$

where $N_{\text{treated}}(p_t)$ is the number classified as high-risk at threshold pt.

### SM1.2 Number Needed to Treat

NNT was calculated as the reciprocal of absolute risk reduction (ARR):

$$\text{NNT} = \frac{1}{\text{ARR}} = \frac{1}{|\text{CER} - \text{EER}|}$$

where:
- CER (Control Event Rate) = event rate in predicted low-risk group
- EER (Experimental Event Rate) = event rate in predicted high-risk group

**Confidence Intervals:**
95% confidence intervals were computed using the Newcombe method for difference in
proportions [2]:

$$\text{SE(ARR)} = \sqrt{\text{SE}^2_{\text{CER}} + \text{SE}^2_{\text{EER}}}$$

where:
$$\text{SE}_{\text{CER}} = \sqrt{\frac{\text{CER} \times (1-\text{CER})}{n_{\text{control}}}}$$
$$\text{SE}_{\text{EER}} = \sqrt{\frac{\text{EER} \times (1-\text{EER})}{n_{\text{experimental}}}}$$

95% CI for NNT: $\left[\frac{1}{\text{ARR} + 1.96 \times \text{SE}}, \frac{1}{\text{ARR} - 1.96 \times \text{SE}}\right]$

Note: Order reverses due to reciprocal transformation.

### SM1.3 Clinical Impact Assessment

For each threshold pt, we computed:

**Positive Predictive Value (PPV):**
$$\text{PPV}(p_t) = \frac{TP(p_t)}{TP(p_t) + FP(p_t)}$$

**Negative Predictive Value (NPV):**
$$\text{NPV}(p_t) = \frac{TN(p_t)}{TN(p_t) + FN(p_t)}$$

**Number Needed to Screen (NNS):**
$$\text{NNS}(p_t) = \frac{N_{\text{screened}}}{TP(p_t)} = \frac{TP(p_t) + FP(p_t)}{TP(p_t)}$$

**References:**
[1] Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating
    prediction models. Med Decis Making. 2006;26(6):565-574.
[2] Newcombe RG. Interval estimation for the difference between independent proportions:
    comparison of eleven methods. Stat Med. 1998;17(8):873-890.
```

---

### Supplementary Methods 2: Fairness Metrics Detailed Definitions

```markdown
## SM2. Fairness Metrics: Mathematical Definitions and Rationale

We evaluated fairness using five complementary metrics, each capturing different
equity concepts [1,2]. Let A denote protected attribute, Y true label, and Ŷ predicted
label.

### SM2.1 Demographic Parity (Independence)

**Definition:**
$$P(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = b) \quad \forall a, b$$

**Metric:** Maximum difference in positive prediction rates across groups

**Acceptance Criterion:** Δ ≤ 0.10 (10%)

**Rationale:**
Demographic parity ensures equal selection rates across groups. Appropriate when:
- Selection for scarce resources
- Procedural fairness is paramount
- True outcome rates do not legitimately differ across groups

**Limitations:**
May be inappropriate when base rates differ (e.g., disease prevalence varies by age).
Does not guarantee equal error rates.

### SM2.2 Equalized Odds (Separation)

**Definition:**
$$P(\hat{Y} = 1 | Y = y, A = a) = P(\hat{Y} = 1 | Y = y, A = b) \quad \forall y \in \{0,1\}$$

Equivalently: TPR and FPR must be equal across groups.

**Metric:**
- TPR difference: $\max_{a,b} |TPR_a - TPR_b|$
- FPR difference: $\max_{a,b} |FPR_a - FPR_b|$

**Acceptance Criterion:** Both differences ≤ 0.10

**Rationale:**
Ensures equal error rates (both false positives and false negatives) across groups.
Generally preferred for medical AI because it:
1. Guarantees equal quality of care
2. Respects legitimate base rate differences
3. Has clear clinical interpretation (equal sensitivity and specificity)

### SM2.3 Equal Opportunity (Sufficiency for Positive Class)

**Definition:**
$$P(\hat{Y} = 1 | Y = 1, A = a) = P(\hat{Y} = 1 | Y = 1, A = b)$$

Equivalently: TPR (sensitivity) must be equal across groups.

**Metric:** TPR difference

**Acceptance Criterion:** Δ ≤ 0.10

**Rationale:**
Ensures equal ability to detect positive cases. Critical for medical AI to avoid
differential miss rates across demographics.

**Note:** Equal opportunity is a relaxation of equalized odds (only TPR, not FPR).

### SM2.4 Disparate Impact (80% Rule)

**Definition:**
$$\text{DI} = \frac{P(\hat{Y} = 1 | A = \text{unprivileged})}{P(\hat{Y} = 1 | A = \text{privileged})}$$

**Acceptance Criterion:** 0.80 ≤ DI ≤ 1.25

**Rationale:**
The "80% rule" (four-fifths rule) from employment discrimination law (UGESP, 1978).
A selection rate for any group less than 80% of the highest rate indicates potential
discrimination requiring investigation.

**Interpretation:**
- DI < 0.80: Unprivileged group under-selected (potential discrimination)
- DI > 1.25: Unprivileged group over-selected
- 0.80 ≤ DI ≤ 1.25: Acceptable range

### SM2.5 Calibration Fairness

**Definition:**
$$P(Y = 1 | \hat{p}(X) = s, A = a) = s \quad \forall s, a$$

Predicted probabilities must match observed frequencies within each group.

**Metric:** Expected Calibration Error (ECE) per group

$$\text{ECE}_a = \frac{1}{B} \sum_{b=1}^{B} \left| \frac{1}{|C_b^a|} \sum_{i \in C_b^a} y_i - \frac{1}{|C_b^a|} \sum_{i \in C_b^a} \hat{p}(x_i) \right|$$

where B is number of bins, $C_b^a$ is set of samples in bin b for group a.

**Acceptance Criterion:** $\max_a \text{ECE}_a \leq 0.10$

**Rationale:**
Poor calibration means predicted probabilities are systematically biased for a group,
leading to incorrect treatment decisions. Calibration is essential for risk-based
decision-making.

### SM2.6 Fairness Metric Selection

**No single fairness metric is universally appropriate.** Different metrics capture
different equity concepts and may be mutually incompatible [3]. We report all five
metrics to provide comprehensive fairness assessment, acknowledging trade-offs.

**Clinical Context Determines Appropriate Metrics:**
- Diagnostic models: Equalized odds (equal error rates)
- Screening programs: Demographic parity or equal opportunity
- Risk stratification: Calibration fairness (accurate probabilities)
- Resource allocation: Depends on scarcity and consequences

**References:**
[1] Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning. NIPS. 2016.
[2] Chouldechova A. Fair prediction with disparate impact. Big data. 2017;5(2):153-163.
[3] Chouldechova A, Roth A. The frontiers of fairness in machine learning. arXiv:1810.08810. 2018.
```

---

### Supplementary Methods 3: Conformal Prediction Algorithm

```markdown
## SM3. Conformal Prediction: Algorithm and Theory

### SM3.1 Split Conformal Prediction for Classification

**Input:**
- Training set: $(X_1, Y_1), ..., (X_n, Y_n)$
- Calibration set: $(X_{n+1}, Y_{n+1}), ..., (X_{n+m}, Y_{n+m})$
- Test sample: $X_{\text{test}}$
- Miscoverage rate: $\alpha$ (e.g., 0.10 for 90% coverage)
- Base model: $f$ (e.g., Random Forest)

**Algorithm:**

1. **Train model:**
   Train $f$ on training set to obtain $\hat{f}$

2. **Compute nonconformity scores on calibration set:**
   For each calibration sample $i$:
   $$s_i = 1 - \hat{p}_i(Y_i)$$
   where $\hat{p}_i(Y_i)$ is model's predicted probability for true label

3. **Compute adjusted quantile:**
   $$q = \text{Quantile}_{\lceil (1-\alpha)(1 + 1/m) \rceil / m} (\{s_1, ..., s_m\})$$

   The adjustment $(1 + 1/m)$ ensures finite-sample validity.

4. **Construct prediction set for test sample:**
   $$C(X_{\text{test}}) = \{y : 1 - \hat{p}_{\text{test}}(y) \leq q\}$$

   Include all labels with nonconformity score ≤ q.

**Output:** Prediction set $C(X_{\text{test}})$ with coverage guarantee

### SM3.2 Theoretical Guarantee

**Theorem (Vovk et al., 2005):**
Under the exchangeability assumption (calibration and test samples are exchangeable),

$$P(Y_{\text{test}} \in C(X_{\text{test}})) \geq 1 - \alpha$$

**Key Properties:**
1. **Distribution-free:** No parametric assumptions about data distribution
2. **Finite-sample:** Guarantee holds for any sample size (not asymptotic)
3. **Model-agnostic:** Works with any base model $f$

**Exchangeability Assumption:**
The guarantee requires that calibration and test samples are drawn i.i.d. from the
same distribution. In non-stationary environments (e.g., concept drift), this may
be violated. Online conformal methods exist for time-varying distributions [1].

### SM3.3 Adaptive Conformal Prediction

To improve efficiency (reduce average set size) while maintaining coverage, we
implemented adaptive conformal prediction based on sample difficulty [2].

**Difficulty Score:**
We measured sample difficulty using entropy of predicted distribution:
$$d(X) = -\sum_{k=1}^K \hat{p}(y_k|X) \log \hat{p}(y_k|X)$$

High entropy indicates uncertain/difficult samples.

**Adaptive Algorithm:**
1. Stratify calibration set into difficulty bins (tertiles)
2. Compute separate quantile threshold for each bin
3. For test sample, assign to difficulty bin and use corresponding threshold

**Result:** Easy samples get smaller prediction sets, difficult samples get larger sets.

**Efficiency Gain:**
In our experiments, adaptive conformal achieved [X]% reduction in average set size
while maintaining [Y]% coverage (target: 90%).

### SM3.4 Implementation Details

**Software:** BASICS-CDSS v2.1.0 (Python 3.10+)

**Code:**
```python
from basics_cdss.clinical_metrics import split_conformal_classification

result = split_conformal_classification(
    model=model,
    X_train=X_train, y_train=y_train,
    X_cal=X_cal, y_cal=y_cal,
    X_test=X_test,
    alpha=0.10,
    score_function=None  # Default: 1 - P(Y=y)
)
```

**Hyperparameters:**
- Split proportions: 60% train, 20% calibration, 20% test
- Miscoverage rate: α = 0.10 (90% coverage target)
- Random seed: 42 (for reproducibility)

**References:**
[1] Gibbs I, Candès E. Adaptive conformal inference under distribution shift.
    NeurIPS. 2021.
[2] Romano Y, et al. Classification with valid and adaptive coverage. NeurIPS. 2020.
```

---

## Supplementary Tables

### Supplementary Table 1: Complete Fairness Metrics

```markdown
## Table S1. Comprehensive Fairness Assessment Across Protected Attributes

| Protected Attribute | Subgroup | n | Positive Rate | TPR | FPR | PPV | NPV | ECE |
|-------------------|----------|---|---------------|-----|-----|-----|-----|-----|
| **Age Group** |
| | <40 years | 150 | 0.25 | 0.85 | 0.12 | 0.72 | 0.91 | 0.08 |
| | 40-60 years | 250 | 0.32 | 0.88 | 0.10 | 0.78 | 0.93 | 0.06 |
| | >60 years | 100 | 0.38 | 0.82 | 0.15 | 0.69 | 0.89 | 0.09 |
| | **Δ_max** | - | **0.13** | **0.06** | **0.05** | - | - | **0.03** |
| | **Status** | - | FAIL | PASS | PASS | - | - | PASS |
| **Sex** |
| | Male | 260 | 0.30 | 0.86 | 0.11 | 0.75 | 0.92 | 0.07 |
| | Female | 240 | 0.31 | 0.84 | 0.13 | 0.73 | 0.90 | 0.08 |
| | **Δ_max** | - | **0.01** | **0.02** | **0.02** | - | - | **0.01** |
| | **Status** | - | PASS | PASS | PASS | - | - | PASS |
| **Race/Ethnicity** |
| | White | 300 | 0.35 | 0.88 | 0.10 | 0.78 | 0.93 | 0.06 |
| | Black | 75 | 0.23 | 0.76 | 0.18 | 0.65 | 0.87 | 0.12 |
| | Asian | 75 | 0.30 | 0.85 | 0.12 | 0.73 | 0.91 | 0.08 |
| | Hispanic | 50 | 0.28 | 0.82 | 0.14 | 0.70 | 0.89 | 0.09 |
| | **Δ_max** | - | **0.12** | **0.12** | **0.08** | - | - | **0.06** |
| | **Status** | - | FAIL | FAIL | PASS | - | - | PASS |

**Abbreviations:**
n = sample size; TPR = True Positive Rate (sensitivity); FPR = False Positive Rate;
PPV = Positive Predictive Value; NPV = Negative Predictive Value; ECE = Expected
Calibration Error; Δ_max = Maximum difference across subgroups within attribute

**Acceptance Thresholds:**
Positive Rate Δ ≤ 0.10, TPR Δ ≤ 0.10, FPR Δ ≤ 0.10, ECE ≤ 0.10

**Interpretation:**
Model passed fairness criteria for Sex but exhibited disparities in Age Group (demographic
parity violation) and Race/Ethnicity (demographic parity and equalized odds violations).
```

---

### Supplementary Table 2: Net Benefit at Multiple Thresholds

```markdown
## Table S2. Net Benefit and Clinical Impact Across Probability Thresholds

| Threshold | Sensitivity | Specificity | PPV | NPV | NB_model | NB_all | Status | % High-Risk | NNS |
|-----------|-------------|-------------|-----|-----|----------|--------|--------|-------------|-----|
| 0.10 | 0.95 | 0.45 | 0.42 | 0.96 | 0.185 | 0.118 | Useful | 68% | 2.4 |
| 0.20 | 0.90 | 0.68 | 0.58 | 0.93 | 0.231 | 0.098 | Useful | 48% | 2.1 |
| 0.30 | 0.85 | 0.82 | 0.72 | 0.91 | 0.252 | 0.070 | **Optimal** | 35% | 2.0 |
| 0.40 | 0.78 | 0.90 | 0.82 | 0.88 | 0.241 | 0.035 | Useful | 25% | 2.1 |
| 0.50 | 0.70 | 0.94 | 0.88 | 0.85 | 0.215 | 0.000 | Useful | 19% | 2.4 |
| 0.60 | 0.60 | 0.97 | 0.92 | 0.82 | 0.173 | -0.042 | Useful | 15% | 2.9 |
| 0.70 | 0.48 | 0.99 | 0.96 | 0.79 | 0.118 | -0.098 | Useful | 11% | 3.8 |
| 0.80 | 0.32 | 0.995 | 0.98 | 0.75 | 0.062 | -0.196 | Marginal | 7% | 6.2 |
| 0.90 | 0.15 | 0.999 | 0.99 | 0.72 | 0.018 | -0.441 | Marginal | 3% | 14.3 |

**Abbreviations:**
PPV = Positive Predictive Value; NPV = Negative Predictive Value; NB = Net Benefit;
NNS = Number Needed to Screen; Status: Useful = NB_model > max(NB_all, 0)

**Interpretation:**
Maximum net benefit (0.252) achieved at threshold 0.30, identifying this as the optimal
operating point for our clinical context (screening/triage). Model provides clinical
utility for thresholds 0.10-0.70.
```

---

## Supplementary Figures

### Supplementary Figure Captions

```markdown
## Figure S1. Decision Curves Stratified by Demographic Group

Decision curve analysis showing net benefit across probability thresholds, stratified
by (A) age group, (B) sex, and (C) race/ethnicity. Solid lines represent the prediction
model, dashed lines represent "treat all" strategy, and dotted lines represent "treat
none" strategy. All demographic subgroups achieved positive net benefit within threshold
range 0.15-0.65, though optimal thresholds varied slightly by group ([detailed findings]).

## Figure S2. Calibration Curves by Protected Attribute

Calibration plots showing predicted probability (x-axis) versus observed frequency
(y-axis) for (A) age groups, (B) sex, and (C) race/ethnicity. Diagonal line represents
perfect calibration. All groups exhibited good calibration (Expected Calibration Error
< 0.10), with [describe any systematic deviations].

## Figure S3. Conformal Prediction Set Size Distribution by Sample Difficulty

Distribution of prediction set sizes stratified by sample difficulty (entropy tertiles).
(A) Easy samples (low entropy) predominantly yielded singleton sets (85%), indicating
high confidence. (B) Medium difficulty samples yielded average set size 2.1. (C)
Difficult samples (high entropy) yielded average set size 3.8, appropriately reflecting
uncertainty.

## Figure S4. Clinical Impact Heatmap Across Thresholds and Subgroups

Heatmap showing clinical impact metrics (PPV, NPV, Number Needed to Screen) across
probability thresholds (0.1-0.9) and demographic subgroups. Darker colors indicate
better performance. Optimal threshold of 0.30 provides balanced performance across
all subgroups.

## Figure S5. Disparate Impact Ratios with 80% Rule

Disparate impact ratios comparing each demographic subgroup to reference group (White
patients), with 80% rule bounds (horizontal dashed lines at 0.80 and 1.25). Ratios
within bounds indicate acceptable disparate impact. [Describe findings: which groups
pass/fail 80% rule].
```

---

## Supplementary Results

### Supplementary Results 1: Subgroup Analysis

```markdown
## SR1. Performance Stratified by Clinical Subgroups

Beyond protected attributes, we evaluated model performance across clinically relevant
subgroups:

**By Comorbidity Burden:**
- 0-1 comorbidities (n=200): AUC 0.88, NB 0.24 at threshold 0.30
- 2-3 comorbidities (n=180): AUC 0.85, NB 0.26 at threshold 0.30
- ≥4 comorbidities (n=120): AUC 0.81, NB 0.22 at threshold 0.30

**By Disease Severity:**
- Low severity (n=250): AUC 0.86, NB 0.23
- Moderate severity (n=150): AUC 0.84, NB 0.25
- High severity (n=100): AUC 0.82, NB 0.24

**Interpretation:**
Model performance was relatively consistent across clinical subgroups, with AUC
varying by ≤0.07 and net benefit by ≤0.04. This suggests the model generalizes well
across patient heterogeneity.

## SR2. Sensitivity Analyses

### SR2.1 Alternative Thresholds

We repeated fairness assessment at thresholds 0.20, 0.30, and 0.40:

| Threshold | Demographic Parity | Equalized Odds (TPR) | Equalized Odds (FPR) |
|-----------|-------------------|---------------------|---------------------|
| 0.20 | PASS (Δ=0.08) | PASS (Δ=0.09) | PASS (Δ=0.07) |
| 0.30 | FAIL (Δ=0.12) | FAIL (Δ=0.12) | PASS (Δ=0.08) |
| 0.40 | PASS (Δ=0.09) | PASS (Δ=0.10) | PASS (Δ=0.06) |

Fairness violations were most pronounced at threshold 0.30, our proposed operating point.

### SR2.2 Alternative Fairness Thresholds

Using more stringent fairness threshold (Δ ≤ 0.05):
- Demographic Parity: FAIL at all thresholds
- Equalized Odds: FAIL for TPR, PASS for FPR

Using more lenient fairness threshold (Δ ≤ 0.15):
- All metrics PASS

This sensitivity analysis highlights the importance of explicit fairness threshold
selection based on clinical context and stakeholder input.

## SR3. Missing Data Analysis

[Describe handling of missing data, missingness patterns by demographic group, and
sensitivity to imputation method]
```

---

## Code/Data Availability

### Code Availability Statement

```markdown
## Code Availability

All analyses were performed using BASICS-CDSS v2.1.0, an open-source Python framework
for comprehensive medical AI evaluation. The framework implements Phase 1 Clinical
Metrics including:

1. Clinical Utility Assessment (Decision Curve Analysis, Net Benefit, NNT)
2. Fairness Evaluation (5 complementary metrics)
3. Conformal Prediction (distribution-free uncertainty quantification)

**Public Repository:**
https://github.com/[YourUsername]/BASICS-CDSS

**Installation:**
```
pip install basics-cdss
```

**Documentation:**
Comprehensive guides available at:
- Clinical Metrics Guide: docs/CLINICAL_METRICS_GUIDE.md
- API Reference: docs/API_REFERENCE.md
- Tutorials: examples/

**Reproducibility:**
All figures in this manuscript can be reproduced using:
```
python examples/generate_clinical_metrics_figures.py --n-samples [N] --output-dir figures/
```

**Software Dependencies:**
- Python ≥3.10
- NumPy ≥1.21
- pandas ≥1.3
- scikit-learn ≥1.0
- scipy ≥1.7
- matplotlib ≥3.4
- shap ≥0.42 (for XAI methods)

**License:** MIT License (open source, commercial use allowed)

**Contact:** [Your email] for questions about code implementation
```

---

### Data Availability Statement

**Option 1 (Public Data):**
```markdown
## Data Availability

This study used publicly available data from [DATA SOURCE]. The dataset can be
accessed at [URL] following [REGISTRATION / APPROVAL PROCESS if applicable].

**Preprocessing Code:**
Data preprocessing scripts are available in the BASICS-CDSS repository:
`examples/data_preprocessing/[DATASET_NAME].py`
```

**Option 2 (Restricted Data - PHI):**
```markdown
## Data Availability

The dataset used in this study contains protected health information (PHI) and cannot
be publicly shared per HIPAA regulations and institutional IRB requirements. Qualified
researchers may request access through [INSTITUTION]'s data sharing agreement process:

**Data Request Process:**
1. Submit research proposal to [CONTACT]
2. Obtain IRB approval from requesting institution
3. Execute Data Use Agreement
4. Access de-identified data through secure enclave

**Synthetic Example Data:**
To facilitate methods replication, we provide synthetic example data (n=500) with
similar statistical properties to the original dataset:
`examples/synthetic_data/example_dataset.csv`

This synthetic data can be used to test BASICS-CDSS workflows but should not be used
for clinical validation.
```

**Option 3 (Simulated Data):**
```markdown
## Data Availability

This methodological study used simulated data to demonstrate the BASICS-CDSS framework.
The data generation code is available in:
`examples/generate_synthetic_clinical_data.py`

Researchers can generate custom synthetic datasets matching their clinical scenario
using:
```python
from basics_cdss.examples import generate_synthetic_data
X, y, protected_attrs = generate_synthetic_data(
    n_samples=500,
    n_features=15,
    prevalence=0.30,
    demographics=['age', 'sex', 'race']
)
```
```

---

## Checklists for AI/ML Papers

### CONSORT-AI Checklist (for RCTs)

```markdown
## CONSORT-AI Checklist

[If applicable - for randomized controlled trials involving AI intervention]

This study followed CONSORT-AI guidelines for transparent reporting of AI interventions
in clinical trials [1].

| Item | Recommendation | Page | Status |
|------|---------------|------|--------|
| 1a | AI intervention clearly described | 5 | ✓ |
| 1b | Algorithm version specified | 5 | ✓ |
| 6a | Training/validation/test split described | 8 | ✓ |
| 6b | Handling of missing data described | 9 | ✓ |
| 11a | Hyperparameter selection described | 7 | ✓ |
| 11b | Model performance metrics reported | 12-14 | ✓ |
| 12a | Fairness assessment across subgroups | 15-16 | ✓ |
| 12b | Uncertainty quantification provided | 17 | ✓ |
| ... | [Continue for all items] | | |

[1] Liu X, et al. Reporting guidelines for clinical trial reports for interventions
involving artificial intelligence. Nat Med. 2020;26(9):1364-1374.
```

---

### TRIPOD-AI Checklist (for Prediction Models)

```markdown
## TRIPOD-AI Checklist

This study followed TRIPOD-AI guidelines for transparent reporting of multivariable
prediction models involving AI [1].

| Section/Topic | Item | Checklist Item | Page |
|--------------|------|---------------|------|
| **Title** | 1 | Identify as prediction model study | 1 | ✓ |
| **Abstract** | 2 | Structured summary (TRIPOD-AI format) | 2 | ✓ |
| **Introduction** |
| Background | 3a | Medical context and rationale | 4-5 | ✓ |
| | 3b | Existing prediction models | 5 | ✓ |
| Objectives | 4 | Specify study objectives | 5 | ✓ |
| **Methods** |
| Source of data | 5a | Source and setting | 6 | ✓ |
| | 5b | Inclusion/exclusion criteria | 6 | ✓ |
| | 5c | Train/validation/test split | 7 | ✓ |
| Participants | 6a | Eligibility criteria | 6 | ✓ |
| Outcome | 7a | Outcome definition | 7 | ✓ |
| Predictors | 8a | Predictor definition | 7-8 | ✓ |
| Sample size | 9 | Sample size justification | 8 | ✓ |
| Missing data | 10 | Missing data handling | 8 | ✓ |
| **Model Development** |
| Algorithm | 11a | Model type and rationale | 9 | ✓ |
| | 11b | Hyperparameter selection | 9 | ✓ |
| | 11c | Training procedure | 9-10 | ✓ |
| **Model Evaluation** |
| Performance | 12a | Discrimination, calibration | 12 | ✓ |
| Clinical utility | 12b | Decision curve analysis | 13 | ✓ |
| Fairness | 12c | Fairness across subgroups | 15 | ✓ |
| Uncertainty | 12d | Uncertainty quantification | 17 | ✓ |
| **Results** |
| [Continue...] | | | |

[1] Collins GS, et al. TRIPOD-AI statement. BMJ. 2024 (in press).
```

---

### PROBAST Checklist (Risk of Bias Assessment)

```markdown
## PROBAST: Prediction Model Risk of Bias Assessment

**Domain 1: Participants**
- Risk of Bias: Low / High / Unclear
- Concerns about Applicability: Low / High / Unclear
- Justification: [Describe inclusion criteria, representativeness]

**Domain 2: Predictors**
- Risk of Bias: Low / High / Unclear
- Concerns about Applicability: Low / High / Unclear
- Justification: [Describe predictor availability, measurement]

**Domain 3: Outcome**
- Risk of Bias: Low / High / Unclear
- Concerns about Applicability: Low / High / Unclear
- Justification: [Describe outcome definition, blinding]

**Domain 4: Analysis**
- Risk of Bias: Low / High / Unclear
- Concerns about Applicability: Low / High / Unclear
- Justification: [Describe sample size, methods, validation]

**Overall Risk of Bias:** Low / High / Unclear
```

---

## Quick Checklist: What to Include in Supplementary Materials

**Minimum (Required for most journals):**
- [ ] Detailed statistical methods
- [ ] Complete fairness metrics table (Table S1)
- [ ] Code/data availability statements
- [ ] TRIPOD-AI or CONSORT-AI checklist (if applicable)

**Recommended (Strengthens paper):**
- [ ] Supplementary figures (calibration curves, stratified DCAs)
- [ ] Net benefit at multiple thresholds (Table S2)
- [ ] Subgroup analyses
- [ ] Sensitivity analyses

**Optional (For comprehensive reporting):**
- [ ] Complete hyperparameter search results
- [ ] Feature importance rankings
- [ ] Confusion matrices by subgroup
- [ ] Additional fairness metrics
- [ ] Model card (Google's format)

---

## File Naming Convention

**Recommended Structure:**
```
SupplementaryMaterials_[YourPaperName].pdf

Contents:
- Supplementary Methods 1-3
- Supplementary Tables S1-SX
- Supplementary Figures S1-SX
- Supplementary Results SR1-SRX
- Checklists (TRIPOD-AI, CONSORT-AI)
- Code/Data Availability Statements
```

**Separate Files (if required by journal):**
```
SupplementaryMethods.pdf
SupplementaryTables.xlsx
SupplementaryFigures.pdf
SupplementaryCode.zip
```

---

**All templates are ready to fill in with your specific results from BASICS-CDSS figure generation**

**For manuscript text, see MANUSCRIPT_UPDATES.md**
**For paper structures, see PAPER_TEMPLATES.md**
**For metric explanations, see METRIC_EXPLANATIONS_FOR_PAPERS.md**
