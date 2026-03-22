# Manuscript Update Guide for Phase 1 Clinical Metrics

**Version:** 2.1.0
**Date:** 2025-01-25
**Purpose:** Ready-to-use text for adding Phase 1 Clinical Metrics to your manuscript

---

## 📝 Quick Start

Copy and paste the sections below into your manuscript. Replace `[X.X]` with your actual numbers from the clinical_test results.

---

## 1. Methods Section Updates

### 1.1 Add New Subsection: "Clinical Utility Assessment"

**Location:** After your main evaluation metrics section

**Text to add:**

```markdown
### 2.X Clinical Utility Assessment

To evaluate clinical applicability beyond traditional accuracy metrics, we implemented
Decision Curve Analysis (DCA) [1,2]. DCA quantifies the net clinical benefit of using
a prediction model compared to alternative strategies across a range of probability
thresholds.

Net benefit was calculated as:

$$\text{NB}(p_t) = \frac{TP}{N} - \frac{FP}{N} \times \frac{p_t}{1-p_t}$$

where $p_t$ represents the probability threshold, $TP$ and $FP$ are true and false
positives, $N$ is the total sample size, and $\frac{p_t}{1-p_t}$ quantifies the
harm-to-benefit ratio at threshold $p_t$.

We compared our model's net benefit curve to two baseline strategies:
- **Treat All**: All patients receive intervention (maximize sensitivity)
- **Treat None**: No patients receive intervention (maximize specificity)

The model is considered clinically useful when its net benefit exceeds both baselines
within a clinically relevant threshold range.

We additionally computed Number Needed to Treat (NNT) to quantify clinical effectiveness:

$$\text{NNT} = \frac{1}{\text{ARR}} = \frac{1}{|\text{CER} - \text{EER}|}$$

where ARR is absolute risk reduction, CER is control event rate (baseline), and EER
is experimental event rate (with model-guided treatment). NNT represents how many
patients must be treated to prevent one adverse event. Values < 10 indicate excellent
effectiveness, 10-20 indicate moderate effectiveness, and > 20 suggest limited
clinical benefit.
```

**References to add:**
```
[1] Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating
    prediction models. Medical Decision Making. 2006;26(6):565-574.
[2] Vickers AJ, Van Calster B, Steyerberg EW. Net benefit approaches to the evaluation
    of prediction models, molecular markers, and diagnostic tests. BMJ. 2016;352:i6.
```

---

### 1.2 Add New Subsection: "Fairness and Bias Assessment"

**Location:** After clinical utility section

**Text to add:**

```markdown
### 2.X Fairness and Bias Assessment

To ensure equitable performance across demographic subgroups and comply with ethical
AI guidelines [3,4], we evaluated algorithmic fairness using five complementary metrics:

**1. Demographic Parity**
Requires equal positive prediction rates across protected groups:
$$P(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = b) \quad \forall a, b$$
where $A$ represents the protected attribute. We considered parity satisfied when the
maximum difference was ≤ 0.10 (10%).

**2. Equalized Odds**
Requires equal true positive rate (TPR) and false positive rate (FPR) across groups:
$$P(\hat{Y} = 1 | Y = y, A = a) = P(\hat{Y} = 1 | Y = y, A = b) \quad \forall y \in \{0,1\}$$
Acceptance threshold: TPR difference ≤ 0.10 and FPR difference ≤ 0.10.

**3. Equal Opportunity**
Requires equal sensitivity (TPR) across groups, ensuring equal ability to identify
positive cases regardless of protected attribute.

**4. Disparate Impact**
Following the "80% rule" from employment discrimination law [5], we computed:
$$\text{DI} = \frac{P(\hat{Y} = 1 | A = \text{unprivileged})}{P(\hat{Y} = 1 | A = \text{privileged})}$$
Fair range: 0.8 ≤ DI ≤ 1.25.

**5. Calibration Fairness**
We assessed whether predicted probabilities match observed frequencies within each
demographic subgroup. Calibration was evaluated using Expected Calibration Error (ECE)
with acceptance threshold ≤ 0.10.

We evaluated fairness across three protected attributes: age group (<40, 40-60, >60),
sex (male, female), and race/ethnicity (following NIH categories).
```

**References to add:**
```
[3] Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning.
    NIPS. 2016.
[4] Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an
    algorithm used to manage the health of populations. Science. 2019;366(6464):447-453.
[5] Biddle RE. Adverse impact and test validation: A practitioner's guide to valid
    and defensible employment testing. Gower Publishing. 2006.
```

---

### 1.3 Add New Subsection: "Uncertainty Quantification with Conformal Prediction"

**Location:** After fairness section

**Text to add:**

```markdown
### 2.X Uncertainty Quantification with Conformal Prediction

To provide rigorous uncertainty estimates with mathematical guarantees, we implemented
conformal prediction [6,7], a distribution-free framework that ensures:

$$P(Y \in C(X)) \geq 1 - \alpha$$

where $C(X)$ is the prediction set for input $X$, $Y$ is the true label, and $\alpha$
is the miscoverage rate. Unlike traditional confidence intervals, this guarantee holds
for any data distribution without parametric assumptions.

**Split Conformal Algorithm:**

1. Split data into training (60%), calibration (20%), and test (20%) sets
2. Train model on training set
3. Compute nonconformity scores on calibration set: $s_i = 1 - P(\hat{Y} = Y_i)$
4. Calculate threshold: $q = \text{Quantile}_{(1-\alpha)(1 + 1/n)}(s_{\text{cal}})$
5. For test sample $x$, include all labels with score ≤ $q$ in prediction set

We set $\alpha = 0.10$ for 90% coverage guarantee. Prediction sets can be:
- **Singleton** (e.g., {Disease A}): High confidence, proceed with diagnosis
- **Multiple labels** (e.g., {Disease A, B, C}): Uncertain, order additional tests
- **Large set**: Defer to specialist or senior clinician

This approach enables the system to appropriately abstain when uncertain, a critical
safety feature for clinical deployment.

**Adaptive Conformal Prediction**
We additionally implemented adaptive conformal prediction to adjust prediction set
sizes based on sample difficulty, measured by entropy of the predicted distribution:
$$H(p) = -\sum_k p_k \log p_k$$

High-entropy (difficult) samples receive larger prediction sets, while low-entropy
(easy) samples receive smaller sets, improving efficiency while maintaining coverage.
```

**References to add:**
```
[6] Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. Springer. 2005.
[7] Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and
    distribution-free uncertainty quantification. arXiv:2107.07511. 2021.
```

---

## 2. Results Section Updates

### 2.1 Add New Subsection: "Clinical Utility Results"

**Location:** After main performance results

**Text to replace [X.X] with your actual numbers:**

```markdown
### 3.X Clinical Utility Results

**Decision Curve Analysis**
Decision curve analysis revealed that our model provided positive net benefit for
probability thresholds between [X.XX] and [X.XX] (Figure X), substantially outperforming
both "treat all" and "treat none" strategies. The maximum net benefit of [X.XXX] was
achieved at threshold [X.XX], corresponding to [X.X] additional patients correctly
managed per 100 evaluated.

At the clinically relevant threshold of 0.30, the model achieved:
- Net Benefit: [X.XXX] (vs. 0 for treat none, [X.XXX] for treat all)
- True Positive Rate: [X.XX]
- False Positive Rate: [X.XX]

**Number Needed to Treat**
The model achieved an NNT of [X.X] (95% CI: [X.X]-[X.X]), indicating that treating
[X] patients based on model predictions would prevent one adverse event. This compares
favorably to established clinical interventions [cite benchmark if available].

Using the model's high-risk classification (threshold = 0.30):
- Control event rate (predicted low-risk): [X.XX]
- Treatment event rate (predicted high-risk): [X.XX]
- Absolute risk reduction: [X.X]%

**Clinical Impact**
At the operating threshold of 0.30:
- [X.X]% of patients classified as high-risk
- Positive Predictive Value (PPV): [X.XXX]
- Negative Predictive Value (NPV): [X.XXX]
- Number needed to screen: [X.X] (screen [X] patients to find 1 true case)

These metrics indicate that the model provides clinically actionable predictions with
acceptable precision and recall for real-world deployment.
```

**Suggested Figure X Caption:**
```
Figure X. Decision Curve Analysis showing net benefit of the prediction model
(blue solid line) compared to "treat all" (red dashed line) and "treat none"
(gray dotted line) strategies across probability thresholds. Shaded region
indicates threshold range where model outperforms both alternatives ([X.XX]-[X.XX]).
Net benefit represents the proportion of patients with improved outcomes per 100
evaluated, accounting for the harm-to-benefit ratio at each threshold.
```

---

### 2.2 Add New Subsection: "Fairness Assessment Results"

**Text to add:**

```markdown
### 3.X Fairness Assessment Results

We evaluated algorithmic fairness across three protected attributes (age group, sex,
race/ethnicity) using five complementary metrics. Table X summarizes results.

**Overall Fairness**
[IF ALL METRICS PASS:]
The model satisfied all fairness criteria across demographic subgroups, with maximum
differences below acceptance thresholds for demographic parity ([X.XX] < 0.10),
equalized odds (TPR: [X.XX], FPR: [X.XX] < 0.10), and calibration error ([X.XX] < 0.10).
Disparate impact ratios ranged from [X.XX] to [X.XX], all within the fair range
(0.8-1.25).

[IF SOME METRICS FAIL:]
The model exhibited statistically significant disparities in [specific metric(s)]
across [specific groups]. Specifically:

- **Demographic Parity**: [PASS/FAIL]. Maximum difference = [X.XX]
  - [Group A]: [X.XX]% positive rate
  - [Group B]: [X.XX]% positive rate
  - Difference: [X.XX] ([> 0.10 threshold])

- **Equalized Odds**: [PASS/FAIL]
  - TPR difference: [X.XX] (threshold: 0.10)
  - FPR difference: [X.XX] (threshold: 0.10)
  - [If fail:] [Group A] TPR = [X.XX], [Group B] TPR = [X.XX]

- **Disparate Impact**: [PASS/FAIL]
  - [Unprivileged group] vs [Privileged group]: DI = [X.XX]
  - [If < 0.8:] Fails 80% rule, indicating potential discrimination

- **Calibration**: [PASS/FAIL]
  - [Group A] calibration error: [X.XX]
  - [Group B] calibration error: [X.XX]
  - [If fail:] Predicted probabilities systematically [over/under]estimate risk in [Group]

Figure Y presents a radar chart summarizing all fairness metrics, where values closer
to 1.0 indicate better fairness.

**Subgroup Analysis**
Performance metrics stratified by protected attributes are presented in Table X+1.
[If disparities exist:] The model showed differential sensitivity for [Group A] ([X.XX])
vs [Group B] ([X.XX]), p < 0.05, suggesting the need for group-specific threshold
calibration or model retraining with balanced sampling.
```

**Suggested Figure Y Caption:**
```
Figure Y. Comprehensive fairness assessment radar chart showing five fairness metrics
(demographic parity, equalized odds, equal opportunity, calibration, disparate impact)
normalized to [0,1] scale where 1 = perfect fairness. Red dashed circle indicates
fairness threshold (0.8). The model [passes/fails] overall fairness criteria with
[specific metrics] requiring attention before deployment.
```

**Suggested Table X:**
```
Table X. Fairness Metrics Across Protected Attributes

Metric                  | Age Group | Sex   | Race/Ethnicity | Threshold | Status
------------------------|-----------|-------|----------------|-----------|--------
Demographic Parity Δ    | 0.XX      | 0.XX  | 0.XX          | ≤ 0.10    | PASS/FAIL
Equalized Odds (TPR) Δ  | 0.XX      | 0.XX  | 0.XX          | ≤ 0.10    | PASS/FAIL
Equalized Odds (FPR) Δ  | 0.XX      | 0.XX  | 0.XX          | ≤ 0.10    | PASS/FAIL
Equal Opportunity Δ     | 0.XX      | 0.XX  | 0.XX          | ≤ 0.10    | PASS/FAIL
Disparate Impact Ratio  | 0.XX      | 0.XX  | 0.XX          | 0.8-1.25  | PASS/FAIL
Calibration Error       | 0.XX      | 0.XX  | 0.XX          | ≤ 0.10    | PASS/FAIL

Δ = Maximum difference across subgroups within attribute
PASS = Within acceptance threshold, FAIL = Exceeds threshold
```

---

### 2.3 Add New Subsection: "Uncertainty Quantification Results"

**Text to add:**

```markdown
### 3.X Uncertainty Quantification Results

**Conformal Prediction Coverage**
Split conformal prediction with miscoverage rate α = 0.10 (target: 90% coverage)
achieved [X.X]% empirical coverage on the test set, validating the theoretical
guarantee. The average prediction set size was [X.XX], indicating [excellent/good/moderate]
efficiency.

Distribution of prediction sets:
- Singleton sets (high confidence): [X]% ([Y]/[N] samples)
- 2-3 labels (moderate uncertainty): [X]% ([Y]/[N] samples)
- > 3 labels (high uncertainty): [X]% ([Y]/[N] samples)

Figure Z shows the distribution of prediction set sizes and their relationship to
model confidence (entropy).

**Adaptive Conformal Prediction**
Adaptive conformal prediction, which adjusts set sizes based on sample difficulty,
improved efficiency by [X.X]% while maintaining the 90% coverage guarantee. For easy
samples (low entropy < [X.X]), the average set size was [X.XX], while difficult samples
(high entropy > [X.X]) had average set size [X.XX].

**Risk-Controlled Predictions**
Using Learn-Then-Test risk control to ensure false negative rate ≤ 5%, we calibrated
a threshold of [X.XX]. At this threshold:
- Empirical FNR: [X.X]% (target: ≤ 5%)
- Rejection rate: [X.X]% (model deferred to clinician)
- True positive rate among non-rejected: [X.X]%

This demonstrates that rigorous FNR control is achievable with moderate rejection rates,
enabling safe clinical deployment.

**Coverage Validation Across Alpha Levels**
We validated coverage guarantees across miscoverage rates α ∈ {0.05, 0.10, 0.15, 0.20}.
All empirical coverages were within 2% of theoretical guarantees, confirming the
validity of the conformal framework for this application (Figure W).
```

**Suggested Figure Z Caption:**
```
Figure Z. Distribution of conformal prediction set sizes. (A) Histogram showing
frequency of singleton, 2-label, 3-label, and larger prediction sets. Singleton
sets indicate high confidence where model can make definitive predictions. (B)
Average prediction set size stratified by model uncertainty (entropy), showing
adaptive conformal prediction adjusts set size based on difficulty.
```

**Suggested Figure W Caption:**
```
Figure W. Conformal prediction coverage validation. Empirical coverage (blue circles)
vs theoretical guarantee (black diagonal line) across miscoverage rates α from 0.05
to 0.30. All empirical coverages meet or exceed theoretical guarantees, confirming
validity of the distribution-free framework.
```

---

## 3. Discussion Section Updates

### 3.1 Add Subsection: "Clinical Utility and Deployment Readiness"

**Text to add:**

```markdown
### 4.X Clinical Utility and Deployment Readiness

Our Decision Curve Analysis demonstrates that the model provides clinically meaningful
benefit across a wide range of probability thresholds ([X.XX]-[X.XX]), substantially
outperforming both universal treatment and no treatment strategies. The NNT of [X.X]
compares favorably to established clinical interventions [cite benchmarks if available],
suggesting the model is ready for prospective validation and clinical deployment.

The clinically useful threshold range aligns well with real-world decision contexts.
For example, at threshold 0.30 (appropriate for screening scenarios), the model achieves
PPV of [X.XX] and NPV of [X.XX], providing actionable guidance for triage decisions.
At higher thresholds (0.50-0.70), suitable for high-stakes interventions, the model
maintains positive net benefit while reducing false positives.

This threshold flexibility enables deployment across different clinical settings with
varying risk tolerances and resource constraints. The comprehensive clinical impact
assessment (Figure X) provides decision-makers with transparent information about
expected outcomes at each operating point.
```

---

### 3.2 Add Subsection: "Fairness, Equity, and Ethical AI"

**Text to add (customize based on your results):**

**[IF PASSED ALL FAIRNESS CRITERIA:]**
```markdown
### 4.X Fairness, Equity, and Ethical AI

Our comprehensive fairness assessment demonstrates that the model performs equitably
across demographic subgroups, satisfying demographic parity, equalized odds, equal
opportunity, and calibration criteria. This finding is particularly important given
well-documented disparities in healthcare AI systems [cite Obermeyer 2019, others].

The model's achievement of calibration fairness—where predicted probabilities accurately
reflect risk across all demographic groups—is especially critical for clinical trust
and appropriate use. Miscalibration could lead to systematic over- or under-treatment
of certain populations, perpetuating or exacerbating existing healthcare disparities.

Compliance with the 80% rule for disparate impact (0.8 ≤ DI ≤ 1.25) across all
protected attributes suggests the model can be deployed without legal or ethical
concerns regarding discriminatory impact.

However, continuous monitoring will be essential during deployment to ensure fairness
is maintained as patient populations and clinical practices evolve.
```

**[IF FAILED SOME FAIRNESS CRITERIA:]**
```markdown
### 4.X Fairness, Equity, and Ethical AI

While our model achieved high overall performance, fairness assessment revealed
statistically significant disparities in [specific metric] across [specific groups].
Specifically, [describe key finding, e.g., "the model exhibited lower sensitivity
for Black patients (0.XX) compared to White patients (0.XX), p < 0.05"].

This finding underscores the critical importance of comprehensive fairness evaluation
in medical AI systems. Similar disparities have been documented in other healthcare
algorithms [cite Obermeyer 2019, Rajkomar 2018], often arising from:
1. Underrepresentation of minority groups in training data
2. Differential feature distributions across populations
3. Historical biases in clinical decision-making embedded in training labels

**Mitigation Strategies:**
To address these disparities before deployment, we recommend:

1. **Group-Specific Threshold Calibration**: Apply different classification thresholds
   per demographic group to equalize TPR/FPR while maintaining overall performance

2. **Balanced Sampling**: Retrain model with stratified sampling to ensure adequate
   representation of underrepresented groups

3. **Fairness-Constrained Learning**: Incorporate fairness constraints (e.g., equalized
   odds) directly into the model training objective

4. **Post-Processing Calibration**: Apply isotonic regression separately per group to
   ensure calibrated probabilities

5. **Continuous Monitoring**: Implement automated fairness audits during deployment
   to detect emerging disparities

The transparency provided by our Phase 1 Clinical Metrics framework enables early
detection of such issues, allowing for remediation before clinical deployment.
Addressing algorithmic fairness is not merely a technical challenge but an ethical
imperative to ensure equitable healthcare for all populations.
```

---

### 3.3 Add Subsection: "Uncertainty Quantification and Clinical Trust"

**Text to add:**

```markdown
### 4.X Uncertainty Quantification and Clinical Trust

The conformal prediction framework provides rigorous uncertainty estimates with
mathematical coverage guarantees (P(Y ∈ C(X)) ≥ 1 - α), a critical advance over
traditional ML confidence scores which lack theoretical grounding. Achieving [X.X]%
empirical coverage against a 90% guarantee validates the approach for this application.

The distribution-free nature of conformal prediction is particularly valuable in
clinical settings where data distributions may be non-Gaussian and evolve over time.
Unlike Bayesian or parametric approaches, conformal prediction requires no assumptions
about the underlying distribution, providing robust uncertainty estimates across
diverse patient populations.

**Clinical Decision Support Integration:**
The prediction set framework naturally maps to clinical workflows:

- **Singleton sets** ([X]% of cases): Model provides definitive prediction; clinician
  can proceed with high confidence

- **2-3 label sets** ([X]% of cases): Differential diagnosis provided; order additional
  tests to disambiguate (e.g., if set = {Pneumonia, COVID-19}, order PCR test)

- **Large sets** ([X]% of cases): High uncertainty; defer to specialist or senior
  clinician for comprehensive evaluation

This "I don't know" capability is essential for safe AI deployment in high-stakes
clinical settings. Systems that provide overconfident point predictions without
uncertainty estimates risk dangerous overreliance and automation bias.

**Risk Control:**
The ability to calibrate thresholds for controlled false negative rates (e.g., FNR ≤ 5%)
enables deployment with safety guarantees aligned with clinical risk tolerance. This
addresses a key barrier to clinical AI adoption—the inability to bound worst-case
error rates.

Future work will evaluate temporal stability of conformal guarantees and develop
online updating procedures for non-stationary clinical environments.
```

---

## 4. Limitations Section Updates

**Add new limitation subsection:**

```markdown
### 5.X Limitations of Phase 1 Clinical Metrics Assessment

While our Phase 1 Clinical Metrics evaluation demonstrates readiness for FDA submission
and ethical AI compliance, this study has several limitations:

**Temporal Validation:**
We have not yet evaluated model performance over time or under concept drift. Medical
AI systems deployed in real-world settings face evolving patient populations, changing
clinical practices, and temporal shifts in disease prevalence. Continuous monitoring
and validation will be critical for maintaining performance and fairness guarantees
over time.

**External Validation:**
Fairness metrics were evaluated on [describe your dataset/population]. Results may not
generalize to other healthcare settings, geographic regions, or patient populations
with different demographic distributions or comorbidity profiles. Multi-site external
validation is needed before widespread deployment.

**Conformal Prediction Assumptions:**
The conformal prediction coverage guarantee assumes exchangeability—that calibration
and test samples are drawn from the same distribution. This assumption may be violated
in non-stationary clinical environments or during population health crises (e.g.,
pandemics). Adaptive conformal methods or online recalibration may be necessary for
long-term deployment.

**Threshold Selection:**
While Decision Curve Analysis identifies clinically useful threshold ranges, optimal
threshold selection in practice requires input from domain experts, consideration of
local resource constraints, and alignment with institutional risk preferences. Our
analysis provides quantitative decision support but does not prescribe a universal
operating point.

**Fairness-Accuracy Trade-offs:**
[If applicable:] Achieving perfect fairness across all metrics simultaneously may
require accepting some reduction in overall accuracy. The optimal fairness-accuracy
trade-off is context-dependent and requires stakeholder engagement including patients,
clinicians, and ethicists.

**Causality:**
Our fairness assessment is associational, not causal. Achieving demographic parity
does not necessarily imply counterfactual fairness—that individuals would receive
the same prediction had they belonged to a different demographic group. Future work
should incorporate causal fairness frameworks to more rigorously assess equity.

Future research will address these limitations through prospective validation,
continuous monitoring frameworks (Phase 3), and causal inference methods (Phase 4).
```

---

## 5. Conclusion Section Updates

**Add to your conclusion:**

```markdown
In summary, our comprehensive Phase 1 Clinical Metrics assessment—including decision
curve analysis, fairness evaluation, and uncertainty quantification with conformal
prediction—demonstrates that the BASICS-CDSS framework provides [clinically useful /
requires fairness improvements before] predictions [with rigorous uncertainty estimates
and equity guarantees]. The model achieves NNT of [X.X], [passes/requires attention
for] fairness criteria across demographic subgroups, and provides 90% coverage guarantee
for uncertainty quantification.

These results indicate that the system is [ready for prospective clinical validation /
requires the following improvements before deployment: [list]]. The transparent,
comprehensive evaluation framework presented here can serve as a template for rigorous
medical AI validation, addressing critical gaps in current practice and supporting
regulatory approval, ethical deployment, and clinical adoption.
```

---

## 6. Figure List Summary

**Recommended new figures to add:**

| # | Figure Title | Type | Priority |
|---|-------------|------|----------|
| X | Decision Curve Analysis | 2D Line | **HIGH** |
| Y | Fairness Radar Chart | 2D Radar | **HIGH** |
| Z | Conformal Prediction Set Sizes | 2D Histogram | MEDIUM |
| W | Coverage Validation | 2D Line | MEDIUM |
| X+1 | Clinical Impact 3D | 3D Surface | LOW |
| X+2 | Standardized Net Benefit | 2D Bar | LOW |

**Minimum viable:** Add Figures X and Y
**Recommended:** Add Figures X, Y, Z, W
**Complete:** Add all figures

---

## 7. Tables Summary

**Recommended new tables:**

| # | Table Title | Priority |
|---|------------|----------|
| X | Fairness Metrics Across Protected Attributes | **HIGH** |
| X+1 | Performance Stratified by Demographics | MEDIUM |
| X+2 | Net Benefit at Key Thresholds | MEDIUM |
| X+3 | Conformal Prediction Statistics | LOW |

---

## 8. References to Add

**Critical references (add these):**

**Decision Curve Analysis:**
1. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making. 2006;26(6):565-574.
2. Vickers AJ, Van Calster B, Steyerberg EW. Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests. BMJ. 2016;352:i6.

**Fairness:**
3. Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning. NIPS. 2016.
4. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447-453.
5. Chouldechova A. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big data. 2017;5(2):153-163.

**Conformal Prediction:**
6. Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. Springer. 2005.
7. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. 2021.
8. Angelopoulos AN, Bates S, Fisch A, et al. Learn then test: Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052. 2022.

**Additional recommended:**
9. Rajkomar A, Hardt M, Howell MD, Corrado G, Chin MH. Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine. 2018;169(12):866-872.
10. Chen IY, Pierson E, Rose S, Joshi S, Ferryman K, Ghassemi M. Ethical machine learning in healthcare. Annual Review of Biomedical Data Science. 2021;4:123-144.

---

## 9. Quick Checklist for Manuscript Updates

Use this checklist to track your updates:

### Methods Section
- [ ] Added "Clinical Utility Assessment" subsection
- [ ] Added "Fairness and Bias Assessment" subsection
- [ ] Added "Uncertainty Quantification with Conformal Prediction" subsection
- [ ] Added equations for Net Benefit, NNT, fairness metrics
- [ ] Added description of conformal algorithm
- [ ] Added new references (1-8 minimum)

### Results Section
- [ ] Added "Clinical Utility Results" subsection with DCA, NNT results
- [ ] Filled in [X.X] placeholders with actual numbers
- [ ] Added "Fairness Assessment Results" subsection
- [ ] Created Table X (Fairness Metrics)
- [ ] Added "Uncertainty Quantification Results" subsection
- [ ] Added Figure X (Decision Curve)
- [ ] Added Figure Y (Fairness Radar)
- [ ] Added Figure Z and/or W (Conformal Prediction)

### Discussion Section
- [ ] Added "Clinical Utility and Deployment Readiness" subsection
- [ ] Added "Fairness, Equity, and Ethical AI" subsection (customized to your results)
- [ ] Added "Uncertainty Quantification and Clinical Trust" subsection
- [ ] Discussed clinical implications and decision-making integration

### Limitations Section
- [ ] Added "Limitations of Phase 1 Clinical Metrics Assessment"
- [ ] Mentioned temporal validation, external validation, conformal assumptions

### Conclusion
- [ ] Updated to mention Phase 1 Clinical Metrics results
- [ ] Stated deployment readiness or required improvements

### Supporting Materials
- [ ] Created high-resolution figures (300 DPI) from clinical_test/
- [ ] Wrote figure captions
- [ ] Created Table X with your actual fairness metrics
- [ ] Added all new references to bibliography

---

## 10. Where to Get Your Numbers

Run this to generate production figures with more samples:

```bash
cd D:\PhD\Manuscript\GitHub\BASICS-CDSS\examples
python generate_clinical_metrics_figures.py --n-samples 500 --output-dir manuscript_figures
```

Then extract numbers from the output:

**For Clinical Utility ([X.X] values):**
- Look for: "Useful threshold range: (X.XX, X.XX)"
- Look for: "Model NNT: X.X (ARR: XX.X%)"
- Check clinical_impact_0.3.pdf for PPV, NPV, etc.

**For Fairness:**
- Look for: "Fairness Report for race: Overall Fair: True/False"
- Look for: "Failed Criteria: [...]"
- Measure differences from the generated fairness PDFs

**For Conformal Prediction:**
- Look for: "Target Coverage: 90.0%"
- Look for: "Average Set Size: X.XX"
- Look for: "Singleton Sets: X/Y"

---

## 11. Timeline Estimate

**Minimum viable update (Figures X, Y only):** 2-3 hours
- Copy Methods sections → 45 min
- Fill in Results with numbers → 45 min
- Write Discussion additions → 45 min
- Add figures and captions → 30 min

**Recommended update (Figures X, Y, Z, W + Table X):** 4-5 hours
- All of above + create/format tables → +1 hour
- More thorough Discussion → +1 hour

**Complete update (all figures, all tables, supplementary):** 6-8 hours
- All of above + supplementary materials
- Multiple rounds of revision

---

## 12. Final Notes

**Important:**
- Replace ALL [X.X] placeholders with your actual numbers
- Customize fairness discussion based on whether you PASSED or FAILED criteria
- Adjust figure numbers (X, Y, Z) to match your manuscript numbering
- Have co-authors/advisors review fairness interpretations (sensitive topic)

**Pro tips:**
- Start with Methods section (easiest, factual)
- Then Results (fill in numbers from output)
- Discussion requires most thought—do last
- Get feedback early on fairness section if you failed any criteria

**Need help?**
- See [CLINICAL_METRICS_GUIDE.md](CLINICAL_METRICS_GUIDE.md) for detailed metric explanations
- See generated PDFs in clinical_test/ for visual examples
- Check arXiv papers referenced for more context

---

**Good luck with your manuscript updates! 🚀**

Let me know if you need help with specific sections or interpreting your results.
