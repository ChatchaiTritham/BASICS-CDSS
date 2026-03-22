# Detailed Metric Explanations for Different Paper Types

**How to explain each metric depending on your paper type and audience**

---

## Table of Contents

1. [Decision Curve Analysis (DCA)](#decision-curve-analysis)
2. [Net Benefit](#net-benefit)
3. [Number Needed to Treat (NNT)](#number-needed-to-treat)
4. [Demographic Parity](#demographic-parity)
5. [Equalized Odds](#equalized-odds)
6. [Calibration Fairness](#calibration-fairness)
7. [Disparate Impact](#disparate-impact)
8. [Conformal Prediction](#conformal-prediction)

---

## Decision Curve Analysis (DCA)

### For Clinical Audience (JAMA, NEJM, Lancet):

**Simple Explanation:**
```
Decision curve analysis helps answer a critical question: "Should we use this model
in clinical practice?" Unlike accuracy, which treats all errors equally, decision
curve analysis accounts for the clinical consequences of false positives (unnecessary
treatment) versus false negatives (missed diagnosis).

The curve shows "net benefit"—essentially, how many additional patients are correctly
managed per 100 evaluated, compared to either treating everyone or treating no one.
When the model's curve is above both alternatives, it provides clinical value.
```

**How to Present in Results:**
```
"Figure 1 shows the decision curve for our model. The model curve (blue) exceeds both
'treat all' (red dashed) and 'treat none' (gray) strategies between threshold 0.20
and 0.65, indicating clinical utility within this range.

At threshold 0.30—appropriate for screening in our context—the model's net benefit
is 0.25, meaning 25 additional patients per 100 screened would be correctly managed
compared to current practice (treating no one preventively).

For comparison, treating all patients would result in excessive interventions with
net benefit of only 0.12, while treating no one provides zero benefit."
```

**Key Points for Discussion:**
- Threshold selection depends on clinical context (screening vs high-stakes intervention)
- Net benefit > 0 is necessary but not sufficient for deployment
- Compare to benchmark interventions (e.g., "comparable to established screening programs")

---

### For Technical Audience (IEEE, AIM, ML Conferences):

**Mathematical Explanation:**
```
Decision Curve Analysis extends traditional ROC analysis by incorporating clinical
preferences through the harm-to-benefit ratio. Net benefit is defined as:

NB(pt) = (TP/N) - (FP/N) × [pt/(1-pt)]

where pt is the probability threshold, and pt/(1-pt) represents the odds—quantifying
how much harm from a false positive is acceptable relative to benefit from a true
positive.

The term pt/(1-pt) can be interpreted as the number of patients a clinician would
accept treating unnecessarily to identify one true case. For example:
- pt = 0.10 → willing to treat 9 unnecessarily to find 1 case (screening)
- pt = 0.50 → willing to treat 1 unnecessarily to find 1 case (balanced)
- pt = 0.80 → willing to treat 0.25 unnecessarily to find 1 case (high-stakes)
```

**Implementation Details for Methods:**
```
We computed net benefit across 100 linearly-spaced thresholds from 0.01 to 0.99.
For each threshold pt, we calculated:

1. Model net benefit: NB_model(pt) using the formula above
2. "Treat all" baseline: NB_all(pt) = prev - (1-prev)×[pt/(1-pt)]
3. "Treat none" baseline: NB_none(pt) = 0

where prev is outcome prevalence. The clinically useful threshold range is defined
as {pt : NB_model(pt) > max(NB_all(pt), 0)}.

We additionally computed standardized net benefit (per 100 patients) and interventions
avoided relative to treat-all strategy.
```

---

### For Fairness/Ethics Papers:

**Equity Implications:**
```
Decision curve analysis was computed separately for each demographic subgroup to
assess whether clinical utility varies across populations. Differential net benefit
curves would indicate that:

1. Optimal thresholds differ by group (requiring group-specific deployment strategies)
2. Some groups may not benefit from the model (precluding universal deployment)
3. The harm-to-benefit trade-off may be population-specific

We found [describe findings: uniform utility vs differential utility across groups].
```

---

## Net Benefit

### For Clinical Audience:

**Intuitive Explanation:**
```
Net benefit answers: "How many more patients are correctly managed using the model
versus current practice?"

A net benefit of 0.25 means that for every 100 patients evaluated, 25 additional
patients receive appropriate care (correct treatment or correct withholding of treatment)
compared to the baseline strategy.

Net benefit accounts for both:
- Benefits: Correctly identifying patients who need treatment (true positives)
- Harms: Incorrectly treating patients who don't need it (false positives, weighted
  by the clinical harm-to-benefit ratio)
```

**How to Interpret Values:**
```
Net Benefit Values:
- NB > 0.20: Excellent clinical utility (20+ additional patients correctly managed per 100)
- NB 0.10-0.20: Moderate clinical utility
- NB 0.05-0.10: Minimal but potentially useful
- NB ≤ 0.05: Limited clinical value
- NB < 0: Harmful (worse than doing nothing)
```

---

### For Technical Audience:

**Relationship to Other Metrics:**
```
Net benefit unifies sensitivity and specificity through a utility function:

NB = (sensitivity × prevalence) - (1-specificity) × (1-prevalence) × w

where w = pt/(1-pt) is the weight representing clinical preferences.

This formulation shows that net benefit is essentially weighted accuracy, where:
- True positives receive weight = 1
- False positives receive weight = -w
- True negatives receive weight = 0
- False negatives receive weight = 0

Unlike AUC-ROC (which integrates over all thresholds), net benefit evaluates
performance at a specific clinically-selected threshold.
```

---

## Number Needed to Treat (NNT)

### For Clinical Audience:

**Clear Interpretation:**
```
NNT = 5 means we must treat 5 patients to prevent 1 adverse event.

Lower NNT = More effective intervention
Higher NNT = Less effective intervention

Clinical Benchmarks:
- NNT < 10: Highly effective (e.g., statins for secondary prevention: NNT ~30)
- NNT 10-20: Moderately effective
- NNT 20-50: Minimally effective, consider cost-benefit
- NNT > 50: Limited effectiveness

Our model achieved NNT = [X.X], indicating [interpretation relative to benchmarks].
```

**How to Present:**
```
"To prevent one [ADVERSE EVENT], [X] patients would need to be [INTERVENTION] based
on the model's high-risk classification. This compares favorably to [BENCHMARK from
literature, e.g., 'NNT of 25 for aspirin in secondary prevention']."
```

**Cost-Effectiveness Connection:**
```
NNT directly informs cost-effectiveness:

Cost per event prevented = NNT × Cost per patient treated

Example: If intervention costs $1,000 per patient and NNT = 10:
Cost per event prevented = 10 × $1,000 = $10,000

This can be compared to willingness-to-pay thresholds or cost-effectiveness of
alternative interventions.
```

---

### For Technical Audience:

**Calculation Details:**
```
NNT = 1 / ARR

where ARR (Absolute Risk Reduction) = |CER - EER|

CER = Control Event Rate (baseline, without model)
EER = Experimental Event Rate (with model-guided treatment)

**Implementation:**
1. Classify patients using model (e.g., threshold = 0.30)
2. Compute event rate in predicted high-risk group (EER)
3. Compute event rate in predicted low-risk group (proxy for CER, or use external benchmark)
4. ARR = |CER - EER|
5. NNT = 1/ARR

**Confidence Intervals:**
We computed 95% CI using the Newcombe method:

SE(ARR) = sqrt(SE²_CER + SE²_EER)

where SE_CER = sqrt(CER × (1-CER) / n_control)
      SE_EER = sqrt(EER × (1-EER) / n_experimental)

CI for NNT: [1/(ARR + 1.96×SE), 1/(ARR - 1.96×SE)]

Note: CI order reverses due to reciprocal transformation.
```

---

## Demographic Parity

### For Fairness Papers:

**Definition and Motivation:**
```
Demographic parity (also called statistical parity or independence) requires that
the probability of a positive prediction is independent of protected attributes:

P(Ŷ = 1 | A = a) = P(Ŷ = 1 | A = b)  for all groups a, b

**Why it matters:**
Violations indicate that certain demographic groups receive systematically different
treatment recommendations, which may reflect or perpetuate healthcare disparities.

**Example:**
If 60% of White patients are classified as high-risk but only 40% of Black patients,
demographic parity is violated (difference = 0.20). This could indicate:
1. Model underestimates risk for Black patients (discrimination)
2. True prevalence differs by race (not necessarily unfair)
3. Feature distributions differ (e.g., differential access to diagnostic testing)

**Interpretation requires clinical context**—demographic parity is not always the
appropriate fairness criterion.
```

**When Demographic Parity is Appropriate:**
```
Use demographic parity when:
- Selection for programs/resources where outcome is uncertain
- Screening for conditions with similar prevalence across groups
- Procedural fairness is the goal (equal treatment)

Do NOT require demographic parity when:
- True prevalence legitimately differs across groups
- Features reflect biological/epidemiological differences
- Different base rates are clinically justified
```

---

### For Clinical Audience:

**Plain Language:**
```
Demographic parity assesses whether the model classifies similar proportions of
patients as high-risk across demographic groups.

In our study:
- White patients: [X]% classified high-risk
- Black patients: [Y]% classified high-risk
- Difference: [|X-Y|]%

We considered parity satisfied if the difference was ≤10%.

[IF VIOLATED:]
The model classifies Black patients as high-risk [more/less] often than White patients.
This disparity warrants investigation: it may reflect true prevalence differences,
differential feature availability, or algorithmic bias. We performed [additional
analyses] to disentangle these possibilities.
```

---

## Equalized Odds

### For Fairness Papers:

**Rigorous Definition:**
```
Equalized odds requires that TPR and FPR are equal across groups:

P(Ŷ = 1 | Y = y, A = a) = P(Ŷ = 1 | Y = y, A = b)  for y ∈ {0,1}, all groups a,b

Equivalently:
- TPR(Group A) = TPR(Group B)  [Equal sensitivity]
- FPR(Group A) = FPR(Group B)  [Equal 1-specificity]

**Why it matters:**
Equalized odds ensures the model has equal error rates across groups. Violations
mean the model performs differentially well for different demographics:

- TPR disparity: Some groups have more missed diagnoses (false negatives)
- FPR disparity: Some groups have more false alarms (false positives)

**Connection to fairness concepts:**
Equalized odds satisfies:
1. Separation: Predictions separated from protected attributes given outcome
2. Equal accuracy: Both types of errors are equalized
3. Conditional use accuracy equality: Model equally useful conditional on true outcome
```

**When Equalized Odds is Appropriate:**
```
Equalized odds is generally preferred for medical AI because:
- Ensures equal quality of care (equal error rates)
- Respects outcome base rates (allows P(Ŷ=1|A=a) ≠ P(Ŷ=1|A=b) if prevalence differs)
- Clinically interpretable (equal sensitivity and specificity)

Use equalized odds for:
- Diagnostic models
- Prognostic models
- Treatment allocation where outcomes are known
```

---

### For Clinical Audience:

**Plain Language:**
```
Equalized odds checks whether the model is equally accurate for different demographic
groups. Specifically:

**Sensitivity (True Positive Rate):**
Among patients with the condition, does the model detect it equally well across groups?

Example: If the model has 90% sensitivity for White patients but only 75% for Black
patients, it misses 15 additional cases per 100 Black patients—a serious equity issue.

**Specificity (True Negative Rate = 1 - FPR):**
Among patients without the condition, does the model correctly rule it out equally
across groups?

Example: If specificity is 85% for White patients but 70% for Black patients, Black
patients experience 15 additional false alarms per 100—leading to unnecessary
interventions, costs, and anxiety.

**Our Findings:**
[Report TPR and FPR by group, interpret differences]
```

---

## Calibration Fairness

### For All Audiences:

**What it Means:**
```
Calibration fairness requires that predicted probabilities accurately reflect risk
within each demographic group:

P(Y = 1 | Score = s, A = a) = s  for all groups a

**Why it matters clinically:**
If a model predicts "70% probability of readmission" for both a White patient and a
Black patient, but the actual readmission rate is:
- White patients with 70% predicted: 70% actually readmit (well-calibrated)
- Black patients with 70% predicted: 50% actually readmit (miscalibrated)

Then the model systematically overestimates risk for Black patients, potentially
leading to:
- Overtreatment (unnecessary aggressive interventions)
- Resource misallocation (intensive monitoring for lower-risk patients)
- Patient harm (treatment side effects without proportionate benefit)
```

**How We Assessed Calibration:**
```
1. Grouped predictions into 10 bins by predicted probability (0-0.1, 0.1-0.2, ..., 0.9-1.0)
2. For each group (race/ethnicity, age, sex):
   - Computed mean predicted probability in each bin
   - Computed observed frequency (actual event rate) in each bin
3. Perfect calibration: predicted probability = observed frequency for all bins
4. Measured calibration error: mean absolute difference between predicted and observed
5. Acceptance threshold: calibration error ≤ 0.10
```

**Calibration Plots:**
```
Figure X shows calibration curves for each demographic group. The diagonal line
represents perfect calibration (predicted = observed). Deviations indicate miscalibration:

- Points above diagonal: Model UNDERestimates risk for this group
- Points below diagonal: Model OVERestimates risk for this group

[Describe your findings]
```

---

## Disparate Impact

### For Fairness/Legal Context:

**80% Rule Explanation:**
```
The "80% rule" (four-fifths rule) comes from US employment discrimination law
(Uniform Guidelines on Employee Selection Procedures, 1978):

Disparate Impact Ratio = Selection Rate (Unprivileged) / Selection Rate (Privileged)

**Legal threshold: DI ≥ 0.80**

Example:
- White patients: 50% classified high-risk
- Black patients: 30% classified high-risk
- DI = 0.30 / 0.50 = 0.60 < 0.80 → FAILS 80% rule

This suggests potential discrimination, triggering further investigation.

**Fair range:** 0.8 ≤ DI ≤ 1.25

**Our Analysis:**
We computed disparate impact ratios comparing each minority group to White patients
(privileged group) for high-risk classification (threshold = 0.30):

[Report DI ratios, interpret against 80% rule]
```

---

## Conformal Prediction

### For Clinical Audience:

**What It Provides:**
```
Traditional AI: "This patient has Disease A" (point prediction, no uncertainty)
Conformal Prediction: "This patient has Disease A, B, or C—order Test X to distinguish"
(prediction set with guaranteed coverage)

**The Guarantee:**
With 90% confidence, the true diagnosis is in the prediction set.

This is NOT a confidence interval (which has no finite-sample guarantees).
This is a mathematically proven bound that works for ANY data distribution.

**Clinical Use:**
- Singleton set {Disease A}: High confidence → proceed with treatment
- Small set {Disease A, Disease B}: Moderate uncertainty → order differential diagnostic tests
- Large set {Disease A, B, C, D, E}: High uncertainty → consult specialist

This enables appropriate abstention—a critical safety feature for clinical AI.
```

**Example Clinical Scenario:**
```
Patient presents with chest pain. Conformal prediction output:

**Scenario 1 (High Confidence):**
Prediction set: {STEMI}
Action: Activate cath lab immediately

**Scenario 2 (Moderate Uncertainty):**
Prediction set: {STEMI, NSTEMI}
Action: Order troponin to distinguish, prepare for potential cath

**Scenario 3 (High Uncertainty):**
Prediction set: {STEMI, NSTEMI, Unstable Angina, Aortic Dissection, Pulmonary Embolism}
Action: Comprehensive workup, senior clinician evaluation

Traditional AI would force a single prediction regardless of uncertainty, risking
dangerous overconfidence.
```

---

### For Technical Audience:

**Algorithm Details:**
```
Split Conformal Prediction (Classification):

1. Split data: Training (60%), Calibration (20%), Test (20%)

2. Train model f on training set

3. Compute nonconformity scores on calibration set:
   s_i = 1 - P(f(X_i) = Y_i)

   where P(f(X_i) = Y_i) is model's predicted probability for true label

4. Compute adjusted quantile:
   q = Quantile_{(1-α)(1+1/n)}(s_cal)

   where n = |calibration set|, α = miscoverage rate (e.g., 0.10 for 90% coverage)

5. For test sample X, construct prediction set:
   C(X) = {y : 1 - P(f(X) = y) ≤ q}

   Include all labels with nonconformity score ≤ q

**Coverage Guarantee:**
P(Y ∈ C(X)) ≥ 1 - α

This holds under exchangeability assumption (calibration and test samples drawn from
same distribution).

**Key Properties:**
- Distribution-free: No parametric assumptions
- Finite-sample guarantee: Not asymptotic
- Model-agnostic: Works with any model f

**Adaptive Variant:**
We additionally implemented adaptive conformal prediction, adjusting set sizes based
on sample difficulty (entropy of predicted distribution), improving efficiency while
maintaining coverage.
```

**Implementation Code Reference:**
```python
from basics_cdss.clinical_metrics import split_conformal_classification

conf_result = split_conformal_classification(
    model=trained_model,
    X_train=X_train, y_train=y_train,
    X_cal=X_cal, y_cal=y_cal,
    X_test=X_test,
    alpha=0.10  # 90% coverage
)

print(f"Coverage: {conf_result.target_coverage}")
print(f"Avg set size: {conf_result.efficiency}")
```

---

## Quick Selection Guide for Your Paper

**Which metrics to emphasize?**

| Paper Type | Primary Metrics | Secondary Metrics |
|-----------|----------------|-------------------|
| Clinical Application | DCA, NNT | Equalized Odds, Calibration |
| Fairness-Focused | All 5 fairness | DCA for subgroups |
| Methodological | All metrics equally | - |
| Validation Study | DCA, Calibration | Fairness |
| Short Letter | DCA, 1-2 fairness | Conformal |

**Depth of explanation:**

| Audience | Explanation Level |
|----------|------------------|
| Clinical (JAMA, NEJM) | Intuitive, examples, clinical benchmarks |
| Technical (IEEE, ML) | Mathematical, implementation, proofs |
| Mixed (Nature Med) | Both intuitive + technical details in supplement |
| General Public | Analogies, avoid jargon, focus on implications |

---

## Standard Sentences to Use

**Decision Curve Analysis:**
- "Decision curve analysis revealed the model provided clinical utility for thresholds [X]-[Y]."
- "Net benefit exceeded both treat-all and treat-none strategies."
- "The optimal operating point was threshold [X] with net benefit [Y]."

**Fairness:**
- "Comprehensive fairness assessment across five metrics showed [equitable/disparate] performance."
- "The model satisfied [X]/5 fairness criteria."
- "Equalized odds analysis revealed [TPR/FPR] differences of [X] across [groups]."

**Conformal Prediction:**
- "Conformal prediction achieved [X]% empirical coverage (target: [Y]%)."
- "Average prediction set size was [X], with [Y]% singleton sets indicating high confidence."
- "The model appropriately abstained on [X]% of cases with high uncertainty."

---

**For copy-paste text, see MANUSCRIPT_UPDATES.md**
**For paper structures, see PAPER_TEMPLATES.md**
**For supplementary materials, see next document**
