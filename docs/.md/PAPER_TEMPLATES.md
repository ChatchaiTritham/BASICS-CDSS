# Paper Templates for Different Journal Types

**BASICS-CDSS v2.1.0 supports multiple paper types**

This guide provides ready-to-use templates for different types of manuscripts you can write using BASICS-CDSS Phase 1 Clinical Metrics.

---

## 📚 Table of Contents

1. [Paper Type A: Methodological Paper (Framework Focus)](#paper-type-a-methodological)
2. [Paper Type B: Application Paper (Clinical Focus)](#paper-type-b-application)
3. [Paper Type C: Fairness-Focused Paper](#paper-type-c-fairness)
4. [Paper Type D: Short Communication / Letter](#paper-type-d-short)
5. [Paper Type E: Validation Study](#paper-type-e-validation)

---

## Paper Type A: Methodological Paper (Framework Focus)

**Target Journals:** IEEE JBHI, Artificial Intelligence in Medicine, Methods of Information in Medicine

**Focus:** Present BASICS-CDSS as a comprehensive evaluation framework

**Structure:**

### Title Options:
1. "BASICS-CDSS: A Comprehensive Evaluation Framework for Safety-Critical Clinical Decision Support Systems"
2. "Beyond Accuracy: Integrating Clinical Utility, Fairness, and Uncertainty Quantification in Medical AI Evaluation"
3. "Phase 1 Clinical Metrics for Medical AI: A Framework for FDA Compliance and Ethical Deployment"

### Abstract Template (250-300 words):

```markdown
**Background:** Current medical AI evaluation relies predominantly on accuracy metrics
(sensitivity, specificity, AUC), which fail to capture clinical utility, fairness
across demographic groups, and uncertainty quantification—critical requirements for
FDA approval and ethical deployment.

**Objective:** We developed BASICS-CDSS v2.1.0, a comprehensive evaluation framework
implementing Phase 1 Clinical Metrics essential for medical AI validation: (1) Clinical
Utility Metrics, (2) Fairness Metrics, and (3) Conformal Prediction for uncertainty
quantification.

**Methods:** The framework integrates Decision Curve Analysis for net benefit
quantification, five complementary fairness metrics (demographic parity, equalized
odds, equal opportunity, disparate impact, calibration), and distribution-free
conformal prediction with guaranteed coverage. We demonstrate the framework using
[YOUR CLINICAL APPLICATION] with [N] patients, evaluating [YOUR MODEL] across
[PROTECTED ATTRIBUTES].

**Results:** [YOUR MODEL] achieved net benefit [X.X] at clinically relevant thresholds
([X.XX]-[X.XX]), with NNT of [X.X] (95% CI: [X.X-X.X]). Fairness assessment revealed
[PASS/disparities in METRIC across GROUPS]. Conformal prediction achieved [X.X]%
empirical coverage (target: 90%) with average prediction set size [X.XX].

**Conclusions:** BASICS-CDSS provides a production-ready framework addressing critical
gaps in current medical AI evaluation. The framework supports FDA 510(k) submission
requirements, ethical AI compliance, and safe clinical deployment through rigorous
uncertainty quantification. Open-source implementation enables widespread adoption
and standardized reporting of Phase 1 Clinical Metrics.

**Keywords:** Medical AI evaluation, Decision curve analysis, Algorithmic fairness,
Conformal prediction, Clinical utility, FDA compliance
```

### Introduction (4-5 paragraphs):

```markdown
The rapid proliferation of artificial intelligence (AI) in clinical decision support
demands rigorous evaluation frameworks that extend beyond traditional performance
metrics [cite]. While sensitivity, specificity, and area under the receiver operating
characteristic curve (AUC-ROC) remain important, they fail to address three critical
questions for real-world deployment: (1) Does the model provide clinical utility
compared to alternative strategies? (2) Does the model perform equitably across
demographic groups? (3) Can the model appropriately quantify and communicate
uncertainty?

**Clinical Utility Gap**
Traditional accuracy metrics do not quantify clinical value [Vickers 2006]. A model
with 95% accuracy may provide negative net benefit if false positives lead to harmful
interventions. Decision Curve Analysis (DCA) addresses this gap by quantifying net
benefit across probability thresholds, enabling comparison to "treat all" and "treat
none" strategies [Vickers 2016]. However, DCA remains underutilized in medical AI
publications, with fewer than [X]% of recent papers reporting net benefit or Number
Needed to Treat (NNT) [cite systematic review if available].

**Fairness and Equity Gap**
Algorithmic bias in healthcare AI has emerged as a critical concern, with high-profile
cases demonstrating systematic disparities across demographic groups [Obermeyer 2019].
The FDA now requires fairness assessment for certain medical device applications,
yet standardized fairness evaluation frameworks remain limited. Multiple fairness
definitions exist (demographic parity, equalized odds, calibration), each capturing
different equity concepts, necessitating comprehensive multi-metric assessment.

**Uncertainty Quantification Gap**
Most medical AI systems provide point predictions without rigorous uncertainty
estimates, precluding safe abstention when uncertain—a critical safety feature for
clinical deployment. Conformal prediction offers distribution-free uncertainty
quantification with mathematical coverage guarantees [Vovk 2005, Angelopoulos 2021],
yet remains rarely implemented in clinical applications.

**Our Contribution**
We present BASICS-CDSS v2.1.0, an open-source framework implementing Phase 1 Clinical
Metrics for comprehensive medical AI evaluation. The framework integrates: (1) Clinical
Utility Metrics including DCA, net benefit, and NNT; (2) Five complementary fairness
metrics addressing different equity definitions; (3) Conformal prediction for guaranteed
uncertainty quantification. We demonstrate the framework using [YOUR APPLICATION],
showing how Phase 1 metrics inform deployment decisions and support FDA compliance.
All code, documentation, and visualization tools are publicly available to enable
standardized reporting and reproducible research.
```

### Methods Section:

```markdown
## 2.1 Framework Overview

BASICS-CDSS v2.1.0 implements three modules addressing distinct validation requirements:

**Module 1: Clinical Utility Assessment**
Quantifies clinical value through Decision Curve Analysis (DCA) [Vickers 2006],
net benefit calculation, and Number Needed to Treat (NNT) [Laupacis 1988].

**Module 2: Fairness Evaluation**
Implements five complementary fairness metrics addressing different equity concepts:
demographic parity (independence), equalized odds (separation), equal opportunity
(sufficiency), disparate impact (80% rule), and calibration fairness.

**Module 3: Uncertainty Quantification**
Provides distribution-free prediction sets with coverage guarantee P(Y ∈ C(X)) ≥ 1-α
using conformal prediction [Vovk 2005].

## 2.2 Clinical Utility Metrics

### 2.2.1 Decision Curve Analysis

Net benefit quantifies the clinical value of using a prediction model compared to
treating all patients or treating no patients:

$$\text{NB}(p_t) = \frac{TP}{N} - \frac{FP}{N} \times \frac{p_t}{1-p_t}$$

where $p_t$ is the probability threshold, $TP$ and $FP$ are true and false positives,
$N$ is total sample size, and $\frac{p_t}{1-p_t}$ represents the harm-to-benefit ratio
[Vickers 2006]. We compute net benefit across probability thresholds pt ∈ [0.01, 0.99]
and compare to baseline strategies:

- **Treat All:** $\text{NB}_{\text{all}}(p_t) = \text{prevalence} - (1-\text{prevalence}) \times \frac{p_t}{1-p_t}$
- **Treat None:** $\text{NB}_{\text{none}}(p_t) = 0$

The model is clinically useful when $\text{NB}_{\text{model}}(p_t) > \max(\text{NB}_{\text{all}}, \text{NB}_{\text{none}})$.

### 2.2.2 Number Needed to Treat

NNT quantifies effectiveness as the number of patients requiring treatment to prevent
one adverse event:

$$\text{NNT} = \frac{1}{\text{ARR}} = \frac{1}{|\text{CER} - \text{EER}|}$$

where ARR is absolute risk reduction, CER is control event rate (baseline), and EER
is experimental event rate (with model guidance). We compute 95% confidence intervals
using the Newcombe method [Newcombe 1998].

### 2.2.3 Clinical Impact Assessment

At each operating threshold, we quantify:
- Positive Predictive Value (PPV): Proportion of high-risk classifications that are correct
- Negative Predictive Value (NPV): Proportion of low-risk classifications that are correct
- Number Needed to Screen (NNS): Patients screened to identify one true positive

[Continue with Sections 2.3 Fairness Metrics and 2.4 Conformal Prediction...]
[Copy from MANUSCRIPT_UPDATES.md]

## 2.5 Demonstration Application

We demonstrate the framework using [DESCRIBE YOUR APPLICATION: e.g., "prediction of
30-day hospital readmission using EHR data from X hospital system, 2018-2022"].

**Dataset:** [N] patients, [M] features, [prevalence]% outcome prevalence
**Model:** [e.g., Random Forest with 100 trees, max depth 10]
**Validation:** 60% training, 20% calibration, 20% test split
**Protected Attributes:** Age group (<40, 40-60, >60), sex (M/F), race/ethnicity
(White, Black, Asian, Hispanic)

[Add your specific methods details]
```

### Results Section:

```markdown
## 3.1 Model Performance

[YOUR MODEL] achieved AUC-ROC [X.XX] (95% CI: [X.XX-X.XX]), sensitivity [X.XX],
specificity [X.XX], and accuracy [X.XX] on the test set (Table 1).

[Standard performance metrics table]

## 3.2 Clinical Utility Assessment

### 3.2.1 Decision Curve Analysis

Figure 1 presents the decision curve showing net benefit across probability thresholds.
The model provided positive net benefit for thresholds [X.XX] to [X.XX], outperforming
both "treat all" and "treat none" strategies. Maximum net benefit of [X.XXX] occurred
at threshold [X.XX], corresponding to [X.X] additional patients correctly managed per
100 evaluated.

At the clinically relevant threshold of 0.30 (appropriate for screening/triage):
- Net Benefit: [X.XXX] (vs. 0 for treat none, [X.XXX] for treat all)
- True Positive Rate: [X.XX]
- False Positive Rate: [X.XX]

### 3.2.2 Number Needed to Treat

The model achieved NNT = [X.X] (95% CI: [X.X]-[X.X]), indicating treatment of [X]
patients based on model predictions would prevent one adverse event. This compares
favorably to [benchmark if available].

Using high-risk classification at threshold 0.30:
- Control event rate (predicted low-risk): [X.XX]
- Experimental event rate (predicted high-risk): [X.XX]
- Absolute risk reduction: [X.X]%

### 3.2.3 Clinical Impact

Table 2 presents clinical impact at threshold 0.30:
- [X.X]% classified as high-risk ([Y]/[N] patients)
- PPV: [X.XXX] ([X]/[Y] high-risk predictions correct)
- NPV: [X.XXX] ([X]/[Y] low-risk predictions correct)
- NNS: [X.X] (screen [X] to find 1 true case)

[Continue with 3.3 Fairness Results and 3.4 Uncertainty Quantification...]
[Copy and fill from MANUSCRIPT_UPDATES.md]
```

### Discussion Section:

```markdown
## 4.1 Principal Findings

We developed and validated BASICS-CDSS v2.1.0, a comprehensive framework for medical
AI evaluation addressing three critical gaps: clinical utility quantification, fairness
assessment, and uncertainty quantification with guarantees. Demonstration on [YOUR
APPLICATION] showed [summary of key findings].

## 4.2 Clinical Utility and Deployment Decisions

Decision curve analysis revealed the model provides clinical utility within threshold
range [X.XX]-[X.XX], with optimal net benefit at [X.XX]. This clinically useful range
aligns with [clinical context]. The NNT of [X.X] indicates [interpretation relative
to benchmark].

Critically, traditional accuracy metrics alone would have been insufficient for
deployment decisions. Despite achieving [X]% accuracy and [X.XX] AUC-ROC, net benefit
analysis shows the model only outperforms alternatives within a specific threshold
range, highlighting the importance of DCA for clinical decision-making.

## 4.3 Fairness and Health Equity Implications

[IF PASSED ALL FAIRNESS CRITERIA:]
Comprehensive fairness assessment demonstrated equitable performance across demographic
subgroups, satisfying all five fairness criteria. This finding is particularly
important given documented disparities in healthcare AI [Obermeyer 2019, Rajkomar 2018].

[IF FAILED SOME CRITERIA:]
Fairness assessment revealed disparities in [METRIC] across [GROUPS], with [specific
finding]. This underscores the necessity of multi-metric fairness evaluation—assessing
only [one metric] would have missed these disparities. The transparency provided by
Phase 1 metrics enables proactive bias mitigation before deployment.

[Continue with mitigation strategies if needed...]

## 4.4 Uncertainty Quantification for Safe Deployment

Conformal prediction achieved [X.X]% empirical coverage, validating the theoretical
guarantee for this application. The average prediction set size of [X.XX] indicates
[excellent/good/moderate] efficiency, with [X]% singleton sets (high confidence) and
[X]% requiring additional diagnostic workup.

This "I don't know" capability is essential for clinical deployment. Systems providing
overconfident point predictions risk automation bias and diagnostic errors.

## 4.5 Implications for Medical AI Development and Regulation

Our framework directly addresses FDA requirements for medical device approval:

1. **Clinical Utility Evidence:** DCA and NNT provide quantitative clinical value assessment
2. **Fairness Documentation:** Multi-metric fairness evaluation supports equity requirements
3. **Uncertainty Quantification:** Coverage guarantees enable safe abstention policies

The open-source implementation enables standardized Phase 1 metric reporting across
studies, facilitating meta-analysis and regulatory review.

## 4.6 Comparison to Existing Frameworks

[Compare to other evaluation frameworks if relevant]

## 4.7 Limitations

[Copy from MANUSCRIPT_UPDATES.md and customize]

## 4.8 Future Directions

Phase 1 metrics address pre-deployment validation. Future work should implement
Phase 3 (temporal validation, concept drift detection) and Phase 4 (causal fairness,
multi-objective optimization) for comprehensive lifecycle monitoring.
```

---

## Paper Type B: Application Paper (Clinical Focus)

**Target Journals:** JAMA, NEJM, Lancet Digital Health, NPJ Digital Medicine

**Focus:** Clinical application with Phase 1 metrics as validation evidence

**Title Options:**
1. "Development and Validation of [DISEASE] Prediction Model: Clinical Utility and Fairness Assessment"
2. "Machine Learning for [CLINICAL TASK]: A Decision Curve Analysis"
3. "Equitable Prediction of [OUTCOME]: Fairness-Aware Medical AI for [POPULATION]"

### Abstract Template (300 words):

```markdown
**Importance:** [Clinical problem statement]

**Objective:** To develop and validate a machine learning model for [clinical task]
and assess clinical utility, fairness, and uncertainty quantification.

**Design, Setting, and Participants:** Retrospective cohort study of [N] patients
at [INSTITUTION] from [DATES]. Included patients [inclusion criteria]. Excluded
patients [exclusion criteria].

**Exposures/Interventions:** [If applicable]

**Main Outcomes and Measures:** Primary outcome was [OUTCOME]. Model performance
assessed by AUC-ROC, calibration, decision curve analysis, and fairness across
demographic groups.

**Results:** Among [N] patients (mean age [X] years, [X]% female), [Y] ([Z]%)
experienced the outcome. The model achieved AUC-ROC [X.XX] (95% CI: [X.XX-X.XX]).
Decision curve analysis revealed net benefit for probability thresholds [X.XX]-[X.XX],
with maximum benefit at threshold [X.XX]. Number needed to treat was [X.X] (95% CI:
[X.X-X.X]). Fairness assessment showed [equitable performance across / disparities in]
demographic groups [details]. Conformal prediction provided 90% coverage guarantee
with average prediction set size [X.XX].

**Conclusions and Relevance:** A machine learning model for [clinical task] demonstrated
clinical utility and [equitable/requires bias mitigation for] performance across
demographic groups. Decision curve analysis identified clinically useful threshold
ranges for deployment. Uncertainty quantification enables safe abstention when the
model is uncertain, a critical feature for clinical decision support.
```

### Key Sections for Clinical Journals:

**Methods - Study Population:**
- Clear inclusion/exclusion criteria
- Sample size justification
- IRB approval statement

**Methods - Statistical Analysis:**
- Traditional metrics PLUS Phase 1 metrics
- Subgroup analyses by demographics
- Missing data handling

**Results - Clinical Utility:**
- Decision curves with clinical interpretation
- NNT with benchmarks from literature
- Clinical impact at proposed operating point

**Results - Fairness:**
- Performance stratified by demographics (Table)
- Fairness metrics with clinical interpretation
- Discussion of equity implications

**Discussion - Clinical Implications:**
- How findings change practice
- Implementation considerations
- Cost-effectiveness implications (if available)

---

## Paper Type C: Fairness-Focused Paper

**Target Journals:** Nature Medicine, Science Translational Medicine, PLOS Medicine

**Focus:** Fairness as the primary contribution

**Title:** "Algorithmic Fairness in [APPLICATION]: Multi-Metric Assessment and Bias Mitigation"

### Abstract Template:

```markdown
Medical AI systems risk perpetuating or amplifying healthcare disparities if not
rigorously evaluated for fairness. We present a comprehensive fairness assessment of
[MODEL] for [TASK] across [PROTECTED ATTRIBUTES]. Using five complementary fairness
metrics on [N] patients, we identified [disparities/equitable performance] in [METRICS].
[IF DISPARITIES:] We implemented [mitigation strategy] achieving [improvement]. Our
multi-metric framework reveals that single-metric fairness evaluation is insufficient—
[X]% of disparities would be missed using only [common metric]. These findings
underscore the necessity of comprehensive fairness assessment for medical AI deployment.
```

### Key Sections:

**Introduction:**
- Healthcare disparities background
- AI bias literature review
- Multi-metric fairness rationale

**Methods:**
- All 5 fairness metrics with clear definitions
- Bias mitigation strategies (if applicable)
- Stakeholder engagement (if applicable)

**Results:**
- Fairness radar charts
- Stratified performance tables
- Calibration curves by group
- Disparate impact analysis

**Discussion:**
- Equity implications
- Policy recommendations
- Bias mitigation effectiveness
- Limitations of fairness definitions

---

## Paper Type D: Short Communication / Letter

**Target Journals:** JAMA, NEJM (Letters), Lancet (Correspondence)

**Length:** 600-800 words

**Template:**

```markdown
To the Editor:

Recent publications on [TOPIC] report high accuracy but omit critical clinical utility
and fairness assessments [cite 1-2 recent papers]. We demonstrate that traditional
metrics are insufficient for deployment decisions.

**Methods:** We evaluated [MODEL] for [TASK] using [N] patients, assessing clinical
utility (decision curve analysis), fairness (5 metrics across demographics), and
uncertainty (conformal prediction).

**Results:** Despite [X]% accuracy and [X.XX] AUC-ROC, decision curve analysis showed
the model only provided net benefit for thresholds [X.XX]-[X.XX]. Fairness assessment
revealed [key finding]. Number needed to treat was [X.X].

**Conclusions:** Phase 1 Clinical Metrics reveal insights missed by accuracy alone.
We urge journals to require clinical utility, fairness, and uncertainty quantification
for medical AI publications.

[Reference to BASICS-CDSS framework for reproducibility]
```

---

## Paper Type E: Validation Study

**Target Journals:** Diagnostic and Prognostic Research, BMJ, JAMIA

**Focus:** External validation with Phase 1 metrics

**Title:** "External Validation and Fairness Assessment of [MODEL] for [TASK]: A Multi-Site Study"

### Abstract Template:

```markdown
**Objective:** Externally validate [MODEL] across [K] institutions and assess fairness
across demographic groups.

**Methods:** We applied [MODEL] to [N] patients from [K] sites ([DATES]). Primary
outcome was [OUTCOME]. We assessed discrimination (AUC-ROC), calibration, clinical
utility (decision curves), fairness (5 metrics), and prediction set coverage.

**Results:** Model performance varied across sites: AUC-ROC [range]. Decision curve
analysis showed clinical utility at [X] of [K] sites. Fairness assessment revealed
[site-specific or consistent patterns]. Conformal prediction achieved [X.X]% coverage.

**Conclusions:** External validation revealed [generalizability / site-specific
performance patterns]. Fairness assessment showed [equitable / disparate] performance.
Phase 1 metrics are essential for multi-site validation studies.
```

---

## Quick Selection Guide

**Choose Paper Type Based On:**

| Primary Goal | Paper Type | Target Impact Factor |
|-------------|------------|---------------------|
| Present framework | Type A (Methodological) | 3-8 (IEEE JBHI, AIM) |
| Clinical application | Type B (Application) | 10-50 (JAMA, Lancet) |
| Address bias/equity | Type C (Fairness) | 30-60 (Nature Med) |
| Quick publication | Type D (Letter) | Variable (same as target journal) |
| Multi-site study | Type E (Validation) | 4-10 (BMJ, JAMIA) |

---

## Common Elements Across All Paper Types

### Always Include:
1. Decision curve with clinical interpretation
2. At least 2 fairness metrics
3. Conformal prediction coverage validation
4. GitHub link to BASICS-CDSS for reproducibility

### Always Cite:
- Vickers 2006, 2016 (DCA)
- Hardt 2016 (Fairness)
- Obermeyer 2019 (Healthcare bias)
- Vovk 2005 or Angelopoulos 2021 (Conformal prediction)

### Standard Figures:
- Figure 1: Decision Curve
- Figure 2: Fairness assessment (radar or stratified performance)
- Figure 3: Conformal prediction (set sizes or coverage)

---

## Next Steps

1. **Select paper type** based on your goals
2. **Copy appropriate template** from above
3. **Fill in [PLACEHOLDERS]** with your data
4. **Add your specific results** from figure generation
5. **Customize Discussion** based on your findings

All templates are designed to work with BASICS-CDSS v2.1.0 generated figures and metrics.

---

**For detailed text to copy-paste, see MANUSCRIPT_UPDATES.md**
**For metric explanations, see next document: METRIC_EXPLANATIONS.md**
**For supplementary materials, see SUPPLEMENTARY_TEMPLATES.md**
