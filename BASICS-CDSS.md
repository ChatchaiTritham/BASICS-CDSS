BASICS-CDSS: Beyond Accuracy Simulation-based Evaluation Framework for Safety-Critical Clinical Decision Support Systems

1. Purpose of This Document

This document specifies the complete analytical, design, and development blueprint of BASICS-CDSS (Beyond Accuracy: Simulation-based Integrated Critical-Safety evaluation for Clinical Decision Support Systems).

It is intentionally written as a machine-readable yet academically grounded system specification to enable:

Faithful implementation of Python modules

Reproduction of evaluation logic through executable notebooks

Preservation of methodological intent beyond mere code behavior

This document should be treated as the single source of truth for implementation.

2. Conceptual Positioning

2.1 What BASICS-CDSS Is

BASICS-CDSS is a simulation-based evaluation and governance framework for safety-critical Clinical Decision Support Systems (CDSS).

It is:

❌ Not a diagnostic AI

❌ Not a predictive performance benchmark

❌ Not a clinical effectiveness study

It is:

✅ A pre-deployment evaluation framework

✅ A decision-governance paradigm

✅ A methodology to assess how systems behave under uncertainty

2.2 Core Principle: Beyond Accuracy

Traditional evaluation asks:

"How accurate is the model?"

BASICS-CDSS asks:

"How does the system behave when the decision is uncertain, risky, or ambiguous?"

Accuracy is treated as necessary but insufficient.

3. Governance-First Design Philosophy

3.1 Uncertainty as a Control Signal

BASICS-CDSS explicitly models uncertainty and uses it to govern decisions.

Sources of uncertainty include:

Incomplete symptom reporting

Ambiguous clinical presentations

Conflicting evidence across features

Variability in clinician expertise

Uncertainty determines:

Escalation vs deferral

Abstention vs recommendation

Urgent vs non-urgent pathways

3.2 Decision-Governance Over Prediction

The framework evaluates:

Whether the system knows when not to decide

Whether escalation aligns with safety policies

Whether confidence is calibrated to risk

4. Data Foundation: Synthetic-Only, Archetype-Driven

4.1 Rationale for Synthetic Data

BASICS-CDSS never uses real patient data at this stage.

Reasons:

Avoid privacy and ethical risks

Eliminate hidden expert bias

Enable full transparency and reproducibility

4.2 SynDX Archetypes

Input data are clinical archetypes from SynDX:

Each archetype represents a logic-consistent clinical scenario

Archetypes define:

Symptom structures

Risk tiers

Expected decision categories

Archetypes are not patients.

5. Scenario Instantiation Engine

5.1 Purpose

Transform archetypes into multiple test scenarios that simulate uncertainty.

5.2 Scenario Generation Dimensions

Each archetype can be expanded along:

Missingness (removed features)

Ambiguity (overlapping symptom signals)

Conflict (contradictory cues)

Noise (measurement perturbation)

5.3 Deterministic Reproducibility

All scenario generation must:

Use fixed random seeds

Log parameters

Be fully replayable

6. Evaluation Targets (What Is Evaluated)

BASICS-CDSS evaluates decision behavior, not disease prediction.

Targets include:

Triage tier

Action category (e.g., escalate, defer, abstain)

Confidence level

Explanation stability

7. Metric Families (Beyond Accuracy)

7.1 Calibration Metrics

Expected Calibration Error (ECE)

Reliability curves

7.2 Coverage–Risk Metrics

Selective prediction curves

Abstention rate vs harm

7.3 Harm-Aware Metrics

Weighted penalty for unsafe decisions

Escalation failure cost

7.4 Consistency Metrics

Decision stability under perturbation

Explanation coherence across similar scenarios

8. Explainability-by-Design

Explainability is not post-hoc.

BASICS-CDSS requires:

Feature-attribution explanations (e.g., SHAP-style)

Factor-level structure (e.g., NMF-like decomposition)

Counterfactual sensitivity analysis

Explanations are evaluated for:

Stability

Alignment with decision logic

Governance consistency

9. Evaluation Workflow (Canonical Pipeline)

Load archetypes

Instantiate scenarios

Apply CDSS decision logic (external system)

Collect outputs

Compute beyond-accuracy metrics

Generate governance reports

No step may be skipped.

10. Reporting & Audit Artifacts

The framework must produce:

Metric tables (CSV)

Plots (PDF/PNG)

Configuration logs (YAML/JSON)

Reproducibility manifests

Outputs must be suitable for:

Regulatory review

Editorial inspection

Independent replication

11. Python Package Architecture (Required)

basics_cdss/
  scenario/
    loader.py
    instantiation.py
  metrics/
    calibration.py
    coverage_risk.py
    harm.py
  governance/
    logging.py
    reporting.py

Each module must:

Be deterministic

Be unit-testable

Avoid domain hard-coding

12. Notebook Specification

Required notebooks:

01_basics_scenario_instantiation.ipynb

02_basics_beyond_accuracy_metrics.ipynb

03_basics_coverage_risk_tradeoff.ipynb

04_basics_harm_aware_evaluation.ipynb

05_basics_explanation_consistency.ipynb

Each notebook must:

Run top-to-bottom

Save outputs

Contain minimal narrative

13. Ethical & Governance Constraints

Must explicitly state:

No patient data

No clinical effectiveness claims

No superiority claims

Language must remain evaluation- and governance-focused.

14. Intended Extension Path

BASICS-CDSS is designed to support:

Retrospective validation

Expert-in-the-loop studies

Prospective trials (future)

These are outside the scope of the current implementation.

15. Final Instruction to Implementing AI

When implementing BASICS-CDSS:

Preserve the decision-governance intent

Do not optimize for predictive performance

Prioritize transparency and reproducibility

Failure to follow these principles constitutes non-compliance with BASICS-CDSS.

