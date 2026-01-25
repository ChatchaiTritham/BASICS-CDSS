# Related Projects

**BASICS-CDSS Ecosystem**

This document provides an overview of related research projects and their integration with the BASICS-CDSS evaluation framework.

Date: 2026-01-25
Version: 1.0.0

---

## Overview

BASICS-CDSS is part of a comprehensive research ecosystem for developing and evaluating safety-critical clinical decision support systems. This ecosystem consists of three main components:

1. **SynDX**: Synthetic data generation for privacy-preserving medical AI research
2. **SAFE-Gate**: Formally verified clinical triage system with provable safety guarantees
3. **BASICS-CDSS**: Simulation-based evaluation framework for pre-deployment assessment

---

## 1. SynDX - Synthetic Data Generation

### Repository

[https://github.com/ChatchaiTritham/SynDX](https://github.com/ChatchaiTritham/SynDX)

### Purpose

SynDX is a research framework for generating synthetic medical data focused on vestibular (dizziness/vertigo) disorder diagnosis. The system creates realistic patient records without requiring actual patient data, operating on clinical guidelines and explainable AI principles.

### Key Features

- **Privacy-Preserving Generation**: Creates synthetic patients from clinical guidelines rather than real patient records
- **Clinical Knowledge Integration**: Formalizes medical guidelines into 8,400 computational archetypes
- **Explainability Focus**: Employs SHAP values, counterfactual analysis, and interpretable models
- **Healthcare Interoperability**: Exports data in HL7 FHIR, SNOMED CT, and LOINC standards
- **Multi-Phase Validation**: Statistical realism, diagnostic performance, and XAI fidelity testing
- **Differential Privacy**: Adds privacy-preserving noise to prevent memorization

### Technologies

- Python 3.9+
- TensorFlow/Keras (VAE implementation)
- Scikit-learn (NMF, classification models)
- SHAP (explainability)
- Docker (containerization)

### Core Algorithms

- **Non-negative Matrix Factorization (NMF)**: Symptom pattern decomposition
- **Variational Autoencoders (VAE)**: Latent space generation
- **Differential Privacy Mechanisms**: Privacy-preserving noise injection

### Repository Structure

```
SynDX/
├── phase1_knowledge/          # Clinical guideline formalization
├── phase2_synthesis/          # Patient generation pipeline
├── phase3_validation/         # Performance evaluation
├── notebooks/                 # Tutorial notebooks (5 total)
├── data/                      # Clinical guidelines and reference materials
├── README.md                  # Comprehensive documentation
├── CHANGELOG.md               # Version history
├── DEPLOYMENT_GUIDE.md        # Implementation instructions
└── DATASET_SUMMARY.md         # Data specifications
```

### Clinical Framework

**TiTrATE Framework** (Newman-Toker & Edlow, 2015):
- Timing: Onset pattern analysis
- Triggers: Precipitating factors
- Targeted examination: Vestibular testing

**Referenced Guidelines:**
- Bárány Society ICVD 2025 vestibular classification
- ACEP Clinical Policy on Acute Headache
- AHA/ASA Stroke Guidelines
- BPPV and Vestibular Neuritis protocols

### Relationship to BASICS-CDSS

SynDX provides the **synthetic archetype data** that BASICS-CDSS uses for scenario instantiation:

1. **Data Source**: BASICS-CDSS instantiates scenarios from SynDX archetypes
2. **Uncertainty Injection**: BASICS-CDSS applies perturbations to SynDX data
3. **Evaluation Input**: SynDX archetypes serve as methodological test inputs

**Integration Flow:**
```
SynDX Archetypes → BASICS-CDSS Perturbations → Evaluation Scenarios
```

### Important Note

> SynDX is used in BASICS-CDSS as a *methodological test input* for stress-testing decision behavior. It is **not** presented as a clinically validated representation of patient populations.

### License

MIT License

---

## 2. SAFE-Gate - Clinical Triage System

### Repository

[https://github.com/ChatchaiTritham/SAFE-Gate](https://github.com/ChatchaiTritham/SAFE-Gate)

### Purpose

SAFE-Gate is a formally verified clinical decision support system designed for triage of dizziness and vertigo presentations in emergency departments. The system achieves 95.3% sensitivity with provable safety guarantees through mathematical validation.

### Key Features

- **6-Gate Parallel Architecture**: Independent evaluations across multiple dimensions
  1. Critical Flags Detection
  2. Moderate Risk Assessment
  3. Data Quality Evaluation
  4. Clinical Decision Logic
  5. Uncertainty Quantification
  6. Temporal Pattern Analysis

- **Conservative Merging Mechanism**: Minimum lattice selection (2.5% improvement over ensemble averaging)

- **Explicit Abstention**: Flags uncertain cases for mandatory physician review

- **Real-Time Performance**: ~1.23ms mean latency for clinical deployment

- **Formal Verification**: Zero violations across 6,400 test cases

### Technologies

- Python 3.8+
- XGBoost for risk scoring
- scikit-learn for machine learning components
- NumPy and pandas for data processing
- Monte Carlo dropout for uncertainty quantification
- Jupyter notebooks for reproducibility

### System Architecture

```
Input Patient Data
       ↓
┌──────────────────────────────────────┐
│   6 Parallel Gates (Independent)     │
├──────────────────────────────────────┤
│ Gate 1: Critical Flags               │
│ Gate 2: Moderate Risk                │
│ Gate 3: Data Quality                 │
│ Gate 4: Clinical Logic               │
│ Gate 5: Uncertainty Quantification   │
│ Gate 6: Temporal Patterns            │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  Conservative Merging (Min Lattice)  │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  Risk Tier Assignment (R1-R5)        │
│  + Abstention Decision               │
└──────────────────────────────────────┘
       ↓
  Clinical Action
```

### Risk Lattice

```
R5 (Critical)  ← Immediate intervention
R4 (High)      ← Urgent evaluation
R3 (Moderate)  ← Prompt assessment
R2 (Low)       ← Standard workup
R1 (Minimal)   ← Observation
```

**Conservative Merging**: Selects minimum (most conservative) tier from 6 gates

### Performance Metrics

- **Sensitivity**: 95.3%
- **Specificity**: 87.2%
- **Abstention Rate**: 12.4% (flagged for physician review)
- **Mean Latency**: 1.23ms
- **Formal Verification**: 100% (6,400/6,400 test cases)

### Repository Structure

```
SAFE-Gate/
├── src/
│   ├── safegate.py            # Main system
│   ├── gates/                 # Six parallel evaluation modules
│   ├── merging/               # Risk lattice and conservative merging
│   └── theorems/              # Formal verification
├── data/                      # Synthetic datasets (SynDX)
├── notebooks/
│   └── 00_quickstart.ipynb    # Reproducibility notebook
├── README.md                  # Documentation
└── LICENSE                    # MIT License
```

### Formal Verification

SAFE-Gate implements formal theorem verification:

1. **Safety Theorem**: No high-risk case assigned to low-risk tier
2. **Monotonicity Theorem**: Risk tier never decreases with more concerning features
3. **Completeness Theorem**: All inputs assigned to valid risk tier or abstention
4. **Determinism Theorem**: Same input produces same output (reproducibility)

### Data Source

SAFE-Gate uses **SynDX methodology** for synthetic data generation:
- Counterfactual reasoning across risk tier boundaries
- Negative matrix factorization for symptom consistency
- Differential privacy for patient protection

### Relationship to BASICS-CDSS

BASICS-CDSS provides the **evaluation methodology** that can be applied to systems like SAFE-Gate:

1. **Evaluation Target**: SAFE-Gate can be evaluated using BASICS-CDSS framework
2. **Beyond-Accuracy Metrics**: BASICS-CDSS assesses calibration, harm-awareness, coverage-risk
3. **Scenario Testing**: BASICS-CDSS perturbs SynDX data to stress-test SAFE-Gate behavior
4. **Pre-Deployment Assessment**: BASICS-CDSS validates SAFE-Gate before clinical deployment

**Evaluation Flow:**
```
SAFE-Gate System → BASICS-CDSS Evaluation → Safety Assessment Report
```

### Publications

**IEEE EMBC 2026**: Conference submission
- Authors: Chatchai Tritham, Chakkrit Snae Namahoot
- Title: *SAFE-Gate: Formally Verified Clinical Decision Support for Emergency Triage*
- Status: Under review

### License

MIT License

---

## 3. Integration Architecture

### Complete Research Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                        Research Ecosystem                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │    SynDX     │────→ │  SAFE-Gate   │────→ │ BASICS-CDSS  │  │
│  │              │      │              │      │              │  │
│  │ Synthetic    │      │ Clinical     │      │ Evaluation   │  │
│  │ Data         │      │ Triage       │      │ Framework    │  │
│  │ Generation   │      │ System       │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│        ↓                      ↓                      ↓          │
│  Privacy-Preserving    Provable Safety     Pre-Deployment      │
│  Archetypes            Guarantees          Validation           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **SynDX → SAFE-Gate**: Synthetic archetypes for training and testing
2. **SynDX → BASICS-CDSS**: Archetypes for scenario instantiation
3. **SAFE-Gate → BASICS-CDSS**: System predictions for evaluation

### Evaluation Pipeline

```
┌─────────────────┐
│ SynDX Archetypes│
│ (8,400 cases)   │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────┐
│ BASICS-CDSS Scenario Generation │
│ - Perturbations (mask/noise)    │
│ - Uncertainty injection         │
│ - Stratified sampling           │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────────────────────┐
│ SAFE-Gate System Evaluation     │
│ - 6-gate parallel processing    │
│ - Conservative merging          │
│ - Risk tier assignment          │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────────────────────┐
│ BASICS-CDSS Metrics Analysis    │
│ - Calibration (ECE, Brier)      │
│ - Coverage-Risk (AURC)          │
│ - Harm-Aware (weighted loss)    │
│ - Performance (ROC, PR, F1)     │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────────────────────┐
│ Safety Assessment Report        │
│ - Beyond-accuracy evaluation    │
│ - Stratified analysis           │
│ - Audit-ready artifacts         │
└─────────────────────────────────┘
```

---

## 4. Research Publications

### Submitted Papers

1. **SynDX Framework**
   - Preprint: arXiv (doi pending)
   - Target: Medical informatics journal
   - Focus: Privacy-preserving synthetic data generation

2. **SAFE-Gate System**
   - Conference: IEEE EMBC 2026
   - Authors: Chatchai Tritham, Chakkrit Snae Namahoot
   - Focus: Formally verified clinical triage

3. **BASICS-CDSS Framework**
   - Journal: Healthcare Informatics Research (under review)
   - Authors: Chatchai Tritham, Chakkrit Snae Namahoot
   - Focus: Simulation-based evaluation methodology

### Planned Publications (4 Papers)

1. **Paper 1**: Digital Twin Simulation
   - Target: Journal of Biomedical Informatics (Q1, IF: 8.0)
   - Tier 1 methodology

2. **Paper 2**: Causal Simulation
   - Target: Nature Machine Intelligence (Q1, IF: 25.8)
   - Tier 2 methodology

3. **Paper 3**: Multi-Agent Simulation
   - Target: JAMIA (Q1, IF: 7.9)
   - Tier 3 methodology

4. **Paper 4**: Integrated Framework
   - Target: Nature Medicine (Q1, IF: 87.2)
   - Complete 3-tier framework

---

## 5. Use Cases

### Use Case 1: Privacy-Preserving Development

**Problem**: Need to develop CDSS without accessing real patient data

**Solution:**
1. Use SynDX to generate synthetic archetypes
2. Develop SAFE-Gate on synthetic data
3. Validate with BASICS-CDSS evaluation

### Use Case 2: Pre-Deployment Validation

**Problem**: Need to validate CDSS safety before clinical deployment

**Solution:**
1. Generate evaluation scenarios from SynDX archetypes
2. Apply BASICS-CDSS perturbations for stress testing
3. Assess SAFE-Gate using beyond-accuracy metrics
4. Generate audit-ready safety report

### Use Case 3: Comparative Evaluation

**Problem**: Need to compare multiple CDSS systems

**Solution:**
1. Use standardized SynDX archetypes
2. Evaluate all systems using BASICS-CDSS framework
3. Compare beyond-accuracy metrics (calibration, harm-awareness)
4. Generate comparative performance reports

### Use Case 4: Regulatory Submission

**Problem**: Need evidence for regulatory approval

**Solution:**
1. Document SynDX data generation process (differential privacy)
2. Demonstrate SAFE-Gate formal verification (theorems)
3. Provide BASICS-CDSS evaluation artifacts (audit trails)
4. Submit comprehensive safety assessment

---

## 6. Technical Integration

### Environment Setup

```bash
# Clone all repositories
git clone https://github.com/ChatchaiTritham/SynDX.git
git clone https://github.com/ChatchaiTritham/SAFE-Gate.git
git clone https://github.com/ChatchaiTritham/BASICS-CDSS.git

# Install SynDX
cd SynDX
conda env create -f environment.yml
conda activate syndx
pip install -e .

# Install SAFE-Gate
cd ../SAFE-Gate
conda env create -f environment.yml
conda activate safegate
pip install -e .

# Install BASICS-CDSS
cd ../BASICS-CDSS
conda env create -f environment.yml
conda activate basics-cdss
pip install -e .
```

### Integration Code Example

```python
# 1. Generate synthetic archetypes using SynDX
from syndx import generate_archetypes

archetypes = generate_archetypes(
    n_archetypes=100,
    guideline_source='TiTrATE',
    privacy_epsilon=1.0
)

# 2. Evaluate SAFE-Gate using BASICS-CDSS
from basics_cdss.scenario import instantiate_scenarios
from basics_cdss.metrics import compute_performance_metrics
from safegate import SAFEGate

# Instantiate scenarios with perturbations
scenarios = instantiate_scenarios(
    archetypes=archetypes,
    n_per_archetype=10,
    perturbation_type='composite',
    seed=42
)

# Run SAFE-Gate predictions
safegate = SAFEGate()
predictions = safegate.predict(scenarios)

# Evaluate with BASICS-CDSS metrics
y_true = [s.targets['triage_tier'] for s in scenarios]
y_pred = predictions['tier']
y_prob = predictions['probability']

metrics = compute_performance_metrics(y_true, y_pred, y_prob)

print(f"F1-Score: {metrics.f1_score:.3f}")
print(f"ROC-AUC: {metrics.roc_auc:.3f}")
```

---

## 7. Contact Information

### Research Team

#### Principal Investigator (PhD Candidate)

**Chatchai Tritham**

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- Department: Computer Science and Information Technology
- Faculty: Science
- Institution: Naresuan University
- Location: Phitsanulok 65000, Thailand

**Research Interests:**
- Clinical decision support systems
- Medical AI safety and evaluation
- Formal verification in healthcare
- Privacy-preserving synthetic data

#### Supervisor

**Chakkrit Snae Namahoot**

- Email: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- Department: Computer Science and Information Technology
- Faculty: Science
- Institution: Naresuan University
- Location: Phitsanulok 65000, Thailand

**Research Interests:**
- Healthcare informatics
- Medical data mining
- Clinical decision support
- Knowledge-based systems

### Collaboration

We welcome collaboration and feedback from:

- Healthcare AI researchers
- Clinical informaticians
- Regulatory bodies
- Healthcare institutions
- Medical device companies

For inquiries about:

- **SynDX**: Privacy-preserving data generation
- **SAFE-Gate**: Clinical triage system deployment
- **BASICS-CDSS**: Evaluation methodology application
- **Research Collaboration**: Joint projects or publications

Please contact: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)

---

## 8. Licenses

All three projects use **MIT License**:

- **SynDX**: MIT License
- **SAFE-Gate**: MIT License
- **BASICS-CDSS**: MIT License

This permissive license allows:

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

Requirements:

- ⚠️ License and copyright notice must be included
- ⚠️ Disclaimer of warranty

**Important Disclaimer:**

> All three repositories contain **research code** for methodological development and academic validation. They are **not** production medical software and lack clinical validation for patient care. Use in real-world clinical settings requires additional regulatory approval and clinical trials.

---

## 9. Future Development

### Planned Features

#### SynDX v2.0
- Multi-condition support (beyond vestibular disorders)
- Temporal disease progression modeling
- Enhanced differential privacy mechanisms

#### SAFE-Gate v2.0
- Extended risk tier classification (R1-R10)
- Multi-modal data support (imaging, lab results)
- Real-time monitoring integration

#### BASICS-CDSS v2.0
- Automated benchmark suite
- Interactive evaluation dashboards
- Regulatory submission templates

### Roadmap

**Q1 2026:**
- Complete IEEE EMBC 2026 publication (SAFE-Gate)
- Submit Healthcare Informatics Research paper (BASICS-CDSS)
- Release SynDX preprint on arXiv

**Q2-Q3 2026:**
- Submit 4 journal papers (Digital Twin, Causal, Multi-Agent, Integrated)
- Develop comprehensive benchmark suite
- Clinical pilot study planning

**Q4 2026:**
- Version 2.0 releases for all three projects
- Clinical validation studies
- Regulatory pathway exploration

---

## 10. Acknowledgments

### Funding

This research is supported by:

- Naresuan University
- Faculty of Science, Naresuan University

### Clinical Guidelines

We acknowledge the following clinical guideline sources:

- Bárány Society ICVD 2025
- ACEP Clinical Policies
- AHA/ASA Stroke Guidelines
- TiTrATE Framework (Newman-Toker & Edlow, 2015)

### Open Source Community

We thank the open-source community for:

- Python scientific computing ecosystem
- TensorFlow and scikit-learn frameworks
- Matplotlib and seaborn visualization libraries

---

**Last Updated**: 2026-01-25
**Version**: 1.0.0
**Author**: Chatchai Tritham
**Supervisor**: Chakkrit Snae Namahoot
