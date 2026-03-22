# BASICS-CDSS (Beyond Accuracy)

**BASICS-CDSS** (*Beyond Accuracy: Simulation-based Integrated Critical-Safety evaluation for Clinical Decision Support Systems*) is a **reproducible, simulation-based evaluation harness** for safety-critical clinical decision support.

This repository operationalizes the evaluation philosophy described in the manuscript:

> *Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems*.

It focuses on **pre-deployment behavioral safety under uncertainty** (e.g., escalation, abstention, calibration, harm-aware outcomes), and is intended as a **methodological and governance contribution**—not a claim of clinical effectiveness.

## What this repo provides
- **Archetype → Scenario instantiation** with controlled uncertainty (missingness/ambiguity/conflict)
- **Beyond-accuracy metrics**: calibration, coverage–risk, harm-aware scoring, explanation checks
- **XAI methods** (v2.0.0): SHAP analysis with game-theoretic interpretation, counterfactual explanations
- **Clinical Metrics** (v2.1.0): Decision curves, fairness assessment, conformal prediction for Medical AI validation
- **Audit-friendly artifacts**: versioned configs, seeds, and exportable evidence tables/figures
- **Synthetic-only workflow** (no patient data)

## Related Projects

### SynDX (Synthetic Data Generation)

Scenarios are instantiated from **synthetic archetypes** provided by **SynDX**:

- Repository: [ChatchaiTritham/SynDX](https://github.com/ChatchaiTritham/SynDX)
- Purpose: Privacy-preserving synthetic medical data generation for vestibular disorders
- Features: 8,400 clinical archetypes, HL7 FHIR export, differential privacy

> **Note:** SynDX is used here as a *methodological test input* for stress-testing decision behavior. It is **not** presented as a clinically validated representation of patient populations.

### BASICS-CDSS (This Repository)

This repository contains the complete implementation:

- Repository: [ChatchaiTritham/BASICS-CDSS](https://github.com/ChatchaiTritham/BASICS-CDSS)
- Package: `basics-cdss` v2.1.0
- Python: ≥3.9
- Status: Production-ready, fully tested, publication-quality

### SAFE-Gate (Clinical Triage System)

BASICS-CDSS provides evaluation methodology that can be applied to systems like **SAFE-Gate**:

- Repository: [ChatchaiTritham/SAFE-Gate](https://github.com/ChatchaiTritham/SAFE-Gate)
- Purpose: Formally verified clinical decision support for emergency triage
- Features: 6-gate parallel architecture, provable safety guarantees, 95.3% sensitivity

## Installation

### Option 1: pip (Recommended)
```bash
# Clone repository
git clone https://github.com/ChatchaiTritham/BASICS-CDSS.git
cd BASICS-CDSS

# Install in editable mode
pip install -e .

# Verify installation
python -c "from basics_cdss import metrics, visualization, clinical_metrics; print('Success!')"
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate basics-cdss
pip install -e .
```

## Quickstart

### Python API
```python
from basics_cdss.visualization import plot_reliability_diagram
import numpy as np

# Generate sample data
bin_confidences = np.linspace(0.1, 0.9, 10)
bin_accuracies = bin_confidences + np.random.normal(0, 0.05, 10)
bin_counts = np.random.randint(50, 200, 10)

# Create calibration plot
fig, ax = plot_reliability_diagram(bin_confidences, bin_accuracies, bin_counts)
fig.savefig('calibration.png', dpi=300, bbox_inches='tight')
```

### Jupyter Notebooks
```bash
jupyter lab
```

Open:
- `notebooks/00_quickstart.ipynb` (coming soon)
- `notebooks/01_basics_scenario_instantiation.ipynb` (coming soon)

## Notebooks
- **00** Quickstart end-to-end smoke run
- **01** Archetype → Scenario instantiation (Methods core)
- **02** Beyond-accuracy metrics (calibration, reliability)
- **03** Coverage–risk & abstention
- **04** Harm-aware evaluation
- **05** Explanation stability / consistency checks
- **06** Reporting pack export (tables + mapping)

## Package layout
- `basics_cdss.scenario` — archetype loader, instantiation, perturbations
- `basics_cdss.metrics` — calibration, coverage–risk, harm-aware metrics
- `basics_cdss.governance` — logging, reporting/export utilities
- `basics_cdss.xai` — SHAP analysis, counterfactual explanations (v2.0.0)
- `basics_cdss.clinical_metrics` — Clinical utility, fairness, conformal prediction (v2.1.0)
- `basics_cdss.visualization` — Publication-ready 2D/3D charts for all metrics

## Phase 1: Clinical Metrics for Medical AI (v2.1.0)

### Critical for FDA Approval and Ethical AI

**1. Clinical Utility Metrics**
- **Net Benefit & Decision Curve Analysis**: Quantify clinical value across thresholds
- **Number Needed to Treat (NNT)**: Effectiveness assessment
- **Clinical Impact Analysis**: Practical deployment implications

**2. Fairness Metrics**
- **Demographic Parity**: Equal treatment rates across groups
- **Equalized Odds**: Equal TPR/FPR across demographics
- **Disparate Impact**: 80% rule compliance
- **Calibration Fairness**: Probability accuracy per subgroup

**3. Conformal Prediction**
- **Guaranteed Coverage**: P(Y ∈ C(X)) ≥ 1 - α
- **Prediction Sets**: Distribution-free uncertainty quantification
- **Risk Control**: Calibrate thresholds for controlled FNR
- **Adaptive Efficiency**: Difficulty-based prediction sets

### Quick Start Example

```python
from basics_cdss.clinical_metrics import (
    decision_curve_analysis,
    calculate_nnt,
    fairness_report,
    split_conformal_classification
)

# Clinical Utility Assessment
dca = decision_curve_analysis(y_true, y_pred_proba)
print(f"Model useful for thresholds: {dca.threshold_range}")

nnt = calculate_nnt(y_true, y_pred)
print(f"NNT: {nnt.nnt:.1f} (treat {nnt.nnt:.0f} to prevent 1 event)")

# Fairness Audit
report = fairness_report(y_true, y_pred, y_pred_proba, race)
print(f"Overall Fair: {report.overall_fair}")

# Conformal Prediction with 90% Coverage Guarantee
conf = split_conformal_classification(
    model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1
)
print(f"Average set size: {conf.efficiency:.2f}")
```

### Generate All Clinical Metrics Figures

```bash
cd examples
python generate_clinical_metrics_figures.py --n-samples 500
```

Produces 20+ publication-ready figures (300 DPI, IEEE/Nature/JAMA compliant):
- Decision curves, net benefit analysis, NNT comparisons
- Demographic parity, equalized odds, calibration by group
- Prediction set distributions, conformal intervals, coverage validation

See [CLINICAL_METRICS_GUIDE.md](docs/CLINICAL_METRICS_GUIDE.md) for complete documentation.

## Reproducibility

All experiments are fully reproducible with:
- **Deterministic seeding**: Set random seeds for NumPy, scikit-learn
- **Version pinning**: All dependencies versioned in `requirements.txt`
- **Configuration files**: YAML configs for all experiments
- **Generated figures**: 46+ publication-ready figures (300 DPI) pre-generated in `clinical_test/` and `examples/figures/`

To regenerate all figures:
```bash
cd examples
python generate_clinical_metrics_figures.py --n-samples 500
python generate_performance_figures.py
python generate_xai_figures.py
```

## How to Cite BASICS-CDSS

If you use this framework, codebase, or evaluation protocol, please cite:

**BibTeX**:
```bibtex
@article{tritham2026basics,
  title={Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems},
  author={Tritham, Chatchai and Snae Namahoot, Chakkrit},
  journal={PeerJ Computer Science},
  year={2026},
  note={under review}
}
```

**APA**:
Tritham, C., & Snae Namahoot, C. (2026). Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems. *PeerJ Computer Science*. (under review)

This repository provides a reproducible implementation of the **BASICS-CDSS** framework for simulation-based, pre-deployment evaluation of clinical decision support systems.

## License

MIT (see `LICENSE`).
