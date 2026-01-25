# BASICS-CDSS Quick Start Guide

## 🚀 Environment Setup

### Option 1: Using the provided virtual environment (Recommended)

**Windows:**
```bash
# Activate venv
.\activate_venv.bat

# Or manually:
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
# Activate venv
source venv/bin/activate
```

### Option 2: Create new conda environment

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate basics-cdss

# Install package
pip install -e .
```

---

## ✅ Verify Installation

```bash
# Run tests
pytest tests/ -v

# Should see: 78 passed
```

---

## 📊 Quick Usage Example

```python
import numpy as np
from basics_cdss.scenario import load_archetypes_csv, instantiate_scenarios
from basics_cdss.scenario import PerturbationConfig
from basics_cdss.metrics import (
    expected_calibration_error,
    selective_prediction_metrics,
    compute_harm_metrics
)

# 1. Load archetypes (requires SynDX data)
# archetypes = load_archetypes_csv("path/to/syndx_archetypes.csv")

# 2. Generate scenarios with uncertainty
# scenarios = instantiate_scenarios(
#     archetypes,
#     n_per_archetype=10,
#     seed=42,
#     perturbation_type="composite",
#     perturbation_config=PerturbationConfig(p_mask=0.2, noise_sigma=0.1)
# )

# 3. Mock predictions for demo (replace with actual CDSS output)
n_samples = 100
y_true = np.random.randint(0, 2, n_samples)
y_prob = np.random.random(n_samples)
y_pred = (y_prob >= 0.5).astype(int)
risk_tiers = np.random.choice(["high", "medium", "low"], n_samples)

# 4. Compute beyond-accuracy metrics
ece = expected_calibration_error(y_true, y_prob, n_bins=10)
sp_metrics = selective_prediction_metrics(y_true, y_prob)
harm_metrics = compute_harm_metrics(y_true, y_pred, risk_tiers)

print(f"Expected Calibration Error: {ece:.4f}")
print(f"AURC: {sp_metrics.aurc:.4f}")
print(f"Weighted Harm Loss: {harm_metrics.weighted_harm_loss:.4f}")
print(f"Harm Concentration: {harm_metrics.harm_concentration:.2%}")
```

---

## 📓 Jupyter Notebooks

```bash
# Start JupyterLab
jupyter lab

# Open notebooks in notebooks/ directory:
# - 00_quickstart.ipynb (⚠️ Coming soon)
# - 01_basics_scenario_instantiation.ipynb (⚠️ Coming soon)
# - 02_basics_beyond_accuracy_metrics.ipynb (⚠️ Coming soon)
```

---

## 🔬 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_perturbations.py -v
pytest tests/test_metrics_calibration.py -v
pytest tests/test_metrics_coverage_risk.py -v
pytest tests/test_metrics_harm.py -v

# With coverage
pytest tests/ --cov=basics_cdss --cov-report=html
```

---

## 📦 Key Modules

### Scenario Generation
```python
from basics_cdss.scenario import (
    instantiate_scenarios,
    instantiate_stratified_scenarios,
    PerturbationConfig,
    MaskOperator,
    NoiseOperator,
    ConflictOperator,
    DegradeOperator
)
```

### Calibration Metrics
```python
from basics_cdss.metrics import (
    expected_calibration_error,
    brier_score,
    reliability_curve,
    calibration_summary
)
```

### Coverage-Risk Metrics
```python
from basics_cdss.metrics import (
    coverage_risk_curve,
    area_under_risk_coverage_curve,
    selective_prediction_metrics,
    abstention_rate
)
```

### Harm-Aware Metrics
```python
from basics_cdss.metrics import (
    weighted_harm_loss,
    harm_by_risk_tier,
    escalation_failure_analysis,
    harm_concentration_index,
    compute_harm_metrics
)
```

### Governance & Reporting
```python
from basics_cdss.governance import (
    EvaluationConfig,
    log_evaluation_run,
    generate_evaluation_report,
    export_metrics_table,
    export_calibration_plot,
    export_coverage_risk_plot
)
```

---

## 🛠️ Development Workflow

```bash
# 1. Make changes to code
# 2. Run tests
pytest tests/ -v

# 3. Format code (optional)
# black src/ tests/
# isort src/ tests/

# 4. Re-install if needed
pip install -e .
```

---

## 📚 Documentation

- **README.md**: Project overview
- **BASICS-CDSS.md**: Technical specification (machine-readable)
- **IMPLEMENTATION_SUMMARY.md**: Development summary
- **Manuscript**: `Manuscript/P4 Evaluation HIR/hir/BASICS-CDSS_Evaluation_Framework_manuscript.tex`

---

## 🆘 Troubleshooting

**Import errors:**
```bash
# Reinstall in editable mode
pip install -e .
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Test failures:**
```bash
# Check Python version (requires >=3.10)
python --version

# Update numpy to fix trapezoid warning
pip install --upgrade numpy
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

**Framework Version:** 0.1.0
**Last Updated:** 2026-01-16
