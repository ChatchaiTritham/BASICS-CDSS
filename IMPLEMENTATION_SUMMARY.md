# BASICS-CDSS Implementation Summary

**Date:** 2026-01-16
**Status:** Core Implementation Complete (v0.1.0)
**Framework:** Beyond Accuracy Simulation-based Integrated Critical-Safety evaluation for CDSS

---

## ✅ Completed Components

### 1. **Scenario Instantiation & Perturbation System**

#### Perturbation Operators (`src/basics_cdss/scenario/perturbations.py`)
Implementation of **Table 1** from manuscript:

| Operator | Uncertainty Type | Implementation Status |
|----------|------------------|----------------------|
| **Mask** | Information missingness | ✅ Complete |
| **Noise** | Ambiguity (Gaussian) | ✅ Complete |
| **Conflict** | Internal inconsistency | ✅ Complete |
| **Degrade** | Reduced specificity | ✅ Complete |
| **Composite** | Multi-dimensional | ✅ Complete |

**Key Features:**
- Deterministic with seed control
- Preserves protected features (archetype_id, triage_tier)
- Generates quantitative uncertainty profiles
- Fully tested (24 tests passing)

#### Enhanced Instantiation (`src/basics_cdss/scenario/instantiation.py`)
Implements **Algorithm 2** from manuscript:

```python
# Instantiate scenarios with controlled uncertainty
scenarios = instantiate_scenarios(
    archetypes=df_archetypes,
    n_per_archetype=10,
    seed=42,
    perturbation_type="composite",  # or "mask", "noise", "conflict", "degrade"
    perturbation_config=PerturbationConfig(p_mask=0.2, noise_sigma=0.1)
)

# Stratified generation (baseline + all perturbation types)
stratified = instantiate_stratified_scenarios(archetypes, n_per_archetype=5)
# Returns: {"baseline": [...], "mask": [...], "noise": [...], etc.}
```

---

### 2. **Beyond-Accuracy Metrics**

#### Calibration Metrics (`src/basics_cdss/metrics/calibration.py`)
Implements **Algorithm 3** and calibration evaluation:

| Metric | Formula | Status |
|--------|---------|--------|
| **ECE** | `Σ (|B_m|/N) |conf(B_m) - acc(B_m)|` | ✅ Complete |
| **Brier Score** | `(1/N) Σ (ŷ_i - y_i)²` | ✅ Complete |
| **Reliability Curves** | Bin-wise calibration | ✅ Complete |
| **Stratified by Tier** | Per risk-tier analysis | ✅ Complete |

**Usage:**
```python
from basics_cdss.metrics import (
    expected_calibration_error,
    brier_score,
    reliability_curve,
    calibration_summary
)

# Overall calibration
ece = expected_calibration_error(y_true, y_prob, n_bins=10)
bs = brier_score(y_true, y_prob)

# Stratified calibration
summary = calibration_summary(y_true, y_prob, risk_tiers=tiers)
# Returns: {"overall": {...}, "by_risk_tier": {"high": {...}, "low": {...}}}
```

#### Coverage-Risk Metrics (`src/basics_cdss/metrics/coverage_risk.py`)
Implements selective prediction evaluation:

| Metric | Description | Status |
|--------|-------------|--------|
| **Coverage-Risk Curve** | Coverage(τ) vs Risk(τ) | ✅ Complete |
| **AURC** | Area Under Risk-Coverage Curve | ✅ Complete |
| **Abstention Rate** | Fraction of predictions withheld | ✅ Complete |
| **Stratified Analysis** | Per risk-tier selective prediction | ✅ Complete |

**Usage:**
```python
from basics_cdss.metrics import (
    selective_prediction_metrics,
    abstention_rate
)

metrics = selective_prediction_metrics(
    y_true, y_prob,
    target_coverage=0.8,  # Target 80% coverage
    target_risk=0.1       # Max 10% risk
)

print(f"AURC: {metrics.aurc:.4f}")
print(f"Risk at 80% coverage: {metrics.risk_at_coverage_threshold:.4f}")
```

#### Harm-Aware Metrics (`src/basics_cdss/metrics/harm.py`)
Implements **cost-sensitive evaluation** from manuscript:

| Metric | Formula | Status |
|--------|---------|--------|
| **Weighted Harm Loss** | `(1/N) Σ w_r_i · 𝟙[ŷ_i ≠ y_i]` | ✅ Complete |
| **Harm by Tier** | Tier-specific weighted error rates | ✅ Complete |
| **Escalation Failures** | Missed high-risk cases | ✅ Complete |
| **Harm Concentration** | Fraction of harm in high-risk tier | ✅ Complete |

**Default Harm Weights:**
- **High-risk:** 10.0 (10x baseline)
- **Medium-risk:** 3.0 (3x baseline)
- **Low-risk:** 1.0 (baseline)

**Usage:**
```python
from basics_cdss.metrics import (
    compute_harm_metrics,
    DEFAULT_HARM_WEIGHTS
)

harm_metrics = compute_harm_metrics(y_true, y_pred, risk_tiers)

print(f"Weighted harm loss: {harm_metrics.weighted_harm_loss:.4f}")
print(f"Escalation failures: {harm_metrics.escalation_failures}")
print(f"Harm concentration: {harm_metrics.harm_concentration:.2%}")
```

---

### 3. **Governance & Reproducibility**

#### Configuration Logging (`src/basics_cdss/governance/logging.py`)
Audit trail and reproducibility support:

```python
from basics_cdss.governance import (
    EvaluationConfig,
    log_evaluation_run,
    save_config,
    load_config
)

# Create configuration
config = EvaluationConfig(
    seed=42,
    n_per_archetype=10,
    perturbation_type="composite",
    calibration_bins=10,
    harm_weights={"high": 10.0, "medium": 3.0, "low": 1.0}
)

# Save for reproducibility
save_config(config, "configs/eval_001.yaml")

# Log execution
log = log_evaluation_run(
    config,
    n_scenarios=100,
    n_archetypes=10,
    execution_time=5.2,
    output_path="logs/eval_001.yaml"
)
```

#### Report Generation (`src/basics_cdss/governance/reporting.py`)
Export audit-ready artifacts:

```python
from basics_cdss.governance import (
    generate_evaluation_report,
    export_metrics_table,
    export_calibration_plot,
    export_coverage_risk_plot
)

# Generate comprehensive report
report = generate_evaluation_report(
    calibration_metrics=cal_summary,
    coverage_risk_metrics=cr_metrics,
    harm_metrics=harm_metrics,
    output_dir="results/eval_001",
    config_path="configs/eval_001.yaml",
    generate_plots=True
)

# Outputs:
# - results/eval_001/calibration_metrics.csv
# - results/eval_001/calibration_curve.png
# - results/eval_001/coverage_risk_curve.png
# - results/eval_001/REPRODUCIBILITY.yaml
```

---

## 📊 Test Coverage

**Total Tests:** 78 ✅ (100% passing)

### Test Breakdown:

| Module | Tests | Status |
|--------|-------|--------|
| **Perturbation Operators** | 24 | ✅ All passing |
| **Calibration Metrics** | 16 | ✅ All passing |
| **Coverage-Risk Metrics** | 16 | ✅ All passing |
| **Harm-Aware Metrics** | 21 | ✅ All passing |
| **Smoke Tests** | 1 | ✅ Passing |

**Test Coverage Includes:**
- Deterministic reproducibility checks
- Edge case handling (empty inputs, single values)
- Stratified analysis validation
- Default configuration tests
- Custom configuration tests

---

## 📁 Project Structure

```
BASICS-CDSS/
├── src/basics_cdss/
│   ├── scenario/
│   │   ├── __init__.py
│   │   ├── loader.py                    # Archetype CSV loading
│   │   ├── instantiation.py             # Scenario generation ✨
│   │   └── perturbations.py             # Uncertainty operators ✨
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── calibration.py               # ECE, Brier, reliability ✨
│   │   ├── coverage_risk.py             # AURC, selective prediction ✨
│   │   └── harm.py                      # Weighted harm, escalation ✨
│   └── governance/
│       ├── __init__.py
│       ├── logging.py                   # Config & audit logging ✨
│       └── reporting.py                 # Report generation ✨
├── tests/
│   ├── test_perturbations.py           # 24 tests ✅
│   ├── test_metrics_calibration.py     # 16 tests ✅
│   ├── test_metrics_coverage_risk.py   # 16 tests ✅
│   ├── test_metrics_harm.py            # 21 tests ✅
│   └── test_smoke.py                   # 1 test ✅
├── notebooks/                           # 🚧 To be developed
│   ├── 00_quickstart.ipynb
│   ├── 01_basics_scenario_instantiation.ipynb
│   ├── 02_basics_beyond_accuracy_metrics.ipynb
│   ├── 03_basics_coverage_risk_tradeoff.ipynb
│   ├── 04_basics_harm_aware_evaluation.ipynb
│   └── 05_basics_explanation_consistency.ipynb
├── pyproject.toml
├── environment.yml
├── README.md
├── BASICS-CDSS.md                      # Technical specification
├── LICENSE
└── CITATION.cff
```

✨ = Newly implemented/enhanced

---

## 🎯 Implementation Fidelity to Manuscript

### Algorithms Implemented:

- ✅ **Algorithm 1:** Scenario-based safety evaluation workflow
- ✅ **Algorithm 2:** Archetype-to-scenario instantiation with controlled uncertainty
- ✅ **Algorithm 3:** Calibration evaluation procedure (ECE, Brier, reliability curves)

### Tables Implemented:

- ✅ **Table 1:** Perturbation operators (Mask, Noise, Conflict, Degrade)
- ✅ **Table (Implicit):** Default harm weights by risk tier
- ✅ **Metrics Table:** All metrics from Table in manuscript (ECE, BS, AURC, Harm-weighted loss)

### Equations Implemented:

- ✅ ECE: `Σ (|B_m|/N) |conf(B_m) - acc(B_m)|`
- ✅ Brier Score: `(1/N) Σ (ŷ_i - y_i)²`
- ✅ Coverage: `(1/N) |{i | ŷ_i ≥ τ}|`
- ✅ Risk: `Σ ρ_i / |{i | ŷ_i ≥ τ}|`
- ✅ AURC: `∫₀¹ Risk(c) dc`
- ✅ Harm Loss: `(1/N) Σ w_r_i · 𝟙[ŷ_i ≠ y_i]`

---

## 🚀 Quick Start

### Installation:
```bash
# Create environment
conda env create -f environment.yml
conda activate basics-cdss

# Install package in editable mode
pip install -e .

# Run tests
pytest tests/ -v
```

### Basic Usage:
```python
import numpy as np
from basics_cdss.scenario import load_archetypes_csv, instantiate_scenarios
from basics_cdss.metrics import (
    expected_calibration_error,
    selective_prediction_metrics,
    compute_harm_metrics
)

# 1. Load archetypes
archetypes = load_archetypes_csv("data/syndx_archetypes.csv")

# 2. Generate scenarios with uncertainty
scenarios = instantiate_scenarios(
    archetypes,
    n_per_archetype=10,
    seed=42,
    perturbation_type="composite"
)

# 3. Evaluate system (mock predictions for demo)
y_true = np.random.randint(0, 2, len(scenarios))
y_prob = np.random.random(len(scenarios))
y_pred = (y_prob >= 0.5).astype(int)
risk_tiers = np.array([s.targets["triage_tier"] for s in scenarios])

# 4. Compute beyond-accuracy metrics
ece = expected_calibration_error(y_true, y_prob)
sp_metrics = selective_prediction_metrics(y_true, y_prob)
harm_metrics = compute_harm_metrics(y_true, y_pred, risk_tiers)

print(f"ECE: {ece:.4f}")
print(f"AURC: {sp_metrics.aurc:.4f}")
print(f"Weighted Harm: {harm_metrics.weighted_harm_loss:.4f}")
```

---

## 📝 Next Steps

### Immediate Priorities:

1. **Working Notebooks** (In Progress 🚧)
   - 00_quickstart.ipynb: End-to-end demonstration
   - 01-05: Detailed analysis notebooks

2. **SynDX Integration** (Pending)
   - Validate with real SynDX archetype data
   - Test domain-specific perturbation configurations
   - Verify clinical plausibility constraints

3. **Additional Tests** (Optional)
   - Integration tests for full pipeline
   - Performance benchmarks
   - Edge case stress testing

### Future Enhancements:

- **Explainability Module:** SHAP-style feature attribution consistency
- **Visualization Suite:** Interactive calibration/coverage-risk dashboards
- **CI/CD:** Automated testing and deployment
- **Documentation:** Sphinx-based API documentation

---

## 📚 Citation

If you use this framework, please cite:

```bibtex
@software{basics_cdss_2026,
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  title = {BASICS-CDSS: Beyond Accuracy Simulation-based Evaluation Framework},
  year = {2026},
  url = {https://github.com/ChatchaiTritham/basics-cdss},
  version = {0.1.0}
}
```

**Manuscript:**
> Tritham C, Snae Namahoot C. Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems. *Healthcare Informatics Research.* (under review).

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Implementation Completion:** ~85%
**Core Modules:** ✅ Complete
**Test Coverage:** ✅ 78 tests passing
**Documentation:** ✅ Comprehensive
**Production Ready:** Pending SynDX validation & notebook completion
