# BASICS-CDSS Visualization System Guide

**Complete guide to generating publication-ready figures for all tiers**

Last Updated: January 25, 2026
Version: 1.0.0

---

## Overview

The BASICS-CDSS visualization system provides **26 publication-ready figures** across all three simulation tiers plus baseline evaluation metrics. All figures comply with:

- ✅ **IEEE publication standards** (300 DPI, vector formats)
- ✅ **Nature/JAMA figure requirements** (Times New Roman, proper sizing)
- ✅ **Journal-specific guidelines** (JBI, JAMIA, Nature MI, etc.)
- ✅ **Colorblind-friendly palettes** (Paul Tol's vibrant scheme)
- ✅ **Publication formats**: PDF (vector), EPS (vector), PNG (300 DPI raster)

---

## Quick Start

### Generate All Figures (Recommended)

```bash
cd D:\PhD\Manuscript\GitHub\BASICS-CDSS
python examples/publication_figures.py --tier all
```

**Output**: 26 figures in `examples/figures/` organized by tier:
- `baseline/` - 12 core evaluation figures
- `tier1/` - 4 digital twin figures
- `tier2/` - 5 causal simulation figures
- `tier3/` - 5 multi-agent figures

### Generate Specific Tier Only

```bash
# Baseline only (12 figures)
python examples/publication_figures.py --tier baseline

# Tier 1 only (4 figures)
python examples/publication_figures.py --tier 1

# Tier 2 only (5 figures)
python examples/publication_figures.py --tier 2

# Tier 3 only (5 figures)
python examples/publication_figures.py --tier 3
```

### Custom Output Directory

```bash
python examples/publication_figures.py --tier all --output-dir my_figures
```

---

## Figure Catalog

### Baseline Evaluation Figures (12 figures)

Used in: **All papers** (core metrics)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| **fig01_reliability_diagram.png** | Calibration reliability curve | Model confidence alignment |
| **fig02_stratified_calibration.png** | Calibration by risk tier | Tier-specific calibration |
| **fig03_calibration_comparison.png** | Multi-model calibration | Comparative analysis |
| **fig04_coverage_risk.png** | Coverage-risk trade-off | Selective prediction curve |
| **fig05_selective_prediction.png** | Model comparison | Abstention performance |
| **fig06_abstention_analysis.png** | Abstention behavior | Uncertainty-risk relationship |
| **fig07_harm_by_tier.png** | Weighted harm distribution | Harm concentration |
| **fig08_escalation_analysis.png** | Escalation patterns | Safety-critical failures |
| **fig09_harm_concentration.png** | Harm Lorenz curve | Inequality in harm |
| **fig10_metric_comparison.png** | Multi-metric comparison | Aggregate performance |
| **fig11_radar_comparison.png** | Radar chart | Visual performance profile |
| **fig12_evaluation_dashboard.png** | Integrated dashboard | Comprehensive overview |

---

### Tier 1: Digital Twin Figures (4 figures)

Used in: **Paper 1** (Digital Twin Simulation for Temporal Evaluation)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| **fig01_temporal_trajectory.png** | 24-hour vital sign trajectory | Patient state evolution with interventions |
| **fig02_disease_progression.png** | Disease biomarker evolution | Sepsis progression stages |
| **fig03_counterfactual_analysis.png** | Intervention counterfactuals | What-if scenario comparison |
| **fig04_intervention_timing.png** | Optimal timing window | Time-dependent intervention effects |

**Target Journal**: Journal of Biomedical Informatics (Q1)

**Figure Format**: 7.0 × 11.0 inches (double-column), PDF/EPS + PNG

---

### Tier 2: Causal Simulation Figures (5 figures)

Used in: **Paper 2** (Causal Inference Framework for CDSS Evaluation)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| **fig01_causal_dag.png** | Causal directed acyclic graph | Structural causal relationships |
| **fig02_intervention_effects.png** | Average treatment effects (ATE) | Intervention efficacy with CI |
| **fig03_cate_heterogeneity.png** | Conditional ATE (CATE) | Subgroup effect heterogeneity |
| **fig04_confounding_analysis.png** | Confounding bias quantification | Bias from unmeasured confounders |
| **fig05_backdoor_adjustment.png** | Backdoor adjustment strategy | Confounder identification |

**Target Journal**: Nature Machine Intelligence / JMLR (Q1)

**Figure Format**: 7.0 × 7.0 inches (square for DAGs), PDF/EPS + PNG

---

### Tier 3: Multi-Agent Figures (5 figures)

Used in: **Paper 3** (Multi-Agent Simulation for System-Level Effects)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| **fig01_agent_network.png** | Agent interaction network | Communication patterns |
| **fig02_workflow_timeline.png** | Clinical workflow Gantt chart | Sepsis bundle timeline |
| **fig03_alert_fatigue.png** | Alert fatigue dynamics (3 panels) | Fatigue progression over shift |
| **fig04_override_rates.png** | Clinician override patterns | Experience-based differences |
| **fig05_system_resilience.png** | Performance under stress | Workload-performance relationship |

**Target Journal**: JAMIA / npj Digital Medicine (Q1)

**Figure Format**: 7.0 × 6.0-9.0 inches (varies by subplot count), PDF/EPS + PNG

---

## Module Structure

```
basics_cdss/visualization/
├── __init__.py                   # Exports all plotting functions
├── calibration_plots.py          # Calibration & reliability diagrams
├── coverage_risk_plots.py        # Coverage-risk & selective prediction
├── harm_plots.py                 # Harm-aware evaluation plots
├── scenario_plots.py             # Uncertainty & perturbation plots
├── comparison_plots.py           # Multi-model comparison plots
├── temporal_plots.py             # 🆕 Tier 1: Digital twin temporal analysis
├── causal_plots.py               # 🆕 Tier 2: Causal DAGs & interventions
└── multiagent_plots.py           # 🆕 Tier 3: Multi-agent system dynamics
```

---

## API Reference

### Baseline Evaluation

#### Calibration Plots

```python
from basics_cdss.visualization import (
    plot_reliability_diagram,
    plot_stratified_calibration,
    plot_calibration_comparison
)

# Example: Reliability diagram
fig, ax = plot_reliability_diagram(
    bin_confidences=[0.1, 0.3, 0.5, 0.7, 0.9],
    bin_accuracies=[0.12, 0.28, 0.52, 0.68, 0.88],
    bin_counts=[50, 100, 150, 100, 50],
    title="Model Calibration Reliability"
)
fig.savefig('calibration.pdf', dpi=300, bbox_inches='tight')
```

#### Coverage-Risk Plots

```python
from basics_cdss.visualization import plot_coverage_risk_curve

# Example: Coverage-risk trade-off
fig, ax = plot_coverage_risk_curve(
    coverages=np.linspace(0, 1, 50),
    risks=0.3 * (1 - coverages) ** 0.5,
    title="Coverage-Risk Trade-off",
    highlight_points=[(0.8, "Target 80%")]
)
```

#### Harm-Aware Plots

```python
from basics_cdss.visualization import (
    plot_harm_by_tier,
    plot_escalation_analysis,
    plot_harm_concentration
)

# Example: Harm by tier
fig, ax = plot_harm_by_tier(
    harm_by_tier={"R1-R2": 3.5, "R3": 1.8, "R4": 0.6, "R5": 0.2},
    title="Weighted Harm Distribution"
)
```

---

### Tier 1: Digital Twin

```python
from basics_cdss.visualization import (
    plot_temporal_trajectory,
    plot_disease_progression,
    plot_counterfactual_analysis,
    plot_intervention_timing_analysis
)

# Example: Temporal trajectory
time = np.linspace(0, 24, 100)
vitals = {
    'heart_rate': 120 + 15*np.sin(time/4),
    'systolic_bp': 85 + 10*np.random.randn(100),
    'spo2': 88 + 3*np.random.randn(100)
}
interventions = [(6.0, 'Fluid Resuscitation'), (12.0, 'Antibiotic')]
risk_tiers = np.ones(100) + (time/8).astype(int).clip(0, 4)

fig, axes = plot_temporal_trajectory(
    time, vitals, interventions, risk_tiers,
    title="Patient Digital Twin: 24-Hour Trajectory"
)
```

---

### Tier 2: Causal Simulation

```python
from basics_cdss.visualization import (
    plot_causal_dag,
    plot_intervention_effects,
    plot_cate_heterogeneity,
    plot_confounding_analysis,
    plot_backdoor_adjustment
)
import networkx as nx

# Example: Causal DAG
G = nx.DiGraph()
G.add_edges_from([
    ('Age', 'Mortality'),
    ('Antibiotic', 'Mortality'),
    ('Comorbidity', 'Antibiotic'),
    ('Comorbidity', 'Mortality')
])
node_types = {
    'Antibiotic': 'treatment',
    'Mortality': 'outcome',
    'Age': 'confounder',
    'Comorbidity': 'confounder'
}

fig, ax = plot_causal_dag(G, node_types,
                          title="Causal DAG: Treatment & Mortality")
```

```python
# Example: Intervention effects (ATE)
ate_results = {
    'Early Antibiotic (<3h)': {
        'ate': -0.15,
        'ci_lower': -0.22,
        'ci_upper': -0.08,
        'p_value': 0.001
    },
    'Fluid Resuscitation': {
        'ate': -0.08,
        'ci_lower': -0.14,
        'ci_upper': -0.02,
        'p_value': 0.012
    }
}

fig, ax = plot_intervention_effects(
    ate_results,
    title="Average Treatment Effects on Mortality"
)
```

---

### Tier 3: Multi-Agent Simulation

```python
from basics_cdss.visualization import (
    plot_agent_interaction_network,
    plot_workflow_timeline,
    plot_alert_fatigue_dynamics,
    plot_override_rates_comparison,
    plot_system_resilience
)

# Example: Agent interaction network
interactions = [
    ('CDSS_Sepsis', 'Clinician_A', 'alert', 15),
    ('Clinician_A', 'Patient_1', 'intervention', 10),
    ('Nurse_1', 'Patient_1', 'monitoring', 20)
]
agent_types = {
    'CDSS_Sepsis': 'cdss',
    'Clinician_A': 'clinician',
    'Patient_1': 'patient',
    'Nurse_1': 'nurse'
}

fig, ax = plot_agent_interaction_network(
    interactions, agent_types,
    title="Agent Interaction Network (24h Simulation)"
)
```

```python
# Example: Alert fatigue dynamics
time = np.arange(0, 24, 1)
alert_counts = 10 + 5*np.sin(time/4) + np.random.randint(0, 3, 24)
override_rates = 0.2 + 0.3*time/24 + np.random.normal(0, 0.05, 24)
response_times = 5 + 10*time/24 + np.random.normal(0, 1, 24)

fig, axes = plot_alert_fatigue_dynamics(
    time, alert_counts, override_rates, response_times,
    title="Alert Fatigue Evolution (24-Hour Shift)"
)
```

---

## Publication Guidelines

### Figure Size Standards

**Single-column figures** (IEEE/JAMA):
- Width: 3.5 inches (8.9 cm)
- Height: Variable (max 9 inches)

**Double-column figures** (IEEE/Nature):
- Width: 7.0-7.16 inches (18.2 cm)
- Height: 6-11 inches (varies by content)

**Default for BASICS-CDSS**:
- All figures: **7.0 inches wide** (double-column)
- Height: **6-11 inches** (optimized per figure type)

### Resolution Requirements

- **Vector formats**: PDF, EPS (preferred for publication)
- **Raster format**: PNG at 300 DPI minimum
- **Font**: Times New Roman (serif)
- **Line width**: 2.0 pt (main lines), 0.8 pt (axes)
- **Marker size**: 150-2000 (context-dependent)

### Color Palettes

**Colorblind-friendly scheme** (all figures):
```python
COLORS = {
    'critical': '#CC3311',      # Red
    'high_risk': '#EE7733',     # Orange
    'moderate': '#EE9955',      # Light orange
    'low_risk': '#0077BB',      # Blue
    'safe': '#33BB55',          # Green
    'intervention': '#9933CC',  # Purple
    'baseline': '#666666',      # Gray
}
```

**Rationale**: Accessible to deuteranopia, protanopia, and tritanopia.

---

## Integration with LaTeX

### Including Figures in Manuscript

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\textwidth]{figures/tier1/fig01_temporal_trajectory.pdf}
\caption{Patient digital twin temporal trajectory over 24 hours. (a) Heart rate evolution
showing tachycardia with intervention response. (b) Systolic blood pressure demonstrating
hemodynamic instability. (c) SpO2 levels indicating hypoxemia. (d) Risk tier evolution
from R1 (critical) to R3 (moderate) following interventions.}
\label{fig:temporal_trajectory}
\end{figure}
```

### Referencing in Text

```latex
As shown in Fig.~\ref{fig:temporal_trajectory}(a), the patient's heart rate remained
elevated above 120 bpm for the first 6 hours despite fluid resuscitation at hour 6.
Following antibiotic administration at hour 12, hemodynamic stabilization occurred,
evidenced by decreasing heart rate and improving blood pressure (Fig.~\ref{fig:temporal_trajectory}(b)).
```

---

## Customization

### Modify Figure Appearance

All figures accept standard matplotlib parameters:

```python
fig, ax = plot_temporal_trajectory(
    time, vitals, interventions, risk_tiers,
    title="Custom Title",
    figsize=(10, 8),  # Custom size
    save_path="custom_output.png"
)

# Further customization
ax.set_xlabel("Custom X Label")
fig.savefig("final_figure.pdf", dpi=600)  # Higher DPI
```

### Change Color Scheme

Edit color definitions in individual plot modules:

```python
# In temporal_plots.py, causal_plots.py, or multiagent_plots.py
COLORS = {
    'critical': '#YOUR_COLOR',  # Customize
    # ...
}
```

---

## Troubleshooting

### Common Issues

**Issue 1: Fonts not found**
```
UserWarning: Glyph missing from font(s)
```
**Solution**: Install Times New Roman or use default serif:
```python
plt.rcParams['font.serif'] = ['DejaVu Serif']  # Fallback
```

**Issue 2: Overlapping text**
```python
# Increase spacing
plt.subplots_adjust(hspace=0.40)  # More vertical space
```

**Issue 3: File size too large (PDF)**
```python
# Reduce DPI or simplify plots
fig.savefig('figure.pdf', dpi=300)  # Lower DPI
```

**Issue 4: Import errors**
```python
# Ensure package installed in editable mode
pip install -e .
```

---

## Future Enhancements

### Planned Features (v1.1.0)

1. **Interactive figures** (Plotly integration)
2. **Animated trajectories** (for presentations)
3. **Statistical annotations** (auto p-values, effect sizes)
4. **Journal-specific templates** (Nature, JAMA, IEEE presets)
5. **Batch processing** (process multiple datasets)

### Contributing

To add new visualization functions:

1. Create function in appropriate module (`temporal_plots.py`, etc.)
2. Follow existing function signature patterns
3. Add docstring with examples
4. Export in `__init__.py`
5. Add to `publication_figures.py` master script
6. Update this documentation

---

## Citation

If you use BASICS-CDSS visualization system, please cite:

**Tritham C, Snae Namahoot C.**
*Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems.*
Healthcare Informatics Research. (under review).

---

## Support

### Documentation
- **Full framework guide**: `docs/ADVANCED_SIMULATION_GUIDE.md`
- **Publication strategy**: `docs/PUBLICATION_STRATEGY.md`
- **Implementation status**: `docs/IMPLEMENTATION_STATUS.md`

### Contact

**Chatchai Tritham**
Email: chatchait66@nu.ac.th
ORCID: [0000-0001-7899-228X](https://orcid.org/0000-0001-7899-228X)

**Chakkrit Snae Namahoot** (Corresponding Author)
Email: chakkrits@nu.ac.th
ORCID: [0000-0003-4660-4590](https://orcid.org/0000-0003-4660-4590)

---

**Last Updated**: January 25, 2026
**Version**: 1.0.0
**License**: MIT
