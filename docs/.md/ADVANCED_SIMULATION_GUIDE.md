# BASICS-CDSS Advanced Simulation Guide

## Overview

BASICS-CDSS provides a **three-tier simulation framework** for evaluating clinical decision support systems with increasing levels of realism and complexity:

1. **Tier 1: Digital Twin Simulation** - Temporal patient evolution
2. **Tier 2: Causal Simulation** - Structural causal models with interventions
3. **Tier 3: Multi-Agent Simulation** - System-level effects and agent interactions

Each tier builds on the previous, enabling comprehensive CDSS evaluation from individual patient trajectories to emergent system-level phenomena.

---

## When to Use Each Tier

### Tier 1: Digital Twin Simulation
**Use when you need:**
- Time-evolving patient states
- Counterfactual "what-if" analysis
- Temporal perturbations (time-varying uncertainty)
- Disease progression modeling

**Best for:**
- Individual patient-level CDSS evaluation
- Trajectory-based metrics (temporal consistency, delayed intervention risk)
- Testing CDSS with physiologically realistic patient evolution

**Example:**
```python
from basics_cdss.temporal import PatientDigitalTwin, SepsisModel

twin = PatientDigitalTwin(
    archetype_id='A001',
    initial_state={'temperature': 38.5, 'heart_rate': 110},
    disease_model=SepsisModel(),
    seed=42
)

trajectory = twin.simulate(horizon_hours=24, dt=1.0)
```

---

### Tier 2: Causal Simulation
**Use when you need:**
- Causal reasoning about interventions
- Do-calculus and counterfactuals
- Confounding adjustment
- Average Treatment Effect (ATE) estimation

**Best for:**
- Evaluating causal effects of CDSS recommendations
- Identifying and controlling for confounders
- Ensuring generated data satisfies causal constraints
- Comparing observational vs interventional effects

**Example:**
```python
from basics_cdss.causal import CausalGraph, StructuralCausalModel, compute_ate

# Define causal graph
graph = CausalGraph()
graph.add_edge('infection', 'temperature')
graph.add_edge('infection', 'white_blood_cell_count')
graph.add_edge('temperature', 'heart_rate')

# Create SCM
scm = StructuralCausalModel(graph, seed=42)

# Sample observational data
obs_data = scm.sample(n=1000)

# Perform intervention: do(antibiotic=True)
int_data = scm.do_intervention({'antibiotic': True}, n=1000)

# Compute ATE
ate = compute_ate(scm, treatment='antibiotic', outcome='mortality')
print(f"ATE: {ate['ate']:.3f}")
```

---

### Tier 3: Multi-Agent Simulation
**Use when you need:**
- System-level effects (alert fatigue, workflow disruption)
- Agent interactions (Patient, Clinician, CDSS, Nurse)
- Clinical workflow modeling
- Emergent phenomena from CDSS deployment

**Best for:**
- Evaluating CDSS in realistic clinical environments
- Understanding systemic impacts (override rates, coordination)
- Testing CDSS integration with workflows
- Studying human-AI interaction in healthcare

**Example:**
```python
from basics_cdss.multiagent import (
    HospitalEnvironment, PatientAgent, ClinicianAgent, CDSSAgent
)

# Create environment
hospital = HospitalEnvironment(n_beds=20, icu_beds=8)

# Create agents
patient = PatientAgent(archetype_id='A001', digital_twin=twin)
clinician = ClinicianAgent(experience_level='senior', cdss_trust=0.8)
cdss = CDSSAgent(model=sepsis_model, alert_threshold=0.8)

# Add to environment
hospital.add_agent(patient)
hospital.add_agent(clinician)
hospital.add_agent(cdss)

# Run simulation
results = hospital.simulate(duration_hours=24, dt=1.0)

# Analyze systemic effects
from basics_cdss.multiagent import compute_alert_fatigue, compute_override_rate

fatigue = compute_alert_fatigue(results)
override = compute_override_rate(results)

print(f"Alert fatigue score: {fatigue['fatigue_score']:.2f}")
print(f"Override rate: {override['overall']:.2%}")
```

---

## API Reference

### Tier 1: Digital Twin Simulation

#### `PatientDigitalTwin`
Temporal patient simulator with disease progression.

**Key Methods:**
- `simulate(horizon_hours, dt)` - Simulate trajectory
- `step(dt, interventions)` - Single time step
- `apply_intervention(interventions)` - Apply medical intervention
- `reset()` - Reset to initial state

**Disease Models:**
- `SepsisModel` - Sepsis progression
- `RespiratoryDistressModel` - ARDS/respiratory failure
- `CardiacEventModel` - Acute coronary syndrome

**Metrics:**
- `temporal_consistency_score()` - Temporal coherence
- `delayed_intervention_risk()` - Risk from delays
- `counterfactual_regret()` - Regret from suboptimal decisions

---

### Tier 2: Causal Simulation

#### `CausalGraph`
Directed acyclic graph for causal relationships.

**Key Methods:**
- `add_edge(cause, effect)` - Add causal edge
- `get_parents(variable)` - Get direct causes
- `d_separated(X, Y, Z)` - Test conditional independence
- `topological_order()` - Get causal ordering

#### `StructuralCausalModel`
SCM with structural equations.

**Key Methods:**
- `sample(n)` - Sample observational data
- `do_intervention(interventions, n)` - Interventional sampling
- `counterfactual(observation, intervention, query)` - Counterfactual reasoning

#### Interventions
- `compute_ate(scm, treatment, outcome)` - Average Treatment Effect
- `compute_cate(scm, treatment, outcome, conditioning)` - Conditional ATE
- `perform_do_intervention(scm, intervention)` - Do-calculus

#### Confounding
- `identify_confounders(graph, treatment, outcome)` - Backdoor criterion
- `backdoor_adjustment(data, treatment, outcome, confounders)` - Adjustment formula
- `frontdoor_adjustment(data, treatment, outcome, mediator)` - Frontdoor criterion

**Metrics:**
- `causal_consistency_score()` - Data-graph consistency
- `intervention_effect_size()` - Effect size of interventions
- `confounding_bias_estimate()` - Confounding bias

---

### Tier 3: Multi-Agent Simulation

#### Agents
- `PatientAgent` - Patient with evolving state
- `ClinicianAgent` - Physician making decisions
- `CDSSAgent` - CDSS generating alerts
- `NurseAgent` - Nurse monitoring patients

#### `HospitalEnvironment`
Simulated hospital with wards and resources.

**Key Methods:**
- `add_agent(agent)` - Add agent to environment
- `simulate(duration_hours, dt)` - Run simulation
- `step(dt)` - Single time step
- `get_state()` - Current environment state

#### `ClinicalWorkflow`
Clinical protocol as task graph.

**Key Methods:**
- `add_task(task)` - Add task to workflow
- `get_ready_tasks()` - Get executable tasks
- `start(current_time)` - Start workflow
- `update(current_time)` - Update workflow state

**Pre-defined Workflows:**
- `create_sepsis_workflow()` - Sepsis 3-hour bundle
- `create_acs_workflow()` - STEMI management
- `create_respiratory_distress_workflow()` - Respiratory failure

#### Systemic Metrics
- `compute_alert_fatigue(results)` - Alert fatigue score
- `compute_override_rate(results)` - Override rate
- `compute_workflow_disruption(results)` - Workflow impact
- `compute_time_to_action(results)` - Time from alert to action
- `compute_coordination_efficiency(results)` - Agent coordination

---

## Common Use Cases

### Use Case 1: Evaluate CDSS with Temporal Perturbations

```python
from basics_cdss.temporal import (
    PatientDigitalTwin, SepsisModel, TemporalNoiseOperator
)

# Create digital twin
twin = PatientDigitalTwin(
    archetype_id='A001',
    initial_state={'temperature': 38.5},
    disease_model=SepsisModel()
)

# Add temporal noise
noise_op = TemporalNoiseOperator(noise_std=0.5)
trajectory = twin.simulate(horizon_hours=24)
noisy_trajectory = noise_op.apply(trajectory)

# Evaluate CDSS on noisy data
predictions = cdss_model.predict(noisy_trajectory)
```

### Use Case 2: Estimate Causal Effect of CDSS Recommendation

```python
from basics_cdss.causal import StructuralCausalModel, compute_ate

# Create SCM
scm = StructuralCausalModel(causal_graph)

# Estimate ATE of recommended intervention
ate_result = compute_ate(
    scm=scm,
    treatment='antibiotic_early',
    outcome='mortality',
    treatment_values=[False, True]
)

print(f"ATE: {ate_result['ate']:.3f}")
print(f"Control mortality: {ate_result['control_mean']:.3f}")
print(f"Treatment mortality: {ate_result['treatment_mean']:.3f}")
```

### Use Case 3: Simulate CDSS Deployment and Measure Alert Fatigue

```python
from basics_cdss.multiagent import (
    HospitalEnvironment, ClinicianAgent, CDSSAgent,
    compute_alert_fatigue, generate_systemic_report
)

# Create environment
hospital = HospitalEnvironment(n_beds=20, icu_beds=8)

# Add agents
clinician = ClinicianAgent(
    experience_level='senior',
    workload_capacity=5,
    cdss_trust=0.8
)
cdss = CDSSAgent(
    model=sepsis_model,
    alert_threshold=0.8,
    alert_cooldown=2.0
)

hospital.add_agent(clinician)
hospital.add_agent(cdss)

# Add patients
for i in range(10):
    patient = PatientAgent(archetype_id=f'A{i:03d}', digital_twin=twins[i])
    hospital.add_agent(patient)

# Run simulation
results = hospital.simulate(duration_hours=24, dt=1.0)

# Generate systemic report
report = generate_systemic_report(results)

print(f"Alert Fatigue: {report['alert_fatigue']['fatigue_score']:.2f}")
print(f"Override Rate: {report['override']['overall']:.2%}")
print(f"Workflow Disruption: {report['disruption']['disruption_score']:.2f}")
```

---

## Comparison Table: Tier 1 vs 2 vs 3

| Feature | Tier 1: Digital Twin | Tier 2: Causal | Tier 3: Multi-Agent |
|---------|---------------------|----------------|---------------------|
| **Temporal Evolution** | ✅ Yes | ⚠️ Limited | ✅ Yes |
| **Causal Reasoning** | ❌ No | ✅ Yes | ⚠️ Limited |
| **Interventions** | ⚠️ Simple | ✅ Do-calculus | ✅ Complex |
| **Agent Interactions** | ❌ No | ❌ No | ✅ Yes |
| **System-Level Effects** | ❌ No | ❌ No | ✅ Yes |
| **Computational Cost** | Low | Medium | High |
| **Realism** | Medium | High (causal) | Very High (system) |
| **Primary Use** | Individual patients | Causal effects | System evaluation |
| **Evaluation Focus** | Temporal metrics | Causal metrics | Systemic metrics |

**Recommendations:**
- Start with **Tier 1** for basic temporal evaluation
- Add **Tier 2** when causal reasoning is needed
- Use **Tier 3** for comprehensive system-level evaluation

---

## Integration Example: Using All Three Tiers

```python
# Tier 1: Create digital twins
from basics_cdss.temporal import PatientDigitalTwin, SepsisModel

twins = []
for i in range(100):
    twin = PatientDigitalTwin(
        archetype_id=f'A{i:03d}',
        initial_state=initial_states[i],
        disease_model=SepsisModel()
    )
    twins.append(twin)

# Tier 2: Define causal structure
from basics_cdss.causal import create_sepsis_causal_graph, StructuralCausalModel

graph = create_sepsis_causal_graph()
scm = StructuralCausalModel(graph)

# Generate causally-consistent data
causal_data = scm.sample(n=1000)

# Tier 3: Multi-agent simulation
from basics_cdss.multiagent import HospitalEnvironment, ClinicianAgent, CDSSAgent

hospital = HospitalEnvironment(n_beds=20)

# Add clinicians
for i in range(5):
    clinician = ClinicianAgent(experience_level='senior')
    hospital.add_agent(clinician)

# Add CDSS
cdss = CDSSAgent(model=sepsis_model, alert_threshold=0.8)
hospital.add_agent(cdss)

# Add patients (with digital twins)
for twin in twins[:20]:
    patient = PatientAgent(archetype_id=twin.archetype_id, digital_twin=twin)
    hospital.add_agent(patient)

# Simulate
results = hospital.simulate(duration_hours=24)

# Comprehensive evaluation
from basics_cdss.temporal import temporal_consistency_score
from basics_cdss.causal import causal_consistency_score
from basics_cdss.multiagent import generate_systemic_report

temporal_score = temporal_consistency_score(trajectories)
causal_score = causal_consistency_score(causal_data, graph)
systemic_report = generate_systemic_report(results)

print(f"Temporal Consistency: {temporal_score:.2f}")
print(f"Causal Consistency: {causal_score['consistency_score']:.2f}")
print(f"Alert Fatigue: {systemic_report['alert_fatigue']['fatigue_score']:.2f}")
```

---

## Best Practices

### 1. Start Simple, Add Complexity
- Begin with Tier 1 for basic evaluation
- Add Tier 2 only if causal reasoning is needed
- Use Tier 3 for final system-level validation

### 2. Validate Assumptions
- Validate disease models against real data (Tier 1)
- Validate causal graphs with domain experts (Tier 2)
- Validate agent behaviors with clinicians (Tier 3)

### 3. Use Appropriate Seeds
- Set random seeds for reproducibility
- Use different seeds for sensitivity analysis

### 4. Monitor Computational Resources
- Tier 1: Fast, can simulate thousands of patients
- Tier 2: Medium, sample sizes of 1000-10000
- Tier 3: Slow, limit to 10-50 agents

### 5. Combine with Real Data
- Use real archetypes as initial states
- Calibrate disease models to real trajectories
- Validate causal graphs with observational data
- Compare simulated metrics to real-world benchmarks

---

## Troubleshooting

### Issue: Digital twin trajectories are unrealistic
**Solution:**
- Adjust disease model parameters
- Validate against real patient trajectories
- Use domain expert input for physiological bounds

### Issue: Causal graph has cycles
**Solution:**
- Remove feedback loops (SCMs require DAGs)
- Consider time-lagged variables for dynamic systems
- Use separate graphs for different time scales

### Issue: Multi-agent simulation is too slow
**Solution:**
- Reduce number of agents
- Increase time step (dt)
- Use fewer patients or shorter simulation duration
- Profile code to identify bottlenecks

### Issue: High alert fatigue in simulations
**Solution:**
- Adjust CDSS alert threshold
- Increase alert cooldown period
- Improve CDSS specificity
- Add alert prioritization logic

---

## Further Reading

- **Tier 1 Theory:**
  - Digital Twin concept: Glaessgen & Stargel (2012)
  - ODE/SDE models: Edelstein-Keshet (2004)

- **Tier 2 Theory:**
  - Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
  - Hernán & Robins (2020). *Causal Inference: What If*

- **Tier 3 Theory:**
  - Wooldridge (2009). *An Introduction to MultiAgent Systems*
  - Berg et al. (2005). Sociotechnical systems in healthcare

---

## Citation

If you use BASICS-CDSS Advanced Simulation in your research, please cite:

```
@software{basics_cdss,
  title={BASICS-CDSS: Bayesian Archetypes for Simulating Inconsistencies in Clinical Decision Support Systems},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/BASICS-CDSS}
}
```
