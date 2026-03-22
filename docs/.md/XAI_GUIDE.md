# XAI (Explainable AI) Guide for BASICS-CDSS

**Version**: 2.0.0
**Date**: 2026-01-25
**Authors**: Chatchai Tritham, Chakkrit Snae Namahoot
**Affiliation**: Department of Computer Science and Information Technology, Faculty of Science, Naresuan University

---

## Table of Contents

1. [Introduction](#introduction)
2. [Game-Theoretic Foundation](#game-theoretic-foundation)
3. [SHAP Analysis](#shap-analysis)
4. [Counterfactual Explanations](#counterfactual-explanations)
5. [Quick Start Examples](#quick-start-examples)
6. [API Reference](#api-reference)
7. [Visualization Gallery](#visualization-gallery)
8. [Clinical Interpretation](#clinical-interpretation)
9. [Integration with BASICS-CDSS](#integration-with-basics-cdss)
10. [Publication Guidelines](#publication-guidelines)
11. [References](#references)

---

## Introduction

BASICS-CDSS v2.0.0 introduces comprehensive Explainable AI (XAI) capabilities specifically designed for clinical decision support systems. The XAI module provides two complementary explanation methods:

### 1. SHAP (SHapley Additive exPlanations)
- **Foundation**: Cooperative game theory (Shapley values)
- **Purpose**: Quantify feature importance and contribution
- **Interpretation**: Features as "players," predictions as "payoff"
- **Clinical Value**: Identify critical symptoms (major players) vs uncertain symptoms (minor players)

### 2. Counterfactual Explanations
- **Foundation**: Causal reasoning and minimal intervention
- **Purpose**: Answer "what-if" questions for clinical decisions
- **Interpretation**: "What needs to change for a different outcome?"
- **Clinical Value**: Actionable intervention suggestions

### Why XAI for Clinical AI?

1. **Trust & Transparency**: Clinicians need to understand AI recommendations
2. **Regulatory Compliance**: FDA requires explainability for AI/ML devices
3. **Clinical Validation**: Verify AI reasoning aligns with medical knowledge
4. **Actionable Insights**: Translate predictions into interventions
5. **Bias Detection**: Identify unfair or unintended feature dependencies

---

## Game-Theoretic Foundation

### Shapley Values from Cooperative Game Theory

The SHAP method is rooted in **cooperative game theory**, specifically the concept of Shapley values introduced by Lloyd Shapley in 1953 (Nobel Prize in Economics, 2012).

#### Game Setup

In the context of clinical decision support:

- **Players**: Clinical features (symptoms, vital signs, lab results)
- **Coalition**: A subset of features used for prediction
- **Payoff**: Prediction accuracy or model output
- **Goal**: Fairly distribute the payoff among players

#### Mathematical Definition

For a feature $i$, the Shapley value $\phi_i$ is:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} [v(S \cup \{i\}) - v(S)]
$$

Where:
- $N$ = set of all features
- $S$ = a coalition (subset) not containing feature $i$
- $v(S)$ = model prediction with features in $S$
- The sum is over all possible coalitions

#### Intuitive Interpretation

The Shapley value is the **average marginal contribution** of feature $i$ across all possible coalitions:

1. Consider all possible orderings of features
2. For each ordering, calculate how much feature $i$ improves the prediction when added
3. Average these contributions

#### Clinical Example

**Patient Triage Scenario**:

```
Features:
- Chest pain (present)
- Troponin = 0.08 ng/mL (elevated)
- Systolic BP = 95 mmHg (low)
- Age = 45 years
- Gender = Male

Prediction: HIGH RISK (85% probability)
```

**Shapley Value Interpretation**:

| Feature | Shapley Value | Interpretation |
|---------|---------------|----------------|
| Troponin | +0.35 | **Major player**: Strong evidence for high risk |
| Chest pain | +0.28 | **Major player**: Critical symptom |
| Systolic BP | +0.22 | **Major player**: Hypotension increases risk |
| Age | +0.02 | **Minor player**: Weak contribution |
| Gender | +0.01 | **Minor player**: Minimal influence |

**Game-Theoretic Explanation**:

- **Troponin** is the MVP (most valuable player) — elevated cardiac biomarker is the strongest indicator
- **Chest pain** and **Systolic BP** are key players — together they form a critical coalition
- **Age** and **Gender** are bench players — contribute minimally to the decision

This aligns with clinical reasoning:
- **Critical symptoms** (troponin elevation, hypotension) → Major players
- **Demographic factors** (age, gender) → Minor players
- **Uncertain findings** → Low Shapley values

### Axioms of Shapley Values

Shapley values satisfy four desirable properties:

1. **Efficiency**: $\sum_{i=1}^{n} \phi_i = v(N) - v(\emptyset)$
   The contributions sum to the total prediction

2. **Symmetry**: If features $i$ and $j$ contribute equally, $\phi_i = \phi_j$

3. **Dummy**: If feature $i$ never changes predictions, $\phi_i = 0$

4. **Additivity**: For combined models, Shapley values add linearly

These axioms ensure **fair attribution** — the only allocation scheme satisfying all four.

---

## SHAP Analysis

### Core Concepts

#### 1. SHAP Values

SHAP values explain individual predictions by decomposing them into feature contributions:

$$
f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i
$$

Where:
- $f(x)$ = model prediction for instance $x$
- $\phi_0$ = base value (expected model output)
- $\phi_i$ = SHAP value for feature $i$

#### 2. Feature Importance

Global feature importance is computed as:

$$
\text{Importance}(i) = \frac{1}{m} \sum_{j=1}^{m} |\phi_i^{(j)}|
$$

Mean absolute SHAP value across all instances $j = 1, \ldots, m$.

#### 3. Interaction Effects

SHAP interaction values capture how features work together:

$$
\phi_{i,j} = \text{contribution of feature } i \text{ depending on feature } j
$$

### Usage Examples

#### Example 1: Compute SHAP Values

```python
from basics_cdss.xai import compute_shap_values
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute SHAP values
shap_values = compute_shap_values(
    model=model,
    X=X_test.values,
    feature_names=list(X_test.columns),
    model_type='tree'  # auto-detect available
)

print(f"SHAP values shape: {shap_values.values.shape}")
print(f"Base value: {shap_values.base_value:.4f}")
```

#### Example 2: Feature Importance Ranking

```python
from basics_cdss.xai import feature_importance_ranking

# Rank features by importance
importance = feature_importance_ranking(
    shap_values,
    method='mean_abs',
    threshold_percentile=75  # Top 25% are "critical"
)

print("Critical Features (Major Players):")
for feat in importance.critical_features:
    idx = importance.feature_names.index(feat)
    score = importance.importance_scores[idx]
    print(f"  {feat}: {score:.4f}")

print("\nNon-Critical Features (Minor Players):")
for feat in importance.non_critical_features:
    idx = importance.feature_names.index(feat)
    score = importance.importance_scores[idx]
    print(f"  {feat}: {score:.4f}")
```

#### Example 3: Game-Theoretic Explanation

```python
from basics_cdss.xai import game_theoretic_explanation

# Explain a single prediction
explanation = game_theoretic_explanation(
    shap_values,
    sample_idx=0,
    major_player_percentile=75
)

print("Game-Theoretic Explanation:")
print("\nMAJOR PLAYERS (Critical Symptoms):")
for player, value in explanation.major_players.items():
    direction = "INCREASES" if value > 0 else "DECREASES"
    print(f"  {player}: {value:+.4f} ({direction} risk)")

print("\nMINOR PLAYERS (Uncertain Symptoms):")
for player, value in explanation.minor_players.items():
    print(f"  {player}: {value:+.4f}")
```

#### Example 4: SHAP Interaction Values

```python
from basics_cdss.xai import compute_shap_interaction_values

# Compute pairwise interactions (tree models only)
interactions = compute_shap_interaction_values(
    model=model,
    X=X_test.values,
    feature_names=list(X_test.columns)
)

# Find strongest interaction
import numpy as np
mean_interaction = np.abs(interactions.values).mean(axis=0)

# Get top interaction pair
i, j = np.unravel_index(
    np.argmax(mean_interaction[np.triu_indices_from(mean_interaction, k=1)]),
    mean_interaction.shape
)

print(f"Strongest interaction: {feature_names[i]} <-> {feature_names[j]}")
print(f"Interaction strength: {mean_interaction[i, j]:.4f}")
```

#### Example 5: Stratified SHAP Analysis

```python
from basics_cdss.xai import stratified_shap_analysis

# Analyze by risk tier
stratified = stratified_shap_analysis(
    shap_values,
    strata=risk_tiers,
    strata_names=['Low', 'Medium', 'High']
)

for tier, importance in stratified.items():
    print(f"\n{tier} Risk Tier:")
    print(f"  Critical features: {importance.critical_features}")
```

### SHAP Visualization Examples

#### Waterfall Plot (Single Prediction)

```python
from basics_cdss.visualization import plot_shap_waterfall

fig, ax = plot_shap_waterfall(
    shap_values,
    sample_idx=0,
    max_display=10,
    save_path='shap_waterfall.pdf'
)
```

Shows how features push prediction from base value to final value.

#### Summary Plot (Global Importance)

```python
from basics_cdss.visualization import plot_shap_summary

# Beeswarm plot
fig, ax = plot_shap_summary(
    shap_values,
    plot_type='dot',
    max_display=20,
    save_path='shap_summary.pdf'
)

# Bar plot
fig, ax = plot_shap_summary(
    shap_values,
    plot_type='bar',
    save_path='shap_bar.pdf'
)
```

#### Dependence Plot (Feature Interaction)

```python
from basics_cdss.visualization import plot_shap_dependence

fig, ax = plot_shap_dependence(
    shap_values,
    feature_name='troponin',
    interaction_feature='age',
    save_path='shap_dependence_troponin.pdf'
)
```

Shows how feature value affects SHAP value, colored by interaction feature.

---

## Counterfactual Explanations

### Core Concepts

#### 1. Counterfactual Definition

A counterfactual explanation answers:
**"What minimal changes would lead to a different prediction?"**

Formally, for instance $x$ with prediction $f(x) = y$, find $x'$ such that:

1. $f(x') = y'$ (desired outcome)
2. $d(x, x') $ is minimized (minimal change)
3. $x'$ satisfies constraints (feasibility)
4. Changes are actionable (modifiable features only)

#### 2. Clinical Relevance

**Original**: Patient triaged as **HIGH RISK**

**Counterfactual**: "If systolic BP was 20 mmHg higher AND troponin was in normal range, patient would be **LOW RISK**"

**Clinical Value**:
- Identifies modifiable risk factors
- Suggests intervention priorities
- Supports shared decision-making
- Provides personalized explanations

### Usage Examples

#### Example 1: Generate Single Counterfactual

```python
from basics_cdss.xai import generate_counterfactual

# Find counterfactual for high-risk patient
patient = X_test.values[0]  # Currently HIGH RISK

cf = generate_counterfactual(
    model=model,
    x=patient,
    feature_names=list(X_test.columns),
    desired_class=0,  # LOW RISK
    method='random',  # or 'gradient', 'genetic'
    feature_ranges={
        'sbp': (80, 200),
        'heart_rate': (40, 180),
        'troponin': (0, 1.0),
    },
    immutable_features=['age', 'gender'],  # Cannot change
    max_iterations=1000
)

print(f"Original prediction: {cf.original_prediction}")
print(f"Counterfactual prediction: {cf.counterfactual_prediction}")
print(f"Distance: {cf.distance:.4f}")
print(f"Feasible: {cf.feasible}")

print("\nRequired changes:")
for feat, (old, new) in cf.feature_changes.items():
    print(f"  {feat}: {old:.2f} → {new:.2f}")
```

#### Example 2: Diverse Counterfactuals

```python
from basics_cdss.xai import generate_diverse_counterfactuals

# Generate multiple diverse explanations
cf_set = generate_diverse_counterfactuals(
    model=model,
    x=patient,
    feature_names=list(X_test.columns),
    num_counterfactuals=5,
    desired_class=0,
    feature_ranges=feature_ranges,
    immutable_features=['age', 'gender']
)

print(f"Generated {cf_set.num_counterfactuals} counterfactuals")
print(f"Diversity score: {cf_set.diversity_score:.3f}")

for i, cf in enumerate(cf_set.counterfactuals):
    print(f"\nOption {i+1}:")
    for feat, (old, new) in cf.feature_changes.items():
        print(f"  {feat}: {old:.2f} → {new:.2f}")
```

#### Example 3: Minimal Counterfactual

```python
from basics_cdss.xai import minimal_counterfactual

# Find counterfactual with fewest changes
minimal_cf = minimal_counterfactual(
    model=model,
    x=patient,
    feature_names=list(X_test.columns),
    max_features_changed=3,  # At most 3 features
    desired_class=0
)

print(f"Minimum changes: {len(minimal_cf.feature_changes)}")
print("Changes:")
for feat, (old, new) in minimal_cf.feature_changes.items():
    print(f"  {feat}: {old:.2f} → {new:.2f}")
```

#### Example 4: Actionable Interventions

```python
from basics_cdss.xai import actionable_interventions

# Translate to clinical interventions
intervention_types = {
    'sbp': 'medication',
    'heart_rate': 'medication',
    'troponin': 'medical workup',
    'lactate': 'resuscitation',
    'spo2': 'oxygen therapy',
}

clinical_priority = {
    'troponin': 1,  # Highest priority
    'sbp': 2,
    'lactate': 3,
}

interventions = actionable_interventions(
    cf,
    intervention_types=intervention_types,
    clinical_priority=clinical_priority
)

print("Recommended Interventions (Priority Order):")
for interv in interventions:
    print(f"{interv.priority}. {interv.feature_name} ({interv.intervention_type}):")
    print(f"   Current: {interv.current_value:.2f}")
    print(f"   Target: {interv.target_value:.2f}")
    print(f"   Change: {interv.change_magnitude:.2f} ({interv.change_percentage:.1f}%)")
```

#### Example 5: What-If Analysis

```python
from basics_cdss.xai import whatif_analysis

# Vary systolic BP and see effect on prediction
whatif_df = whatif_analysis(
    model=model,
    x=patient,
    feature_names=list(X_test.columns),
    feature_to_vary='sbp',
    value_range=(80, 200),
    num_points=50
)

# Plot results
from basics_cdss.visualization import plot_whatif_curve

fig, ax = plot_whatif_curve(
    whatif_df,
    feature_name='sbp',
    threshold=0.5,
    save_path='whatif_sbp.pdf'
)
```

### Counterfactual Visualization Examples

#### Comparison Plot

```python
from basics_cdss.visualization import plot_counterfactual_comparison

fig, ax = plot_counterfactual_comparison(
    cf,
    max_features=10,
    save_path='cf_comparison.pdf'
)
```

#### Feature Changes

```python
from basics_cdss.visualization import plot_feature_changes

fig, ax = plot_feature_changes(
    cf,
    show_percentage=True,
    save_path='feature_changes.pdf'
)
```

#### Intervention Priority

```python
from basics_cdss.visualization import plot_intervention_priority

fig, ax = plot_intervention_priority(
    interventions,
    max_display=8,
    save_path='intervention_priority.pdf'
)
```

---

## Quick Start Examples

### Complete XAI Workflow

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import BASICS-CDSS XAI
from basics_cdss.xai import (
    compute_shap_values,
    feature_importance_ranking,
    game_theoretic_explanation,
    generate_counterfactual,
    actionable_interventions,
)
from basics_cdss.visualization import (
    plot_shap_waterfall,
    plot_shap_summary,
    plot_shap_bar,
    plot_counterfactual_comparison,
    plot_intervention_priority,
)

# ========== 1. Load and Prepare Data ==========
X, y = load_clinical_data()  # Your data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ========== 2. Train Model ==========
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")

# ========== 3. SHAP Analysis ==========
print("\n[SHAP Analysis]")

# Compute SHAP values
shap_values = compute_shap_values(
    model, X_test.values,
    feature_names=list(X_test.columns),
    model_type='tree'
)

# Feature importance
importance = feature_importance_ranking(shap_values)

print("\nCritical Features (Major Players):")
for feat in importance.critical_features[:5]:
    idx = importance.feature_names.index(feat)
    print(f"  {feat}: {importance.importance_scores[idx]:.4f}")

# Visualize
plot_shap_summary(shap_values, plot_type='dot', save_path='shap_summary.pdf')
plot_shap_bar(importance, save_path='shap_importance.pdf')

# Explain specific prediction
explanation = game_theoretic_explanation(shap_values, sample_idx=0)

print("\nMajor Players for Patient 0:")
for player, value in sorted(explanation.major_players.items(),
                            key=lambda x: abs(x[1]), reverse=True):
    print(f"  {player}: {value:+.4f}")

# ========== 4. Counterfactual Analysis ==========
print("\n[Counterfactual Analysis]")

# Find high-risk patient
high_risk_idx = np.where(model.predict(X_test.values) == 1)[0][0]
patient = X_test.values[high_risk_idx]

print(f"Analyzing Patient {high_risk_idx} (HIGH RISK)")

# Generate counterfactual
cf = generate_counterfactual(
    model, patient, list(X_test.columns),
    desired_class=0,  # LOW RISK
    feature_ranges={col: (X[col].min(), X[col].max()) for col in X.columns},
    immutable_features=['age', 'gender'],
    max_iterations=1000
)

print(f"\nCounterfactual found: {cf.counterfactual_prediction} (LOW RISK)")
print(f"Distance: {cf.distance:.4f}")
print(f"Changes required: {len(cf.feature_changes)}")

# Get interventions
interventions = actionable_interventions(
    cf,
    intervention_types={'sbp': 'medication', 'heart_rate': 'medication'},
    clinical_priority={'troponin': 1, 'sbp': 2}
)

print("\nTop Interventions:")
for interv in interventions[:3]:
    print(f"{interv.priority}. {interv.feature_name}: "
          f"{interv.current_value:.2f} → {interv.target_value:.2f} "
          f"({interv.intervention_type})")

# Visualize
plot_counterfactual_comparison(cf, save_path='cf_comparison.pdf')
plot_intervention_priority(interventions, save_path='cf_interventions.pdf')

print("\n[Complete] XAI analysis finished!")
```

### Batch Analysis

```python
# Analyze all test samples
all_explanations = []

for i in range(len(X_test)):
    explanation = game_theoretic_explanation(shap_values, sample_idx=i)
    all_explanations.append({
        'sample_idx': i,
        'prediction': model.predict(X_test.values[i:i+1])[0],
        'major_players': list(explanation.major_players.keys()),
        'top_contributor': max(explanation.major_players.items(),
                              key=lambda x: abs(x[1]))[0]
    })

df_explanations = pd.DataFrame(all_explanations)
print(df_explanations.head())

# Most common critical features
from collections import Counter

all_major = [feat for ex in all_explanations for feat in ex['major_players']]
print("\nMost Frequent Major Players:")
for feat, count in Counter(all_major).most_common(10):
    print(f"  {feat}: {count} times")
```

---

## API Reference

### SHAP Analysis Functions

#### `compute_shap_values()`

Compute SHAP values for model predictions.

**Parameters**:
- `model`: Trained model
- `X`: Feature matrix (numpy array)
- `feature_names`: List of feature names
- `model_type`: 'tree', 'linear', 'deep', 'kernel', or 'auto'
- `algorithm`: 'auto', 'permutation', 'partition', 'tree'
- `background_data`: Background dataset for KernelExplainer
- `n_background_samples`: Number of background samples (default: 100)
- `check_additivity`: Verify SHAP values sum correctly (default: True)
- `random_state`: Random seed (default: 42)

**Returns**: `SHAPValues` object

#### `compute_shap_interaction_values()`

Compute SHAP interaction values (tree models only).

**Parameters**:
- `model`: Trained tree-based model
- `X`: Feature matrix
- `feature_names`: List of feature names
- `random_state`: Random seed

**Returns**: `SHAPInteractionValues` object

#### `feature_importance_ranking()`

Rank features by SHAP-based importance.

**Parameters**:
- `shap_values`: SHAPValues object
- `method`: 'mean_abs', 'mean', 'max_abs', 'std'
- `threshold_percentile`: Percentile for critical/non-critical split (0-100)

**Returns**: `FeatureImportance` object

#### `game_theoretic_explanation()`

Generate game-theoretic explanation of prediction.

**Parameters**:
- `shap_values`: SHAPValues object
- `sample_idx`: Sample index to explain
- `major_player_percentile`: Threshold for major/minor classification
- `interaction_values`: Optional SHAPInteractionValues

**Returns**: `GameTheoreticExplanation` object

#### `stratified_shap_analysis()`

SHAP analysis stratified by subgroups.

**Parameters**:
- `shap_values`: SHAPValues object
- `strata`: Group labels for each sample
- `strata_names`: Names for strata groups

**Returns**: Dict mapping stratum name to FeatureImportance

#### `explain_prediction()`

Quick explanation of a single prediction (convenience function).

**Parameters**:
- `model`: Trained model
- `X`: Feature matrix
- `sample_idx`: Index of sample to explain
- `feature_names`: Feature names
- `background_data`: Background data (optional)
- `compute_interactions`: Whether to compute interactions (default: False)

**Returns**: `GameTheoreticExplanation` object

### Counterfactual Functions

#### `generate_counterfactual()`

Generate a single counterfactual explanation.

**Parameters**:
- `model`: Trained classifier
- `x`: Original instance (1D array)
- `feature_names`: List of feature names
- `desired_class`: Target class (optional)
- `method`: 'gradient', 'random', 'genetic'
- `distance_metric`: 'euclidean', 'manhattan', 'cosine'
- `feature_ranges`: Dict of {feature: (min, max)}
- `immutable_features`: List of unchangeable features
- `actionable_features`: List of changeable features
- `categorical_features`: List of categorical features
- `max_iterations`: Maximum optimization iterations (default: 1000)
- `learning_rate`: For gradient-based methods (default: 0.01)
- `tolerance`: Convergence tolerance (default: 1e-3)
- `random_state`: Random seed (default: 42)

**Returns**: `CounterfactualExample` object

#### `generate_diverse_counterfactuals()`

Generate diverse set of counterfactual explanations.

**Parameters**:
- `model`: Trained classifier
- `x`: Original instance
- `feature_names`: Feature names
- `num_counterfactuals`: Number to generate (default: 5)
- `diversity_weight`: Weight for diversity term (default: 1.0)
- `**kwargs`: Additional arguments for generate_counterfactual()

**Returns**: `CounterfactualSet` object

#### `minimal_counterfactual()`

Generate counterfactual with minimal feature changes.

**Parameters**:
- `model`: Trained classifier
- `x`: Original instance
- `feature_names`: Feature names
- `max_features_changed`: Maximum number of features to change (default: 3)
- `**kwargs`: Additional arguments for generate_counterfactual()

**Returns**: `CounterfactualExample` object

#### `actionable_interventions()`

Generate clinical intervention suggestions from counterfactual.

**Parameters**:
- `counterfactual`: CounterfactualExample object
- `intervention_types`: Dict {feature: intervention_type}
- `clinical_priority`: Dict {feature: priority_score}

**Returns**: List of `InterventionSuggestion` objects

#### `whatif_analysis()`

Perform what-if analysis by varying a single feature.

**Parameters**:
- `model`: Trained classifier
- `x`: Original instance
- `feature_names`: Feature names
- `feature_to_vary`: Name of feature to vary
- `value_range`: (min, max) range for feature
- `num_points`: Number of points to sample (default: 20)

**Returns**: DataFrame with feature values and predictions

---

## Visualization Gallery

### SHAP Visualizations

1. **Waterfall Plot**: Individual prediction explanation
2. **Summary Plot (Beeswarm)**: Global feature importance with value distribution
3. **Summary Plot (Bar)**: Simple feature importance ranking
4. **Feature Importance Bar**: Critical vs non-critical features
5. **Dependence Plot**: Feature value vs SHAP value relationship
6. **Heatmap**: SHAP values across samples and features
7. **Interaction Heatmap**: Pairwise feature interactions

### Counterfactual Visualizations

1. **Comparison Plot**: Original vs counterfactual feature values
2. **Feature Changes**: Magnitude of required changes
3. **Intervention Priority**: Ranked clinical interventions
4. **What-If Curve**: Prediction vs single feature variation
5. **Diversity Plot**: Multiple counterfactual options

---

## Clinical Interpretation

### Interpreting SHAP Values

#### Sign of SHAP Value

- **Positive** (+): Feature increases prediction (e.g., increases risk)
- **Negative** (−): Feature decreases prediction (e.g., decreases risk)

#### Magnitude of SHAP Value

- **High absolute value**: Strong contribution (major player)
- **Low absolute value**: Weak contribution (minor player)

#### Clinical Examples

| Feature | Value | SHAP | Interpretation |
|---------|-------|------|----------------|
| Troponin | 0.08 ng/mL | +0.35 | Elevated troponin strongly increases risk |
| Age | 45 years | +0.02 | Age has minimal effect in this case |
| Systolic BP | 95 mmHg | +0.22 | Low BP moderately increases risk |
| Gender | Male | +0.01 | Gender has negligible effect |

### Interpreting Counterfactuals

#### Actionable Changes

Focus on features that can be modified through interventions:

**Modifiable**:
- Vital signs (BP, HR, RR) → Medications, fluids
- Lab values (lactate, troponin) → Medical interventions
- Oxygen saturation → Supplemental oxygen

**Non-modifiable**:
- Demographics (age, gender)
- Medical history
- Time-based features

#### Intervention Feasibility

Consider clinical feasibility:

- **Realistic**: Lower BP by 20 mmHg (medication)
- **Unrealistic**: Reduce age by 10 years (impossible)
- **Uncertain**: Normalize troponin immediately (takes time)

#### Multiple Pathways

Diverse counterfactuals show multiple intervention strategies:

**Option 1**: Lower BP + reduce heart rate (cardiovascular focus)
**Option 2**: Normalize lactate + improve oxygenation (perfusion focus)
**Option 3**: Reduce troponin + control rhythm (cardiac focus)

Clinicians can choose the most appropriate strategy based on:
- Patient condition
- Available resources
- Time constraints
- Side effects and risks

---

## Integration with BASICS-CDSS

### Using XAI with Scenario Perturbations

```python
from basics_cdss.scenario import create_scenario
from basics_cdss.xai import compute_shap_values

# Create perturbation scenario
scenario = create_scenario(
    profile='severe_uncertainty',
    perturbations=['mask', 'noise']
)

# Apply to clean data
X_perturbed = scenario.apply(X_clean)

# SHAP analysis on perturbed data
shap_clean = compute_shap_values(model, X_clean.values, feature_names)
shap_perturbed = compute_shap_values(model, X_perturbed.values, feature_names)

# Compare feature importance
importance_clean = feature_importance_ranking(shap_clean)
importance_perturbed = feature_importance_ranking(shap_perturbed)

print("Critical features changed due to perturbation:")
for feat in set(importance_clean.critical_features) ^ set(importance_perturbed.critical_features):
    print(f"  {feat}")
```

### XAI for Harm-Aware Evaluation

```python
from basics_cdss.metrics import compute_harm_metrics
from basics_cdss.xai import stratified_shap_analysis

# Compute harm metrics
harm = compute_harm_metrics(y_true, y_pred, risk_tiers)

# SHAP analysis by risk tier
stratified_shap = stratified_shap_analysis(
    shap_values, strata=risk_tiers
)

# Identify features contributing to high-harm errors
high_harm_indices = np.where(harm.harm_by_tier['high'] > 0)[0]
high_harm_shap = SHAPValues(
    values=shap_values.values[high_harm_indices],
    base_value=shap_values.base_value,
    data=shap_values.data[high_harm_indices],
    feature_names=shap_values.feature_names
)

# What features drive high-harm errors?
high_harm_importance = feature_importance_ranking(high_harm_shap)
print("Features driving high-harm errors:")
for feat in high_harm_importance.critical_features:
    print(f"  {feat}")
```

---

## Publication Guidelines

### Figures for Tier 1 Journals

All XAI figures follow publication standards:

- **Format**: PDF (vector graphics)
- **Resolution**: 300 DPI minimum
- **Font**: Times New Roman
- **Color scheme**: Colorblind-friendly (Paul Tol's palette)
- **Size**: 7.0 × 6.0 to 11.0 × 8.0 inches
- **Standards**: IEEE/Nature/JAMA compliant

### Recommended Figures for Papers

#### Main Text

1. **SHAP Summary (Beeswarm)**: Global feature importance
2. **SHAP Waterfall**: Example prediction explanation
3. **Counterfactual Comparison**: Example intervention suggestion

#### Supplementary Materials

1. **SHAP Feature Importance Bar**: Complete ranking
2. **SHAP Dependence Plots**: Top 3-5 features
3. **SHAP Interaction Heatmap**: Feature interactions
4. **Multiple Counterfactual Examples**: Diverse cases
5. **What-If Curves**: Sensitivity analysis

### Writing the Methods Section

**Example Methods Text**:

```
We employed Shapley Additive exPlanations (SHAP) to quantify feature
importance based on cooperative game theory [1]. SHAP values represent
the average marginal contribution of each feature across all possible
coalitions, providing a fair attribution of prediction to features [2].

Features were classified as "critical" (major players with Shapley values
above the 75th percentile) or "non-critical" (minor players below the
threshold). This classification aligns with clinical reasoning where certain
symptoms (e.g., elevated cardiac biomarkers) are clearly diagnostic while
others are ambiguous.

To provide actionable insights, we generated counterfactual explanations
using optimization-based methods [3]. Counterfactuals identify minimal
changes to modifiable features that would result in a different triage
decision, translating predictions into concrete intervention suggestions
while respecting clinical constraints (e.g., demographic features are
immutable).
```

**References**:
1. Shapley, L. S. (1953). A value for n-person games.
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
3. Wachter, S., et al. (2018). Counterfactual explanations without opening the black box. Harvard JL & Tech.

### Reporting Results

#### SHAP Results

**Example Results Text**:

```
Feature importance analysis revealed that cardiac biomarkers (troponin,
lactate) and vital signs (systolic BP, heart rate) were the strongest
predictors (mean |SHAP| > 0.25), consistent with clinical expectations.
Demographic factors (age, gender) had minimal influence (mean |SHAP| < 0.05).

For high-risk patients (n=150), elevated troponin was the most frequent
major player (85% of cases), followed by hypotension (62%) and tachycardia
(58%). This game-theoretic interpretation aligns with established clinical
risk factors for acute cardiac events.
```

#### Counterfactual Results

**Example Results Text**:

```
Counterfactual analysis of high-risk triage decisions (n=150) identified
an average of 3.2 ± 1.1 modifiable features requiring change for low-risk
classification. The most frequent intervention suggestions were:
  1. Normalize cardiac biomarkers (88% of cases)
  2. Improve blood pressure (75%)
  3. Control heart rate (62%)

Diverse counterfactual sets (k=5) provided multiple intervention pathways
with mean pairwise distance of 2.4 ± 0.8, offering clinicians flexibility
in choosing appropriate strategies based on patient context and resource
availability.
```

---

## References

### Foundational Papers

1. **Shapley, L. S.** (1953). A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307-317.

2. **Lundberg, S. M., & Lee, S. I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

3. **Lundberg, S. M., et al.** (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.

4. **Wachter, S., Mittelstadt, B., & Russell, C.** (2018). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *Harvard Journal of Law & Technology*, 31(2), 841.

5. **Mothilal, R. K., Sharma, A., & Tan, C.** (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607-617.

### Clinical AI Explainability

6. **Tonekaboni, S., et al.** (2019). What clinicians want: Contextualizing explainable machine learning for clinical end use. *Machine Learning for Healthcare Conference*, 359-380.

7. **Ghassemi, M., et al.** (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*, 3(11), e745-e750.

8. **Amann, J., et al.** (2020). Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC Medical Informatics and Decision Making*, 20(1), 1-9.

### Regulatory Guidance

9. **FDA** (2021). Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan.

10. **European Commission** (2021). Proposal for a Regulation on Artificial Intelligence (AI Act).

### Technical References

11. **Molnar, C.** (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd edition.

12. **Barocas, S., Hardt, M., & Narayanan, A.** (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.

---

## Appendix: Installation and Requirements

### Install BASICS-CDSS with XAI

```bash
# Install from source
git clone https://github.com/ChatchaiTritham/BASICS-CDSS.git
cd BASICS-CDSS
pip install -e .

# Install SHAP (required for XAI)
pip install shap>=0.42.0
```

### Dependencies

- Python >= 3.10
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- **shap >= 0.42.0** (for SHAP analysis)

### Optional

- jupyter: For interactive notebooks
- xgboost: For tree-based models
- pytorch: For deep learning models

---

**Document Version**: 2.0.0
**Last Updated**: 2026-01-25
**Maintainers**: Chatchai Tritham (chatchait66@nu.ac.th), Chakkrit Snae Namahoot (chakkrits@nu.ac.th)
**License**: MIT

---
