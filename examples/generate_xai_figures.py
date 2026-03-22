"""
Generate XAI (Explainable AI) Figures for BASICS-CDSS

This script demonstrates the complete XAI capabilities of BASICS-CDSS:
1. SHAP (Shapley value) analysis with game-theoretic interpretation
2. Counterfactual explanations for clinical decisions

Generates publication-ready figures for:
- Feature importance rankings (major vs minor players)
- SHAP waterfall, summary, and dependence plots
- Counterfactual comparisons and intervention suggestions

Author: Chatchai Tritham
Date: 2026-01-25
Version: 2.0.0 (XAI Enhancement)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from basics_cdss.xai import (
        compute_shap_values,
        compute_shap_interaction_values,
        feature_importance_ranking,
        game_theoretic_explanation,
        generate_counterfactual,
        generate_diverse_counterfactuals,
        actionable_interventions,
        whatif_analysis,
    )
    from basics_cdss.visualization import (
        plot_shap_waterfall,
        plot_shap_summary,
        plot_shap_bar,
        plot_shap_dependence,
        plot_shap_heatmap,
        plot_shap_interaction_heatmap,
        plot_counterfactual_comparison,
        plot_feature_changes,
        plot_intervention_priority,
        plot_whatif_curve,
        plot_counterfactual_diversity,
    )
except ImportError as e:
    print(f"Error importing BASICS-CDSS: {e}")
    print("Make sure you have installed the package with: pip install -e .")
    print("And installed SHAP: pip install shap>=0.42.0")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')


def create_synthetic_clinical_data(n_samples=500, random_state=42):
    """
    Create synthetic clinical data for triage prediction.

    Simulates emergency department triage with features like:
    - Vital signs (critical symptoms = major players)
    - Lab results (important for diagnosis)
    - Demographics (minor players, non-modifiable)
    - Symptoms (varying importance)

    Returns:
        X: Feature matrix (DataFrame)
        y: Target (0=low risk, 1=high risk)
        feature_names: List of feature names with descriptions
    """
    np.random.seed(random_state)

    # Define features with clinical interpretation
    data = {}

    # ========== MAJOR PLAYERS (Critical Symptoms) ==========
    # These should have high Shapley values

    # Systolic Blood Pressure (critical vital sign)
    # Low BP = shock risk (high risk), Normal = stable
    data['sbp'] = np.random.normal(120, 25, n_samples)

    # Heart Rate (critical vital sign)
    # Tachycardia or bradycardia = concerning
    data['heart_rate'] = np.random.normal(80, 20, n_samples)

    # Respiratory Rate (critical for oxygenation)
    # Tachypnea = respiratory distress
    data['resp_rate'] = np.random.normal(16, 4, n_samples)

    # Troponin (cardiac biomarker)
    # Elevated = myocardial injury
    data['troponin'] = np.abs(np.random.normal(0.02, 0.05, n_samples))

    # Lactate (tissue perfusion marker)
    # High lactate = shock
    data['lactate'] = np.abs(np.random.normal(1.5, 1.0, n_samples))

    # ========== MODERATE PLAYERS ==========

    # Temperature
    data['temperature'] = np.random.normal(37.0, 0.8, n_samples)

    # Oxygen Saturation
    data['spo2'] = np.clip(np.random.normal(97, 3, n_samples), 80, 100)

    # White Blood Cell Count
    data['wbc'] = np.abs(np.random.normal(8.0, 3.0, n_samples))

    # Creatinine (kidney function)
    data['creatinine'] = np.abs(np.random.normal(1.0, 0.5, n_samples))

    # ========== MINOR PLAYERS (Non-modifiable or less predictive) ==========

    # Age (non-modifiable demographic)
    data['age'] = np.random.randint(18, 90, n_samples)

    # Gender (non-modifiable demographic)
    data['gender'] = np.random.randint(0, 2, n_samples)

    # Time of arrival (less predictive)
    data['arrival_hour'] = np.random.randint(0, 24, n_samples)

    # Pain score (subjective, uncertain)
    data['pain_score'] = np.random.randint(0, 11, n_samples)

    # Convert to DataFrame
    X = pd.DataFrame(data)

    # ========== Generate Target (High Risk = 1, Low Risk = 0) ==========
    # Use logistic function based on weighted features

    # Major players have high weights
    risk_score = (
        (X['sbp'] < 90) * 3.0 +  # Hypotension (critical)
        (X['heart_rate'] > 120) * 2.5 +  # Tachycardia
        (X['resp_rate'] > 24) * 2.0 +  # Tachypnea
        (X['troponin'] > 0.04) * 3.5 +  # Elevated troponin (critical)
        (X['lactate'] > 2.5) * 2.5 +  # Elevated lactate
        (X['spo2'] < 92) * 2.0 +  # Hypoxia
        (X['temperature'] > 38.5) * 1.0 +  # Fever
        (X['wbc'] > 12) * 0.8 +  # Leukocytosis
        (X['creatinine'] > 1.5) * 1.0 +  # Acute kidney injury
        # Minor players have low weights
        (X['age'] > 70) * 0.5 +  # Elderly
        (X['pain_score'] > 7) * 0.3  # Severe pain
    )

    # Convert to probability
    prob = 1 / (1 + np.exp(-risk_score + 5))

    # Generate binary labels
    y = (prob > np.random.rand(n_samples)).astype(int)

    feature_descriptions = {
        'sbp': 'Systolic Blood Pressure (mmHg)',
        'heart_rate': 'Heart Rate (bpm)',
        'resp_rate': 'Respiratory Rate (breaths/min)',
        'troponin': 'Troponin I (ng/mL)',
        'lactate': 'Serum Lactate (mmol/L)',
        'temperature': 'Body Temperature (°C)',
        'spo2': 'Oxygen Saturation (%)',
        'wbc': 'White Blood Cell Count (×10⁹/L)',
        'creatinine': 'Serum Creatinine (mg/dL)',
        'age': 'Age (years)',
        'gender': 'Gender (0=F, 1=M)',
        'arrival_hour': 'Arrival Hour (0-23)',
        'pain_score': 'Pain Score (0-10)',
    }

    return X, y, list(X.columns), feature_descriptions


def train_demo_model(X, y):
    """Train a simple Random Forest classifier for demonstration."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"  Model trained: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

    return model, X_train, X_test, y_train, y_test


def generate_shap_figures(model, X, y, feature_names, output_dir):
    """Generate all SHAP visualization figures."""
    print("\n[*] Generating SHAP visualization figures...")

    shap_dir = output_dir / 'shap'
    shap_dir.mkdir(exist_ok=True)

    # Compute SHAP values
    print("  - Computing SHAP values...")
    shap_values = compute_shap_values(
        model, X.values, feature_names,
        model_type='tree'
    )

    # Compute feature importance
    importance = feature_importance_ranking(
        shap_values,
        threshold_percentile=75
    )

    # Compute interaction values
    print("  - Computing SHAP interaction values...")
    try:
        interaction_values = compute_shap_interaction_values(
            model, X.values, feature_names
        )
        has_interactions = True
    except:
        print("    (Interaction values not available)")
        has_interactions = False

    # ========== Generate SHAP Plots ==========

    # 1. SHAP Waterfall (single instance explanation)
    print("  - Generating SHAP waterfall plot...")
    plot_shap_waterfall(
        shap_values, sample_idx=0,
        save_path=str(shap_dir / 'shap_waterfall.pdf')
    )

    # 2. SHAP Summary (beeswarm)
    print("  - Generating SHAP summary plot (beeswarm)...")
    plot_shap_summary(
        shap_values, plot_type='dot',
        save_path=str(shap_dir / 'shap_summary_beeswarm.pdf')
    )

    # 3. SHAP Summary (bar)
    print("  - Generating SHAP summary plot (bar)...")
    plot_shap_summary(
        shap_values, plot_type='bar',
        save_path=str(shap_dir / 'shap_summary_bar.pdf')
    )

    # 4. SHAP Bar (feature importance)
    print("  - Generating SHAP feature importance bar...")
    plot_shap_bar(
        importance,
        highlight_critical=True,
        save_path=str(shap_dir / 'shap_feature_importance.pdf')
    )

    # 5. SHAP Dependence (top 3 features)
    print("  - Generating SHAP dependence plots...")
    top_3_indices = np.where(importance.importance_rank <= 3)[0]
    top_3_features = [importance.feature_names[i] for i in top_3_indices]
    for feat in top_3_features:
        plot_shap_dependence(
            shap_values, feat,
            save_path=str(shap_dir / f'shap_dependence_{feat}.pdf')
        )

    # 6. SHAP Heatmap
    print("  - Generating SHAP heatmap...")
    plot_shap_heatmap(
        shap_values,
        max_features=12,
        max_samples=50,
        save_path=str(shap_dir / 'shap_heatmap.pdf')
    )

    # 7. SHAP Interaction Heatmap
    if has_interactions:
        print("  - Generating SHAP interaction heatmap...")
        plot_shap_interaction_heatmap(
            interaction_values,
            top_k=10,
            save_path=str(shap_dir / 'shap_interaction_heatmap.pdf')
        )

    # 8. Game-theoretic explanation (text output)
    print("  - Generating game-theoretic explanation...")
    explanation = game_theoretic_explanation(
        shap_values, sample_idx=0,
        interaction_values=interaction_values if has_interactions else None
    )

    # Save text explanation
    with open(shap_dir / 'game_theoretic_explanation.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Game-Theoretic Explanation (Sample 0)\n")
        f.write("=" * 80 + "\n\n")

        f.write("MAJOR PLAYERS (Critical Symptoms - High Shapley Values):\n")
        f.write("-" * 80 + "\n")
        for player, value in sorted(explanation.major_players.items(),
                                    key=lambda x: abs(x[1]), reverse=True):
            direction = "INCREASES" if value > 0 else "DECREASES"
            f.write(f"  {player:20s}: {value:+.4f}  ({direction} risk)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("MINOR PLAYERS (Uncertain/Non-Critical - Low Shapley Values):\n")
        f.write("-" * 80 + "\n")
        for player, value in sorted(explanation.minor_players.items(),
                                    key=lambda x: abs(x[1]), reverse=True):
            direction = "increases" if value > 0 else "decreases"
            f.write(f"  {player:20s}: {value:+.4f}  ({direction} risk slightly)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        f.write("In cooperative game theory, features are 'players' in a game where the\n")
        f.write("'payoff' is the prediction accuracy. Shapley values provide a fair\n")
        f.write("attribution of this payoff to each player.\n\n")
        f.write("- MAJOR PLAYERS have high Shapley values → strong influence on decision\n")
        f.write("- MINOR PLAYERS have low Shapley values → weak influence on decision\n\n")
        f.write("This aligns with clinical intuition:\n")
        f.write("  • Critical symptoms (e.g., troponin, BP) → major players\n")
        f.write("  • Uncertain/ambiguous signs → minor players\n")
        f.write("=" * 80 + "\n")

    print(f"[OK] Generated {7 if has_interactions else 6} SHAP figure types")


def generate_counterfactual_figures(model, X, y, feature_names, output_dir):
    """Generate all counterfactual visualization figures."""
    print("\n[*] Generating counterfactual visualization figures...")

    cf_dir = output_dir / 'counterfactual'
    cf_dir.mkdir(exist_ok=True)

    # Find a high-risk patient to generate counterfactuals for
    high_risk_indices = np.where(model.predict(X.values) == 1)[0]
    if len(high_risk_indices) == 0:
        print("  [WARNING] No high-risk predictions found. Skipping counterfactuals.")
        return

    patient_idx = high_risk_indices[0]
    patient = X.values[patient_idx]

    print(f"  - Generating counterfactuals for Patient {patient_idx} (HIGH RISK)")

    # Define feature ranges for constraints
    feature_ranges = {
        'sbp': (80, 200),
        'heart_rate': (40, 180),
        'resp_rate': (10, 40),
        'troponin': (0, 1.0),
        'lactate': (0, 10),
        'temperature': (35, 41),
        'spo2': (85, 100),
        'wbc': (2, 25),
        'creatinine': (0.5, 5.0),
        'age': (18, 90),
        'pain_score': (0, 10),
    }

    # Immutable features (demographics)
    immutable = ['age', 'gender', 'arrival_hour']

    # ========== Generate Counterfactuals ==========

    # 1. Single counterfactual
    print("  - Generating single counterfactual...")
    cf = generate_counterfactual(
        model, patient, feature_names,
        desired_class=0,  # LOW RISK
        method='random',
        feature_ranges=feature_ranges,
        immutable_features=immutable,
        max_iterations=500
    )

    # 2. Plot: Counterfactual comparison
    print("  - Plotting counterfactual comparison...")
    plot_counterfactual_comparison(
        cf,
        save_path=str(cf_dir / 'counterfactual_comparison.pdf')
    )

    # 3. Plot: Feature changes
    print("  - Plotting feature changes...")
    plot_feature_changes(
        cf,
        show_percentage=True,
        save_path=str(cf_dir / 'feature_changes.pdf')
    )

    # 4. Generate intervention suggestions
    print("  - Generating intervention suggestions...")
    intervention_types = {
        'sbp': 'medication',
        'heart_rate': 'medication',
        'troponin': 'medical workup',
        'lactate': 'resuscitation',
        'temperature': 'medication',
        'resp_rate': 'oxygen therapy',
        'spo2': 'oxygen therapy',
        'wbc': 'antibiotics',
        'creatinine': 'hydration',
        'pain_score': 'analgesia',
    }

    clinical_priority = {
        'troponin': 1,
        'sbp': 2,
        'lactate': 3,
        'spo2': 4,
        'heart_rate': 5,
        'resp_rate': 6,
    }

    interventions = actionable_interventions(
        cf,
        intervention_types=intervention_types,
        clinical_priority=clinical_priority
    )

    # 5. Plot: Intervention priority
    print("  - Plotting intervention priority...")
    plot_intervention_priority(
        interventions,
        save_path=str(cf_dir / 'intervention_priority.pdf')
    )

    # 6. What-if analysis (vary systolic BP)
    print("  - Performing what-if analysis (systolic BP)...")
    whatif_df = whatif_analysis(
        model, patient, feature_names,
        feature_to_vary='sbp',
        value_range=(80, 200),
        num_points=50
    )

    # 7. Plot: What-if curve
    print("  - Plotting what-if curve...")
    plot_whatif_curve(
        whatif_df,
        feature_name='sbp',
        threshold=0.5,
        save_path=str(cf_dir / 'whatif_sbp.pdf')
    )

    # 8. Generate diverse counterfactuals
    print("  - Generating diverse counterfactuals...")
    cf_set = generate_diverse_counterfactuals(
        model, patient, feature_names,
        num_counterfactuals=5,
        desired_class=0,
        method='random',
        feature_ranges=feature_ranges,
        immutable_features=immutable,
        max_iterations=300
    )

    # 9. Plot: Counterfactual diversity
    print("  - Plotting counterfactual diversity...")
    plot_counterfactual_diversity(
        cf_set,
        save_path=str(cf_dir / 'counterfactual_diversity.pdf')
    )

    # 10. Save text explanation
    print("  - Saving counterfactual explanation...")
    with open(cf_dir / 'counterfactual_explanation.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Counterfactual Explanation for Patient {patient_idx}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"ORIGINAL PREDICTION: Class {cf.original_prediction} (HIGH RISK)\n")
        f.write(f"COUNTERFACTUAL PREDICTION: Class {cf.counterfactual_prediction} (LOW RISK)\n")
        f.write(f"DISTANCE: {cf.distance:.4f}\n")
        f.write(f"FEASIBLE: {cf.feasible}\n")
        f.write(f"ACTIONABLE: {cf.actionable}\n\n")

        f.write("REQUIRED CHANGES:\n")
        f.write("-" * 80 + "\n")
        for feat, (old, new) in cf.feature_changes.items():
            change = new - old
            pct = abs(change / old * 100) if old != 0 else 0
            f.write(f"  {feat:20s}: {old:8.2f} → {new:8.2f}  "
                   f"(change: {change:+.2f}, {pct:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CLINICAL INTERVENTIONS (Priority Order):\n")
        f.write("-" * 80 + "\n")
        for interv in interventions:
            f.write(f"{interv.priority}. {interv.feature_name} ({interv.intervention_type}):\n")
            f.write(f"   Current: {interv.current_value:.2f}\n")
            f.write(f"   Target:  {interv.target_value:.2f}\n")
            f.write(f"   Change:  {interv.change_magnitude:.2f} "
                   f"({interv.change_percentage:.1f}%)\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        f.write("Counterfactual explanations answer: 'What needs to change for this\n")
        f.write("patient to be triaged as LOW risk instead of HIGH risk?'\n\n")
        f.write("This provides:\n")
        f.write("  1. Actionable insights for clinicians\n")
        f.write("  2. Identification of modifiable risk factors\n")
        f.write("  3. Potential intervention strategies\n")
        f.write("  4. Support for clinical decision-making\n")
        f.write("=" * 80 + "\n")

    print(f"[OK] Generated 9 counterfactual figure types")


def main():
    parser = argparse.ArgumentParser(
        description='Generate XAI (SHAP + Counterfactual) figures for BASICS-CDSS'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='xai_figures',
        help='Output directory for figures (default: xai_figures)'
    )
    parser.add_argument(
        '--shap-only',
        action='store_true',
        help='Generate only SHAP figures'
    )
    parser.add_argument(
        '--cf-only',
        action='store_true',
        help='Generate only counterfactual figures'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of synthetic samples (default: 500)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BASICS-CDSS XAI Figure Generation")
    print("=" * 80)
    print(f"Output directory: {Path(args.output_dir).absolute()}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate synthetic data
    print("[*] Generating synthetic clinical data...")
    X, y, feature_names, feature_descriptions = create_synthetic_clinical_data(
        n_samples=args.n_samples
    )
    print(f"  Generated {len(X)} samples with {len(feature_names)} features")
    print(f"  Class distribution: {np.sum(y==0)} low-risk, {np.sum(y==1)} high-risk")

    # Train model
    print("\n[*] Training demonstration model...")
    model, X_train, X_test, y_train, y_test = train_demo_model(X, y)

    # Generate figures
    if not args.cf_only:
        generate_shap_figures(model, X_test, y_test, feature_names, output_dir)

    if not args.shap_only:
        generate_counterfactual_figures(model, X_test, y_test, feature_names, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("[SUCCESS] All XAI figures generated successfully!")
    print("=" * 80)
    print(f"Output location: {output_dir.absolute()}")
    print()
    print("Figure Summary:")
    print("-" * 80)
    if not args.cf_only:
        print("  SHAP Figures:")
        print("    - shap_waterfall.pdf")
        print("    - shap_summary_beeswarm.pdf")
        print("    - shap_summary_bar.pdf")
        print("    - shap_feature_importance.pdf")
        print("    - shap_dependence_*.pdf (top 3 features)")
        print("    - shap_heatmap.pdf")
        print("    - shap_interaction_heatmap.pdf")
        print("    - game_theoretic_explanation.txt")
    if not args.shap_only:
        print("  Counterfactual Figures:")
        print("    - counterfactual_comparison.pdf")
        print("    - feature_changes.pdf")
        print("    - intervention_priority.pdf")
        print("    - whatif_sbp.pdf")
        print("    - counterfactual_diversity.pdf")
        print("    - counterfactual_explanation.txt")
    print("-" * 80)


if __name__ == '__main__':
    main()
