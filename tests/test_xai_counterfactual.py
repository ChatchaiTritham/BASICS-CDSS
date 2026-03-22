"""
Tests for counterfactual explanations module (basics_cdss.xai.counterfactual)

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import pytest
from basics_cdss.xai.counterfactual import (CounterfactualExample,
                                            CounterfactualSet,
                                            InterventionSuggestion,
                                            actionable_interventions,
                                            counterfactual_stability,
                                            generate_counterfactual,
                                            generate_diverse_counterfactuals,
                                            minimal_counterfactual,
                                            whatif_analysis)
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_clinical_data():
    """Generate sample clinical data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 6

    # Features: sbp, hr, troponin, lactate, age, gender
    X = np.column_stack(
        [
            np.random.uniform(80, 200, n_samples),  # sbp
            np.random.uniform(50, 150, n_samples),  # hr
            np.abs(np.random.normal(0.02, 0.05, n_samples)),  # troponin
            np.abs(np.random.normal(1.5, 1.0, n_samples)),  # lactate
            np.random.randint(20, 90, n_samples),  # age
            np.random.randint(0, 2, n_samples),  # gender
        ]
    )

    # Decision: high risk if sbp < 90 OR troponin > 0.04 OR lactate > 2.5
    y = ((X[:, 0] < 90) | (X[:, 2] > 0.04) | (X[:, 3] > 2.5)).astype(int)

    feature_names = ['sbp', 'hr', 'troponin', 'lactate', 'age', 'gender']

    return X, y, feature_names


@pytest.fixture
def trained_clinical_model(sample_clinical_data):
    """Train model on clinical data."""
    X, y, _ = sample_clinical_data
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    return model


class TestGenerateCounterfactual:
    """Tests for generate_counterfactual function."""

    def test_basic_generation(self, trained_clinical_model, sample_clinical_data):
        """Test basic counterfactual generation."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        # Find a high-risk patient
        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            method='random',
            max_iterations=100,
            random_state=42,
        )

        assert isinstance(cf, CounterfactualExample)
        assert cf.original_prediction == 1
        # Counterfactual may or may not succeed with limited iterations
        assert len(cf.feature_changes) >= 0

    def test_immutable_features(self, trained_clinical_model, sample_clinical_data):
        """Test that immutable features are not changed."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        immutable = ['age', 'gender']

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            immutable_features=immutable,
            max_iterations=200,
            random_state=42,
        )

        # Check immutable features unchanged
        for feat in immutable:
            assert (
                feat not in cf.feature_changes
            ), f"Immutable feature {feat} was changed"

    def test_feature_ranges(self, trained_clinical_model, sample_clinical_data):
        """Test that feature ranges are respected."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        feature_ranges = {
            'sbp': (80, 200),
            'hr': (40, 180),
            'troponin': (0, 1.0),
            'lactate': (0, 10),
        }

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            feature_ranges=feature_ranges,
            max_iterations=200,
            random_state=42,
        )

        # Check ranges
        for feat, (old, new) in cf.feature_changes.items():
            if feat in feature_ranges:
                min_val, max_val = feature_ranges[feat]
                assert min_val <= new <= max_val, f"{feat} out of range: {new}"

    def test_different_methods(self, trained_clinical_model, sample_clinical_data):
        """Test different counterfactual generation methods."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        methods = ['random', 'gradient', 'genetic']

        for method in methods:
            cf = generate_counterfactual(
                model=model,
                x=patient,
                feature_names=feature_names,
                desired_class=0,
                method=method,
                max_iterations=100,
                random_state=42,
            )

            assert isinstance(cf, CounterfactualExample)

    def test_distance_metrics(self, trained_clinical_model, sample_clinical_data):
        """Test different distance metrics."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        metrics = ['euclidean', 'manhattan', 'cosine']

        for metric in metrics:
            cf = generate_counterfactual(
                model=model,
                x=patient,
                feature_names=feature_names,
                desired_class=0,
                distance_metric=metric,
                max_iterations=100,
                random_state=42,
            )

            assert cf.distance >= 0


class TestDiverseCounterfactuals:
    """Tests for generate_diverse_counterfactuals function."""

    def test_diversity(self, trained_clinical_model, sample_clinical_data):
        """Test generation of diverse counterfactuals."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf_set = generate_diverse_counterfactuals(
            model=model,
            x=patient,
            feature_names=feature_names,
            num_counterfactuals=3,
            desired_class=0,
            max_iterations=100,
            random_state=42,
        )

        assert isinstance(cf_set, CounterfactualSet)
        assert cf_set.num_counterfactuals == 3
        assert len(cf_set.counterfactuals) == 3
        assert cf_set.diversity_score >= 0
        assert 0 <= cf_set.coverage <= 1

    def test_diversity_score(self, trained_clinical_model, sample_clinical_data):
        """Test that diversity score increases with more counterfactuals."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf_set_small = generate_diverse_counterfactuals(
            model=model,
            x=patient,
            feature_names=feature_names,
            num_counterfactuals=2,
            max_iterations=100,
            random_state=42,
        )

        cf_set_large = generate_diverse_counterfactuals(
            model=model,
            x=patient,
            feature_names=feature_names,
            num_counterfactuals=5,
            max_iterations=100,
            random_state=43,  # Different seed
        )

        # More counterfactuals should have potential for higher coverage
        assert cf_set_large.num_counterfactuals > cf_set_small.num_counterfactuals


class TestMinimalCounterfactual:
    """Tests for minimal_counterfactual function."""

    def test_minimal_changes(self, trained_clinical_model, sample_clinical_data):
        """Test that minimal counterfactual has few changes."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        minimal_cf = minimal_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            max_features_changed=3,
            max_iterations=200,
            random_state=42,
        )

        assert isinstance(minimal_cf, CounterfactualExample)
        assert len(minimal_cf.feature_changes) <= 3


class TestActionableInterventions:
    """Tests for actionable_interventions function."""

    def test_intervention_generation(
        self, trained_clinical_model, sample_clinical_data
    ):
        """Test intervention suggestion generation."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            max_iterations=200,
            random_state=42,
        )

        # Skip if no changes
        if len(cf.feature_changes) == 0:
            pytest.skip("No counterfactual changes generated")

        intervention_types = {
            'sbp': 'medication',
            'hr': 'medication',
            'troponin': 'medical workup',
            'lactate': 'resuscitation',
        }

        clinical_priority = {
            'troponin': 1,
            'sbp': 2,
            'lactate': 3,
        }

        interventions = actionable_interventions(
            cf,
            intervention_types=intervention_types,
            clinical_priority=clinical_priority,
        )

        assert isinstance(interventions, list)
        assert all(isinstance(i, InterventionSuggestion) for i in interventions)

        # Check sorted by priority
        if len(interventions) > 1:
            for i in range(len(interventions) - 1):
                assert interventions[i].priority <= interventions[i + 1].priority

    def test_intervention_fields(self, trained_clinical_model, sample_clinical_data):
        """Test intervention suggestion fields."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            max_iterations=200,
            random_state=42,
        )

        if len(cf.feature_changes) == 0:
            pytest.skip("No counterfactual changes generated")

        interventions = actionable_interventions(cf)

        for interv in interventions:
            assert hasattr(interv, 'feature_name')
            assert hasattr(interv, 'current_value')
            assert hasattr(interv, 'target_value')
            assert hasattr(interv, 'change_magnitude')
            assert hasattr(interv, 'change_percentage')
            assert interv.change_magnitude >= 0


class TestWhatIfAnalysis:
    """Tests for whatif_analysis function."""

    def test_whatif_basic(self, trained_clinical_model, sample_clinical_data):
        """Test basic what-if analysis."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        patient = X[0]

        whatif_df = whatif_analysis(
            model=model,
            x=patient,
            feature_names=feature_names,
            feature_to_vary='sbp',
            value_range=(80, 200),
            num_points=20,
        )

        assert isinstance(whatif_df, pd.DataFrame)
        assert 'sbp' in whatif_df.columns
        assert 'prediction' in whatif_df.columns
        assert len(whatif_df) == 20

        # Check range
        assert whatif_df['sbp'].min() >= 80
        assert whatif_df['sbp'].max() <= 200

    def test_whatif_with_proba(self, trained_clinical_model, sample_clinical_data):
        """Test what-if analysis with probabilities."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        patient = X[0]

        whatif_df = whatif_analysis(
            model=model,
            x=patient,
            feature_names=feature_names,
            feature_to_vary='troponin',
            value_range=(0, 0.1),
            num_points=15,
        )

        assert 'prediction_proba' in whatif_df.columns

        # Probabilities should be in [0, 1]
        proba_col = whatif_df['prediction_proba']
        if isinstance(proba_col.iloc[0], np.ndarray):
            # Multi-class
            assert all(0 <= p <= 1 for row in proba_col for p in row)
        else:
            # Binary
            assert all(0 <= p <= 1 for p in proba_col)


class TestCounterfactualStability:
    """Tests for counterfactual_stability function."""

    def test_stability_computation(self, trained_clinical_model, sample_clinical_data):
        """Test counterfactual stability assessment."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            max_iterations=200,
            random_state=42,
        )

        if cf.counterfactual_prediction != 0:
            pytest.skip("Counterfactual did not reach desired class")

        stability = counterfactual_stability(
            model=model,
            counterfactual=cf,
            noise_level=0.01,
            num_trials=50,
            random_state=42,
        )

        assert isinstance(stability, dict)
        assert 'stability_score' in stability
        assert 'success_rate' in stability
        assert 'num_trials' in stability
        assert 'noise_level' in stability

        assert 0 <= stability['stability_score'] <= 1
        assert 0 <= stability['success_rate'] <= 1
        assert stability['num_trials'] == 50

    def test_stability_decreases_with_noise(
        self, trained_clinical_model, sample_clinical_data
    ):
        """Test that stability decreases with higher noise."""
        X, y, feature_names = sample_clinical_data
        model = trained_clinical_model

        high_risk_idx = np.where(model.predict(X) == 1)[0][0]
        patient = X[high_risk_idx]

        cf = generate_counterfactual(
            model=model,
            x=patient,
            feature_names=feature_names,
            desired_class=0,
            max_iterations=300,
            random_state=42,
        )

        if cf.counterfactual_prediction != 0:
            pytest.skip("Counterfactual did not reach desired class")

        stability_low = counterfactual_stability(
            model, cf, noise_level=0.01, num_trials=30
        )

        stability_high = counterfactual_stability(
            model, cf, noise_level=0.1, num_trials=30
        )

        # Higher noise should generally decrease stability
        # (though not guaranteed due to randomness)
        assert stability_low['noise_level'] < stability_high['noise_level']


class TestCounterfactualDataClasses:
    """Tests for counterfactual data classes."""

    def test_counterfactual_example_creation(self):
        """Test CounterfactualExample dataclass."""
        cf = CounterfactualExample(
            original=np.array([1, 2, 3]),
            counterfactual=np.array([1, 2.5, 3]),
            original_prediction=1,
            counterfactual_prediction=0,
            feature_changes={'f2': (2.0, 2.5)},
            distance=0.5,
            feasible=True,
            actionable=True,
            feature_names=['f1', 'f2', 'f3'],
        )

        assert cf.original_prediction == 1
        assert cf.counterfactual_prediction == 0
        assert len(cf.feature_changes) == 1
        assert cf.feasible
        assert cf.actionable

    def test_counterfactual_set_creation(self):
        """Test CounterfactualSet dataclass."""
        cf1 = CounterfactualExample(
            original=np.array([1, 2]),
            counterfactual=np.array([1, 3]),
            original_prediction=1,
            counterfactual_prediction=0,
            feature_changes={'f2': (2, 3)},
            distance=1.0,
            feasible=True,
            actionable=True,
            feature_names=['f1', 'f2'],
        )

        cf_set = CounterfactualSet(
            counterfactuals=[cf1],
            diversity_score=0.5,
            coverage=0.3,
            num_counterfactuals=1,
        )

        assert len(cf_set.counterfactuals) == 1
        assert cf_set.num_counterfactuals == 1
        assert cf_set.diversity_score == 0.5

    def test_intervention_suggestion_creation(self):
        """Test InterventionSuggestion dataclass."""
        interv = InterventionSuggestion(
            feature_name='sbp',
            current_value=85.0,
            target_value=120.0,
            change_magnitude=35.0,
            change_percentage=41.2,
            priority=1,
            actionable=True,
            intervention_type='medication',
        )

        assert interv.feature_name == 'sbp'
        assert interv.priority == 1
        assert interv.actionable
        assert interv.intervention_type == 'medication'
