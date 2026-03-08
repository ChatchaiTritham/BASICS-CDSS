"""
Tests for SHAP analysis module (basics_cdss.xai.shap_analysis)

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Test if SHAP is available
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from basics_cdss.xai.shap_analysis import (FeatureImportance,
                                           GameTheoreticExplanation,
                                           SHAPValues, compute_shap_values,
                                           feature_importance_ranking,
                                           game_theoretic_explanation,
                                           shap_based_feature_selection,
                                           shapley_coalition_values,
                                           stratified_shap_analysis)


@pytest.fixture
def sample_data():
    """Generate sample clinical data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple decision boundary

    feature_names = ['troponin', 'sbp', 'age', 'hr', 'lactate']

    return X, y, feature_names


@pytest.fixture
def trained_model(sample_data):
    """Train a simple model for testing."""
    X, y, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


class TestComputeSHAPValues:
    """Tests for compute_shap_values function."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_basic_computation(self, trained_model, sample_data):
        """Test basic SHAP value computation."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        assert isinstance(shap_vals, SHAPValues)
        assert shap_vals.values.shape[:2] == X.shape
        assert len(shap_vals.feature_names) == len(feature_names)
        base_val = shap_vals.base_value
        if hasattr(base_val, '__len__'):
            base_val = base_val[0] if len(base_val) > 0 else base_val
        assert isinstance(base_val, (int, float, np.number))

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_additivity(self, trained_model, sample_data):
        """Test SHAP additivity property: sum of SHAP values equals prediction - base."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model,
            X=X[:10],  # Small sample for speed
            feature_names=feature_names,
            model_type='tree',
        )

        predictions = trained_model.predict_proba(X[:10])[:, 1]

        for i in range(10):
            shap_values_2d = shap_vals.values[i]
            base_val = shap_vals.base_value
            if hasattr(base_val, '__len__'):
                base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
            if shap_values_2d.ndim == 2:
                shap_sum = base_val + shap_values_2d[:, 1].sum() if shap_values_2d.shape[1] > 1 else base_val + shap_values_2d.sum()
            else:
                shap_sum = base_val + shap_values_2d.sum()
            pred = float(predictions[i])
            assert np.abs(shap_sum - pred) < 0.01, f"Additivity violated for sample {i}"

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_auto_model_type(self, trained_model, sample_data):
        """Test automatic model type detection."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model,
            X=X[:20],
            feature_names=feature_names,
            model_type='auto',  # Should detect tree
        )

        assert shap_vals is not None
        assert shap_vals.values.shape[0] == 20

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_linear_model(self, sample_data):
        """Test SHAP with linear model."""
        X, y, feature_names = sample_data

        # Train linear model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        shap_vals = compute_shap_values(
            model=model, X=X[:20], feature_names=feature_names, model_type='linear'
        )

        assert shap_vals.values.shape == (20, len(feature_names))


class TestFeatureImportanceRanking:
    """Tests for feature_importance_ranking function."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_ranking(self, trained_model, sample_data):
        """Test feature importance ranking."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        importance = feature_importance_ranking(
            shap_vals, method='mean_abs', threshold_percentile=75
        )

        assert isinstance(importance, FeatureImportance)
        assert len(importance.feature_names) == len(feature_names)
        assert len(importance.importance_scores) == len(feature_names)
        assert len(importance.importance_rank) == len(feature_names)

        # Check that scores are non-negative
        assert np.all(importance.importance_scores >= 0)

        # Check that ranks are 1-indexed and consecutive
        assert set(importance.importance_rank) == set(range(1, len(feature_names) + 1))

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_critical_noncritical_split(self, trained_model, sample_data):
        """Test critical/non-critical feature split."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        importance = feature_importance_ranking(
            shap_vals, threshold_percentile=80  # Top 20% are critical
        )

        # Critical + non-critical should equal all features
        all_features = set(
            importance.critical_features + importance.non_critical_features
        )
        assert all_features == set(feature_names)

        # No overlap
        assert set(importance.critical_features).isdisjoint(
            set(importance.non_critical_features)
        )

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_different_methods(self, trained_model, sample_data):
        """Test different importance calculation methods."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        methods = ['mean_abs', 'mean', 'max_abs', 'std']

        for method in methods:
            importance = feature_importance_ranking(shap_vals, method=method)
            assert importance is not None
            assert len(importance.importance_scores) == len(feature_names)


class TestShapleyCoalitionValues:
    """Tests for shapley_coalition_values function."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_coalition_values(self, trained_model, sample_data):
        """Test coalition value computation."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        coalitions = shapley_coalition_values(shap_vals, sample_idx=0, top_k=5)

        assert 'individual_contributions' in coalitions
        assert 'top_coalitions' in coalitions
        assert 'cumulative_contribution' in coalitions
        assert 'base_value' in coalitions

        # Check all features present
        assert set(coalitions['individual_contributions'].keys()) == set(feature_names)

        # Top coalitions should be subset
        assert len(coalitions['top_coalitions']) <= 5


class TestGameTheoreticExplanation:
    """Tests for game_theoretic_explanation function."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_game_explanation(self, trained_model, sample_data):
        """Test game-theoretic explanation generation."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        explanation = game_theoretic_explanation(
            shap_vals, sample_idx=0, major_player_percentile=75
        )

        assert isinstance(explanation, GameTheoreticExplanation)
        assert isinstance(explanation.major_players, dict)
        assert isinstance(explanation.minor_players, dict)
        assert isinstance(explanation.marginal_contributions, dict)

        # Major + minor should equal all features
        all_players = set(explanation.major_players.keys()) | set(
            explanation.minor_players.keys()
        )
        assert all_players == set(feature_names)

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_major_minor_split(self, trained_model, sample_data):
        """Test major/minor player classification."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        explanation = game_theoretic_explanation(
            shap_vals, sample_idx=0, major_player_percentile=80
        )

        # Should have at least one minor player with 80th percentile
        assert len(explanation.minor_players) > 0

        # Major players should have higher absolute values than minor
        if len(explanation.major_players) > 0 and len(explanation.minor_players) > 0:
            max_minor = max(abs(v) for v in explanation.minor_players.values())
            min_major = min(abs(v) for v in explanation.major_players.values())
            assert min_major >= max_minor


class TestStratifiedSHAPAnalysis:
    """Tests for stratified_shap_analysis function."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_stratified_analysis(self, trained_model, sample_data):
        """Test SHAP analysis stratified by groups."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        # Create strata (risk tiers)
        strata = np.random.choice(['low', 'high'], size=len(X))

        stratified = stratified_shap_analysis(
            shap_vals, strata=strata, strata_names=['low', 'high']
        )

        assert 'low' in stratified
        assert 'high' in stratified

        for tier, importance in stratified.items():
            assert isinstance(importance, FeatureImportance)
            assert len(importance.feature_names) == len(feature_names)


class TestSHAPFeatureSelection:
    """Tests for shap_based_feature_selection function."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_top_n_selection(self, trained_model, sample_data):
        """Test selecting top N features."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        selected = shap_based_feature_selection(shap_vals, n_features=3)

        assert len(selected) == 3
        assert all(feat in feature_names for feat in selected)

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not installed")
    def test_threshold_selection(self, trained_model, sample_data):
        """Test selecting features by threshold."""
        X, y, feature_names = sample_data

        shap_vals = compute_shap_values(
            model=trained_model, X=X, feature_names=feature_names, model_type='tree'
        )

        selected = shap_based_feature_selection(shap_vals, importance_threshold=0.1)

        assert len(selected) > 0
        assert all(feat in feature_names for feat in selected)


class TestSHAPDataClasses:
    """Tests for SHAP data classes."""

    def test_shap_values_creation(self):
        """Test SHAPValues dataclass creation."""
        values = np.random.randn(10, 5)
        base_value = 0.5
        data = np.random.randn(10, 5)
        features = ['f1', 'f2', 'f3', 'f4', 'f5']

        shap_vals = SHAPValues(
            values=values, base_value=base_value, data=data, feature_names=features
        )

        assert np.array_equal(shap_vals.values, values)
        assert shap_vals.base_value == base_value
        assert shap_vals.expected_value == base_value  # Auto-set
        assert shap_vals.feature_names == features

    def test_feature_importance_creation(self):
        """Test FeatureImportance dataclass creation."""
        features = ['f1', 'f2', 'f3']
        scores = np.array([0.5, 0.3, 0.1])
        ranks = np.array([1, 2, 3])

        importance = FeatureImportance(
            feature_names=features,
            importance_scores=scores,
            importance_rank=ranks,
            critical_features=['f1'],
            non_critical_features=['f2', 'f3'],
            threshold=0.4,
        )

        assert importance.feature_names == features
        assert len(importance.critical_features) == 1
        assert len(importance.non_critical_features) == 2
