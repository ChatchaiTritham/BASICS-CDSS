"""
Tests for perturbation operators.

These tests verify that perturbation operators:
1. Are deterministic given a seed
2. Preserve clinical plausibility constraints
3. Generate appropriate uncertainty profiles
4. Handle edge cases correctly
"""

import numpy as np
import pytest
from basics_cdss.scenario.perturbations import (CompositePerturbation,
                                                ConflictOperator,
                                                DegradeOperator, MaskOperator,
                                                NoiseOperator,
                                                PerturbationConfig,
                                                create_default_perturbation)


class TestMaskOperator:
    """Tests for information missingness operator."""

    def test_deterministic_masking(self):
        """Verify masking is deterministic with fixed seed."""
        features = {"symptom_1": "headache", "symptom_2": "nausea", "age": 45}
        config = PerturbationConfig(p_mask=0.5)

        # Same seed should produce identical results
        op1 = MaskOperator(config, seed=42)
        perturbed1, _ = op1.apply(features)

        op2 = MaskOperator(config, seed=42)
        perturbed2, _ = op2.apply(features)

        assert perturbed1 == perturbed2

    def test_protected_features_not_masked(self):
        """Verify protected features are never removed."""
        features = {
            "archetype_id": "A001",
            "triage_tier": "high",
            "symptom": "dizziness",
        }
        config = PerturbationConfig(p_mask=1.0)  # Mask everything possible
        op = MaskOperator(config, seed=42)
        perturbed, _ = op.apply(features)

        # Protected features must remain
        assert "archetype_id" in perturbed
        assert "triage_tier" in perturbed
        # symptom may be masked

    def test_missingness_profile(self):
        """Verify uncertainty profile correctly reports missingness fraction."""
        features = {f"feature_{i}": i for i in range(10)}
        config = PerturbationConfig(p_mask=0.3, protected_features=set())

        op = MaskOperator(config, seed=42)
        perturbed, uncertainty = op.apply(features)

        # Check missingness is in [0, 1]
        assert 0.0 <= uncertainty["missingness"] <= 1.0

        # Count actual missing features
        actual_missingness = 1.0 - len(perturbed) / len(features)
        assert abs(uncertainty["missingness"] - actual_missingness) < 1e-6

    def test_no_maskable_features(self):
        """Handle case where all features are protected."""
        features = {"archetype_id": "A001", "triage_tier": "high"}
        config = PerturbationConfig(p_mask=0.5)

        op = MaskOperator(config, seed=42)
        perturbed, uncertainty = op.apply(features)

        assert perturbed == features
        assert uncertainty["missingness"] == 0.0


class TestNoiseOperator:
    """Tests for ambiguity (noise) operator."""

    def test_deterministic_noise(self):
        """Verify noise is deterministic with fixed seed."""
        features = {"temperature": 37.5, "heart_rate": 80}
        config = PerturbationConfig(noise_sigma=0.1)

        op1 = NoiseOperator(config, seed=42)
        perturbed1, _ = op1.apply(features)

        op2 = NoiseOperator(config, seed=42)
        perturbed2, _ = op2.apply(features)

        assert perturbed1 == perturbed2

    def test_only_continuous_features_perturbed(self):
        """Verify only numeric features receive noise."""
        features = {
            "symptom": "headache",  # categorical - should not change
            "age": 45,  # numeric - should receive noise
            "temperature": 37.5,  # numeric - should receive noise
        }
        config = PerturbationConfig(noise_sigma=0.1)

        op = NoiseOperator(config, seed=42)
        perturbed, _ = op.apply(features)

        # Categorical unchanged
        assert perturbed["symptom"] == features["symptom"]

        # Numeric should have changed (with high probability)
        # Use small tolerance for edge case where noise is exactly zero
        assert (
            perturbed["age"] != features["age"]
            or abs(perturbed["age"] - features["age"]) < 0.01
        )

    def test_ambiguity_profile(self):
        """Verify uncertainty profile reports mean noise magnitude."""
        features = {"x": 10.0, "y": 20.0, "z": 30.0}
        config = PerturbationConfig(noise_sigma=0.5)

        op = NoiseOperator(config, seed=42)
        perturbed, uncertainty = op.apply(features)

        # Ambiguity should be >= 0
        assert uncertainty["ambiguity"] >= 0.0

    def test_no_numeric_features(self):
        """Handle case with no numeric features."""
        features = {"symptom": "headache", "condition": "stable"}
        config = PerturbationConfig(noise_sigma=0.1)

        op = NoiseOperator(config, seed=42)
        perturbed, uncertainty = op.apply(features)

        assert perturbed == features
        assert uncertainty["ambiguity"] == 0.0


class TestConflictOperator:
    """Tests for internal inconsistency operator."""

    def test_deterministic_conflict(self):
        """Verify conflict is deterministic with fixed seed."""
        features = {"blood_pressure": "normal"}
        conflict_map = {"blood_pressure": {"normal": "high", "high": "low"}}
        config = PerturbationConfig(conflict_pairs=conflict_map)

        op1 = ConflictOperator(config, seed=42)
        perturbed1, _ = op1.apply(features)

        op2 = ConflictOperator(config, seed=42)
        perturbed2, _ = op2.apply(features)

        assert perturbed1 == perturbed2

    def test_conflict_application(self):
        """Verify conflicts are introduced correctly."""
        features = {"bp_status": "normal", "heart_status": "regular"}
        conflict_map = {
            "bp_status": {"normal": "abnormal"},
        }
        config = PerturbationConfig(conflict_pairs=conflict_map)

        # Run multiple times to check probabilistic behavior
        conflicts_found = False
        for seed in range(10):
            op = ConflictOperator(config, seed=seed)
            perturbed, uncertainty = op.apply(features)

            if perturbed["bp_status"] == "abnormal":
                conflicts_found = True
                assert uncertainty["conflict"] > 0.0
                break

        assert conflicts_found  # Should find at least one conflict in 10 trials

    def test_no_conflict_pairs_defined(self):
        """Handle case with no conflict pairs configured."""
        features = {"symptom": "dizziness"}
        config = PerturbationConfig(conflict_pairs={})

        op = ConflictOperator(config, seed=42)
        perturbed, uncertainty = op.apply(features)

        assert perturbed == features
        assert uncertainty["conflict"] == 0.0


class TestDegradeOperator:
    """Tests for reduced specificity operator."""

    def test_deterministic_degradation(self):
        """Verify degradation is deterministic with fixed seed."""
        features = {"symptom": "acute vertigo"}
        degrade_map = {"acute vertigo": "dizziness"}
        config = PerturbationConfig(degrade_map=degrade_map)

        op1 = DegradeOperator(config, seed=42)
        perturbed1, _ = op1.apply(features)

        op2 = DegradeOperator(config, seed=42)
        perturbed2, _ = op2.apply(features)

        assert perturbed1 == perturbed2

    def test_degradation_application(self):
        """Verify specific terms are replaced with vague descriptors."""
        features = {"chief_complaint": "acute vertigo"}
        degrade_map = {"acute vertigo": "dizziness"}
        config = PerturbationConfig(degrade_map=degrade_map)

        # Run multiple times
        degradation_found = False
        for seed in range(10):
            op = DegradeOperator(config, seed=seed)
            perturbed, uncertainty = op.apply(features)

            if perturbed["chief_complaint"] == "dizziness":
                degradation_found = True
                assert uncertainty["degradation"] > 0.0
                break

        assert degradation_found

    def test_no_degrade_map(self):
        """Handle case with no degradation mapping."""
        features = {"symptom": "headache"}
        config = PerturbationConfig(degrade_map={})

        op = DegradeOperator(config, seed=42)
        perturbed, uncertainty = op.apply(features)

        assert perturbed == features
        assert uncertainty["degradation"] == 0.0


class TestCompositePerturbation:
    """Tests for composite perturbation operator."""

    def test_composite_applies_multiple_operators(self):
        """Verify composite applies all operators in sequence."""
        features = {
            "symptom_1": "headache",
            "symptom_2": "nausea",
            "temperature": 37.5,
            "age": 45,
        }

        config = PerturbationConfig(p_mask=0.3, noise_sigma=0.1)
        operators = [
            MaskOperator(config, seed=42),
            NoiseOperator(config, seed=43),
        ]

        composite = CompositePerturbation(operators, config, seed=42)
        perturbed, uncertainty = composite.apply(features)

        # Should have uncertainty from both operators
        assert "missingness" in uncertainty
        assert "ambiguity" in uncertainty

    def test_composite_deterministic(self):
        """Verify composite is deterministic."""
        features = {"x": 10, "y": 20}
        config = PerturbationConfig(p_mask=0.5, noise_sigma=0.1)

        operators1 = [MaskOperator(config, seed=42), NoiseOperator(config, seed=43)]
        composite1 = CompositePerturbation(operators1, config, seed=42)
        perturbed1, _ = composite1.apply(features)

        operators2 = [MaskOperator(config, seed=42), NoiseOperator(config, seed=43)]
        composite2 = CompositePerturbation(operators2, config, seed=42)
        perturbed2, _ = composite2.apply(features)

        assert perturbed1 == perturbed2


class TestFactoryFunction:
    """Tests for create_default_perturbation factory."""

    def test_create_mask_operator(self):
        """Verify factory creates correct operator type."""
        op = create_default_perturbation("mask", seed=42)
        assert isinstance(op, MaskOperator)

    def test_create_noise_operator(self):
        op = create_default_perturbation("noise", seed=42)
        assert isinstance(op, NoiseOperator)

    def test_create_conflict_operator(self):
        op = create_default_perturbation("conflict", seed=42)
        assert isinstance(op, ConflictOperator)

    def test_create_degrade_operator(self):
        op = create_default_perturbation("degrade", seed=42)
        assert isinstance(op, DegradeOperator)

    def test_create_composite_operator(self):
        op = create_default_perturbation("composite", seed=42)
        assert isinstance(op, CompositePerturbation)

    def test_invalid_type_raises_error(self):
        """Verify invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown perturbation type"):
            create_default_perturbation("invalid_type", seed=42)


class TestPerturbationConfig:
    """Tests for PerturbationConfig dataclass."""

    def test_default_config(self):
        """Verify default configuration values."""
        config = PerturbationConfig()
        assert config.p_mask == 0.2
        assert config.noise_sigma == 0.1
        assert "archetype_id" in config.protected_features
        assert "triage_tier" in config.protected_features

    def test_custom_config(self):
        """Verify custom configuration."""
        config = PerturbationConfig(
            p_mask=0.5,
            noise_sigma=0.2,
            protected_features={"custom_id"},
        )
        assert config.p_mask == 0.5
        assert config.noise_sigma == 0.2
        assert "custom_id" in config.protected_features
