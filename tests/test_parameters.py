"""Tests for algorithm parameter dataclasses."""

from xyzgraph.parameters import BondThresholds, GeometryThresholds, OptimizerConfig, ScoringWeights


def test_geometry_strict_equals_default():
    """strict() produces same values as default constructor."""
    assert GeometryThresholds.strict() == GeometryThresholds()


def test_geometry_relaxed_more_permissive():
    """Relaxed thresholds are more permissive than strict."""
    s = GeometryThresholds.strict()
    r = GeometryThresholds.relaxed()
    assert r.acute_threshold_nonmetal < s.acute_threshold_nonmetal


def test_all_defaults_instantiate():
    """All parameter classes instantiate with defaults."""
    assert ScoringWeights().violation_weight == 1000.0
    assert OptimizerConfig().max_iter == 50
    assert BondThresholds().threshold == 1.0
