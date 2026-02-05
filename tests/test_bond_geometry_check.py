"""Basic tests for bond geometry checking.

Verifies extraction from GraphBuilder works correctly.
Tests use molecular contexts: acute angles, collinear bonds, 3-ring diagonals.
"""

import networkx as nx
import pytest

from xyzgraph.bond_geometry_check import BondGeometryChecker
from xyzgraph.config_classes import GeometryThresholds
from xyzgraph.data_loader import DATA
from xyzgraph.geometry import GeometryCalculator


@pytest.fixture
def checker_strict():
    """BondGeometryChecker with strict thresholds."""
    return BondGeometryChecker(
        geometry=GeometryCalculator(),
        thresholds=GeometryThresholds.strict(),
        data=DATA,
    )


@pytest.fixture
def checker_relaxed():
    """BondGeometryChecker with relaxed thresholds."""
    return BondGeometryChecker(
        geometry=GeometryCalculator(),
        thresholds=GeometryThresholds.relaxed(),
        data=DATA,
    )


def _make_graph_with_atoms(atom_data):
    """Helper: create graph with atoms but no edges.

    atom_data: list of (symbol, position, atomic_number) tuples
    """
    G = nx.Graph()
    for idx, (sym, pos, z) in enumerate(atom_data):
        G.add_node(idx, symbol=sym, position=pos, atomic_number=z)
    return G


def test_isolated_atoms_always_valid(checker_strict):
    """Bond between two isolated atoms is always valid."""
    G = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("H", (1.09, 0.0, 0.0), 1),
    ])
    assert checker_strict.check(G, 0, 1, distance=1.09, confidence=0.8)


def test_acute_angle_rejected(checker_strict):
    """Bond creating acute angle with existing bond is rejected."""
    # C at origin with existing bond to H1, proposed bond to H2 at acute angle
    G = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("H", (1.0, 0.0, 0.0), 1),
        ("H", (1.0, 0.1, 0.0), 1),  # ~5.7° from H1
    ])
    G.add_edge(0, 1, bond_order=1.0, distance=1.0)

    # Angle C-H1 to C-H2 is ~5.7° which is below 35° (nonmetal strict threshold)
    assert not checker_strict.check(G, 0, 2, distance=1.005, confidence=0.8)


def test_normal_tetrahedral_angle_accepted(checker_strict):
    """Bond at tetrahedral angle with existing bond is accepted."""
    # C at origin, existing bond to H1, proposed bond to H2 at ~109.5°
    G = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("H", (0.629, 0.629, 0.629), 1),
        ("H", (-0.629, -0.629, 0.629), 1),
    ])
    G.add_edge(0, 1, bond_order=1.0, distance=1.09)

    assert checker_strict.check(G, 0, 2, distance=1.09, confidence=0.8)


def test_relaxed_allows_tighter_angles(checker_strict, checker_relaxed):
    """Relaxed thresholds accept angles that strict rejects."""
    # Angle of ~25° - rejected by strict (35°) but accepted by relaxed (20°)
    G_strict = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("C", (1.5, 0.0, 0.0), 6),
        ("C", (1.5, 0.65, 0.0), 6),  # ~23° angle at atom 0
    ])
    G_strict.add_edge(0, 1, bond_order=1.0, distance=1.5)

    G_relaxed = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("C", (1.5, 0.0, 0.0), 6),
        ("C", (1.5, 0.65, 0.0), 6),
    ])
    G_relaxed.add_edge(0, 1, bond_order=1.0, distance=1.5)

    result_strict = checker_strict.check(G_strict, 0, 2, distance=1.63, confidence=0.8)
    result_relaxed = checker_relaxed.check(G_relaxed, 0, 2, distance=1.63, confidence=0.8)

    assert not result_strict  # 23° < 35° strict threshold
    assert result_relaxed  # 23° > 20° relaxed threshold


def test_overlapping_bond_direction_rejected(checker_strict):
    """Bond nearly parallel to existing bond is rejected (acute angle ~0°)."""
    G = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("H", (1.0, 0.0, 0.0), 1),
        ("H", (0.5, 0.0, 0.0), 1),  # Between C and H1, ~0° angle
        ("H", (0.0, 1.0, 0.0), 1),
    ])
    G.add_edge(0, 1, bond_order=1.0, distance=1.0)
    G.add_edge(0, 3, bond_order=1.0, distance=1.0)

    # ~0° angle at C between C-H1 and C-H2, well below 35° strict threshold
    assert not checker_strict.check(G, 0, 2, distance=0.5, confidence=0.3)


def test_collinear_opposite_direction_accepted(checker_strict):
    """Collinear bond in opposite direction (trans) is accepted."""
    # Linear arrangement: H1-C-H2 with H2 on opposite side
    G = _make_graph_with_atoms([
        ("C", (0.0, 0.0, 0.0), 6),
        ("H", (1.0, 0.0, 0.0), 1),
        ("H", (-1.0, 0.0, 0.0), 1),  # Opposite direction
        ("H", (0.0, 1.0, 0.0), 1),  # Extra neighbor for degree >= 2
    ])
    G.add_edge(0, 1, bond_order=1.0, distance=1.0)
    G.add_edge(0, 3, bond_order=1.0, distance=1.0)

    assert checker_strict.check(G, 0, 2, distance=1.0, confidence=0.8)
