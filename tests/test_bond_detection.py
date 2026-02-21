"""Basic tests for bond detection.

Verifies extraction from GraphBuilder works correctly.
Tests use molecular contexts: methane, water, diatomic H2.
"""

import pytest

from xyzgraph.bond_detection import BondDetector
from xyzgraph.bond_geometry_check import BondGeometryChecker
from xyzgraph.data_loader import DATA
from xyzgraph.geometry import GeometryCalculator
from xyzgraph.parameters import BondThresholds, GeometryThresholds


@pytest.fixture
def detector():
    """BondDetector with default thresholds."""
    geometry = GeometryCalculator()
    bond_checker = BondGeometryChecker(
        geometry=geometry,
        thresholds=GeometryThresholds.strict(),
        data=DATA,
    )
    return BondDetector(
        geometry=geometry,
        bond_checker=bond_checker,
        thresholds=BondThresholds(),
        data=DATA,
    )


def test_methane_connectivity(detector):
    """Methane: 5 atoms, 4 C-H bonds."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    # All bonds are C-H
    for i, j in G.edges():
        syms = {G.nodes[i]["symbol"], G.nodes[j]["symbol"]}
        assert syms == {"C", "H"}


def test_water_connectivity(detector):
    """Water: 3 atoms, 2 O-H bonds."""
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.757, 0.586, 0.0)),
        ("H", (-0.757, 0.586, 0.0)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2


def test_h2_diatomic(detector):
    """H2: 2 atoms, 1 H-H bond."""
    atoms = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.74, 0.0, 0.0)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_formula_hill_system(detector):
    """Chemical formula follows Hill system (C, H, then alphabetical)."""
    from xyzgraph.utils import compute_formula

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("C", (1.2, 0.0, 0.0)),
        ("H", (1.8, 0.9, 0.0)),
    ]
    G = detector.detect(atoms)
    compute_formula(G)
    assert G.graph["formula"] == "CHO"


def test_user_specified_bond(detector):
    """User-specified bonds are added."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (5.0, 0.0, 0.0)),  # Too far for automatic detection
    ]
    G = detector.detect(atoms, bond=[(0, 1)])
    assert G.has_edge(0, 1)


def test_user_specified_unbond(detector):
    """User-specified unbonds remove detected bonds."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629)),
    ]
    G = detector.detect(atoms, unbond=[(0, 1)])
    assert not G.has_edge(0, 1)
    assert G.number_of_edges() == 3  # 4 - 1


def test_benzene_ring_detection(detector):
    """Benzene: 6 C-C bonds form a ring."""
    atoms = [
        ("C", (1.396, 0.000, 0.000)),
        ("C", (0.698, 1.209, 0.000)),
        ("C", (-0.698, 1.209, 0.000)),
        ("C", (-1.396, 0.000, 0.000)),
        ("C", (-0.698, -1.209, 0.000)),
        ("C", (0.698, -1.209, 0.000)),
        ("H", (2.479, 0.000, 0.000)),
        ("H", (1.240, 2.147, 0.000)),
        ("H", (-1.240, 2.147, 0.000)),
        ("H", (-2.479, 0.000, 0.000)),
        ("H", (-1.240, -2.147, 0.000)),
        ("H", (1.240, -2.147, 0.000)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 12
    assert G.number_of_edges() == 12  # 6 C-C + 6 C-H
    assert len(G.graph["_rings"]) >= 1  # At least one ring detected


def test_unbond_invalidates_ring_cache(detector):
    """Ring cache must not reference edges removed by unbond.

    Regression test: unbond removing a ring edge left a stale _rings cache,
    causing KeyError in downstream bond-order optimization when iterating
    cached ring edges that no longer exist in the graph.
    """
    # Benzene geometry — has a 6-membered ring
    atoms = [
        ("C", (1.396, 0.000, 0.000)),
        ("C", (0.698, 1.209, 0.000)),
        ("C", (-0.698, 1.209, 0.000)),
        ("C", (-1.396, 0.000, 0.000)),
        ("C", (-0.698, -1.209, 0.000)),
        ("C", (0.698, -1.209, 0.000)),
        ("H", (2.479, 0.000, 0.000)),
        ("H", (1.240, 2.147, 0.000)),
        ("H", (-1.240, 2.147, 0.000)),
        ("H", (-2.479, 0.000, 0.000)),
        ("H", (-1.240, -2.147, 0.000)),
        ("H", (1.240, -2.147, 0.000)),
    ]
    # Remove edge (0,1) which is part of the ring
    G = detector.detect(atoms, unbond=[(0, 1)])
    assert not G.has_edge(0, 1)

    # Ring cache must not contain any ring that references the removed edge
    for ring in G.graph["_rings"]:
        edges = list(zip(ring, ring[1:] + [ring[0]]))
        for i, j in edges:
            assert G.has_edge(i, j) or G.has_edge(j, i), (
                f"Stale ring cache: ring {ring} references removed edge ({i}, {j})"
            )
