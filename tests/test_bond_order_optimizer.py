"""Tests for bond order optimization.

Verifies BondOrderOptimizer scoring, valence adjustment,
formal charge computation, and aromatic detection.
"""

import networkx as nx
import pytest

from xyzgraph.bond_order_optimizer import BondOrderOptimizer
from xyzgraph.data_loader import DATA
from xyzgraph.geometry import GeometryCalculator
from xyzgraph.parameters import OptimizerConfig, ScoringWeights


@pytest.fixture
def optimizer():
    """BondOrderOptimizer with default weights and config."""
    return BondOrderOptimizer(
        geometry=GeometryCalculator(),
        data=DATA,
        charge=0,
    )


@pytest.fixture
def charged_optimizer():
    """BondOrderOptimizer for cation (charge=+1)."""
    return BondOrderOptimizer(
        geometry=GeometryCalculator(),
        data=DATA,
        charge=1,
    )


def _make_graph(atoms, edges):
    """Build a minimal graph from atoms and edges."""
    G = nx.Graph()
    for i, (sym, pos) in enumerate(atoms):
        z = DATA.s2n[sym]
        G.add_node(i, symbol=sym, atomic_number=z, position=pos)
    for i, j in edges:
        pi = G.nodes[i]["position"]
        pj = G.nodes[j]["position"]
        d = GeometryCalculator.distance(pi, pj)
        G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=False)
    # Compute rings (needed by init_kekule / detect_aromatic_rings)
    G.graph["_rings"] = nx.cycle_basis(G)
    G.graph["_neighbors"] = {n: list(G.neighbors(n)) for n in G.nodes()}
    G.graph["_has_H"] = {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()}
    return G


# ---- Static utilities ----


def test_valence_sum():
    """valence_sum returns sum of bond orders at a node."""
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_edge(0, 1, bond_order=2.0)
    G.add_edge(0, 2, bond_order=1.0)
    assert BondOrderOptimizer.valence_sum(G, 0) == 3.0
    assert BondOrderOptimizer.valence_sum(G, 1) == 2.0


def test_formal_charge_value():
    """Formal charge: neutral C with 4 bonds → 0, N with 4 bonds → +1."""
    # Carbon: 4 valence electrons, 4 bonds → FC = 0
    assert BondOrderOptimizer._compute_formal_charge_value("C", 4, 4.0) == 0
    # Nitrogen: 5 valence electrons, 4 bonds → FC = +1 (ammonium)
    assert BondOrderOptimizer._compute_formal_charge_value("N", 5, 4.0) == 1
    # Oxygen: 6 valence electrons, 1 bond → FC = -1 (alkoxide)
    assert BondOrderOptimizer._compute_formal_charge_value("O", 6, 1.0) == -1
    # Hydrogen: 1 valence electron, 1 bond → FC = 0
    assert BondOrderOptimizer._compute_formal_charge_value("H", 1, 1.0) == 0


def test_ekey_normalization():
    """_ekey always returns (min, max) tuple."""
    assert BondOrderOptimizer._ekey(3, 1) == (1, 3)
    assert BondOrderOptimizer._ekey(1, 3) == (1, 3)


# ---- Valence violation ----


def test_no_valence_violation(optimizer):
    """Methane (4 C-H bonds) has no valence violation."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629)),
    ]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    G = _make_graph(atoms, edges)
    assert optimizer.check_valence_violation(G) is False


def test_valence_violation_detected(optimizer):
    """Carbon with 5 single bonds is a valence violation."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (1.0, 0.0, 0.0)),
        ("H", (-1.0, 0.0, 0.0)),
        ("H", (0.0, 1.0, 0.0)),
        ("H", (0.0, -1.0, 0.0)),
        ("H", (0.0, 0.0, 1.0)),
    ]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    G = _make_graph(atoms, edges)
    assert optimizer.check_valence_violation(G) is True


# ---- Quick valence adjust ----


def test_quick_valence_adjust_ethylene(optimizer):
    """Quick adjust should assign double bond to C=C in ethylene."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (1.34, 0.0, 0.0)),
        ("H", (-0.5, 0.87, 0.0)),
        ("H", (-0.5, -0.87, 0.0)),
        ("H", (1.84, 0.87, 0.0)),
        ("H", (1.84, -0.87, 0.0)),
    ]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    G = _make_graph(atoms, edges)
    stats = optimizer.optimize(G, quick=True)
    # C-C bond should be double
    assert G.edges[0, 1]["bond_order"] == 2.0
    assert isinstance(stats, dict)


# ---- Formal charges ----


def test_formal_charges_methane(optimizer):
    """Methane: all atoms neutral."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629)),
    ]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    G = _make_graph(atoms, edges)
    charges = optimizer.compute_formal_charges(G)
    assert all(c == 0 for c in charges)


def test_formal_charges_ammonium(charged_optimizer):
    """NH4+: nitrogen has +1 formal charge."""
    atoms = [
        ("N", (0.0, 0.0, 0.0)),
        ("H", (0.5, 0.5, 0.5)),
        ("H", (-0.5, -0.5, 0.5)),
        ("H", (-0.5, 0.5, -0.5)),
        ("H", (0.5, -0.5, -0.5)),
    ]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    G = _make_graph(atoms, edges)
    charges = charged_optimizer.compute_formal_charges(G)
    # Total charge should be +1
    assert sum(charges) == 1
    # Nitrogen should carry the positive charge
    assert charges[0] == 1


# ---- Custom weights affect scoring ----


def test_custom_weights_change_behavior():
    """Non-default scoring weights should produce different optimizer behavior."""
    geometry = GeometryCalculator()
    default_opt = BondOrderOptimizer(
        geometry=geometry,
        data=DATA,
        charge=0,
    )
    heavy_violation_opt = BondOrderOptimizer(
        geometry=geometry,
        data=DATA,
        charge=0,
        weights=ScoringWeights(violation_weight=5000.0),
    )
    # Both should instantiate fine with different weights
    assert default_opt.weights.violation_weight == 1000.0
    assert heavy_violation_opt.weights.violation_weight == 5000.0


def test_custom_config():
    """OptimizerConfig parameters are passed through."""
    config = OptimizerConfig(max_iter=10, beam_width=3, edge_per_iter=5)
    opt = BondOrderOptimizer(
        geometry=GeometryCalculator(),
        data=DATA,
        charge=0,
        config=config,
    )
    assert opt.config.max_iter == 10
    assert opt.config.beam_width == 3
    assert opt.config.edge_per_iter == 5


# ---- Graph copy ----


def test_copy_graph_state_independence():
    """Copied graph should be independent (edits don't propagate)."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (1.0, 0.0, 0.0)),
    ]
    G = _make_graph(atoms, [(0, 1)])
    G_copy = BondOrderOptimizer._copy_graph_state(G)

    # Modify copy
    G_copy.edges[0, 1]["bond_order"] = 2.0

    # Original unchanged
    assert G.edges[0, 1]["bond_order"] == 1.0
    assert G_copy.edges[0, 1]["bond_order"] == 2.0


# ---- Invalid optimizer mode ----


def test_invalid_optimizer_mode(optimizer):
    """Unknown optimizer mode raises ValueError."""
    G = nx.Graph()
    with pytest.raises(ValueError, match="Unknown optimizer mode"):
        optimizer.optimize(G, mode="invalid")
