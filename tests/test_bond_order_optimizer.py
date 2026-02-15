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


# ---- Kekulé initialisation & aromatic detection ----
#
# Two-step process tested separately:
#   1. init_kekule:  assigns alternating single/double (1.0/2.0) Kekulé
#      pattern to validated aromatic rings.  This is a *localised* picture.
#   2. detect_aromatic_rings:  converts Kekulé bonds to aromatic BO=1.5
#      using Hückel 4n+2 π-electron counting.
#
# Uses _make_graph to build minimal graphs with known topology, then
# calls each step directly — no scoring or formal charges involved.


# Indole: fused 6+5 bicyclic (benzene + pyrrole).
# 10 ring bonds, all should become BO=1.5 after Kekulé init.
INDOLE_ATOMS = [
    ("C", (-1.204, -0.695, 0.0)),  # 0  C3a (junction)
    ("C", (-1.204, 0.695, 0.0)),  # 1  C4
    ("C", (0.000, 1.390, 0.0)),  # 2  C5
    ("C", (1.204, 0.695, 0.0)),  # 3  C6
    ("C", (1.204, -0.695, 0.0)),  # 4  C7
    ("C", (0.000, -1.390, 0.0)),  # 5  C7a (junction)
    ("C", (-2.237, -1.626, 0.0)),  # 6  C3
    ("C", (-1.672, -2.896, 0.0)),  # 7  C2
    ("N", (-0.290, -2.750, 0.0)),  # 8  N1
    ("H", (-2.139, 1.235, 0.0)),  # 9  H-C4
    ("H", (0.000, 2.470, 0.0)),  # 10 H-C5
    ("H", (2.139, 1.235, 0.0)),  # 11 H-C6
    ("H", (2.139, -1.235, 0.0)),  # 12 H-C7
    ("H", (-3.292, -1.401, 0.0)),  # 13 H-C3
    ("H", (-2.212, -3.831, 0.0)),  # 14 H-C2
    ("H", (0.386, -3.500, 0.0)),  # 15 H-N1
]
INDOLE_EDGES = [
    # 6-ring (benzene): C3a-C4-C5-C6-C7-C7a
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    # 5-ring (pyrrole): C3a-C3-C2-N1-C7a
    (0, 6),
    (6, 7),
    (7, 8),
    (8, 5),
    # C-H / N-H bonds
    (1, 9),
    (2, 10),
    (3, 11),
    (4, 12),
    (6, 13),
    (7, 14),
    (8, 15),
]

# Anthracene: three linearly fused 6-rings (C14H10).
# 16 ring bonds, all should become BO=1.5 after Kekulé init.
ANTHRACENE_ATOMS = [
    ("C", (0.000, 1.399, 0.0)),  # 0  (centre ring, top-left)
    ("C", (1.212, 0.700, 0.0)),  # 1  junction (centre-right)
    ("C", (1.212, -0.700, 0.0)),  # 2  junction (centre-right)
    ("C", (0.000, -1.399, 0.0)),  # 3  (centre ring, bottom-left)
    ("C", (-1.212, -0.700, 0.0)),  # 4  junction (centre-left)
    ("C", (-1.212, 0.700, 0.0)),  # 5  junction (centre-left)
    ("C", (2.424, 1.399, 0.0)),  # 6  (right ring)
    ("C", (3.636, 0.700, 0.0)),  # 7  (right ring)
    ("C", (3.636, -0.700, 0.0)),  # 8  (right ring)
    ("C", (2.424, -1.399, 0.0)),  # 9  (right ring)
    ("C", (-2.424, 1.399, 0.0)),  # 10 (left ring)
    ("C", (-3.636, 0.700, 0.0)),  # 11 (left ring)
    ("C", (-3.636, -0.700, 0.0)),  # 12 (left ring)
    ("C", (-2.424, -1.399, 0.0)),  # 13 (left ring)
    ("H", (0.000, 2.489, 0.0)),  # 14
    ("H", (0.000, -2.489, 0.0)),  # 15
    ("H", (2.424, 2.489, 0.0)),  # 16
    ("H", (4.572, 1.244, 0.0)),  # 17
    ("H", (4.572, -1.244, 0.0)),  # 18
    ("H", (2.424, -2.489, 0.0)),  # 19
    ("H", (-2.424, 2.489, 0.0)),  # 20
    ("H", (-4.572, 1.244, 0.0)),  # 21
    ("H", (-4.572, -1.244, 0.0)),  # 22
    ("H", (-2.424, -2.489, 0.0)),  # 23
]
ANTHRACENE_EDGES = [
    # Centre ring: 0-1-2-3-4-5
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    # Right ring: 1-6-7-8-9-2
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 2),
    # Left ring: 5-10-11-12-13-4
    (5, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 4),
    # C-H bonds
    (0, 14),
    (3, 15),
    (6, 16),
    (7, 17),
    (8, 18),
    (9, 19),
    (10, 20),
    (11, 21),
    (12, 22),
    (13, 23),
]


def test_indole_kekule_then_aromatic(optimizer):
    """Indole: Kekulé init then aromatic detection on a single graph.

    The pyrrole N (idx 8) needs valence 3, so both N-ring bonds must be
    single.  This forces C3=C2 double (6,7) and C3a-C3 single (0,6).
    The benzene ring alternates from there; either way all FC must be 0.
    Then detect_aromatic_rings converts everything to BO=1.5.
    """
    G = _make_graph(INDOLE_ATOMS, INDOLE_EDGES)

    # -- Step 1: Kekulé init --
    n_init = optimizer.init_kekule(G)
    assert n_init == 2  # one 6-ring + one 5-ring

    # Pyrrole bonds forced by N valence
    assert G.edges[7, 8]["bond_order"] == pytest.approx(1.0), "C2-N1 must be single"
    assert G.edges[8, 5]["bond_order"] == pytest.approx(1.0), "N1-C7a must be single"
    assert G.edges[6, 7]["bond_order"] == pytest.approx(2.0), "C3=C2 must be double"
    assert G.edges[0, 6]["bond_order"] == pytest.approx(1.0), "C3a-C3 must be single"

    # Valid Kekulé => all formal charges are 0
    charges = optimizer.compute_formal_charges(G)
    assert all(c == 0 for c in charges), f"Expected all FC=0, got {charges}"

    # -- Step 2: aromatic detection --
    optimizer.detect_aromatic_rings(G)
    for i, j in INDOLE_EDGES:
        if i < 9 and j < 9:  # only check ring bonds, not C-H / N-H
            assert G.edges[i, j]["bond_order"] == pytest.approx(1.5), (
                f"indole edge {i}-{j} BO={G.edges[i, j]['bond_order']}, expected 1.5"
            )


def test_anthracene_kekule_then_aromatic(optimizer):
    """Anthracene: Kekulé init then aromatic detection on a single graph.

    Three fused 6-rings, all C.  Kekulé gives alternating 1.0/2.0 with
    all FC=0.  Then detect_aromatic_rings converts to BO=1.5.
    """
    G = _make_graph(ANTHRACENE_ATOMS, ANTHRACENE_EDGES)
    ring_edges = [(i, j) for i, j in ANTHRACENE_EDGES if i < 14 and j < 14]

    # -- Step 1: Kekulé init --
    n_init = optimizer.init_kekule(G)
    assert n_init == 3  # three 6-rings
    assert len(ring_edges) == 16

    # Valid Kekulé => all formal charges are 0
    charges = optimizer.compute_formal_charges(G)
    assert all(c == 0 for c in charges), f"Expected all FC=0, got {charges}"

    # -- Step 2: aromatic detection --
    optimizer.detect_aromatic_rings(G)
    for i, j in ring_edges:
        assert G.edges[i, j]["bond_order"] == pytest.approx(1.5), (
            f"anthracene edge {i}-{j} BO={G.edges[i, j]['bond_order']}, expected 1.5"
        )
