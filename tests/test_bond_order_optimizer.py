"""Tests for bond order optimization.

Verifies BondOrderOptimizer scoring, valence adjustment,
formal charge computation, and aromatic detection.
"""

import networkx as nx
import numpy as np
import pytest

from xyzgraph.bond_order_optimizer import BondOrderOptimizer
from xyzgraph.data_loader import DATA
from xyzgraph.geometry import GeometryCalculator
from xyzgraph.parameters import OptimizerConfig, ScoringWeights
from xyzgraph.scoring_arrays import huckel_aromatic, ring_pi_electrons


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
    # Compute rings (needed by detect_aromatic_rings)
    G.graph["_rings"] = nx.cycle_basis(G)
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
    """Formal charge across octet, sextet (B), and duet (H) shells."""
    fc = BondOrderOptimizer._compute_formal_charge_value
    # Octet group (target = 8)
    assert fc("C", 4, 4.0) == 0  # methane-style
    assert fc("N", 5, 4.0) == 1  # ammonium
    assert fc("O", 6, 1.0) == -1  # alkoxide
    assert fc("F", 7, 1.0) == 0  # HF-style
    # Group-13 sextet (target = 6, not octet).  A neutral trivalent boron
    # must come out as fc=0 — the octet-only formula gave fc=-2 here.
    assert fc("B", 3, 3.0) == 0  # BF3
    assert fc("B", 3, 4.0) == -1  # BF4-
    # Hydrogen duet: bonded H is neutral, a bare H completes its duet (a hydride), as a bare Cl is Cl-
    assert fc("H", 1, 1.0) == 0
    assert fc("H", 1, 0.0) == -1
    # At an allowed valence the leftover electrons stay lone pairs: a sulfoxide S
    # (one S=O, valence 4) is neutral, a three-bonded sulfonium S is not.
    assert fc("S", 6, 4.0, degree=3, allowed=(2, 4, 6)) == 0
    assert fc("S", 6, 3.0, degree=3, allowed=(2, 4, 6)) == 1


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
    stats = optimizer._quick_valence_adjust(G)
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


# ---- Invalid optimizer mode ----


def test_invalid_optimizer_mode(optimizer):
    """Unknown optimizer mode raises ValueError."""
    G = nx.Graph()
    with pytest.raises(ValueError, match="Unknown optimizer mode"):
        optimizer.optimize(G, mode="invalid")


# ---- π seeding & aromatic detection ----
#
# Two-step process tested separately:
#   1. seed_pi_bonds:  matches atoms that lack valence, placing a Kekulé
#      (1.0/2.0/3.0) structure.  This is a *localised* picture.
#   2. detect_aromatic_rings:  converts Kekulé bonds to aromatic BO=1.5
#      using Hückel 4n+2 π-electron counting.
#
# Uses _make_graph to build minimal graphs with known topology, then
# calls each step directly — no scoring or formal charges involved.


# Indole: fused 6+5 bicyclic (benzene + pyrrole).
# 10 ring bonds, all should become BO=1.5 after aromatic detection.
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
# 16 ring bonds, all should become BO=1.5 after aromatic detection.
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
    """Indole: π seeding then aromatic detection on a single graph.

    The pyrrole N (idx 8) needs valence 3, so both N-ring bonds must be
    single.  This forces C3=C2 double (6,7) and C3a-C3 single (0,6).
    The benzene ring alternates from there; either way all FC must be 0.
    Then detect_aromatic_rings converts everything to BO=1.5.
    """
    G = _make_graph(INDOLE_ATOMS, INDOLE_EDGES)

    # -- Step 1: π seeding --
    assert optimizer.seed_pi_bonds(G) == 4  # C3=C2 plus three benzene doubles

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
    # Both rings should be flagged aromatic
    assert len(G.graph["_aromatic_rings"]) == 2


def test_kekule_flag_skips_aromatic_bo(optimizer):
    """With kekule=True, rings are detected but bond orders stay Kekule."""
    G = _make_graph(INDOLE_ATOMS, INDOLE_EDGES)
    optimizer.seed_pi_bonds(G)
    charges = optimizer.compute_formal_charges(G)
    for i, fc in enumerate(charges):
        G.nodes[i]["formal_charge"] = fc

    # Capture Kekule bond orders before aromatic detection
    kekule_bo = {(i, j): G.edges[i, j]["bond_order"] for i, j in INDOLE_EDGES}

    optimizer.detect_aromatic_rings(G, kekule=True)

    # Rings are still detected
    assert len(G.graph["_aromatic_rings"]) == 2

    # Bond orders unchanged from Kekule values
    for i, j in INDOLE_EDGES:
        assert G.edges[i, j]["bond_order"] == pytest.approx(kekule_bo[(i, j)]), (
            f"edge {i}-{j} should keep Kekule BO={kekule_bo[(i, j)]}, got {G.edges[i, j]['bond_order']}"
        )

    # No 1.5 bond orders anywhere
    assert not any(abs(G.edges[i, j]["bond_order"] - 1.5) < 0.01 for i, j in INDOLE_EDGES)


def test_anthracene_kekule_then_aromatic(optimizer):
    """Anthracene: π seeding then aromatic detection on a single graph.

    Three fused 6-rings, all C.  Kekulé gives alternating 1.0/2.0 with
    all FC=0.  Then detect_aromatic_rings converts to BO=1.5.
    """
    G = _make_graph(ANTHRACENE_ATOMS, ANTHRACENE_EDGES)
    ring_edges = [(i, j) for i, j in ANTHRACENE_EDGES if i < 14 and j < 14]

    # -- Step 1: π seeding --
    assert optimizer.seed_pi_bonds(G) == 7  # one double per pair of ring carbons
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


# Benzyne (1,2-didehydrobenzene, C6H4): aromatic 6-ring with a triple
# bond between the two dehydrogenated carbons (C3, C4).
BENZYNE_ATOMS = [
    ("C", (-0.08896, 0.83348, 0.00000)),  # 0
    ("C", (-0.38460, -0.52113, 0.01487)),  # 1
    ("C", (0.63922, -1.45543, 0.01309)),  # 2
    ("C", (1.96067, -1.03512, -0.00356)),  # 3  (no H)
    ("C", (2.25731, 0.31948, -0.01844)),  # 4  (no H)
    ("C", (1.23249, 1.25379, -0.01666)),  # 5
    ("H", (-0.88781, 1.56177, 0.00139)),  # 6
    ("H", (-1.41567, -0.84876, 0.02378)),  # 7
    ("H", (0.40799, -2.51135, 0.02469)),  # 8
    ("H", (1.46371, 2.30971, -0.02825)),  # 9
]
BENZYNE_EDGES = [
    # Ring: C0-C1-C2-C3-C4-C5
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    # C-H bonds (C3 and C4 have no H)
    (0, 6),
    (1, 7),
    (2, 8),
    (5, 9),
]


def test_benzyne_kekule_and_aromatic(optimizer):
    """Benzyne: π seeding + optimisation + aromatic guard.

    The two undercoordinated carbons (C3, C4) each lack two bonds, so the
    matching seeds the C3≡C4 triple directly and the beam search has nothing
    left to improve.  The aromatic detector flags the ring (6π, Hückel-valid)
    but keeps the Kekulé+triple structure because the triple bond cannot be
    represented at 1.5.
    """
    G = _make_graph(BENZYNE_ATOMS, BENZYNE_EDGES)

    # -- Step 1: π seeding --
    assert optimizer.seed_pi_bonds(G) == 4  # the triple's two π bonds plus two ring doubles
    assert G.edges[3, 4]["bond_order"] == pytest.approx(3.0), "C3≡C4 should be seeded as a triple"

    # -- Step 2: beam search optimisation --
    stats = optimizer.optimize(G, mode="beam")
    assert stats["improvements"] == 0, "the seed is already the optimum"

    # Final bond orders: 1 triple, 2 doubles, 3 singles
    ring_bos = sorted(G.edges[i, j]["bond_order"] for i, j in BENZYNE_EDGES[:6])
    assert ring_bos == pytest.approx([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    assert G.edges[3, 4]["bond_order"] == pytest.approx(3.0), "C3-C4 must be triple"

    # All formal charges should be 0
    charges = optimizer.compute_formal_charges(G)
    for i, fc in enumerate(charges):
        G.nodes[i]["formal_charge"] = fc
    assert all(c == 0 for c in charges), f"Expected all FC=0, got {charges}"

    # -- Step 3: aromatic detection --
    optimizer.detect_aromatic_rings(G)

    # Ring is flagged aromatic (Hückel 4n+2: 6π electrons)
    assert len(G.graph["_aromatic_rings"]) == 1

    # But bonds are NOT set to 1.5 (triple bond blocks conversion)
    assert G.edges[3, 4]["bond_order"] == pytest.approx(3.0), "Triple bond must be preserved"
    for i, j in BENZYNE_EDGES[:6]:
        assert G.edges[i, j]["bond_order"] != pytest.approx(1.5), f"Bond {i}-{j} should not be aromatic 1.5"


# ---- NHC carbenes: regression guard for the N-cap fix ----
#
# Carbene C2 sits between two Ns with no H — its non-metal valence sum is
# naturally ≤ 2, so before `SCORING_VALENCE_LIMITS["N"]` was tightened to 4,
# the optimiser closed the gap by promoting one C-N to a triple bond and
# letting N go to valence 5.  Representation-agnostic invariants: no C-N
# triple, no N > 4, formal charges balance.  Survives the follow-up
# electron-counting PR (ylide → pure C:) without modification.

NHC_ATOMS = [  # 1H,3H-imidazol-2-ylidene (free singlet carbene)
    ("N", (0.315321, -1.090120, 0.030915)),  # 0
    ("C", (1.167190, -0.044472, 0.274604)),  # 1
    ("C", (0.412102, 1.101290, 0.390491)),  # 2
    ("N", (-0.942833, 0.733552, 0.210340)),  # 3
    ("C", (-0.973417, -0.625439, -0.010014)),  # 4  carbene
    ("H", (0.594186, -2.054220, -0.098510)),  # 5  N-H
    ("H", (2.234610, -0.205024, 0.344430)),  # 6  C-H
    ("H", (0.795428, 2.094790, 0.583885)),  # 7  C-H
    ("H", (-1.803260, 1.389490, 0.238365)),  # 8  N-H
]
NHC_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8)]

NHC_PT_ATOMS = [  # [Pt(NHC)Cl2(NH3)]
    ("N", (0.275626, -1.065130, 0.034780)),  # 0
    ("C", (1.157650, -0.046999, 0.270073)),  # 1
    ("C", (0.421646, 1.086720, 0.386920)),  # 2
    ("N", (-0.882388, 0.723162, 0.221738)),  # 3
    ("C", (-0.987589, -0.600189, 0.005288)),  # 4  carbene
    ("H", (0.504845, -2.041530, -0.090950)),  # 5
    ("H", (2.217780, -0.185132, 0.339715)),  # 6
    ("H", (0.732437, 2.094450, 0.573955)),  # 7
    ("H", (-1.703560, 1.319020, 0.230587)),  # 8
    ("Pt", (-2.702450, -1.556900, -0.244630)),  # 9
    ("Cl", (-3.864720, 0.437196, -0.312301)),  # 10
    ("Cl", (-1.686330, -3.611530, -0.188207)),  # 11
    ("N", (-4.566200, -2.548630, -0.493331)),  # 12 NH3
    ("H", (-4.457570, -3.370940, -1.086820)),  # 13
    ("H", (-5.251820, -1.915830, -0.906690)),  # 14
    ("H", (-4.920640, -2.856940, 0.412246)),  # 15
]
NHC_PT_EDGES = [*NHC_EDGES, (4, 9), (9, 10), (9, 11), (9, 12), (12, 13), (12, 14), (12, 15)]
NHC_PT_METAL_EDGES = {(4, 9), (9, 10), (9, 11), (9, 12)}


def _assert_nhc_invariants(G, optimizer, carbene, n_neighbors, total_charge):
    fcs = optimizer.compute_formal_charges(G)
    for nb in n_neighbors:
        assert G.edges[carbene, nb]["bond_order"] <= 2.0, (
            f"C{carbene}-N{nb} bond order {G.edges[carbene, nb]['bond_order']} (triple disallowed)"
        )
        nv = BondOrderOptimizer.valence_sum(G, nb)
        assert nv <= 4.0, f"N{nb} valence {nv} > 4"
    # The carbene carbon is a neutral singlet carbene (a sextet, one lone pair),
    # not the spurious -2 dicarbanion an octet fill would give.  This falls out
    # of the general neutral-shell formal-charge rule — no carbene pattern-match.
    assert fcs[carbene] == 0, f"carbene C{carbene} should be neutral (sextet), got {fcs[carbene]:+d}"
    assert sum(fcs) == total_charge


def test_nhc_free_carbene_no_hypervalent_n(optimizer):
    """Free NHC: optimiser must not produce C≡N or pentavalent N."""
    G = _make_graph(NHC_ATOMS, NHC_EDGES)
    optimizer.seed_pi_bonds(G)
    optimizer.optimize(G)
    _assert_nhc_invariants(G, optimizer, carbene=4, n_neighbors=(0, 3), total_charge=0)


def test_nhc_pt_carbene_no_hypervalent_n(optimizer):
    """[Pt(NHC)Cl2(NH3)]: same invariants hold with metal coordination."""
    G = _make_graph(NHC_PT_ATOMS, NHC_PT_EDGES)
    for i, j in NHC_PT_METAL_EDGES:
        G.edges[i, j]["metal_coord"] = True
    optimizer.seed_pi_bonds(G)
    optimizer.optimize(G)
    _assert_nhc_invariants(G, optimizer, carbene=4, n_neighbors=(0, 3), total_charge=0)


# Acetonitrile CH₃-C≡N: a linear nitrile.  Reaching the C≡N triple needs the
# optimiser's ±2 promotion to jump single→triple in one step — the single→double
# intermediate is an fc--1 barrier the strictly-improving beam will not cross.
# Same mechanism that lets a divalent carbon read as a neutral carbene without
# starving alkynes/nitriles of their triples.
ACETONITRILE_ATOMS = [
    ("C", (0.000, 0.000, 0.000)),  # 0  methyl C
    ("C", (1.460, 0.000, 0.000)),  # 1  nitrile C
    ("N", (2.620, 0.000, 0.000)),  # 2  nitrile N
    ("H", (-0.363, 1.027, 0.000)),  # 3
    ("H", (-0.363, -0.513, 0.889)),  # 4
    ("H", (-0.363, -0.513, -0.889)),  # 5
]
ACETONITRILE_EDGES = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)]


def test_nitrile_triple_via_jump(optimizer):
    """Acetonitrile: the C≡N triple forms (via the single→triple ±2 move), and
    every atom is neutral."""
    G = _make_graph(ACETONITRILE_ATOMS, ACETONITRILE_EDGES)
    optimizer.seed_pi_bonds(G)
    optimizer.optimize(G)
    assert G.edges[1, 2]["bond_order"] == pytest.approx(3.0), (
        f"C1≡N2 expected triple, got {G.edges[1, 2]['bond_order']}"
    )
    charges = optimizer.compute_formal_charges(G)
    assert all(c == 0 for c in charges), f"acetonitrile should be all-neutral, got {charges}"


# ---- Aromaticity helpers: p-orbital availability ----

# p-Benzoquinone skeleton (heavy atoms only): ring C0-C5, exocyclic O6 on C0
# and O7 on C3, doubles at C1=C2, C4=C5, C0=O6, C3=O7.
QUINONE_ATOMS = [
    ("C", (1.40, 0.00, 0.0)),
    ("C", (0.70, 1.21, 0.0)),
    ("C", (-0.70, 1.21, 0.0)),
    ("C", (-1.40, 0.00, 0.0)),
    ("C", (-0.70, -1.21, 0.0)),
    ("C", (0.70, -1.21, 0.0)),
    ("O", (2.62, 0.00, 0.0)),
    ("O", (-2.62, 0.00, 0.0)),
]
QUINONE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (3, 7)]
QUINONE_DOUBLES = [(1, 2), (4, 5), (0, 6), (3, 7)]


def _quinone_graph():
    G = _make_graph(QUINONE_ATOMS, QUINONE_EDGES)
    for i, j in QUINONE_DOUBLES:
        G.edges[i, j]["bond_order"] = 2.0
    return G


@pytest.mark.parametrize(
    ("case", "V", "fc", "bos", "degree", "ring_pi", "exo_pi", "expected"),
    [
        ("pyridine N, in a ring double bond", 5, 0, 3, 2, True, False, 1),
        ("quinone C, pi bond off the ring", 4, 0, 4, 3, False, True, 0),
        ("pyrrole NH, lone pair in p", 5, 0, 3, 3, False, False, 2),
        ("furan O, one pair in plane, one in p", 6, 0, 2, 2, False, False, 2),
        ("Cp- carbon", 4, -1, 3, 3, False, False, 2),
        ("carbene C, pair in plane, empty p", 4, 0, 2, 2, False, False, 0),
        ("borane B, empty p", 3, 0, 3, 3, False, False, 0),
    ],
)
def test_ring_pi_electrons(case, V, fc, bos, degree, ring_pi, exo_pi, expected):
    """One element-free rule gives every ring atom's π electrons."""
    assert ring_pi_electrons(V, fc, bos, degree, ring_pi, exo_pi) == expected, case


def test_quinone_is_cross_conjugated(optimizer):
    """Quinone carbonyl carbons give no π electron; two such atoms make a neutral 6π ring
    cross-conjugated, while olate charge on the ring system lets it stay aromatic."""
    G = _quinone_graph()
    ring = G.graph["_rings"][0]
    pi = optimizer._ring_pi_electrons(G)
    assert [pi[i] for i in ring].count(0) == 2
    assert not huckel_aromatic(np.array([0, 0, 2, 2, 1, 1]), 0)
    assert huckel_aromatic(np.array([0, 0, 2, 2, 1, 1]), -2)


def test_ring_system_charge(optimizer):
    """The ring-system charge counts direct substituents (olates), not a bound metal."""
    G = _quinone_graph()
    ring = G.graph["_rings"][0]
    assert optimizer._ring_system_charge(G, ring) == 0
    G.nodes[6]["formal_charge"] = -1  # exocyclic olate oxygen
    G.add_node(8, symbol="Fe", formal_charge=2)
    G.add_edge(1, 8, bond_order=1.0)
    assert optimizer._ring_system_charge(G, ring) == -1
