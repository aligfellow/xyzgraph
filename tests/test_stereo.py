"""Tests for stereochemistry assignment."""

import math

import networkx as nx

from xyzgraph.stereo import (
    _bond_multiplicity,
    annotate_stereo,
    assign_axial,
    assign_ez,
    assign_helical,
    assign_planar,
    assign_rs,
)


def _add_node(G: nx.Graph, idx: int, symbol: str, pos: tuple[float, float, float]) -> None:
    G.add_node(idx, symbol=symbol, position=pos)


def _make_allene_graph() -> nx.Graph:
    G = nx.Graph()
    # C=C=C with distinct substituents on each end
    _add_node(G, 0, "C", (-1.0, 0.0, 0.0))
    _add_node(G, 1, "C", (0.0, 0.0, 0.0))
    _add_node(G, 2, "C", (1.0, 0.0, 0.0))
    _add_node(G, 3, "I", (-1.0, 1.0, 0.0))
    _add_node(G, 4, "F", (-1.0, -1.0, 0.0))
    _add_node(G, 5, "Br", (1.0, 0.0, 1.0))
    _add_node(G, 6, "Cl", (1.0, -1.0, 0.0))

    G.add_edge(0, 1, bond_order=2.0)
    G.add_edge(1, 2, bond_order=2.0)
    G.add_edge(0, 3, bond_order=1.0)
    G.add_edge(0, 4, bond_order=1.0)
    G.add_edge(2, 5, bond_order=1.0)
    G.add_edge(2, 6, bond_order=1.0)
    return G


def _make_planar_metallocene_graph() -> nx.Graph:
    G = nx.Graph()
    ring = []
    for i in range(5):
        angle = 2 * math.pi * i / 5
        pos = (math.cos(angle), math.sin(angle), 0.0)
        _add_node(G, i, "C", pos)
        ring.append(i)

    metal = 5
    _add_node(G, metal, "Fe", (0.0, 0.0, 1.0))

    # Substituents on two ring atoms to define planar chirality
    _add_node(G, 6, "Cl", (1.6, 0.0, 0.0))
    _add_node(G, 7, "F", (0.0, 1.6, 0.0))
    G.add_edge(0, 6, bond_order=1.0)
    G.add_edge(1, 7, bond_order=1.0)

    # Ring edges
    for i in range(5):
        G.add_edge(ring[i], ring[(i + 1) % 5], bond_order=1.5)

    # Metal coordination edges
    for i in ring:
        G.add_edge(metal, i, bond_order=1.0)

    G.graph["rings"] = [ring]
    return G


def _make_helical_graph() -> nx.Graph:
    """5 fused 3-atom rings in a helical arrangement (sharing 2 atoms each)."""
    G = nx.Graph()
    rings: list[list[int]] = []
    # First ring: 3 unique atoms
    idx = 0
    t, z = 0.0, 0.0
    cx, cy = math.cos(t), math.sin(t)
    first_ring = []
    for offset in [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (-0.1, 0.0, 0.0)]:
        pos = (cx + offset[0], cy + offset[1], z + offset[2])
        _add_node(G, idx, "C", pos)
        first_ring.append(idx)
        idx += 1
    rings.append(first_ring)

    # Each subsequent ring shares last 2 atoms of previous ring + 1 new atom
    for k in range(1, 5):
        t = 0.5 * k
        z = 0.7 * k
        cx, cy = math.cos(t), math.sin(t)
        prev = rings[-1]
        shared = [prev[-2], prev[-1]]  # share 2 atoms
        pos = (cx - 0.1, cy, z)
        _add_node(G, idx, "C", pos)
        new_ring = [*shared, idx]
        idx += 1
        rings.append(new_ring)

    G.graph["aromatic_rings"] = rings
    return G


# ---------------------------------------------------------------------------
# R/S tests
# ---------------------------------------------------------------------------


def test_assign_rs_tetrahedral() -> None:
    """Tetrahedral C with F, Cl, Br, I — must get a specific label."""
    G = nx.Graph()
    # S-configuration geometry (verified by hand)
    _add_node(G, 0, "C", (0.0, 0.0, 0.0))
    _add_node(G, 1, "F", (0.0, 0.0, -1.0))
    _add_node(G, 2, "I", (0.0, 0.943, 0.333))
    _add_node(G, 3, "Cl", (0.816, -0.471, 0.333))
    _add_node(G, 4, "Br", (-0.816, -0.471, 0.333))

    for i in range(1, 5):
        G.add_edge(0, i, bond_order=1.0)

    rs = assign_rs(G)
    assert rs[0] == "S"


def test_rs_noncontiguous_nodes() -> None:
    """Nodes with gaps must work — regression for pos indexing bug."""
    G = nx.Graph()
    _add_node(G, 10, "C", (0.0, 0.0, 0.0))
    _add_node(G, 20, "F", (0.0, 0.0, -1.0))
    _add_node(G, 30, "I", (0.0, 0.943, 0.333))
    _add_node(G, 40, "Cl", (0.816, -0.471, 0.333))
    _add_node(G, 50, "Br", (-0.816, -0.471, 0.333))

    for n in [20, 30, 40, 50]:
        G.add_edge(10, n, bond_order=1.0)

    rs = assign_rs(G)
    assert rs[10] == "S"


def test_rs_rejects_planar_center() -> None:
    """4-coordinate but planar (sp2) center must NOT get R/S."""
    G = nx.Graph()
    _add_node(G, 0, "C", (0.0, 0.0, 0.0))
    _add_node(G, 1, "F", (1.0, 0.0, 0.0))
    _add_node(G, 2, "Cl", (0.0, 1.0, 0.0))
    _add_node(G, 3, "Br", (-1.0, 0.0, 0.0))
    _add_node(G, 4, "I", (0.0, -1.0, 0.0))

    for n in [1, 2, 3, 4]:
        G.add_edge(0, n, bond_order=1.0)

    assert len(assign_rs(G)) == 0


def test_rs_rejects_near_linear() -> None:
    """4-coordinate with nearly linear pair must NOT get R/S."""
    G = nx.Graph()
    _add_node(G, 0, "C", (0.0, 0.0, 0.0))
    _add_node(G, 1, "F", (0.0, 0.0, 1.0))
    _add_node(G, 2, "Cl", (0.0, 0.087, -0.996))  # ~175° from F
    _add_node(G, 3, "Br", (1.0, 0.0, 0.0))
    _add_node(G, 4, "I", (0.0, 1.0, 0.0))

    for n in [1, 2, 3, 4]:
        G.add_edge(0, n, bond_order=1.0)

    assert len(assign_rs(G)) == 0


def test_rs_accepts_ts_geometry() -> None:
    """4-coordinate with 3 in-plane + 1 out-of-plane (Buergi-Dunitz TS) gets R/S."""
    G = nx.Graph()
    _add_node(G, 0, "C", (0.0, 0.0, 0.0))
    # 3 substituents in the xy-plane at ~120 degrees
    _add_node(G, 1, "F", (1.4, 0.0, 0.0))
    _add_node(G, 2, "Cl", (-0.7, 1.212, 0.0))
    _add_node(G, 3, "Br", (-0.7, -1.212, 0.0))
    # Nucleophile approaching from above at ~17 degrees (Buergi-Dunitz)
    _add_node(G, 4, "I", (0.0, 0.0, 2.5))

    for n in [1, 2, 3, 4]:
        G.add_edge(0, n, bond_order=1.0)

    rs = assign_rs(G)
    assert len(rs) == 1
    assert rs[0] in {"R", "S"}


def test_sulfone_not_stereocenter() -> None:
    """S(=O)2 with different S-O bond orders must NOT be a stereocenter."""
    G = nx.Graph()
    _add_node(G, 0, "S", (0.0, 0.0, 0.0))
    _add_node(G, 1, "C", (1.5, 0.0, 0.0))
    _add_node(G, 2, "N", (-1.5, 0.0, 0.0))
    _add_node(G, 3, "O", (0.0, 1.2, 0.8))
    _add_node(G, 4, "O", (0.0, -1.2, 0.8))

    G.add_edge(0, 1, bond_order=1.0)
    G.add_edge(0, 2, bond_order=1.0)
    G.add_edge(0, 3, bond_order=1.4)
    G.add_edge(0, 4, bond_order=1.6)

    assert len(assign_rs(G)) == 0


def test_no_backward_compat_attrs() -> None:
    """annotate_stereo must NOT set generic 'stereo' attribute."""
    G = nx.Graph()
    _add_node(G, 0, "C", (0.0, 0.0, 0.0))
    _add_node(G, 1, "F", (0.0, 0.0, -1.0))
    _add_node(G, 2, "I", (0.0, 0.943, 0.333))
    _add_node(G, 3, "Cl", (0.816, -0.471, 0.333))
    _add_node(G, 4, "Br", (-0.816, -0.471, 0.333))
    for n in [1, 2, 3, 4]:
        G.add_edge(0, n, bond_order=1.0)

    annotate_stereo(G)
    assert "stereo_rs" in G.nodes[0]
    assert "stereo" not in G.nodes[0]


# ---------------------------------------------------------------------------
# E/Z tests
# ---------------------------------------------------------------------------


def test_assign_ez_simple() -> None:
    """Cl same side = Z (verified)."""
    G = nx.Graph()
    _add_node(G, 0, "C", (-0.67, 0.0, 0.0))
    _add_node(G, 1, "C", (0.67, 0.0, 0.0))
    _add_node(G, 2, "Cl", (-1.2, 1.0, 0.0))
    _add_node(G, 3, "F", (-1.2, -1.0, 0.0))
    _add_node(G, 4, "Cl", (1.2, 1.0, 0.0))
    _add_node(G, 5, "F", (1.2, -1.0, 0.0))

    G.add_edge(0, 1, bond_order=2.0)
    G.add_edge(0, 2, bond_order=1.0)
    G.add_edge(0, 3, bond_order=1.0)
    G.add_edge(1, 4, bond_order=1.0)
    G.add_edge(1, 5, bond_order=1.0)

    assert assign_ez(G)[(0, 1)] == "Z"


def test_ez_noncontiguous_nodes() -> None:
    """E/Z with gapped node IDs — regression for pos indexing bug."""
    G = nx.Graph()
    _add_node(G, 100, "C", (-0.67, 0.0, 0.0))
    _add_node(G, 200, "C", (0.67, 0.0, 0.0))
    _add_node(G, 300, "Cl", (-1.2, 1.0, 0.0))
    _add_node(G, 400, "F", (-1.2, -1.0, 0.0))
    _add_node(G, 500, "Cl", (1.2, 1.0, 0.0))
    _add_node(G, 600, "F", (1.2, -1.0, 0.0))

    G.add_edge(100, 200, bond_order=2.0)
    G.add_edge(100, 300, bond_order=1.0)
    G.add_edge(100, 400, bond_order=1.0)
    G.add_edge(200, 500, bond_order=1.0)
    G.add_edge(200, 600, bond_order=1.0)

    assert assign_ez(G)[(100, 200)] == "Z"


def test_ez_rejects_small_ring() -> None:
    """Double bond in 6-membered ring must NOT get E/Z (geometry-locked)."""
    G = nx.Graph()
    angles = [2 * math.pi * i / 6 for i in range(6)]
    for i, a in enumerate(angles):
        _add_node(G, i, "C", (math.cos(a), math.sin(a), 0.0))
    G.add_edge(0, 1, bond_order=2.0)
    for i in range(1, 6):
        G.add_edge(i, (i + 1) % 6, bond_order=1.0)
    _add_node(G, 6, "F", (1.5, 0.5, 0.0))
    _add_node(G, 7, "Cl", (1.5, -0.5, 0.0))
    _add_node(G, 8, "F", (0.0, 1.5, 0.0))
    _add_node(G, 9, "Br", (1.0, 1.5, 0.0))
    G.add_edge(0, 6, bond_order=1.0)
    G.add_edge(0, 7, bond_order=1.0)
    G.add_edge(1, 8, bond_order=1.0)
    G.add_edge(1, 9, bond_order=1.0)
    G.graph["rings"] = [[0, 1, 2, 3, 4, 5]]

    assert len(assign_ez(G)) == 0


def test_ez_allows_large_ring() -> None:
    """Double bond in 8-membered ring CAN have E/Z."""
    G = nx.Graph()
    angles = [2 * math.pi * i / 8 for i in range(8)]
    for i, a in enumerate(angles):
        _add_node(G, i, "C", (2.0 * math.cos(a), 2.0 * math.sin(a), 0.0))
    G.add_edge(0, 1, bond_order=2.0)
    for i in range(1, 8):
        G.add_edge(i, (i + 1) % 8, bond_order=1.0)
    _add_node(G, 8, "F", (3.0, 0.5, 0.0))
    _add_node(G, 9, "Cl", (3.0, -0.5, 0.0))
    _add_node(G, 10, "F", (1.0, 2.5, 0.0))
    _add_node(G, 11, "Br", (2.0, 2.5, 0.0))
    G.add_edge(0, 8, bond_order=1.0)
    G.add_edge(0, 9, bond_order=1.0)
    G.add_edge(1, 10, bond_order=1.0)
    G.add_edge(1, 11, bond_order=1.0)
    G.graph["rings"] = [[0, 1, 2, 3, 4, 5, 6, 7]]

    assert len(assign_ez(G)) == 1


# ---------------------------------------------------------------------------
# CIP / bond multiplicity tests
# ---------------------------------------------------------------------------


def test_bond_multiplicity_values() -> None:
    """Exact expected values for all bond order ranges."""
    assert _bond_multiplicity(None) == 1
    assert _bond_multiplicity(1.0) == 1
    assert _bond_multiplicity(1.4) == 1  # rounds to 1
    assert _bond_multiplicity(1.5) == 2  # aromatic special case
    assert _bond_multiplicity(1.501) == 2  # rounds to 2
    assert _bond_multiplicity(1.6) == 2  # rounds to 2
    assert _bond_multiplicity(2.0) == 2
    assert _bond_multiplicity(2.4) == 2  # rounds to 2, not 3
    assert _bond_multiplicity(2.6) == 3  # rounds to 3
    assert _bond_multiplicity(3.0) == 3


# ---------------------------------------------------------------------------
# Axial chirality tests
# ---------------------------------------------------------------------------


def test_assign_axial_allene_and_annotate_axes() -> None:
    """Allene with distinct ends gets axial label via non-edge axis."""
    G = _make_allene_graph()
    axial, axes = assign_axial(G)
    assert not axial  # no direct edge between allene termini
    assert len(axes) == 1
    assert axes[0][2] in {"Rₐ", "Sₐ"}

    annotate_stereo(G)
    assert "stereo_axes" in G.graph
    assert any(axis["kind"] == "axial" for axis in G.graph["stereo_axes"])


def _make_axial_metallocene_graph() -> nx.Graph:
    """Sandwich metallocene with different subs on each ring — axial chirality."""
    G = nx.Graph()
    # Ring A at z=0
    for i in range(5):
        angle = 2 * math.pi * i / 5
        _add_node(G, i, "C", (math.cos(angle), math.sin(angle), 0.0))
    # Ring B at z=3.3
    for i in range(5):
        angle = 2 * math.pi * i / 5
        _add_node(G, i + 5, "C", (math.cos(angle), math.sin(angle), 3.3))
    # Fe at midpoint
    _add_node(G, 10, "Fe", (0.0, 0.0, 1.65))

    for i in range(5):
        G.add_edge(i, (i + 1) % 5, bond_order=1.5)
        G.add_edge(10, i, bond_order=1.0)
        G.add_edge(i + 5, ((i + 1) % 5) + 5, bond_order=1.5)
        G.add_edge(10, i + 5, bond_order=1.0)

    # CH₃ on ring A, atom 0 (angle 0)
    _add_node(G, 11, "C", (2.0, 0.0, 0.0))
    G.add_edge(0, 11, bond_order=1.0)

    # Cl on ring B, atom 7 (angle 2·2π/5 ≈ 144° — NOT eclipsed with CH₃)
    angle_7 = 2 * math.pi * 2 / 5
    _add_node(G, 12, "Cl", (math.cos(angle_7) * 2.0, math.sin(angle_7) * 2.0, 3.3))
    G.add_edge(7, 12, bond_order=1.0)

    G.graph["rings"] = [list(range(5)), list(range(5, 10))]
    return G


def test_assign_axial_metallocene() -> None:
    """Sandwich metallocene with different substituents on each ring gets axial label."""
    G = _make_axial_metallocene_graph()
    axial, axes = assign_axial(G)
    total = len(axial) + len(axes)
    assert total == 1
    label = next(iter(axial.values())) if axial else axes[0][2]
    assert label in {"Rₐ", "Sₐ"}


def test_axial_metallocene_identical_subs_rejected() -> None:
    """Sandwich metallocene with identical substituents on each ring — achiral."""
    G = nx.Graph()
    for i in range(5):
        angle = 2 * math.pi * i / 5
        _add_node(G, i, "C", (math.cos(angle), math.sin(angle), 0.0))
    for i in range(5):
        angle = 2 * math.pi * i / 5
        _add_node(G, i + 5, "C", (math.cos(angle), math.sin(angle), 3.3))
    _add_node(G, 10, "Fe", (0.0, 0.0, 1.65))
    for i in range(5):
        G.add_edge(i, (i + 1) % 5, bond_order=1.5)
        G.add_edge(10, i, bond_order=1.0)
        G.add_edge(i + 5, ((i + 1) % 5) + 5, bond_order=1.5)
        G.add_edge(10, i + 5, bond_order=1.0)

    # Same substituent (CH₃) on both rings
    _add_node(G, 11, "C", (2.0, 0.0, 0.0))
    G.add_edge(0, 11, bond_order=1.0)
    angle_7 = 2 * math.pi * 2 / 5
    _add_node(G, 12, "C", (math.cos(angle_7) * 2.0, math.sin(angle_7) * 2.0, 3.3))
    G.add_edge(7, 12, bond_order=1.0)

    G.graph["rings"] = [list(range(5)), list(range(5, 10))]
    axial, axes = assign_axial(G)
    # Metallocene-specific labels only — ring bridge / allene should also be empty
    assert len(axial) + len(axes) == 0


def test_symmetric_biaryl_no_label() -> None:
    """Two identical rings joined by single bond — no axial chirality.

    Rings have no ortho substituents and identical substitution patterns,
    so both the ortho steric gate and the symmetric-ends check reject it.
    """
    G = nx.Graph()
    for i in range(6):
        angle = 2 * math.pi * i / 6
        _add_node(G, i, "C", (math.cos(angle), math.sin(angle), 0.0))
    for i in range(6):
        angle = 2 * math.pi * i / 6
        _add_node(G, i + 6, "C", (math.cos(angle) + 3.0, math.sin(angle), 0.0))
    for i in range(6):
        G.add_edge(i, (i + 1) % 6, bond_order=1.5)
        G.add_edge(i + 6, ((i + 1) % 6) + 6, bond_order=1.5)
    G.add_edge(0, 6, bond_order=1.0)
    G.graph["rings"] = [list(range(6)), list(range(6, 12))]
    G.graph["aromatic_rings"] = [list(range(6)), list(range(6, 12))]

    assert len(assign_axial(G)[0]) == 0


# ---------------------------------------------------------------------------
# Planar chirality tests
# ---------------------------------------------------------------------------


def _make_paracyclophane_graph() -> nx.Graph:
    """[2.2]paracyclophane with Cl at position 4 on ring A.

    Ring A (z=0): 6 C atoms, Cl on atom 3.
    Ring B (z=2.8): 6 C atoms, no non-H substituents.
    Two CH₂-CH₂ bridges connecting atom 0↔6 and atom 3↔9 (para positions),
    with bridge carbons at z ≈ 1.4 (out of both ring planes by ~1.4 Å).
    """
    G = nx.Graph()
    # Ring A at z=0
    for i in range(6):
        angle = 2 * math.pi * i / 6
        _add_node(G, i, "C", (1.4 * math.cos(angle), 1.4 * math.sin(angle), 0.0))
    for i in range(6):
        G.add_edge(i, (i + 1) % 6, bond_order=1.5)

    # Ring B at z=2.8
    for i in range(6):
        angle = 2 * math.pi * i / 6
        _add_node(G, i + 6, "C", (1.4 * math.cos(angle), 1.4 * math.sin(angle), 2.8))
    for i in range(6):
        G.add_edge(i + 6, ((i + 1) % 6) + 6, bond_order=1.5)

    # Bridge 1: atom 0 (ring A) — CH₂(12) — CH₂(13) — atom 6 (ring B)
    _add_node(G, 12, "C", (1.4, 0.0, 0.9))
    _add_node(G, 13, "C", (1.4, 0.0, 1.9))
    G.add_edge(0, 12, bond_order=1.0)
    G.add_edge(12, 13, bond_order=1.0)
    G.add_edge(13, 6, bond_order=1.0)

    # Bridge 2: atom 3 (ring A) — CH₂(14) — CH₂(15) — atom 9 (ring B)
    _add_node(G, 14, "C", (-1.4, 0.0, 0.9))
    _add_node(G, 15, "C", (-1.4, 0.0, 1.9))
    G.add_edge(3, 14, bond_order=1.0)
    G.add_edge(14, 15, bond_order=1.0)
    G.add_edge(15, 9, bond_order=1.0)

    # Cl on ring A, atom 1 (ortho to bridge)
    angle_1 = 2 * math.pi * 1 / 6
    _add_node(G, 16, "Cl", (2.2 * math.cos(angle_1), 2.2 * math.sin(angle_1), -0.3))
    G.add_edge(1, 16, bond_order=1.0)

    # F on ring A, atom 5 (other ortho to bridge, different sub for chirality)
    angle_5 = 2 * math.pi * 5 / 6
    _add_node(G, 17, "F", (2.2 * math.cos(angle_5), 2.2 * math.sin(angle_5), -0.3))
    G.add_edge(5, 17, bond_order=1.0)

    G.graph["rings"] = [list(range(6)), list(range(6, 12))]
    return G


def test_assign_planar_paracyclophane() -> None:
    """Substituted [2.2]paracyclophane gets planar chirality label."""
    G = _make_paracyclophane_graph()
    planar, axes = assign_planar(G)
    total = len(planar) + len(axes)
    assert total == 1
    label = next(iter(planar.values())) if planar else axes[0][2]
    assert label in {"Rₚ", "Sₚ"}


def test_assign_planar_metallocene() -> None:
    """Substituted Cp ring with Fe gets planar chirality label."""
    G = _make_planar_metallocene_graph()
    planar, axes = assign_planar(G)
    assert axes == []
    assert len(planar) == 1
    label = next(iter(planar.values()))
    assert label in {"Rₚ", "Sₚ"}


# ---------------------------------------------------------------------------
# Helical chirality tests
# ---------------------------------------------------------------------------


def test_assign_helical_synthetic() -> None:
    """Fused helical rings get P or M label."""
    G = _make_helical_graph()
    labels = assign_helical(G)
    assert len(labels) == 1
    assert labels[0][2] in {"P", "M"}


def test_helical_disconnected_rejected() -> None:
    """5 non-fused aromatic rings must NOT get P/M."""
    G = nx.Graph()
    rings = []
    idx = 0
    for k in range(5):
        ring = []
        for offset in [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (-0.1, 0.0, 0.0)]:
            pos = (10.0 * k + offset[0], offset[1], 0.7 * k + offset[2])
            _add_node(G, idx, "C", pos)
            ring.append(idx)
            idx += 1
        rings.append(ring)
    G.graph["aromatic_rings"] = rings

    assert len(assign_helical(G)) == 0
