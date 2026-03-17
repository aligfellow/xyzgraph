"""Minimal tests for stereochemistry assignment."""

import math

import networkx as nx

from xyzgraph.stereo import (
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
    G = nx.Graph()
    rings: list[list[int]] = []
    idx = 0
    for k in range(5):
        t = 0.5 * k
        z = 0.7 * k
        cx = math.cos(t)
        cy = math.sin(t)
        ring = []
        for offset in [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (-0.1, 0.0, 0.0)]:
            pos = (cx + offset[0], cy + offset[1], z + offset[2])
            _add_node(G, idx, "C", pos)
            ring.append(idx)
            idx += 1
        rings.append(ring)

    G.graph["aromatic_rings"] = rings
    return G


def test_assign_rs_tetrahedral() -> None:
    G = nx.Graph()
    _add_node(G, 0, "C", (0.0, 0.0, 0.0))
    _add_node(G, 1, "F", (1.0, 1.0, 1.0))
    _add_node(G, 2, "Cl", (-1.0, -1.0, 1.0))
    _add_node(G, 3, "Br", (-1.0, 1.0, -1.0))
    _add_node(G, 4, "I", (1.0, -1.0, -1.0))

    for i in range(1, 5):
        G.add_edge(0, i, bond_order=1.0)

    rs = assign_rs(G)
    assert len(rs) == 1
    label = next(iter(rs.values()))
    assert label in {"R", "S"}


def test_assign_ez_simple() -> None:
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

    ez = assign_ez(G)
    assert ez[(0, 1)] == "Z"


def test_assign_axial_allene_and_annotate_axes() -> None:
    G = _make_allene_graph()
    axial, axes = assign_axial(G)
    assert not axial
    assert len(axes) == 1
    assert axes[0][2] in {"Rₐ", "Sₐ"}

    summary = annotate_stereo(G)
    assert "axial" in summary
    assert "stereo_axes" in G.graph
    assert any(axis["kind"] == "axial" for axis in G.graph["stereo_axes"])


def test_assign_planar_metallocene() -> None:
    G = _make_planar_metallocene_graph()
    planar, axes = assign_planar(G)
    assert axes == []
    assert len(planar) == 1
    label = next(iter(planar.values()))
    assert label in {"Rₚ", "Sₚ"}


def test_assign_helical_synthetic() -> None:
    G = _make_helical_graph()
    labels = assign_helical(G)
    assert len(labels) == 1
    assert labels[0][2] in {"P", "M"}
