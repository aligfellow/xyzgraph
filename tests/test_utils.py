"""Tests for utility functions."""

import networkx as nx

from xyzgraph.utils import smallest_rings


def test_smallest_rings_empty_graph():
    """Empty or edge-less graphs return an empty ring list."""
    assert smallest_rings(nx.Graph()) == []
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    assert smallest_rings(G) == []


def test_smallest_rings_benzene():
    """Single benzene-like 6-ring: one ring of size 6."""
    G = nx.cycle_graph(6)
    rings = smallest_rings(G)
    assert len(rings) == 1
    assert len(rings[0]) == 6


def test_smallest_rings_azulene_topology():
    """5+7 fused rings (azulene topology): returns [5, 7], not [6, 6] or larger."""
    G = nx.Graph()
    # 5-ring on atoms 0..4, sharing edge 0-4 with a 7-ring through 5..9
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    G.add_edges_from([(0, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)])
    sizes = sorted(len(r) for r in smallest_rings(G))
    assert sizes == [5, 7]
