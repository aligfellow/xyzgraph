"""Tests for RDKit-based graph building and comparison."""

import pytest

rdkit = pytest.importorskip("rdkit")

from xyzgraph import build_graph, build_graph_rdkit, compare_with_rdkit  # noqa: E402

# Ethene (C2H4)
ETHENE = [
    ("C", (0.0, 0.0, 0.0)),
    ("C", (1.34, 0.0, 0.0)),
    ("H", (-0.5, 0.87, 0.0)),
    ("H", (-0.5, -0.87, 0.0)),
    ("H", (1.84, 0.87, 0.0)),
    ("H", (1.84, -0.87, 0.0)),
]

# Water
WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.757, 0.586, 0.0)),
    ("H", (-0.757, 0.586, 0.0)),
]


class TestBuildGraphRdkit:
    def test_ethene_double_bond(self):
        G = build_graph_rdkit(ETHENE)
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 5  # 1 C=C + 4 C-H
        # Find C-C bond
        cc_bo = None
        for i, j, d in G.edges(data=True):
            if G.nodes[i]["symbol"] == "C" and G.nodes[j]["symbol"] == "C":
                cc_bo = d["bond_order"]
        assert cc_bo == pytest.approx(2.0)

    def test_water(self):
        G = build_graph_rdkit(WATER)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

    def test_node_attributes(self):
        G = build_graph_rdkit(WATER)
        o = G.nodes[0]
        assert o["symbol"] == "O"
        assert o["atomic_number"] == 8
        assert o["position"] == (0.0, 0.0, 0.0)
        assert isinstance(o["charges"], dict)
        assert "formal_charge" in o
        assert "valence" in o

    def test_edge_attributes(self):
        G = build_graph_rdkit(WATER)
        for _i, _j, d in G.edges(data=True):
            assert "bond_order" in d
            assert "distance" in d
            assert "bond_type" in d
            assert "metal_coord" in d
            assert d["distance"] > 0

    def test_metadata(self):
        G = build_graph_rdkit(WATER)
        assert G.graph["method"] == "rdkit"
        assert G.graph["total_charge"] == 0
        assert G.graph["metadata"]["source"] == "rdkit"


class TestCompareWithRdkit:
    def test_water_agreement(self):
        G_native = build_graph(WATER, charge=0)
        G_rdkit = build_graph_rdkit(WATER, charge=0)
        report = compare_with_rdkit(G_native, G_rdkit)
        assert "RDKIT" in report
        # Water should agree on topology
        assert "only_in_native=0" in report
        assert "only_in_rdkit=0" in report

    def test_auto_build_rdkit(self):
        """compare_with_rdkit builds RDKit graph when not provided."""
        G_native = build_graph(WATER, charge=0)
        report = compare_with_rdkit(G_native)
        assert "RDKIT" in report

    def test_bond_order_diffs_reported(self):
        """Differences >= 0.25 are reported."""
        G_native = build_graph(ETHENE, charge=0)
        G_rdkit = build_graph_rdkit(ETHENE, charge=0)
        report = compare_with_rdkit(G_native, G_rdkit)
        assert "Bond differences" in report
