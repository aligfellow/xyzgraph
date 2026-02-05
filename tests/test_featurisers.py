"""Tests for graph featurisers."""

import pytest

rdkit = pytest.importorskip("rdkit")

from xyzgraph import build_graph_rdkit, compute_gasteiger_charges  # noqa: E402

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.8, 0.6, 0.0)),
    ("H", (-0.8, 0.6, 0.0)),
]

ETHENE = [
    ("C", (0.0, 0.0, 0.0)),
    ("C", (1.34, 0.0, 0.0)),
    ("H", (-0.5, 0.87, 0.0)),
    ("H", (-0.5, -0.87, 0.0)),
    ("H", (1.84, 0.87, 0.0)),
    ("H", (1.84, -0.87, 0.0)),
]


class TestComputeGasteigerCharges:
    def test_adds_charges(self):
        G = build_graph_rdkit(WATER)
        G = compute_gasteiger_charges(G)
        for node in G.nodes():
            assert "gasteiger" in G.nodes[node]["charges"]
            assert "gasteiger_raw" in G.nodes[node]["charges"]

    def test_charges_sum_to_target(self):
        G = build_graph_rdkit(WATER)
        G = compute_gasteiger_charges(G, target_charge=0)
        total = sum(G.nodes[n]["charges"]["gasteiger"] for n in G.nodes())
        assert total == pytest.approx(0.0, abs=1e-6)

    def test_returns_same_graph(self):
        G = build_graph_rdkit(ETHENE)
        G2 = compute_gasteiger_charges(G)
        assert G2 is G

    def test_agg_charge_updated(self):
        G = build_graph_rdkit(WATER)
        G = compute_gasteiger_charges(G)
        # Oxygen's agg_charge should include its own + H neighbors
        o_node = next(n for n in G.nodes() if G.nodes[n]["symbol"] == "O")
        o_agg = G.nodes[o_node]["agg_charge"]
        expected = G.nodes[o_node]["charges"]["gasteiger"]
        for nbr in G.neighbors(o_node):
            if G.nodes[nbr]["symbol"] == "H":
                expected += G.nodes[nbr]["charges"]["gasteiger"]
        assert o_agg == pytest.approx(expected)
