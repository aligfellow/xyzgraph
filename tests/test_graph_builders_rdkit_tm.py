"""Tests for RDKit-TM graph building and graph isomorphism matching."""

from pathlib import Path

import networkx as nx
import pytest

scipy = pytest.importorskip("scipy")

from xyzgraph.graph_builders_rdkit_tm import _partial_graph_matching  # noqa: E402

PT_COMPLEX_XYZ = Path(__file__).parent.parent / "examples" / "pt_complex.xyz"


class TestPartialGraphMatching:
    """Test graph-distance + neighbour-symbol partial matching."""

    def _make_chain(self, symbols):
        """Build a linear chain graph: A-B-C-..."""
        G = nx.Graph()
        for i, sym in enumerate(symbols):
            G.add_node(i, symbol=sym)
        for i in range(len(symbols) - 1):
            G.add_edge(i, i + 1)
        return G

    def test_identical_graphs(self):
        """Perfect match when graphs are identical."""
        G1 = self._make_chain(["C", "O", "C"])
        G2 = self._make_chain(["C", "O", "C"])
        mapping = _partial_graph_matching(G1, G2)
        assert len(mapping) == 3
        for r, x in mapping.items():
            assert G1.nodes[r]["symbol"] == G2.nodes[x]["symbol"]

    def test_relabeled_graphs(self):
        """Matching when node indices differ but structure is same."""
        G1 = self._make_chain(["C", "O", "C"])
        G2 = nx.relabel_nodes(G1, {0: 10, 1: 11, 2: 12})
        mapping = _partial_graph_matching(G1, G2)
        assert len(mapping) == 3
        for r, x in mapping.items():
            assert G1.nodes[r]["symbol"] == G2.nodes[x]["symbol"]

    def test_element_count_mismatch_raises(self):
        """Raises when element counts don't match."""
        G1 = self._make_chain(["C", "O", "C"])
        G2 = self._make_chain(["C", "O", "N"])
        with pytest.raises(ValueError, match="count mismatch"):
            _partial_graph_matching(G1, G2)

    def test_branched_graph(self):
        """Matching on a branched structure (star graph)."""
        G1 = nx.Graph()
        G1.add_node(0, symbol="C")
        for i in range(1, 4):
            G1.add_node(i, symbol="H")
            G1.add_edge(0, i)

        G2 = nx.Graph()
        G2.add_node(10, symbol="C")
        for i in range(11, 14):
            G2.add_node(i, symbol="H")
            G2.add_edge(10, i)

        mapping = _partial_graph_matching(G1, G2)
        assert len(mapping) == 4
        assert G2.nodes[mapping[0]]["symbol"] == "C"


class TestGraphIsomorphism:
    """Test graph isomorphism used in build_graph_rdkit_tm alignment."""

    def test_isomorphic_detection(self):
        from networkx.algorithms import isomorphism

        G1 = nx.Graph()
        G1.add_node(0, symbol="C")
        G1.add_node(1, symbol="O")
        G1.add_node(2, symbol="C")
        G1.add_edges_from([(0, 1), (1, 2)])

        G2 = nx.Graph()
        G2.add_node(0, symbol="C")
        G2.add_node(1, symbol="O")
        G2.add_node(2, symbol="C")
        G2.add_edges_from([(0, 1), (1, 2)])

        nm = isomorphism.categorical_node_match("symbol", "")
        GM = isomorphism.GraphMatcher(G1, G2, node_match=nm)
        assert GM.is_isomorphic()

    def test_non_isomorphic_different_connectivity(self):
        from networkx.algorithms import isomorphism

        G1 = nx.Graph()
        for i, s in enumerate(["C", "O", "C"]):
            G1.add_node(i, symbol=s)
        G1.add_edges_from([(0, 1), (1, 2)])

        G2 = nx.Graph()
        for i, s in enumerate(["C", "O", "C"]):
            G2.add_node(i, symbol=s)
        G2.add_edges_from([(0, 1), (1, 2), (2, 0)])

        nm = isomorphism.categorical_node_match("symbol", "")
        GM = isomorphism.GraphMatcher(G1, G2, node_match=nm)
        assert not GM.is_isomorphic()


class TestBuildGraphRdkitTm:
    """Integration tests for build_graph_rdkit_tm with TM complexes."""

    @pytest.fixture
    def xyz2mol_tm(self):
        """Skip if xyz2mol_tm is not installed."""
        return pytest.importorskip("xyz2mol_tm")

    def test_pt_complex(self, xyz2mol_tm):
        """Test Pt complex from examples/pt_complex.xyz."""
        from xyzgraph import build_graph_rdkit_tm

        G = build_graph_rdkit_tm(str(PT_COMPLEX_XYZ), charge=-1)

        # Should have 10 atoms: Pt, 3 Cl, 2 C, 4 H
        assert G.number_of_nodes() == 10
        symbols = [G.nodes[n]["symbol"] for n in G.nodes()]
        assert "Pt" in symbols
        assert symbols.count("Cl") == 3
        assert symbols.count("C") == 2
        assert symbols.count("H") == 4

    def test_pt_complex_has_metal_coord(self, xyz2mol_tm):
        """Metal-ligand bonds should have metal_coord=True."""
        from xyzgraph import build_graph_rdkit_tm

        # if not PT_COMPLEX_XYZ.exists():
        #     pytest.skip(f"Example file not found: {PT_COMPLEX_XYZ}")

        G = build_graph_rdkit_tm(str(PT_COMPLEX_XYZ), charge=-1)
        if G.number_of_edges() == 0:
            pytest.skip("xyz2mol_tmc failed or timed out")

        # Check that Pt bonds are marked as metal_coord
        pt_node = next(n for n in G.nodes() if G.nodes[n]["symbol"] == "Pt")
        for nbr in G.neighbors(pt_node):
            edge_data = G.edges[pt_node, nbr]
            assert edge_data["metal_coord"] is True
