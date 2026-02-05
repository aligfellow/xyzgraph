"""Tests for ORCA-based graph building."""

from pathlib import Path

import pytest

from xyzgraph import OrcaParseError, build_graph_orca

ORCA_WATER = Path(__file__).parent / "orca_water.out"


class TestBuildGraphOrca:
    def test_water(self):
        G = build_graph_orca(str(ORCA_WATER))
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.graph["method"] == "orca"

    def test_node_attributes(self):
        G = build_graph_orca(str(ORCA_WATER))
        symbols = [G.nodes[i]["symbol"] for i in G.nodes()]
        assert sorted(symbols) == ["H", "H", "O"]
        for n in G.nodes():
            assert "atomic_number" in G.nodes[n]
            assert "position" in G.nodes[n]
            assert "mulliken" in G.nodes[n]["charges"]
            assert "formal_charge" in G.nodes[n]
            assert "valence" in G.nodes[n]

    def test_metadata(self):
        G = build_graph_orca(str(ORCA_WATER))
        assert G.graph["metadata"]["source"] == "orca"
        assert G.graph["metadata"]["source_file"] == str(ORCA_WATER)
        assert "bond_threshold" in G.graph["metadata"]

    def test_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.out"
        bad_file.write_text("not an orca output")
        with pytest.raises(OrcaParseError):
            build_graph_orca(str(bad_file))
