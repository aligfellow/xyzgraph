"""Tests for xTB-based graph building (using pre-existing output)."""

from pathlib import Path

import pytest

from xyzgraph.graph_builders_xtb import _find_file, build_graph_xtb

XTB_WATER_DIR = Path(__file__).parent

# Water geometry matching the xTB calculation
WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.757, 0.586, 0.0)),
    ("H", (-0.757, 0.586, 0.0)),
]


class TestBuildGraphXtb:
    def test_from_xtb_dir(self):
        """Read pre-existing xTB output from the tests directory."""
        G = build_graph_xtb(WATER, xtb_dir=str(XTB_WATER_DIR))
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.graph["method"] == "xtb"

    def test_node_attributes(self):
        """Check that node attributes are populated from xTB output."""
        G = build_graph_xtb(WATER, xtb_dir=str(XTB_WATER_DIR))
        o = G.nodes[0]
        assert o["symbol"] == "O"
        assert o["atomic_number"] == 8
        # Real xTB Mulliken charge for oxygen
        assert o["charges"]["mulliken"] == pytest.approx(-0.562, abs=0.01)

    def test_edge_attributes(self):
        """Check that edge attributes are populated from xTB WBO."""
        G = build_graph_xtb(WATER, xtb_dir=str(XTB_WATER_DIR))
        for _i, _j, d in G.edges(data=True):
            # Real xTB WBO for O-H bonds in water
            assert d["bond_order"] == pytest.approx(0.92, abs=0.01)
            assert d["distance"] > 0
            assert "bond_type" in d

    def test_charges_sum_near_zero(self):
        """Mulliken charges should sum close to total charge (0 for neutral water)."""
        G = build_graph_xtb(WATER, xtb_dir=str(XTB_WATER_DIR))
        total = sum(G.nodes[n]["charges"]["mulliken"] for n in G.nodes())
        assert total == pytest.approx(0.0, abs=0.01)

    def test_valence_computed(self):
        """Valence should be computed from bond orders."""
        G = build_graph_xtb(WATER, xtb_dir=str(XTB_WATER_DIR))
        # Oxygen has 2 O-H bonds, each ~0.92 WBO
        o_valence = G.nodes[0]["valence"]
        assert o_valence == pytest.approx(1.84, abs=0.05)
        # Hydrogen has 1 O-H bond
        h_valence = G.nodes[1]["valence"]
        assert h_valence == pytest.approx(0.92, abs=0.01)

    def test_fallback_no_bonds(self, tmp_path):
        """Falls back to geometric build_graph when no xTB bonds found."""
        # Write wbo with all bonds below threshold (0.5)
        wbo = tmp_path / "wbo"
        wbo.write_text("  1  2  0.10\n")
        charges = tmp_path / "charges"
        charges.write_text("\n".join(["0.0"] * 3) + "\n")

        G = build_graph_xtb(WATER, xtb_dir=str(tmp_path))
        # Fallback should still produce a valid graph
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() > 0

    def test_missing_charges_defaults_to_zero(self, tmp_path):
        """When charges file is missing, default to 0.0."""
        wbo = tmp_path / "wbo"
        wbo.write_text("  1  2  0.95\n")

        G = build_graph_xtb(WATER, xtb_dir=str(tmp_path))
        for node in G.nodes():
            assert G.nodes[node]["charges"]["mulliken"] == pytest.approx(0.0)


class TestFindFile:
    def test_bare_name(self, tmp_path):
        (tmp_path / "wbo").write_text("test")
        assert _find_file(str(tmp_path), "wbo") is not None

    def test_prefixed_name(self, tmp_path):
        (tmp_path / "xtb_wbo").write_text("test")
        assert _find_file(str(tmp_path), "wbo") is not None

    def test_extension_name(self, tmp_path):
        """Find files with .{name} extension (e.g., xtb_water.wbo)."""
        (tmp_path / "xtb_water.wbo").write_text("test")
        assert _find_file(str(tmp_path), "wbo") is not None

    def test_not_found(self, tmp_path):
        assert _find_file(str(tmp_path), "wbo") is None
