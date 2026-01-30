"""Tests for xyzgraph graph building functionality."""

from xyzgraph import build_graph


class TestBuildGraph:
    """Tests for build_graph function."""

    def test_simple_molecule(self):
        """Test building a graph from a simple atom list."""
        # Methane: C with 4 H atoms
        atoms = [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (0.629, 0.629, 0.629)),
            ("H", (-0.629, -0.629, 0.629)),
            ("H", (-0.629, 0.629, -0.629)),
            ("H", (0.629, -0.629, -0.629)),
        ]
        G = build_graph(atoms, charge=0)

        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 4  # 4 C-H bonds

    def test_benzene(self, tmp_path):
        """Test building a graph for benzene."""
        xyz_content = """12
benzene
C  0.000  1.396  0.000
C  1.209  0.698  0.000
C  1.209 -0.698  0.000
C  0.000 -1.396  0.000
C -1.209 -0.698  0.000
C -1.209  0.698  0.000
H  0.000  2.479  0.000
H  2.147  1.240  0.000
H  2.147 -1.240  0.000
H  0.000 -2.479  0.000
H -2.147 -1.240  0.000
H -2.147  1.240  0.000
"""
        xyz_file = tmp_path / "benzene.xyz"
        xyz_file.write_text(xyz_content)

        G = build_graph(str(xyz_file), charge=0)

        assert G.number_of_nodes() == 12
        # 6 C-C bonds + 6 C-H bonds = 12 bonds
        assert G.number_of_edges() == 12

    def test_charged_molecule(self):
        """Test building a graph with a net charge."""
        # Ammonium: NH4+
        atoms = [
            ("N", (0.0, 0.0, 0.0)),
            ("H", (0.5, 0.5, 0.5)),
            ("H", (-0.5, -0.5, 0.5)),
            ("H", (-0.5, 0.5, -0.5)),
            ("H", (0.5, -0.5, -0.5)),
        ]
        G = build_graph(atoms, charge=1)

        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 4
