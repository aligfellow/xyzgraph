"""Tests for utility functions."""

import re

import networkx as nx
import pytest

from xyzgraph import count_frames_and_atoms, read_xyz_file, read_xyz_frames
from xyzgraph.data_loader import BOHR_TO_ANGSTROM
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
    """5+7 fused rings (azulene topology): returns [5, 7]."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    G.add_edges_from([(0, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)])
    assert sorted(map(len, smallest_rings(G))) == [5, 7]


def test_xyz_readers_support_variable_atom_counts(tmp_path):
    """Each frame uses its own atom-count header; trailing blank lines are ignored."""
    path = tmp_path / "variable.xyz"
    path.write_text("2\nframe 0\nH 0 0 0\nH 1 0 0\n3\nframe 1\nO 0 0 0\nH 1 0 0\nH 0 1 0\n\n\n", encoding="utf-8")

    assert [len(frame) for frame in read_xyz_frames(str(path))] == [2, 3]
    assert read_xyz_file(str(path), frame=1) == [("O", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)), ("H", (0.0, 1.0, 0.0))]
    with pytest.raises(ValueError, match="File has 2 frame"):
        read_xyz_file(str(path), frame=2)
    with pytest.raises(ValueError, match="use read_xyz_frames"):
        count_frames_and_atoms(str(path))


def test_xyz_readers_uniform_trajectory(tmp_path):
    """Atomic numbers map to symbols, Bohr converts to Angstrom, uniform frames count."""
    path = tmp_path / "uniform.xyz"
    path.write_text("2\nframe 0\n8 0 0 0\n1 1 0 0\n2\nframe 1\nO 0 0 0\nH 0 -2 0.5\n", encoding="utf-8")

    frames = read_xyz_frames(str(path), bohr_units=True)
    assert [[symbol for symbol, _ in atoms] for atoms in frames] == [["O", "H"], ["O", "H"]]
    assert frames[1][1][1] == pytest.approx((0.0, -2 * BOHR_TO_ANGSTROM, 0.5 * BOHR_TO_ANGSTROM))
    assert count_frames_and_atoms(str(path)) == (2, 2)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("\n\n", "Empty XYZ file"),
        ("1\nc\nH 0 0 0\n\nx\n", "Frame 1: expected atom count at line 5"),
        ("-1\nc\n", "Frame 0: atom count must be non-negative"),
        ("1\n", "Frame 0: missing comment line"),
        ("1\nc\nH 0 0 0\n2\nc\nH 0 0 0\n", "Frame 1 truncated: expected 2 atoms, found 1"),
        ("1\nc\nH 0 0\n", "Frame 0, atom 0: expected at least 4 columns"),
        ("1\nc\nH 0 zero 0\n", "Frame 0, atom 0: invalid coordinates"),
        ("1\nc\n999 0 0 0\n", "Frame 0, atom 0: unknown atomic number 999"),
        ("1\nc\nXx 0 0 0\n", "Frame 0, atom 0: unknown element symbol 'Xx'"),
    ],
)
def test_read_xyz_frames_rejects_malformed_input(tmp_path, text, message):
    path = tmp_path / "bad.xyz"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_frames(str(path))
