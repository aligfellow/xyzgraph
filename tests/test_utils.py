"""Tests for utility functions."""

import networkx as nx
import pytest

from xyzgraph import count_frames_and_atoms, read_xyz_file, read_xyz_frames
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


def test_read_xyz_frames_uniform_trajectory(tmp_path):
    xyz_file = tmp_path / "uniform.xyz"
    xyz_file.write_text(
        "2\nframe 0\nH 0 0 0\n8 1 0 0\n2\nframe 1\nH 0 1 0\nO 1 1 0\n",
        encoding="utf-8",
    )

    assert read_xyz_frames(str(xyz_file)) == [
        [("H", (0.0, 0.0, 0.0)), ("O", (1.0, 0.0, 0.0))],
        [("H", (0.0, 1.0, 0.0)), ("O", (1.0, 1.0, 0.0))],
    ]
    assert count_frames_and_atoms(str(xyz_file)) == (2, 2)


def test_read_xyz_frames_variable_atom_counts(tmp_path):
    xyz_file = tmp_path / "variable.xyz"
    xyz_file.write_text(
        "2\nframe 0\nH 0 0 0\nH 1 0 0\n"
        "3\nframe 1\nH 0 1 0\nO 1 1 0\nH 2 1 0\n"
        "2\nframe 2\nH 0 2 0\nH 1 2 0\n"
        "2\nframe 3\nH 0 3 0\nH 1 3 0\n",
        encoding="utf-8",
    )

    frames = read_xyz_frames(str(xyz_file))

    assert [len(frame) for frame in frames] == [2, 3, 2, 2]
    with pytest.raises(ValueError, match="read_xyz_frames"):
        count_frames_and_atoms(str(xyz_file))


def test_read_xyz_file_selects_frame_after_atom_count_change(tmp_path):
    xyz_file = tmp_path / "variable.xyz"
    xyz_file.write_text(
        "2\nframe 0\nH 0 0 0\nH 1 0 0\n3\nframe 1\nH 0 1 0\nO 1 1 0\nH 2 1 0\n2\nframe 2\nC 0 2 0\nO 1 2 0\n",
        encoding="utf-8",
    )

    assert read_xyz_file(str(xyz_file), frame=2) == [
        ("C", (0.0, 2.0, 0.0)),
        ("O", (1.0, 2.0, 0.0)),
    ]


def test_read_xyz_frames_rejects_truncated_frame(tmp_path):
    xyz_file = tmp_path / "truncated.xyz"
    xyz_file.write_text(
        "2\nframe 0\nH 0 0 0\nH 1 0 0\n3\nframe 1\nH 0 1 0\nO 1 1 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Frame 1 truncated: expected 3 atoms, found 2"):
        read_xyz_frames(str(xyz_file))


def test_count_frames_and_atoms_rejects_negative_atom_count(tmp_path):
    """Malformed negative counts must fail rather than stalling frame iteration."""
    xyz_file = tmp_path / "negative.xyz"
    xyz_file.write_text("-2\ninvalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="atom count must be non-negative"):
        count_frames_and_atoms(str(xyz_file))
