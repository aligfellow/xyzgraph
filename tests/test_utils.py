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


def _write_xyz(tmp_path, text: str) -> str:
    xyz_file = tmp_path / "input.xyz"
    xyz_file.write_text(text, encoding="utf-8")
    return str(xyz_file)


def test_read_xyz_frames_converts_bohr_to_angstrom(tmp_path):
    path = _write_xyz(tmp_path, "1\nbohr\nH 1.0 -2.0 0.5\n")

    [[(symbol, coords)]] = read_xyz_frames(path, bohr_units=True)

    assert symbol == "H"
    assert coords == pytest.approx((BOHR_TO_ANGSTROM, -2.0 * BOHR_TO_ANGSTROM, 0.5 * BOHR_TO_ANGSTROM))
    assert read_xyz_file(path, bohr_units=True) == [(symbol, coords)]


def test_read_xyz_frames_accepts_atomic_numbers(tmp_path):
    path = _write_xyz(tmp_path, "3\nwater\n8 0 0 0\n1 1 0 0\n1 0 1 0\n")

    assert [symbol for symbol, _ in read_xyz_file(path)] == ["O", "H", "H"]


@pytest.mark.parametrize(
    ("atom_line", "message"),
    [
        ("999 0 0 0", "Frame 0, atom 1: unknown atomic number 999"),
        ("Xx 0 0 0", "Frame 0, atom 1: unknown element symbol 'Xx'"),
        ("H 0 zero 0", "Frame 0, atom 1: invalid coordinates"),
        ("H 0 0", "Frame 0, atom 1: expected at least 4 columns"),
        ("", "Frame 0, atom 1: expected at least 4 columns"),
    ],
)
def test_xyz_readers_reject_bad_atom_records(tmp_path, atom_line, message):
    path = _write_xyz(tmp_path, f"3\ncomment\nH 0 0 0\n{atom_line}\nH 1 0 0\n")

    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_frames(path)
    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_file(path)


@pytest.mark.parametrize(
    ("text", "message", "frame"),
    [
        ("", "Empty XYZ file", 0),
        ("\n\n", "Empty XYZ file", 0),
        ("two\ncomment\nH 0 0 0\nH 1 0 0\n", "Frame 0: expected atom count at line 1", 0),
        ("1\nc\nH 0 0 0\n\n1\nc\nH 0 0 0\n", "Frame 1: expected atom count at line 4", 1),
        ("-1\ncomment\n", "Frame 0: atom count must be non-negative", 0),
        ("2\n", "Frame 0: missing comment line", 0),
        ("1\nc\nH 0 0 0\n0\n\n", "Frame 1: missing comment line", 1),
        ("2\ncomment\nH 0 0 0\n", "Frame 0 truncated: expected 2 atoms, found 1", 0),
        ("2\ncomment\nH 0 0 0\n\n", "Frame 0 truncated: expected 2 atoms, found 1", 0),
    ],
)
def test_xyz_readers_reject_malformed_layout(tmp_path, text, message, frame):
    path = _write_xyz(tmp_path, text)

    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_frames(path)
    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_file(path, frame=frame)


def test_xyz_readers_ignore_trailing_blank_lines(tmp_path):
    path = _write_xyz(tmp_path, "1\nframe 0\nH 0 0 0\n1\nframe 1\nH 0 1 0\n\n   \n\n")

    assert read_xyz_frames(path) == [[("H", (0.0, 0.0, 0.0))], [("H", (0.0, 1.0, 0.0))]]
    assert read_xyz_file(path, frame=1) == [("H", (0.0, 1.0, 0.0))]
    assert count_frames_and_atoms(path) == (2, 1)
    with pytest.raises(ValueError, match=r"Frame 2 out of range\. File has 2 frame\(s\)\."):
        read_xyz_file(path, frame=2)


def test_xyz_readers_keep_blank_comment_lines(tmp_path):
    path = _write_xyz(tmp_path, "0\n\n1\n\nH 0 0 0\n")

    assert read_xyz_frames(path) == [[], [("H", (0.0, 0.0, 0.0))]]
    assert read_xyz_file(path, frame=1) == [("H", (0.0, 0.0, 0.0))]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (-1, "Frame -1 out of range. Frame index must be non-negative."),
        (2, "Frame 2 out of range. File has 2 frame(s)."),
    ],
)
def test_read_xyz_file_rejects_out_of_range_frame(tmp_path, frame, message):
    path = _write_xyz(tmp_path, "1\nframe 0\nH 0 0 0\n2\nframe 1\nH 0 1 0\nH 1 1 0\n")

    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_file(path, frame=frame)


def test_read_xyz_file_negative_frame_raises_before_reading(tmp_path):
    missing = str(tmp_path / "missing.xyz")

    with pytest.raises(ValueError, match="must be non-negative"):
        read_xyz_file(missing, frame=-1)


BAD_CONTENT_FRAME = "2\nbad content\nXx 0 0 0\nH 0 zero 0\n"
GOOD_FRAME = "3\ngood\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
GOOD_ATOMS = [("O", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)), ("H", (0.0, 1.0, 0.0))]


def test_read_xyz_file_skips_atom_records_before_requested_frame(tmp_path):
    path = _write_xyz(tmp_path, BAD_CONTENT_FRAME + GOOD_FRAME)

    assert read_xyz_file(path, frame=1) == GOOD_ATOMS
    with pytest.raises(ValueError, match="Frame 0, atom 0: unknown element symbol 'Xx'"):
        read_xyz_frames(path)


@pytest.mark.parametrize(
    ("bad_frame", "message"),
    [
        ("x\ncomment\n", "Frame 0: expected atom count at line 1"),
        ("-2\ncomment\n", "Frame 0: atom count must be non-negative"),
    ],
)
def test_read_xyz_file_rejects_bad_layout_before_requested_frame(tmp_path, bad_frame, message):
    path = _write_xyz(tmp_path, bad_frame + GOOD_FRAME)

    with pytest.raises(ValueError, match=re.escape(message)):
        read_xyz_file(path, frame=1)


def test_read_xyz_file_rejects_truncated_frame_before_requested_frame(tmp_path):
    path = _write_xyz(tmp_path, GOOD_FRAME + "4\ntruncated\nH 0 0 0\n")

    with pytest.raises(ValueError, match=re.escape("Frame 1 truncated: expected 4 atoms, found 1")):
        read_xyz_file(path, frame=2)


@pytest.mark.parametrize(
    ("trailing", "strict_message"),
    [
        (BAD_CONTENT_FRAME, "Frame 1, atom 0: unknown element symbol 'Xx'"),
        ("not a header\n", "Frame 1: expected atom count at line 6"),
        ("5\ntruncated\nH 0 0 0\n", "Frame 1 truncated: expected 5 atoms, found 1"),
        ("\n\ngarbage after blank lines\n", "Frame 1: expected atom count at line 6"),
    ],
)
def test_read_xyz_file_ignores_content_after_requested_frame(tmp_path, trailing, strict_message):
    path = _write_xyz(tmp_path, GOOD_FRAME + trailing)

    assert read_xyz_file(path, frame=0) == GOOD_ATOMS
    with pytest.raises(ValueError, match=re.escape(strict_message)):
        read_xyz_frames(path)


def test_read_xyz_file_out_of_range_reads_layout_to_end(tmp_path):
    """Frames are counted to EOF, so a later layout error wins over the range error."""
    path = _write_xyz(tmp_path, GOOD_FRAME + BAD_CONTENT_FRAME)
    with pytest.raises(ValueError, match=re.escape("Frame 5 out of range. File has 2 frame(s).")):
        read_xyz_file(path, frame=5)

    path = _write_xyz(tmp_path, GOOD_FRAME + "x\n")
    with pytest.raises(ValueError, match="Frame 1: expected atom count at line 6"):
        read_xyz_file(path, frame=5)


def test_count_frames_and_atoms_rejects_malformed_trajectory(tmp_path):
    with pytest.raises(ValueError, match="read_xyz_frames"):
        count_frames_and_atoms(_write_xyz(tmp_path, GOOD_FRAME + "1\nsmaller\nH 0 0 0\n"))
    with pytest.raises(ValueError, match="not evenly divisible"):
        count_frames_and_atoms(_write_xyz(tmp_path, GOOD_FRAME + "3\ntruncated\nH 0 0 0\n"))
