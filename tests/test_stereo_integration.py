"""Integration tests for stereochemistry assignment using real structures."""

from __future__ import annotations

from pathlib import Path

from xyzgraph import build_graph
from xyzgraph.stereo import assign_axial, assign_ez, assign_helical, assign_planar

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

STRUCTURES = Path(__file__).resolve().parent.parent / "examples" / "stereo"


def _label_from_map(labels: dict[tuple[int, int], str]) -> str:
    assert len(labels) == 1
    return next(iter(labels.values()))


def test_axial_binol_pair() -> None:
    r_binol = build_graph(str(STRUCTURES / "R_binol.xyz"))
    s_binol = build_graph(str(STRUCTURES / "S_binol.xyz"))
    r_label = _label_from_map(assign_axial(r_binol)[0])
    s_label = _label_from_map(assign_axial(s_binol)[0])
    # Enantiomers must get opposite labels
    assert {r_label, s_label} == {"Rₐ", "Sₐ"}


def test_axial_metallocene_ferrocene_pair() -> None:
    a = build_graph(str(STRUCTURES / "Ra_ferrocene_axial.xyz"))
    b = build_graph(str(STRUCTURES / "Sa_ferrocene_axial.xyz"))
    a_label = _label_from_map(assign_axial(a)[0])
    b_label = _label_from_map(assign_axial(b)[0])
    assert {a_label, b_label} == {"Rₐ", "Sₐ"}


def test_axial_allene_pair() -> None:
    ra = build_graph(str(STRUCTURES / "Ra_allene.xyz"))
    sa = build_graph(str(STRUCTURES / "Sa_allene.xyz"))
    # Allene axes are non-edge (terminal C's aren't bonded)
    ra_axes = assign_axial(ra)[1]
    sa_axes = assign_axial(sa)[1]
    assert len(ra_axes) == 1
    assert len(sa_axes) == 1
    assert {ra_axes[0][2], sa_axes[0][2]} == {"Rₐ", "Sₐ"}


def test_planar_ferrocene_pair() -> None:
    rp = build_graph(str(STRUCTURES / "Rp_ferrocene.xyz"))
    sp = build_graph(str(STRUCTURES / "Sp_ferrocene.xyz"))
    rp_label = _label_from_map(assign_planar(rp)[0])
    sp_label = _label_from_map(assign_planar(sp)[0])
    # Enantiomers must get opposite labels
    assert {rp_label, sp_label} == {"Rₚ", "Sₚ"}


def test_22paracyclophane_achiral() -> None:
    """Unsubstituted [2.2]paracyclophane is achiral — no stereo labels."""
    from xyzgraph.stereo import annotate_stereo

    G = build_graph(str(STRUCTURES / "22paracyclophane.xyz"))
    s = annotate_stereo(G)
    assert not any(s.values())


def test_22paracyclophane_F_planar() -> None:
    """Mono-F [2.2]paracyclophane has planar chirality."""
    from xyzgraph.stereo import annotate_stereo

    G = build_graph(str(STRUCTURES / "22paracyclophane_F.xyz"))
    s = annotate_stereo(G)
    planar = s["planar"]
    assert len(planar) >= 1
    assert all(entry["label"] in {"Rₚ", "Sₚ"} for entry in planar)


def test_hindered_biaryl_axial_pair() -> None:
    """Hindered biaryl (bridged biphenyl) pair gets opposite axial labels."""
    ra = build_graph(str(STRUCTURES / "Ra_hindered_biaryl.xyz"))
    sa = build_graph(str(STRUCTURES / "Sa_hindered_biaryl.xyz"))
    ra_label = _label_from_map(assign_axial(ra)[0])
    sa_label = _label_from_map(assign_axial(sa)[0])
    assert {ra_label, sa_label} == {"Rₐ", "Sₐ"}


def test_helicene_pair() -> None:
    m = build_graph(str(STRUCTURES / "M_helicene.xyz"))
    p = build_graph(str(STRUCTURES / "P_helicene.xyz"))
    assert assign_helical(m)[0][2] == "M"
    assert assign_helical(p)[0][2] == "P"


def test_ez_2_butene_pair() -> None:
    e = build_graph(str(STRUCTURES / "E_2butene.xyz"))
    z = build_graph(str(STRUCTURES / "Z_2butene.xyz"))
    assert _label_from_map(assign_ez(e)) == "E"
    assert _label_from_map(assign_ez(z)) == "Z"
