"""Integration tests for stereochemistry assignment using real structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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


def test_planar_ferrocene_pair() -> None:
    rp = build_graph(str(STRUCTURES / "Rp_ferrocene.xyz"))
    sp = build_graph(str(STRUCTURES / "Sp_ferrocene.xyz"))
    rp_label = _label_from_map(assign_planar(rp)[0])
    sp_label = _label_from_map(assign_planar(sp)[0])
    # Enantiomers must get opposite labels
    assert {rp_label, sp_label} == {"Rₚ", "Sₚ"}


@pytest.mark.xfail(
    reason="aromatic ring fusion not detected in 132-atom aza-helicene — xyzgraph ring detection issue",
    strict=True,
)
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
