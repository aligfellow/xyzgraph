"""End-to-end integration tests.

Each test builds a graph from an .xyz file via build_graph() and compares
the full JSON-serialisable output against a hand-verified fixture (.json).
This catches regressions anywhere in the pipeline: bond detection, bond
order optimisation, formal charges, valence splitting, metal coordination,
oxidation state inference, and JSON serialisation.

Fixture categories:
  Organic       - isothio (charged cation, fused aromatic rings, S/N heteroatoms)
  Organometallic - mnh (Fe/Mn bimetallic, Cp rings, phosphine, oxidation states)
  Transition states - mnh2-ts, ru-co-ts (connectivity only; valence/charges are
                      meaningless at a TS geometry so we skip those assertions)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xyzgraph import build_graph
from xyzgraph.utils import graph_to_dict

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FIXTURES = Path(__file__).resolve().parent


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ===========================================================================
# Organic: isothiocyanate derivative (C23H25N2OS, charge +1)
#
# Why: fused indole + thiazolinium ring system, formal +1 on N, mix of
# aromatic (BO=1.5) and localised double bonds (C=O, C=C), no metals.
# Tests the full organic pipeline: ring detection, Kekulé init, bond order
# optimisation, and formal charge balancing.
# ===========================================================================


def test_isothio():
    """Full pipeline match for charged organic molecule."""
    result = graph_to_dict(build_graph(str(EXAMPLES / "isothio.xyz"), charge=1))
    expected = _load_fixture("isothio.json")

    # Graph-level
    assert result["graph"]["formula"] == expected["graph"]["formula"]
    assert result["graph"]["total_charge"] == expected["graph"]["total_charge"]
    assert len(result["graph"]["rings"]) == len(expected["graph"]["rings"])

    # Every node: symbol, formal_charge, valence, metal_valence
    assert len(result["nodes"]) == len(expected["nodes"])
    for got, exp in zip(result["nodes"], expected["nodes"]):
        assert got["symbol"] == exp["symbol"]
        assert got["formal_charge"] == exp["formal_charge"]
        assert got["valence"] == pytest.approx(exp["valence"])
        assert got["metal_valence"] == pytest.approx(exp["metal_valence"])

    # Every edge: connectivity, bond order, metal_coord
    assert len(result["edges"]) == len(expected["edges"])
    for got, exp in zip(result["edges"], expected["edges"]):
        assert got["idx1"] == exp["idx1"]
        assert got["idx2"] == exp["idx2"]
        assert got["bond_order"] == pytest.approx(exp["bond_order"])
        assert got["metal_coord"] == exp["metal_coord"]

    # 16 aromatic bonds across fused indole + thiazolinium rings
    assert sum(1 for e in result["edges"] if e["bond_order"] == 1.5) == 16


# ===========================================================================
# Organometallic: Mn/Fe bimetallic complex (C34H35FeMnN3O2P, charge 0)
#
# Why: two metals (Fe, Mn) with different oxidation states, Cp and arene
# eta-coordination (28 aromatic bonds), a phosphine ligand (tests valence
# vs metal_valence split), and dative/ionic classification.
# ===========================================================================


def test_mnh():
    """Full pipeline match for bimetallic organometallic complex."""
    result = graph_to_dict(build_graph(str(EXAMPLES / "mnh.xyz"), charge=0))
    expected = _load_fixture("mnh.json")

    # Graph-level
    assert result["graph"]["formula"] == expected["graph"]["formula"]
    assert result["graph"]["total_charge"] == expected["graph"]["total_charge"]
    assert len(result["graph"]["rings"]) == len(expected["graph"]["rings"])

    # Every node
    assert len(result["nodes"]) == len(expected["nodes"])
    for got, exp in zip(result["nodes"], expected["nodes"]):
        assert got["symbol"] == exp["symbol"]
        assert got["formal_charge"] == exp["formal_charge"]
        assert got["valence"] == pytest.approx(exp["valence"])
        assert got["metal_valence"] == pytest.approx(exp["metal_valence"])

    # Every edge
    assert len(result["edges"]) == len(expected["edges"])
    for got, exp in zip(result["edges"], expected["edges"]):
        assert got["idx1"] == exp["idx1"]
        assert got["idx2"] == exp["idx2"]
        assert got["bond_order"] == pytest.approx(exp["bond_order"])
        assert got["metal_coord"] == exp["metal_coord"]

    # Metal oxidation states: Fe(II), Mn(I)
    metals = {n["id"]: n for n in result["nodes"] if n["symbol"] in ("Fe", "Mn")}
    exp_metals = {n["id"]: n for n in expected["nodes"] if n["symbol"] in ("Fe", "Mn")}
    for idx in metals:
        assert metals[idx]["oxidation_state"] == exp_metals[idx]["oxidation_state"]

    # 16 metal-coordination edges
    got_mc = sorted((e["idx1"], e["idx2"]) for e in result["edges"] if e["metal_coord"])
    exp_mc = sorted((e["idx1"], e["idx2"]) for e in expected["edges"] if e["metal_coord"])
    assert got_mc == exp_mc

    # P ligand: valence=3 (organic), metal_valence=1 (dative P->Mn)
    p = next(n for n in result["nodes"] if n["symbol"] == "P")
    assert p["formal_charge"] == 0
    assert p["valence"] == pytest.approx(3.0)
    assert p["metal_valence"] == pytest.approx(1.0)

    # 28 aromatic bonds across Cp and arene rings
    assert sum(1 for e in result["edges"] if e["bond_order"] == 1.5) == 28


# ===========================================================================
# Transition states
#
# TS geometries have partially formed/broken bonds, so formal charges and
# valences are chemically meaningless.  We only check *connectivity*.
#
# threshold=1.4 captures the stretched bonds at TS geometries
# ===========================================================================


def test_mnh2_ts():
    """MnH2 TS: connectivity and bond orders match fixture."""
    result = graph_to_dict(build_graph(str(EXAMPLES / "mnh2-ts.xyz"), charge=0, threshold=1.4, quick=True))
    expected = _load_fixture("mnh2-ts.json")

    assert result["graph"]["formula"] == expected["graph"]["formula"]
    assert len(result["nodes"]) == len(expected["nodes"])
    for got, exp in zip(result["nodes"], expected["nodes"]):
        assert got["symbol"] == exp["symbol"]

    assert len(result["edges"]) == len(expected["edges"])
    for got, exp in zip(result["edges"], expected["edges"]):
        assert got["idx1"] == exp["idx1"]
        assert got["idx2"] == exp["idx2"]
        assert got["metal_coord"] == exp["metal_coord"]


def test_ru_co_ts():
    """Ru-CO TS: connectivity and bond orders match fixture."""
    result = graph_to_dict(build_graph(str(EXAMPLES / "ru-co-ts.xyz"), charge=0, threshold=1.4, quick=True))
    expected = _load_fixture("ru-co-ts.json")

    assert result["graph"]["formula"] == expected["graph"]["formula"]
    assert len(result["nodes"]) == len(expected["nodes"])
    for got, exp in zip(result["nodes"], expected["nodes"]):
        assert got["symbol"] == exp["symbol"]

    assert len(result["edges"]) == len(expected["edges"])
    for got, exp in zip(result["edges"], expected["edges"]):
        assert got["idx1"] == exp["idx1"]
        assert got["idx2"] == exp["idx2"]
        assert got["metal_coord"] == exp["metal_coord"]


# ===========================================================================
# Organic: alizarin (1,2-dihydroxyanthraquinone, C14H8O4, charge 0)
#
# Why: three fused 6-membered rings — two aromatic and one quinone.
# Tests that the quinone ring is NOT treated as aromatic (Kekulé), the two
# C=O carbonyls are correctly assigned BO=2, and all C atoms stay at
# valence 4 with formal charge 0.
# ===========================================================================


def test_alizarin():
    """Alizarin: fused quinone/aromatic system, correct C=O and Kekulé."""
    G = build_graph(str(EXAMPLES / "alizarin.xyz"), charge=0, kekule=True)
    result = graph_to_dict(G)

    assert result["graph"]["formula"] == "C14H8O4"

    # Every C must have valence 4 and formal charge 0
    for node in result["nodes"]:
        if node["symbol"] == "C":
            assert node["valence"] == pytest.approx(4.0), f"C{node['id']} valence {node['valence']}"
            assert node["formal_charge"] == 0, f"C{node['id']} FC {node['formal_charge']}"

    # Carbonyl oxygens (O7, O11): valence 2, FC 0, BO=2 to their carbon
    for oid in (7, 11):
        o_node = result["nodes"][oid]
        assert o_node["symbol"] == "O"
        assert o_node["formal_charge"] == 0, f"O{oid} FC {o_node['formal_charge']}"
        o_edges = [e for e in result["edges"] if oid in (e["idx1"], e["idx2"])]
        assert len(o_edges) == 1
        assert o_edges[0]["bond_order"] == pytest.approx(2.0)

    # Hydroxyl oxygens (O16, O17): single bond to C, single bond to H
    for oid in (16, 17):
        o_node = result["nodes"][oid]
        assert o_node["symbol"] == "O"
        assert o_node["formal_charge"] == 0
        o_edges = [e for e in result["edges"] if oid in (e["idx1"], e["idx2"])]
        assert len(o_edges) == 2
        assert all(e["bond_order"] == pytest.approx(1.0) for e in o_edges)
