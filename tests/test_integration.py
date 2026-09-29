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


def test_second_hydrogen_contact_is_a_hydrogen_bond():
    """At equilibrium an H's longer contact to an O is an NCI hydrogen bond; a transition state keeps it bonded."""
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (1.0, 0.0, 0.0)),
        ("H", (-0.33, 0.94, 0.0)),
        ("O", (2.4, 0.0, 0.0)),
        ("H", (2.73, 0.94, 0.0)),
        ("H", (2.73, -0.47, 0.82)),
    ]
    G = build_graph(atoms, charge=0)
    assert G.edges[1, 3]["NCI"]
    assert G.edges[1, 3]["bond_order"] == 0.0
    assert not any(G.nodes[n]["formal_charge"] for n in G)
    assert not build_graph(atoms, charge=0, relaxed=True).edges[1, 3].get("NCI")


def test_agostic_hydrogen_bonds_only_in_a_transition_state():
    """A C-H pointing at a metal is agostic at equilibrium; a transition state keeps the M-H, an H in flight."""
    atoms = [("Pd", (0.0, 0.0, 0.0)), ("Cl", (-2.3, 0.0, 0.0)), ("H", (1.9, 0.0, 0.0)), ("C", (3.02, 0.0, 0.0))]
    atoms += [("H", (3.38, 1.03, 0.0)), ("H", (3.38, -0.51, 0.89)), ("H", (3.38, -0.51, -0.89))]
    assert not build_graph(atoms, charge=0).has_edge(0, 2)
    assert build_graph(atoms, charge=0, relaxed=True).has_edge(0, 2)


def test_cation_takes_the_lone_pair_across_the_ring():
    """4-Aminobenzyl cation: the charge sits on the iminium N+, four bonds from the CH2, not on a carbocation."""
    atoms = [
        ("N", (-2.825, 0.132, -0.117)),
        ("C", (-1.433, -0.004, -0.071)),
        ("C", (-0.664, 0.228, -1.213)),
        ("C", (0.734, 0.21, -1.15)),
        ("C", (1.383, 0.001, 0.068)),
        ("C", (2.836, -0.017, 0.136)),
        ("C", (0.618, -0.172, 1.223)),
        ("C", (-0.779, -0.153, 1.154)),
        ("H", (-3.226, -0.078, -1.026)),
        ("H", (-3.307, -0.346, 0.637)),
        ("H", (-1.145, 0.426, -2.166)),
        ("H", (1.305, 0.376, -2.061)),
        ("H", (3.335, -0.166, 1.086)),
        ("H", (3.425, 0.132, -0.76)),
        ("H", (1.096, -0.31, 2.189)),
        ("H", (-1.353, -0.257, 2.071)),
    ]
    G = build_graph(atoms, charge=1)
    assert [n for n in G if G.nodes[n]["formal_charge"]] == [0]
    assert G.edges[0, 1]["bond_order"] == 2.0


def test_charge_inferred_when_not_given():
    """charge=None reads a metal-free molecule's closed-shell total; a complex assumes 0."""
    G = build_graph(str(EXAMPLES / "isothio.xyz"))
    assert (G.graph["total_charge"], G.graph["multiplicity"]) == (1, 1)
    assert build_graph(str(EXAMPLES / "mnh.xyz")).graph["total_charge"] == 0


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
    # Cp ring carbons are symmetry-equivalent — formal charge can land on
    # any one of the 5 carbons in each ring, so we compare per-ring totals
    # rather than per-atom values for those atoms.
    CP_RINGS = [{7, 8, 9, 11, 13}, {15, 17, 19, 21, 23}]  # 0-indexed
    cp_atoms = CP_RINGS[0] | CP_RINGS[1]

    assert len(result["nodes"]) == len(expected["nodes"])
    for i, (got, exp) in enumerate(zip(result["nodes"], expected["nodes"])):
        assert got["symbol"] == exp["symbol"]
        if i not in cp_atoms:
            assert got["formal_charge"] == exp["formal_charge"]
        assert got["valence"] == pytest.approx(exp["valence"])
        assert got["metal_valence"] == pytest.approx(exp["metal_valence"])

    # Cp ring formal charge totals must match (any permutation within ring is OK)
    for ring in CP_RINGS:
        got_sum = sum(result["nodes"][i]["formal_charge"] for i in ring)
        exp_sum = sum(expected["nodes"][i]["formal_charge"] for i in ring)
        assert got_sum == exp_sum, f"Cp ring {ring}: fc sum {got_sum} != {exp_sum}"

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


# ===========================================================================
# Organic: caffeine (C8H10N4O2, charge 0)
#
# The imidazole ring is aromatic; the pyrimidine-2,6-dione ring is not. The
# dione's two carbonyl carbons each donate no p-electron to the ring, so the
# Hückel count reaches an aromatic total; the cross-conjugation guard (a ring
# admits at most one such carbon) is what keeps the ring non-aromatic.
# ===========================================================================


def test_caffeine_dione_ring_not_aromatic():
    """Caffeine: imidazole aromatic, pyrimidinedione ring stays Kekulé."""
    G = build_graph(str(EXAMPLES / "caffeine.xyz"), charge=0)
    result = graph_to_dict(G)

    assert result["graph"]["formula"] == "C8H10N4O2"

    # Only the five-membered imidazole ring is aromatic.
    arom = [sorted(r) for r in result["graph"]["aromatic_rings"]]
    assert arom == [[0, 2, 4, 9, 10]], f"aromatic rings: {arom}"
    assert sum(1 for e in result["edges"] if e["bond_order"] == pytest.approx(1.5)) == 5

    # Each carbonyl keeps a localised C=O double bond, not a delocalised 1.5.
    for cid, oid in ((5, 12), (6, 13)):
        edge = next(e for e in result["edges"] if {e["idx1"], e["idx2"]} == {cid, oid})
        assert edge["bond_order"] == pytest.approx(2.0)
        assert result["nodes"][oid]["formal_charge"] == 0
