"""Tests for xyzgraph protein semantics extraction."""

from __future__ import annotations

import math

import networkx as nx

from xyzgraph.protein import (
    ProteinConfidenceTier,
    annotate_protein_semantics,
    protein_semantics_from_dict,
)


def _add_node(g: nx.Graph, idx: int, sym: str, x: float) -> None:
    g.add_node(idx, symbol=sym, position=(x, 0.0, 0.0))


def _build_helical_backbone(n_residues: int = 12) -> tuple[nx.Graph, list[dict[str, object]]]:
    g = nx.Graph()
    annotations: list[dict[str, object]] = []
    serial = 0
    prev_c: int | None = None
    for i in range(n_residues):
        seq = i + 1
        theta = math.radians(100.0 * i)
        ca = (2.3 * math.cos(theta), 2.3 * math.sin(theta), 1.5 * i)
        n_pos = (ca[0] - 1.1, ca[1], ca[2])
        c_pos = (ca[0] + 1.1, ca[1], ca[2])
        o_pos = (ca[0] + 1.8, ca[1] + 0.7, ca[2])

        n_idx = serial
        g.add_node(n_idx, symbol="N", position=n_pos)
        annotations.append(
            {"record_type": "ATOM", "atom_name": "N", "res_name": "ALA", "res_seq": seq, "chain_id": "A", "ss_type": "C"}
        )
        serial += 1

        ca_idx = serial
        g.add_node(ca_idx, symbol="C", position=ca)
        annotations.append(
            {
                "record_type": "ATOM",
                "atom_name": "CA",
                "res_name": "ALA",
                "res_seq": seq,
                "chain_id": "A",
                "ss_type": "C",
            }
        )
        serial += 1

        c_idx = serial
        g.add_node(c_idx, symbol="C", position=c_pos)
        annotations.append(
            {"record_type": "ATOM", "atom_name": "C", "res_name": "ALA", "res_seq": seq, "chain_id": "A", "ss_type": "C"}
        )
        serial += 1

        o_idx = serial
        g.add_node(o_idx, symbol="O", position=o_pos)
        annotations.append(
            {"record_type": "ATOM", "atom_name": "O", "res_name": "ALA", "res_seq": seq, "chain_id": "A", "ss_type": "C"}
        )
        serial += 1

        g.add_edges_from([(n_idx, ca_idx), (ca_idx, c_idx), (c_idx, o_idx)])
        if prev_c is not None:
            g.add_edge(prev_c, n_idx)
        prev_c = c_idx

    return g, annotations


def test_annotation_extraction_full_ribbon_and_ligand_partition():
    g = nx.Graph()
    # Residue 1: N-CA-C-O
    _add_node(g, 0, "N", 0.0)
    _add_node(g, 1, "C", 1.0)
    _add_node(g, 2, "C", 2.0)
    _add_node(g, 3, "O", 3.0)
    # Residue 2: N-CA-C-O
    _add_node(g, 4, "N", 4.0)
    _add_node(g, 5, "C", 5.0)
    _add_node(g, 6, "C", 6.0)
    _add_node(g, 7, "O", 7.0)
    # Ligand atoms
    _add_node(g, 8, "C", 9.0)
    _add_node(g, 9, "O", 10.0)

    g.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (8, 9),
        ]
    )

    annotations = [
        {"record_type": "ATOM", "atom_name": "N", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "CA", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "C", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "O", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "N", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "CA", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "C", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "H"},
        {"record_type": "ATOM", "atom_name": "O", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "H"},
        {
            "record_type": "HETATM",
            "atom_name": "C1",
            "res_name": "LIG",
            "res_seq": 101,
            "chain_id": "A",
            "ss_type": "C",
        },
        {
            "record_type": "HETATM",
            "atom_name": "O1",
            "res_name": "LIG",
            "res_seq": 101,
            "chain_id": "A",
            "ss_type": "C",
        },
    ]

    report = annotate_protein_semantics(g, atom_annotations=annotations, format_hint=".pdb")
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.FULL_RIBBON
    assert "A" in report.semantics.chains
    assert len(report.semantics.chains["A"].residues) == 2
    assert report.semantics.ligand_indices == {8, 9}

    payload = g.graph["protein_semantics"]
    sem2 = protein_semantics_from_dict(payload)
    assert sem2.confidence_tier == ProteinConfidenceTier.FULL_RIBBON
    protein_atoms = {i for ch in sem2.chains.values() for r in ch.residues for i in r.atom_indices}
    assert 8 not in protein_atoms
    assert 9 not in protein_atoms


def test_heuristic_fallback_largest_component_trace_only():
    g = nx.Graph()
    # Largest component, peptide-like chain with 3 residues.
    coords = [
        ("N", 0.0),
        ("C", 1.0),
        ("C", 2.0),
        ("O", 3.0),
        ("N", 4.0),
        ("C", 5.0),
        ("C", 6.0),
        ("O", 7.0),
        ("N", 8.0),
        ("C", 9.0),
        ("C", 10.0),
        ("O", 11.0),
    ]
    for i, (sym, x) in enumerate(coords):
        _add_node(g, i, sym, x)
    g.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (8, 9),
            (9, 10),
            (10, 11),
        ]
    )

    # Separate ligand component.
    _add_node(g, 12, "C", 20.0)
    _add_node(g, 13, "O", 21.0)
    _add_node(g, 14, "N", 22.0)
    g.add_edges_from([(12, 13), (13, 14)])

    report = annotate_protein_semantics(g, atom_annotations=None, protein_requested=True)
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.TRACE_ONLY
    assert report.semantics.ligand_indices == {12, 13, 14}
    assert "A" in report.semantics.trace_chains
    assert len(report.semantics.trace_chains["A"]) >= 3


def test_heuristic_requires_peptide_links_not_just_ca_like_motifs():
    g = nx.Graph()

    # Build 6 residue-like local motifs without peptide C(=O)-N links:
    # each "CA" carbon has N and carbonyl neighbors, but CA-CA is linked directly.
    idx = 0
    ca_nodes: list[int] = []
    for _ in range(6):
        n_atom = idx
        ca_atom = idx + 1
        c_atom = idx + 2
        o_atom = idx + 3
        _add_node(g, n_atom, "N", float(n_atom))
        _add_node(g, ca_atom, "C", float(ca_atom))
        _add_node(g, c_atom, "C", float(c_atom))
        _add_node(g, o_atom, "O", float(o_atom))
        g.add_edges_from([(n_atom, ca_atom), (ca_atom, c_atom), (c_atom, o_atom)])
        ca_nodes.append(ca_atom)
        idx += 4

    # Connect motifs as a single large component without carbonyl->amide links.
    for a, b in zip(ca_nodes[:-1], ca_nodes[1:], strict=False):
        g.add_edge(a, b)

    report = annotate_protein_semantics(g, protein_requested=True)
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.INSUFFICIENT
    assert report.semantics.chains == {}


def test_no_metadata_and_not_requested_returns_none():
    g = nx.Graph()
    _add_node(g, 0, "C", 0.0)
    _add_node(g, 1, "O", 1.0)
    g.add_edge(0, 1)

    report = annotate_protein_semantics(g, atom_annotations=None, protein_requested=False)
    assert report is None
    assert "protein_semantics" not in g.graph


def test_geometry_inference_supplements_unlabeled_secondary_structure():
    g, annotations = _build_helical_backbone(n_residues=12)
    report = annotate_protein_semantics(g, atom_annotations=annotations, format_hint=".pdb")
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.FULL_RIBBON
    chain = report.semantics.chains["A"]
    labels = [res.ss_type for res in chain.residues]
    assert any(ss in {"H", "E"} for ss in labels)
    assert any("inferred from geometry" in msg for msg in report.confidence_reasons)
    assert "inferred" in report.semantics.provenance


def test_geometry_inference_preserves_explicit_secondary_structure_labels():
    g, annotations = _build_helical_backbone(n_residues=12)
    for row in annotations:
        if row["res_seq"] <= 4:
            row["ss_type"] = "H"

    report = annotate_protein_semantics(g, atom_annotations=annotations, format_hint=".pdb")
    assert report is not None
    chain = report.semantics.chains["A"]
    by_seq = {res.res_seq: res.ss_type for res in chain.residues}
    for seq in range(1, 5):
        assert by_seq[seq] == "H"
    assert any("supplemented by geometry inference" in msg for msg in report.confidence_reasons)


def test_requested_with_weak_signal_returns_insufficient():
    g = nx.Graph()
    for i, sym in enumerate(["C", "N", "O", "H", "C"]):
        _add_node(g, i, sym, float(i))
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    report = annotate_protein_semantics(g, protein_requested=True)
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.INSUFFICIENT
    assert report.semantics.chains == {}


def test_annotation_alias_keys_are_normalized():
    g = nx.Graph()
    for i, (sym, x) in enumerate(
        [
            ("N", 0.0),
            ("C", 1.0),
            ("C", 2.0),
            ("O", 3.0),
            ("N", 4.0),
            ("C", 5.0),
            ("C", 6.0),
            ("O", 7.0),
        ]
    ):
        _add_node(g, i, sym, x)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7)])

    annotations = [
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "N",
            "auth_comp_id": "ALA",
            "auth_seq_id": 1,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "CA",
            "auth_comp_id": "ALA",
            "auth_seq_id": 1,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "C",
            "auth_comp_id": "ALA",
            "auth_seq_id": 1,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "O",
            "auth_comp_id": "ALA",
            "auth_seq_id": 1,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "N",
            "auth_comp_id": "GLY",
            "auth_seq_id": 2,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "CA",
            "auth_comp_id": "GLY",
            "auth_seq_id": 2,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "C",
            "auth_comp_id": "GLY",
            "auth_seq_id": 2,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
        {
            "group_pdb": "ATOM",
            "auth_atom_id": "O",
            "auth_comp_id": "GLY",
            "auth_seq_id": 2,
            "auth_asym_id": "A",
            "secondary_structure": "H",
        },
    ]

    report = annotate_protein_semantics(g, atom_annotations=annotations, format_hint=".cif")
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.FULL_RIBBON
    assert "A" in report.semantics.chains
    assert len(report.semantics.chains["A"].residues) == 2


def test_atom_coded_waters_and_ions_not_forced_to_ligand():
    g = nx.Graph()
    for i, (sym, x) in enumerate(
        [
            ("N", 0.0),
            ("C", 1.0),
            ("C", 2.0),
            ("O", 3.0),
            ("N", 4.0),
            ("C", 5.0),
            ("C", 6.0),
            ("O", 7.0),
            ("O", 12.0),
            ("H", 13.0),
            ("H", 14.0),
            ("NA", 16.0),
        ]
    ):
        _add_node(g, i, sym, x)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (8, 9), (8, 10)])

    annotations = [
        {"record_type": "ATOM", "atom_name": "N", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "CA", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "C", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "O", "res_name": "ALA", "res_seq": 1, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "N", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "CA", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "C", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "O", "res_name": "GLY", "res_seq": 2, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "O", "res_name": "HOH", "res_seq": 3, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "H1", "res_name": "HOH", "res_seq": 3, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "H2", "res_name": "HOH", "res_seq": 3, "chain_id": "A", "ss_type": "C"},
        {"record_type": "ATOM", "atom_name": "NA", "res_name": "NA", "res_seq": 4, "chain_id": "A", "ss_type": "C"},
    ]

    report = annotate_protein_semantics(g, atom_annotations=annotations, format_hint=".cif")
    assert report is not None
    assert report.confidence_tier == ProteinConfidenceTier.FULL_RIBBON
    assert report.semantics.water_indices == {8, 9, 10}
    assert report.semantics.ion_indices == {11}
    assert report.semantics.ligand_indices == set()
