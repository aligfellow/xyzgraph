"""Protein semantics extraction for downstream rendering workflows.

This module provides a graph-level, format-agnostic protein semantics layer
that can be consumed by renderers (e.g. xyzrender) without hard-coding
format-specific logic into drawing code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable

import networkx as nx
import numpy as np

_ANNOTATION_ALIASES: dict[str, tuple[str, ...]] = {
    "record_type": ("record_type", "group_pdb", "pdb_record_type", "_atom_site.group_pdb"),
    "atom_name": ("atom_name", "auth_atom_id", "label_atom_id", "atomtypes"),
    "res_name": ("res_name", "auth_comp_id", "label_comp_id", "residuenames"),
    "res_seq": ("res_seq", "auth_seq_id", "label_seq_id", "residuenumbers"),
    "chain_id": ("chain_id", "auth_asym_id", "label_asym_id", "chainids"),
    "ss_type": ("ss_type", "secondary_structure"),
}

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})
_ION_RESNAMES: frozenset[str] = frozenset({"NA", "K", "CA", "MG", "ZN", "CL", "FE", "CU", "MN", "CO", "NI", "SO4", "PO4"})
_SS_INFERENCE = {
    "helix_torsion_min": 20.0,
    "helix_torsion_max": 105.0,
    "helix_d3_min": 4.7,
    "helix_d3_max": 6.8,
    "helix_d2_min": 5.0,
    "helix_d2_max": 6.4,
    "sheet_abs_torsion_min": 130.0,
    "sheet_d3_min": 8.0,
    "sheet_d2_min": 6.1,
    "bridge_gap": 1,
    "min_helix_run": 4,
    "min_sheet_run": 3,
}


class ProteinConfidenceTier(Enum):
    """Confidence tiers for protein semantics extraction."""

    FULL_RIBBON = "full_ribbon"
    TRACE_ONLY = "trace_only"
    INSUFFICIENT = "insufficient"


@dataclass
class ProteinResidueSemantics:
    """Normalized residue-level protein semantics."""

    res_name: str
    res_seq: int
    chain_id: str
    atom_indices: list[int]
    ca_index: int | None
    c_index: int | None
    o_index: int | None
    n_index: int | None
    ss_type: str = "C"


@dataclass
class ProteinChainSemantics:
    """Ordered residue semantics for a chain."""

    chain_id: str
    residues: list[ProteinResidueSemantics]


@dataclass
class ProteinSemantics:
    """Format-agnostic protein semantics used by downstream renderers."""

    chains: dict[str, ProteinChainSemantics]
    hetatm_indices: set[int]
    backbone_indices: set[int]
    sidechain_indices: set[int]
    helix_spans: list[tuple[str, int, int]]
    sheet_spans: list[tuple[str, int, int]]
    ligand_indices: set[int] = field(default_factory=set)
    water_indices: set[int] = field(default_factory=set)
    ion_indices: set[int] = field(default_factory=set)
    confidence_tier: ProteinConfidenceTier = ProteinConfidenceTier.FULL_RIBBON
    confidence_reasons: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    trace_chains: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ProteinExtractionReport:
    """Structured result from protein semantics extraction."""

    semantics: ProteinSemantics
    confidence_tier: ProteinConfidenceTier
    confidence_reasons: list[str]


def _normalize_ss(value: object) -> str:
    ss = str(value or "C").strip().upper()
    return ss if ss in {"H", "E", "C"} else "C"


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_annotation_value(row: dict[str, object], key: str, default: object) -> object:
    aliases = _ANNOTATION_ALIASES.get(key, (key,))
    for alias in aliases:
        if alias not in row:
            continue
        value = row.get(alias)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() in {"", ".", "?"}:
            continue
        return value
    return default


def _normalize_annotations(raw: Iterable[dict[str, object]] | None, n_atoms: int) -> list[dict[str, object]] | None:
    if raw is None:
        return None
    rows = list(raw)
    if len(rows) != n_atoms:
        return None

    out: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        row = row or {}
        rec = str(_pick_annotation_value(row, "record_type", "ATOM")).strip().upper() or "ATOM"
        atom_name = str(_pick_annotation_value(row, "atom_name", "")).strip()
        res_name = str(_pick_annotation_value(row, "res_name", "UNK")).strip().upper() or "UNK"
        chain_id = str(_pick_annotation_value(row, "chain_id", "A")).strip() or "A"
        res_seq = _to_int(_pick_annotation_value(row, "res_seq", idx + 1), default=idx + 1)
        ss_type = _normalize_ss(_pick_annotation_value(row, "ss_type", "C"))

        out.append(
            {
                "record_type": "HETATM" if rec == "HETATM" else "ATOM",
                "atom_name": atom_name,
                "res_name": res_name,
                "chain_id": chain_id,
                "res_seq": res_seq,
                "ss_type": ss_type,
            }
        )

    return out


def _derive_ss_spans(chains: dict[str, ProteinChainSemantics], ss_type: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for cid in sorted(chains):
        residues = chains[cid].residues
        if not residues:
            continue
        start = end = None
        prev_seq = None
        for res in residues:
            if res.ss_type == ss_type:
                if start is None:
                    start = end = res.res_seq
                elif prev_seq is not None and res.res_seq == prev_seq + 1:
                    end = res.res_seq
                else:
                    spans.append((cid, int(start), int(end)))
                    start = end = res.res_seq
            else:
                if start is not None:
                    spans.append((cid, int(start), int(end)))
                    start = end = None
            prev_seq = res.res_seq
        if start is not None:
            spans.append((cid, int(start), int(end)))
    return spans


def _node_position(graph: nx.Graph, node_id: int) -> tuple[float, float, float] | None:
    node = graph.nodes[node_id]
    pos = node.get("position", node.get("pos"))
    if pos is None:
        return None
    try:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    except Exception:
        return None
    return (x, y, z)


def _anchor_position(graph: nx.Graph, residue: ProteinResidueSemantics) -> tuple[float, float, float] | None:
    for idx in (residue.ca_index, residue.n_index, residue.c_index):
        if idx is None:
            continue
        pos = _node_position(graph, idx)
        if pos is not None:
            return pos
    return None


def _dihedral_degrees(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    a = np.asarray(p0, dtype=float)
    b = np.asarray(p1, dtype=float)
    c = np.asarray(p2, dtype=float)
    d = np.asarray(p3, dtype=float)
    b0 = a - b
    b1 = c - b
    b2 = d - c
    n = np.linalg.norm(b1)
    if n <= 1.0e-12:
        return 0.0
    b1n = b1 / n
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1n, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _bridge_gaps(labels: list[str], symbol: str, max_gap: int, locked: list[bool]) -> None:
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != symbol:
            i += 1
            continue
        j = i
        while j < n and labels[j] == symbol:
            j += 1
        k = j
        while k < n and labels[k] == "C" and not locked[k]:
            k += 1
        gap = k - j
        if 0 < gap <= max_gap and k < n and labels[k] == symbol:
            for u in range(j, k):
                labels[u] = symbol
        i = k


def _trim_short_runs(labels: list[str], symbol: str, min_len: int, locked: list[bool]) -> None:
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != symbol:
            i += 1
            continue
        j = i
        has_locked = False
        while j < n and labels[j] == symbol:
            has_locked = has_locked or locked[j]
            j += 1
        if not has_locked and (j - i) < min_len:
            for u in range(i, j):
                labels[u] = "C"
        i = j


def _infer_missing_ss_in_chain(graph: nx.Graph, residues: list[ProteinResidueSemantics]) -> list[str]:
    n = len(residues)
    if n < 4:
        return ["C"] * n

    anchors = [_anchor_position(graph, r) for r in residues]
    helix_votes = [0] * n
    sheet_votes = [0] * n

    for i in range(1, n - 2):
        win = residues[i - 1 : i + 3]
        if any(anchors[idx] is None for idx in (i - 1, i, i + 1, i + 2)):
            continue
        if not (
            win[1].res_seq == win[0].res_seq + 1
            and win[2].res_seq == win[1].res_seq + 1
            and win[3].res_seq == win[2].res_seq + 1
        ):
            continue
        p0 = anchors[i - 1]
        p1 = anchors[i]
        p2 = anchors[i + 1]
        p3 = anchors[i + 2]
        assert p0 is not None and p1 is not None and p2 is not None and p3 is not None

        torsion = _dihedral_degrees(p0, p1, p2, p3)
        abs_torsion = abs(torsion)
        d3 = _distance(p0, p3)
        d2 = _distance(p0, p2)

        helix_like = (
            _SS_INFERENCE["helix_torsion_min"] <= torsion <= _SS_INFERENCE["helix_torsion_max"]
            and _SS_INFERENCE["helix_d3_min"] <= d3 <= _SS_INFERENCE["helix_d3_max"]
            and _SS_INFERENCE["helix_d2_min"] <= d2 <= _SS_INFERENCE["helix_d2_max"]
        )
        sheet_like = (
            abs_torsion >= _SS_INFERENCE["sheet_abs_torsion_min"]
            and d3 >= _SS_INFERENCE["sheet_d3_min"]
            and d2 >= _SS_INFERENCE["sheet_d2_min"]
        )

        if helix_like and not sheet_like:
            helix_votes[i] += 1
            helix_votes[i + 1] += 1
        elif sheet_like and not helix_like:
            sheet_votes[i] += 1
            sheet_votes[i + 1] += 1

    inferred = ["C"] * n
    for i in range(n):
        h = helix_votes[i]
        e = sheet_votes[i]
        if h == 0 and e == 0:
            continue
        inferred[i] = "H" if h >= e else "E"
    return inferred


def _supplement_secondary_structure(graph: nx.Graph, semantics: ProteinSemantics) -> int:
    inferred_residues = 0
    explicit_has_ss = any(res.ss_type in {"H", "E"} for chain in semantics.chains.values() for res in chain.residues)

    for chain in semantics.chains.values():
        residues = chain.residues
        if not residues:
            continue
        locked = [r.ss_type in {"H", "E"} for r in residues]
        if all(locked):
            continue
        inferred = _infer_missing_ss_in_chain(graph, residues)
        labels = [r.ss_type if r.ss_type in {"H", "E"} else inferred[i] for i, r in enumerate(residues)]

        _bridge_gaps(labels, "H", int(_SS_INFERENCE["bridge_gap"]), locked)
        _bridge_gaps(labels, "E", int(_SS_INFERENCE["bridge_gap"]), locked)
        _trim_short_runs(labels, "H", int(_SS_INFERENCE["min_helix_run"]), locked)
        _trim_short_runs(labels, "E", int(_SS_INFERENCE["min_sheet_run"]), locked)

        for i, res in enumerate(residues):
            if res.ss_type != "C":
                continue
            if labels[i] in {"H", "E"}:
                res.ss_type = labels[i]
                inferred_residues += 1

    semantics.helix_spans = _derive_ss_spans(semantics.chains, "H")
    semantics.sheet_spans = _derive_ss_spans(semantics.chains, "E")
    inferred_any = inferred_residues > 0
    if inferred_any:
        if "inferred" not in semantics.provenance:
            semantics.provenance.append("inferred")
        if explicit_has_ss:
            msg = "secondary structure supplemented by geometry inference"
        else:
            msg = "secondary structure inferred from geometry"
        if msg not in semantics.confidence_reasons:
            semantics.confidence_reasons.append(msg)
    return inferred_residues


def _is_peptide_residue(atom_names: set[str], res_name: str) -> bool:
    if res_name in _WATER_RESNAMES or res_name in _ION_RESNAMES:
        return False
    # Conservative: require canonical peptide backbone naming to avoid ligand leakage.
    return {"N", "CA", "C"}.issubset(atom_names)


def _extract_from_annotations(
    graph: nx.Graph,
    annotations: list[dict[str, object]],
    *,
    provenance: str,
) -> ProteinSemantics | None:
    n_atoms = graph.number_of_nodes()
    if n_atoms == 0:
        return None

    residues: dict[tuple[str, int, str], list[int]] = {}
    residue_order: list[tuple[str, int, str]] = []
    residue_names: dict[tuple[str, int, str], set[str]] = {}
    residue_ss: dict[tuple[str, int, str], str] = {}

    hetatm_indices: set[int] = set()
    ligand_indices: set[int] = set()
    water_indices: set[int] = set()
    ion_indices: set[int] = set()

    for idx, row in enumerate(annotations):
        rec = str(row["record_type"])
        atom_name = str(row["atom_name"]).upper().strip()
        res_name = str(row["res_name"]).upper().strip() or "UNK"
        chain_id = str(row["chain_id"]).strip() or "A"
        res_seq = _to_int(row["res_seq"], default=idx + 1)
        ss_type = _normalize_ss(row["ss_type"])

        if rec == "HETATM":
            hetatm_indices.add(idx)
            if res_name in _WATER_RESNAMES:
                water_indices.add(idx)
            elif res_name in _ION_RESNAMES:
                ion_indices.add(idx)
            else:
                ligand_indices.add(idx)
            continue

        key = (chain_id, int(res_seq), res_name)
        if key not in residues:
            residues[key] = []
            residue_order.append(key)
            residue_names[key] = set()
            residue_ss[key] = ss_type
        residues[key].append(idx)
        if atom_name:
            residue_names[key].add(atom_name)
        # Keep first provided SS label for deterministic behavior.

    if not residues:
        return None

    chains_raw: dict[str, list[ProteinResidueSemantics]] = {}
    backbone_indices: set[int] = set()
    protein_atoms: set[int] = set()

    for chain_id, res_seq, res_name in residue_order:
        idxs = residues[(chain_id, res_seq, res_name)]
        names = residue_names[(chain_id, res_seq, res_name)]

        ca_index = c_index = o_index = n_index = None
        for idx in idxs:
            atom_name = str(annotations[idx]["atom_name"]).upper().strip()
            if atom_name == "CA":
                ca_index = idx
                backbone_indices.add(idx)
            elif atom_name == "C":
                c_index = idx
                backbone_indices.add(idx)
            elif atom_name == "N":
                n_index = idx
                backbone_indices.add(idx)
            elif atom_name in {"O", "OXT"}:
                if atom_name == "O" and o_index is None:
                    o_index = idx
                backbone_indices.add(idx)

        has_trace_anchor = ca_index is not None or (n_index is not None and c_index is not None)
        if not has_trace_anchor or not _is_peptide_residue(names, res_name):
            if res_name in _WATER_RESNAMES:
                water_indices.update(idxs)
            elif res_name in _ION_RESNAMES:
                ion_indices.update(idxs)
            else:
                ligand_indices.update(idxs)
            continue

        residue = ProteinResidueSemantics(
            res_name=res_name,
            res_seq=int(res_seq),
            chain_id=chain_id,
            atom_indices=list(idxs),
            ca_index=ca_index,
            c_index=c_index,
            o_index=o_index,
            n_index=n_index,
            ss_type=residue_ss[(chain_id, res_seq, res_name)],
        )
        chains_raw.setdefault(chain_id, []).append(residue)
        protein_atoms.update(idxs)

    # Require chain continuity signal: at least 2 protein-like residues per chain.
    chains: dict[str, ProteinChainSemantics] = {}
    for cid, residues_list in chains_raw.items():
        ordered = sorted(residues_list, key=lambda r: (r.res_seq, r.atom_indices[0] if r.atom_indices else -1))
        if len(ordered) < 2:
            continue
        chains[cid] = ProteinChainSemantics(chain_id=cid, residues=ordered)

    if not chains:
        return None

    protein_atoms = {idx for chain in chains.values() for res in chain.residues for idx in res.atom_indices}
    sidechain_indices = protein_atoms - backbone_indices

    # In metadata-driven inputs without HETATM annotations (e.g. MOL2), treat
    # non-protein residues as ligand only when a protein partition exists.
    for idx in range(n_atoms):
        if idx in protein_atoms or idx in hetatm_indices or idx in water_indices or idx in ion_indices:
            continue
        ligand_indices.add(idx)

    trace_chains: dict[str, list[int]] = {}
    for cid, chain in chains.items():
        trace: list[int] = []
        for res in chain.residues:
            ai = res.ca_index if res.ca_index is not None else (res.n_index if res.n_index is not None else res.c_index)
            if ai is not None:
                trace.append(ai)
        if len(trace) >= 2:
            trace_chains[cid] = trace

    semantics = ProteinSemantics(
        chains=chains,
        hetatm_indices=hetatm_indices,
        backbone_indices=backbone_indices,
        sidechain_indices=sidechain_indices,
        helix_spans=_derive_ss_spans(chains, "H"),
        sheet_spans=_derive_ss_spans(chains, "E"),
        ligand_indices=ligand_indices,
        water_indices=water_indices,
        ion_indices=ion_indices,
        confidence_tier=ProteinConfidenceTier.FULL_RIBBON,
        confidence_reasons=[f"{provenance} annotations parsed"],
        provenance=[provenance],
        trace_chains=trace_chains,
    )
    has_ca = any(res.ca_index is not None for chain in chains.values() for res in chain.residues)
    inferred_count = _supplement_secondary_structure(graph, semantics)
    has_ss = any(res.ss_type in {"H", "E"} for chain in chains.values() for res in chain.residues)

    if not has_ca:
        semantics.confidence_tier = ProteinConfidenceTier.TRACE_ONLY
        semantics.confidence_reasons.append("CA atoms missing; downgraded to TRACE_ONLY")
    elif not has_ss and inferred_count == 0:
        semantics.confidence_reasons.append("no explicit helix/sheet labels; loops-only ribbon")

    return semantics


def _extract_heuristic(graph: nx.Graph) -> ProteinSemantics | None:
    """Topology-first fallback used only when explicit protein intent is set."""
    node_ids = sorted(graph.nodes())
    if not node_ids:
        return None

    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
    if not components:
        return None

    largest = set(components[0])
    if len(largest) < 12:
        return None

    symbols = {n: str(graph.nodes[n].get("symbol", "")).upper() for n in largest}
    peptide_like = [n for n in largest if symbols.get(n, "") in {"C", "N", "O", "S"}]
    if len(peptide_like) < 12:
        return None

    carbonyl_carbons: set[int] = set()
    for nid in peptide_like:
        if symbols.get(nid) != "C":
            continue
        if any(symbols.get(nb, "") == "O" for nb in graph.neighbors(nid) if nb in largest):
            carbonyl_carbons.add(int(nid))

    # Candidate CA-like sites with local peptide-like chemistry:
    # - bonded to at least one N
    # - bonded to a carbonyl carbon (C bonded to O)
    candidates: list[tuple[int, int, int]] = []  # (ca_atom, amide_n, carbonyl_c)
    for nid in sorted(peptide_like):
        if symbols.get(nid) != "C":
            continue
        n_neigh = sorted(int(nb) for nb in graph.neighbors(nid) if nb in largest and symbols.get(nb, "") == "N")
        c_neigh = sorted(
            int(nb)
            for nb in graph.neighbors(nid)
            if nb in largest and int(nb) in carbonyl_carbons and int(nb) != int(nid)
        )
        if not n_neigh or not c_neigh:
            continue
        candidates.append((int(nid), n_neigh[0], c_neigh[0]))

    if len(candidates) < 3:
        return None

    # Build peptide-link graph between candidate residues:
    # residue i -> residue j when carbonyl(i) bonds to amide-N(j).
    n_to_candidates: dict[int, list[int]] = {}
    for rid, (_, n_atom, _) in enumerate(candidates):
        n_to_candidates.setdefault(n_atom, []).append(rid)

    succ: dict[int, set[int]] = {rid: set() for rid in range(len(candidates))}
    pred_count: dict[int, int] = {rid: 0 for rid in range(len(candidates))}
    for rid, (_, _, carbonyl_c) in enumerate(candidates):
        for nb in graph.neighbors(carbonyl_c):
            for nxt in n_to_candidates.get(int(nb), []):
                if nxt == rid or nxt in succ[rid]:
                    continue
                succ[rid].add(nxt)
                pred_count[nxt] += 1

    def _walk(start: int, seen: set[int]) -> list[int]:
        seg: list[int] = []
        cur = start
        while True:
            if cur in seen:
                break
            seen.add(cur)
            seg.append(candidates[cur][0])
            next_nodes = [nxt for nxt in sorted(succ[cur]) if nxt not in seen]
            if not next_nodes:
                break
            cur = next_nodes[0]
        return seg

    seen_candidates: set[int] = set()
    segments: list[list[int]] = []

    starts = [rid for rid in range(len(candidates)) if pred_count[rid] == 0 and succ[rid]]
    for rid in sorted(starts):
        seg = _walk(rid, seen_candidates)
        if len(seg) >= 3:
            segments.append(seg)

    remainder = [rid for rid in range(len(candidates)) if rid not in seen_candidates and (succ[rid] or pred_count[rid] > 0)]
    for rid in sorted(remainder):
        seg = _walk(rid, seen_candidates)
        if len(seg) >= 3:
            segments.append(seg)

    if not segments:
        return None

    segments.sort(key=lambda seg: (seg[0], len(seg)))
    chain_ids: list[str] = [chr(ord("A") + i) if i < 26 else f"C{i + 1}" for i in range(len(segments))]

    chains: dict[str, ProteinChainSemantics] = {}
    trace_chains: dict[str, list[int]] = {}
    backbone_indices: set[int] = set()
    for cid, trace in zip(chain_ids, segments, strict=False):
        residues = [
            ProteinResidueSemantics(
                res_name="UNK",
                res_seq=i,
                chain_id=cid,
                atom_indices=[idx],
                ca_index=idx,
                c_index=None,
                o_index=None,
                n_index=None,
                ss_type="C",
            )
            for i, idx in enumerate(trace, start=1)
        ]
        chains[cid] = ProteinChainSemantics(chain_id=cid, residues=residues)
        trace_chains[cid] = list(trace)
        backbone_indices.update(trace)

    ligand_indices: set[int] = set()
    for comp in components[1:]:
        ligand_indices.update(int(n) for n in comp)

    return ProteinSemantics(
        chains=chains,
        hetatm_indices=set(),
        backbone_indices=backbone_indices,
        sidechain_indices={int(n) for n in peptide_like if int(n) not in backbone_indices},
        helix_spans=[],
        sheet_spans=[],
        ligand_indices=ligand_indices,
        water_indices=set(),
        ion_indices=set(),
        confidence_tier=ProteinConfidenceTier.TRACE_ONLY,
        confidence_reasons=["graph-only heuristic trace from peptide-link topology"],
        provenance=["heuristic"],
        trace_chains=trace_chains,
    )


def protein_semantics_to_dict(semantics: ProteinSemantics) -> dict[str, object]:
    """Serialize :class:`ProteinSemantics` to a graph-storable dictionary."""
    chain_payload: dict[str, dict[str, object]] = {}
    for cid, chain in semantics.chains.items():
        chain_payload[cid] = {
            "chain_id": chain.chain_id,
            "residues": [asdict(res) for res in chain.residues],
        }
    return {
        "chains": chain_payload,
        "hetatm_indices": sorted(semantics.hetatm_indices),
        "backbone_indices": sorted(semantics.backbone_indices),
        "sidechain_indices": sorted(semantics.sidechain_indices),
        "helix_spans": [list(span) for span in semantics.helix_spans],
        "sheet_spans": [list(span) for span in semantics.sheet_spans],
        "ligand_indices": sorted(semantics.ligand_indices),
        "water_indices": sorted(semantics.water_indices),
        "ion_indices": sorted(semantics.ion_indices),
        "confidence_tier": semantics.confidence_tier.value,
        "confidence_reasons": list(semantics.confidence_reasons),
        "provenance": list(semantics.provenance),
        "trace_chains": {cid: list(idxs) for cid, idxs in semantics.trace_chains.items()},
    }


def protein_semantics_from_dict(payload: dict[str, object]) -> ProteinSemantics:
    """Deserialize graph payload into :class:`ProteinSemantics`."""
    raw_chains = payload.get("chains", {})
    chains: dict[str, ProteinChainSemantics] = {}
    if isinstance(raw_chains, dict):
        for cid, chain_obj in raw_chains.items():
            if not isinstance(chain_obj, dict):
                continue
            residues_obj = chain_obj.get("residues", [])
            residues: list[ProteinResidueSemantics] = []
            if isinstance(residues_obj, list):
                for r in residues_obj:
                    if not isinstance(r, dict):
                        continue
                    residues.append(
                        ProteinResidueSemantics(
                            res_name=str(r.get("res_name", "UNK")),
                            res_seq=_to_int(r.get("res_seq", 0), 0),
                            chain_id=str(r.get("chain_id", cid)),
                            atom_indices=[int(x) for x in r.get("atom_indices", [])],
                            ca_index=(None if r.get("ca_index") is None else int(r["ca_index"])),
                            c_index=(None if r.get("c_index") is None else int(r["c_index"])),
                            o_index=(None if r.get("o_index") is None else int(r["o_index"])),
                            n_index=(None if r.get("n_index") is None else int(r["n_index"])),
                            ss_type=_normalize_ss(r.get("ss_type", "C")),
                        )
                    )
            chains[str(cid)] = ProteinChainSemantics(chain_id=str(chain_obj.get("chain_id", cid)), residues=residues)

    confidence_raw = str(payload.get("confidence_tier", ProteinConfidenceTier.INSUFFICIENT.value))
    try:
        confidence = ProteinConfidenceTier(confidence_raw)
    except ValueError:
        confidence = ProteinConfidenceTier.INSUFFICIENT

    return ProteinSemantics(
        chains=chains,
        hetatm_indices={int(x) for x in payload.get("hetatm_indices", [])},
        backbone_indices={int(x) for x in payload.get("backbone_indices", [])},
        sidechain_indices={int(x) for x in payload.get("sidechain_indices", [])},
        helix_spans=[(str(c), int(s), int(e)) for c, s, e in payload.get("helix_spans", [])],
        sheet_spans=[(str(c), int(s), int(e)) for c, s, e in payload.get("sheet_spans", [])],
        ligand_indices={int(x) for x in payload.get("ligand_indices", [])},
        water_indices={int(x) for x in payload.get("water_indices", [])},
        ion_indices={int(x) for x in payload.get("ion_indices", [])},
        confidence_tier=confidence,
        confidence_reasons=[str(x) for x in payload.get("confidence_reasons", [])],
        provenance=[str(x) for x in payload.get("provenance", [])],
        trace_chains={str(cid): [int(x) for x in idxs] for cid, idxs in payload.get("trace_chains", {}).items()},
    )


def annotate_protein_semantics(
    graph: nx.Graph,
    *,
    atom_annotations: Iterable[dict[str, object]] | None = None,
    format_hint: str | None = None,
    protein_requested: bool = False,
) -> ProteinExtractionReport | None:
    """Extract and attach protein semantics to ``graph``.

    Parameters
    ----------
    graph:
        Input molecular graph.
    atom_annotations:
        Optional canonical per-atom annotations with keys:
        ``record_type``, ``atom_name``, ``res_name``, ``res_seq``,
        ``chain_id``, ``ss_type``.
    format_hint:
        Optional source-format hint for provenance logging.
    protein_requested:
        When ``True``, run graph-only fallback heuristics when metadata
        extraction is insufficient.

    Returns
    -------
    ProteinExtractionReport | None
        ``None`` when no semantics are available and protein mode was not
        requested; otherwise a report with extracted semantics.
    """
    n_atoms = graph.number_of_nodes()
    normalized = _normalize_annotations(atom_annotations, n_atoms)

    provenance = "annotations"
    if format_hint:
        provenance = f"annotations:{format_hint.lower()}"

    semantics: ProteinSemantics | None = None
    if normalized is not None:
        semantics = _extract_from_annotations(graph, normalized, provenance=provenance)

    if semantics is None and protein_requested:
        semantics = _extract_heuristic(graph)

    if semantics is None:
        if not protein_requested:
            return None
        semantics = ProteinSemantics(
            chains={},
            hetatm_indices=set(),
            backbone_indices=set(),
            sidechain_indices=set(),
            helix_spans=[],
            sheet_spans=[],
            ligand_indices=set(),
            water_indices=set(),
            ion_indices=set(),
            confidence_tier=ProteinConfidenceTier.INSUFFICIENT,
            confidence_reasons=["insufficient metadata and weak graph-only signal"],
            provenance=["none"],
            trace_chains={},
        )

    graph.graph["protein_semantics"] = protein_semantics_to_dict(semantics)
    return ProteinExtractionReport(
        semantics=semantics,
        confidence_tier=semantics.confidence_tier,
        confidence_reasons=list(semantics.confidence_reasons),
    )


__all__ = [
    "ProteinChainSemantics",
    "ProteinConfidenceTier",
    "ProteinExtractionReport",
    "ProteinResidueSemantics",
    "ProteinSemantics",
    "annotate_protein_semantics",
    "protein_semantics_from_dict",
    "protein_semantics_to_dict",
]
