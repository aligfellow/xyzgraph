"""Stereochemistry assignment from 3D geometry."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .data_loader import DATA


class StereoSummary(TypedDict):
    """Return type for :func:`annotate_stereo`."""

    rs: dict[int, str]
    ez: dict[tuple[int, int], str]
    axial: dict[tuple[int, int], str]
    planar: dict[tuple[int, int], str]
    helical: list[tuple[int, int, str]]


_EPS = 1e-8
_CIP_MAX_DEPTH = 20
_SMALL_RING_MAX = 6  # rings with ≤6 members can't have E/Z
_HELICAL_MIN_RINGS = 5  # minimum fused rings for helicene
_HELICAL_MIN_SPAN = 1.5  # Å — minimum vertical extent of helix
_HELICAL_MIN_AVG_TWIST = 0.35  # radians (~20°) — minimum average twist per step


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_pos(graph) -> dict[int, np.ndarray]:
    """Build position cache {node_id: position_vector}."""
    return {n: np.asarray(graph.nodes[n]["position"], dtype=float) for n in graph.nodes()}


def _is_metal(graph, n: int) -> bool:
    return graph.nodes[n].get("symbol", "") in DATA.metals


def _is_dummy(graph, n: int) -> bool:
    return graph.nodes[n].get("symbol", "") == "*"


def _is_tetrahedral(pos: dict[int, np.ndarray], center: int, nbrs: list[int]) -> bool:
    """Check that 4 neighbors form tetrahedral geometry (not planar/linear)."""
    p = [pos[n] - pos[center] for n in nbrs]
    # Signed volume of tetrahedron — near zero means coplanar (sp2)
    vol = abs(float(np.dot(p[0], np.cross(p[1], p[2]))))
    if vol < 0.01:
        return False
    # Reject if any angle > 145° (nearly linear)
    for i in range(4):
        for j in range(i + 1, 4):
            cos_a = float(np.dot(p[i], p[j])) / (np.linalg.norm(p[i]) * np.linalg.norm(p[j]) + _EPS)
            if cos_a < -0.82:  # cos(145°) ≈ -0.82
                return False
    return True


def _fused_ring_domains(rings: list[list[int]]) -> dict[int, list[int]]:
    """Group rings into fused domains by shared atoms.

    Returns {root_idx: [ring_indices]}.
    """
    ring_sets = [set(r) for r in rings]
    parent = list(range(len(ring_sets)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ring_sets)):
        for j in range(i + 1, len(ring_sets)):
            if ring_sets[i] & ring_sets[j]:
                _union(i, j)

    domains: dict[int, list[int]] = {}
    for idx in range(len(rings)):
        domains.setdefault(_find(idx), []).append(idx)
    return domains


# ---------------------------------------------------------------------------
# CIP ranking
# ---------------------------------------------------------------------------


def _atomic_number(graph, idx: int) -> int:
    sym = graph.nodes[idx].get("symbol", "")
    return DATA.s2n.get(sym, 0)


def _bond_multiplicity(order: float | None) -> int:
    if order is None:
        return 1
    if abs(order - 1.5) < 0.01:  # aromatic — keep as double
        return 2
    rounded = round(order)  # 1.4→1, 1.6→2, 2.4→2, 2.6→3
    return max(1, min(3, rounded))


def _substituent_signature(
    graph,
    start: int,
    center: int,
    *,
    use_bond_orders: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """BFS signature for a substituent tree rooted at *start* (away from *center*).

    With *use_bond_orders* (default) bond multiplicity duplicates atoms in
    each layer (CIP phantom-atom convention).  Without, only element
    connectivity matters — used to detect resonance-equivalent substituents
    (e.g. the two O's in S(=O)₂ that the optimizer gave different bond orders).
    """
    frontier: list[tuple[int, int]] = [(start, center)]
    visited_nodes = {center, start}
    layers: list[tuple[int, ...]] = []

    for _ in range(_CIP_MAX_DEPTH):
        if not frontier:
            break
        values: list[int] = []
        next_frontier: list[tuple[int, int]] = []
        for node, parent in frontier:
            anum = _atomic_number(graph, node)
            if use_bond_orders:
                if graph.has_edge(node, parent):
                    order = graph.edges[node, parent].get("bond_order", 1.0)
                else:
                    order = 1.0
                values.extend([anum] * _bond_multiplicity(order))
            else:
                values.append(anum)
            for nb in graph.neighbors(node):
                if nb == parent:
                    continue
                if nb in visited_nodes:
                    continue
                visited_nodes.add(nb)
                next_frontier.append((nb, node))
        values.sort(reverse=True)
        layers.append(tuple(values))
        frontier = next_frontier

    return tuple(layers)


def _rank_neighbors(graph, center: int, neighbors: list[int]) -> list[tuple[tuple[tuple[int, ...], ...], int]]:
    ranks = [(_substituent_signature(graph, nb, center), nb) for nb in neighbors]
    ranks.sort(key=lambda x: x[0], reverse=True)
    return ranks


def _count_unique_substituents(graph, center: int, neighbors: list[int]) -> int:
    """Count genuinely distinct substituents using topology-only signatures."""
    topo_sigs = {_substituent_signature(graph, nb, center, use_bond_orders=False) for nb in neighbors}
    return len(topo_sigs)


def _count_ortho_subs(
    graph,
    bridge: int,
    other_bridge: int,
    bridge_nbrs: list[int],
    ring_atoms: set[int],
) -> int:
    """Count non-H substituents on ring neighbors of a bridge atom (ortho positions)."""
    count = 0
    for n in bridge_nbrs:
        if n not in ring_atoms:
            continue  # not an ortho ring neighbor
        for sub in graph.neighbors(n):
            if sub == bridge or sub in ring_atoms or sub == other_bridge:
                continue
            if graph.nodes[sub].get("symbol", "") != "H":
                count += 1
    return count


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rs_from_vectors(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> str | None:
    n4 = np.linalg.norm(v4)
    if n4 < _EPS:
        return None

    w = -v4 / n4  # viewer direction (v4 points away)
    if abs(w[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(w, a)
    nu = np.linalg.norm(u)
    if nu < _EPS:
        return None
    u /= nu
    v = np.cross(w, u)

    def _proj(vec: np.ndarray) -> tuple[float, float]:
        p = vec - w * np.dot(vec, w)
        return float(np.dot(p, u)), float(np.dot(p, v))

    x1, y1 = _proj(v1)
    x2, y2 = _proj(v2)
    x3, y3 = _proj(v3)

    orient = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if abs(orient) < _EPS:
        return None
    return "R" if orient < 0 else "S"


def _axis_label(axis: np.ndarray, v_front: np.ndarray, v_back: np.ndarray) -> str | None:
    """Assign Rₐ/Sₐ from front/back substituent vectors around an axis."""
    n = np.linalg.norm(axis)
    if n < _EPS:
        return None
    bh = axis / n

    vf = v_front - bh * np.dot(v_front, bh)
    vb = v_back - bh * np.dot(v_back, bh)
    if np.linalg.norm(vf) < _EPS or np.linalg.norm(vb) < _EPS:
        return None

    orient = float(np.dot(np.cross(vf, vb), bh))
    if abs(orient) < _EPS:
        return None
    return "Rₐ" if orient > 0 else "Sₐ"


def _plane_normal(coords: np.ndarray) -> np.ndarray | None:
    """Best-fit plane normal for a set of points."""
    if coords.shape[0] < 3:
        return None
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    n = np.linalg.norm(normal)
    if n < _EPS:
        return None
    return normal / n


def _determine_front_back(
    pos: dict[int, np.ndarray],
    ranks_a: list[tuple[tuple[tuple[int, ...], ...], int]],
    ranks_b: list[tuple[tuple[tuple[int, ...], ...], int]],
    end_a: int,
    end_b: int,
    *,
    symmetric_is_chiral: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Determine axis direction and front/back substituent vectors.

    Returns (axis_vec, v_front, v_back) or None if axis is achiral.

    Parameters
    ----------
    symmetric_is_chiral : bool
        If True, symmetric ends still produce a result (use index tiebreaker).
        Allenes with identical ends are still chiral due to the 90° twist.
        If False, symmetric ends return None (biaryls with identical
        substitution are achiral).
    """
    sig_a, sig_b = ranks_a[0][0], ranks_b[0][0]
    if sig_a > sig_b:
        front, back = end_a, end_b
        v_front = pos[ranks_a[0][1]] - pos[end_a]
        v_back = pos[ranks_b[0][1]] - pos[end_b]
    elif sig_b > sig_a:
        front, back = end_b, end_a
        v_front = pos[ranks_b[0][1]] - pos[end_b]
        v_back = pos[ranks_a[0][1]] - pos[end_a]
    else:
        # Primary sigs tied — compare second-highest
        sig_a2 = ranks_a[1][0] if len(ranks_a) > 1 else ()
        sig_b2 = ranks_b[1][0] if len(ranks_b) > 1 else ()
        if sig_a2 == sig_b2:
            if not symmetric_is_chiral:
                return None  # achiral by symmetry (biaryls)
            # Allenes: symmetric ends still chiral, use index tiebreaker
            if end_a < end_b:
                front, back = end_a, end_b
                v_front = pos[ranks_a[0][1]] - pos[end_a]
                v_back = pos[ranks_b[0][1]] - pos[end_b]
            else:
                front, back = end_b, end_a
                v_front = pos[ranks_b[0][1]] - pos[end_b]
                v_back = pos[ranks_a[0][1]] - pos[end_a]
        elif sig_a2 > sig_b2:
            front, back = end_a, end_b
            v_front = pos[ranks_a[0][1]] - pos[end_a]
            v_back = pos[ranks_b[0][1]] - pos[end_b]
        else:
            front, back = end_b, end_a
            v_front = pos[ranks_b[0][1]] - pos[end_b]
            v_back = pos[ranks_a[0][1]] - pos[end_a]
    return pos[back] - pos[front], v_front, v_back


# ---------------------------------------------------------------------------
# R/S assignment
# ---------------------------------------------------------------------------


def assign_rs(graph) -> dict[int, str]:
    """Assign R/S labels to stereocenters using 3D geometry."""
    rs: dict[int, str] = {}
    pos = _build_pos(graph)

    # Aromatic atoms are sp2 — never tetrahedral stereocenters
    aromatic_atoms: set[int] = set()
    for ring in graph.graph.get("aromatic_rings") or []:
        aromatic_atoms.update(ring)

    for center in graph.nodes():
        if _is_dummy(graph, center) or _is_metal(graph, center):
            continue
        if center in aromatic_atoms:
            continue
        nbrs = list(graph.neighbors(center))
        if len(nbrs) != 4:
            continue
        if any(_is_dummy(graph, n) for n in nbrs):
            continue
        if not _is_tetrahedral(pos, center, nbrs):
            continue
        # Check topology-only equivalence — catches resonance-equivalent
        # substituents (e.g. two O's in S(=O)₂) that the optimizer may
        # have given different bond orders
        if _count_unique_substituents(graph, center, nbrs) < 4:
            continue

        ranks = _rank_neighbors(graph, center, nbrs)
        ordered = [nb for _, nb in ranks]
        v1 = pos[ordered[0]] - pos[center]
        v2 = pos[ordered[1]] - pos[center]
        v3 = pos[ordered[2]] - pos[center]
        v4 = pos[ordered[3]] - pos[center]

        label = _rs_from_vectors(v1, v2, v3, v4)
        if label is None:
            continue
        rs[center] = label

    return rs


# ---------------------------------------------------------------------------
# E/Z assignment
# ---------------------------------------------------------------------------


def assign_ez(graph) -> dict[tuple[int, int], str]:
    """Assign E/Z labels to double bonds using 3D geometry."""
    ez: dict[tuple[int, int], str] = {}
    pos = _build_pos(graph)

    # Exclude aromatic ring bonds
    excluded_bonds: set[tuple[int, int]] = set()
    aromatic_rings = graph.graph.get("aromatic_rings") or []
    for ring in aromatic_rings:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            excluded_bonds.add((a, b) if a < b else (b, a))

    # Exclude double bonds in small rings (≤6 members) — geometry-locked
    all_rings = graph.graph.get("rings") or []
    for ring in all_rings:
        if len(ring) <= _SMALL_RING_MAX:
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                excluded_bonds.add((a, b) if a < b else (b, a))

    for i, j, data in graph.edges(data=True):
        bo = data.get("bond_order", 1.0)
        if bo < 1.9:
            continue
        if _is_dummy(graph, i) or _is_dummy(graph, j):
            continue
        if _is_metal(graph, i) or _is_metal(graph, j):
            continue
        key = (i, j) if i < j else (j, i)
        if key in excluded_bonds:
            continue
        if any(_is_metal(graph, n) for n in graph.neighbors(i)):
            continue
        if any(_is_metal(graph, n) for n in graph.neighbors(j)):
            continue

        i_nbrs = [n for n in graph.neighbors(i) if n != j]
        j_nbrs = [n for n in graph.neighbors(j) if n != i]
        if len(i_nbrs) < 2 or len(j_nbrs) < 2:
            continue

        i_ranks = _rank_neighbors(graph, i, i_nbrs)
        j_ranks = _rank_neighbors(graph, j, j_nbrs)
        if i_ranks[0][0] == i_ranks[1][0]:
            continue
        if j_ranks[0][0] == j_ranks[1][0]:
            continue

        vi = pos[i_ranks[0][1]] - pos[i]
        vj = pos[j_ranks[0][1]] - pos[j]
        b = pos[j] - pos[i]
        bn = np.linalg.norm(b)
        if bn < _EPS:
            continue
        bh = b / bn
        vi_p = vi - bh * np.dot(vi, bh)
        vj_p = vj - bh * np.dot(vj, bh)
        if np.linalg.norm(vi_p) < _EPS or np.linalg.norm(vj_p) < _EPS:
            continue

        dot = float(np.dot(vi_p, vj_p))
        if abs(dot) < _EPS:
            continue

        label = "Z" if dot > 0 else "E"
        ez[key] = label

    return ez


# ---------------------------------------------------------------------------
# Axial chirality
# ---------------------------------------------------------------------------


def assign_axial(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    """Assign axial chirality (Rₐ/Sₐ) on suitable axes.

    Returns
    -------
    tuple
        Labels on existing graph edges (axis bond present), plus labels on non-edge axes
        (as i, j, label).
    """
    axial: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    axial.update(_assign_axial_ring_bridge(graph))
    allene_labels, allene_axes = _assign_axial_allene(graph)
    axial.update(allene_labels)
    axes.extend(allene_axes)
    return axial, axes


def _assign_axial_ring_bridge(graph) -> dict[tuple[int, int], str]:
    axial: dict[tuple[int, int], str] = {}
    all_rings = graph.graph.get("rings") or []
    if not all_rings:
        return axial

    # Map each atom to the set of ring indices it belongs to
    atom_rings: dict[int, set[int]] = {}
    ring_sets = [set(r) for r in all_rings]
    for r_idx, rset in enumerate(ring_sets):
        for atom in rset:
            atom_rings.setdefault(atom, set()).add(r_idx)

    pos = _build_pos(graph)

    for i, j, data in graph.edges(data=True):
        if data.get("bond_order", 1.0) > 1.3:
            continue
        # Both atoms must be in rings
        i_rings = atom_rings.get(i)
        j_rings = atom_rings.get(j)
        if not i_rings or not j_rings:
            continue
        # Must not share a ring (bridge connects different ring systems)
        if i_rings & j_rings:
            continue

        n_i = [n for n in graph.neighbors(i) if n != j]
        n_j = [n for n in graph.neighbors(j) if n != i]
        if len(n_i) < 2 or len(n_j) < 2:
            continue

        # Ortho steric gating: count non-H substituents on ring neighbors
        # of each bridge atom (these create the rotation barrier)
        i_ring_atoms = set()
        for r_idx in i_rings:
            i_ring_atoms.update(ring_sets[r_idx])
        j_ring_atoms = set()
        for r_idx in j_rings:
            j_ring_atoms.update(ring_sets[r_idx])

        ortho_i = _count_ortho_subs(graph, i, j, n_i, i_ring_atoms)
        ortho_j = _count_ortho_subs(graph, j, i, n_j, j_ring_atoms)

        # Need ≥2 ortho substituents total (1+1 or 2+0)
        if ortho_i + ortho_j < 2:
            continue

        ranks_i = _rank_neighbors(graph, i, n_i)
        ranks_j = _rank_neighbors(graph, j, n_j)
        if ranks_i[0][0] == ranks_i[1][0] or ranks_j[0][0] == ranks_j[1][0]:
            continue

        result = _determine_front_back(pos, ranks_i, ranks_j, i, j)
        if result is None:
            continue
        axis, v_front, v_back = result
        label = _axis_label(axis, v_front, v_back)
        if label is None:
            continue

        key = (i, j) if i < j else (j, i)
        axial[key] = label

    return axial


def _assign_axial_allene(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    axial: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    pos = _build_pos(graph)

    for center in graph.nodes():
        nbrs = list(graph.neighbors(center))
        if len(nbrs) != 2:
            continue
        bo1 = graph.edges[center, nbrs[0]].get("bond_order", 1.0)
        bo2 = graph.edges[center, nbrs[1]].get("bond_order", 1.0)
        if bo1 < 1.9 or bo2 < 1.9:
            continue

        end_a, end_b = nbrs
        a_subs = [n for n in graph.neighbors(end_a) if n != center]
        b_subs = [n for n in graph.neighbors(end_b) if n != center]
        if len(a_subs) < 2 or len(b_subs) < 2:
            continue

        rank_a = _rank_neighbors(graph, end_a, a_subs)
        rank_b = _rank_neighbors(graph, end_b, b_subs)
        if rank_a[0][0] == rank_a[1][0] or rank_b[0][0] == rank_b[1][0]:
            continue

        result = _determine_front_back(
            pos,
            rank_a,
            rank_b,
            end_a,
            end_b,
            symmetric_is_chiral=True,
        )
        if result is None:
            continue
        axis, v_front, v_back = result
        label = _axis_label(axis, v_front, v_back)
        if label is None:
            continue

        key = (end_a, end_b) if end_a < end_b else (end_b, end_a)
        if graph.has_edge(end_a, end_b):
            axial[key] = label
        else:
            axes.append((end_a, end_b, label))

    return axial, axes


# ---------------------------------------------------------------------------
# Planar chirality
# ---------------------------------------------------------------------------


def assign_planar(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    """Assign planar chirality (Rₚ/Sₚ) on suitable planar systems.

    Currently targets substituted metallocene-like Cp rings.
    """
    planar: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    planar_labels, planar_axes = _assign_planar_metallocene(graph)
    planar.update(planar_labels)
    axes.extend(planar_axes)
    return planar, axes


def _assign_planar_metallocene(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    planar: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    pos = _build_pos(graph)
    rings = graph.graph.get("rings") or []
    ring_candidates: list[tuple[list[int], int]] = []
    for ring in rings:
        if len(ring) != 5:
            continue
        if not all(graph.nodes[a].get("symbol", "") == "C" for a in ring):
            continue
        metals = {nb for a in ring for nb in graph.neighbors(a) if _is_metal(graph, nb)}
        for metal in metals:
            ring_candidates.append((ring, metal))

    for ring, metal in ring_candidates:
        ring_set = set(ring)
        coords = np.array([pos[a] for a in ring], dtype=float)
        normal = _plane_normal(coords)
        if normal is None:
            continue
        centroid = coords.mean(axis=0)

        to_metal = pos[metal] - centroid
        if np.dot(normal, to_metal) < 0:
            normal = -normal

        candidates: list[tuple[tuple[tuple[int, ...], ...], int]] = []
        for atom in ring:
            externals = [n for n in graph.neighbors(atom) if n not in ring_set and n != metal]
            if not externals:
                continue
            ranks = _rank_neighbors(graph, atom, externals)
            candidates.append((ranks[0][0], atom))

        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates[0][0] == candidates[1][0]:
            continue

        a1 = candidates[0][1]
        a2 = candidates[1][1]
        v1 = pos[a1] - centroid
        v2 = pos[a2] - centroid
        v1 = v1 - normal * np.dot(v1, normal)
        v2 = v2 - normal * np.dot(v2, normal)
        if np.linalg.norm(v1) < _EPS or np.linalg.norm(v2) < _EPS:
            continue

        orient = float(np.dot(normal, np.cross(v1, v2)))
        if abs(orient) < _EPS:
            continue
        label = "Rₚ" if orient > 0 else "Sₚ"

        key = (metal, a1) if metal < a1 else (a1, metal)
        if graph.has_edge(metal, a1):
            planar[key] = label
        else:
            axes.append((metal, a1, label))

    return planar, axes


# ---------------------------------------------------------------------------
# Helical chirality
# ---------------------------------------------------------------------------


def assign_helical(graph) -> list[tuple[int, int, str]]:
    """Assign helical chirality (P/M) for polycyclic helices (heuristic)."""
    rings = graph.graph.get("aromatic_rings") or []
    if len(rings) < _HELICAL_MIN_RINGS:
        return []

    # Only analyze fused domains with enough rings
    fused = _fused_ring_domains(rings)
    pos = _build_pos(graph)

    results: list[tuple[int, int, str]] = []
    for ring_indices in fused.values():
        if len(ring_indices) < _HELICAL_MIN_RINGS:
            continue
        domain_rings = [rings[i] for i in ring_indices]
        result = _helical_from_fused_rings(pos, domain_rings)
        if result:
            results.extend(result)
    return results


def _helical_from_fused_rings(pos: dict[int, np.ndarray], rings: list[list[int]]) -> list[tuple[int, int, str]]:
    """Detect P/M helicity in a set of fused rings."""
    centroids: list[np.ndarray] = []
    for ring in rings:
        coords = np.array([pos[a] for a in ring], dtype=float)
        centroids.append(coords.mean(axis=0))
    centroid_arr = np.vstack(centroids)

    centered = centroid_arr - centroid_arr.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    proj = centered @ axis
    order = np.argsort(proj)
    ordered = centroid_arr[order]

    # Build orthonormal basis around axis
    a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, a)
    if np.linalg.norm(u) < _EPS:
        return []
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    proj_pts = ordered - ordered.mean(axis=0)
    x = proj_pts @ u
    y = proj_pts @ v
    angles = np.arctan2(y, x)
    dtheta = np.diff(angles)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    span = float(proj.max() - proj.min())
    if span < _HELICAL_MIN_SPAN:
        return []
    if np.count_nonzero(np.sign(dtheta)) < len(dtheta):
        return []
    if np.mean(np.abs(dtheta)) < _HELICAL_MIN_AVG_TWIST:
        return []

    handed = float(np.sum(dtheta))
    if abs(handed) < _EPS:
        return []

    label = "P" if handed > 0 else "M"
    first_ring = rings[int(order[0])]
    last_ring = rings[int(order[-1])]
    i = min(first_ring)
    j = min(last_ring)
    return [(i, j, label)]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def annotate_stereo(graph) -> StereoSummary:
    """Assign stereochemistry and store labels on node/edge attributes.

    Keys:
      - Nodes: ``stereo_rs`` (R/S)
      - Edges: ``stereo_ez`` (E/Z), ``stereo_axial`` (Rₐ/Sₐ), ``stereo_planar`` (Rₚ/Sₚ)

    For axes that are not graph edges (e.g., allenes, metallocenes),
    entries are added to ``graph.graph["stereo_axes"]`` as
    ``{"i": i, "j": j, "label": label, "kind": "axial"|"planar"}``.
    """
    rs = assign_rs(graph)
    for idx, label in rs.items():
        graph.nodes[idx]["stereo_rs"] = label

    ez = assign_ez(graph)
    for (i, j), label in ez.items():
        if graph.has_edge(i, j):
            graph.edges[i, j]["stereo_ez"] = label

    axial, axial_axes = assign_axial(graph)
    for (i, j), label in axial.items():
        if graph.has_edge(i, j):
            graph.edges[i, j]["stereo_axial"] = label

    planar, planar_axes = assign_planar(graph)
    for (i, j), label in planar.items():
        if graph.has_edge(i, j):
            graph.edges[i, j]["stereo_planar"] = label

    axes: list[dict[str, object]] = []
    for i, j, label in axial_axes:
        axes.append({"i": i, "j": j, "label": label, "kind": "axial"})
    for i, j, label in planar_axes:
        axes.append({"i": i, "j": j, "label": label, "kind": "planar"})

    helical_axes = assign_helical(graph)
    for i, j, label in helical_axes:
        axes.append({"i": i, "j": j, "label": label, "kind": "helical"})
    if axes:
        graph.graph["stereo_axes"] = axes

    return {"rs": rs, "ez": ez, "axial": axial, "planar": planar, "helical": helical_axes}
