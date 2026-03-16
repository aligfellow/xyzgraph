"""Stereochemistry assignment from 3D geometry."""

from __future__ import annotations

import numpy as np

from .data_loader import DATA


def assign_rs(graph) -> dict[int, str]:
    """Assign R/S labels to stereocenters using 3D geometry."""
    rs: dict[int, str] = {}
    nodes = list(graph.nodes())
    pos = np.array([graph.nodes[n]["position"] for n in nodes], dtype=float)

    for center in nodes:
        sym = graph.nodes[center].get("symbol", "")
        if sym == "*" or sym in DATA.metals:
            continue
        nbrs = list(graph.neighbors(center))
        if len(nbrs) != 4:
            continue
        if any(graph.nodes[n].get("symbol", "") == "*" for n in nbrs):
            continue
        if any(graph.nodes[n].get("symbol", "") in DATA.metals for n in nbrs):
            continue

        ranks = _rank_neighbors(graph, center, nbrs)
        sigs = [sig for sig, _ in ranks]
        if len(set(sigs)) < 4:
            continue

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


def assign_ez(graph) -> dict[tuple[int, int], str]:
    """Assign E/Z labels to double bonds using 3D geometry."""
    ez: dict[tuple[int, int], str] = {}
    nodes = list(graph.nodes())
    pos = np.array([graph.nodes[n]["position"] for n in nodes], dtype=float)
    ring_bonds: set[tuple[int, int]] = set()
    rings = graph.graph.get("rings") or []
    for ring in rings:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            ring_bonds.add((a, b) if a < b else (b, a))

    for i, j, data in graph.edges(data=True):
        bo = data.get("bond_order", 1.0)
        if bo < 1.9:
            continue
        if graph.nodes[i].get("symbol", "") == "*" or graph.nodes[j].get("symbol", "") == "*":
            continue
        if graph.nodes[i].get("symbol", "") in DATA.metals or graph.nodes[j].get("symbol", "") in DATA.metals:
            continue
        key = (i, j) if i < j else (j, i)
        if key in ring_bonds:
            continue

        i_nbrs = [n for n in graph.neighbors(i) if n != j]
        j_nbrs = [n for n in graph.neighbors(j) if n != i]
        if len(i_nbrs) < 2 or len(j_nbrs) < 2:
            continue

        i_ranks = _rank_neighbors(graph, i, i_nbrs)
        j_ranks = _rank_neighbors(graph, j, j_nbrs)
        if len(i_ranks) < 2 or len(j_ranks) < 2:
            continue
        if i_ranks[0][0] == i_ranks[1][0]:
            continue
        if j_ranks[0][0] == j_ranks[1][0]:
            continue

        vi = pos[i_ranks[0][1]] - pos[i]
        vj = pos[j_ranks[0][1]] - pos[j]
        b = pos[j] - pos[i]
        bn = np.linalg.norm(b)
        if bn < 1e-6:
            continue
        bh = b / bn
        vi_p = vi - bh * np.dot(vi, bh)
        vj_p = vj - bh * np.dot(vj, bh)
        if np.linalg.norm(vi_p) < 1e-6 or np.linalg.norm(vj_p) < 1e-6:
            continue

        dot = float(np.dot(vi_p, vj_p))
        if abs(dot) < 1e-8:
            continue

        label = "Z" if dot > 0 else "E"
        ez[key] = label

    return ez


def assign_axial(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    """Assign axial chirality (Rₐ/Sₐ) on suitable axes.

    Returns:
      - labels on existing graph edges (axis bond present)
      - labels on non-edge axes (as i, j, label)
    """
    axial: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    axial.update(_assign_axial_biaryl(graph))
    allene_labels, allene_axes = _assign_axial_allene(graph)
    axial.update(allene_labels)
    axes.extend(allene_axes)
    return axial, axes


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


def assign_helical(graph) -> list[tuple[int, int, str]]:
    """Assign helical chirality (P/M) for polycyclic helices (heuristic)."""
    rings = graph.graph.get("aromatic_rings") or []
    if len(rings) < 5:
        return []

    centroids: list[np.ndarray] = []
    for ring in rings:
        coords = np.array([graph.nodes[a]["position"] for a in ring], dtype=float)
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
    if np.linalg.norm(u) < 1e-8:
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
    if span < 1.5:
        return []
    if np.count_nonzero(np.sign(dtheta)) < len(dtheta):
        return []
    if np.mean(np.abs(dtheta)) < 0.35:
        return []

    handed = float(np.sum(dtheta))
    if abs(handed) < 1e-6:
        return []

    label = "P" if handed > 0 else "M"
    first_ring = rings[int(order[0])]
    last_ring = rings[int(order[-1])]
    i = min(first_ring)
    j = min(last_ring)
    return [(i, j, label)]


def annotate_stereo(graph) -> dict[str, dict]:
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
        graph.nodes[idx]["stereo"] = label  # backward compatibility

    ez = assign_ez(graph)
    for (i, j), label in ez.items():
        if graph.has_edge(i, j):
            graph.edges[i, j]["stereo_ez"] = label
            graph.edges[i, j]["stereo"] = label  # backward compatibility

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


def _atomic_number(graph, idx: int) -> int:
    sym = graph.nodes[idx].get("symbol", "")
    return DATA.s2n.get(sym, 0)


def _bond_multiplicity(order: float | None) -> int:
    if order is None:
        return 1
    if order >= 2.5:
        return 3
    if order >= 1.5:
        return 2
    return 1


def _cip_signature(graph, start: int, center: int) -> tuple[tuple[int, ...], ...]:
    """Return a lexicographic CIP-like signature for a substituent tree."""
    max_depth = graph.number_of_nodes()
    frontier: list[tuple[int, int]] = [(start, center)]
    visited_nodes = {center, start}
    layers: list[tuple[int, ...]] = []

    for _ in range(max_depth):
        if not frontier:
            break
        values: list[int] = []
        next_frontier: list[tuple[int, int]] = []
        for node, parent in frontier:
            if graph.has_edge(node, parent):
                order = graph.edges[node, parent].get("bond_order", 1.0)
            else:
                order = 1.0
            mult = _bond_multiplicity(order)
            anum = _atomic_number(graph, node)
            values.extend([anum] * mult)
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
    ranks = [(_cip_signature(graph, nb, center), nb) for nb in neighbors]
    ranks.sort(key=lambda x: x[0], reverse=True)
    return ranks


def _rs_from_vectors(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> str | None:
    eps = 1e-6
    n4 = np.linalg.norm(v4)
    if n4 < eps:
        return None

    w = -v4 / n4  # viewer direction (v4 points away)
    if abs(w[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(w, a)
    nu = np.linalg.norm(u)
    if nu < eps:
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
    if abs(orient) < eps:
        return None
    return "R" if orient < 0 else "S"


def _axis_label(axis: np.ndarray, v_front: np.ndarray, v_back: np.ndarray) -> str | None:
    """Assign Rₐ/Sₐ from front/back substituent vectors around an axis."""
    eps = 1e-8
    n = np.linalg.norm(axis)
    if n < eps:
        return None
    bh = axis / n

    vf = v_front - bh * np.dot(v_front, bh)
    vb = v_back - bh * np.dot(v_back, bh)
    if np.linalg.norm(vf) < eps or np.linalg.norm(vb) < eps:
        return None

    orient = float(np.dot(np.cross(vf, vb), bh))
    if abs(orient) < eps:
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
    if n < 1e-8:
        return None
    return normal / n


def _assign_axial_biaryl(graph) -> dict[tuple[int, int], str]:
    axial: dict[tuple[int, int], str] = {}
    rings = graph.graph.get("aromatic_rings") or []
    if not rings:
        return axial

    # Build aromatic domains (fused ring groups)
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

    domains: dict[int, set[int]] = {}
    for idx, rset in enumerate(ring_sets):
        root = _find(idx)
        domains.setdefault(root, set()).update(rset)

    atom_domain: dict[int, int] = {}
    for d_idx, atoms in enumerate(domains.values()):
        for atom in atoms:
            atom_domain[atom] = d_idx

    pos = np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)

    for i, j, data in graph.edges(data=True):
        if atom_domain.get(i) is None or atom_domain.get(j) is None:
            continue
        if atom_domain[i] == atom_domain[j]:
            continue
        if data.get("bond_order", 1.0) > 1.3:
            continue

        n_i = [n for n in graph.neighbors(i) if n != j]
        n_j = [n for n in graph.neighbors(j) if n != i]
        if len(n_i) < 2 or len(n_j) < 2:
            continue

        ranks_i = _rank_neighbors(graph, i, n_i)
        ranks_j = _rank_neighbors(graph, j, n_j)
        if ranks_i[0][0] == ranks_i[1][0] or ranks_j[0][0] == ranks_j[1][0]:
            continue

        sig_i = ranks_i[0][0]
        sig_j = ranks_j[0][0]

        if sig_i > sig_j:
            front, back = i, j
            v_front = pos[ranks_i[0][1]] - pos[i]
            v_back = pos[ranks_j[0][1]] - pos[j]
        elif sig_j > sig_i:
            front, back = j, i
            v_front = pos[ranks_j[0][1]] - pos[j]
            v_back = pos[ranks_i[0][1]] - pos[i]
        else:
            # symmetric ends: fall back to index order for deterministic label
            if i < j:
                front, back = i, j
                v_front = pos[ranks_i[0][1]] - pos[i]
                v_back = pos[ranks_j[0][1]] - pos[j]
            else:
                front, back = j, i
                v_front = pos[ranks_j[0][1]] - pos[j]
                v_back = pos[ranks_i[0][1]] - pos[i]

        axis = pos[back] - pos[front]
        label = _axis_label(axis, v_front, v_back)
        if label is None:
            continue

        key = (i, j) if i < j else (j, i)
        axial[key] = label

    return axial


def _assign_axial_allene(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    axial: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    pos = np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)

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

        sig_a = rank_a[0][0]
        sig_b = rank_b[0][0]

        if sig_a > sig_b:
            front, back = end_a, end_b
            v_front = pos[rank_a[0][1]] - pos[end_a]
            v_back = pos[rank_b[0][1]] - pos[end_b]
        elif sig_b > sig_a:
            front, back = end_b, end_a
            v_front = pos[rank_b[0][1]] - pos[end_b]
            v_back = pos[rank_a[0][1]] - pos[end_a]
        else:
            # symmetric ends: fall back to index order for deterministic label
            if end_a < end_b:
                front, back = end_a, end_b
                v_front = pos[rank_a[0][1]] - pos[end_a]
                v_back = pos[rank_b[0][1]] - pos[end_b]
            else:
                front, back = end_b, end_a
                v_front = pos[rank_b[0][1]] - pos[end_b]
                v_back = pos[rank_a[0][1]] - pos[end_a]

        axis = pos[back] - pos[front]
        label = _axis_label(axis, v_front, v_back)
        if label is None:
            continue

        key = (front, back) if front < back else (back, front)
        if graph.has_edge(front, back):
            axial[key] = label
        else:
            axes.append((front, back, label))

    return axial, axes


def _assign_planar_metallocene(graph) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    planar: dict[tuple[int, int], str] = {}
    axes: list[tuple[int, int, str]] = []
    pos = np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)
    rings = graph.graph.get("rings") or []
    ring_sets = [set(r) for r in rings]

    # If no rings were detected, attempt Cp-like rings around metals by geometry.
    ring_candidates: list[tuple[list[int], int]] = []
    metals = [idx for idx in graph.nodes() if graph.nodes[idx].get("symbol", "") in DATA.metals]
    if not rings and metals:
        for metal in metals:
            carbon_idx = [
                i
                for i in graph.nodes()
                if graph.nodes[i].get("symbol", "") == "C"
                and np.linalg.norm(pos[i] - pos[metal]) < 2.6
            ]
            if len(carbon_idx) < 8:
                continue
            coords = pos[carbon_idx] - pos[metal]
            _, _, vh = np.linalg.svd(coords, full_matrices=False)
            axis = vh[0]
            proj = coords @ axis
            top = [i for i, p in zip(carbon_idx, proj) if p >= 0]
            bottom = [i for i, p in zip(carbon_idx, proj) if p < 0]
            for ring in (top, bottom):
                if len(ring) < 5:
                    continue
                ccoords = pos[ring]
                centroid = ccoords.mean(axis=0)
                ring = sorted(ring, key=lambda i: np.linalg.norm(pos[i] - centroid))[:5]
                ring_candidates.append((ring, metal))

    for ring_idx, ring in enumerate(rings):
        if len(ring) != 5:
            continue
        if not all(graph.nodes[a].get("symbol", "") == "C" for a in ring):
            continue
        metals = {
            nb
            for a in ring
            for nb in graph.neighbors(a)
            if graph.nodes[nb].get("symbol", "") in DATA.metals
        }
        if not metals:
            metals = {
                idx
                for idx in graph.nodes()
                if graph.nodes[idx].get("symbol", "") in DATA.metals
                and np.min(np.linalg.norm(pos[ring] - pos[idx], axis=1)) < 2.6
            }
        for metal in metals:
            ring_candidates.append((ring, metal))

    for ring, metal in ring_candidates:
        ring_set = set(ring)
        coords = np.array([graph.nodes[a]["position"] for a in ring], dtype=float)
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
        if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
            continue

        orient = float(np.dot(normal, np.cross(v1, v2)))
        if abs(orient) < 1e-8:
            continue
        label = "Rₚ" if orient > 0 else "Sₚ"

        key = (metal, a1) if metal < a1 else (a1, metal)
        if graph.has_edge(metal, a1):
            planar[key] = label
        else:
            axes.append((metal, a1, label))

    return planar, axes
