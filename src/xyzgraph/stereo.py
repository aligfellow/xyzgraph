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
        if sym == "*":
            continue
        nbrs = list(graph.neighbors(center))
        if len(nbrs) != 4:
            continue
        if any(graph.nodes[n].get("symbol", "") == "*" for n in nbrs):
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

    for i, j, data in graph.edges(data=True):
        bo = data.get("bond_order", 1.0)
        if bo < 1.9:
            continue
        if graph.nodes[i].get("symbol", "") == "*" or graph.nodes[j].get("symbol", "") == "*":
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
        key = (i, j) if i < j else (j, i)
        ez[key] = label

    return ez


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
