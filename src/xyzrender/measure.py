"""Geometry measurements for molecular graphs."""

from __future__ import annotations

import math

import numpy as np


def bond_length(pos_i: np.ndarray, pos_j: np.ndarray) -> float:
    """Distance between two atoms in Å."""
    return float(np.linalg.norm(pos_j - pos_i))


def bond_angle(pos_i: np.ndarray, pos_j: np.ndarray, pos_k: np.ndarray) -> float:
    """Angle i-j-k in degrees (j is the vertex)."""
    v1 = pos_i - pos_j
    v2 = pos_k - pos_j
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def dihedral_angle(pos_i: np.ndarray, pos_j: np.ndarray, pos_k: np.ndarray, pos_l: np.ndarray) -> float:
    """Dihedral angle i-j-k-l in degrees (-180 to 180)."""
    b1 = pos_j - pos_i
    b2 = pos_k - pos_j
    b3 = pos_l - pos_k
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / (np.linalg.norm(b2) + 1e-10)
    m1 = np.cross(n1, b2_hat)
    return math.degrees(math.atan2(float(np.dot(m1, n2)), float(np.dot(n1, n2))))


def _pos(graph, i: int) -> np.ndarray:
    return np.array(graph.nodes[i]["position"], dtype=float)


def all_bond_lengths(graph) -> list[tuple[int, int, float]]:
    """All bonded distances as (i, j, Å), i < j, sorted."""
    result = []
    for i, j in graph.edges():
        a, b = (i, j) if i < j else (j, i)
        result.append((a, b, bond_length(_pos(graph, i), _pos(graph, j))))
    return sorted(result)


def all_bond_angles(graph) -> list[tuple[int, int, int, float]]:
    """All bonded angles as (i, j=center, k, degrees), i < k."""
    result = []
    for j in graph.nodes():
        nbrs = list(graph.neighbors(j))
        for a_idx, i in enumerate(nbrs):
            for k in nbrs[a_idx + 1 :]:
                theta = bond_angle(_pos(graph, i), _pos(graph, j), _pos(graph, k))
                result.append((i, j, k, theta))
    return result


def all_dihedrals(graph) -> list[tuple[int, int, int, int, float]]:
    """All bonded dihedral angles as (i, j, k, m, degrees)."""
    result = []
    seen: set[tuple[int, int]] = set()
    for j, k in graph.edges():
        key = (min(j, k), max(j, k))
        if key in seen:
            continue
        seen.add(key)
        j_nbrs = [n for n in graph.neighbors(j) if n != k]
        k_nbrs = [n for n in graph.neighbors(k) if n != j]
        for i in j_nbrs:
            for m in k_nbrs:
                if i == m:
                    continue
                phi = dihedral_angle(_pos(graph, i), _pos(graph, j), _pos(graph, k), _pos(graph, m))
                result.append((i, j, k, m, phi))
    return result


def print_measurements(graph, modes: str | list[str] = "all") -> None:
    """Print bonded measurements as a formatted table to stdout (1-indexed labels).

    modes may be a single string ("all", "d", "a", "t"/"tor"/"dih") or a list of
    such strings to select multiple sections, e.g. ["d", "a"].
    """
    symbols = {i: graph.nodes[i]["symbol"] for i in graph.nodes()}

    def lbl(i: int) -> str:
        return f"{symbols[i]}{i + 1}"

    if isinstance(modes, str):
        modes = [modes]

    active: set[str] = set()
    for m in modes:
        token = m.lower()
        if token == "all":
            active = {"d", "a", "t"}
            break
        elif token == "d":
            active.add("d")
        elif token == "a":
            active.add("a")
        elif token in ("t", "tor", "dih"):
            active.add("t")

    if "d" in active:
        rows = all_bond_lengths(graph)
        if rows:
            print("Bond Distances:")
            for i, j, d in rows:
                print(f"  {lbl(i):>5s} - {lbl(j):<5s}  {d:.3f}Å")

    if "a" in active:
        rows = all_bond_angles(graph)
        if rows:
            print("Bond Angles:")
            for i, j, k, theta in rows:
                print(f"  {lbl(i):>5s} - {lbl(j)} - {lbl(k):<5s}  {theta:6.2f}°")

    if "t" in active:
        rows = all_dihedrals(graph)
        if rows:
            print("Dihedral Angles:")
            for i, j, k, m, phi in rows:
                print(f"  {lbl(i):>5s} - {lbl(j)} - {lbl(k)} - {lbl(m):<5s}  {phi:7.2f}°")
