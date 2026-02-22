"""Pi-system identification: aromatic rings and non-ring conjugated domains."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from . import geometry as geom

logger = logging.getLogger(__name__)


def analyse_pi_systems(
    G: nx.Graph,
    positions: np.ndarray,
    bo_thresh: float = 1.20,
    cc_dist_thresh: float = 1.5,
    planarity_sigma_max: float = 0.02,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Identify aromatic rings and non-ring pi-domains from graph topology.

    Returns
    -------
    rings : list[tuple[int, ...]]
        Planar 5-7 membered rings.
    pi_domains : list[tuple[int, ...]]
        Non-ring conjugated systems (olefins, conjugated chains).
    """
    rings = _find_rings(G, positions)
    ring_bonds = _ring_bond_set(rings)
    olefin_edges = _find_olefin_bonds(G, positions, ring_bonds, bo_thresh, cc_dist_thresh, planarity_sigma_max)
    pi_domains = _build_pi_domains(olefin_edges, rings)

    logger.debug("Pi systems: %d rings, %d non-ring domains", len(rings), len(pi_domains))
    return rings, pi_domains


def _find_rings(G: nx.Graph, positions: np.ndarray) -> list[tuple[int, ...]]:
    """Find planar 5-7 membered rings."""
    # Use aromatic_rings from graph if available, otherwise cycle_basis
    aromatic = G.graph.get("aromatic_rings", [])
    if aromatic:
        return [tuple(r) for r in aromatic]

    rings = []
    for cyc in nx.cycle_basis(G):
        if 5 <= len(cyc) <= 7:
            pts = positions[cyc]
            normal = geom.plane_normal(pts)
            centroid = pts.mean(axis=0)
            max_dev = max(geom.point_plane_distance(p, centroid, normal) for p in pts)
            if max_dev < 0.15:
                rings.append(tuple(cyc))
    return rings


def _ring_bond_set(rings: list[tuple[int, ...]]) -> set[tuple[int, int]]:
    """Build set of (i,j) bond pairs belonging to any ring."""
    bonds: set[tuple[int, int]] = set()
    for ring in rings:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            bonds.add((min(a, b), max(a, b)))
    return bonds


def _find_olefin_bonds(
    G: nx.Graph,
    positions: np.ndarray,
    ring_bonds: set[tuple[int, int]],
    bo_thresh: float,
    cc_dist_thresh: float,
    planarity_sigma_max: float,
) -> list[tuple[int, int]]:
    """Find non-ring pi-bonds (olefins, conjugated bonds)."""
    olefins: list[tuple[int, int]] = []
    for i, j in G.edges():
        if G.nodes[i]["symbol"] == "H" or G.nodes[j]["symbol"] == "H":
            continue
        bond_key = tuple(sorted((i, j)))
        if bond_key in ring_bonds:
            continue

        bo = G[i][j].get("bond_order")
        if bo is not None and float(bo) > bo_thresh:
            olefins.append((i, j))
            continue

        # C-C distance filter
        if G.nodes[i]["symbol"] == "C" and G.nodes[j]["symbol"] == "C":
            if np.linalg.norm(positions[i] - positions[j]) > cc_dist_thresh:
                continue

        # Planarity check for heteroatom bonds
        ni = [n for n in G.neighbors(i) if n != j and G.nodes[n]["symbol"] != "H"]
        nj = [n for n in G.neighbors(j) if n != i and G.nodes[n]["symbol"] != "H"]
        if not (ni and nj):
            continue
        pts = np.array([positions[i], positions[j], positions[ni[0]], positions[nj[0]]])
        _, s, _ = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        if float(s[-1]) <= planarity_sigma_max:
            olefins.append((i, j))

    return olefins


def _build_pi_domains(
    olefin_edges: list[tuple[int, int]],
    rings: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Group connected olefin bonds into pi-domains, excluding ring atoms."""
    ring_atoms = {a for ring in rings for a in ring}
    pi_graph = nx.Graph()
    for i, j in olefin_edges:
        if i not in ring_atoms and j not in ring_atoms:
            pi_graph.add_edge(i, j)

    return [tuple(sorted(comp)) for comp in nx.connected_components(pi_graph) if len(comp) >= 2]
