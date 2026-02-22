"""NCI graph decoration: add NCI edges and pi-centroid nodes to a molecular graph."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from .interaction import NCIData


def build_nci_graph(G: nx.Graph, ncis: list[NCIData] | None = None) -> nx.Graph:
    """Return a copy of *G* decorated with NCI edges and pi-centroid nodes.

    NCI edges are added with ``bond_order=0.0``, ``NCI=True``, and
    ``nci_type`` set to the interaction type string.  For pi-system
    sites, centroid dummy nodes (``symbol="*"``) are inserted at the
    mean 3-D position of the site atoms.

    Parameters
    ----------
    G : nx.Graph
        Molecular graph (unchanged).
    ncis : list[NCIData], optional
        Interactions to decorate.  Defaults to ``G.graph["ncis"]``.

    Returns
    -------
    nx.Graph
        Deep copy of *G* with NCI edges and centroid nodes added.
        ``result.graph["nci_centroid"]`` lists the centroid node IDs.
        Centroid nodes also have ``symbol="*"``.
    """
    nci_G = copy.deepcopy(G)

    if ncis is None:
        ncis = nci_G.graph.get("ncis", [])

    if not ncis:
        return nci_G

    atom_atom = [n for n in ncis if len(n.site_a) == 1 and len(n.site_b) == 1]
    pi_ncis = [n for n in ncis if n not in atom_atom]

    # Atom-atom NCI edges
    for nci in atom_atom:
        if nci.aux_atoms and nci_G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            a, b = nci.aux_atoms[0], nci.site_b[0]
        else:
            a, b = nci.site_a[0], nci.site_b[0]
        if not nci_G.has_edge(a, b):
            nci_G.add_edge(a, b, bond_order=0.0, NCI=True, nci_type=nci.type)

    # Pi-system centroid nodes
    centroid_nodes: dict[tuple[int, ...], int] = {}
    next_id = max(nci_G.nodes()) + 1

    for nci in pi_ncis:
        for site in [nci.site_a, nci.site_b]:
            if len(site) > 1:
                key = tuple(sorted(site))
                if key not in centroid_nodes:
                    positions = np.array([nci_G.nodes[i]["position"] for i in site])
                    centroid_pos = tuple(positions.mean(axis=0).tolist())
                    cid = next_id
                    next_id += 1
                    nci_G.add_node(cid, symbol="*", position=centroid_pos)
                    centroid_nodes[key] = cid

    # Pi NCI edges (to/from centroid nodes)
    for nci in pi_ncis:
        key_a = tuple(sorted(nci.site_a)) if len(nci.site_a) > 1 else None
        key_b = tuple(sorted(nci.site_b)) if len(nci.site_b) > 1 else None
        if nci.aux_atoms and nci_G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            node_a = nci.aux_atoms[0]
        else:
            node_a = centroid_nodes[key_a] if key_a else nci.site_a[0]
        node_b = centroid_nodes[key_b] if key_b else nci.site_b[0]
        if not nci_G.has_edge(node_a, node_b):
            nci_G.add_edge(node_a, node_b, bond_order=0.0, NCI=True, nci_type=nci.type)

    nci_G.graph["nci_centroid"] = list(centroid_nodes.values())
    return nci_G
