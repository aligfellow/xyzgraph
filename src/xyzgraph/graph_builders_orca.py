"""ORCA-based molecular graph construction."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from .data_loader import DATA
from .orca_parser import OrcaParseError, parse_orca_output

logger = logging.getLogger(__name__)

ORCA_BOND_THRESHOLD = 0.25
"""Default minimum Mayer bond order to count as a bond."""


def build_graph_orca(
    orca_file: str,
    bond_threshold: float = ORCA_BOND_THRESHOLD,
    debug: bool = False,
) -> nx.Graph:
    """Build molecular graph from ORCA quantum chemistry output file.

    Uses Mayer bond orders and Mulliken charges from ORCA calculations.
    Coordinates, charge, and multiplicity are read from the output file.

    Parameters
    ----------
    orca_file : str
        Path to ORCA output file.
    bond_threshold : float
        Minimum Mayer bond order to consider as a bond.
    debug : bool
        Enable debug logging.  Not needed when called via
        ``build_graph()`` which configures logging upstream.

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing:
        ``symbol``, ``atomic_number``, ``position``, ``charges``,
        ``formal_charge``, ``valence``, ``agg_charge``.

    Raises
    ------
    OrcaParseError
        If ORCA output cannot be parsed or required data is missing.
    """
    if debug:
        from .utils import configure_debug_logging

        configure_debug_logging()

    # Parse ORCA output
    try:
        orca_data = parse_orca_output(orca_file)
    except OrcaParseError as e:
        raise OrcaParseError(f"Failed to parse ORCA output: {e}") from e

    atoms = orca_data["atoms"]
    bonds = orca_data["bonds"]
    mulliken_charges = orca_data["charges"]
    charge = orca_data["charge"]
    multiplicity = orca_data["multiplicity"]

    logger.debug("Parsed ORCA output: %d atoms, %d bonds (before threshold)", len(atoms), len(bonds))
    logger.debug("Charge: %d, Multiplicity: %d", charge, multiplicity)

    # Build graph
    G = nx.Graph()

    for i, (symbol, pos) in enumerate(atoms):
        atomic_number = DATA.s2n.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

        G.add_node(
            i,
            symbol=symbol,
            atomic_number=atomic_number,
            position=pos,
            charges={"mulliken": mulliken_charges[i]},
        )

    # Add edges (filter by threshold)
    bonds_added = 0
    for i, j, mayer_bo in bonds:
        if mayer_bo >= bond_threshold:
            pos_i = np.array(atoms[i][1])
            pos_j = np.array(atoms[j][1])
            distance = float(np.linalg.norm(pos_i - pos_j))

            si = G.nodes[i]["symbol"]
            sj = G.nodes[j]["symbol"]

            G.add_edge(
                i,
                j,
                bond_order=float(mayer_bo),
                distance=distance,
                bond_type=(si, sj),
                metal_coord=(si in DATA.metals or sj in DATA.metals),
            )
            bonds_added += 1

    logger.debug("Added %d bonds (threshold=%s)", bonds_added, bond_threshold)

    # Compute derived properties
    for node in G.nodes():
        valence = sum(
            G[node][nbr].get("bond_order", 1.0)
            for nbr in G.neighbors(node)
            if G.nodes[nbr]["symbol"] not in DATA.metals
        )
        G.nodes[node]["valence"] = valence

        # Formal charge
        sym = G.nodes[node]["symbol"]
        if sym in DATA.metals:
            formal_charge = 0
        else:
            V = DATA.electrons.get(sym, 0)
            if V == 0:
                formal_charge = 0
            elif sym == "H":
                formal_charge = int(V - valence)
            else:
                B = 2 * valence
                target = 8
                L = max(0, target - B)
                formal_charge = round(V - L - B / 2)

        G.nodes[node]["formal_charge"] = formal_charge

        # Aggregated charge (include H neighbors)
        agg_charge = mulliken_charges[node]
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg_charge += mulliken_charges[nbr]
        G.nodes[node]["agg_charge"] = agg_charge

    # Graph metadata
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "orca",
        "source_file": orca_file,
        "bond_threshold": bond_threshold,
    }
    G.graph["total_charge"] = charge
    G.graph["multiplicity"] = multiplicity
    G.graph["method"] = "orca"

    logger.debug("Final graph: %d atoms, %d bonds", G.number_of_nodes(), G.number_of_edges())

    return G
