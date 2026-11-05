"""NCI site detection: classify atoms as donors, acceptors, etc."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

from xyzgraph.data_loader import DATA

if TYPE_CHECKING:
    from .thresholds import NCIThresholds

logger = logging.getLogger(__name__)


def sigma_reference_atoms(G: nx.Graph, donor: int, kind: str) -> tuple[int, ...]:
    """Return covalent sigma references in neighbour order; XB/HALPI require a heavy reference."""
    return tuple(
        atom
        for atom, edge in G[donor].items()
        if "dative_donor" not in edge
        and not edge.get("metal_coord", False)
        and not edge.get("NCI", False)
        and G.nodes[atom]["symbol"] not in DATA.metals
        and (kind not in ("XB", "HALPI") or G.nodes[atom]["symbol"] != "H")
    )


def detect_sites(G: nx.Graph, thr: NCIThresholds) -> dict[str, list[int]]:
    """Classify NCI sites, accounting for explicit dative donation and terminal hydrides."""
    donors: list[int] = []
    acceptors: list[int] = []
    lp_donors: list[int] = []
    halogens: list[int] = []
    chalcogens: list[int] = []
    pnictogens: list[int] = []
    cationic: list[int] = []
    anionic: list[int] = []
    spent: set[int] = set()

    for i, d in G.nodes(data=True):
        sym = d["symbol"]
        nH = sum(1 for j in G.neighbors(i) if G.nodes[j]["symbol"] == "H")
        n_neighbors = len(list(G.neighbors(i)))

        q = float(d.get("formal_charge", 0))
        valence = float(d.get("valence", 0))
        donated = sum(edge.get("dative_donor") == i for edge in G[i].values())
        if donated:
            covalent = [edge for edge in G[i].values() if "dative_donor" not in edge]
            n_neighbors = len(covalent)
            valence = sum(float(edge.get("bond_order", 1.0)) for edge in covalent)
            if DATA.electrons[sym] - q - valence - 2 * donated < 2:
                spent.add(i)

        # Sigma-hole halogens (XB, HALPI); covalent F has essentially no sigma hole
        if sym in ("Cl", "Br", "I"):
            halogens.append(i)
        # Chalcogens
        if sym in ("S", "Se", "Te"):
            chalcogens.append(i)
        # Pnictogens
        if sym in ("P", "As", "Sb", "Bi"):
            pnictogens.append(i)

        # Cationic/anionic
        if q >= thr.ionic_min_charge:
            cationic.append(i)
        elif q <= -thr.ionic_min_charge:
            anionic.append(i)

        # Lone pair donors
        if sym == "O" and n_neighbors <= 2:
            lp_donors.append(i)
        elif sym == "N" and n_neighbors <= 3:
            lp_donors.append(i)
        elif sym == "S" and n_neighbors <= 2:
            lp_donors.append(i)
        elif sym == "P" and n_neighbors <= 4 and valence < 5:
            lp_donors.append(i)
        elif sym in ("F", "Cl", "Br", "I"):
            lp_donors.append(i)
        elif sym in ("C", "Si") and n_neighbors == 2:
            # Carbene-like
            if all(G.nodes[j]["symbol"] not in ("O", "S") for j in G.neighbors(i)):
                lp_donors.append(i)

        # HB donors
        if sym != "C" and sym not in DATA.metals and nH > 0:
            donors.append(i)

        # HB acceptors
        if sym in ("O", "S", "Se", "Te", "P", "As", "Sb", "Bi"):
            # Bonding every valence electron (S(VI), P(V)) leaves no lone pair to accept with
            if valence < DATA.electrons[sym] and not (q > 0.25):
                acceptors.append(i)
        elif sym == "N":
            if not (q > 0.20):
                acceptors.append(i)
        elif sym in ("F", "Cl", "Br", "I"):
            if not (q > 0.10):
                acceptors.append(i)
        elif sym == "H":
            bonded = [j for j, edge in G[i].items() if not edge.get("NCI") and edge.get("bond_order", 1.0) > 0]
            if len(bonded) == 1 and G.nodes[bonded[0]]["symbol"] in DATA.metals:
                acceptors.append(i)

    # Charges cancelling within a covalent group (nitro, N-oxide) make no ion; a net-charged group (azide)
    # is an ion at its atoms of that sign.
    groups = nx.Graph()
    groups.add_nodes_from(cationic + anionic)
    groups.add_edges_from((i, j) for i in cationic + anionic for j in sigma_reference_atoms(G, i, "ion") if j in groups)
    ions: set[int] = set()
    for group in nx.connected_components(groups):
        net = sum(float(G.nodes[i].get("formal_charge", 0)) for i in group)
        ions |= {i for i in group if net * float(G.nodes[i].get("formal_charge", 0)) > 0}
    cationic = [i for i in cationic if i in ions]
    anionic = [i for i in anionic if i in ions]

    # Hydride acceptance uses the M-H bond pair, rather than an unspent lone pair.
    acceptors = [atom for atom in acceptors if atom not in spent or G.nodes[atom]["symbol"] == "H"]
    lp_donors = [atom for atom in lp_donors if atom not in spent]
    logger.debug(
        "Sites: donors=%d acceptors=%d lp=%d cations=%d anions=%d hal=%d chalc=%d pnic=%d",
        len(donors),
        len(acceptors),
        len(lp_donors),
        len(cationic),
        len(anionic),
        len(halogens),
        len(chalcogens),
        len(pnictogens),
    )
    return {
        "hbond_donors": donors,
        "hbond_acceptors": acceptors,
        "lp_donors": lp_donors,
        "halogen_atoms": halogens,
        "chalcogen_atoms": chalcogens,
        "pnictogen_atoms": pnictogens,
        "cationic_atoms": cationic,
        "anionic_atoms": anionic,
    }
