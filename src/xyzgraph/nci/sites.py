"""NCI site detection: classify atoms as donors, acceptors, etc."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from .thresholds import NCIThresholds

logger = logging.getLogger(__name__)


def detect_sites(G: nx.Graph, thr: NCIThresholds) -> dict[str, list[int]]:
    """Classify atoms into NCI-relevant site categories.

    Uses ``formal_charge`` from the bond order optimizer (always available).
    """
    donors: list[int] = []
    acceptors: list[int] = []
    lp_donors: list[int] = []
    halogens: list[int] = []
    chalcogens: list[int] = []
    pnictogens: list[int] = []
    cationic: list[int] = []
    anionic: list[int] = []

    for i, d in G.nodes(data=True):
        sym = d["symbol"]
        nH = sum(1 for j in G.neighbors(i) if G.nodes[j]["symbol"] == "H")
        n_neighbors = len(list(G.neighbors(i)))

        q = float(d.get("formal_charge", 0))

        # Halogens
        if sym in ("F", "Cl", "Br", "I"):
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
        elif sym == "P" and n_neighbors <= 4 and float(d.get("valence", 0)) < 5:
            lp_donors.append(i)
        elif sym in ("F", "Cl", "Br", "I"):
            lp_donors.append(i)
        elif sym in ("C", "Si") and n_neighbors == 2:
            # Carbene-like
            if all(G.nodes[j]["symbol"] not in ("O", "S") for j in G.neighbors(i)):
                lp_donors.append(i)

        # HB donors: any non-C with at least one H
        if sym != "C" and nH > 0:
            donors.append(i)

        # HB acceptors
        if sym in ("O", "S", "Se", "Te"):
            if not (q > 0.25):
                acceptors.append(i)
        elif sym in ("P", "As", "Sb", "Bi"):
            # Pnictogens only accept if they have a lone pair (valence < 5 means spare electron pair)
            val = float(d.get("valence", 0))
            if val < 5 and not (q > 0.25):
                acceptors.append(i)
        elif sym == "N":
            if not (q > 0.20):
                acceptors.append(i)
        elif sym in ("F", "Cl", "Br", "I"):
            if not (q > 0.10):
                acceptors.append(i)

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
