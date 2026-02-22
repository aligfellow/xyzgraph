"""Enumerate candidate NCI pairs from detected sites and pi-systems."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .thresholds import NCIThresholds

logger = logging.getLogger(__name__)

# Minimum topological distance (bonds) for non-covalent interactions.
_MIN_GRAPH_DIST = 4


def enumerate_pairs(
    G: nx.Graph,
    sites: dict[str, list[int]],
    pi_rings: list[tuple[int, ...]],
    pi_domains: list[tuple[int, ...]],
    thr: NCIThresholds,
) -> dict[str, list]:
    """Generate all candidate NCI pairs, filtered by type and bonding."""
    pairs: dict[str, list] = {
        "HB": [],
        "XB": [],
        "ChB": [],
        "PnB": [],
        "PIPI": [],
        "CATPI": [],
        "ANPI": [],
        "IONIC": [],
        "CHPI": [],
        "HBPI": [],
        "HALPI": [],
        "CATLP": [],
    }

    donors = sites["hbond_donors"]
    accs = sites["hbond_acceptors"]
    lp = sites["lp_donors"]
    hal = sites["halogen_atoms"]
    chalc = sites["chalcogen_atoms"]
    pnic = sites["pnictogen_atoms"]
    cations = sites["cationic_atoms"]
    anions = sites["anionic_atoms"]

    hb_set: set[tuple[int, int]] = set()

    # HB: donor-acceptor pairs (not bonded)
    for d in donors:
        for a in accs:
            if d != a and not G.has_edge(d, a):
                hb_set.add((d, a))
        for a in lp:
            if d != a and not G.has_edge(d, a):
                hb_set.add((d, a))
    pairs["HB"] = list(hb_set)

    # HB-pi: donor H to pi-system
    for d in donors:
        if d in cations:
            continue  # handled by cation-pi
        for n in G.neighbors(d):
            if G.nodes[n]["symbol"] != "H":
                continue
            h = n
            for ring in pi_rings:
                if d not in ring and h not in ring:
                    pairs["HBPI"].append(((d, h), ring))
            for domain in pi_domains:
                if d not in domain and h not in domain and len(domain) >= thr.hbpi_min_pi_atoms:
                    if any(a in domain for a in lp) or any(a in domain for a in accs):
                        continue
                    pairs["HBPI"].append(((d, h), domain))

    # Halogen bonds
    for x in hal:
        for a in accs:
            if x != a and not G.has_edge(x, a):
                pairs["XB"].append((x, a))

    # Chalcogen bonds
    for y in chalc:
        for a in lp:
            if y != a and not G.has_edge(y, a):
                pairs["ChB"].append((y, a))

    # Pnictogen bonds (min graph distance to exclude metal-bridged ligands)
    pnb_set: set[tuple[int, int]] = set()
    for p in pnic:
        for a in lp:
            if p == a:
                continue
            try:
                gdist = nx.shortest_path_length(G, p, a)
            except nx.NetworkXNoPath:
                gdist = _MIN_GRAPH_DIST
            if gdist >= _MIN_GRAPH_DIST:
                pnb_set.add((p, a))
        for a in accs:
            if p == a:
                continue
            try:
                gdist = nx.shortest_path_length(G, p, a)
            except nx.NetworkXNoPath:
                gdist = _MIN_GRAPH_DIST
            if gdist >= _MIN_GRAPH_DIST:
                pnb_set.add((p, a))
    pairs["PnB"] = list(pnb_set)

    # Halogen-pi
    for x in hal:
        for ring in pi_rings:
            if x not in ring:
                pairs["HALPI"].append((x, ring))
        for domain in pi_domains:
            if x in domain:
                continue
            if len(domain) == 2 and any(a in lp or a in accs for a in domain):
                continue
            symbols = [G.nodes[a]["symbol"] for a in domain]
            if len(domain) in (3, 4) and sum(1 for s in symbols if s == "C") < 2:
                continue
            pairs["HALPI"].append((x, domain))

    # Ionic (min graph distance to exclude ring neighbors)
    for c in cations:
        for a in anions:
            if c == a:
                continue
            try:
                gdist = nx.shortest_path_length(G, c, a)
            except nx.NetworkXNoPath:
                gdist = _MIN_GRAPH_DIST
            if gdist >= _MIN_GRAPH_DIST:
                pairs["IONIC"].append((c, a))

    # Pi-pi (all combinations of rings + domains, excluding metal-bridged pairs)
    all_pi = list(pi_rings) + list(pi_domains)
    for i in range(len(all_pi)):
        for j in range(i + 1, len(all_pi)):
            si, sj = set(all_pi[i]), set(all_pi[j])
            if not si.isdisjoint(sj):
                continue
            # Exclude pairs that share a common neighbor (e.g. ferrocene Cp rings via Fe)
            nbrs_i = {n for a in si for n in G.neighbors(a)} - si
            nbrs_j = {n for a in sj for n in G.neighbors(a)} - sj
            if nbrs_i & nbrs_j:
                continue
            pairs["PIPI"].append((all_pi[i], all_pi[j]))

    # Cation-pi (exclude if cation is bonded to any ring atom)
    for c in cations:
        c_nbrs = set(G.neighbors(c))
        for ring in pi_rings:
            if c not in ring and c_nbrs.isdisjoint(ring):
                pairs["CATPI"].append((c, ring))

    # Anion-pi (exclude if anion is bonded to any ring atom)
    for a in anions:
        a_nbrs = set(G.neighbors(a))
        for ring in pi_rings:
            if a not in ring and a_nbrs.isdisjoint(ring):
                pairs["ANPI"].append((a, ring))

    # Cation-lone pair (no H on cation, min graph distance to exclude ring neighbors)
    for c in cations:
        if any(G.nodes[n]["symbol"] == "H" for n in G.neighbors(c)):
            continue
        for lp_atom in lp:
            if lp_atom in anions or lp_atom in cations:
                continue
            if c == lp_atom:
                continue
            try:
                gdist = nx.shortest_path_length(G, c, lp_atom)
            except nx.NetworkXNoPath:
                gdist = _MIN_GRAPH_DIST  # disconnected fragments are fine
            if gdist >= _MIN_GRAPH_DIST:
                pairs["CATLP"].append((c, lp_atom))

    # CH-pi
    pi_atoms = {a for ring in pi_rings for a in ring} | {a for dom in pi_domains for a in dom}
    for c in range(G.number_of_nodes()):
        if G.nodes[c]["symbol"] != "C" or c in pi_atoms:
            continue
        for n in G.neighbors(c):
            if G.nodes[n]["symbol"] != "H":
                continue
            h = n
            for ring in pi_rings:
                if c not in ring:
                    pairs["CHPI"].append(((c, h), ring))
            for domain in pi_domains:
                if c not in domain and (len(domain) >= thr.chpi_min_pi_atoms or thr.chpi_detailed_mode):
                    if any(a in domain for a in lp) or any(a in domain for a in accs):
                        continue
                    pairs["CHPI"].append(((c, h), domain))

    logger.debug(
        "Pairs: HB=%d XB=%d ChB=%d PnB=%d PIPI=%d CATPI=%d CHPI=%d IONIC=%d HALPI=%d CATLP=%d ANPI=%d HBPI=%d",
        len(pairs["HB"]),
        len(pairs["XB"]),
        len(pairs["ChB"]),
        len(pairs["PnB"]),
        len(pairs["PIPI"]),
        len(pairs["CATPI"]),
        len(pairs["CHPI"]),
        len(pairs["IONIC"]),
        len(pairs["HALPI"]),
        len(pairs["CATLP"]),
        len(pairs["ANPI"]),
        len(pairs["HBPI"]),
    )
    return pairs
