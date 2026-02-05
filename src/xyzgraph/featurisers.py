"""Optional graph featurisers (require extra dependencies)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


def compute_gasteiger_charges(G: nx.Graph, target_charge: int = 0) -> nx.Graph:
    """Add Gasteiger partial charges to every node.

    Parameters
    ----------
    G : nx.Graph
        Graph produced by any ``build_graph*`` function.
    target_charge : int
        Molecular charge. Raw Gasteiger charges are shifted so that
        the sum matches this value.

    Returns
    -------
    nx.Graph
        Same graph (mutated in-place) with updated node attributes:
        ``charges["gasteiger"]``, ``charges["gasteiger_raw"]``,
        and ``agg_charge``.

    Notes
    -----
    Requires ``rdkit``.  On failure the graph is returned unchanged
    (charges default to 0.0).
    """
    try:
        from rdkit import Chem  # lazy import — RDKit is optional

        rw = Chem.RWMol()
        for i in G.nodes():
            rw.AddAtom(Chem.Atom(G.nodes[i]["symbol"]))

        for i, j, data in G.edges(data=True):
            bo = data["bond_order"]
            if bo >= 2.5:
                bt = Chem.BondType.TRIPLE
            elif bo >= 1.75:
                bt = Chem.BondType.DOUBLE
            elif bo >= 1.25:
                bt = Chem.BondType.AROMATIC
            else:
                bt = Chem.BondType.SINGLE
            rw.AddBond(int(i), int(j), bt)

        mol = rw.GetMol()

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)

        Chem.AllChem.ComputeGasteigerCharges(mol)  # ty: ignore

        raw: list[float] = []
        for atom in mol.GetAtoms():
            try:
                c = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(c):
                    c = 0.0
            except Exception:
                c = 0.0
            raw.append(c)

    except Exception:
        logger.warning("Gasteiger charge calculation failed — charges left unchanged")
        return G

    # Shift so sum matches target_charge
    raw_sum = sum(raw)
    n = G.number_of_nodes()
    delta = (target_charge - raw_sum) / n if n else 0.0
    adj = [c + delta for c in raw]

    # Write into graph
    for node in G.nodes():
        charges = G.nodes[node].get("charges", {})
        charges["gasteiger_raw"] = raw[node]
        charges["gasteiger"] = adj[node]
        G.nodes[node]["charges"] = charges

        # Recompute agg_charge from gasteiger
        agg = adj[node]
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg += adj[nbr]
        G.nodes[node]["agg_charge"] = agg

    return G
