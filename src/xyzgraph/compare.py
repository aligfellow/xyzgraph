"""Graph comparison utilities."""

from typing import Optional

import networkx as nx
from rdkit import Chem

from .ascii_renderer import graph_to_ascii
from .utils import graph_debug_report


def _bond_order_from_rdkit(bond) -> float:
    """Convert RDKit bond to numeric order."""
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
        return 1.5
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return 1.0
    elif bt == Chem.BondType.DOUBLE:
        return 2.0
    elif bt == Chem.BondType.TRIPLE:
        return 3.0
    return 1.0


def compare_with_rdkit(
    reference_graph: nx.Graph,
    rdkit_graph: Optional[nx.Graph] = None,
    verbose: bool = False,
    ascii: bool = False,
    ascii_scale: float = 2.0,
    ascii_include_h: bool = True,
) -> str:
    """
    Compare a reference graph to an RDKit-built graph.

    Parameters
    ----------
    reference_graph : nx.Graph
        Reference molecular graph to compare against
    rdkit_graph : nx.Graph, optional
        RDKit-built graph. If None, will be built from reference_graph nodes
    verbose : bool, default=False
        Include detailed graph report in output
    ascii : bool, default=False
        Include ASCII visualization in output
    ascii_scale : float, default=2.0
        Scale factor for ASCII visualization
    ascii_include_h : bool, default=True
        Include hydrogen atoms in ASCII visualization

    Returns
    -------
    str
        Formatted comparison report

    Examples
    --------
    >>> from xyzgraph import build_graph, build_graph_rdkit, compare_with_rdkit
    >>> G_cheminf = build_graph("structure.xyz")
    >>> G_rdkit = build_graph_rdkit("structure.xyz")
    >>> print(compare_with_rdkit(G_cheminf, G_rdkit))
    """
    # If no RDKit graph provided, build it from reference graph nodes
    if rdkit_graph is None:
        from .graph_builders import build_graph_rdkit

        atoms = [
            (reference_graph.nodes[i]["symbol"], reference_graph.nodes[i]["position"]) for i in reference_graph.nodes()
        ]
        charge = reference_graph.graph.get("total_charge", 0)
        try:
            rdkit_graph = build_graph_rdkit(atoms, charge=charge)
        except ValueError as e:
            return f"\n{'=' * 60}\nRDKIT COMPARISON\n{'=' * 60}\n# {e!s}\n"

    out = []
    out.append("\n" + "=" * 60)
    out.append("RDKIT (XYZ2MOL) COMPARISON")
    out.append("=" * 60)

    charge = reference_graph.graph.get("total_charge", 0)
    out.append(
        f"# RDKit graph: {rdkit_graph.number_of_nodes()} atoms, {rdkit_graph.number_of_edges()} bonds (charge={charge})"
    )

    # --- ASCII depiction (optionally aligned to reference layout) ---
    if ascii:
        _, layout = graph_to_ascii(
            reference_graph,
            scale=ascii_scale,
            include_h=ascii_include_h,
        )

        ascii_txt, _ = graph_to_ascii(
            rdkit_graph,
            scale=ascii_scale,
            include_h=ascii_include_h,
            reference_layout=layout,
        )
        out.append("# RDKit ASCII (aligned to reference)")
        out.append(ascii_txt)
        out.append("")

    # --- Compare edges (presence + bond-order diffs) ---
    ref_edges = {}
    for i, j, d in reference_graph.edges(data=True):
        ref_edges[tuple(sorted((i, j)))] = float(d.get("bond_order", 1.0))

    rdkit_edges = {}
    for i, j, d in rdkit_graph.edges(data=True):
        rdkit_edges[tuple(sorted((i, j)))] = float(d.get("bond_order", 1.0))

    only_ref = sorted(e for e in ref_edges if e not in rdkit_edges)
    only_rdkit = sorted(e for e in rdkit_edges if e not in ref_edges)
    shared = sorted(e for e in ref_edges if e in rdkit_edges)

    bo_diffs = []
    for e in shared:
        r = ref_edges[e]
        rd = rdkit_edges[e]
        if abs(r - rd) >= 0.25:
            bo_diffs.append((e, r, rd, r - rd))

    out.append(
        f"# Bond differences: only_in_native={len(only_ref):,}   "
        f"only_in_rdkit={len(only_rdkit):,}   bond_order_diffs={len(bo_diffs):,}"
    )

    if only_ref:
        out.append("#   only_in_native: " + " ".join(f"{a}-{b}" for a, b in only_ref))
    if only_rdkit:
        out.append("#   only_in_rdkit: " + " ".join(f"{a}-{b}" for a, b in only_rdkit))
    if bo_diffs:
        out.append("#   bond_order_diffs (Δ≥0.25):")
        for e, r, rd, delta in bo_diffs[:40]:
            a, b = e
            out.append(f"#     {a:>3}-{b:<3}   native={r:>4.2f}   rdkit={rd:>4.2f}   Δ={delta:+6.2f}")

        if len(bo_diffs) > 40:
            out.append("#     ...")

    # --- Verbose report ---
    if verbose:
        out.append(graph_debug_report(rdkit_graph, include_h=ascii_include_h))

    return "\n".join(out) + "\n"
