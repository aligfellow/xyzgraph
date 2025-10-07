import networkx as nx
from typing import List, Optional, Dict, Any

PREF_CHARGE_ORDER = ['gasteiger', 'mulliken', 'hirshfeld', 'gasteiger_raw']

def _pick_charge(d):
    for k in PREF_CHARGE_ORDER:
        if k in d.get('charges', {}):
            return d['charges'][k]
    return next(iter(d.get('charges', {}).values()), 0.0)

def _visible_nodes(G: nx.Graph, include_h: bool) -> List[int]:
    """
    Return node indices to display.
    If include_h is False, hide only hydrogens bound solely to carbon (typical C–H hydrogens).
    Hydrogens attached to any heteroatom (O, N, S, etc.) are retained.
    """
    if include_h:
        return list(G.nodes())
    keep = []
    for n, data in G.nodes(data=True):
        sym = data.get('symbol')
        if sym != 'H':
            keep.append(n)
            continue
        # Hydrogen: inspect neighbors
        nbrs = list(G.neighbors(n))
        if not nbrs:
            # isolated H (rare) – hide it
            continue
        # If every neighbor is carbon, hide; else keep
        if all(G.nodes[nbr].get('symbol') == 'C' for nbr in nbrs):
            continue  # hide C–H hydrogen
        keep.append(n)
    return keep

# -----------------------------
# Debug (tabular) representation
# -----------------------------
def graph_debug_report(G: nx.Graph, include_h: bool = False) -> str:
    """
    Debug listing (optionally hides hydrogens / C–H bonds if include_h=False).
    Valence shown is the full valence (including hidden H contributions).
    """
    lines = []
    lines.append(f"# Molecular Graph: {G.number_of_nodes()} atoms, {G.number_of_edges()} bonds")
    if 'total_charge' in G.graph or 'multiplicity' in G.graph:
        # gather charge sums
        def sum_method(m):
            return sum(d['charges'].get(m,0.0) for _,d in G.nodes(data=True))
        reported = None
        for m in PREF_CHARGE_ORDER:
            if any(m in d['charges'] for _,d in G.nodes(data=True)):
                reported = (m, sum_method(m))
                break
        raw_sum = None
        if any('gasteiger_raw' in d['charges'] for _,d in G.nodes(data=True)):
            raw_sum = ('gasteiger_raw', sum_method('gasteiger_raw'))
        meta = []
        if 'total_charge' in G.graph:
            meta.append(f"total_charge={G.graph['total_charge']}")
        if 'multiplicity' in G.graph:
            meta.append(f"multiplicity={G.graph['multiplicity']}")
        if reported:
            meta.append(f"sum({reported[0]})={reported[1]:+.3f}")
        if raw_sum and reported and raw_sum[0] != reported[0]:
            meta.append(f"sum({raw_sum[0]})={raw_sum[1]:+.3f}")
        lines.append("# " + "  ".join(meta))
    if not include_h:
        lines.append("# (C–H hydrogens hidden; heteroatom-bound hydrogens shown; valences still include all H)")
    lines.append("# [idx] Sym  val=.. chg=.. agg=.. | neighbors: idx(order / aromatic flag)")
    arom_edges = {tuple(sorted((i,j))) for i,j,d in G.edges(data=True)
                  if 1.4 < d.get('bond_order',1.0) < 1.6}
    visible = set(_visible_nodes(G, include_h))
    for idx,data in G.nodes(data=True):
        if idx not in visible:
            continue
        # full valence (all neighbors, including hidden hydrogens)
        full_val = data.get('valence', sum(G.edges[idx,n].get('bond_order',1.0) for n in G.neighbors(idx)))
        chg = _pick_charge(data)
        agg = data.get('agg_charge', chg)
        formal = data.get('formal_charge', 0)
        nbrs = []
        for n in sorted(G.neighbors(idx)):
            if n not in visible:
                # skip listing hidden H neighbor
                continue
            bo = G.edges[idx,n].get('bond_order',1.0)
            arom = '*' if tuple(sorted((idx,n))) in arom_edges else ''
            nbrs.append(f"{n}({bo:.2f}{arom})")
        formal = data.get('formal_charge', 0)
        lines.append(f"[{idx:>2}] {data.get('symbol','?'):>2}  val={full_val:.2f}  "
                     f"formal={formal:+d}  chg={chg:+.3f}  agg={agg:+.3f} | " +
                     (" ".join(nbrs) if nbrs else "-"))
    # Edge summary (filtered)
    lines.append("")
    lines.append("# Bonds (i-j: order) (filtered)")

    max_idx = max(max(i, j) for i, j, d in G.edges(data=True))
    idx_width = max(2, len(str(max_idx)))
    for i,j,d in sorted(G.edges(data=True)):
        if i in visible and j in visible:
            lines.append(f"[{i:>{idx_width}}-{j:>{idx_width}}]: {d.get('bond_order', 1.0):>4.2f}")
    return "\n".join(lines)