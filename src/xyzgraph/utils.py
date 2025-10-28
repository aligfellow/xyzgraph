import networkx as nx
from typing import List, Optional, Dict, Any, Tuple
from .data_loader import DATA, BOHR_TO_ANGSTROM

PREF_CHARGE_ORDER = ['gasteiger', 'mulliken', 'gasteiger_raw']

def _pick_charge(d):
    for k in PREF_CHARGE_ORDER:
        if k in d.get('charges', {}):
            return d['charges'][k]
    return next(iter(d.get('charges', {}).values()), 0.0)

def _visible_nodes(G: nx.Graph, include_h: bool, show_h_indices: Optional[List[int]] = None) -> List[int]:
    """
    Return node indices to display.
    
    Parameters
    ----------
    G : nx.Graph
        Molecular graph with node attributes including 'symbol'
    include_h : bool
        If True, show all hydrogen atoms
    show_h_indices : Optional[List[int]], default=None
        List of specific hydrogen atom indices to show, overriding the default
        hiding behavior. Useful for highlighting specific hydrogens in transition
        states or other special cases. If include_h is True, this parameter is ignored.
    
    Returns
    -------
    List[int]
        List of node indices to display
    
    Notes
    -----
    If include_h is False, hydrogens bound solely to carbon (typical C-H hydrogens)
    are hidden by default. Hydrogens attached to any heteroatom (O, N, S, etc.) are
    retained. The show_h_indices parameter can override this to show specific C-H
    hydrogens when needed.
    """
    if include_h:
        return list(G.nodes())
    
    # Convert show_h_indices to set for efficient lookup
    show_h_set = set(show_h_indices) if show_h_indices else set()
    
    keep = []
    for n, data in G.nodes(data=True):
        sym = data.get('symbol')
        if sym != 'H':
            keep.append(n)
            continue
        
        # Check if this hydrogen is explicitly requested to be shown
        if n in show_h_set:
            keep.append(n)
            continue
        
        # Hydrogen: inspect neighbors
        nbrs = list(G.neighbors(n))
        if not nbrs:
            # isolated H (rare) – show (could be a problem)
            keep.append(n)
            continue
        # Hide C-H protons
        if all(G.nodes[nbr].get('symbol') == 'C' for nbr in nbrs):
            continue  
        keep.append(n)
    return keep

# -----------------------------
# Debug (tabular) representation
# -----------------------------
def graph_debug_report(G: nx.Graph, include_h: bool = False, show_h_indices: Optional[List[int]] = None) -> str:
    """
    Debug listing (optionally hides hydrogens / C–H bonds if include_h=False).
    Valence shown is the full valence (including hidden H contributions).
    
    Parameters
    ----------
    G : nx.Graph
        Molecular graph to report
    include_h : bool, default=False
        If True, show all hydrogen atoms
    show_h_indices : Optional[List[int]], default=None
        List of specific hydrogen atom indices to show
    
    Returns
    -------
    str
        Formatted debug report
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
    lines.append("# [idx] Sym  val=.. metal=.. formal=.. chg=.. agg=.. | neighbors: idx(order / aromatic flag)")
    lines.append("# (val = organic valence excluding metal bonds; metal = metal coordination bonds)")
    arom_edges = {tuple(sorted((i,j))) for i,j,d in G.edges(data=True)
                  if 1.4 < d.get('bond_order',1.0) < 1.6}
    visible = set(_visible_nodes(G, include_h, show_h_indices))
    for idx,data in G.nodes(data=True):
        if idx not in visible:
            continue
        # Calculate organic valence (excluding metal bonds) and metal valence separately
        organic_val = 0.0
        metal_val = 0.0
        for n in G.neighbors(idx):
            bo = G.edges[idx,n].get('bond_order',1.0)
            if G.nodes[n]['symbol'] in DATA.metals:
                metal_val += bo
            else:
                organic_val += bo
        
        chg = _pick_charge(data)
        agg = data.get('agg_charge', chg)
        formal = data.get('formal_charge', 0)
        # Format formal charge: " 0" for zero, "+1"/"-1" for non-zero
        formal_str = f"{formal} " if formal == 0 else f"{formal:+d}"
        nbrs = []
        for n in sorted(G.neighbors(idx)):
            if n not in visible:
                continue
            bo = G.edges[idx,n].get('bond_order',1.0)
            arom = '*' if tuple(sorted((idx,n))) in arom_edges else ''
            nbrs.append(f"{n}({bo:.2f}{arom})")
        lines.append(f"[{idx:>3}] {data.get('symbol','?'):>2}  val={organic_val:.2f}  metal={metal_val:.2f}  "
                     f"formal={formal_str}  chg={chg:+.3f}  agg={agg:+.3f} | " +
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


def read_xyz_file(filepath: str, bohr_units: bool = False) -> List[Tuple[str, Tuple[float, float, float]]]:
    """
    Read XYZ file and return list of (symbol, (x, y, z)).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"Invalid XYZ format: first line should be atom count")

    # Skip comment line
    atom_lines = lines[2:2+num_atoms]

    atoms = []
    for i, line in enumerate(atom_lines):
        parts = line.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Line {i+3}: expected at least 4 columns")

        element_or_symbol = parts[0].strip()
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as e:
            raise ValueError(f"Line {i+3}: invalid coordinates") from e

        # Determine symbol - if it's an atomic number, convert to symbol
        if element_or_symbol.isdigit():
            atomic_num = int(element_or_symbol)
            if atomic_num in DATA.n2s:
                symbol = DATA.n2s[atomic_num]
            else:
                raise ValueError(f"Line {i+3}: unknown atomic number {atomic_num}")
        else:
            # Assume it's already a symbol
            symbol = element_or_symbol

        if symbol not in DATA.s2n:
            raise ValueError(f"Line {i+3}: unknown element symbol '{symbol}'")

        # Convert Bohr to Angstrom if needed
        if bohr_units:
            x *= BOHR_TO_ANGSTROM
            y *= BOHR_TO_ANGSTROM
            z *= BOHR_TO_ANGSTROM

        atoms.append((symbol, (x, y, z)))

    if len(atoms) != num_atoms:
        raise ValueError(f"Expected {num_atoms} atoms, found {len(atoms)}")

    return atoms

def _parse_pairs(arg_value: str):
    """
    Parse '--bond "i,j a,b"' or '--unbond "i,j a,b"' into [(i,j), (a,b)].
    """
    pairs = []
    for pair_str in arg_value.split():
        i_str, j_str = pair_str.split(",")
        pairs.append((int(i_str), int(j_str)))
    return pairs
