"""Utility functions."""

import logging
from collections import Counter, deque
from typing import List, Optional, Tuple

import networkx as nx

from .data_loader import BOHR_TO_ANGSTROM, DATA

PREF_CHARGE_ORDER = ["gasteiger", "mulliken", "gasteiger_raw"]


def smallest_rings(G: nx.Graph) -> List[List[int]]:
    """SSSR-like smallest set of rings via shortest-cycle-per-edge BFS.

    For each edge ``(u, v)``, find the shortest cycle containing it (BFS in
    ``G {(u,v)}`` from ``u`` to ``v``).  Deduplicate by canonical rotation.
    Returns chemically natural smallest rings, avoiding larger ring artefacts.

    Two orders of magnitude faster than ``nx.minimum_cycle_basis``
    """
    n = G.number_of_nodes()
    if n == 0 or G.number_of_edges() == 0:
        return []

    nodes = list(G.nodes())
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    adj: List[List[int]] = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []
    for a, b in G.edges():
        ia, ib = node_idx[a], node_idx[b]
        adj[ia].append(ib)
        adj[ib].append(ia)
        edges.append((ia, ib))

    seen: set = set()
    rings: List[List[int]] = []
    # `parent` is allocated once and reused; `touched` tracks which entries
    # were set this BFS so we reset only those (avoids an O(N) blind reset
    # loop per edge — wins on large graphs with small-radius BFS).
    parent = [-1] * n

    for u, v in edges:
        parent[u] = u  # root sentinel
        touched = [u]
        q = deque([u])
        while q:
            x = q.popleft()
            if x == v:
                break
            for nb in adj[x]:
                if nb == v and x == u:
                    continue  # skip the direct edge
                if parent[nb] != -1:
                    continue
                parent[nb] = x
                touched.append(nb)
                q.append(nb)
        if parent[v] != -1:
            path = [v]
            cur = parent[v]
            while cur != u:
                path.append(cur)
                cur = parent[cur]
            path.append(u)
            m = min(range(len(path)), key=lambda i: path[i])
            rot = path[m:] + path[:m]
            if len(rot) > 2 and rot[1] > rot[-1]:
                rot = [rot[0], *rot[:0:-1]]
            key = tuple(rot)
            if key not in seen:
                seen.add(key)
                rings.append([nodes[i] for i in rot])
        for i in touched:
            parent[i] = -1
    return rings


def compute_formula(G: nx.Graph) -> None:
    """Compute chemical formula and element counts on a molecular graph.

    Sets ``G.graph["_element_counts"]`` and ``G.graph["formula"]`` using the
    Hill system (C first, then H, then remaining elements alphabetically).
    If ``_element_counts`` is already present it is reused rather than
    recomputed.
    """
    if "_element_counts" in G.graph:
        element_counts = G.graph["_element_counts"]
    else:
        element_counts = dict(Counter(d["symbol"] for _, d in G.nodes(data=True)))
        G.graph["_element_counts"] = element_counts

    formula_parts: list[str] = []
    if "C" in element_counts:
        formula_parts.append(f"C{element_counts['C']}" if element_counts["C"] > 1 else "C")
    if "H" in element_counts:
        formula_parts.append(f"H{element_counts['H']}" if element_counts["H"] > 1 else "H")
    for elem in sorted(element_counts.keys()):
        if elem not in ("C", "H"):
            formula_parts.append(f"{elem}{element_counts[elem]}" if element_counts[elem] > 1 else elem)
    G.graph["formula"] = "".join(formula_parts)


def graph_to_dict(G: nx.Graph) -> dict:
    """Convert molecular graph to dictionary for JSON serialization.

    Useful for generating test fixtures or exporting graph data.

    Parameters
    ----------
    G : nx.Graph
        Molecular graph to convert.

    Returns
    -------
    dict
        Dictionary with keys:
        - graph: graph-level attributes (method, total_charge, etc.)
        - nodes: list of node dicts (id, symbol, formal_charge, etc.)
        - edges: list of edge dicts (idx1, idx2, bond_order, distance)

    Notes
    -----
    The 'position' attribute is converted from tuple to list for JSON compatibility.
    """
    nodes = []
    for n, d in G.nodes(data=True):
        node_dict = {"id": n}
        for k, v in d.items():
            if k == "position":
                # Convert tuple to list for JSON serialization
                node_dict[k] = list(v)
            else:
                node_dict[k] = v
        nodes.append(node_dict)

    edges = []
    for i, j, d in G.edges(data=True):
        edge_dict = {"idx1": i, "idx2": j}
        edge_dict.update(d)
        edges.append(edge_dict)

    # Filter out internal/derived keys from graph attributes
    # - _ prefixed: internal caches (_rings before rename, _element_counts, etc.)
    # - ligand_classification: derived from formal charges, redundant in JSON
    # - build_log: debug info, not molecule data
    # - ncis: serialized separately below via NCIData.to_dict()
    exclude_keys = {"ligand_classification", "build_log", "ncis"}
    graph_attrs = {k: v for k, v in G.graph.items() if not k.startswith("_") and k not in exclude_keys}

    result = {
        "graph": graph_attrs,
        "nodes": nodes,
        "edges": edges,
    }

    if "ncis" in G.graph:
        result["ncis"] = [nci.to_dict() for nci in G.graph["ncis"]]

    return result


def configure_debug_logging():
    """Enable DEBUG-level console output for the xyzgraph package."""
    pkg_logger = logging.getLogger("xyzgraph")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)


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
        sym = data.get("symbol")
        if sym != "H":
            keep.append(n)
            continue

        # Check if this hydrogen is explicitly requested to be shown
        if n in show_h_set:
            keep.append(n)
            continue

        # Hydrogen: inspect neighbors
        nbrs = list(G.neighbors(n))
        if not nbrs:
            # isolated H (rare) - show (could be a problem)
            keep.append(n)
            continue
        # Hide C-H protons
        if all(G.nodes[nbr].get("symbol") == "C" for nbr in nbrs):
            continue
        keep.append(n)
    return keep


# -----------------------------
# Debug (tabular) representation
# -----------------------------
def graph_debug_report(G: nx.Graph, include_h: bool = False, show_h_indices: Optional[List[int]] = None) -> str:
    """Generate debug listing of molecular graph.

    Optionally hides hydrogens / C-H bonds if include_h=False.
    Valence shown is the full valence (including hidden H contributions).

    Parameters
    ----------
    G : nx.Graph
        Molecular graph to report.
    include_h : bool, default=False
        If True, show all hydrogen atoms.
    show_h_indices : list of int, optional
        List of specific hydrogen atom indices to show.

    Returns
    -------
    str
        Formatted debug report.
    """
    lines = []
    lines.append(f"# Molecular Graph: {G.number_of_nodes()} atoms, {G.number_of_edges()} bonds")
    if "total_charge" in G.graph or "multiplicity" in G.graph:
        # gather charge sums (handle missing charges dict)
        def sum_method(m):
            return sum(d.get("charges", {}).get(m, 0.0) for _, d in G.nodes(data=True))

        reported = None
        for m in PREF_CHARGE_ORDER:
            if any(m in d.get("charges", {}) for _, d in G.nodes(data=True)):
                reported = (m, sum_method(m))
                break
        raw_sum = None
        if any("gasteiger_raw" in d.get("charges", {}) for _, d in G.nodes(data=True)):
            raw_sum = ("gasteiger_raw", sum_method("gasteiger_raw"))
        meta = []
        if "total_charge" in G.graph:
            meta.append(f"total_charge={G.graph['total_charge']}")
        if "multiplicity" in G.graph:
            meta.append(f"multiplicity={G.graph['multiplicity']}")
        if reported:
            meta.append(f"sum({reported[0]})={reported[1]:+.3f}")
        if raw_sum and reported and raw_sum[0] != reported[0]:
            meta.append(f"sum({raw_sum[0]})={raw_sum[1]:+.3f}")
        lines.append("# " + "  ".join(meta))
    if not include_h:
        lines.append("# (C-H hydrogens hidden; heteroatom-bound hydrogens shown; valences still include all H)")
    lines.append("# [idx] Sym  val=.. metal=.. formal=.. | neighbors: idx(order / aromatic flag)")
    lines.append("# (val = organic valence excluding metal bonds; metal = metal coordination bonds)")
    arom_edges = {tuple(sorted((i, j))) for i, j, d in G.edges(data=True) if 1.4 < d.get("bond_order", 1.0) < 1.6}
    visible = set(_visible_nodes(G, include_h, show_h_indices))
    for idx, data in G.nodes(data=True):
        if idx not in visible:
            continue
        # Calculate organic valence (excluding metal bonds) and metal valence separately
        organic_val = 0.0
        metal_val = 0.0
        for n in G.neighbors(idx):
            bo = G.edges[idx, n].get("bond_order", 1.0)
            if G.nodes[n]["symbol"] in DATA.metals:
                metal_val += bo
            else:
                organic_val += bo

        formal = data.get("formal_charge", 0)
        # Format formal charge: " 0" for zero, "+1"/"-1" for non-zero
        formal_str = f"{formal} " if formal == 0 else f"{formal:+d}"
        nbrs = []
        for n in sorted(G.neighbors(idx)):
            if n not in visible:
                continue
            bo = G.edges[idx, n].get("bond_order", 1.0)
            arom = "*" if tuple(sorted((idx, n))) in arom_edges else ""
            nbrs.append(f"{n}({bo:.2f}{arom})")
        lines.append(
            f"[{idx:>3}] {data.get('symbol', '?'):>2}  val={organic_val:.2f}  metal={metal_val:.2f}  "
            f"formal={formal_str} | " + (" ".join(nbrs) if nbrs else "-")
        )
    # Edge summary (filtered)
    lines.append("")
    lines.append("# Bonds (i-j: order) (filtered)")

    if G.number_of_edges() > 0:
        max_idx = max(max(i, j) for i, j, d in G.edges(data=True))
        idx_width = max(2, len(str(max_idx)))
        for i, j, d in sorted(G.edges(data=True)):
            if i in visible and j in visible:
                lines.append(f"[{i:>{idx_width}}-{j:>{idx_width}}]: {d.get('bond_order', 1.0):>4.2f}")
    else:
        lines.append("# (no bonds detected)")

    # Stereo section — read from unified G.graph["stereo"]
    stereo = G.graph.get("stereo")
    if stereo and any(stereo.values()):
        lines.append("")
        lines.append("# Stereochemistry")
        for entry in stereo.get("point", []):
            idx = entry["atom"]
            sym = G.nodes[idx].get("symbol", "?")
            lines.append(f"#   [{idx:>3}] {sym}: {entry['label']}")
        for entry in stereo.get("ez", []):
            i, j = entry["bond"]
            lines.append(f"#   [{i:>3}={j:>3}]: {entry['label']}")
        for key in ("axial", "helical"):
            for entry in stereo.get(key, []):
                i, j = entry["atoms"]
                lines.append(f"#   [{i:>3}-{j:>3}]: {entry['label']} ({key})")
        for entry in stereo.get("planar", []):
            ring_str = ",".join(str(a) for a in entry["ring"])
            lines.append(f"#   [{ring_str}]: {entry['label']} (planar)")

    return "\n".join(lines)


def count_frames_and_atoms(filepath: str) -> tuple[int, int]:
    """Count frames and atoms in an XYZ trajectory file.

    Parameters
    ----------
    filepath : str
        Path to XYZ file.

    Returns
    -------
    tuple[int, int]
        (num_frames, num_atoms_per_frame)
    """
    with open(filepath, "r") as f:
        lines = f.read().rstrip().splitlines()

    if not lines:
        raise ValueError("Empty XYZ file")
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("Invalid XYZ format: first line should be atom count") from None

    frame_size = num_atoms + 2

    if len(lines) % frame_size != 0:
        raise ValueError(f"File has {len(lines)} lines, not evenly divisible by frame size {frame_size}")

    return len(lines) // frame_size, num_atoms


def read_xyz_file(
    filepath: str, bohr_units: bool = False, frame: int = 0
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Read XYZ file and return list of (symbol, (x, y, z)) for specified frame.

    Supports single and multi-frame (trajectory) files. Streams to requested frame.
    """
    num_frames, num_atoms = count_frames_and_atoms(filepath)

    if frame < 0 or frame >= num_frames:
        raise ValueError(f"Frame {frame} out of range. File has {num_frames} frame(s).")

    start_line = frame * (num_atoms + 2)

    with open(filepath, "r") as f:
        for _ in range(start_line):
            f.readline()
        f.readline()  # Skip header
        f.readline()  # Skip comment

        atoms = []
        for i in range(num_atoms):
            parts = f.readline().strip().split()
            if len(parts) < 4:
                raise ValueError(f"Frame {frame}, atom {i}: expected at least 4 columns")

            elem = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError as e:
                raise ValueError(f"Frame {frame}, atom {i}: invalid coordinates") from e

            # Convert atomic number to symbol if needed
            if elem.isdigit():
                atomic_num = int(elem)
                if atomic_num not in DATA.n2s:
                    raise ValueError(f"Frame {frame}, atom {i}: unknown atomic number {atomic_num}")
                symbol = DATA.n2s[atomic_num]
            else:
                symbol = elem

            if symbol not in DATA.s2n:
                raise ValueError(f"Frame {frame}, atom {i}: unknown element symbol '{symbol}'")

            if bohr_units:
                x, y, z = (
                    x * BOHR_TO_ANGSTROM,
                    y * BOHR_TO_ANGSTROM,
                    z * BOHR_TO_ANGSTROM,
                )

            atoms.append((symbol, (x, y, z)))

    return atoms


def _parse_pairs(arg_value: str):
    """Parse '--bond "i,j a,b"' or '--unbond "i,j a,b"' into [(i,j), (a,b)]."""
    pairs = []
    for pair_str in arg_value.split():
        i_str, j_str = pair_str.split(",")
        pairs.append((int(i_str), int(j_str)))
    return pairs
