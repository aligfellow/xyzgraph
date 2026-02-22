"""NCI display formatting and ASCII rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from .interaction import NCIData


def _format_site(G: nx.Graph, site: tuple[int, ...]) -> str:
    """Format a site tuple as element+index string."""
    if len(site) == 1:
        return f"{G.nodes[site[0]]['symbol']}{site[0]}"
    syms = [f"{G.nodes[i]['symbol']}{i}" for i in site]
    return f"[{','.join(syms)}]"


def format_nci_table(G: nx.Graph, ncis: list[NCIData], debug: bool = False) -> str:
    """Format NCI detection results as a summary table.

    Returns the full text block including header, interaction lines,
    and pi-centroid note.
    """
    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append("# Non-Covalent Interactions")
    lines.append("=" * 80)

    if not ncis:
        lines.append("\n  No non-covalent interactions detected.\n")
        return "\n".join(lines)

    has_pi = any(len(nci.site_a) > 1 or len(nci.site_b) > 1 for nci in ncis)

    lines.append(f"\n  {len(ncis)} interaction(s) detected:\n")
    for nci in ncis:
        site_a_str = _format_site(G, nci.site_a)
        site_b_str = _format_site(G, nci.site_b)
        aux_str = ""
        if nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            aux_syms = [f"{G.nodes[a]['symbol']}{a}" for a in nci.aux_atoms]
            aux_str = f" via {','.join(aux_syms)}"
        line = f"  {nci.type:<22s}  {site_a_str} ... {site_b_str}{aux_str}"
        if debug:
            geom_parts = [f"{k}={v:.2f}" for k, v in nci.geometry.items()]
            line += f"  ({', '.join(geom_parts)})"
        lines.append(line)

    if has_pi:
        lines.append("\n  * Pi-system centroids shown as '*' in ASCII depiction.")
    lines.append("")

    return "\n".join(lines)


def render_nci_ascii(
    G: nx.Graph,
    ncis: list[NCIData],
    scale: float = 3.0,
    include_h: bool = False,
    show_h_indices: list[int] | None = None,
) -> str:
    """Render ASCII depiction with NCI dotted lines and pi centroids.

    Temporarily mutates *G* (adds NCI edges and centroid dummy nodes)
    for layout, then cleans up before returning.
    """
    from xyzgraph.ascii_renderer import graph_to_ascii

    if not ncis:
        header = f"{'=' * 80}\n# ASCII Depiction\n{'=' * 80}\n"
        ascii_out, _ = graph_to_ascii(
            G, scale=scale, include_h=include_h, show_h_indices=show_h_indices,
        )
        return header + "\n" + ascii_out + "\n"

    # Partition into atom-atom and pi-system NCIs
    atom_atom = [n for n in ncis if len(n.site_a) == 1 and len(n.site_b) == 1]
    pi_ncis = [n for n in ncis if n not in atom_atom]

    # Collect H atoms involved in NCI display
    nci_show_h: set[int] = set()
    for nci in ncis:
        if nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            nci_show_h.add(nci.aux_atoms[0])

    h_indices = list(nci_show_h)
    if show_h_indices:
        h_indices = list(set(h_indices) | set(show_h_indices))

    # --- Graph mutation (cleaned up at the end) ---
    nci_edges: list[tuple[int, int]] = []
    layout_hints: list[tuple[int, int]] = []

    # Add atom-atom NCI edges for layout
    for nci in atom_atom:
        if nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            a, b = nci.aux_atoms[0], nci.site_b[0]
        else:
            a, b = nci.site_a[0], nci.site_b[0]
        if not G.has_edge(a, b):
            G.add_edge(a, b, bond_order=0.0, NCI=True)
            nci_edges.append((a, b))

    # Temp layout-hint edges for pi NCIs
    for nci in pi_ncis:
        if len(nci.site_a) == 1 and len(nci.site_b) > 1:
            has_h_aux = nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H"
            src = nci.aux_atoms[0] if has_h_aux else nci.site_a[0]
            tgt = nci.site_b[0]
        elif len(nci.site_b) == 1 and len(nci.site_a) > 1:
            src, tgt = nci.site_b[0], nci.site_a[0]
        else:
            continue
        if not G.has_edge(src, tgt):
            G.add_edge(src, tgt, bond_order=0.0, NCI=True)
            layout_hints.append((src, tgt))

    # Compute 2D layout (NCI + hint edges present for better placement)
    _, layout = graph_to_ascii(
        G, scale=scale, include_h=include_h, show_h_indices=h_indices or None,
    )

    # Remove layout hints
    for a, b in layout_hints:
        if G.has_edge(a, b):
            G.remove_edge(a, b)

    # Add centroid dummy nodes for pi-system NCIs
    centroid_nodes: dict[tuple[int, ...], int] = {}
    ref_layout = dict(layout)
    next_id = max(G.nodes()) + 1

    for nci in pi_ncis:
        for site in [nci.site_a, nci.site_b]:
            if len(site) > 1:
                key = tuple(sorted(site))
                if key not in centroid_nodes:
                    pos_2d = [layout[i] for i in site if i in layout]
                    if pos_2d:
                        cx = sum(p[0] for p in pos_2d) / len(pos_2d)
                        cy = sum(p[1] for p in pos_2d) / len(pos_2d)
                        cid = next_id
                        next_id += 1
                        G.add_node(cid, symbol="*", position=[0.0, 0.0, 0.0])
                        centroid_nodes[key] = cid
                        ref_layout[cid] = (cx, cy)

    for nci in pi_ncis:
        key_a = tuple(sorted(nci.site_a)) if len(nci.site_a) > 1 else None
        key_b = tuple(sorted(nci.site_b)) if len(nci.site_b) > 1 else None
        if nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            node_a = nci.aux_atoms[0]
        else:
            node_a = centroid_nodes[key_a] if key_a else nci.site_a[0]
        node_b = centroid_nodes[key_b] if key_b else nci.site_b[0]
        if not G.has_edge(node_a, node_b):
            G.add_edge(node_a, node_b, bond_order=0.0, NCI=True)
            nci_edges.append((node_a, node_b))

    # Render
    title = "# ASCII Depiction (with NCI dotted lines)"
    if centroid_nodes:
        title += "  (* = centroid)"
    header = f"{'=' * 80}\n{title}\n{'=' * 80}\n"

    ascii_out, _ = graph_to_ascii(
        G, scale=scale, include_h=include_h,
        show_h_indices=h_indices or None,
        reference_layout=ref_layout,
    )

    # Clean up temp edges and centroid nodes
    for a, b in nci_edges:
        if G.has_edge(a, b):
            G.remove_edge(a, b)
    for cid in centroid_nodes.values():
        G.remove_node(cid)

    return header + "\n" + ascii_out + "\n"
