"""NCI display formatting and ASCII rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .graph import build_nci_graph

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

    # Group bifurcated HBs by acceptor for combined display
    bifurcated = [n for n in ncis if n.type == "hbond_bifurcated"]
    non_bifurcated = [n for n in ncis if n.type != "hbond_bifurcated"]
    bif_by_acc: dict[tuple[int, ...], list] = {}
    for nci in bifurcated:
        bif_by_acc.setdefault(nci.site_b, []).append(nci)

    for nci in non_bifurcated:
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

    for acc, group in bif_by_acc.items():
        acc_str = _format_site(G, acc)
        donors = []
        for nci in group:
            d_str = _format_site(G, nci.site_a)
            if nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
                h_sym = f"{G.nodes[nci.aux_atoms[0]]['symbol']}{nci.aux_atoms[0]}"
                d_str += f" via {h_sym}"
            donors.append(d_str)
        line = f"  {'hbond_bifurcated':<22s}  {' + '.join(donors)} ... {acc_str}"
        if debug:
            geom_parts = []
            for nci in group:
                parts = [f"{k}={v:.2f}" for k, v in nci.geometry.items()]
                geom_parts.append(", ".join(parts))
            line += f"  ({' | '.join(geom_parts)})"
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

    Uses :func:`build_nci_graph` to produce a decorated copy, then
    renders it via ``graph_to_ascii``.  The original graph is never mutated.

    Centroid nodes are positioned via a two-pass layout: first the
    molecule is laid out without centroids, then each centroid is placed
    at the mean 2-D position of its ring atoms.
    """
    from xyzgraph.ascii_renderer import graph_to_ascii

    if not ncis:
        header = f"{'=' * 80}\n# ASCII Depiction\n{'=' * 80}\n"
        ascii_out, _ = graph_to_ascii(
            G,
            scale=scale,
            include_h=include_h,
            show_h_indices=show_h_indices,
        )
        return header + "\n" + ascii_out + "\n"

    nci_G = build_nci_graph(G, ncis)

    # Collect H atoms involved in NCI display
    nci_show_h: set[int] = set()
    for nci in ncis:
        if nci.aux_atoms and G.nodes[nci.aux_atoms[0]].get("symbol") == "H":
            nci_show_h.add(nci.aux_atoms[0])

    h_indices = list(nci_show_h)
    if show_h_indices:
        h_indices = list(set(h_indices) | set(show_h_indices))

    centroid_nodes = nci_G.graph.get("nci_centroid", [])
    centroid_sites = nci_G.graph.get("nci_centroid_sites", {})
    title = "# ASCII Depiction (with NCI dotted lines)"
    if centroid_nodes:
        title += "  (* = centroid)"
    header = f"{'=' * 80}\n{title}\n{'=' * 80}\n"

    # First pass: layout with NCI atom-atom edges (for fragment orientation)
    # but without centroid nodes (which have only one edge and float away).
    # Centroids are then placed at the mean 2-D position of their ring atoms.
    layout_nodes = sorted(n for n in nci_G.nodes() if n not in centroid_nodes)
    _, layout_2d = graph_to_ascii(
        nci_G,
        scale=scale,
        include_h=include_h,
        show_h_indices=h_indices or None,
        nodes=layout_nodes,
    )
    for cid, atoms in centroid_sites.items():
        ring_pos = [layout_2d[a] for a in atoms if a in layout_2d]
        if ring_pos:
            cx = sum(p[0] for p in ring_pos) / len(ring_pos)
            cy = sum(p[1] for p in ring_pos) / len(ring_pos)
            layout_2d[cid] = (cx, cy)
    ascii_out, _ = graph_to_ascii(
        nci_G,
        scale=scale,
        include_h=include_h,
        show_h_indices=h_indices or None,
        reference_layout=layout_2d,
    )

    return header + "\n" + ascii_out + "\n"
