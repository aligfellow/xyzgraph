"""ASCII rendering for molecular graphs."""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from rdkit import Chem

from .utils import _visible_nodes


# --- special edge glyph helper ---
def _edge_char(attrs: Dict[str, Any], bo: float, orient: str, dx: int, dy: int) -> Tuple[str, bool, str]:
    """Return (glyph, special_flag, bond_type).

    special_flag True => skip multi-line double/triple drawing.
    bond_type in ['single', 'double', 'triple'] for parallel line logic.
    Precedence: TS > NCI > normal.
    """
    if attrs.get("TS"):
        return "*", True, "special"
    if attrs.get("NCI"):
        return ".", True, "special"

    # Triple bonds
    if bo >= 2.5:
        return "#", False, "triple"

    # Double bonds - use more distinctive characters
    if bo >= 1.8:
        if orient == "h":
            return "=", False, "double"
        if orient == "v":
            return "‖", False, "double"  # Unicode double vertical line
        # For diagonals, we'll handle spacing differently
        if (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
            return "/", False, "double"
        else:
            return "\\", False, "double"

    #     # --- Aromatic bonds (1.35-1.8 range, typical) ---
    # if 1.35 <= bo < 1.8:
    #     # choose base single bond symbol by orientation
    #     if orient == 'h':
    #         return '-.', False, 'aromatic'
    #     if orient == 'v':
    #         return '|.', False, 'aromatic'
    #     if (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
    #         return '/.', False, 'aromatic'
    #     else:
    #         return '\\.', False, 'aromatic'

    # Single bonds
    if orient == "h":
        return "-", False, "single"
    if orient == "v":
        return "|", False, "single"
    if (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
        return "/", False, "single"
    else:
        return "\\", False, "single"


# --- end helper ---


# -----------------------------
# 2D depiction (RDKit-based)
# -----------------------------
class GraphToASCII:
    """Core renderer: build RDKit 2D layout and rasterize to ASCII.

    Prefer using graph_to_ascii() unless you need layout reuse across many graphs.
    """

    def __init__(self): ...

    def _build_rdkit_mol(
        self,
        graph: nx.Graph,
        nodes: List[int],
        reference_layout: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Tuple[Chem.Mol, Dict[int, int]]:
        idx_map = {orig: new for new, orig in enumerate(nodes)}
        mol = Chem.RWMol()
        for n in nodes:
            sym = graph.nodes[n].get("symbol", "C")
            mol.AddAtom(Chem.Atom(sym))
        for i, j, data in graph.edges(data=True):
            if i in idx_map and j in idx_map:
                bo = float(data.get("bond_order", 1.0))
                if bo >= 2.5:
                    bt = Chem.BondType.TRIPLE
                elif bo >= 1.75:
                    bt = Chem.BondType.DOUBLE
                elif 1.4 < bo < 1.6:
                    bt = Chem.BondType.AROMATIC
                else:
                    bt = Chem.BondType.SINGLE
                mol.AddBond(idx_map[i], idx_map[j], bt)
        if reference_layout is not None:
            conf = Chem.Conformer(len(nodes))
            for orig, new in idx_map.items():
                x, y = reference_layout.get(orig, (0.0, 0.0))
                conf.SetAtomPosition(new, (float(x), float(y), 0.0))
            mol.AddConformer(conf, assignId=True)
        else:
            from rdkit.Chem import rdDepictor

            try:
                rdDepictor.Compute2DCoords(mol)
            except Exception:
                conf = Chem.Conformer(len(nodes))
                mol.AddConformer(conf, assignId=True)
        return mol, idx_map

    def _mol_to_ascii(
        self,
        mol: Chem.Mol,
        nodes: List[int],
        bond_orders_map: Dict[Tuple[int, int], float],
        edge_attr_map: Dict[Tuple[int, int], Dict[str, Any]],
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        scale: float = 1.0,
    ) -> str:
        if mol.GetNumAtoms() == 0:
            return "<empty>"
        try:
            conf = mol.GetConformer()
        except Exception:
            return "<no conformer>"

        n = mol.GetNumAtoms()
        if scale_x is None or scale_y is None:
            if n <= 10:
                mult_x, mult_y = 11, 6
                scale_x, scale_y = 1.35, 1.05
            elif n <= 25:
                mult_x, mult_y = 14, 8
                scale_x, scale_y = 1.45, 1.10
            else:
                mult_x, mult_y = 17, 10
                scale_x, scale_y = 1.55, 1.15
        else:
            mult_x, mult_y = 14, 8
        # Global user scale
        mult_x = int(max(1, round(mult_x * scale)))
        mult_y = int(max(1, round(mult_y * scale)))

        coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n)]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        span_x = max(max(xs) - min(xs), 1e-3)
        span_y = max(max(ys) - min(ys), 1e-3)

        grid = []
        for x, y in coords:
            gx = round(((x - min(xs)) / span_x) * scale_x * mult_x)
            gy = round(((y - min(ys)) / span_y) * scale_y * mult_y)
            grid.append((gx, gy))
        padding = 1
        max_gx = max(g for g, _ in grid) + padding
        max_gy = max(g for _, g in grid) + padding
        canvas = [[" "] * (max_gx + 1) for _ in range(max_gy + 1)]

        def classify(x1, y1, x2, y2):
            dx, dy = x2 - x1, y2 - y1
            adx, ady = abs(dx), abs(dy)
            if ady < 0.35 * adx:
                return "h", dx, dy
            if adx < 0.35 * ady:
                return "v", dx, dy
            return "d", dx, dy

        def draw_line(x1, y1, x2, y2, ch):
            steps = max(abs(x2 - x1), abs(y2 - y1))
            steps = max(1, steps)
            for t in range(steps + 1):
                xt = round(x1 + (x2 - x1) * t / steps)
                yt = round(y1 + (y2 - y1) * t / steps)
                if 0 <= yt < len(canvas) and 0 <= xt < len(canvas[0]):
                    if canvas[yt][xt] == " ":
                        canvas[yt][xt] = ch

        def draw_parallel(x1, y1, x2, y2, ox, oy, ch):
            draw_line(x1 + ox, y1 + oy, x2 + ox, y2 + oy, ch)

        # IMPROVED: Draw double diagonal bonds with wider spacing
        def draw_double_diagonal(x1, y1, x2, y2, glyph):
            """Draw double diagonal bond with improved spacing."""
            steps = max(abs(x2 - x1), abs(y2 - y1))
            steps = max(1, steps)

            # Determine perpendicular offset direction
            _dx, _dy = x2 - x1, y2 - y1
            if glyph == "/":
                # For /, offset perpendicular is along \
                offset_pairs = [(0, 0), (1, 0), (-1, 0)]  # center, right, left
            else:  # '\\'
                # For \, offset perpendicular is along /
                offset_pairs = [(0, 0), (-1, 0), (1, 0)]  # center, left, right

            # Draw main line
            for t in range(steps + 1):
                xt = round(x1 + (x2 - x1) * t / steps)
                yt = round(y1 + (y2 - y1) * t / steps)
                for ox, oy in offset_pairs[:2]:  # Draw two parallel lines
                    xp, yp = xt + ox, yt + oy
                    if 0 <= yp < len(canvas) and 0 <= xp < len(canvas[0]):
                        if canvas[yp][xp] == " ":
                            canvas[yp][xp] = glyph

        rev_map = {i: nodes[i] for i in range(n)}

        # --- IMPROVED edge drawing with better double bond rendering ---
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            o1 = rev_map[i]
            o2 = rev_map[j]
            bo = bond_orders_map.get((o1, o2), bond_orders_map.get((o2, o1), 1.0))
            attrs = edge_attr_map.get((o1, o2), edge_attr_map.get((o2, o1), {}))
            (x1, y1) = grid[i]
            (x2, y2) = grid[j]
            orient, dx, dy = classify(x1, y1, x2, y2)
            glyph, special, bond_type = _edge_char(attrs, bo, orient, dx, dy)

            # Draw main bond line
            draw_line(x1, y1, x2, y2, glyph)

            # Handle double bonds with improved spacing
            if not special and bond_type == "double":
                if orient == "h":
                    # Horizontal: add line above
                    draw_parallel(x1, y1, x2, y2, 0, 1, glyph)
                elif orient == "v":
                    # Vertical: add line to the right
                    draw_parallel(x1, y1, x2, y2, 1, 0, "‖")
                # Diagonal: use improved double diagonal rendering
                # Draw second parallel line with better spacing
                elif glyph == "/":
                    draw_parallel(x1, y1, x2, y2, 1, 0, "/")
                else:
                    draw_parallel(x1, y1, x2, y2, 1, 0, "\\")

            # Triple bonds: just draw single '#' (no additional lines)
        # --- end improved edge drawing ---

        for m_idx, _orig in enumerate(nodes):
            gx, gy = grid[m_idx]
            if 0 <= gy < len(canvas) and 0 <= gx < len(canvas[0]):
                sym = mol.GetAtomWithIdx(m_idx).GetSymbol()
                canvas[gy][gx] = sym[0]
                if len(sym) > 1:
                    sx = gx + 1
                    if sx < len(canvas[0]):
                        # Overwrite if blank or bond glyph
                        if canvas[gy][sx] in (
                            " ",
                            "-",
                            "=",
                            "|",
                            "/",
                            "\\",
                            "#",
                            "*",
                            ".",
                            "‖",
                        ):
                            canvas[gy][sx] = sym[1]

        lines = ["".join(r).rstrip() for r in canvas]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines) if lines else "<empty>"

    def render(
        self,
        graph: nx.Graph,
        nodes: Optional[List[int]] = None,
        reference_layout: Optional[Dict[int, Tuple[float, float]]] = None,
        scale: float = 1.0,
        include_h: bool = False,
        show_h_indices: Optional[List[int]] = None,
    ) -> Tuple[str, Dict[int, Tuple[float, float]]]:
        """Render graph to ASCII."""
        if nodes is None:
            nodes = _visible_nodes(graph, include_h, show_h_indices)
        else:
            # When nodes are explicitly provided, still respect show_h_indices
            show_h_set = set(show_h_indices) if show_h_indices else set()
            nodes = [
                n
                for n in nodes
                if include_h
                or graph.nodes[n].get("symbol") != "H"
                or n in show_h_set
                or any(graph.nodes[nbr].get("symbol") != "C" for nbr in graph.neighbors(n))
            ]
        if not nodes:
            return "<no heavy atoms>", {}
        nodes = sorted(nodes)
        mol, idx_map = self._build_rdkit_mol(graph, nodes, reference_layout=reference_layout)
        try:
            conf = mol.GetConformer()
            layout = {orig: (conf.GetAtomPosition(new).x, conf.GetAtomPosition(new).y) for orig, new in idx_map.items()}
        except Exception:
            layout = dict.fromkeys(nodes, (0.0, 0.0))
        bond_orders_map: Dict[Tuple[int, int], float] = {}
        edge_attr_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for i, j, data in graph.edges(data=True):
            if i in idx_map and j in idx_map:
                bo = float(data.get("bond_order", 1.0))
                bond_orders_map[(i, j)] = bo
                bond_orders_map[(j, i)] = bo
                edge_attr_map[(i, j)] = data
                edge_attr_map[(j, i)] = data
        ascii_str = self._mol_to_ascii(mol, nodes, bond_orders_map, edge_attr_map, scale=scale)
        return ascii_str, layout


def graph_to_ascii(
    G: nx.Graph,
    scale: float = 3.0,
    include_h: bool = False,
    reference: Optional[nx.Graph] = None,
    reference_layout: Optional[Dict[int, Tuple[float, float]]] = None,
    nodes: Optional[List[int]] = None,
    show_h_indices: Optional[List[int]] = None,
) -> Tuple[str, Dict[int, Tuple[float, float]]]:
    """
    Render graph to ASCII.

    Returns
    -------
    str or Tuple[str, Dict[int, Tuple[float,float]]]
        ASCII rendering string, or tuple of (ascii string, layout dict) if return_layout=True

    Notes
    -----
    Alignment only uses intersection of node sets; if no overlap layout fallback occurs.
    """
    gta = GraphToASCII()
    layout = None
    base_nodes = nodes
    if reference is not None and reference_layout is None:
        _, ref_layout = gta.render(
            reference,
            nodes=base_nodes,
            scale=scale,
            include_h=include_h,
            show_h_indices=show_h_indices,
        )
        layout = ref_layout
    if reference_layout is not None:
        layout = reference_layout
    target_nodes = base_nodes
    if layout is not None:
        allowed = set(layout.keys())
        if target_nodes is None:
            target_nodes = sorted(n for n in _visible_nodes(G, include_h, show_h_indices) if n in allowed)
        else:
            target_nodes = [n for n in target_nodes if n in allowed]
        if not target_nodes:
            layout = None
            target_nodes = base_nodes
    ascii_out, out_layout = gta.render(
        G,
        nodes=target_nodes,
        reference_layout=layout,
        scale=scale,
        include_h=include_h,
        show_h_indices=show_h_indices,
    )
    return ascii_out, out_layout


__all__ = ["GraphToASCII", "graph_to_ascii"]
