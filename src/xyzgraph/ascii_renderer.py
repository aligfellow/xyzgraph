from typing import List, Dict, Tuple, Optional
import networkx as nx
from rdkit import Chem

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
    for i,j,d in sorted(G.edges(data=True)):
        if i in visible and j in visible:
            lines.append(f"{i}-{j}: {d.get('bond_order',1.0):.2f}")
    return "\n".join(lines)

# -----------------------------
# 2D depiction (RDKit-based)
# -----------------------------
class GraphToASCII:
    """
    Core renderer (public but light): build RDKit 2D layout and rasterize to ASCII.
    Prefer using graph_to_ascii() unless you need layout reuse across many graphs.
    """
    def __init__(self): ...

    def _build_rdkit_mol(self,
                         graph: nx.Graph,
                         nodes: List[int],
                         reference_layout: Optional[Dict[int, Tuple[float,float]]] = None) -> Tuple[Chem.Mol, Dict[int,int]]:
        idx_map = {orig: new for new, orig in enumerate(nodes)}
        mol = Chem.RWMol()
        for n in nodes:
            sym = graph.nodes[n].get('symbol', 'C')
            mol.AddAtom(Chem.Atom(sym))
        for i,j,data in graph.edges(data=True):
            if i in idx_map and j in idx_map:
                bo = float(data.get('bond_order', 1.0))
                if bo >= 2.5: bt = Chem.BondType.TRIPLE
                elif bo >= 1.75: bt = Chem.BondType.DOUBLE
                elif 1.4 < bo < 1.6: bt = Chem.BondType.AROMATIC
                else: bt = Chem.BondType.SINGLE
                mol.AddBond(idx_map[i], idx_map[j], bt)
        if reference_layout is not None:
            conf = Chem.Conformer(len(nodes))
            for orig, new in idx_map.items():
                x,y = reference_layout.get(orig, (0.0,0.0))
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

    def _mol_to_ascii(self,
                      mol: Chem.Mol,
                      nodes: List[int],
                      bond_orders_map: Dict[Tuple[int,int], float],
                      scale_x: Optional[float] = None,
                      scale_y: Optional[float] = None,
                      padding: int = 1,
                      scale: float = 1.0) -> str:
        if mol.GetNumAtoms() == 0:
            return "<empty>"
        try:
            conf = mol.GetConformer()
        except Exception:
            return "<no conformer>"

        n = mol.GetNumAtoms()
        if scale_x is None or scale_y is None:
            if n <= 10:
                mult_x, mult_y = 11, 6; scale_x, scale_y = 1.35, 1.05
            elif n <= 25:
                mult_x, mult_y = 14, 8; scale_x, scale_y = 1.45, 1.10
            else:
                mult_x, mult_y = 17, 10; scale_x, scale_y = 1.55, 1.15
        else:
            mult_x, mult_y = 14, 8
        # NEW: global user scale
        mult_x = int(max(1, round(mult_x * scale)))
        mult_y = int(max(1, round(mult_y * scale)))

        coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n)]
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        span_x = max(max(xs)-min(xs), 1e-3)
        span_y = max(max(ys)-min(ys), 1e-3)

        grid = []
        for (x,y) in coords:
            gx = int(round(((x - min(xs))/span_x) * scale_x * mult_x))
            gy = int(round(((y - min(ys))/span_y) * scale_y * mult_y))
            grid.append((gx,gy))
        max_gx = max(g for g,_ in grid) + padding
        max_gy = max(g for _,g in grid) + padding
        canvas = [[' ']*(max_gx+1) for _ in range(max_gy+1)]

        def classify(x1,y1,x2,y2):
            dx, dy = x2-x1, y2-y1
            adx, ady = abs(dx), abs(dy)
            if ady < 0.35*adx: return 'h', dx, dy
            if adx < 0.35*ady: return 'v', dx, dy
            return 'd', dx, dy

        def bond_class(bo: float):
            if bo >= 2.5: return 'triple'
            if bo >= 1.9: return 'double'
            return 'single'

        def draw_line(x1,y1,x2,y2,ch):
            steps = max(abs(x2-x1), abs(y2-y1))
            steps = max(1, steps)
            for t in range(steps+1):
                xt = int(round(x1 + (x2-x1)*t/steps))
                yt = int(round(y1 + (y2-y1)*t/steps))
                if 0 <= yt < len(canvas) and 0 <= xt < len(canvas[0]):
                    if canvas[yt][xt] == ' ':
                        canvas[yt][xt] = ch

        def draw_parallel(x1,y1,x2,y2,ox,oy,ch):
            draw_line(x1+ox,y1+oy,x2+ox,y2+oy,ch)

        rev_map = {i: nodes[i] for i in range(n)}

        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
            o1 = rev_map[i]; o2 = rev_map[j]
            bo = bond_orders_map.get((o1,o2), bond_orders_map.get((o2,o1), 1.0))
            (x1,y1) = grid[i]; (x2,y2) = grid[j]
            orient, dx, dy = classify(x1,y1,x2,y2)
            bcls = bond_class(bo)
            if bcls == 'triple':
                glyph = '#'
            elif bcls == 'double':
                if orient == 'h': glyph = '='
                elif orient == 'v': glyph = '|'
                else: glyph = '/' if (dx>0 and dy<0) or (dx<0 and dy>0) else '\\'
            else:
                if orient == 'h': glyph = '-'
                elif orient == 'v': glyph = '|'
                else: glyph = '/' if (dx>0 and dy<0) or (dx<0 and dy>0) else '\\'
            draw_line(x1,y1,x2,y2,glyph)
            if bcls == 'double':
                if orient == 'h': draw_parallel(x1,y1,x2,y2,0,1,'=')
                elif orient == 'v': draw_parallel(x1,y1,x2,y2,1,0,'|')
                else:
                    if glyph == '/': draw_parallel(x1,y1,x2,y2,-1,0,'/')
                    else: draw_parallel(x1,y1,x2,y2,1,0,'\\')

        for m_idx, orig in enumerate(nodes):
            gx,gy = grid[m_idx]
            if 0 <= gy < len(canvas) and 0 <= gx < len(canvas[0]):
                sym = mol.GetAtomWithIdx(m_idx).GetSymbol()
                canvas[gy][gx] = sym[0]
                if len(sym) > 1:
                    sx = gx + 1
                    if sx < len(canvas[0]):
                        # Overwrite if blank or bond glyph
                        if canvas[gy][sx] in (' ', '-', '=', '|', '/', '\\', '#'):
                            canvas[gy][sx] = sym[1]

        lines = ["".join(r).rstrip() for r in canvas]
        while lines and not lines[0].strip(): lines.pop(0)
        while lines and not lines[-1].strip(): lines.pop()
        return "\n".join(lines) if lines else "<empty>"

    def render(self,
               graph: nx.Graph,
               nodes: Optional[List[int]] = None,
               reference_layout: Optional[Dict[int, Tuple[float,float]]] = None,
               scale: float = 1.0,
               include_h: bool = False) -> Tuple[str, Dict[int, Tuple[float,float]]]:
        if nodes is None:
            nodes = _visible_nodes(graph, include_h)
        else:
            nodes = [n for n in nodes if include_h or graph.nodes[n].get('symbol') != 'H' or
                     any(graph.nodes[nbr].get('symbol') != 'C' for nbr in graph.neighbors(n))]
        if not nodes:
            return "<no heavy atoms>", {}
        nodes = sorted(nodes)
        mol, idx_map = self._build_rdkit_mol(graph, nodes, reference_layout=reference_layout)
        try:
            conf = mol.GetConformer()
            layout = {orig: (conf.GetAtomPosition(new).x, conf.GetAtomPosition(new).y)
                      for orig, new in idx_map.items()}
        except Exception:
            layout = {orig: (0.0,0.0) for orig in nodes}
        bond_orders_map: Dict[Tuple[int,int], float] = {}
        for i,j,data in graph.edges(data=True):
            if i in idx_map and j in idx_map:
                bo = float(data.get('bond_order', 1.0))
                bond_orders_map[(i,j)] = bo
                bond_orders_map[(j,i)] = bo
        ascii_str = self._mol_to_ascii(mol, nodes, bond_orders_map, scale=scale)
        return ascii_str, layout

def graph_to_ascii(G: nx.Graph,
                   scale: float = 3.0,
                   include_h: bool = False,
                   reference: Optional[nx.Graph] = None,
                   reference_layout: Optional[Dict[int, Tuple[float,float]]] = None,
                   nodes: Optional[List[int]] = None,
                   return_layout: bool = False) -> str | Tuple[str, Dict[int, Tuple[float,float]]]:
    """
    Render graph to ASCII.
      reference: optional graph providing canonical layout.
      reference_layout: explicit node->(x,y) dict (overrides reference).
      nodes: optional subset (auto-filtered for H hiding unless include_h=True).
      return_layout=True returns (ascii, layout).
    Alignment only uses intersection of node sets; if no overlap layout fallback occurs.
    """
    gta = GraphToASCII()
    layout = None
    base_nodes = nodes
    if reference is not None and reference_layout is None:
        _, ref_layout = gta.render(reference,
                                   nodes=base_nodes,
                                   scale=scale,
                                   include_h=include_h)
        layout = ref_layout
    if reference_layout is not None:
        layout = reference_layout
    target_nodes = base_nodes
    if layout is not None:
        allowed = set(layout.keys())
        if target_nodes is None:
            target_nodes = sorted(n for n in _visible_nodes(G, include_h) if n in allowed)
        else:
            target_nodes = [n for n in target_nodes if n in allowed]
        if not target_nodes:
            layout = None
            target_nodes = base_nodes
    ascii_out, out_layout = gta.render(G,
                                       nodes=target_nodes,
                                       reference_layout=layout,
                                       scale=scale,
                                       include_h=include_h)
    if return_layout:
        return ascii_out, out_layout
    return ascii_out

__all__ = [
    "graph_debug_report",
    "graph_to_ascii",
    "GraphToASCII"
]
