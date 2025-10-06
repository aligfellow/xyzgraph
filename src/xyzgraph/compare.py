from typing import Optional, Dict, List, Tuple
import numpy as np
import networkx as nx
from rdkit import Chem
from ase import Atoms
from .graph_builders import _bond_rdkit_order
from .ascii_renderer import graph_to_ascii

def xyz2mol_compare(atoms: Atoms,
                    charge: int = 0,
                    xyz_path: Optional[str] = None,
                    verbose: bool = False,
                    ascii: bool = False,
                    ascii_scale: float = 2.0,
                    ascii_include_h: bool = True,
                    reference_graph: Optional[nx.Graph] = None) -> str:
    """
    xyz2mol comparison / fallback diagnostic.
    verbose=True  -> include full atom neighbor table AND bond list.
    ascii=True    -> append ASCII depiction (scale, H visibility configurable).
    reference_graph -> if provided and atom counts match, xyz2mol ASCII is aligned
                       to the layout of the reference graph (same node indices).
    """
    try:
        from xyz2mol import xyz2mol
    except ImportError:
        return "# xyz2mol not installed (pip install xyz2mol)\n"

    def _load_source():
        if xyz_path:
            try:
                lines = open(xyz_path).read().splitlines()
                n = int(lines[0].strip())
                block = lines[2:2+n]
                syms = []; coords=[]
                for ln in block:
                    p=ln.split()
                    if len(p) >= 4:
                        syms.append(p[0]); coords.append([float(p[1]), float(p[2]), float(p[3])])
                if len(syms) != n:
                    return None, None, "# xyz parse mismatch\n"
                pt = Chem.GetPeriodicTable()
                return [pt.GetAtomicNumber(s) for s in syms], np.array(coords,float), None
            except Exception as e:
                return None, None, f"# xyz parse error: {e}\n"
        return list(map(int, atoms.get_atomic_numbers())), np.array(atoms.positions,float), None

    atomic_numbers, coords, err = _load_source()
    if err: return err
    if not atomic_numbers: return "# xyz2mol: no atoms\n"

    pt = Chem.GetPeriodicTable()
    try:
        nat=len(atomic_numbers)
        adj={i:[] for i in range(nat)}
        simple=[]; short=0; moderate=0
        for i in range(nat):
            ri=pt.GetRcovalent(int(atomic_numbers[i])) or 0.77
            for j in range(i+1,nat):
                rj=pt.GetRcovalent(int(atomic_numbers[j])) or 0.77
                d=float(np.linalg.norm(coords[i]-coords[j]))
                if d<1.25*(ri+rj):
                    simple.append((i,j,d)); adj[i].append(j); adj[j].append(i)
                if d<0.90*(ri+rj): short+=1
                elif d<1.10*(ri+rj): moderate+=1
        visited=set(); fragments=[]
        for i in range(nat):
            if i in visited: continue
            stack=[i]; comp=[]
            while stack:
                k=stack.pop()
                if k in visited: continue
                visited.add(k); comp.append(k); stack.extend(adj[k])
            fragments.append(sorted(comp))
    except Exception:
        simple=[]; short=moderate=0; fragments=[[i] for i in range(len(atomic_numbers))]

    total_e = sum(atomic_numbers) - charge
    param_sets = [
        dict(allow_charged_fragments=True,use_huckel=True),
        dict(allow_charged_fragments=False,use_huckel=False),
        dict(allow_charged_fragments=True,use_huckel=False),
        dict(allow_charged_fragments=False,use_huckel=True),
    ]
    mol=None; last=None; used=None
    for ps in param_sets:
        try:
            mols=xyz2mol(atomic_numbers,coords,charge=charge,**ps)
            if mols: mol=mols[0]; used=ps; break
        except Exception as e:
            last=str(e)
    if not mol:
        lines=["# xyz2mol failed",
               f"# last_error: {last or 'unknown'}",
               f"# electrons={total_e} ({'even' if total_e%2==0 else 'odd'}) fragments={len(fragments)} edges={len(simple)} short={short} moderate={moderate}"]
        if len(fragments)>1:
            lines.append("# note: multiple fragments may hinder perception")
        lines.append("# fallback heuristic connectivity listing (truncated):")
        for (i,j,d) in simple[:30]:
            lines.append(f"#   {i}-{j} d={d:.2f}")
        if len(simple)>30: lines.append("#   ...")
        return "\n".join(lines)+"\n"

    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass

    bond_map: Dict[int, List[Tuple[int, float, bool]]] = {}
    for b in mol.GetBonds():
        i=b.GetBeginAtomIdx(); j=b.GetEndAtomIdx()
        o=_bond_rdkit_order(b); ar=b.GetIsAromatic()
        bond_map.setdefault(i, []).append((j,o,ar))
        bond_map.setdefault(j, []).append((i,o,ar))

    out=[f"# xyz2mol graph: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds (charge={charge})"]

    # NEW: connectivity / bond order diff if reference provided
    if reference_graph is not None and reference_graph.number_of_nodes() == mol.GetNumAtoms():
        # xyzgraph edges
        ref_edges = {}
        for i,j,d in reference_graph.edges(data=True):
            ref_edges[tuple(sorted((i,j)))] = float(d.get('bond_order',1.0))
        # xyz2mol edges
        x2_edges = {}
        for b in mol.GetBonds():
            e = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            x2_edges[e] = float(_bond_rdkit_order(b))
        only_ref = sorted(e for e in ref_edges.keys() if e not in x2_edges)
        only_x2  = sorted(e for e in x2_edges.keys() if e not in ref_edges)
        shared = sorted(e for e in ref_edges.keys() if e in x2_edges)
        bo_diffs = []
        for e in shared:
            r = ref_edges[e]; x = x2_edges[e]
            if abs(r - x) >= 0.25:
                bo_diffs.append((e,r,x, r-x))
        out.append("# edge_diff: only_in_xyzgraph={:,} only_in_xyz2mol={:,} bond_order_diffs={:,}".format(
            len(only_ref), len(only_x2), len(bo_diffs)))
        if only_ref:
            out.append("#   only_in_xyzgraph: " + " ".join(f"{a}-{b}" for a,b in only_ref))
        if only_x2:
            out.append("#   only_in_xyz2mol: " + " ".join(f"{a}-{b}" for a,b in only_x2))
        if bo_diffs:
            out.append("#   bond_order_diffs (Δ≥0.25):")
            for (a,b,r,x,delta) in bo_diffs[:40]:
                out.append(f"#     {a}-{b}: xyzgraph={r:.2f} xyz2mol={x:.2f} Δ={delta:+.2f}")
            if len(bo_diffs) > 40:
                out.append("#     ...")

    if verbose and used:
        out.append("# used_params: "+", ".join(f"{k}={v}" for k,v in used.items()))

    if verbose:
        out.append("# [idx] Sym formal val | neighbors idx(order*)")
        for a in mol.GetAtoms():
            i=a.GetIdx(); sym=a.GetSymbol(); f=a.GetFormalCharge()
            nbrs=sorted(bond_map.get(i,[]), key=lambda x:x[0])
            val=sum(o for _,o,_ in nbrs)
            nbrs_str=" ".join(f"{n}({o:.2f}{'*' if ar else ''})" for n,o,ar in nbrs) if nbrs else "-"
            out.append(f"[{i:>2}] {sym:>2} {f:+d} {val:.2f} | {nbrs_str}")
        out.append("")

    if verbose:
        out.append("# Bonds (i-j: order)")
        for b in mol.GetBonds():
            out.append(f"{b.GetBeginAtomIdx()}-{b.GetEndAtomIdx()}: {_bond_rdkit_order(b):.2f}")

    if ascii:
        Gx = nx.Graph()
        for a in mol.GetAtoms():
            i=a.GetIdx()
            Gx.add_node(i, symbol=a.GetSymbol(), charges={}, agg_charge=0.0,
                        formal_charge=a.GetFormalCharge(), valence=0.0)
        for b in mol.GetBonds():
            i=b.GetBeginAtomIdx(); j=b.GetEndAtomIdx()
            bo=_bond_rdkit_order(b)
            Gx.add_edge(i,j,bond_order=bo,bond_type=(Gx.nodes[i]['symbol'], Gx.nodes[j]['symbol']))
        for n in Gx.nodes:
            Gx.nodes[n]['valence']=sum(Gx.edges[n,m].get('bond_order',1.0) for m in Gx.neighbors(n))

        aligned = False
        layout = None
        if reference_graph is not None and reference_graph.number_of_nodes() == Gx.number_of_nodes():
            # Get layout from reference (do not print its ASCII here; CLI already printed)
            _, layout = graph_to_ascii(reference_graph,
                                       scale=ascii_scale,
                                       include_h=ascii_include_h,
                                       return_layout=True)
            aligned = True

        if aligned and layout:
            xyz2mol_ascii = graph_to_ascii(Gx,
                                           scale=ascii_scale,
                                           include_h=ascii_include_h,
                                           reference_layout=layout)
            header = "# xyz2mol ASCII (aligned to reference)"
        else:
            xyz2mol_ascii = graph_to_ascii(Gx,
                                           scale=ascii_scale,
                                           include_h=ascii_include_h)
            header = "# xyz2mol ASCII"

        out.append("")
        out.append(header)
        out.append(xyz2mol_ascii)

    return "\n".join(out)+"\n"
