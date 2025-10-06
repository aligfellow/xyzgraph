import os
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from .data_loader import load_vdw_radii, load_expected_valences, load_valence_electrons

# ----------------------------------------------------------------------------------
# High-level flow (cheminf path):
#   1. Distance-based bond guess (VDW + metal-aware thresholds)
#   2. Initial aromatic guess (5/6 rings with allowed atoms)
#   3. Valence-driven bond order refinement (locks metal bonds to 1.0)
#   4. RDKit aromatic perception pass
#   5. Charge model (Gasteiger) + formal charges
#   6. Sanitation pass (valence recheck, H-aggregation)
#
# Design intent:
#   - Be permissive enough for heterogeneous / metal-containing systems without
#     collapsing into incorrect over-bonded cores.
#   - Keep metal coordination "graphical" (single bonds) while still allowing
#     ligand internal bond order refinement.
# ----------------------------------------------------------------------------------

# Central metal set (used in multiple stages)
METALS = {
    'Li','Na','K','Mg','Ca','Zn','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os',
    'Ir','Pt','Au','Hg','Al','Ga','In','Sn','Pb','La','Ce','Pr','Nd','Sm','Eu',
    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'
}

# Lazy-load caches (added)
VDW = None
VALENCES = None
VALENCE_ELECTRONS = None

def _get_vdw():
    global VDW 
    if VDW is None:
        VDW = load_vdw_radii()
    return VDW

def _get_expected_valences():
    global VALENCES
    if VALENCES is None:
        VALENCES = load_expected_valences()  # FIX: call the loader
    return VALENCES

def _get_valence_electrons():
    global VALENCE_ELECTRONS
    if VALENCE_ELECTRONS is None:
        VALENCE_ELECTRONS = load_valence_electrons()
    return VALENCE_ELECTRONS

# Public accessors (lightweight wrappers) ---------------------------
def get_vdw() -> Dict[str, float]:
    """
    Public accessor for Van der Waals radii.
    Ensures lazy cache is populated.
    """
    return _get_vdw()

def get_expected_valences() -> Dict[str, List[int]]:
    """
    Public accessor for expected valence list per element.
    """
    return _get_expected_valences()

def get_valence_electrons() -> Dict[str, int]:
    """
    Public accessor for valence electron counts.
    """
    return _get_valence_electrons()


# ------------------------
# Bond perception helpers
# ------------------------

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _should_bond_metal(sym_i: str, sym_j: str, distance: float, vdw: Dict[str,float]) -> bool:
    """
    Chemical filter for metal bonds (called AFTER distance check).
    
    Returns False only for implausible metal pairings:
      - Metal-metal (unless bridging ligand expected)
      - Metal to non-donor heavy atom (e.g., Ru-Ne)
    
    Accepts:
      - Metal to donor atoms (O, N, C, P, S)
      - Metal to halides/oxo (ionic)
      - Metal to H (hydrides)
    """
    if sym_i not in METALS and sym_j not in METALS:
        return True
        
    # Identify metal and other
    metal = sym_i if sym_i in METALS else sym_j
    other = sym_j if metal == sym_i else sym_i
    
    # Accept common ligands
    if other in ('O', 'N', 'C', 'P', 'S', 'H'):
        return True
    
    # Accept halides (ionic)
    if other in ('F', 'Cl', 'Br', 'I'):
        return True
    
    # Reject other pairings (noble gases, metal-metal, etc.)
    return False

def _initial_bonds(atoms: Atoms,
                   vdw: Dict[str, float],
                   vdw_scale_h=0.45,
                   vdw_scale_h_metal=0.5,
                   vdw_scale=0.55,
                   vdw_scale_metal=0.65) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Return (bonds, bond_dists). Distances cached for later refinement / pruning.
    """
    bonds = []
    dists: List[float] = []
    pos = atoms.positions
    for i in range(len(atoms)):
        si = atoms[i].symbol
        for j in range(i+1, len(atoms)):
            sj = atoms[j].symbol
            d = _distance(pos[i], pos[j])
            r_sum_vdw = vdw.get(si, 2.0) + vdw.get(sj, 2.0)
            is_metal_i = si in METALS
            is_metal_j = sj in METALS
            has_metal = is_metal_i or is_metal_j
            has_h = 'H' in (si, sj)
            if has_h and not has_metal:
                threshold = vdw_scale_h * r_sum_vdw
            elif has_h and has_metal:
                threshold = vdw_scale_h_metal * r_sum_vdw
            elif has_metal:
                threshold = vdw_scale_metal * r_sum_vdw
            else:
                threshold = vdw_scale * r_sum_vdw
            if d < threshold:
                if has_metal:
                    if _should_bond_metal(si, sj, d, vdw):
                        bonds.append((i, j)); dists.append(d)
                else:
                    bonds.append((i, j)); dists.append(d)
    return bonds, dists

def _compute_bond_distances(atoms: Atoms,
                            bonds: List[Tuple[int,int]]) -> List[float]:
    pos = atoms.positions
    return [float(np.linalg.norm(pos[i]-pos[j])) for i,j in bonds]

def _build_temp_graph(atoms: Atoms,
                      bonds: List[Tuple[int,int]],
                      orders: List[float]) -> nx.Graph:
    G = nx.Graph()
    for i, a in enumerate(atoms):
        G.add_node(i, symbol=a.symbol, Z=a.number)
    for k,(i,j) in enumerate(bonds):
        G.add_edge(i, j, bond_order=orders[k])
    return G

def _identify_valence_issues(G: nx.Graph,
                             expected: Dict[str, List[int]],
                             tol: float = 0.5) -> Dict[int, Dict[str, Any]]:
    issues = {}
    for n,data in G.nodes(data=True):
        sym = data['symbol']
        if sym not in expected:
            continue
        total = 0.0
        for nbr in G.neighbors(n):
            total += G.edges[n, nbr].get('bond_order', 1.0)
        best_err = min(abs(total - v) for v in expected[sym])
        if best_err > tol:
            issues[n] = {'current': total, 'expected': expected[sym], 'error': best_err}
    return issues

def _valence_sum(G: nx.Graph, n: int) -> float:
    """Sum fractional bond order around node n (default 1.0 if missing attribute)."""
    return sum(G.edges[n, nbr].get('bond_order', 1.0) for nbr in G.neighbors(n))

# NEW: shared helper for list-based bond containers (used before graph assembly)
def _bond_order_sum(index: int,
                    bonds: List[Tuple[int,int]],
                    bond_orders: List[float]) -> float:
    return sum(bond_orders[k] for k,(a,b) in enumerate(bonds) if index == a or index == b)

def _adjust_valences(atoms: Atoms,
                     bonds: List[Tuple[int,int]],
                     bond_orders: List[float],
                     expected: Dict[str,List[int]],
                     vdw: Dict[str,float],
                     max_iter: int = 5,
                     multiplicity: int = 1,
                     bond_dists: Optional[List[float]] = None) -> Dict[str,int]:
    """
    Bond order refinement (non-metals):
      - Metals: coordination only, all bonds locked to 1.0 (no π promotion).
      - Compute per-atom deficit relative to closest allowed valence.
      - Adjust only when (a) geometric proximity acceptable, (b) deficits complementary.
      - Avoid oscillation by minimum increment (0.5) and single-pass modifications.
    """
    if bond_dists is None:
        bond_dists = _compute_bond_distances(atoms, bonds)
    for k,(i,j) in enumerate(bonds):
        if atoms[i].symbol in METALS or atoms[j].symbol in METALS:
            bond_orders[k] = 1.0
    unpaired_budget = multiplicity - 1
    promotions = 0
    reductions = 0
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        G = _build_temp_graph(atoms, bonds, bond_orders)
        deficits: Dict[int, float] = {}
        any_issue = False
        for n in range(len(atoms)):
            sym = atoms[n].symbol
            cur = _valence_sum(G, n)
            if sym in METALS:
                deficits[n] = 0.0
                continue
            allowed = expected.get(sym, [])
            if not allowed:
                deficits[n] = 0.0
                continue
            target = min(allowed, key=lambda v: (abs(v - cur), -v))
            diff = target - cur
            if unpaired_budget > 0 and -1.3 < diff < -0.7:
                deficits[n] = 0.0
                unpaired_budget -= 1
            else:
                deficits[n] = diff
            if abs(diff) > 0.5:
                any_issue = True
        if not any_issue:
            break
        changed = 0
        for k,(i,j) in enumerate(bonds):
            si, sj = atoms[i].symbol, atoms[j].symbol
            bo = bond_orders[k]
            if si in METALS or sj in METALS:
                if abs(bo - 1.0) > 1e-6:
                    bond_orders[k] = 1.0; changed += 1
                continue
            if bo >= 3.0: continue
            di, dj = deficits[i], deficits[j]
            if 'H' in (si, sj): continue
            dist_ratio = bond_dists[k] / (vdw.get(si,2.0)+vdw.get(sj,2.0))
            if dist_ratio > 0.60: continue
            if di < -0.6 and dj < -0.6 and bo > 1.0:
                bond_orders[k] = max(1.0, bo - 0.5); reductions += 1; changed += 1; continue
            if abs(di) < 0.3 and abs(dj) < 0.3:
                continue
            inc = 0.0
            if di > 0.3 and dj > 0.3:
                inc = min(di, dj, (3.0 - bo if dist_ratio < 0.35 else 2.0 - bo))
            elif di > 0.3 and dj < -0.3:
                inc = min(di, -dj, 2.0 - bo)
            elif dj > 0.3 and di < -0.3:
                inc = min(dj, -di, 2.0 - bo)
            if inc >= 0.5:
                bond_orders[k] = bo + inc; promotions += 1; changed += 1
        if not changed:
            break
    return {"iterations": iterations, "promotions": promotions, "reductions": reductions}

def _detect_aromatic_cycles(atoms: Atoms,
                            bonds: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    G.add_edges_from(bonds)
    aromatic_edges = set()
    for cycle in nx.cycle_basis(G):
        if len(cycle) in (5,6):
            if all(atoms[i].symbol in ('C','N','O','S') for i in cycle):
                # mark alternating edges as aromatic candidate
                for k in range(len(cycle)):
                    a = cycle[k]; b = cycle[(k+1)%len(cycle)]
                    aromatic_edges.add(tuple(sorted((a,b))))
    return list(aromatic_edges)

def _apply_aromatic(bonds: List[Tuple[int,int]],
                    bond_orders: List[float],
                    aromatic_edges: List[Tuple[int,int]]):
    arom = set(aromatic_edges)
    for idx,(i,j) in enumerate(bonds):
        if tuple(sorted((i,j))) in arom:
            bond_orders[idx] = max(bond_orders[idx], 1.5)

def _rdkit_aromatic_refine(atoms: Atoms,
                           bonds: List[Tuple[int,int]],
                           bond_orders: List[float]) -> int:
    """
    Build RDKit molecule and use its aromaticity perception to refine bond orders:
    - Any aromatic bond set to max(current, 1.5)
    - Non-aromatic bonds > 1.5 left as-is (kept from valence logic)
    """
    upgrades = 0
    try:
        rw = Chem.RWMol()
        for atom in atoms:
            rw.AddAtom(Chem.Atom(atom.symbol))
        for (i,j) in bonds:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        aromatic_pairs = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                          for b in mol.GetBonds() if b.GetIsAromatic()}
        for idx,(i,j) in enumerate(bonds):
            if tuple(sorted((i,j))) in aromatic_pairs and bond_orders[idx] < 1.5:
                bond_orders[idx] = 1.5
                upgrades += 1
    except Exception:
        pass
    return upgrades

def _compute_gasteiger(atoms: Atoms,
                       bonds: List[Tuple[int,int]],
                       bond_orders: List[float],
                       attempt_formal_charge: bool = True) -> List[float]:
    """
    Compute (Gasteiger) charges with optional legacy-like formal charge recovery.

    Steps:
      1. Build RWMol and add bonds mapped from fractional bond orders.
      2. Attempt full sanitize.
      3. If sanitize fails with an explicit valence error AND attempt_formal_charge=True:
           - Parse atom index from the error message.
           - Assign +1 formal charge (up to two distinct offending atoms / passes).
           - Retry sanitize each time.
         Falls back to minimal sanitize flags if still failing.
      4. Compute charges; any NaN replaced by 0.0.
    """
    try:
        rw = Chem.RWMol()
        for a in atoms:
            rw.AddAtom(Chem.Atom(a.symbol))
        for k,(i,j) in enumerate(bonds):
            bo = bond_orders[k]
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

        def _try_sanitize(m: Chem.Mol) -> Optional[str]:
            try:
                Chem.SanitizeMol(m)
                return None
            except Exception as e:
                return str(e)

        err = _try_sanitize(mol)
        attempted = 0
        processed_atoms = set()

        # Legacy-style formal charge promotion for valence errors
        while err and attempt_formal_charge and "Explicit valence for atom" in err and attempted < 2:
            # Extract atom index after '#' marker if present
            parts = err.split("#")
            atom_idx = None
            if len(parts) > 1:
                tail = parts[1].strip().split()
                if tail:
                    try:
                        atom_idx = int(tail[0])
                    except ValueError:
                        atom_idx = None
            if atom_idx is not None and atom_idx not in processed_atoms and atom_idx < mol.GetNumAtoms():
                mol.GetAtomWithIdx(atom_idx).SetFormalCharge(mol.GetAtomWithIdx(atom_idx).GetFormalCharge() + 1)
                processed_atoms.add(atom_idx)
                attempted += 1
                err = _try_sanitize(mol)
            else:
                break

        if err:
            # Final reduced sanitize attempt (properties only) if still failing
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass  # give up gracefully; RDKit may still allow charge calc

        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            return [0.0]*len(atoms)

        charges: List[float] = []
        for atom in mol.GetAtoms():
            try:
                c = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                c = 0.0
            if np.isnan(c):
                c = 0.0
            charges.append(c)
        if len(charges) != len(atoms):
            return [0.0]*len(atoms)
        return charges

    except Exception:
        return [0.0]*len(atoms)

def _annotate_valences(G: nx.Graph):
        for n in G.nodes:
            G.nodes[n]['valence'] = _valence_sum(G, n)

def _aggregate_hydrogen_charges(G: nx.Graph):
    # Choose preferred charge field
    preferred_order = ['gasteiger', 'mulliken']
    for n,data in G.nodes(data=True):
        # find a base numeric charge (fallback 0.0)
        base_charge = 0.0
        for m in preferred_order:
            if m in data.get('charges', {}):
                base_charge = data['charges'][m]
                break
        if data['symbol'] == 'H':
            G.nodes[n]['agg_charge'] = base_charge
            continue
        agg = base_charge
        for nbr in G.neighbors(n):
            if G.nodes[nbr]['symbol'] == 'H':
                # add hydrogen charge
                h_charge = 0.0
                for m in preferred_order:
                    if m in G.nodes[nbr]['charges']:
                        h_charge = G.nodes[nbr]['charges'][m]
                        break
                agg += h_charge
        G.nodes[n]['agg_charge'] = agg

def _sanitize_graph(G: nx.Graph, atoms: Atoms, expected: Dict[str, List[int]], vdw: Dict[str,float], max_iter: int = 3):
    # Re-run valence adjustment if needed
    _annotate_valences(G)
    problems = [n for n,d in G.nodes(data=True)
                if d['symbol'] in expected and
                min(abs(d['valence'] - v) for v in expected[d['symbol']]) > 0.6]
    if problems:
        bonds = list(G.edges())
        bond_orders = [G.edges[b]['bond_order'] for b in bonds]
        _adjust_valences(atoms, bonds, bond_orders, expected, vdw, max_iter=max_iter)
        for idx,(i,j) in enumerate(bonds):
            G.edges[i,j]['bond_order'] = float(bond_orders[idx])
        _annotate_valences(G)
    _aggregate_hydrogen_charges(G)

def _guess_multiplicity(atoms: Atoms, charge: int) -> int:
    """Multiplicity = 1 for even electrons, 2 for odd (simple heuristic)."""
    ne = int(np.sum(atoms.get_atomic_numbers())) - charge
    return 1 if ne % 2 == 0 else 2

def _valence_error_score(atoms: Atoms,
                         bonds: List[Tuple[int,int]],
                         bond_orders: List[float],
                         expected: Dict[str,List[int]]) -> float:
    """Sum of squared deviations from nearest allowed valence (non-metals only)."""
    deg = {i:0.0 for i in range(len(atoms))}
    for (i,j),bo in zip(bonds,bond_orders):
        deg[i]+=bo; deg[j]+=bo
    score = 0.0
    for i,a in enumerate(atoms):
        sym = a.symbol
        if sym in METALS or sym not in expected:
            continue
        cur = deg[i]
        target = min(expected[sym], key=lambda v: (abs(v-cur), -v))
        diff = cur - target
        score += diff*diff
    return score

def _prune_small_rings(atoms: Atoms,
                       bonds: List[Tuple[int,int]],
                       bond_orders: List[float],
                       ring_sizes: Tuple[int,...] = (3,4),
                       ratio_cutoff: float = 1.18,
                       max_passes: int = 4,
                       skip_metal_cycles: bool = True,
                       bond_dists: Optional[List[float]] = None,
                       expected: Optional[Dict[str,List[int]]] = None,
                       adaptive: bool = True,
                       ratio_cutoff_3: float = 1.18,
                       ratio_cutoff_4: float = 1.22,
                       min_improvement: float = 0.0) -> int:
    """
    Adaptive pruning of distorted small rings (size in ring_sizes):
      - Distortion test via (max_dist / min_dist) > threshold (size-specific for 3/4; else ratio_cutoff).
      - If adaptive=True: simulate single-edge removal (longest edge) and accept only
        if valence error score does not worsen (and improves by >= min_improvement if set).
      - No atom-type or topology special casing (no explicit spiro handling).
    """
    if bond_dists is None:
        bond_dists = _compute_bond_distances(atoms, bonds)
    if expected is None and adaptive:
        expected = _get_expected_valences()
    removed_total = 0
    for _ in range(max_passes):
        G = nx.Graph(); G.add_nodes_from(range(len(atoms))); G.add_edges_from(bonds)
        cycles = nx.cycle_basis(G)
        if not cycles:
            break
        removed = False
        dist_lookup = {tuple(sorted((i,j))): d for (i,j), d in zip(bonds, bond_dists)}
        for cyc in cycles:
            L = len(cyc)
            if L not in ring_sizes:
                continue
            if skip_metal_cycles and any(atoms[a].symbol in METALS for a in cyc):
                continue
            cyc_edges = [tuple(sorted((cyc[k], cyc[(k+1)%L]))) for k in range(L)]
            dvec = [dist_lookup[e] for e in cyc_edges if e in dist_lookup]
            if len(dvec) != L:
                continue
            dmin = min(dvec); dmax = max(dvec)
            if dmin < 1e-6:
                continue
            thr = ratio_cutoff_3 if L == 3 else (ratio_cutoff_4 if L == 4 else ratio_cutoff)
            if dmax/dmin <= thr:
                continue
            worst = cyc_edges[dvec.index(dmax)]
            if not adaptive or expected is None:
                for idx,(e,(i,j)) in enumerate(zip(bond_dists, bonds)):
                    if tuple(sorted((i,j))) == worst:
                        bonds.pop(idx); bond_orders.pop(idx); bond_dists.pop(idx)
                        removed_total += 1; removed = True
                        break
            else:
                current_score = _valence_error_score(atoms, bonds, bond_orders, expected)
                for idx,(e,(i,j)) in enumerate(zip(bond_dists, bonds)):
                    if tuple(sorted((i,j))) == worst:
                        tbonds = bonds[:idx] + bonds[idx+1:]
                        torders = bond_orders[:idx] + bond_orders[idx+1:]
                        trial_score = _valence_error_score(atoms, tbonds, torders, expected)
                        improvement = current_score - trial_score
                        if trial_score <= current_score and improvement >= min_improvement:
                            bonds.pop(idx); bond_orders.pop(idx); bond_dists.pop(idx)
                            removed_total += 1; removed = True
                        break
            if removed:
                break
        if not removed:
            break
    return removed_total

def build_graph_cheminf(atoms: Atoms,
                        charge: int = 0,
                        multiplicity: Optional[int] = None,
                        sanitize_iterations: int = 5) -> nx.Graph:
    vdw = _get_vdw()
    expected = _get_expected_valences()
    if multiplicity is None:
        multiplicity = _guess_multiplicity(atoms, charge)
    bonds, bond_dists = _initial_bonds(atoms, vdw)
    bond_orders = [1.0]*len(bonds)
    _prune_small_rings(atoms, bonds, bond_orders,
                       bond_dists=bond_dists,
                       expected=expected,
                       adaptive=True)
    arom = _detect_aromatic_cycles(atoms, bonds)
    _apply_aromatic(bonds, bond_orders, arom)
    _adjust_valences(atoms, bonds, bond_orders, expected, vdw,
                     multiplicity=multiplicity, bond_dists=bond_dists)
    rdkit_up = _rdkit_aromatic_refine(atoms, bonds, bond_orders)
    formal_charges = _compute_formal_charges(atoms, bonds, bond_orders, charge)
    charges_raw = _compute_gasteiger(atoms, bonds, bond_orders)
    raw_sum = sum(charges_raw) if charges_raw else 0.0
    charges_adj = charges_raw[:]
    if charges_adj and abs(raw_sum - charge) > 1e-6:
        delta = (charge - raw_sum)/len(charges_adj)
        charges_adj = [c + delta for c in charges_adj]
    G = _assemble_graph(atoms, bonds, bond_orders,
                        {'gasteiger_raw': charges_raw, 'gasteiger': charges_adj},
                        formal_charges=formal_charges,
                        bond_dists=bond_dists)
    G.graph['total_charge'] = charge
    G.graph['multiplicity'] = multiplicity
    _sanitize_graph(G, atoms, expected, vdw, max_iter=sanitize_iterations)
    return G

def build_graph_xtb(atoms: Atoms,
                    charge: int = 0,
                    multiplicity: Optional[int] = None,
                    basename: str = 'xtb',
                    clean_up: bool = True,
                    sanitize_iterations: int = 3) -> nx.Graph:
    if multiplicity is None:
        multiplicity = _guess_multiplicity(atoms, charge)
    work = 'xtb_tmp_local'
    if os.system('which xtb > /dev/null 2>&1') != 0:
        raise RuntimeError("xtb command not found; please install xtb or use 'cheminf' method")
    os.makedirs(work, exist_ok=True)
    import ase.io
    ase.io.write(os.path.join(work, f'{basename}.xyz'), atoms, format='xyz')
    os.system(f'cd {work} && xtb {basename}.xyz --chrg {charge} --uhf {multiplicity-1} --gfn2 > {basename}.out')
    if not all(os.path.exists(os.path.join(work, basename + ext)) for ext in ('.out','_wbo','_charges')):
        if os.path.exists(os.path.join(work,'wbo')):
            os.rename(os.path.join(work,'wbo'), os.path.join(work,f'{basename}_wbo'))
        if os.path.exists(os.path.join(work,'charges')):
            os.rename(os.path.join(work,'charges'), os.path.join(work,f'{basename}_charges'))
    bonds=[]; bond_orders=[]
    try:
        with open(os.path.join(work,f'{basename}_wbo')) as fh:
            for line in fh:
                p=line.split()
                if len(p)==3 and float(p[2])>0.5:
                    bonds.append((int(p[0])-1,int(p[1])-1))
                    bond_orders.append(float(p[2]))
    except FileNotFoundError:
        pass
    charges=[]
    try:
        with open(os.path.join(work,f'{basename}_charges')) as fh:
            for line in fh:
                charges.append(float(line.split()[0]))
    except FileNotFoundError:
        charges=[0.0]*len(atoms)
    if clean_up:
        for f in os.listdir(work):
            os.remove(os.path.join(work,f))
        os.rmdir(work)
    if not bonds:
        vdw = _get_vdw()
        expected = _get_expected_valences()
        bonds, bond_dists = _initial_bonds(atoms, vdw)
        bond_orders = [1.0]*len(bonds)
        _prune_small_rings(atoms, bonds, bond_orders,
                           bond_dists=bond_dists,
                           expected=expected,
                           adaptive=True)
        arom = _detect_aromatic_cycles(atoms, bonds)
        _apply_aromatic(bonds, bond_orders, arom)
        _rdkit_aromatic_refine(atoms, bonds, bond_orders)
    else:
        bond_dists = _compute_bond_distances(atoms, bonds)
        expected = _get_expected_valences()
    G = _assemble_graph(atoms, bonds, bond_orders, {'mulliken': charges},
                        bond_dists=bond_dists)
    G.graph['total_charge'] = charge
    G.graph['multiplicity'] = multiplicity
    vdw = _get_vdw()
    _sanitize_graph(G, atoms, expected, vdw, max_iter=sanitize_iterations)
    return G

def build_graph(atoms: Atoms,
                method: str = 'cheminf',
                charge: int = 0,
                multiplicity: Optional[int] = None,
                **kwargs) -> nx.Graph:
    """
    Unified public wrapper.
    method: 'cheminf' or 'xtb'
    Additional kwargs passed to underlying builder.
    """
    if method == 'cheminf':
        return build_graph_cheminf(atoms, charge=charge, multiplicity=multiplicity,
                                   sanitize_iterations=kwargs.get('sanitize_iterations', 5))
    if method == 'xtb':
        return build_graph_xtb(atoms, charge=charge, multiplicity=multiplicity,
                               basename=kwargs.get('basename','xtb'),
                               clean_up=kwargs.get('clean_up', True),
                               sanitize_iterations=kwargs.get('sanitize_iterations', 3))
    raise ValueError("method must be 'cheminf' or 'xtb'")

def _assemble_graph(atoms: Atoms,
                    bonds: List[Tuple[int,int]],
                    bond_orders: List[float],
                    charge_dict: Dict[str,List[float]],
                    formal_charges: Optional[List[int]] = None,
                    bond_dists: Optional[List[float]] = None) -> nx.Graph:
    """
    Build NetworkX graph and attach:
      Node: symbol, atomic_number, charges{method:value}, agg_charge placeholder, formal_charge
      Edge: bond_order (float), bond_type (tuple), metal_coord (bool).
    """
    G = nx.Graph()
    for i,a in enumerate(atoms):
        charges = {m: arr[i] if i < len(arr) else 0.0 for m, arr in charge_dict.items()}
        G.add_node(i, symbol=a.symbol, atomic_number=a.number,
                   charges=charges, agg_charge=0.0,
                   formal_charge=formal_charges[i] if formal_charges else 0)
    if bond_dists is None:
        bond_dists = _compute_bond_distances(atoms, bonds)
    for ((i,j), bo, dist) in zip(bonds, bond_orders, bond_dists):
        si, sj = atoms[i].symbol, atoms[j].symbol
        G.add_edge(i, j,
                   bond_order=float(bo),
                   bond_type=(si, sj),
                   metal_coord=(si in METALS or sj in METALS),
                   distance=float(dist))
    return G

def _bond_rdkit_order(b):
    """Map RDKit bond type to numeric order (aromatic=1.5)."""
    if b.GetIsAromatic():
        return 1.5
    t = b.GetBondType()
    if t == Chem.BondType.SINGLE: return 1.0
    if t == Chem.BondType.DOUBLE: return 2.0
    if t == Chem.BondType.TRIPLE: return 3.0
    if t == Chem.BondType.AROMATIC: return 1.5
    return 1.0

def _compute_formal_charges(atoms: Atoms,
                            bonds: List[Tuple[int,int]],
                            bond_orders: List[float],
                            total_charge: int) -> List[int]:
    """
    Compute simple formal charges using:
        formal = V - (L + B/2)
      where:
        V = valence electrons (loaded from data/valence_electrons.json; fallback heuristic)
        B = 2 * sum(bond orders involving atom
        L = max(0, target - B) ; target = 2 (H) else 8
    Metals forced to 0 (coordination model). Residual charge distributed.
    """
    valence_electrons = _get_valence_electrons()
    formal = [0] * len(atoms)
    for i, atom in enumerate(atoms):
        sym = atom.symbol
        if sym in METALS:
            formal[i] = 0
            continue
        V = valence_electrons.get(sym)
        if V is None:
            formal[i] = 0
            continue
        bond_order_sum = _bond_order_sum(i, bonds, bond_orders)
        B = 2.0 * bond_order_sum
        target = 2 if sym == 'H' else 8
        L = max(0, target - B)
        formal[i] = int(round(V - L - B/2))
    residual = total_charge - sum(formal)  
    
    if residual != 0:
        bonded_atoms = []
        for idx in range(len(atoms)):
            bond_count = _bond_order_sum(idx, bonds, bond_orders)
            if bond_count > 0:
                bonded_atoms.append((abs(formal[idx]), idx))
        if not bonded_atoms:
            bonded_atoms = [(0, idx) for idx in range(len(atoms))]
        bonded_atoms.sort(reverse=True)
        sign = 1 if residual > 0 else -1
        for _, idx in bonded_atoms:
            if residual == 0:
                break
            formal[idx] += sign
            residual -= sign
    return formal
