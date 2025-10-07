import os
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations
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

DEBUG_BOND_ORDER = True

def set_bond_order_debug(enabled: bool):
    """Enable/disable bond order refinement debugging"""
    global DEBUG_BOND_ORDER
    DEBUG_BOND_ORDER = enabled

def _debug_print(msg: str, level: int = 0):
    """Print debug message with indentation"""
    if DEBUG_BOND_ORDER:
        indent = "  " * level
        print(f"{indent}{msg}")

def _compute_formal_charge_simple(symbol: str, 
                                   valence_electrons: int,
                                   bond_order_sum: float, 
                                   lone_pairs: int = 0) -> int:
    """
    Formal charge = V - (L + B/2)
    where V=valence electrons, L=2*lone_pairs, B=bond electrons (2*bond_order_sum)
    
    For convenience, we assume lone pairs fill to octet:
    L = max(0, target - B) where target = 2 for H, 8 for others
    """
    if symbol == 'H':
        return valence_electrons - int(bond_order_sum)
    
    target = 8
    B = 2 * bond_order_sum
    L = max(0, target - B)
    return int(round(valence_electrons - L - B/2))

def _score_bond_assignment(atoms, bonds, bond_orders, 
                           valence_electrons_dict, expected_valences,
                           total_charge: int,
                           debug: bool = False) -> Tuple[float, List[int]]:
    """
    Score a bond order assignment based on:
    1. Sum of absolute formal charges (minimize this - PRIMARY)
    2. Number of charged atoms (prefer fewer - SECONDARY)
    3. Ring conjugation preservation (IMPORTANT)
    4. Electronegativity-aware charge placement
    5. Protonation preference
    6. Valence satisfaction error
    7. Charge balance error
    8. Chemical reasonableness penalties
    
    Returns (score, formal_charges) where lower score is better
    """
    EN = {
        'H': 2.2, 'C': 2.5, 'N': 3.0, 'O': 3.5, 'F': 4.0,
        'P': 2.2, 'S': 2.6, 'Cl': 3.2, 'Br': 3.0, 'I': 2.7
    }
    
    n_atoms = len(atoms)
    bond_sums = [0.0] * n_atoms
    
    for (i, j), bo in zip(bonds, bond_orders):
        bond_sums[i] += bo
        bond_sums[j] += bo
    
    # Build a temporary graph to find rings
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    for (i, j) in bonds:
        G.add_edge(i, j)
    
    # Find all rings (cycles)
    try:
        rings = list(nx.cycle_basis(G))
    except:
        rings = []
    
    # Identify which atoms are in rings
    atoms_in_rings = set()
    for ring in rings:
        atoms_in_rings.update(ring)
    
    # NEW: Identify which atoms have H neighbors
    has_hydrogen = [False] * n_atoms
    for (i, j), bo in zip(bonds, bond_orders):
        if atoms[i].symbol == 'H':
            has_hydrogen[j] = True
        elif atoms[j].symbol == 'H':
            has_hydrogen[i] = True
    
    formal_charges = []
    valence_errors = 0.0
    en_penalty = 0.0
    violation_penalty = 0.0
    chemical_weirdness = 0.0
    protonation_penalty = 0.0
    conjugation_penalty = 0.0  # NEW: penalty for breaking ring conjugation
    
    if debug:
        _debug_print("=== Scoring bond assignment ===", 2)
    
    # ========== INSERT RING CONJUGATION CHECK HERE ==========
    for ring in rings:
        if len(ring) not in (5, 6):
            continue
            
        conjugatable = {'C', 'N', 'O', 'S', 'P'}
        if not all(atoms[i].symbol in conjugatable for i in ring):
            continue
        
        ring_set = set(ring)
        
        # Count elevated bonds WITHIN the ring
        ring_edges = [(ring[k], ring[(k+1) % len(ring)]) for k in range(len(ring))]
        elevated_bonds = 0
        
        for i, j in ring_edges:
            for bond_idx, (bi, bj) in enumerate(bonds):
                if {bi, bj} == {i, j}:
                    bo = bond_orders[bond_idx]
                    if bo > 1.3:
                        elevated_bonds += 1
                    break
        
        # Check for exocyclic double bonds FROM ring atoms
        exocyclic_double = 0
        for ring_atom in ring:
            ring_atom_sym = atoms[ring_atom].symbol
            
            for bond_idx, (i, j) in enumerate(bonds):
                if ring_atom in (i, j):
                    other = j if i == ring_atom else i
                    
                    if other not in ring_set:
                        bo = bond_orders[bond_idx]
                        if bo >= 1.8:
                            other_sym = atoms[other].symbol
                            
                            if ring_atom_sym == 'C' and other_sym != 'O':
                                exocyclic_double += 1
                                if debug:
                                    _debug_print(f"!!! Ring C{ring_atom}={other_sym}{other} breaks conjugation", 3)
                            
                            elif ring_atom_sym == 'N' and other_sym in ('C', 'P', 'S'):
                                exocyclic_double += 1
                                if debug:
                                    _debug_print(f"!!! Ring N{ring_atom}={other_sym}{other} breaks conjugation", 3)
        
        # Score conjugation quality
        expected_elevated = len(ring) // 2
        
        if elevated_bonds >= expected_elevated - 1:
            if exocyclic_double > 0:
                conjugation_penalty += exocyclic_double * 12.0
                if debug:
                    _debug_print(f"Ring conjugated but has {exocyclic_double} bad exocyclic doubles", 3)
        else:
            deficit = (expected_elevated - 1) - elevated_bonds
            if deficit > 0:
                conjugation_penalty += deficit * 5.0
                if exocyclic_double > 0:
                    conjugation_penalty += exocyclic_double * 12.0
                    if debug:
                        _debug_print(f"Ring poorly conjugated AND has exocyclic doubles", 3)
    # ========== END RING CONJUGATION CHECK ==========
    
    for idx, atom in enumerate(atoms):
        sym = atom.symbol
        
        if sym in METALS:
            formal_charges.append(0)
            continue
        
        V = valence_electrons_dict.get(sym, 0)
        fc = _compute_formal_charge_simple(sym, V, bond_sums[idx])
        formal_charges.append(fc)
        
        # Protonation preference
        if has_hydrogen[idx]:
            if fc == 0:
                for (i, j), bo in zip(bonds, bond_orders):
                    if idx in (i, j):
                        other = j if i == idx else i
                        if atoms[other].symbol != 'H':
                            other_V = valence_electrons_dict.get(atoms[other].symbol, 0)
                            other_fc = _compute_formal_charge_simple(atoms[other].symbol, other_V, bond_sums[other])
                            if other_fc > 0:
                                if sym in ('N', 'O'):
                                    protonation_penalty += 8.0
                                    if debug:
                                        _debug_print(f"!!! PROTONATION MISMATCH: {sym}{idx}-H is neutral but neighbor {atoms[other].symbol}{other} is +{other_fc}", 3)
                                else:
                                    protonation_penalty += 3.0
            elif fc > 0 and sym in ('N', 'O', 'S'):
                # Expected protonation - small bonus
                en_penalty -= 1.5
                if debug:
                    _debug_print(f"  ✓ Expected protonation: {sym}{idx}-H is +{fc}", 3)
        
        # Electronegativity penalty
        en = EN.get(sym, 2.5)
        if fc > 0:
            if sym == 'N':
                en_penalty += fc * 0.2
            elif sym == 'S':
                en_penalty += fc * 0.8
            elif sym == 'P':
                en_penalty += fc * 0.6
            elif sym == 'O':
                en_penalty += fc * (en - 2.5) * 0.8
            else:
                en_penalty += fc * (en - 2.5) * 0.5
        elif fc < 0:
            en_penalty += abs(fc) * (3.5 - en) * 0.5
        
        if debug and fc != 0:
            _debug_print(f"Atom {idx} ({sym}): valence={bond_sums[idx]:.1f}, formal_charge={fc}, in_ring={idx in atoms_in_rings}", 3)
        
        # Chemical weirdness check
        if sym == 'N' and bond_sums[idx] >= 4.5:
            double_bonds_to_O = 0
            for (i, j), bo in zip(bonds, bond_orders):
                if idx in (i, j) and bo >= 1.8:
                    other = j if i == idx else i
                    if atoms[other].symbol == 'O':
                        double_bonds_to_O += 1
            
            if double_bonds_to_O >= 2:
                chemical_weirdness += 30.0
                if debug:
                    _debug_print(f"!!! CHEMICAL WEIRDNESS: N{idx} has {double_bonds_to_O} double bonds to O", 3)
        
        # Valence error
        if sym in expected_valences:
            allowed = expected_valences[sym]
            current = bond_sums[idx]
            min_error = min(abs(current - v) for v in allowed)
            
            absolute_limits = {
                'C': 4, 'N': 5, 'O': 3, 'F': 1,
                'S': 6, 'P': 5, 'Cl': 7, 'Br': 7, 'I': 7,
            }
            
            if sym in absolute_limits:
                absolute_max = absolute_limits[sym]
                if current > absolute_max + 0.1:
                    violation_penalty += 1000.0
                    if debug:
                        _debug_print(f"!!! VALENCE VIOLATION: {sym}{idx} has {current:.1f} > absolute max {absolute_max} (fc={fc:+d})", 3)
            
            valence_errors += min_error ** 2
            
            if sym in ('O', 'N', 'F', 'Cl', 'S') and current < 2 and fc == 0:
                valence_errors += 1.0
            
            if debug and min_error > 0.3:
                target = min(allowed, key=lambda v: abs(v - current))
                _debug_print(f"Atom {idx} ({sym}): current={current:.1f}, target={target}, error={min_error:.2f}", 3)
    
    # Scoring components
    formal_charge_penalty = sum(abs(fc) for fc in formal_charges)
    num_charged_atoms = sum(1 for fc in formal_charges if fc != 0)
    charge_balance_error = abs(sum(formal_charges) - total_charge)
    
    # Weighted score
    score = (
        1000.0 * violation_penalty +         # Catastrophic if violates absolute max
        30.0 * chemical_weirdness +          # Penalty for weird bonding (N(=O)₂)
        12.0 * conjugation_penalty +         # NEW: Penalty for breaking ring conjugation
        8.0 * protonation_penalty +          # Penalty for wrong charge placement near H
        10.0 * formal_charge_penalty +       # Minimize total formal charge
        3.0 * num_charged_atoms +            # Prefer fewer charged atoms
        2.0 * en_penalty +                   # EN-aware charge placement
        5.0 * charge_balance_error +         # Balance total charge
        5.0 * valence_errors                 # Satisfy valences
    )
    
    if debug:
        _debug_print(f"Score breakdown: violation={violation_penalty}, weirdness={chemical_weirdness}, conjugation={conjugation_penalty:.1f}, protonation={protonation_penalty:.1f}, fc_penalty={formal_charge_penalty}, n_charged={num_charged_atoms}, en_penalty={en_penalty:.2f}, charge_error={charge_balance_error}, valence_error={valence_errors:.2f}", 3)
        _debug_print(f"Total score: {score:.2f}", 3)
    
    return score, formal_charges

def _get_adjustable_bonds(atoms, bonds, bond_orders, 
                         max_bond_order: float = 3.0,
                         debug: bool = False) -> List[int]:
    """
    Return indices of bonds that can be adjusted:
    - Not involving metals (keep those at 1.0)
    - Not H-X bonds (those are almost always single)
    - Not already at max bond order
    """
    adjustable = []
    for idx, ((i, j), bo) in enumerate(zip(bonds, bond_orders)):
        si, sj = atoms[i].symbol, atoms[j].symbol
        
        if si in METALS or sj in METALS:
            continue
        if 'H' in (si, sj):
            continue
        if bo >= max_bond_order:
            continue
        
        adjustable.append(idx)
    
    if debug:
        _debug_print(f"Found {len(adjustable)} adjustable bonds out of {len(bonds)} total", 2)
        if adjustable and len(adjustable) <= 10:
            for idx in adjustable:
                i, j = bonds[idx]
                _debug_print(f"  Bond {idx}: {atoms[i].symbol}{i}-{atoms[j].symbol}{j} (order={bond_orders[idx]:.1f})", 3)
    
    return adjustable

def _enumerate_bond_order_candidates(atoms, bonds, bond_orders,
                                     valence_electrons_dict, expected_valences,
                                     total_charge: int,
                                     max_candidates: int = 100,
                                     debug: bool = False) -> List[Tuple[float, List[float], List[int]]]:
    """
    Generate candidate bond order assignments using local search:
    1. Start from current assignment
    2. Try incrementing bond orders for adjustable bonds
    3. Score each candidate
    4. Return top candidates sorted by score
    
    Returns list of (score, bond_orders, formal_charges)
    """
    adjustable = _get_adjustable_bonds(atoms, bonds, bond_orders, debug=debug)
    
    if not adjustable:
        score, fc = _score_bond_assignment(atoms, bonds, bond_orders, 
                                           valence_electrons_dict, expected_valences,
                                           total_charge, debug=debug)
        if debug:
            _debug_print("No adjustable bonds found", 2)
        return [(score, bond_orders[:], fc)]
    
    candidates = []
    
    # Current assignment
    score, fc = _score_bond_assignment(atoms, bonds, bond_orders,
                                       valence_electrons_dict, expected_valences,
                                       total_charge, debug=debug)
    candidates.append((score, bond_orders[:], fc))
    
    if debug:
        _debug_print(f"Current assignment score: {score:.2f}", 2)
    
    # Try single bond increments (1.0 -> 2.0, 2.0 -> 3.0)
    for idx in adjustable:
        new_orders = bond_orders[:]
        increment = 1.0 if new_orders[idx] < 2.0 else 1.0
        new_orders[idx] = min(3.0, new_orders[idx] + increment)
        
        score, fc = _score_bond_assignment(atoms, bonds, new_orders,
                                           valence_electrons_dict, expected_valences,
                                           total_charge, debug=False)
        candidates.append((score, new_orders, fc))
    
    # Try pairs of increments (conjugated systems)
    if len(adjustable) > 1 and len(adjustable) <= 10:
        pair_count = 0
        for idx1, idx2 in combinations(adjustable, 2):
            # Check if these bonds share an atom (potential conjugation)
            b1 = bonds[idx1]
            b2 = bonds[idx2]
            shared = set(b1) & set(b2)
            
            if shared:  # Only try pairs that share an atom
                new_orders = bond_orders[:]
                new_orders[idx1] = min(3.0, new_orders[idx1] + 1.0)
                new_orders[idx2] = min(3.0, new_orders[idx2] + 1.0)
                
                score, fc = _score_bond_assignment(atoms, bonds, new_orders,
                                                   valence_electrons_dict, expected_valences,
                                                   total_charge, debug=False)
                candidates.append((score, new_orders, fc))
                pair_count += 1
        
        if debug and pair_count > 0:
            _debug_print(f"Tried {pair_count} conjugated bond pairs", 2)
    
    # Sort by score and limit
    candidates.sort(key=lambda x: x[0])
    
    if debug:
        _debug_print(f"Generated {len(candidates)} total candidates", 2)
        _debug_print(f"Best 3 scores: {[f'{c[0]:.2f}' for c in candidates[:3]]}", 2)
    
    return candidates[:max_candidates]

def _find_kekule_structures(atoms, bonds, bond_orders, G: nx.Graph,
                           valence_electrons_dict, expected_valences,
                           total_charge: int,
                           max_structures: int = 10,
                           debug: bool = False) -> List[Tuple[float, List[float]]]:
    """
    Generate valid Kekulé structures for rings by trying different
    double bond placements. Uses formal charge minimization to rank.
    
    Strategy:
    1. Find all cycles (rings)
    2. For each ring, identify "double-bondable" edges
    3. Generate alternating double bond patterns
    4. Score each pattern and keep best
    
    Returns list of (score, bond_orders)
    """
    if debug:
        _debug_print("=== Searching for Kekulé structures ===", 1)
    
    cycles = [c for c in nx.cycle_basis(G) if len(c) in (5, 6)]
    if not cycles:
        return []
    
    # Build edge map (bond index lookup)
    edge_to_bond_idx = {}
    for idx, (i, j) in enumerate(bonds):
        edge_to_bond_idx[tuple(sorted((i, j)))] = idx
    
    structures = []
    
    for ring_idx, cycle in enumerate(cycles):
        if debug:
            _debug_print(f"Ring {ring_idx}: {[f'{atoms[i].symbol}{i}' for i in cycle]}", 2)
        
        # Get ring edges
        ring_edges = []
        for k in range(len(cycle)):
            i, j = cycle[k], cycle[(k+1) % len(cycle)]
            edge_key = tuple(sorted((i, j)))
            if edge_key in edge_to_bond_idx:
                bond_idx = edge_to_bond_idx[edge_key]
                si, sj = atoms[i].symbol, atoms[j].symbol
                
                # Check if edge can be double bond (not H, not metal, reasonable atoms)
                if 'H' not in (si, sj) and si not in METALS and sj not in METALS:
                    ring_edges.append((bond_idx, i, j))
        
        if len(ring_edges) < 3:
            continue  # Not enough bondable edges
        
        # Try alternating patterns
        # For a 6-ring: positions 0,2,4 or 1,3,5
        # For a 5-ring: harder, try different combinations
        
        n_edges = len(ring_edges)
        
        # Generate patterns: which edges get double bonds?
        if n_edges == 6:  # Benzene-like
            patterns = [
                [0, 2, 4],  # Alternating pattern 1
                [1, 3, 5],  # Alternating pattern 2
            ]
        elif n_edges == 5:  # Cyclopentadiene-like
            patterns = [
                [0, 2],     # Two separated doubles
                [1, 3],
                [0, 3],
                [1, 4],
                [2, 4],
            ]
        else:
            # Generic: try all combinations of ~n/2 double bonds
            from itertools import combinations as comb
            n_doubles = n_edges // 2
            patterns = list(comb(range(n_edges), n_doubles))
            patterns = patterns[:5]  # Limit to 5 patterns
        
        if debug:
            _debug_print(f"  Trying {len(patterns)} Kekulé patterns", 2)
        
        # Score each pattern
        for pattern in patterns:
            test_orders = bond_orders[:]
            
            # Set double bonds according to pattern
            for pos in pattern:
                if pos < len(ring_edges):
                    bond_idx, _, _ = ring_edges[pos]
                    test_orders[bond_idx] = 2.0
            
            if _has_invalid_valences(atoms, bonds, test_orders):
                continue

            score, fc = _score_bond_assignment(atoms, bonds, test_orders,
                                              valence_electrons_dict, expected_valences,
                                              total_charge, debug=False)
            
            structures.append((score, test_orders))
            
            if debug:
                double_bonds = [f"{atoms[ring_edges[p][1]].symbol}{ring_edges[p][1]}-{atoms[ring_edges[p][2]].symbol}{ring_edges[p][2]}" 
                               for p in pattern if p < len(ring_edges)]
                _debug_print(f"  Pattern {pattern}: score={score:.2f}, doubles={double_bonds}", 3)
    
    # Sort by score
    structures.sort(key=lambda x: x[0])
    
    if debug and structures:
        _debug_print(f"Best Kekulé structure: score={structures[0][0]:.2f}", 2)
    
    return structures[:max_structures]

def _has_invalid_valences(atoms, bonds, bond_orders):
    """
    Quick check: does this assignment have any chemically impossible valences?
    Returns True if invalid (should be rejected immediately).
    
    Focus on the most common violations:
    - C > 4
    - N > 5
    - O > 3
    """
    bond_sums = [0.0] * len(atoms)
    for (i, j), bo in zip(bonds, bond_orders):
        bond_sums[i] += bo
        bond_sums[j] += bo
    
    for idx, atom in enumerate(atoms):
        sym = atom.symbol
        current = bond_sums[idx]
        
        # Hard limits - immediate rejection
        if sym == 'C' and current > 4.05:
            return True
        if sym == 'N' and current > 5.05:
            return True
        if sym == 'O' and current > 3.05:
            return True
        if sym == 'P' and current > 5.05:
            return True
        if sym == 'S' and current > 6.05:
            return True
        if sym == 'F' and current > 1.05:
            return True
    
    return False


def enhanced_adjust_valences(atoms, bonds, bond_orders,
                            expected_valences, valence_electrons_dict,
                            total_charge: int,
                            max_iterations: int = 10,
                            try_kekule: bool = True,
                            debug: bool = None) -> Dict[str, any]:
    """
    Enhanced valence adjustment using:
    1. Kekulé structure enumeration for rings (if enabled)
    2. Formal charge minimization
    3. Local bond order optimization with STRICT valence limits
    
    Returns dict with statistics
    """
    if debug is None:
        debug = DEBUG_BOND_ORDER
    
    if debug:
        _debug_print("=" * 60, 0)
        _debug_print("ENHANCED VALENCE ADJUSTMENT (Kekulé-first)", 0)
        _debug_print("=" * 60, 0)
        _debug_print(f"Atoms: {len(atoms)}, Bonds: {len(bonds)}, Total charge: {total_charge}", 1)
    
    stats = {'iterations': 0, 'initial_score': 0.0, 'final_score': 0.0, 
             'improvements': 0, 'kekule_applied': False, 'bond_order_changes': []}
    
    # Lock metal bonds to 1.0
    metal_bonds = 0
    for idx, (i, j) in enumerate(bonds):
        if atoms[i].symbol in METALS or atoms[j].symbol in METALS:
            if bond_orders[idx] != 1.0:
                bond_orders[idx] = 1.0
                metal_bonds += 1
    
    if debug and metal_bonds > 0:
        _debug_print(f"Locked {metal_bonds} metal bonds to order 1.0", 1)
    
    # Initial score
    initial_score, initial_fc = _score_bond_assignment(atoms, bonds, bond_orders,
                                                       valence_electrons_dict, expected_valences,
                                                       total_charge, debug=debug)
    stats['initial_score'] = initial_score
    
    if debug:
        _debug_print(f"Initial score: {initial_score:.2f}", 1)
        fc_summary = [f"{atoms[i].symbol}{i}:{fc:+d}" for i, fc in enumerate(initial_fc) if fc != 0]
        if fc_summary:
            _debug_print(f"Initial formal charges: {', '.join(fc_summary)}", 1)
    
    current_score = initial_score
    best_orders = bond_orders[:]
    
    # STEP 1: Try Kekulé structures for rings
    if try_kekule:
        G_temp = nx.Graph()
        G_temp.add_nodes_from(range(len(atoms)))
        G_temp.add_edges_from(bonds)
        
        kekule_structures = _find_kekule_structures(atoms, bonds, bond_orders, G_temp, valence_electrons_dict, expected_valences, total_charge, debug=debug)
        
        if kekule_structures and kekule_structures[0][0] < current_score:
            improvement = current_score - kekule_structures[0][0]
            current_score, new_orders = kekule_structures[0]
            
            changes = []
            for idx, (old, new) in enumerate(zip(bond_orders, new_orders)):
                if abs(old - new) > 0.01:
                    i, j = bonds[idx]
                    changes.append(f"{atoms[i].symbol}{i}-{atoms[j].symbol}{j}: {old:.1f}→{new:.1f}")
            
            bond_orders[:] = new_orders
            best_orders = new_orders[:]
            stats['kekule_applied'] = True
            stats['improvements'] += 1
            
            if debug:
                _debug_print(f"✓ Applied Kekulé structure (improvement: {improvement:.2f})", 1)
                if changes:
                    for change in changes:
                        _debug_print(f"  {change}", 2)

    # STEP 2: Enhanced local optimization with strict valence checking
    stagnation_counter = 0
    for iteration in range(max_iterations):
        if debug:
            _debug_print(f"\n--- Iteration {iteration + 1} ---", 1)
        
        stats['iterations'] = iteration + 1
        pre_orders = bond_orders[:]
        
        adjustable = _get_adjustable_bonds(atoms, bonds, bond_orders, debug=debug)
        if not adjustable:
            if debug:
                _debug_print("No adjustable bonds, stopping", 2)
            break
        
        # Generate ALL candidates and pick the best VALID one
        candidates = []
        
        # Try single bond increments
        for idx in adjustable:
            test_orders = bond_orders[:]
            test_orders[idx] = min(3.0, test_orders[idx] + 1.0)
            
            # CRITICAL: Skip if violates valence limits
            if _has_invalid_valences(atoms, bonds, test_orders):
                continue
            
            score, fc = _score_bond_assignment(atoms, bonds, test_orders,
                                              valence_electrons_dict, expected_valences,
                                              total_charge, debug=False)
            
            candidates.append((score, test_orders, idx, 'single'))
        
        # ALSO try pairs of increments (important for conjugated systems!)
        if len(adjustable) > 1:
            # Try adjacent bond pairs (share an atom)
            bond_graph = {}
            for idx in adjustable:
                i, j = bonds[idx]
                if i not in bond_graph:
                    bond_graph[i] = []
                if j not in bond_graph:
                    bond_graph[j] = []
                bond_graph[i].append(idx)
                bond_graph[j].append(idx)
            
            # Find pairs of adjustable bonds that share an atom
            tried_pairs = set()
            for atom_idx, bond_indices in bond_graph.items():
                if len(bond_indices) >= 2:
                    for idx1, idx2 in combinations(bond_indices, 2):
                        pair = tuple(sorted((idx1, idx2)))
                        if pair in tried_pairs:
                            continue
                        tried_pairs.add(pair)
                        
                        test_orders = bond_orders[:]
                        test_orders[idx1] = min(3.0, test_orders[idx1] + 1.0)
                        test_orders[idx2] = min(3.0, test_orders[idx2] + 1.0)
                        
                        # CRITICAL: Skip if violates valence limits
                        if _has_invalid_valences(atoms, bonds, test_orders):
                            continue
                        
                        score, fc = _score_bond_assignment(atoms, bonds, test_orders,
                                                          valence_electrons_dict, expected_valences,
                                                          total_charge, debug=False)
                        
                        candidates.append((score, test_orders, (idx1, idx2), 'pair'))
        
        if not candidates:
            if debug:
                _debug_print("✗ No valid candidates (all violate valence limits)", 2)
            break
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x[0])
        
        if debug and candidates[:3]:
            _debug_print(f"Top 3 candidate scores: {[f'{c[0]:.2f}' for c in candidates[:3]]}", 2)
        
        # Take the best candidate
        best_candidate_score, best_candidate_orders, changed_bonds, change_type = candidates[0]
        
        if best_candidate_score < current_score - 0.001:  # Small threshold for numerical stability
            improvement = current_score - best_candidate_score
            current_score = best_candidate_score
            
            changes = []
            for idx, (old, new) in enumerate(zip(bond_orders, best_candidate_orders)):
                if abs(old - new) > 0.01:
                    i, j = bonds[idx]
                    changes.append(f"{atoms[i].symbol}{i}-{atoms[j].symbol}{j}: {old:.1f}→{new:.1f}")
            
            bond_orders[:] = best_candidate_orders
            best_orders = best_candidate_orders[:]
            stats['improvements'] += 1
            stagnation_counter = 0  # Reset stagnation
            
            if debug:
                _debug_print(f"✓ Improvement: {improvement:.2f} (new score: {current_score:.2f})", 2)
                _debug_print(f"  Change type: {change_type}", 2)
                if changes:
                    for change in changes:
                        _debug_print(f"  {change}", 3)
        else:
            stagnation_counter += 1
            if debug:
                _debug_print(f"✗ No improvement found (stagnation: {stagnation_counter})", 2)
            
            # Allow a few iterations of stagnation before giving up
            if stagnation_counter >= 2:
                if debug:
                    _debug_print("Stopping due to stagnation", 2)
                break
    
    stats['final_score'] = current_score
    bond_orders[:] = best_orders
    
    if debug:
        _debug_print(f"\n{'=' * 60}", 0)
        _debug_print(f"SUMMARY: {stats['improvements']} improvements over {stats['iterations']} iterations", 1)
        _debug_print(f"Kekulé applied: {stats['kekule_applied']}", 1)
        _debug_print(f"Score: {stats['initial_score']:.2f} → {stats['final_score']:.2f} (Δ={stats['initial_score'] - stats['final_score']:.2f})", 1)
        _debug_print(f"{'=' * 60}\n", 0)
    
    return stats

def handle_aromatic_rings_carefully(atoms, bonds, bond_orders, G: nx.Graph,
                                    debug: bool = None) -> int:
    """
    Refined aromatic detection:
    1. Find 5/6-membered rings with conjugation potential
    2. Check Huckel rule (4n+2 pi electrons)
    3. Set appropriate bond orders (1.5 for true aromatic, alternating for others)
    
    Returns number of aromatic bonds set
    """
    if debug is None:
        debug = DEBUG_BOND_ORDER
    
    if debug:
        _debug_print("=" * 60, 0)
        _debug_print("AROMATIC RING DETECTION (Hückel rule)", 0)
        _debug_print("=" * 60, 0)
    
    cycles = nx.cycle_basis(G)
    processed = 0
    aromatic_rings = 0
    
    for ring_idx, cycle in enumerate(cycles):
        if len(cycle) not in (5, 6):
            continue
        
        if debug:
            _debug_print(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {[f'{atoms[i].symbol}{i}' for i in cycle]}", 1)
        
        # Check if all atoms can participate in pi system
        aromatic_atoms = {'C', 'N', 'O', 'S', 'P'}  # Potentially aromatic
        if not all(atoms[i].symbol in aromatic_atoms for i in cycle):
            if debug:
                non_aromatic = [atoms[i].symbol for i in cycle if atoms[i].symbol not in aromatic_atoms]
                _debug_print(f"✗ Contains non-aromatic atoms: {non_aromatic}", 2)
            continue
        
        # Count pi electrons (simplified - assumes sp2 hybridization)
        pi_electrons = 0
        pi_breakdown = []
        
        for idx in cycle:
            sym = atoms[idx].symbol
            contribution = 0
            
            if sym == 'C':
                contribution = 1
                pi_breakdown.append(f"{sym}{idx}:1")
                pi_electrons += 1
            elif sym == 'N':
                # Check if N+ or neutral N with lone pair
                degree = sum(1 for _ in G.neighbors(idx))
                if degree == 2:  # Neutral N with lone pair
                    contribution = 2
                    pi_breakdown.append(f"{sym}{idx}:2(LP)")
                    pi_electrons += 2
                elif degree == 3:  # N+ or N with 3 bonds
                    contribution = 1
                    pi_breakdown.append(f"{sym}{idx}:1")
                    pi_electrons += 1
            elif sym == 'O':
                # O typically contributes 2 electrons (lone pair)
                degree = sum(1 for _ in G.neighbors(idx))
                if degree == 2:  # Two-coordinate O
                    contribution = 2
                    pi_breakdown.append(f"{sym}{idx}:2(LP)")
                    pi_electrons += 2
        
        if debug:
            _debug_print(f"π electrons: {pi_electrons} ({', '.join(pi_breakdown)})", 2)
        
        # Huckel rule: 4n+2 pi electrons
        is_aromatic = (pi_electrons - 2) % 4 == 0 and pi_electrons >= 6
        
        if debug:
            n_value = (pi_electrons - 2) / 4 if pi_electrons >= 2 else None
            if is_aromatic:
                _debug_print(f"✓ AROMATIC (4n+2 rule: n={n_value:.0f})", 2)
            else:
                _debug_print(f"✗ Not aromatic (4n+2 rule violated)", 2)
        
        # Set bond orders for ring edges
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        
        bonds_changed = 0
        for i, j in cycle_edges:
            edge_key = tuple(sorted((i, j)))
            for idx, (bi, bj) in enumerate(bonds):
                if tuple(sorted((bi, bj))) == edge_key:
                    old_order = bond_orders[idx]
                    if is_aromatic:
                        bond_orders[idx] = max(bond_orders[idx], 1.5)
                        if bond_orders[idx] != old_order:
                            bonds_changed += 1
                            if debug:
                                _debug_print(f"  {atoms[i].symbol}{i}-{atoms[j].symbol}{j}: {old_order:.1f}→{bond_orders[idx]:.1f}", 3)
                    processed += 1
                    break
        
        if is_aromatic:
            aromatic_rings += 1
    
    if debug:
        _debug_print(f"\n{'=' * 60}", 0)
        _debug_print(f"SUMMARY: {aromatic_rings} aromatic rings found, {processed} bonds processed", 1)
        _debug_print(f"{'=' * 60}\n", 0)
    
    return processed

def convert_aromatic_to_kekule(atoms, bonds, bond_orders, debug: bool = None):
    """
    Post-processing: convert any remaining 1.5 bonds to proper Kekulé (1.0/2.0).
    This is called AFTER Kekulé structure search and is a cleanup step.
    
    Strategy: Find connected components of 1.5 bonds and alternate them.
    """
    if debug is None:
        debug = DEBUG_BOND_ORDER
    
    aromatic_bonds = [idx for idx, bo in enumerate(bond_orders) if abs(bo - 1.5) < 0.01]
    
    if not aromatic_bonds:
        return
    
    if debug:
        _debug_print("=" * 60, 0)
        _debug_print(f"CONVERTING {len(aromatic_bonds)} AROMATIC (1.5) BONDS TO KEKULÉ", 0)
        _debug_print("=" * 60, 0)
    
    # Build graph of aromatic bonds
    G_arom = nx.Graph()
    for idx in aromatic_bonds:
        i, j = bonds[idx]
        G_arom.add_edge(i, j, bond_idx=idx)
    
    # Process each connected component
    for component in nx.connected_components(G_arom):
        if len(component) < 3:
            continue  # Too small
        
        # Find a cycle in this component
        subgraph = G_arom.subgraph(component)
        try:
            cycle = nx.find_cycle(subgraph)
        except nx.NetworkXNoCycle:
            continue
        
        # Alternate double/single in cycle
        for k, (i, j) in enumerate(cycle):
            bond_idx = G_arom.edges[i, j]['bond_idx']
            if k % 2 == 0:
                bond_orders[bond_idx] = 2.0
            else:
                bond_orders[bond_idx] = 1.0
        
        if debug:
            _debug_print(f"Converted cycle: {[f'{atoms[i].symbol}{i}-{atoms[j].symbol}{j}' for i,j in cycle]}", 1)


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

def _sanitize_graph(G: nx.Graph, atoms: Atoms, expected: Dict[str, List[int]], vdw: Dict[str,float], valence_electrons: Dict[str, int], total_charge: int, max_iter: int = 3):
    # Re-run valence adjustment if needed
    _annotate_valences(G)
    _aggregate_hydrogen_charges(G)
    return None

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
                        sanitize_iterations: int = 10,
                        debug: bool = True) -> nx.Graph:
    vdw = _get_vdw()
    expected = _get_expected_valences()
    valence_electrons = _get_valence_electrons()
    
    if multiplicity is None:
        multiplicity = _guess_multiplicity(atoms, charge)
    
    bonds, bond_dists = _initial_bonds(atoms, vdw)
    bond_orders = [1.0]*len(bonds)
    
    _prune_small_rings(atoms, bonds, bond_orders,
                       bond_dists=bond_dists,
                       expected=expected,
                       adaptive=True)
    
    # Enhanced valence adjustment with Kekulé
    stats = enhanced_adjust_valences(
        atoms, bonds, bond_orders, 
        expected, valence_electrons, 
        charge, 
        max_iterations=50, 
        try_kekule=True,
        debug=debug
    )
    
    # Convert 1.5 bonds to proper Kekulé
    convert_aromatic_to_kekule(atoms, bonds, bond_orders, debug=debug)
    
    # RDKit aromatic perception
    rdkit_up = _rdkit_aromatic_refine(atoms, bonds, bond_orders)
    
    # Formal charges
    formal_charges = _compute_formal_charges(atoms, bonds, bond_orders, charge)
    
    # Gasteiger charges
    charges_raw = _compute_gasteiger(atoms, bonds, bond_orders)
    raw_sum = sum(charges_raw) if charges_raw else 0.0
    charges_adj = charges_raw[:]
    if charges_adj and abs(raw_sum - charge) > 1e-6:
        delta = (charge - raw_sum)/len(charges_adj)
        charges_adj = [c + delta for c in charges_adj]
    
    # Assemble graph
    G = _assemble_graph(atoms, bonds, bond_orders,
                        {'gasteiger_raw': charges_raw, 'gasteiger': charges_adj},
                        formal_charges=formal_charges,
                        bond_dists=bond_dists)
    G.graph['total_charge'] = charge
    G.graph['multiplicity'] = multiplicity
    G.graph['valence_stats'] = stats
    
    # Sanitize
    _sanitize_graph(G, atoms, expected, vdw, valence_electrons, charge, max_iter=sanitize_iterations)
    
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
    valence_electrons = _get_valence_electrons()
    _sanitize_graph(G, atoms, expected, vdw, valence_electrons, charge, max_iter=sanitize_iterations)

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
