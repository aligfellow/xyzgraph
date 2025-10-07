import os
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

from .data_loader import load_vdw_radii, load_expected_valences, load_valence_electrons

# =============================================================================
# DESIGN PHILOSOPHY
# =============================================================================
# Two modes:
#   1. QUICK: Fast heuristic-based bond/valence assignment
#   2. FULL:  Comprehensive Kekulé search + formal charge optimization
#
# Core approach:
#   - Work directly with NetworkX graph throughout (no intermediate lists)
#   - Distance-based initial bonding with metal-aware thresholds
#   - Optional valence refinement with formal charge minimization
#   - RDKit aromatic perception + Gasteiger charges
# =============================================================================

# Central metal set (used in multiple stages)
METALS = {
    'Li','Na','K','Mg','Ca','Zn','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os',
    'Ir','Pt','Au','Hg','Al','Ga','In','Sn','Pb','La','Ce','Pr','Nd','Sm','Eu',
    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'
}

# Lazy-loaded data caches
_VDW = None
_VALENCES = None
_VALENCE_ELECTRONS = None
_DEBUG = False

def set_debug(enabled: bool):
    """Enable/disable debug printing globally"""
    global _DEBUG
    _DEBUG = enabled

def _debug_print(msg: str, level: int = 0):
    """Print debug message with indentation"""
    if _DEBUG:
        print(f"{'  ' * level}{msg}")

def _get_vdw() -> Dict[str, float]:
    global _VDW
    if _VDW is None:
        _VDW = load_vdw_radii()
    return _VDW

def _get_expected_valences() -> Dict[str, List[int]]:
    global _VALENCES
    if _VALENCES is None:
        _VALENCES = load_expected_valences()
    return _VALENCES

def _get_valence_electrons() -> Dict[str, int]:
    global _VALENCE_ELECTRONS
    if _VALENCE_ELECTRONS is None:
        _VALENCE_ELECTRONS = load_valence_electrons()
    return _VALENCE_ELECTRONS

# Public accessors
def get_vdw() -> Dict[str, float]:
    return _get_vdw()

def get_expected_valences() -> Dict[str, List[int]]:
    return _get_expected_valences()

def get_valence_electrons() -> Dict[str, int]:
    return _get_valence_electrons()

# =============================================================================
# GRAPH-BASED BOND CONSTRUCTION
# =============================================================================

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _should_bond_metal(sym_i: str, sym_j: str) -> bool:
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

def _build_initial_graph(atoms: Atoms, vdw: Dict[str, float],
                         vdw_scale_h: float = 0.45,
                         vdw_scale_h_metal: float = 0.5,
                         vdw_scale: float = 0.55,
                         vdw_scale_metal: float = 0.65) -> nx.Graph:
    """
    Build initial graph with distance-based bonds.
    All edges start with bond_order=1.0.
    """
    G = nx.Graph()
    pos = atoms.positions

    # Add all nodes
    for i, atom in enumerate(atoms):
        G.add_node(i, 
                   symbol=atom.symbol,
                   atomic_number=atom.number,
                   position=pos[i])
    
    # Add edges based on distance
    for i in range(len(atoms)):
        si = atoms[i].symbol
        is_metal_i = si in METALS
        
        for j in range(i + 1, len(atoms)):
            sj = atoms[j].symbol
            is_metal_j = sj in METALS
            has_metal = is_metal_i or is_metal_j
            has_h = 'H' in (si, sj)
            
            d = _distance(pos[i], pos[j])
            r_sum = vdw.get(si, 2.0) + vdw.get(sj, 2.0)
            
            # Choose threshold
            if has_h and not has_metal:
                threshold = vdw_scale_h * r_sum
            elif has_h and has_metal:
                threshold = vdw_scale_h_metal * r_sum
            elif has_metal:
                threshold = vdw_scale_metal * r_sum
            else:
                threshold = vdw_scale * r_sum
            
            if d < threshold:
                if has_metal and not _should_bond_metal(si, sj):
                    continue
                
                G.add_edge(i, j,
                          bond_order=1.0,
                          distance=d,
                          metal_coord=has_metal)
                
    # -- graph caching ---
    rings = nx.cycle_basis(G)
    G.graph['_rings'] = rings
    G.graph['_neighbors'] = {n: list(G.neighbors(n)) for n in G.nodes()}
    G.graph['_has_H'] = {n: any(atoms[nbr].symbol == 'H' for nbr in G.neighbors(n)) for n in G.nodes()}

    return G

def _valence_sum(G: nx.Graph, node: int) -> float:
    """Sum bond orders around a node"""
    return sum(G.edges[node, nbr].get('bond_order', 1.0) 
               for nbr in G.neighbors(node))

def _prune_distorted_rings(G: nx.Graph, atoms: Atoms,
                          ring_sizes: Tuple[int, ...] = (3, 4),
                          ratio_3: float = 1.18,
                          ratio_4: float = 1.22,
                          max_passes: int = 4) -> int:
    """
    Remove geometrically distorted small rings.
    Works directly on graph G (modifies in place).
    """
    removed = 0
    
    for _ in range(max_passes):
        cycles = nx.cycle_basis(G)
        if not cycles:
            break
        
        pruned_this_pass = False
        
        for cycle in cycles:
            if len(cycle) not in ring_sizes:
                continue
            
            # Skip metal-containing cycles
            if any(atoms[i].symbol in METALS for i in cycle):
                continue
            
            # Get cycle edges and distances
            cycle_edges = [(cycle[k], cycle[(k+1) % len(cycle)]) 
                          for k in range(len(cycle))]
            
            distances = []
            for i, j in cycle_edges:
                if G.has_edge(i, j):
                    distances.append(G.edges[i, j]['distance'])
            
            if len(distances) != len(cycle):
                continue
            
            dmin, dmax = min(distances), max(distances)
            if dmin < 1e-6:
                continue
            
            # Choose threshold
            threshold = ratio_3 if len(cycle) == 3 else ratio_4
            
            if dmax / dmin > threshold:
                # Remove longest edge
                worst_idx = distances.index(dmax)
                i, j = cycle_edges[worst_idx]
                if G.has_edge(i, j):
                    G.remove_edge(i, j)
                    removed += 1
                    pruned_this_pass = True
                    break
        
        if not pruned_this_pass:
            break
    
    return removed

# =============================================================================
# QUICK MODE: Simple heuristic valence adjustment
# =============================================================================

def _quick_valence_adjust(G: nx.Graph, atoms: Atoms,
                          expected: Dict[str, List[int]],
                          vdw: Dict[str, float],
                          max_iter: int = 3) -> Dict[str, int]:
    """
    Fast heuristic bond order adjustment.
    No formal charge optimization - just satisfy valences.
    """
    stats = {'iterations': 0, 'promotions': 0}
    
    # Lock metal bonds
    for i, j in G.edges():
        if G.edges[i, j].get('metal_coord', False):
            G.edges[i, j]['bond_order'] = 1.0
    
    for iteration in range(max_iter):
        stats['iterations'] = iteration + 1
        changed = False
        
        # Calculate deficits
        deficits = {}
        for node in G.nodes():
            sym = atoms[node].symbol
            if sym in METALS:
                deficits[node] = 0.0
                continue
            
            current = _valence_sum(G, node)
            allowed = expected.get(sym, [])
            if not allowed:
                deficits[node] = 0.0
                continue
            
            target = min(allowed, key=lambda v: abs(v - current))
            deficits[node] = target - current
        
        # Try to promote bonds
        for i, j, data in G.edges(data=True):
            if data.get('metal_coord', False):
                continue
            
            si, sj = atoms[i].symbol, atoms[j].symbol
            if 'H' in (si, sj):
                continue
            
            bo = data['bond_order']
            if bo >= 3.0:
                continue
            
            di, dj = deficits[i], deficits[j]
            
            # Check geometry
            dist_ratio = data['distance'] / (vdw.get(si, 2.0) + vdw.get(sj, 2.0))
            if dist_ratio > 0.60:
                continue
            
            # Promote if both atoms need more valence
            if di > 0.3 and dj > 0.3:
                increment = min(di, dj, 3.0 - bo)
                if increment >= 0.5:
                    data['bond_order'] = bo + increment
                    stats['promotions'] += 1
                    changed = True
        
        if not changed:
            break
    
    return stats

# =============================================================================
# FULL MODE: Formal charge optimization
# =============================================================================

def _compute_formal_charge(symbol: str, valence_electrons: int,
                          bond_order_sum: float) -> int:
    """
    Formal charge = V - (L + B/2)
    where L = lone pair electrons (fills to octet)
    """
    if symbol == 'H':
        return valence_electrons - int(bond_order_sum)
    
    B = 2 * bond_order_sum
    target = 8
    L = max(0, target - B)
    return int(round(valence_electrons - L - B / 2))

def _ring_conjugation_penalty(G: nx.Graph, atoms, rings, debug=False) -> float:
    """
    Assess conjugation and exocyclic double penalties in aromatic rings (5-6 members).
    Returns a numeric penalty (larger = worse).
    """
    conjugation_penalty = 0.0
    for ring in rings:
        if len(ring) not in (5, 6):
            continue

        conjugatable = {'C', 'N', 'O', 'S', 'P'}
        if not all(atoms[i].symbol in conjugatable for i in ring):
            continue

        ring_set = set(ring)
        elevated_bonds = 0
        exocyclic_double = 0

        # --- Bonds within the ring ---
        ring_edges = [(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]
        for i, j in ring_edges:
            bo = G[i][j].get('bond_order', 1.0)
            if bo > 1.3:
                elevated_bonds += 1

        # --- Exocyclic double bonds ---
        for ring_atom in ring:
            ring_sym = atoms[ring_atom].symbol
            for nbr, data in G[ring_atom].items():
                if nbr not in ring_set:
                    bo = data.get('bond_order', 1.0)
                    if bo >= 1.8:
                        nbr_sym = atoms[nbr].symbol
                        if (ring_sym == 'C' and nbr_sym != 'O') or \
                           (ring_sym == 'N' and nbr_sym in ('C', 'P', 'S')):
                            exocyclic_double += 1

        # --- Scoring logic (unchanged) ---
        expected_elevated = len(ring) // 2
        if elevated_bonds >= expected_elevated - 1:
            if exocyclic_double > 0:
                conjugation_penalty += exocyclic_double * 12.0
        else:
            deficit = (expected_elevated - 1) - elevated_bonds
            if deficit > 0:
                conjugation_penalty += deficit * 5.0
                if exocyclic_double > 0:
                    conjugation_penalty += exocyclic_double * 12.0

    return conjugation_penalty

def _check_valence_violation(G: nx.Graph,
                             atoms,
                             limits: Optional[Dict[str, float]] = None,
                             tol: float = 0.3) -> bool:
    """
    Quick structural sanity check for pentavalent carbon.
    Returns True if any atom exceeds its valence limit by >tol.
    """
    if limits is None:
        limits = {'C': 4}

    for i in G.nodes():
        sym = atoms[i].symbol
        if sym in limits:
            val = sum(G[i][j].get('bond_order', 1.0) for j in G.neighbors(i))
            if val > limits[sym] + tol:
                return True
    return False

def _edge_likelihood(G, atoms, expected, valence_cache, k=5):
    scores = {}
    for i, j, data in G.edges(data=True):
        if data.get('metal_coord', False) or data['bond_order'] >= 3.0:
            continue
        si, sj = atoms[i].symbol, atoms[j].symbol
        deficit = (max(expected.get(si, [4])) - valence_cache[i]) + (max(expected.get(sj, [4])) - valence_cache[j])

        scores[(i, j)] = deficit
    # sort descending and pick top k
    top_edges = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [e for e, _ in top_edges]

def _score_assignment(G: nx.Graph, atoms: Atoms,
                     valence_electrons: Dict[str, int],
                     expected: Dict[str, List[int]],
                     total_charge: int,
                     rings=None) -> Tuple[float, List[int]]:
    """
    Score a bond order assignment.
    Returns (score, formal_charges) - lower is better.
    """
    EN = {'H': 2.2, 'C': 2.5, 'N': 3.0, 'O': 3.5, 'F': 4.0,
          'P': 2.2, 'S': 2.6, 'Cl': 3.2, 'Br': 3.0, 'I': 2.7}
    
    if _check_valence_violation(G, atoms):
        return 1e9, [0 for _ in G.nodes()]
    
    # --- ring cache ---
    if rings is None:
        rings = G.graph.get('_rings')
        if rings is None:
            rings = nx.cycle_basis(G)
            G.graph['_rings'] = rings

    # -- neighbors cache ---
    if '_neighbors' not in G.graph:
        G.graph['_neighbors'] = {n: list(G.neighbors(n)) for n in G.nodes()}
    neighbor_cache = G.graph['_neighbors']

    # --- has_H cache ---
    if '_has_H' not in G.graph:
        G.graph['_has_H'] = {n: any(atoms[nbr].symbol == 'H' for nbr in G.neighbors(n)) for n in G.nodes()}
    has_H = G.graph['_has_H']

    # --- valence cache ---
    valence_cache = {n: sum(G[n][nbr].get('bond_order', 1.0) for nbr in G.neighbors(n)) for n in G.nodes()}
    # --- formal charge cache (keyed by (sym, valence_sum)) ---
    formal_cache = {}
    def get_formal(sym, vsum):
        key = (sym, round(vsum, 2))
        if key not in formal_cache:
            V = valence_electrons.get(sym, 0)
            formal_cache[key] = _compute_formal_charge(sym, V, vsum)
        return formal_cache[key]

    penalties = {'valence': 0.0, 'en': 0.0, 'violation': 0.0, 'protonation': 0.0, 'conjugation': 0.0, 'fc': 0, 'n_charged': 0}
                 
    # --- RING CONJUGATION PENALTY ---
    penalties['conjugation'] = _ring_conjugation_penalty(G, atoms, rings, debug=_DEBUG)

    formal_charges = []

    for node in G.nodes():
        sym = atoms[node].symbol
        vsum = valence_cache[node]
        
        if sym in METALS:
            formal_charges.append(0)
            continue
        
        fc = get_formal(sym, vsum)
        formal_charges.append(fc)

        if fc != 0:
            penalties['fc'] += abs(fc)
            penalties['n_charged'] += 1
        
        nb = neighbor_cache[node]
        if has_H[node]:
            if fc == 0:
                for nbr in nb:
                    if atoms[nbr].symbol != 'H':
                        other_fc = get_formal(atoms[nbr].symbol, valence_cache[nbr])
                        if other_fc > 0:
                            penalties['protonation'] += 8.0 if sym in ('N', 'O') else 3.0
            elif fc > 0 and sym in ('N', 'O', 'S'):
                penalties['en'] -= 1.5  # expected protonation bonus

        # Valence error
        if sym in expected:
            allowed = expected[sym]
            min_error = min(abs(vsum - v) for v in allowed)
            penalties['valence'] += min_error ** 2
            
            # Check absolute limits
            limits = {'C': 4, 'N': 5, 'O': 3, 'S': 6, 'P': 6}
            if sym in limits and vsum > limits[sym] + 0.1:
                penalties['violation'] += 1000.0
        
        # Electronegativity penalty
        en = EN.get(sym, 2.5)
        if fc != 0:
            penalties['en'] += abs(fc) * ((3.5 - en) if fc < 0 else (en - 2.5)) * 0.5
            
    # Total score
    charge_error = abs(sum(formal_charges) - total_charge)
    score = (1000.0 * penalties['violation'] +
        12.0 * penalties['conjugation'] +
        8.0 * penalties['protonation'] +
        10.0 * penalties['fc'] +
        3.0 * penalties['n_charged'] +
        5.0 * charge_error +
        2.0 * penalties['en'] +
        5.0 * penalties['valence'])
    
    return score, formal_charges

def _full_valence_optimize(G: nx.Graph, atoms: Atoms,
                           expected: Dict[str, List[int]],
                           valence_electrons: Dict[str, int],
                           total_charge: int,
                           max_iter: int = 50,
                           edge_per_iter: int = 10) -> Dict[str, Any]:
    """
    Full bond order optimization with formal charge minimization and 
    detailed debugging.

    Returns a stats dict containing:
        - iterations
        - improvements
        - initial_score
        - final_score
        - final formal_charges
    """
    # --- Debug header ---
    if _DEBUG:
        _debug_print("=" * 60)
        _debug_print("FULL VALENCE OPTIMIZATION", 1)
        _debug_print("=" * 60)

    # --- Precompute / cache graph info ---
    rings = G.graph.get('_rings') or nx.cycle_basis(G)
    G.graph['_rings'] = rings

    neighbor_cache = G.graph.get('_neighbors') or {n: list(G.neighbors(n)) for n in G.nodes()}
    G.graph['_neighbors'] = neighbor_cache

    has_H = G.graph.get('_has_H') or {n: any(atoms[nbr].symbol == 'H' for nbr in G[n]) for n in G.nodes()}
    G.graph['_has_H'] = has_H

    valence_cache = {n: sum(G[n][nbr].get('bond_order', 1.0) for nbr in G.neighbors(n)) for n in G.nodes()}

    # --- Lock metal bonds ---
    for i, j, data in G.edges(data=True):
        if data.get('metal_coord', False):
            data['bond_order'] = 1.0

    # --- Initial scoring ---
    current_score, formal_charges = _score_assignment(G, atoms, valence_electrons, expected, total_charge, rings)
    initial_score = current_score

    stats = {
        'iterations': 0,
        'improvements': 0,
        'initial_score': initial_score,
        'final_score': initial_score,
        'final_formal_charges': formal_charges,
    }

    if _DEBUG:
        _debug_print(f"Initial score: {initial_score:.2f}", 1)
   
    stagnation = 0
    improved = True
    
    # --- Optimization loop ---
    for iteration in range(max_iter):
        stats['iterations'] = iteration + 1
        best_delta = 0.0
        best_edge = None

        if _DEBUG:
            _debug_print(f"\nIteration {iteration + 1}", 1)

        # --- Precompute top-k candidate edges ---
        top_edges = _edge_likelihood(G, atoms, expected, valence_cache, k=edge_per_iter)

        # --- Evaluate top-k edges using local delta scoring ---
        for i, j in top_edges:
            bo = G[i][j]['bond_order']
            if bo >= 3.0:
                continue

            # Temporarily increment bond
            G[i][j]['bond_order'] += 1
            valence_cache[i] += 1
            valence_cache[j] += 1

            # Compute full score
            new_score, _ = _score_assignment(G, atoms, valence_electrons, expected, total_charge, rings)
            delta = current_score - new_score

            # Rollback
            G[i][j]['bond_order'] -= 1
            valence_cache[i] -= 1
            valence_cache[j] -= 1

            if delta > best_delta:
                best_delta = delta
                best_edge = (i, j)

        # --- Apply best improvement ---
        if best_edge and best_delta > 1e-6:
            i, j = best_edge
            G[i][j]['bond_order'] += 1
            valence_cache[i] += 1
            valence_cache[j] += 1
            current_score, _ = _score_assignment(G, atoms, valence_electrons, expected, total_charge, rings)


            stats['improvements'] += 1
            stagnation = 0

            if _DEBUG:
                si, sj = atoms[i].symbol, atoms[j].symbol
                edge_label = f"{si}{i}-{sj}{j}"
                _debug_print(f"✓ {edge_label:<10}  Δscore = {best_delta:6.2f}  new_score = {current_score:8.2f}", 2)

        else:
            stagnation += 1
            if stagnation >= 2:
                break  # stop if no improvement

    # --- Final scoring ---
    final_formal_charges = _score_assignment(G, atoms, valence_electrons, expected, total_charge, rings)[1]
    stats['final_score'] = current_score
    stats['final_formal_charges'] = final_formal_charges

    if _DEBUG:
        _debug_print(f"\n{'=' * 60}")
        _debug_print(f"Optimized: {stats['improvements']} improvements", 1)
        _debug_print(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        _debug_print("=" * 60)

    return stats

# =============================================================================
# RDKIT INTEGRATION
# =============================================================================

def _rdkit_aromatic_refine(G: nx.Graph, atoms: Atoms) -> int:
    """Use RDKit aromatic perception to refine bond orders"""
    upgrades = 0
    
    try:
        rw = Chem.RWMol()
        for atom in atoms:
            rw.AddAtom(Chem.Atom(atom.symbol))
        
        for i, j in G.edges():
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        
        aromatic_pairs = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                         for b in mol.GetBonds() if b.GetIsAromatic()}
        
        for i, j, data in G.edges(data=True):
            if tuple(sorted((i, j))) in aromatic_pairs:
                old_order = data['bond_order']
                data['bond_order'] = 1.5
                if abs(old_order - 1.5) > 0.01:
                    upgrades += 1
    
    except Exception:
        pass
    
    return upgrades

def _detect_aromatic_rings(G: nx.Graph, atoms: Atoms) -> int:
    """
    Detect aromatic rings using Hückel rule (4n+2 π electrons).
    Sets all ring bonds to 1.5 for aromatic systems.
    
    Returns number of bonds set to aromatic.
    """
    if _DEBUG:
        _debug_print("=" * 60)
        _debug_print("AROMATIC RING DETECTION (Hückel 4n+2)")
        _debug_print("=" * 60)
    
    cycles = nx.cycle_basis(G)
    aromatic_count = 0
    aromatic_rings = 0
    
    for ring_idx, cycle in enumerate(cycles):
        # Only check 5- and 6-membered rings
        if len(cycle) not in (5, 6):
            continue
        
        if _DEBUG:
            ring_atoms = [f"{atoms[i].symbol}{i}" for i in cycle]
            _debug_print(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)
        
        # Check if all atoms can be aromatic - just means that other atoms will be kekule structures
        aromatic_atoms = {'C', 'N', 'O', 'S', 'P'}
        if not all(atoms[i].symbol in aromatic_atoms for i in cycle):
            if _DEBUG:
                non_aromatic = [atoms[i].symbol for i in cycle 
                               if atoms[i].symbol not in aromatic_atoms]
                _debug_print(f"✗ Contains non-aromatic atoms: {non_aromatic}", 2)
            continue
        
        # Count π electrons (simplified)
        pi_electrons = 0
        pi_breakdown = []
        
        for idx in cycle:
            sym = atoms[idx].symbol
            contribution = 0
            
            if sym == 'C':
                # sp2 carbon contributes 1 π electron
                contribution = 1
                pi_breakdown.append(f"{sym}{idx}:1")
            elif sym == 'N':
                # N with 2 neighbors (pyrrole-like) contributes 2
                # N with 3 neighbors (pyridine-like) contributes 1
                degree = sum(1 for _ in G.neighbors(idx))
                if degree == 2:
                    contribution = 2
                    pi_breakdown.append(f"{sym}{idx}:2(LP)")
                elif degree == 3:
                    contribution = 1
                    pi_breakdown.append(f"{sym}{idx}:1")
            elif sym in ('O', 'S'):
                # O/S with 2 neighbors contributes 2 (furan-like)
                degree = sum(1 for _ in G.neighbors(idx))
                if degree == 2:
                    contribution = 2
                    pi_breakdown.append(f"{sym}{idx}:2(LP)")
            
            pi_electrons += contribution
        
        if _DEBUG:
            _debug_print(f"π electrons: {pi_electrons} ({', '.join(pi_breakdown)})", 2)
        
        # Hückel rule: 4n+2 π electrons (n = 0, 1, 2, ...)
        is_aromatic = (pi_electrons >= 2 and 
                      pi_electrons in (2, 6, 10, 14, 18))
        
        if _DEBUG:
            if is_aromatic:
                n = (pi_electrons - 2) // 4
                _debug_print(f"✓ AROMATIC (4n+2 rule: n={n})", 2)
            else:
                _debug_print(f"✗ Not aromatic (4n+2 rule violated)", 2)
        
        if is_aromatic:
            # Set all ring edges to 1.5
            ring_edges = [(cycle[k], cycle[(k+1) % len(cycle)]) 
                         for k in range(len(cycle))]
            
            bonds_set = 0
            for i, j in ring_edges:
                edge_key = tuple(sorted((i, j)))
                if G.has_edge(i, j):
                    old_order = G.edges[i, j]['bond_order']
                    G.edges[i, j]['bond_order'] = 1.5
                    if abs(old_order - 1.5) > 0.01:
                        bonds_set += 1
                        aromatic_count += 1
            
            if bonds_set > 0:
                aromatic_rings += 1
    
    if _DEBUG:
        _debug_print(f"\n{'=' * 60}", 0)
        _debug_print(f"SUMMARY: {aromatic_rings} aromatic rings, "
                    f"{aromatic_count} bonds set to 1.5", 1)
        _debug_print(f"{'=' * 60}\n", 0)
    
    return aromatic_count


def _compute_gasteiger_charges(G: nx.Graph, atoms: Atoms) -> List[float]:
    """Compute Gasteiger charges using RDKit"""
    try:
        rw = Chem.RWMol()
        for atom in atoms:
            rw.AddAtom(Chem.Atom(atom.symbol))
        
        for i, j, data in G.edges(data=True):
            bo = data['bond_order']
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
        
        try:
            Chem.SanitizeMol(mol)
        except:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        
        AllChem.ComputeGasteigerCharges(mol)
        
        charges = []
        for atom in mol.GetAtoms():
            try:
                c = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(c):
                    c = 0.0
            except:
                c = 0.0
            charges.append(c)
        
        return charges
    
    except Exception:
        return [0.0] * len(atoms)

def _compute_formal_charges(G: nx.Graph, atoms: Atoms,
                           valence_electrons: Dict[str, int],
                           total_charge: int) -> List[int]:
    """Compute formal charges and balance to total"""
    formal = []
    
    for node in G.nodes():
        sym = atoms[node].symbol
        
        if sym in METALS:
            formal.append(0)
            continue
        
        V = valence_electrons.get(sym)
        if V is None:
            formal.append(0)
            continue
        
        bond_sum = _valence_sum(G, node)
        B = 2.0 * bond_sum
        target = 2 if sym == 'H' else 8
        L = max(0, target - B)
        fc = int(round(V - L - B / 2))
        formal.append(fc)
    
    # Balance residual charge
    residual = total_charge - sum(formal)
    if residual != 0:
        bonded = [(abs(formal[i]), i) for i in range(len(atoms))
                 if _valence_sum(G, i) > 0]
        bonded.sort(reverse=True)
        
        sign = 1 if residual > 0 else -1
        for _, idx in bonded:
            if residual == 0:
                break
            formal[idx] += sign
            residual -= sign
    
    return formal


# =============================================================================
# MAIN BUILD FUNCTIONS
# =============================================================================

def build_graph_cheminf(atoms: Atoms,
                       charge: int = 0,
                       multiplicity: Optional[int] = None,
                       quick: bool = False,
                       max_iter: int = 50,
                       edge_per_iter: int = 10) -> nx.Graph:
    """
    Build molecular graph using cheminformatics approach.
    
    Args:
        atoms: ASE Atoms object
        charge: Total molecular charge
        multiplicity: Spin multiplicity (auto-guessed if None)
        quick: Use fast heuristics instead of full optimization
    """
    vdw = _get_vdw()
    expected = _get_expected_valences()
    valence_electrons = _get_valence_electrons()
    
    if multiplicity is None:
        ne = int(np.sum(atoms.get_atomic_numbers())) - charge
        multiplicity = 1 if ne % 2 == 0 else 2
    
    if _DEBUG:
        mode = "QUICK" if quick else "FULL"
        _debug_print(f"\n{'=' * 60}")
        _debug_print(f"BUILDING GRAPH ({mode} MODE)")
        _debug_print(f"Atoms: {len(atoms)}, Charge: {charge}, "
                    f"Multiplicity: {multiplicity}")
        _debug_print(f"{'=' * 60}\n")
    
    # Build initial graph
    G = _build_initial_graph(atoms, vdw)
    
    if _DEBUG:
        _debug_print(f"Initial bonds: {G.number_of_edges()}", 1)
    
    # Prune distorted rings
    removed = _prune_distorted_rings(G, atoms)
    if _DEBUG and removed > 0:
        _debug_print(f"Pruned {removed} distorted ring bonds", 1)
    
    # Valence adjustment
    if quick:
        stats = _quick_valence_adjust(G, atoms, expected, vdw)
    else:
        stats = _full_valence_optimize(G, atoms, expected, 
                                       valence_electrons, charge, max_iter=max_iter, edge_per_iter=edge_per_iter)
    
    # Aromatic detection (Hückel rule)
    arom_count = _detect_aromatic_rings(G, atoms)
    if _DEBUG and arom_count > 0:
        _debug_print(f"Aromatic detection: {arom_count} bonds set to 1.5", 1)

    # RDKit aromatic refinement
    arom_count = _rdkit_aromatic_refine(G, atoms)
    if _DEBUG and arom_count > 0:
        _debug_print(f"RDKit aromatic: {arom_count} bonds upgraded", 1)
    
    # Compute charges
    gasteiger_raw = _compute_gasteiger_charges(G, atoms)
    raw_sum = sum(gasteiger_raw)
    delta = (charge - raw_sum) / len(atoms) if atoms else 0.0
    gasteiger_adj = [c + delta for c in gasteiger_raw]
    
    formal_charges = _compute_formal_charges(G, atoms, valence_electrons, charge)
    
    # Annotate graph
    for node in G.nodes():
        G.nodes[node]['charges'] = {
            'gasteiger_raw': gasteiger_raw[node],
            'gasteiger': gasteiger_adj[node]
        }
        G.nodes[node]['formal_charge'] = formal_charges[node]
        G.nodes[node]['valence'] = _valence_sum(G, node)
        
        # Aggregate charge (add H contributions)
        agg = gasteiger_adj[node]
        for nbr in G.neighbors(node):
            if atoms[nbr].symbol == 'H':
                agg += gasteiger_adj[nbr]
        G.nodes[node]['agg_charge'] = agg
    
    # Add bond types
    for i, j, data in G.edges(data=True):
        data['bond_type'] = (atoms[i].symbol, atoms[j].symbol)
    
    G.graph['total_charge'] = charge
    G.graph['multiplicity'] = multiplicity
    G.graph['valence_stats'] = stats
    G.graph['method'] = 'cheminf-quick' if quick else 'cheminf-full'
    
    if _DEBUG:
        _debug_print(f"\n{'=' * 60}")
        _debug_print("GRAPH CONSTRUCTION COMPLETE")
        _debug_print(f"{'=' * 60}\n")
    
    return G

def build_graph_xtb(atoms: Atoms,
                   charge: int = 0,
                   multiplicity: Optional[int] = None,
                   basename: str = 'xtb',
                   clean_up: bool = True) -> nx.Graph:
    """Build graph using xTB quantum chemistry calculations"""
    if multiplicity is None:
        ne = int(np.sum(atoms.get_atomic_numbers())) - charge
        multiplicity = 1 if ne % 2 == 0 else 2
    
    work = 'xtb_tmp_local'
    if os.system('which xtb > /dev/null 2>&1') != 0:
        raise RuntimeError("xTB not found - install or use 'cheminf' method")
    
    os.makedirs(work, exist_ok=True)
    
    import ase.io
    ase.io.write(os.path.join(work, f'{basename}.xyz'), atoms, format='xyz')
    os.system(f'cd {work} && xtb {basename}.xyz --chrg {charge} '
             f'--uhf {multiplicity-1} --gfn2 > {basename}.out')
    
    # Parse WBO
    bonds = []
    bond_orders = []
    wbo_file = os.path.join(work, f'{basename}_wbo')
    if not os.path.exists(wbo_file) and os.path.exists(os.path.join(work, 'wbo')):
        os.rename(os.path.join(work, 'wbo'), wbo_file)
    
    try:
        with open(wbo_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3 and float(parts[2]) > 0.5:
                    bonds.append((int(parts[0])-1, int(parts[1])-1))
                    bond_orders.append(float(parts[2]))
    except FileNotFoundError:
        pass
    
    # Parse charges
    charges = []
    charges_file = os.path.join(work, f'{basename}_charges')
    if not os.path.exists(charges_file) and os.path.exists(os.path.join(work, 'charges')):
        os.rename(os.path.join(work, 'charges'), charges_file)
    
    try:
        with open(charges_file) as f:
            for line in f:
                charges.append(float(line.split()[0]))
    except FileNotFoundError:
        charges = [0.0] * len(atoms)
    
    if clean_up:
        for f in os.listdir(work):
            os.remove(os.path.join(work, f))
        os.rmdir(work)
    
    # Build graph
    G = nx.Graph()
    pos = atoms.positions
    
    for i, atom in enumerate(atoms):
        G.add_node(i,
                  symbol=atom.symbol,
                  atomic_number=atom.number,
                  position=pos[i],
                  charges={'mulliken': charges[i] if i < len(charges) else 0.0})
    
    if bonds:
        for (i, j), bo in zip(bonds, bond_orders):
            d = _distance(pos[i], pos[j])
            si, sj = atoms[i].symbol, atoms[j].symbol
            G.add_edge(i, j,
                      bond_order=float(bo),
                      distance=d,
                      bond_type=(si, sj),
                      metal_coord=(si in METALS or sj in METALS))
    else:
        # Fallback to distance-based
        vdw = _get_vdw()
        G = _build_initial_graph(atoms, vdw)
    
    # Add derived properties
    for node in G.nodes():
        G.nodes[node]['valence'] = _valence_sum(G, node)
        agg = G.nodes[node]['charges'].get('mulliken', 0.0)
        for nbr in G.neighbors(node):
            if atoms[nbr].symbol == 'H':
                agg += G.nodes[nbr]['charges'].get('mulliken', 0.0)
        G.nodes[node]['agg_charge'] = agg
    
    G.graph['total_charge'] = charge
    G.graph['multiplicity'] = multiplicity
    G.graph['method'] = 'xtb'
    
    return G

def build_graph(atoms: Atoms,
               method: str = 'cheminf',
               charge: int = 0,
               multiplicity: Optional[int] = None,
               quick: bool = False,
               **kwargs) -> nx.Graph:
    """
    Unified graph builder.
    
    Args:
        atoms: ASE Atoms object
        method: 'cheminf' or 'xtb'
        charge: Total molecular charge
        multiplicity: Spin multiplicity (auto-guessed if None)
        quick: Fast mode for cheminf (skips comprehensive optimization)
        **kwargs: Additional arguments for specific methods
    
    Returns:
        NetworkX graph with molecular structure
    """
    if method == 'cheminf':
        return build_graph_cheminf(atoms, charge=charge, 
                                  multiplicity=multiplicity, quick=quick, max_iter=kwargs.get('max_iter', 50),edge_per_iter=kwargs.get('edge_per_iter', 5))
    elif method == 'xtb':
        return build_graph_xtb(atoms, charge=charge,
                              multiplicity=multiplicity,
                              basename=kwargs.get('basename', 'xtb'),
                              clean_up=kwargs.get('clean_up', True))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cheminf' or 'xtb'")