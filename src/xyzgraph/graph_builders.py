import os
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Try to suppress RDKit warnings

from .data_loader import DATA
from . import DEFAULT_PARAMS
from .utils import read_xyz_file

# =============================================================================
# GRAPH-BASED BOND CONSTRUCTION CLASS
# =============================================================================

class GraphBuilder:
    """
    Molecular graph construction with integrated state management.
    atoms: List of (symbol, (x, y, z)) tuples
    """

    def __init__(
            self,
            atoms: List[Tuple[str, Tuple[float, float, float]]],
            charge: int = DEFAULT_PARAMS['charge'],
            multiplicity: Optional[int] = DEFAULT_PARAMS['multiplicity'],
            method: str = DEFAULT_PARAMS['method'],
            quick: bool = DEFAULT_PARAMS['quick'],
            optimizer: str = DEFAULT_PARAMS['optimizer'],
            max_iter: int = DEFAULT_PARAMS['max_iter'],
            edge_per_iter: int = DEFAULT_PARAMS['edge_per_iter'],
            beam_width: int = DEFAULT_PARAMS['beam_width'],
            bond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS['bond'],
            unbond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS['unbond'],
            clean_up: bool = DEFAULT_PARAMS['clean_up'],
            debug: bool = DEFAULT_PARAMS['debug'],
            threshold: float = DEFAULT_PARAMS['threshold'],
            threshold_h_h: float = DEFAULT_PARAMS['threshold_h_h'],
            threshold_h_nonmetal: float = DEFAULT_PARAMS['threshold_h_nonmetal'],
            threshold_h_metal: float = DEFAULT_PARAMS['threshold_h_metal'],
            threshold_metal_ligand: float = DEFAULT_PARAMS['threshold_metal_ligand'],
            threshold_nonmetal_nonmetal: float = DEFAULT_PARAMS['threshold_nonmetal_nonmetal']
            ):
        self.atoms = atoms  # List of (symbol, (x,y,z))
        self.charge = charge
        self.method = method
        self.optimizer = optimizer.lower()
        self.quick = quick
        self.max_iter = max_iter
        self.edge_per_iter = edge_per_iter
        self.beam_width = beam_width
        self.bond = bond
        self.unbond = unbond
        self.clean_up = clean_up
        self.debug = debug

        if self.optimizer not in ('greedy', 'beam'):
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Choose from: 'greedy', 'beam'")

        # Auto-detect multiplicity
        if multiplicity is None:
            total_electrons = sum(DATA.s2n[symbol] for symbol, _ in atoms) - charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2
        else:
            self.multiplicity = multiplicity

        self.threshold = threshold
        self.threshold_h_h = threshold_h_h
        self.threshold_h_nonmetal = threshold_h_nonmetal
        self.threshold_h_metal = threshold_h_metal
        self.threshold_metal_ligand = threshold_metal_ligand
        self.threshold_nonmetal_nonmetal = threshold_nonmetal_nonmetal

        # Reference to global data
        self.data = DATA

        # Pre-compute atom properties from tuples
        self.symbols = [symbol for symbol, _ in self.atoms]
        self.atomic_numbers = [DATA.s2n[symbol] for symbol, _ in self.atoms]
        self.positions = [(x, y, z) for _, (x, y, z) in self.atoms]

        # State
        self.graph: Optional[nx.Graph] = None
        self.log_buffer = []

        # Optimization state (for caching)
        self.valence_cache = {}
        self.edge_scores_cache = None
        self._edge_score_map = None
    
    def log(self, msg: str, level: int = 0):
        """Log message"""
        if self.debug:
            indent = "  " * level
            line = f"{indent}{msg}"
            print(line)
            self.log_buffer.append(line)
    
    def get_log(self) -> str:
        """Get full build log as string"""
        return "\n".join(self.log_buffer)

    # =========================================================================
    # Helper methods 
    # =========================================================================

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _should_bond_metal(self, sym_i: str, sym_j: str) -> bool:
        """
        Chemical filter for metal bonds (called AFTER distance check).
        
        Returns False only for implausible metal pairings:
        - Metal-metal (unless bridging ligand expected) # NOTE: this may be a limitation
        
        Accepts:
        - Metal to donor atoms (O, N, C, P, S) # NOTE: this may be a limitation
        - Metal to halides/oxo (ionic)
        - Metal to H (hydrides)
        """
        if sym_i not in self.data.metals and sym_j not in self.data.metals:
            return True
        
        # Identify metal and other
        metal = sym_i if sym_i in self.data.metals else sym_j
        other = sym_j if metal == sym_i else sym_i
        
        # Accept common ligands
        if other in ('O', 'N', 'C', 'P', 'S', 'H'):
            return True
        
        # Accept halides
        if other in ('F', 'Cl', 'Br', 'I'):
            return True
        
        return False

    @staticmethod
    def _valence_sum(G: nx.Graph, node: int) -> float:
        """Sum bond orders around a node"""
        return sum(G.edges[node, nbr].get('bond_order', 1.0) for nbr in G.neighbors(node))
    
    def _compute_formal_charge_value(self, symbol: str, valence_electrons: int, 
                                     bond_order_sum: float) -> int:
        """Compute formal charge for an atom"""
        if symbol == 'H':
            return valence_electrons - int(bond_order_sum)
        
        B = 2 * bond_order_sum
        target = 8
        L = max(0, target - B)
        return int(round(valence_electrons - L - B / 2))

    def _compute_formal_charges(self, G: nx.Graph) -> List[int]:
        """Compute formal charges for all atoms and balance to total charge"""
        formal = []

        for node in G.nodes():
            sym = G.nodes[node]['symbol']

            if sym in DATA.metals:
                formal.append(0)
                continue

            V = DATA.electrons.get(sym)
            if V is None:
                formal.append(0)
                continue

            bond_sum = self._valence_sum(G, node)
            fc = self._compute_formal_charge_value(sym, V, bond_sum)
            formal.append(fc)

        # # Balance residual charge
        residual = self.charge - sum(formal)
        if residual != 0:
            bonded = [(abs(formal[i]), i) for i in range(len(self.atoms)) if self._valence_sum(G, i) > 0]
            bonded.sort(reverse=True)

            sign = 1 if residual > 0 else -1
            for _, idx in bonded:
                if residual == 0:
                    break
                formal[idx] += sign
                residual -= sign

        return formal
    
    def _check_valence_violation(self, G: nx.Graph,
                                 limits: Optional[Dict[str, float]] = None,
                                 tol: float = 0.3) -> bool:
        """Check for pentavalent carbon etc."""
        if limits is None:
            limits = {'C': 4}

        for i in G.nodes():
            sym = G.nodes[i]['symbol']
            if sym in limits:
                val = sum(G[i][j].get('bond_order', 1.0) for j in G.neighbors(i))
                if val > limits[sym] + tol:
                    return True
        return False

    # =========================================================================
    # Main build method
    # =========================================================================

    def build(self) -> nx.Graph:
        """Build molecular graph using configured method"""
        mode = "QUICK" if self.quick else "FULL"
        self.log(f"\n{'=' * 60}")
        self.log(f"BUILDING GRAPH ({self.method.upper()}, {mode} MODE)")
        self.log(f"Atoms: {len(self.atoms)}, Charge: {self.charge}, "
                f"Multiplicity: {self.multiplicity}")
        self.log(f"{'=' * 60}\n")
        
        if self.method == 'cheminf':
            self.graph = self._build_cheminf()
        elif self.method == 'xtb':
            self.graph = self._build_xtb()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store build log in graph
        self.graph.graph['build_log'] = self.get_log()
        
        self.log(f"\n{'=' * 60}")
        self.log("GRAPH CONSTRUCTION COMPLETE")
        self.log(f"{'=' * 60}\n")
        
        return self.graph

    # =========================================================================
    # Cheminformatics path
    # =========================================================================

    def _build_initial_graph(self) -> nx.Graph:
        """Build initial graph with distance-based bonds"""
        G = nx.Graph()

        pos = np.array(self.positions)

        # Add nodes
        for i, atomic_num, symbol in zip(range(len(self.atoms)), self.atomic_numbers, self.symbols):
            G.add_node(i,
                      symbol=symbol,
                      atomic_number=atomic_num,
                      position=tuple(pos[i]))

        self.log(f"Added {len(self.atoms)} atoms", 1)

        # Add edges based on distance
        edge_count = 0
        for i in range(len(self.atoms)):
            si = self.symbols[i]
            is_metal_i = si in self.data.metals

            for j in range(i + 1, len(self.atoms)):
                sj = self.symbols[j]
                is_metal_j = sj in self.data.metals
                has_metal = is_metal_i or is_metal_j
                has_h = 'H' in (si, sj)

                d = self._distance(pos[i], pos[j])
                r_sum = DATA.vdw.get(si, 2.0) + DATA.vdw.get(sj, 2.0)

                # Choose threshold (scaled by threshold)
                # Check H-H bonds first
                if si == 'H' and sj == 'H':
                    threshold = self.threshold_h_h * r_sum * self.threshold
                # Then H-metal bonds
                elif has_h and has_metal:
                    threshold = self.threshold_h_metal * r_sum * self.threshold
                # Then H-nonmetal bonds
                elif has_h and not has_metal:
                    threshold = self.threshold_h_nonmetal * r_sum * self.threshold
                # Then metal-ligand bonds
                elif has_metal:
                    threshold = self.threshold_metal_ligand * r_sum * self.threshold
                # Finally nonmetal-nonmetal bonds
                else:
                    threshold = self.threshold_nonmetal_nonmetal * r_sum * self.threshold

                if d < threshold:
                    if has_metal and not self._should_bond_metal(si, sj):
                        continue

                    G.add_edge(i, j,
                              bond_order=1.0,
                              distance=d,
                              metal_coord=has_metal)
                    edge_count += 1

        if self.bond:
            for i, j in self.bond:
                if not G.has_edge(i, j):
                    d = self._distance(pos[i], pos[j])
                    G.add_edge(i, j, bond_order=1, distance=d)
                    self.log(f"Added bond {i}-{j} (d={d:.3f} Å)", 2)

        if self.unbond:
            for i, j in self.unbond:
                if G.has_edge(i, j):
                    G.remove_edge(i, j)
                    self.log(f"Removed bond {i}-{j}")

        self.log(f"Initial bonds: {edge_count}", 1)

        # Cache graph properties
        rings = nx.cycle_basis(G)
        G.graph['_rings'] = rings
        G.graph['_neighbors'] = {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph['_has_H'] = {n: any(G.nodes[nbr]['symbol'] == 'H' for nbr in G.neighbors(n)) for n in G.nodes()}

        self.log(f"Found {len(rings)} rings", 1)

        return G

    def _prune_distorted_rings(self, G: nx.Graph) -> int:
        """Remove geometrically distorted small rings"""

        removed = 0
        max_passes = 4
        ring_prune_ratios = {3: 1.18, 4:1.22}
        self.log(f"Pruning distorted rings (sizes: {list(ring_prune_ratios.keys())})", 1)

        for _ in range(max_passes):
            cycles = nx.cycle_basis(G) # recompute since graph may change
            if not cycles:
                break

            pruned_this_pass = False

            for cycle in cycles:
                if len(cycle) not in ring_prune_ratios.keys():
                    continue

                # Skip metal-containing cycles
                if any(G.nodes[i]['symbol'] in DATA.metals for i in cycle):
                    continue

                # Get cycle edges and distances
                cycle_edges = [(cycle[k], cycle[(k+1) % len(cycle)]) for k in range(len(cycle))]

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
                threshold = ring_prune_ratios[len(cycle)]

                if dmax / dmin > threshold:
                    # Remove longest edge
                    worst_idx = distances.index(dmax)
                    i, j = cycle_edges[worst_idx]
                    if G.has_edge(i, j):
                        G.remove_edge(i, j)
                        removed += 1
                        pruned_this_pass = True
                        self.log(f"  Removed edge {i}-{j} in {len(cycle)}-ring", 2)
                        break # One edge per pass

            if not pruned_this_pass:
                break

        # Update cached rings after pruning
        G.graph['_rings'] = nx.cycle_basis(G)

        return removed

    def _init_kekule_for_6rings(self, G: nx.Graph) -> int:
        """
        Initialize alternating single/double bonds for 6-membered aromatic rings.
            - Find 6-membered rings with aromatic atoms (C, N, O, S, B)
            - Check planarity
            - Apply simple alternating 1-2-1-2 pattern as starting point
            - If edges are already set (from fused rings), validate consistency
            - Optimizer will refine based on formal charges, electronegativity, etc.

        Parameters:
        - G: NetworkX graph to modify in-place
        """
        cycles = G.graph.get('_rings')
        if cycles is None:
            cycles = nx.cycle_basis(G)
            G.graph['_rings'] = cycles

        initialized_rings = 0

        # Atoms that can participate in aromatic conjugation
        aromatic_atoms = {'C', 'N', 'O', 'S', 'B'}

        for cycle in cycles:
            # Must be 6-membered
            if len(cycle) != 6:
                continue

            # Allow some heteroatoms for extended conjugation
            if not all(G.nodes[i]['symbol'] in aromatic_atoms for i in cycle):
                continue

            # Check planarity (aromatic rings are planar)
            if not self._check_planarity(cycle, G, threshold=0.15):
                continue

            # Get ring edges and their current bond orders
            ring_edges = [(cycle[k], cycle[(k+1) % len(cycle)]) for k in range(len(cycle))]
            current_orders = []
            for i, j in ring_edges:
                if G.has_edge(i, j):
                    current_orders.append(G.edges[i, j]['bond_order'])
                else:
                    current_orders.append(1.0)

            # Check for already-set edges (from previously processed fused rings)
            set_edges = [(idx, bo) for idx, bo in enumerate(current_orders) if abs(bo - 1.0) > 0.01] # ie. not currently a single bond

            if len(set_edges) == 0:
                # No constraints - use default pattern starting with double
                for idx, (i, j) in enumerate(ring_edges):
                    if G.has_edge(i, j):
                        bond_order = 2.0 if idx % 2 == 0 else 1.0
                        G.edges[i, j]['bond_order'] = bond_order
                initialized_rings += 1

            elif len(set_edges) == 1:
                # One anchor - alternate from it
                anchor_idx, anchor_bo = set_edges[0]

                for idx, (i, j) in enumerate(ring_edges):
                    if G.has_edge(i, j):
                        if idx == anchor_idx:
                            continue  # Keep anchor as is

                        offset = (idx - anchor_idx) % 6
                        if abs(anchor_bo - 2.0) < 0.1:
                            bond_order = 2.0 if offset % 2 == 0 else 1.0
                        else:
                            bond_order = 1.0 if offset % 2 == 0 else 2.0

                        G.edges[i, j]['bond_order'] = bond_order
                initialized_rings += 1

            else:
                # Multiple constraints - verify consistency before applying
                # Use first anchor to generate expected pattern
                anchor_idx, anchor_bo = set_edges[0]

                # Generate expected pattern based on first anchor
                expected = []
                for idx in range(6):
                    offset = (idx - anchor_idx) % 6
                    if abs(anchor_bo - 2.0) < 0.1:
                        expected.append(2.0 if offset % 2 == 0 else 1.0)
                    else:
                        expected.append(1.0 if offset % 2 == 0 else 2.0)

                # Check if all set edges match expected pattern
                consistent = True
                for idx, bo in set_edges:
                    if abs(expected[idx] - bo) > 0.1:
                        consistent = False
                        break

                if consistent:
                    # Apply pattern to unset edges only
                    for idx, (i, j) in enumerate(ring_edges):
                        if G.has_edge(i, j):
                            # Only set if currently at default (1.0)
                            if abs(current_orders[idx] - 1.0) < 0.01:
                                G.edges[i, j]['bond_order'] = expected[idx]
                    initialized_rings += 1
                else:
                    # Conflict detected - skip this ring, let optimizer handle it
                    pass

        if initialized_rings > 0:
            self.log(f"Initialized {initialized_rings} 6-membered carbon rings with Kekulé pattern", 1)

        return initialized_rings

# =============================================================================
# QUICK MODE: Simple heuristic valence adjustment
# =============================================================================

    def _quick_valence_adjust(self, G: nx.Graph) -> Dict[str, int]:
        """
        Fast heuristic bond order adjustment.
        No formal charge optimization - just satisfy valences.
        """
        stats = {'iterations': 0, 'promotions': 0}

        # Lock metal bonds
        for i, j in G.edges():
            if G.edges[i, j].get('metal_coord', False):
                G.edges[i, j]['bond_order'] = 1.0

        for iteration in range(3):
            stats['iterations'] = iteration + 1
            changed = False

            # Calculate deficits
            deficits = {}
            for node in G.nodes():
                sym = G.nodes[node]['symbol']
                if sym in DATA.metals:
                    deficits[node] = 0.0
                    continue

                current = self._valence_sum(G, node)
                allowed = DATA.valences.get(sym, [])
                if not allowed:
                    deficits[node] = 0.0
                    continue

                target = min(allowed, key=lambda v: abs(v - current))
                deficits[node] = target - current

            # Try to promote bonds
            for i, j, data in G.edges(data=True):
                if data.get('metal_coord', False):
                    continue

                si, sj = G.nodes[i]['symbol'], G.nodes[j]['symbol']
                if 'H' in (si, sj):
                    continue

                bo = data['bond_order']
                if bo >= 3.0:
                    continue

                di, dj = deficits[i], deficits[j]

                # Check geometry
                dist_ratio = data['distance'] / (DATA.vdw.get(si, 2.0) + DATA.vdw.get(sj, 2.0))
                if dist_ratio > 0.60:
                    continue

                # Promote if both atoms need more valence
                if di > 0.3 and dj > 0.3:
                    increment = min(di, dj, 3.0 - bo)
                    if increment >= 0.5:
                        data['bond_order'] = bo + increment
                        stats['promotions'] += 1
                        changed = True
            self.log(f"Iteration {iteration+1}: Promotions={stats['promotions']}", 1)

            if not changed:
                break

        return stats

    def _edge_score(self, G: nx.Graph, i: int, j: int) -> float:
        """Check scoring of edge."""
        if not self._eligible_edge(G, i, j):
            return float('-inf')
        si, sj = G.nodes[i]['symbol'], G.nodes[j]['symbol']
        vmax_i = max(DATA.valences.get(si, [4]))
        vmax_j = max(DATA.valences.get(sj, [4]))
        di = vmax_i - self.valence_cache[i]
        dj = vmax_j - self.valence_cache[j]
        return di + dj


    def _eligible_edge(self, G: nx.Graph, i: int, j: int) -> bool:
        data = G[i][j]
        if data.get('metal_coord', False):
            return False
        if data.get('locked', False):
            return False
        if data.get('bond_order', 1.0) >= 3.0:
            return False
        return True

    def _ekey(self, i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _edge_likelihood(self, G: nx.Graph, *, init: bool = False, touch_nodes: Optional[set] = None):
        """
        Cache-aware candidate selection:
        - init=True: build score map for all edges once
        - touch_nodes={u,v}: update edges belonging to these nodes
        - return current top-k edges as a list [(i,j), ...]
        """
        # Build / refresh full score map
        if init or self._edge_score_map is None:
            self._edge_score_map = {}
            for i, j in G.edges():
                e = self._ekey(i, j)
                self._edge_score_map[e] = self._edge_score(G, *e)

        # Incremental update: only recompute scores for edges touching changed nodes
        if touch_nodes:
            for n in touch_nodes:
                for nbr in G.neighbors(n):
                    e = self._ekey(n, nbr)
                    # only update existing edges
                    if G.has_edge(*e):
                        self._edge_score_map[e] = self._edge_score(G, *e)
        # Return top-k edges
        items = [(s, e) for e, s in self._edge_score_map.items() if s != float('-inf')]
        items.sort(key=lambda t: (-t[0], t[1][0], t[1][1])) # sort by score desc, then by edge
        top = [e for _, e in items[: self.edge_per_iter]]
        self.edge_scores_cache = top
        return top

    # =============================================================================
    # FULL MODE: Formal charge optimization
    # =============================================================================

    def _ring_conjugation_penalty(self, G: nx.Graph, rings) -> float:
        """
        Assess conjugation and exocyclic double penalties in aromatic rings (5-6 members).
        Returns a numeric penalty (larger = worse).
        """
        conjugation_penalty = 0.0
        for ring in rings:
            if len(ring) not in (5, 6):
                continue

            conjugatable = {'C', 'N', 'O', 'S', 'P'}
            if not all(G.nodes[i]['symbol'] in conjugatable for i in ring):
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
                ring_sym = G.nodes[ring_atom]['symbol']
                for nbr, data in G[ring_atom].items():
                    if nbr not in ring_set:
                        bo = data.get('bond_order', 1.0)
                        if bo >= 1.8:
                            nbr_sym = G.nodes[nbr]['symbol']
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

    def _full_valence_optimize(self, G: nx.Graph) -> Dict[str, Any]:
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
        self.log(f"\n{'=' * 60}", 0)
        self.log("FULL VALENCE OPTIMIZATION", 1)
        self.log("=" * 60, 0)

        # --- Precompute / cache graph info ---
        rings = G.graph.get('_rings') or nx.cycle_basis(G)
        G.graph['_rings'] = rings

        neighbor_cache = G.graph.get('_neighbors') or {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph['_neighbors'] = neighbor_cache

        has_H = G.graph.get('_has_H') or {n: any(G.nodes[nbr]['symbol'] == 'H' for nbr in G[n]) for n in G.nodes()}
        G.graph['_has_H'] = has_H

        valence_cache = {n: sum(G[n][nbr].get('bond_order', 1.0) for nbr in G.neighbors(n)) for n in G.nodes()}
        self.valence_cache = valence_cache

        # --- Lock metal bonds ---
        metal_count = 0
        for i, j, data in G.edges(data=True):
            if data.get('metal_coord', False):
                data['bond_order'] = 1.0
                metal_count += 1
        if metal_count > 0:
            self.log(f"Locked {metal_count} metal bonds", 1)

        # --- Initial scoring ---
        current_score, formal_charges = self._score_assignment(G, rings)
        initial_score = current_score

        stats = {
            'iterations': 0,
            'improvements': 0,
            'initial_score': initial_score,
            'final_score': initial_score,
            'final_formal_charges': formal_charges,
        }

        self.log(f"Initial score: {initial_score:.2f}", 1)
    
        stagnation = 0
        self.edge_scores_cache = None
        last_promoted_edge = None   
        self._edge_score_map = None 
        
        # --- Optimization loop ---
        for iteration in range(self.max_iter):
            stats['iterations'] = iteration + 1
            best_delta = 0.0
            best_edge = None

            self.log(f"\nIteration {iteration + 1}:", 1)

            # --- Precompute top-k candidate edges (with cache) ---
            if self.edge_scores_cache is None:
                self.edge_scores_cache = self._edge_likelihood(G, init=True)
            elif last_promoted_edge is not None:
                self.log(f"Recalculating candidates (promoted {last_promoted_edge})", 2)
                # update only edges incident to last promoted atoms
                i, j = last_promoted_edge
                self.edge_scores_cache = self._edge_likelihood(G, touch_nodes={i, j})

            # --- Evaluate top-k edges using local delta scoring ---
            # Test both promotion (+1) and demotion (-1) for Kekulé flexibility
            for i, j in self.edge_scores_cache:
                bo = G[i][j]['bond_order']
                
                # Test both directions
                for change in [+1, -1]:
                    new_bo = bo + change
                    
                    # Skip invalid bond orders
                    if new_bo < 1.0 or new_bo > 3.0:
                        continue
                    
                    # Temporarily apply change
                    G[i][j]['bond_order'] = new_bo
                    valence_cache[i] += change
                    valence_cache[j] += change

                    # Compute full score
                    new_score, _ = self._score_assignment(G, rings)
                    delta = current_score - new_score

                    # Rollback
                    G[i][j]['bond_order'] = bo
                    valence_cache[i] -= change
                    valence_cache[j] -= change

                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j, change)

            # --- Apply best improvement ---
            if best_edge and best_delta > 1e-6:
                i, j, change = best_edge
                G[i][j]['bond_order'] += change
                valence_cache[i] += change
                valence_cache[j] += change
                current_score, _ = self._score_assignment(G, rings)

                stats['improvements'] += 1
                stagnation = 0
                last_promoted_edge = (i, j)
                self._edge_likelihood(G, touch_nodes={i, j})  # update cache

                si, sj = G.nodes[i]['symbol'], G.nodes[j]['symbol']
                edge_label = f"{si}{i}-{sj}{j}"
                action = "promoted" if change > 0 else "demoted"
                self.log(f"✓ {edge_label:<10}  {action}  Δscore = {best_delta:6.2f}  new_score = {current_score:8.2f}", 2)

            else:
                stagnation += 1
                last_promoted_edge = None
                self.edge_scores_cache = None  # force full recompute next time

                if stagnation >= 3:
                    break  # stop if no improvement

        # --- Final scoring ---
        final_formal_charges = self._score_assignment(G, rings)[1]
        stats['final_score'] = current_score
        stats['final_formal_charges'] = final_formal_charges

        self.log("-" * 60, 0)
        self.log(f"Optimized: {stats['improvements']} improvements", 1)
        self.log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self.log("-" * 60, 0)

        return stats

    def _update_valence_cache(self, G: nx.Graph, nodes: Optional[set] = None) -> None:
        """
        Update valence cache for specific nodes or all nodes.
        """
        if nodes is None:
            # Full rebuild
            self.valence_cache = {
                n: sum(G[n][nbr].get('bond_order', 1.0) for nbr in G.neighbors(n))
                for n in G.nodes()
            }
        else:
            # Incremental update
            for n in nodes:
                self.valence_cache[n] = sum(
                    G[n][nbr].get('bond_order', 1.0) for nbr in G.neighbors(n)
                )

    def _restore_graph_caches(self, G: nx.Graph) -> None:
        """
        Rebuild cached graph properties after modifications.
        Called after applying bond order changes.
        """
        # Update cached rings
        G.graph['_rings'] = nx.cycle_basis(G)
        
        # Update neighbor cache
        G.graph['_neighbors'] = {n: list(G.neighbors(n)) for n in G.nodes()}
        
        # Update H-neighbor cache
        G.graph['_has_H'] = {
            n: any(G.nodes[nbr]['symbol'] == 'H' for nbr in G.neighbors(n)) 
            for n in G.nodes()
        }
    
    def _copy_graph_state(self, G: nx.Graph) -> nx.Graph:
        """
        Create INDEPENDENT copy of graph for beam exploration.
        """
        G_new = nx.Graph()
        
        # Copy nodes with INDEPENDENT attribute dicts
        for node, data in G.nodes(data=True):
            G_new.add_node(
                node,
                symbol=data['symbol'],
                atomic_number=data['atomic_number'],
                position=data['position']  
            )
        
        # Copy edges with INDEPENDENT attribute dicts
        for i, j, data in G.edges(data=True):
            G_new.add_edge(
                i, j,
                bond_order=float(data['bond_order']),
                distance=float(data['distance']),
                metal_coord=bool(data.get('metal_coord', False))
            )
                
        return G_new

    def _score_assignment(self, G: nx.Graph, rings: List[List[int]] = None) -> Tuple[float, List[int]]:
        """
        Scoring that uses pre-computed valence cache.
        """
        EN = {'H': 2.2, 'C': 2.5, 'N': 3.0, 'O': 3.5, 'F': 4.0,
            'P': 2.2, 'S': 2.6, 'Cl': 3.2, 'Br': 3.0, 'I': 2.7}
        
        if self._check_valence_violation(G):
            return 1e9, [0 for _ in G.nodes()]
        
        # Ring cache
        if rings is None:
            rings = G.graph.get('_rings', nx.cycle_basis(G))
        
        # Neighbor cache
        neighbor_cache = G.graph.get('_neighbors', {n: list(G.neighbors(n)) for n in G.nodes()})
        
        # H-neighbor cache
        has_H = G.graph.get('_has_H', {n: any(G.nodes[nbr]['symbol'] == 'H' for nbr in G.neighbors(n)) for n in G.nodes()})
        
        penalties = {'valence': 0.0, 'en': 0.0, 'violation': 0.0, 
                    'protonation': 0.0, 'conjugation': 0.0, 'fc': 0, 'n_charged': 0}
        
        # Conjugation penalty
        penalties['conjugation'] = self._ring_conjugation_penalty(G, rings)
        
        # Formal charge cache
        formal_cache = {}
        def get_formal(sym, vsum):
            key = (sym, round(vsum, 2))
            if key not in formal_cache:
                V = DATA.electrons.get(sym, 0)
                formal_cache[key] = self._compute_formal_charge_value(sym, V, vsum)
            return formal_cache[key]
        
        formal_charges = []
        
        for node in G.nodes():
            sym = G.nodes[node]['symbol']

            vsum = self.valence_cache[node]
            
            if sym in DATA.metals:
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
                        if G.nodes[nbr]['symbol'] != 'H':
                            other_fc = get_formal(G.nodes[nbr]['symbol'], self.valence_cache[nbr])
                            if other_fc > 0:
                                penalties['protonation'] += 8.0 if sym in ('N', 'O') else 3.0
                elif fc > 0 and sym in ('N', 'O', 'S'):
                    penalties['en'] -= 1.5
            
            # Valence error
            if sym in DATA.valences:
                allowed = DATA.valences[sym]
                min_error = min(abs(vsum - v) for v in allowed)
                penalties['valence'] += min_error ** 2
                
                limits = {'C': 4, 'N': 5, 'O': 3, 'S': 6, 'P': 6}
                if sym in limits and vsum > limits[sym] + 0.1:
                    penalties['violation'] += 1000.0
            
            # Electronegativity penalty
            en = EN.get(sym, 2.5)
            if fc != 0:
                penalties['en'] += abs(fc) * ((3.5 - en) if fc < 0 else (en - 2.5)) * 0.5
        
        # Total score
        charge_error = abs(sum(formal_charges) - self.charge)
        score = (1000.0 * penalties['violation'] +
                12.0 * penalties['conjugation'] +
                8.0 * penalties['protonation'] +
                10.0 * penalties['fc'] +
                3.0 * penalties['n_charged'] +
                5.0 * charge_error +
                2.0 * penalties['en'] +
                5.0 * penalties['valence'])
        
        return score, formal_charges


    def _beam_search_optimize(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Memory-efficient beam search with incremental valence cache updates.
        
        Strategy:
        - Maintain valence cache per hypothesis (small dict)
        - When promoting edge (i,j), update valence for nodes i and j
        - Score calculation uses cached valences
        """
        self.log("=" * 60, 0)
        self.log(f"\nBEAM SEARCH OPTIMIZATION (width={self.beam_width})", 1)
        self.log("=" * 60, 0)
        
        # Precompute graph info (shared across beam)
        rings = nx.cycle_basis(G)
        G.graph['_rings'] = rings
        G.graph['_neighbors'] = {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph['_has_H'] = {
            n: any(G.nodes[nbr]['symbol'] == 'H' for nbr in G.neighbors(n)) 
            for n in G.nodes()
        }
        
        # Lock metal bonds
        metal_count = 0
        for i, j, data in G.edges(data=True):
            if data.get('metal_coord', False):
                data['bond_order'] = 1.0
                metal_count += 1
        if metal_count > 0:
            self.log(f"Locked {metal_count} metal bonds", 1)
        
        # Build initial valence cache (shared starting point)
        base_valence_cache = {
            n: sum(G[n][nbr].get('bond_order', 1.0) for nbr in G.neighbors(n))
            for n in G.nodes()
        }
        
        # Initial scoring
        self.valence_cache = base_valence_cache.copy()
        current_score, formal_charges = self._score_assignment(G, rings)
        initial_score = current_score
        
        self.log(f"Initial score: {initial_score:.2f}", 1)

        beam = [(current_score, G, base_valence_cache.copy(), [])]
        
        stats = {
            'iterations': 0,
            'improvements': 0,
            'initial_score': initial_score,
            'final_score': initial_score,
            'final_formal_charges': formal_charges,
            'beam_explored': 0
        }
        
        best_ever_score = current_score
        best_ever_graph = self._copy_graph_state(G)
        best_ever_cache = base_valence_cache.copy()
        
        for iteration in range(self.max_iter):
            stats['iterations'] = iteration + 1
            self.log(f"\nIteration {iteration + 1}:", 1)
            
            candidates = [] 
            
            # Expand each hypothesis in beam
            for beam_idx, (parent_score, parent_graph, parent_cache, parent_history) in enumerate(beam):
                
                self.valence_cache = parent_cache
                
                # Get top candidate edges
                self._edge_score_map = None
                top_edges = self._edge_likelihood(parent_graph, init=True)
                                
                changes_tried = 0
                for i, j in top_edges:
                    if not self._eligible_edge(parent_graph, i, j):
                        continue
                    
                    # Test both promotion (+1) and demotion (-1)
                    for change in [+1, -1]:
                        old_bo = parent_graph[i][j]['bond_order']
                        new_bo = old_bo + change
                        
                        # Skip invalid bond orders
                        if new_bo < 1.0 or new_bo > 3.0:
                            continue
                        
                        changes_tried += 1
                        
                        G_new = self._copy_graph_state(parent_graph)
                        
                        # Apply change
                        G_new[i][j]['bond_order'] = new_bo
                        
                        # Update the two affected nodes
                        new_cache = parent_cache.copy()
                        new_cache[i] = parent_cache[i] + change
                        new_cache[j] = parent_cache[j] + change
                        
                        # Use new cache for scoring
                        self.valence_cache = new_cache
                        new_score, _ = self._score_assignment(G_new, rings)
                        
                        stats['beam_explored'] += 1
                        
                        # Keep if improvement
                        delta = parent_score - new_score
                        if delta > 0:
                            new_history = parent_history + [(i, j, change)]
                            candidates.append((new_score, G_new, new_cache, (i, j, change), new_history))
                            
            if not candidates:
                self.log("  No improvements found in any beam, stopping", 2)
                break
            
            # Sort and keep top beam_width
            candidates.sort(key=lambda x: x[0])
            
            self.log(f"  Generated {len(candidates)} candidates, keeping top {min(self.beam_width, len(candidates))}", 2)

            beam = [(score, graph, cache, history) 
                    for score, graph, cache, edge, history in candidates[:self.beam_width]]
            
            # Track best ever
            best_in_beam = beam[0]
            if best_in_beam[0] < best_ever_score:
                improvement = best_ever_score - best_in_beam[0]
                best_ever_score = best_in_beam[0]
                best_ever_graph = self._copy_graph_state(best_in_beam[1])
                best_ever_cache = best_in_beam[2].copy()
                stats['improvements'] += 1
                
                # Log improvement
                last_edge = best_in_beam[3][-1]
                si = G.nodes[last_edge[0]]['symbol']
                sj = G.nodes[last_edge[1]]['symbol']
                edge_label = f"{si}{last_edge[0]}-{sj}{last_edge[1]}"
                self.log(f"  ✓ New best: {edge_label:<10}  Δtotal = {improvement:6.2f}  score = {best_ever_score:8.2f}", 2)
        
        # Apply best solution
        self.log("\nApplying best solution to graph...", 1)
        for i, j, data in best_ever_graph.edges(data=True):
            G[i][j]['bond_order'] = data['bond_order']
        
        # Restore caches
        self._restore_graph_caches(G)
        self.valence_cache = best_ever_cache
        
        # Final scoring
        final_score, final_formal_charges = self._score_assignment(G, rings)
        stats['final_score'] = final_score
        stats['final_formal_charges'] = final_formal_charges
        
        self.log("-" * 60, 0)
        self.log(f"Explored {stats['beam_explored']} states across {stats['iterations']} iterations", 1)
        self.log(f"Found {stats['improvements']} improvements", 1)
        self.log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self.log("-" * 60, 0)
        
        return stats


    # =============================================================================
    # AROMATIC DETECTION
    # =============================================================================

    def _detect_aromatic_rings(self, G: nx.Graph) -> int:
        """
        Detect aromatic rings using Hückel rule (4n+2 π electrons). 
        Only performed on 5 and 6 mem rings, with 'C', 'N', 'O', 'S', 'P' # NOTE: possible limitation
            Reduces to C only for now.
        This only sets bond orders to 1.5 for aromatic rings - kekule structure is still a valid solution.
        """
        self.log(f"\n{'=' * 60}", 0)
        self.log("AROMATIC RING DETECTION (Hückel 4n+2)", 0)
        self.log("=" * 60, 0)
        
        cycles = nx.cycle_basis(G)
        aromatic_count = 0
        aromatic_rings = 0
        
        for ring_idx, cycle in enumerate(cycles):
            if len(cycle) not in (5, 6):
                continue
            
            ring_atoms = [f"{G.nodes[i]['symbol']}{i}" for i in cycle]

            # Check if all atoms can be aromatic - just means that other atoms will be kekule structures
            # aromatic_atoms = {'C', 'N', 'O', 'S', 'P', 'B'}
            aromatic_atoms = {'C'}
            if not all(G.nodes[i]['symbol'] in aromatic_atoms for i in cycle):
                non_aromatic = [G.nodes[i]['symbol'] for i in cycle if G.nodes[i]['symbol'] not in aromatic_atoms]
                self.log(f"✗ Contains non-aromatic atoms: {non_aromatic}", 2)
                continue

            is_planar = self._check_planarity(cycle, G)
            if not is_planar:
                self.log(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)
                self.log("✗ Not planar, skipping aromaticity check", 2)
                continue

            self.log(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)
            
            # Count π electrons (simplified)
            pi_electrons = 0
            pi_breakdown = []
            
            for idx in cycle:
                sym = G.nodes[idx]['symbol']
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
            
            self.log(f"π electrons: {pi_electrons} ({', '.join(pi_breakdown)})", 2)
            
            # Hückel rule: 4n+2 π electrons (n = 0, 1, 2, ...)
            is_aromatic = (pi_electrons >= 2 and pi_electrons in (2, 6, 10, 14, 18))
            
            if is_aromatic:
                n = (pi_electrons - 2) // 4
                self.log(f"✓ AROMATIC (4n+2 rule: n={n})", 2)
                # Set all ring edges to 1.5
                ring_edges = [(cycle[k], cycle[(k+1) % len(cycle)]) for k in range(len(cycle))]
                
                bonds_set = 0
                for i, j in ring_edges:
                    if G.has_edge(i, j):
                        old_order = G.edges[i, j]['bond_order']
                        G.edges[i, j]['bond_order'] = 1.5
                        if abs(old_order - 1.5) > 0.01:
                            bonds_set += 1
                            aromatic_count += 1
                
                if bonds_set > 0:
                    aromatic_rings += 1
            else:
                self.log(f"✗ Not aromatic (4n+2 rule violated)", 2)

        self.log(f"\n{'-' * 60}", 0)
        self.log(f"SUMMARY: {aromatic_rings} aromatic rings, {aromatic_count} bonds set to 1.5", 1)
        self.log(f"{'-' * 60}\n", 0)

        return aromatic_count

    def _check_planarity(self, cycle: List[int], G: nx.Graph, threshold: float = 0.15) -> bool:
        """
        Check if a ring is approximately planar.
        """
        if len(cycle) < 3:
            return True
        
        coords = np.array([G.nodes[i]['position'] for i in cycle])
        
        # Fit plane using SVD
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        
        # Plane normal is smallest singular vector
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        
        # Check distance of each point to plane
        distances = np.abs(centered @ normal)
        max_deviation = distances.max()
        
        return max_deviation < threshold

    def _compute_gasteiger_charges(self, G: nx.Graph) -> List[float]:
        """Compute Gasteiger charges using RDKit"""
        try:
            rw = Chem.RWMol()
            for i in G.nodes():
                rw.AddAtom(Chem.Atom(G.nodes[i]['symbol']))
            
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
            
            Chem.AllChem.ComputeGasteigerCharges(mol)
            
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
        
        except Exception as e:
            self.log(f"Gasteiger charge calculation failed: {e}", 2)
            return [0.0] * len(self.atoms)

    # =============================================================================
    # MAIN BUILD FUNCTIONS
    # =============================================================================

    def _build_cheminf(self) -> nx.Graph:
        """
        Build molecular graph using cheminformatics approach.
        """

        if self.multiplicity is None:
            total_electrons = sum(self.atomic_numbers) - self.charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2
        
        # Build initial graph
        G = self._build_initial_graph()
        
        self.log(f"Initial bonds: {G.number_of_edges()}", 1)
        
        # Prune distorted rings
        removed = self._prune_distorted_rings(G)
        if removed > 0:
            self.log(f"Pruned {removed} distorted ring bonds", 1)
        
        # Initialize Kekulé patterns for 6-membered carbon rings (gives optimizer a head start)
        init_rings = self._init_kekule_for_6rings(G)
        stats = None
        # Valence adjustment
        if self.quick:
            stats = self._quick_valence_adjust(G)
        else:
            if self.optimizer == 'greedy':
                stats = self._full_valence_optimize(G)
            if self.optimizer == 'beam':
                stats = self._beam_search_optimize(G)
        
        # Aromatic detection (Hückel rule)
        arom_count = self._detect_aromatic_rings(G)
        
        # Compute charges
        gasteiger_raw = self._compute_gasteiger_charges(G)
        raw_sum = sum(gasteiger_raw)
        delta = (self.charge - raw_sum) / len(self.atoms) if self.atoms else 0.0
        gasteiger_adj = [c + delta for c in gasteiger_raw]
        
        formal_charges = self._compute_formal_charges(G)
        
        # Annotate graph
        for node in G.nodes():
            G.nodes[node]['charges'] = {
                'gasteiger_raw': gasteiger_raw[node],
                'gasteiger': gasteiger_adj[node]
            }
            G.nodes[node]['formal_charge'] = formal_charges[node]
            G.nodes[node]['valence'] = self._valence_sum(G, node)
            
            # Aggregate charge (add H contributions)
            agg = gasteiger_adj[node]
            for nbr in G.neighbors(node):
                if G.nodes[nbr]['symbol'] == 'H':
                    agg += gasteiger_adj[nbr]
            G.nodes[node]['agg_charge'] = agg
        
        # Add bond types
        for i, j, data in G.edges(data=True):
            data['bond_type'] = (G.nodes[i]['symbol'], G.nodes[j]['symbol'])
        
        G.graph['total_charge'] = self.charge
        G.graph['multiplicity'] = self.multiplicity
        G.graph['valence_stats'] = stats
        G.graph['method'] = 'cheminf-quick' if self.quick else 'cheminf-full'
        
        return G

    def _build_xtb(self) -> nx.Graph:
        """Build graph using xTB quantum chemistry calculations"""

        if self.multiplicity is None:
            total_electrons = sum(self.atomic_numbers) - self.charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2

        work = 'xtb_tmp_local'
        basename = 'xtb'
        if os.system('which xtb > /dev/null 2>&1') != 0:
            raise RuntimeError("xTB not found in PATH - install xTB or use 'cheminf' method")

        os.makedirs(work, exist_ok=True)

        # Write XYZ file natively
        xyz_path = os.path.join(work, f"{basename}.xyz")
        with open(xyz_path, 'w') as f:
            f.write(f"{len(self.atoms)}\n")
            f.write(f"xyzgraph generated XYZ for xTB\n")
            for symbol, (x, y, z) in self.atoms:
                f.write(f"{symbol:>2} {x:15.8f} {y:15.8f} {z:15.8f}\n")

        cmd = (f'cd {work} && xtb {basename}.xyz --chrg {self.charge} --uhf {self.multiplicity - 1} --gfn2 > {basename}.out')
        ret = os.system(cmd)

        if ret != 0:
            self.log(f"Warning: xTB returned non-zero exit code {ret}", 1)

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
                    if len(parts) == 3 and float(parts[2]) > 0.5: # bonding threshold
                        bonds.append((int(parts[0])-1, int(parts[1])-1)) # xTB uses 1-indexed
                        bond_orders.append(float(parts[2]))
            self.log(f"Parsed {len(bonds)} bonds from xTB WBO", 1)
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
            self.log(f"Parsed {len(charges)} Mulliken charges from xTB", 1)
        except FileNotFoundError:
            charges = [0.0] * len(self.atoms)

        if self.clean_up:
            try:
                for f in os.listdir(work):
                    os.remove(os.path.join(work, f))
                os.rmdir(work)
            except Exception as e:
                self.log(f"Warning: Could not clean up temp files: {e}", 1)

        # Build graph
        G = nx.Graph()
        pos = self.positions  # Use pre-computed positions

        for i, (symbol, _) in enumerate(self.atoms):
            G.add_node(i,
                    symbol=symbol,
                    atomic_number=self.atomic_numbers[i],
                    position=pos[i],
                    charges={'mulliken': charges[i] if i < len(charges) else 0.0})

        if bonds:
            for (i, j), bo in zip(bonds, bond_orders):
                d = self._distance(pos[i], pos[j])
                si, sj = G.nodes[i]['symbol'], G.nodes[j]['symbol']
                G.add_edge(i, j,
                        bond_order=float(bo),
                        distance=d,
                        bond_type=(si, sj),
                        metal_coord=(si in DATA.metals or sj in DATA.metals))
            self.log(f"Built graph with {G.number_of_edges()} bonds from xTB", 1)
        else:
            # Fallback to distance-based if xTB failed
            self.log(f"Warning: No xTB bonds found, falling back to distance-based, try using `--method cheminf`", 1)
            G = self._build_initial_graph()

        # Add derived properties
        for node in G.nodes():
            G.nodes[node]['valence'] = self._valence_sum(G, node)
            agg = G.nodes[node]['charges'].get('mulliken', 0.0)
            for nbr in G.neighbors(node):
                if G.nodes[nbr]['symbol'] == 'H':
                    agg += G.nodes[nbr]['charges'].get('mulliken', 0.0)
            G.nodes[node]['agg_charge'] = agg

        G.graph['total_charge'] = self.charge
        G.graph['multiplicity'] = self.multiplicity
        G.graph['method'] = 'xtb'

        return G

def build_graph(
            atoms: List[Tuple[str, Tuple[float, float, float]]] | str,
            charge: int = DEFAULT_PARAMS['charge'],
            multiplicity: Optional[int] = DEFAULT_PARAMS['multiplicity'],
            method: str = DEFAULT_PARAMS['method'],
            quick: bool = DEFAULT_PARAMS['quick'],
            optimizer: str = DEFAULT_PARAMS['optimizer'],
            max_iter: int = DEFAULT_PARAMS['max_iter'],
            edge_per_iter: int = DEFAULT_PARAMS['edge_per_iter'],
            beam_width: int = DEFAULT_PARAMS['beam_width'],
            bond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS['bond'],
            unbond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS['unbond'],
            clean_up: bool = DEFAULT_PARAMS['clean_up'],
            debug: bool = DEFAULT_PARAMS['debug'],
            threshold: float = DEFAULT_PARAMS['threshold'],
            threshold_h_h: float = DEFAULT_PARAMS['threshold_h_h'],
            threshold_h_nonmetal: float = DEFAULT_PARAMS['threshold_h_nonmetal'],
            threshold_h_metal: float = DEFAULT_PARAMS['threshold_h_metal'],
            threshold_metal_ligand: float = DEFAULT_PARAMS['threshold_metal_ligand'],
            threshold_nonmetal_nonmetal: float = DEFAULT_PARAMS['threshold_nonmetal_nonmetal']
        ) -> nx.Graph:
    """Convenience function that wraps GraphBuilder.

    atoms: Either a list of (symbol, (x,y,z)) tuples, or a filepath to read.
    """
    # Handle filepath input
    if isinstance(atoms, str):
        atoms = read_xyz_file(atoms)

    builder = GraphBuilder(
        atoms=atoms,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        quick=quick,
        optimizer=optimizer,
        max_iter=max_iter,
        edge_per_iter=edge_per_iter,
        beam_width=beam_width,
        bond=bond,
        unbond=unbond,
        clean_up=clean_up,
        debug=debug,
        threshold=threshold,
        threshold_h_h=threshold_h_h,
        threshold_h_nonmetal=threshold_h_nonmetal,
        threshold_h_metal=threshold_h_metal,
        threshold_metal_ligand=threshold_metal_ligand,
        threshold_nonmetal_nonmetal=threshold_nonmetal_nonmetal
    )
    return builder.build()
