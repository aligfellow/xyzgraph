"""Bond order optimization for molecular graphs.

Assigns bond orders (1.0/1.5/2.0/3.0) and formal charges to a
connectivity graph produced by BondDetector.

Three optimization modes:
- quick: Fast heuristic valence adjustment (no formal charges)
- greedy: Greedy optimizer with formal charge minimization
- beam: Beam search optimizer (default, best quality)

Also handles:
- Kekulé pattern initialization for aromatic rings
- Post-optimization aromatic detection (Hückel 4n+2 rule)
- Formal charge computation and balancing
- Metal-ligand classification and oxidation state inference
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import networkx as nx

from .data_loader import MolecularData
from .geometry import GeometryCalculator
from .parameters import OptimizerConfig, ScoringWeights

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = ScoringWeights()
_DEFAULT_CONFIG = OptimizerConfig()

# Valence violation detection (check_valence_violation)
VALENCE_CHECK_LIMITS: Dict[str, float] = {"C": 4}
VALENCE_CHECK_TOLERANCE = 0.3

# Scoring valence limits (max bond order sum before hard penalty)
SCORING_VALENCE_LIMITS: Dict[str, float] = {"C": 4, "N": 5, "O": 3, "S": 6, "P": 6}
SCORING_VALENCE_TOLERANCE = 0.1

# Quick valence adjust thresholds
QUICK_PROMOTE_DIST_RATIO = 0.60
MIN_DEFICIT_FOR_PROMOTION = 0.3
MIN_BOND_INCREMENT = 0.5

# Greedy optimizer convergence
MAX_STAGNATION_ITERATIONS = 3

# Default electronegativity for unknown elements
DEFAULT_ELECTRONEGATIVITY = 2.5


class BondOrderOptimizer:
    """Assigns bond orders and formal charges to molecular graphs.

    Uses valence rules, electronegativity, and aromatic conjugation
    to find optimal bond order assignment that minimizes a weighted
    penalty score.
    """

    def __init__(
        self,
        geometry: GeometryCalculator,
        data: MolecularData,
        charge: int,
        weights: ScoringWeights = _DEFAULT_WEIGHTS,
        config: OptimizerConfig = _DEFAULT_CONFIG,
    ):
        self.geometry = geometry
        self.data = data
        self.charge = charge
        self.weights = weights
        self.config = config
        self.log_buffer: List[str] = []

        # Optimization state (caches)
        self.valence_cache: Dict[int, float] = {}
        self.edge_scores_cache: Optional[List[Tuple[int, int]]] = None
        self._edge_score_map: Optional[Dict[Tuple[int, int], float]] = None

    def _log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        indent = "  " * level
        line = f"{indent}{msg}"
        logger.debug(line)
        self.log_buffer.append(line)

    def get_log(self) -> List[str]:
        """Return accumulated log messages."""
        return self.log_buffer

    # =========================================================================
    # Static utilities
    # =========================================================================

    @staticmethod
    def valence_sum(G: nx.Graph, node: int) -> float:
        """Sum bond orders around a node."""
        return sum(G.edges[node, nbr].get("bond_order", 1.0) for nbr in G.neighbors(node))

    @staticmethod
    def _compute_formal_charge_value(symbol: str, valence_electrons: int, bond_order_sum: float) -> int:
        """Compute formal charge for an atom."""
        if symbol == "H":
            return valence_electrons - int(bond_order_sum)

        B = 2 * bond_order_sum
        target = 8
        L = max(0, target - B)
        return round(valence_electrons - L - B / 2)

    @staticmethod
    def _ekey(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    @staticmethod
    def _copy_graph_state(G: nx.Graph) -> nx.Graph:
        """Create INDEPENDENT copy of graph for beam exploration."""
        G_new = nx.Graph()

        # Copy nodes with INDEPENDENT attribute dicts
        for node, data in G.nodes(data=True):
            G_new.add_node(
                node,
                symbol=data["symbol"],
                atomic_number=data["atomic_number"],
                position=data["position"],
            )

        # Copy edges with INDEPENDENT attribute dicts
        for i, j, data in G.edges(data=True):
            G_new.add_edge(
                i,
                j,
                bond_order=float(data["bond_order"]),
                distance=float(data["distance"]),
                metal_coord=bool(data.get("metal_coord", False)),
            )

        return G_new

    # =========================================================================
    # Public API
    # =========================================================================

    def optimize(self, G: nx.Graph, mode: str = "beam", quick: bool = False) -> Dict[str, Any]:
        """Optimize bond orders.

        Parameters
        ----------
        G : nx.Graph
            Graph with initial bond_order=1.0 edges.
        mode : str
            Optimizer: "greedy" or "beam".
        quick : bool
            Use fast heuristic (no formal charge optimization).

        Returns
        -------
        dict
            Statistics about the optimization run.
        """
        if quick:
            return self._quick_valence_adjust(G)
        if mode == "greedy":
            return self._full_valence_optimize(G)
        if mode == "beam":
            return self._beam_search_optimize(G)
        raise ValueError(f"Unknown optimizer mode: {mode}")

    # =========================================================================
    # Validation
    # =========================================================================

    def check_valence_violation(
        self,
        G: nx.Graph,
        limits: Optional[Dict[str, float]] = None,
        tol: float = VALENCE_CHECK_TOLERANCE,
    ) -> bool:
        """Check for pentavalent carbon etc."""
        if limits is None:
            limits = VALENCE_CHECK_LIMITS

        for i in G.nodes():
            sym = G.nodes[i]["symbol"]
            if sym in limits:
                # Exclude metal bonds from valence
                val = sum(
                    G[i][j].get("bond_order", 1.0)
                    for j in G.neighbors(i)
                    if G.nodes[j]["symbol"] not in self.data.metals
                )
                if val > limits[sym] + tol:
                    return True
        return False

    # =========================================================================
    # Formal charge computation
    # =========================================================================

    def compute_formal_charges(self, G: nx.Graph) -> List[int]:
        """Compute formal charges for all atoms and balance to total charge."""
        formal = []

        self._log("\n" + "=" * 80, 0)
        self._log("FORMAL CHARGE CALCULATION", 0)
        self._log("=" * 80, 0)

        for node in G.nodes():
            sym = G.nodes[node]["symbol"]

            if sym in self.data.metals:
                formal.append(0)
                continue

            V = self.data.electrons.get(sym)
            if V is None:
                formal.append(0)
                continue

            # Exclude metal bonds from ligand valence for formal charge calculation
            bond_sum = sum(
                G.edges[node, nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(node)
                if G.nodes[nbr]["symbol"] not in self.data.metals
            )

            # Special case: H bonded only to metal(s) is hydride (H⁻)
            if sym == "H" and bond_sum == 0:
                if all(G.nodes[nbr]["symbol"] in self.data.metals for nbr in G.neighbors(node)):
                    fc = -1  # Hydride
                    formal.append(fc)
                    continue

            fc = self._compute_formal_charge_value(sym, V, bond_sum)
            formal.append(fc)

        # Check if system has metals
        has_metals = any(G.nodes[i]["symbol"] in self.data.metals for i in G.nodes())

        # Log initial formal charges
        initial_sum = sum(formal)
        self._log("\nInitial formal charges:", 2)
        self._log(f"  Sum: {initial_sum:+d} (target: {self.charge:+d})", 3)

        if has_metals:
            # Show metal coordination summary
            self._log("\nMetal coordination summary:", 3)

            # Compute ligand classification inline, passing formal charges
            ligand_classification = self.classify_metal_ligands(G, formal)

            for metal_idx, ox_state in sorted(ligand_classification["metal_ox_states"].items()):
                metal_sym = G.nodes[metal_idx]["symbol"]
                coord_num = len(list(G.neighbors(metal_idx)))

                # Get ligands for this metal
                metal_dative = [entry for entry in ligand_classification["dative_bonds"] if entry[0] == metal_idx]
                metal_ionic = [entry for entry in ligand_classification["ionic_bonds"] if entry[0] == metal_idx]

                self._log(
                    f"\n[{metal_idx:>3}] {metal_sym}  oxidation_state={ox_state:+d}  coordination={coord_num}",
                    4,
                )

                # Sort and display charged ligands first
                if metal_ionic:
                    sorted_ionic = sorted(metal_ionic, key=lambda x: x[2])
                    for entry in sorted_ionic:
                        _m, donor, chg, ligand_type = entry if len(entry) == 4 else (*entry, "unknown")
                        d_sym = G.nodes[donor]["symbol"]
                        charge_str = f"{chg:+d}" if chg != 0 else " 0"
                        self._log(
                            f"  • {ligand_type:>6} ({charge_str})  [donor: {d_sym}{donor}]",
                            4,
                        )

                # Display neutral ligands
                if metal_dative:
                    for entry in metal_dative:
                        _m, donor, ligand_type = entry if len(entry) == 3 else (*entry, "unknown")
                        d_sym = G.nodes[donor]["symbol"]
                        self._log(f"  • {ligand_type:>6} ( 0)  [donor: {d_sym}{donor}]", 4)
        else:
            # No metals - show traditional formal charge list
            charged_atoms = [(i, formal[i]) for i in range(len(formal)) if formal[i] != 0]
            if charged_atoms:
                self._log("  Charged atoms:", 3)
                for i, fc in charged_atoms:
                    sym = G.nodes[i]["symbol"]
                    self._log(f"    {sym}{i}: {fc:+d}", 4)
            else:
                self._log("  (no charged atoms)", 3)

        # Balance residual charge with priority-based distribution
        residual = self.charge - sum(formal)

        # Check if system has metals - if so, skip redistribution
        has_metals = any(G.nodes[i]["symbol"] in self.data.metals for i in G.nodes())

        if residual != 0 and not has_metals:
            self._log("\nResidual charge distribution needed:", 2)
            self._log(f"  Residual: {residual:+d}", 3)

            candidates = []
            for i in G.nodes():
                if self.valence_sum(G, i) == 0:
                    continue

                sym = G.nodes[i]["symbol"]
                if sym in self.data.metals:
                    continue

                # Skip atoms bonded to metals
                bonded_to_metal = any(G.nodes[nbr]["symbol"] in self.data.metals for nbr in G.neighbors(i))
                if bonded_to_metal:
                    continue

                score = 0

                # Priority: heteroatoms (more electronegative, better charge bearers)
                if sym in ("O", "N", "S", "Cl", "Br", "I", "F", "P"):
                    score += 5

                # Lower priority: already charged
                if abs(formal[i]) > 0:
                    score += 2

                candidates.append((score, i))

            candidates.sort(reverse=True, key=lambda x: x[0])

            self._log("  Top candidates (showing first 10):", 3)
            for score, idx in candidates[:10]:
                sym = G.nodes[idx]["symbol"]
                current_fc = formal[idx]
                self._log(f"    {sym}{idx}: score={score}, current_fc={current_fc:+d}", 4)

            # Distribute charge
            sign = 1 if residual > 0 else -1
            distributed_to = []
            for _, idx in candidates:
                if residual == 0:
                    break
                formal[idx] += sign
                residual -= sign
                distributed_to.append((G.nodes[idx]["symbol"], idx, formal[idx]))

            self._log(f"  Distributed to {len(distributed_to)} atoms:", 3)
            for sym, idx, new_fc in distributed_to:
                self._log(f"    {sym}{idx}: {new_fc:+d}", 4)
        elif residual != 0 and has_metals:
            self._log("\nMetal complex detected: ", 2)
            self._log(f"  Residual: {residual:+d} (represents metal oxidation states)", 3)

            # Assign oxidation states as formal charges on metals
            for metal_idx, ox_state in ligand_classification["metal_ox_states"].items():
                formal[metal_idx] = ox_state
                self._log(
                    f"  {G.nodes[metal_idx]['symbol']}{metal_idx}: formal_charge={ox_state:+d}",
                    3,
                )

            # Handle remaining residual for isolated metals (no bonds, no ox_state)
            residual = self.charge - sum(formal)
            if residual != 0:
                for i in G.nodes():
                    if residual == 0:
                        break
                    if G.nodes[i]["symbol"] in self.data.metals and len(list(G.neighbors(i))) == 0 and formal[i] == 0:
                        sign = 1 if residual > 0 else -1
                        formal[i] += sign
                        residual -= sign
                        self._log(f"  Assigned {sign:+d} to isolated {G.nodes[i]['symbol']}{i}", 3)
        else:
            self._log("\nNo residual charge distribution needed (sum matches target)", 2)

        return formal

    # =========================================================================
    # Aromatic initialization (Kekulé patterns)
    # =========================================================================

    # Typical degree when all bonds are single.  If an exocyclic neighbour's
    # actual degree is below this, it must form a multiple bond to the ring
    # atom, consuming that atom's p-orbital for exocyclic π rather than ring
    # conjugation.
    _EXO_PI_THRESHOLD: ClassVar[Dict[str, int]] = {"N": 3, "O": 2, "S": 2, "P": 3}

    def _estimate_pi_electrons(self, G: nx.Graph, cycle: List[int]) -> int:
        """Estimate π electrons using metal-bonding as hint.

        Heuristic: 5-membered C-ring bonded to metal → likely Cp⁻ → π=6
        A ring carbon whose exocyclic neighbour is unsaturated (degree below
        its typical single-bond valence) has its p-orbital engaged in an
        exocyclic π bond and contributes 0 π electrons to the ring.
        """
        pi_electrons = 0
        bonded_to_metal = any(any(G.nodes[nbr]["symbol"] in self.data.metals for nbr in G.neighbors(c)) for c in cycle)
        cycle_set = set(cycle)

        for idx in cycle:
            sym = G.nodes[idx]["symbol"]
            if sym == "C":
                has_exo_pi = any(
                    G.degree(nbr) < self._EXO_PI_THRESHOLD.get(G.nodes[nbr]["symbol"], 0)
                    for nbr in G.neighbors(idx)
                    if nbr not in cycle_set
                )
                pi_electrons += 0 if has_exo_pi else 1
            elif sym == "N":
                degree = sum(1 for nbr in G.neighbors(idx) if G.nodes[nbr]["symbol"] not in self.data.metals)
                if degree == 3:
                    pi_electrons += 2  # Pyrrole-like
                elif degree == 2:
                    pi_electrons += 1  # Pyridine-like
            elif sym in ("O", "S"):
                pi_electrons += 2

        # 5-ring bonded to metal with 5 π electrons → assume Cp⁻ (6 total)
        if len(cycle) == 5 and bonded_to_metal and pi_electrons == 5:
            pi_electrons += 1

        return pi_electrons

    def init_kekule(self, G: nx.Graph) -> int:
        """Initialize Kekulé patterns for aromatic rings.

        1) Validate rings (planarity, aromatic atoms, sp2 carbons, Huckel, Cp-like).
        2) Initialize Kekulé patterns with propagation respecting fused rings.
        """
        cycles = G.graph.get("_rings")
        if cycles is None:
            cycles = nx.cycle_basis(G)
            G.graph["_rings"] = cycles

        self._log("\n" + "=" * 80, 0)
        self._log("KEKULE INITIALIZATION FOR AROMATIC RINGS", 0)
        self._log("=" * 80, 0)

        # --- Phase 0: Precompute edge info ---
        edge_to_rings: Dict[frozenset, List[int]] = {}
        ring_edges = []
        ring_symbols = []
        for r_idx, cycle in enumerate(cycles):
            edges = []
            for k in range(len(cycle)):
                a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                edges.append((a, b))
                key = frozenset((a, b))
                edge_to_rings.setdefault(key, []).append(r_idx)
            ring_edges.append(edges)
            ring_symbols.append({G.nodes[i]["symbol"] for i in cycle})

        ring_adj: Dict[int, set] = {i: set() for i in range(len(cycles))}
        for _edge_key, rings_list in edge_to_rings.items():
            if len(rings_list) > 1:
                for a in rings_list:
                    for b in rings_list:
                        if a != b:
                            ring_adj[a].add(b)

        # --- Phase 1: Ring validation / logging ---
        valid_rings = set()
        for r_idx, cycle in enumerate(cycles):
            if len(cycle) not in (5, 6):
                continue

            ring_atoms_str = [f"{G.nodes[i]['symbol']}{i}" for i in cycle]
            self._log(f"\nRing {r_idx} ({len(cycle)}-membered): {ring_atoms_str}", 2)

            # Must contain only aromatic atoms
            if not all(G.nodes[i]["symbol"] in self.data.conjugatable_atoms for i in cycle):
                self._log("✗ Contains non-conjugatable atoms", 3)
                continue

            # Check planarity
            if not self.geometry.check_planarity(cycle, G, tolerance=0.15):
                self._log("✗ Not planar", 3)
                continue

            # Check for sp3 carbon
            has_sp3 = False
            for idx in cycle:
                sym = G.nodes[idx]["symbol"]
                if sym == "C":
                    degree = sum(1 for nbr in G.neighbors(idx) if G.nodes[nbr]["symbol"] not in self.data.metals)
                    if degree >= 4:
                        self._log(f"✗ Contains non-sp2 carbon {sym}{idx}", 3)
                        has_sp3 = True
                        break
            if has_sp3:
                continue

            # Cp-like detection for 5-membered carbon rings
            if len(cycle) == 5 and all(G.nodes[i]["symbol"] == "C" for i in cycle):
                metal_neighbors: Dict[int, List[int]] = {}
                for c in cycle:
                    for nbr in G.neighbors(c):
                        if G.nodes[nbr]["symbol"] in self.data.metals:
                            metal_neighbors.setdefault(nbr, []).append(c)
                is_cp_like = any(len(carbons) == 5 for carbons in metal_neighbors.values())
                if is_cp_like:
                    metal_idx = next(m for m, carbons in metal_neighbors.items() if len(carbons) == 5)
                    metal_sym = G.nodes[metal_idx]["symbol"]
                    self._log(
                        f"✓ Detected Cp-like ring (all 5 C bonded to {metal_sym}{metal_idx})",
                        3,
                    )

            # Estimate π electrons and Hückel rule
            pi_electrons = self._estimate_pi_electrons(G, cycle)
            self._log(f"π electrons estimate: {pi_electrons}", 3)
            if pi_electrons not in (6, 10):
                self._log(f"✗ Hückel rule violated (π={pi_electrons})", 3)
                continue

            valid_rings.add(r_idx)

        if not valid_rings:
            self._log("No rings passed validation, skipping Kekulé init", 1)
            return 0

        self._log(f"{'-' * 80}", 0)
        self._log(f"Valid rings for Kekulé initialization: \n\t{sorted(valid_rings)}", 0)

        # --- Phase 2: Kekulé initialization ---
        processed_rings: set = set()  # Track rings handled by any priority

        def max_val(n):
            return self.data.max_aromatic_valence.get(G.nodes[n].get("symbol"), 4)

        def bond_sum(node, ignore_edge=None):
            s = 0.0
            for nbr in G.neighbors(node):
                if ignore_edge is not None:
                    a, b = ignore_edge
                    if (node == a and nbr == b) or (node == b and nbr == a):
                        continue
                s += float(G.edges[node, nbr].get("bond_order", 1.0))
            return s

        def can_set_edge(i, j, new_bo):
            return (
                bond_sum(i, ignore_edge=(i, j)) + new_bo <= max_val(i) + 1e-9
                and bond_sum(j, ignore_edge=(i, j)) + new_bo <= max_val(j) + 1e-9
            )

        def apply_pattern(r_idx, pattern):
            if r_idx not in valid_rings:
                return False
            edges = ring_edges[r_idx]
            assigns = []
            for idx, (i, j) in enumerate(edges):
                existing = float(G.edges[i, j].get("bond_order", 1.0))
                desired = float(pattern[idx])
                if abs(existing - 1.0) > 0.01 and ((existing > 1.5) != (desired > 1.5)):
                    return False
                if not can_set_edge(i, j, desired):
                    if desired > 1.5 and can_set_edge(i, j, 1.0):
                        desired = 1.0
                    else:
                        return False
                assigns.append((i, j, desired))
            for i, j, bo in assigns:
                G.edges[i, j]["bond_order"] = bo
            return True

        def alt_patterns(L, start_with_double=True):
            return (
                [2.0 if k % 2 == 0 else 1.0 for k in range(L)]
                if start_with_double
                else [1.0 if k % 2 == 0 else 2.0 for k in range(L)]
            )

        # --- Priority 1: Cp-like 5-membered rings ---
        for r_idx in valid_rings:
            if len(cycles[r_idx]) != 5:
                continue
            if not all(G.nodes[i]["symbol"] == "C" for i in cycles[r_idx]):
                continue
            # Detect metal-bound Cp
            metal_map: Dict[int, List[int]] = {}
            for c in cycles[r_idx]:
                for nbr in G.neighbors(c):
                    if G.nodes[nbr]["symbol"] in self.data.metals:
                        metal_map.setdefault(nbr, []).append(c)
            if any(len(cs) == 5 for cs in metal_map.values()):
                # Apply alternating pattern [1,2,1,2,1] rotated to best match existing anchors
                L = 5
                base = [1.0, 2.0, 1.0, 2.0, 1.0]
                applied = False
                for rot in range(L):
                    p = base[-rot:] + base[:-rot]
                    if apply_pattern(r_idx, p):
                        processed_rings.add(r_idx)
                        applied = True
                        self._log(f"✓ Cp-like 5-ring {r_idx} initialized (rotation {rot})", 3)
                        break
                if not applied:
                    self._log(f"✗ Cp-like 5-ring {r_idx} could not be safely applied", 3)

        # --- Priority 2: 5-membered heterocycles (LP-in) ---
        hetero_initialized = set()
        for r_idx in valid_rings:
            if len(cycles[r_idx]) != 5:
                continue
            if not any(G.nodes[i]["symbol"] in ("N", "O", "S", "B") for i in cycles[r_idx]):
                continue
            lp = None
            for idx in cycles[r_idx]:
                sym = G.nodes[idx]["symbol"]
                if sym not in ("N", "O", "S", "B"):
                    continue
                neighbors = len(list(G.neighbors(idx)))
                if sym == "N" and neighbors == 3:
                    lp = idx
                    break
                if sym in ("O", "S") and neighbors == 2:
                    lp = idx
                    break
            if lp is not None:
                cycle = cycles[r_idx]
                pos = cycle.index(lp)
                p = [1.0] * 5
                p[pos] = 1.0
                p[(pos + 1) % 5] = 2.0
                p[(pos + 2) % 5] = 1.0
                p[(pos + 3) % 5] = 2.0
                p[(pos + 4) % 5] = 1.0
                if apply_pattern(r_idx, p):
                    hetero_initialized.add(r_idx)
                    processed_rings.add(r_idx)
                    self._log(f"✓ 5-heterocycle {r_idx} (lp {lp}) initialized", 3)
                else:
                    self._log(
                        f"✗ 5-heterocycle {r_idx} (lp {lp}) could not be safely applied",
                        3,
                    )

        # --- Priority 2b: propagate to fused rings ---
        to_propagate: set = set()
        for r in hetero_initialized:
            to_propagate |= ring_adj[r]
        for r_idx in sorted(to_propagate):
            if r_idx not in valid_rings or r_idx in hetero_initialized:
                continue
            L = len(cycles[r_idx])
            success = False
            if L == 6:
                for start_double in (True, False):
                    p = alt_patterns(6, start_with_double=start_double)
                    if apply_pattern(r_idx, p):
                        processed_rings.add(r_idx)
                        success = True
                        self._log(f"✓ Propagated init to fused ring {r_idx} (6-ring)", 3)
                        break
            elif L == 5:
                base = [2.0, 1.0, 2.0, 1.0, 1.0]
                for rot in range(5):
                    p = base[-rot:] + base[:-rot]
                    if apply_pattern(r_idx, p):
                        processed_rings.add(r_idx)
                        success = True
                        self._log(
                            f"✓ Propagated init to fused ring {r_idx} (5-ring rotation {rot})",
                            3,
                        )
                        break
            if not success:
                self._log(f"• Could not propagate safely to fused ring {r_idx}", 4)

        # --- Priority 3 & 4: fused benzene clusters + isolated 6-rings ---
        six_ring_indices = [i for i in valid_rings if len(cycles[i]) == 6]
        if six_ring_indices:
            sub_adj: Dict[int, set] = {i: set() for i in six_ring_indices}
            for i in six_ring_indices:
                for j in ring_adj[i]:
                    if j in sub_adj:
                        sub_adj[i].add(j)
            seen: set = set()
            for start in six_ring_indices:
                if start in seen:
                    continue
                comp: set = set()
                stack = [start]
                while stack:
                    x = stack.pop()
                    if x in comp:
                        continue
                    comp.add(x)
                    for nb in sub_adj.get(x, ()):
                        if nb not in comp:
                            stack.append(nb)
                seen |= comp
                if len(comp) == 1:
                    # isolated 6-ring: handle later
                    continue
                comp_sorted = sorted(comp)

                # global propagation with two parity seeds
                def try_component(seed_parity, comp):
                    assigned = {comp[0]: alt_patterns(6, start_with_double=seed_parity)}
                    queue = [comp[0]]
                    while queue:
                        r = queue.pop(0)
                        patt = assigned[r]
                        for nb in sub_adj[r]:
                            if nb not in comp:
                                continue
                            shared_edges = []
                            for idx_e, (a, b) in enumerate(ring_edges[r]):
                                key = frozenset((a, b))
                                if nb in edge_to_rings.get(key, []):
                                    shared_edges.append((idx_e, key))
                            if nb in assigned:
                                consistent = True
                                for idx_e, key in shared_edges:
                                    for idx_nb, (ua, ub) in enumerate(ring_edges[nb]):
                                        if frozenset((ua, ub)) == key and (assigned[nb][idx_nb] > 1.5) != (
                                            patt[idx_e] > 1.5
                                        ):
                                            consistent = False
                                            break
                                    if not consistent:
                                        break
                                if not consistent:
                                    return None
                                continue
                            ok = False
                            for start_bool in (True, False):
                                candidate = alt_patterns(6, start_with_double=start_bool)
                                good = True
                                for idx_e, key in shared_edges:
                                    for idx_nb, (ua, ub) in enumerate(ring_edges[nb]):
                                        if frozenset((ua, ub)) == key and (candidate[idx_nb] > 1.5) != (
                                            patt[idx_e] > 1.5
                                        ):
                                            good = False
                                            break
                                    if not good:
                                        break
                                if good:
                                    assigned[nb] = candidate
                                    queue.append(nb)
                                    ok = True
                                    break
                            if not ok:
                                return None
                    # valence check and commit (with rollback on failure)
                    originals: Dict[tuple, float] = {}
                    for r in comp:
                        patt = assigned[r]
                        for idx_edge, (i, j) in enumerate(ring_edges[r]):
                            bo = patt[idx_edge]
                            if not can_set_edge(i, j, bo):
                                # Rollback all changes made so far
                                for (oi, oj), orig_bo in originals.items():
                                    G.edges[oi, oj]["bond_order"] = orig_bo
                                return None
                            key = (min(i, j), max(i, j))
                            if key not in originals:
                                originals[key] = G.edges[i, j].get("bond_order", 1.0)
                            G.edges[i, j]["bond_order"] = bo
                    return assigned

                assigned = None
                for seed in (True, False):
                    assigned = try_component(seed, comp_sorted)
                    if assigned is not None:
                        processed_rings.update(comp_sorted)
                        self._log(f"✓ Initialized fused benzene block rings {comp_sorted}", 3)
                        break
                if assigned is None:
                    self._log(
                        f"✗ Could not find consistent Kekulé for fused benzene block {comp_sorted}",
                        3,
                    )

        # --- Priority 4: isolated 6-membered rings ---
        for r_idx in six_ring_indices:
            if r_idx in processed_rings:
                continue
            # Choose parity so double bonds fall on edges that give
            # the optimizer a head start toward the correct bonding.
            preferred = True
            for k, (i, j) in enumerate(ring_edges[r_idx]):
                if (
                    G.nodes[i]["symbol"] == "C"
                    and G.nodes[j]["symbol"] == "C"
                    and sum(1 for nb in G.neighbors(i) if G.nodes[nb]["symbol"] not in self.data.metals) < 3
                    and sum(1 for nb in G.neighbors(j) if G.nodes[nb]["symbol"] not in self.data.metals) < 3
                ):
                    preferred = k % 2 == 0
                    break

            applied = False
            for sd in (preferred, not preferred):
                if apply_pattern(r_idx, alt_patterns(6, start_with_double=sd)):
                    processed_rings.add(r_idx)
                    self._log(f"✓ Initialized isolated 6-ring {r_idx}", 3)
                    applied = True
                    break
            if not applied:
                self._log(f"• Could not safely init isolated 6-ring {r_idx}", 4)

        # --- Priority 5: remaining carbon-only 5-membered rings ---
        for r_idx in valid_rings:
            if len(cycles[r_idx]) != 5:
                continue
            if any(G.nodes[i]["symbol"] != "C" for i in cycles[r_idx]):
                continue
            if r_idx in processed_rings:
                continue
            fused = any(len(edge_to_rings[frozenset((a, b))]) > 1 for a, b in ring_edges[r_idx])
            if fused:
                self._log(f"• Skipping fused carbon-5 ring {r_idx}", 4)
                continue
            pattern = [2.0, 1.0, 2.0, 1.0, 1.0]
            if apply_pattern(r_idx, pattern):
                processed_rings.add(r_idx)
                self._log(f"✓ Initialized isolated carbon-5 ring {r_idx}", 3)
            else:
                self._log(f"• Could not safely init isolated carbon-5 ring {r_idx}", 4)

        self._log("\n" + "-" * 80, 0)
        self._log(f"SUMMARY: Initialized {len(processed_rings)} ring(s) with Kekulé pattern", 1)
        self._log("-" * 80, 0)
        return len(processed_rings)

    # =========================================================================
    # Quick mode: Simple heuristic valence adjustment
    # =========================================================================

    def _quick_valence_adjust(self, G: nx.Graph) -> Dict[str, int]:
        """Perform fast heuristic bond order adjustment.

        No formal charge optimization - just satisfy valences.
        """
        stats: Dict[str, int] = {"iterations": 0, "promotions": 0}

        # Lock metal bonds
        for i, j in G.edges():
            if G.edges[i, j].get("metal_coord", False):
                G.edges[i, j]["bond_order"] = 1.0

        for iteration in range(3):
            stats["iterations"] = iteration + 1
            changed = False

            # Calculate deficits
            deficits: Dict[int, float] = {}
            for node in G.nodes():
                sym = G.nodes[node]["symbol"]
                if sym in self.data.metals:
                    deficits[node] = 0.0
                    continue

                current = self.valence_sum(G, node)
                allowed = self.data.valences.get(sym, [])
                if not allowed:
                    deficits[node] = 0.0
                    continue

                target = min(allowed, key=lambda v: abs(v - current))
                deficits[node] = target - current

            # Try to promote bonds
            for i, j, data in G.edges(data=True):
                if data.get("metal_coord", False):
                    continue

                si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
                if "H" in (si, sj):
                    continue

                bo = data["bond_order"]
                if bo >= 3.0:
                    continue

                di, dj = deficits[i], deficits[j]

                # Check geometry
                dist_ratio = data["distance"] / (self.data.vdw.get(si, 2.0) + self.data.vdw.get(sj, 2.0))
                if dist_ratio > QUICK_PROMOTE_DIST_RATIO:
                    continue

                # Promote if both atoms need more valence
                if di > MIN_DEFICIT_FOR_PROMOTION and dj > MIN_DEFICIT_FOR_PROMOTION:
                    increment = min(di, dj, 3.0 - bo)
                    if increment >= MIN_BOND_INCREMENT:
                        data["bond_order"] = bo + increment
                        stats["promotions"] += 1
                        changed = True
            self._log(f"Iteration {iteration + 1}: Promotions={stats['promotions']}", 1)

            if not changed:
                break

        return stats

    # =========================================================================
    # Edge scoring and selection
    # =========================================================================

    def _edge_score(self, G: nx.Graph, i: int, j: int) -> float:
        """Check scoring of edge."""
        if not self._eligible_edge(G, i, j):
            return float("-inf")
        si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
        vmax_i = max(self.data.valences.get(si, [4]))
        vmax_j = max(self.data.valences.get(sj, [4]))
        di = vmax_i - self.valence_cache[i]
        dj = vmax_j - self.valence_cache[j]
        return di + dj

    @staticmethod
    def _eligible_edge(G: nx.Graph, i: int, j: int) -> bool:
        data = G[i][j]
        if data.get("metal_coord", False):
            return False
        if data.get("locked", False):
            return False
        if data.get("bond_order", 1.0) >= 3.0:
            return False
        return True

    def _edge_likelihood(
        self, G: nx.Graph, *, init: bool = False, touch_nodes: Optional[set] = None
    ) -> List[Tuple[int, int]]:
        """Select candidate edges for bond order optimization.

        - init=True: build score map for all edges once.
        - touch_nodes={u,v}: update edges belonging to these nodes.
        - return current top-k edges as a list [(i,j), ...].
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
        items = [(s, e) for e, s in self._edge_score_map.items() if s != float("-inf")]
        items.sort(key=lambda t: (-t[0], t[1][0], t[1][1]))  # sort by score desc, then by edge
        top = [e for _, e in items[: self.config.edge_per_iter]]
        self.edge_scores_cache = top
        return top

    # =========================================================================
    # Scoring
    # =========================================================================

    def _ring_conjugation_penalty(self, G: nx.Graph, rings: List[List[int]]) -> float:
        """Assess conjugation penalties in aromatic rings (5-6 members).

        Returns a numeric penalty (larger = worse).
        """
        # Atoms belonging to any ring — fused-ring junction bonds (where the
        # neighbour is also a ring member) are not true exocyclic substituents.
        all_ring_atoms: set = set()
        for c in rings:
            all_ring_atoms.update(c)

        conjugation_penalty = 0.0
        for ring in rings:
            if len(ring) not in (5, 6):
                continue

            if not all(G.nodes[i]["symbol"] in self.data.scoring_conjugatable_atoms for i in ring):
                continue

            ring_set = set(ring)
            elevated_bonds = 0
            exocyclic_double = 0

            # --- Bonds within the ring ---
            ring_edges = [(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]
            for i, j in ring_edges:
                bo = G[i][j].get("bond_order", 1.0)
                if bo > 1.3:
                    elevated_bonds += 1

            # --- Exocyclic double bonds ---
            for ring_atom in ring:
                ring_sym = G.nodes[ring_atom]["symbol"]
                for nbr, data in G[ring_atom].items():
                    if nbr not in ring_set:
                        nbr_sym = G.nodes[nbr]["symbol"]

                        # Skip fused-ring junction bonds
                        if nbr in all_ring_atoms:
                            continue

                        # Skip metal bonds
                        if nbr_sym in self.data.metals:
                            continue

                        bo = data.get("bond_order", 1.0)
                        if bo >= 1.8:
                            if (ring_sym == "C" and nbr_sym != "O") or (ring_sym == "N" and nbr_sym in ("C", "P", "S")):
                                exocyclic_double += 1

            # --- Scoring logic ---
            expected_elevated = len(ring) // 2
            if elevated_bonds >= expected_elevated - 1:
                if exocyclic_double > 0:
                    conjugation_penalty += exocyclic_double * self.weights.exocyclic_double_penalty
            else:
                deficit = (expected_elevated - 1) - elevated_bonds
                if deficit > 0:
                    conjugation_penalty += deficit * self.weights.conjugation_deficit_penalty
                    if exocyclic_double > 0:
                        conjugation_penalty += exocyclic_double * self.weights.exocyclic_double_penalty

        return conjugation_penalty

    def _score_assignment(self, G: nx.Graph, rings: Optional[List[List[int]]] = None) -> Tuple[float, List[int]]:
        """Scoring that uses pre-computed valence cache."""
        if self.check_valence_violation(G):
            return 1e9, [0 for _ in G.nodes()]

        # Ring cache
        if rings is None:
            rings = G.graph.get("_rings", nx.cycle_basis(G))

        # Neighbor cache
        neighbor_cache = G.graph.get("_neighbors", {n: list(G.neighbors(n)) for n in G.nodes()})

        # H-neighbor cache
        has_H = G.graph.get(
            "_has_H",
            {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G[n]) for n in G.nodes()},
        )

        # Penalty accumulators
        penalties = {
            "violation": 0.0,
            "conjugation": 0.0,
            "protonation": 0.0,
            "valence": 0.0,
            "en": 0.0,
            "fc": 0,
            "n_charged": 0,
        }

        # Conjugation penalty
        penalties["conjugation"] = self._ring_conjugation_penalty(G, rings)

        # Formal charge cache
        formal_cache: Dict[Tuple[str, float], int] = {}

        def get_formal(sym, vsum):
            key = (sym, round(vsum, 2))
            if key not in formal_cache:
                V = self.data.electrons.get(sym, 0)
                formal_cache[key] = self._compute_formal_charge_value(sym, V, vsum)
            return formal_cache[key]

        formal_charges = []

        for node in G.nodes():
            sym = G.nodes[node]["symbol"]

            vsum = self.valence_cache[node]

            if sym in self.data.metals:
                formal_charges.append(0)
                continue

            fc = get_formal(sym, vsum)
            formal_charges.append(fc)

            if fc != 0:
                penalties["fc"] += abs(fc)
                penalties["n_charged"] += 1

            nb = neighbor_cache[node]
            if has_H[node]:
                if fc == 0:
                    for nbr in nb:
                        if G.nodes[nbr]["symbol"] != "H":
                            other_fc = get_formal(G.nodes[nbr]["symbol"], self.valence_cache[nbr])
                            if other_fc > 0:
                                penalties["protonation"] += 8.0 if sym in ("N", "O") else 3.0
                elif fc > 0 and sym in ("N", "O", "S"):
                    penalties["en"] -= 1.5

            # Valence error
            if sym in self.data.valences:
                allowed = self.data.valences[sym]
                min_error = min(abs(vsum - v) for v in allowed)
                penalties["valence"] += min_error**2

                if sym in SCORING_VALENCE_LIMITS and vsum > SCORING_VALENCE_LIMITS[sym] + SCORING_VALENCE_TOLERANCE:
                    penalties["violation"] += self.weights.violation_weight

            # Electronegativity penalty
            en = self.data.electronegativity.get(sym, DEFAULT_ELECTRONEGATIVITY)
            if fc != 0:
                penalties["en"] += abs(fc) * ((3.5 - en) if fc < 0 else (en - 2.5)) * 0.5

        # Total score
        charge_error = abs(sum(formal_charges) - self.charge)
        score = (
            self.weights.violation_weight * penalties["violation"]
            + self.weights.conjugation_weight * penalties["conjugation"]
            + self.weights.protonation_weight * penalties["protonation"]
            + self.weights.formal_charge_weight * penalties["fc"]
            + self.weights.charged_atoms_weight * penalties["n_charged"]
            + self.weights.charge_error_weight * charge_error
            + self.weights.electronegativity_weight * penalties["en"]
            + self.weights.valence_error_weight * penalties["valence"]
        )

        return score, formal_charges

    # =========================================================================
    # Valence cache management
    # =========================================================================

    def _update_valence_cache(self, G: nx.Graph, nodes: Optional[set] = None) -> None:
        """Update valence cache for specific nodes or all nodes.

        Excludes metal bonds to match behavior in optimization methods.
        """
        if nodes is None:
            # Full rebuild (excluding metal bonds)
            self.valence_cache = {
                n: sum(
                    G[n][nbr].get("bond_order", 1.0)
                    for nbr in G.neighbors(n)
                    if G.nodes[nbr]["symbol"] not in self.data.metals
                )
                for n in G.nodes()
            }
        else:
            # Incremental update (excluding metal bonds)
            for n in nodes:
                self.valence_cache[n] = sum(
                    G[n][nbr].get("bond_order", 1.0)
                    for nbr in G.neighbors(n)
                    if G.nodes[nbr]["symbol"] not in self.data.metals
                )

    @staticmethod
    def _restore_graph_caches(G: nx.Graph) -> None:
        """Rebuild cached graph properties after modifications."""
        G.graph["_neighbors"] = {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph["_has_H"] = {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()}

    # =========================================================================
    # Full mode: Greedy optimizer
    # =========================================================================

    def _full_valence_optimize(self, G: nx.Graph) -> Dict[str, Any]:
        """Optimize bond orders with formal charge minimization.

        Returns a stats dict containing iterations, improvements,
        initial_score, final_score, and final formal_charges.
        """
        self._log(f"\n{'=' * 80}", 0)
        self._log("FULL VALENCE OPTIMIZATION", 1)
        self._log("=" * 80, 0)

        # --- Precompute / cache graph info ---
        rings = G.graph.get("_rings") or nx.cycle_basis(G)
        G.graph["_rings"] = rings

        neighbor_cache = G.graph.get("_neighbors") or {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph["_neighbors"] = neighbor_cache

        has_H = G.graph.get("_has_H") or {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G[n]) for n in G.nodes()}
        G.graph["_has_H"] = has_H

        # Build valence cache excluding metal bonds
        valence_cache = {
            n: sum(
                G[n][nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(n)
                if G.nodes[nbr]["symbol"] not in self.data.metals
            )
            for n in G.nodes()
        }
        self.valence_cache = valence_cache

        # --- Lock metal bonds ---
        metal_count = 0
        for _i, _j, data in G.edges(data=True):
            if data.get("metal_coord", False):
                data["bond_order"] = 1.0
                metal_count += 1
        if metal_count > 0:
            self._log(f"Locked {metal_count} metal bonds", 1)

        # --- Initial scoring ---
        current_score, formal_charges = self._score_assignment(G, rings)
        initial_score = current_score

        stats: dict[str, Any] = {
            "iterations": 0,
            "improvements": 0,
            "initial_score": initial_score,
            "final_score": initial_score,
            "final_formal_charges": formal_charges,
        }

        self._log(f"Initial score: {initial_score:.2f}", 1)

        stagnation = 0
        self.edge_scores_cache = None
        last_promoted_edge = None
        self._edge_score_map = None

        # --- Optimization loop ---
        for iteration in range(self.config.max_iter):
            stats["iterations"] = iteration + 1
            best_delta = 0.0
            best_edge = None

            self._log(f"\nIteration {iteration + 1}:", 1)

            # --- Precompute top-k candidate edges (with cache) ---
            if self.edge_scores_cache is None:
                self.edge_scores_cache = self._edge_likelihood(G, init=True)
            elif last_promoted_edge is not None:
                self._log(f"Recalculating candidates (promoted {last_promoted_edge})", 2)
                i, j = last_promoted_edge
                self.edge_scores_cache = self._edge_likelihood(G, touch_nodes={i, j})

            # --- Evaluate top-k edges using local delta scoring ---
            for i, j in self.edge_scores_cache:
                bo = G[i][j]["bond_order"]

                # Test both directions
                for change in [+1, -1]:
                    new_bo = bo + change

                    # Skip invalid bond orders
                    if new_bo < 1.0 or new_bo > 3.0:
                        continue

                    # Temporarily apply change
                    G[i][j]["bond_order"] = new_bo
                    valence_cache[i] += change
                    valence_cache[j] += change

                    # Compute full score
                    new_score, _ = self._score_assignment(G, rings)
                    delta = current_score - new_score

                    # Rollback
                    G[i][j]["bond_order"] = bo
                    valence_cache[i] -= change
                    valence_cache[j] -= change

                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j, change)

            # --- Apply best improvement ---
            if best_edge and best_delta > 1e-6:
                i, j, change = best_edge
                G[i][j]["bond_order"] += change
                valence_cache[i] += change
                valence_cache[j] += change
                current_score, _ = self._score_assignment(G, rings)

                stats["improvements"] += 1
                stagnation = 0
                last_promoted_edge = (i, j)
                self._edge_likelihood(G, touch_nodes={i, j})  # update cache

                si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
                edge_label = f"{si}{i}-{sj}{j}"
                action = "promoted" if change > 0 else "demoted"
                self._log(
                    f"✓ {edge_label:<10}  {action}  Δscore = {best_delta:6.2f}  new_score = {current_score:8.2f}",
                    2,
                )

            else:
                stagnation += 1
                last_promoted_edge = None
                self.edge_scores_cache = None  # force full recompute next time

                if stagnation >= MAX_STAGNATION_ITERATIONS:
                    break  # stop if no improvement

        # --- Final scoring ---
        final_formal_charges = self._score_assignment(G, rings)[1]
        stats["final_score"] = current_score
        stats["final_formal_charges"] = final_formal_charges

        self._log("-" * 80, 0)
        self._log(f"Optimized: {stats['improvements']} improvements", 1)
        self._log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self._log("-" * 80, 0)

        return stats

    # =========================================================================
    # Beam search optimizer
    # =========================================================================

    def _beam_search_optimize(self, G: nx.Graph) -> Dict[str, Any]:
        """Memory-efficient beam search with incremental valence cache updates.

        Strategy:
        - Maintain valence cache per hypothesis (small dict)
        - When promoting edge (i,j), update valence for nodes i and j
        - Score calculation uses cached valences
        """
        self._log(f"\n{'=' * 80}", 0)
        self._log(f"BEAM SEARCH OPTIMIZATION (width={self.config.beam_width})", 0)
        self._log("=" * 80, 0)

        # Use cached graph info (don't recompute - preserves metal-free rings)
        rings = G.graph.get("_rings", nx.cycle_basis(G))
        G.graph["_neighbors"] = {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph["_has_H"] = {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()}

        # Lock metal bonds
        metal_count = 0
        for _i, _j, data in G.edges(data=True):
            if data.get("metal_coord", False):
                data["bond_order"] = 1.0
                metal_count += 1
        if metal_count > 0:
            self._log(f"Locked {metal_count} metal bonds", 1)

        # Build initial valence cache excluding metal bonds (shared starting point)
        base_valence_cache = {
            n: sum(
                G[n][nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(n)
                if G.nodes[nbr]["symbol"] not in self.data.metals
            )
            for n in G.nodes()
        }

        # Initial scoring
        self.valence_cache = base_valence_cache.copy()
        current_score, formal_charges = self._score_assignment(G, rings)
        initial_score = current_score

        self._log(f"Initial score: {initial_score:.2f}", 1)

        beam = [(current_score, G, base_valence_cache.copy(), [])]

        stats: dict[str, Any] = {
            "iterations": 0,
            "improvements": 0,
            "initial_score": initial_score,
            "final_score": initial_score,
            "final_formal_charges": formal_charges,
            "beam_explored": 0,
        }

        best_ever_score = current_score
        best_ever_graph = self._copy_graph_state(G)
        best_ever_cache = base_valence_cache.copy()

        for iteration in range(self.config.max_iter):
            stats["iterations"] = iteration + 1
            self._log(f"\nIteration {iteration + 1}:", 1)

            candidates = []

            # Expand each hypothesis in beam
            for _beam_idx, (
                parent_score,
                parent_graph,
                parent_cache,
                parent_history,
            ) in enumerate(beam):
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
                        old_bo = parent_graph[i][j]["bond_order"]
                        new_bo = old_bo + change

                        # Skip invalid bond orders
                        if new_bo < 1.0 or new_bo > 3.0:
                            continue

                        changes_tried += 1

                        G_new = self._copy_graph_state(parent_graph)

                        # Apply change
                        G_new[i][j]["bond_order"] = new_bo

                        # Update the two affected nodes
                        new_cache = parent_cache.copy()
                        new_cache[i] = parent_cache[i] + change
                        new_cache[j] = parent_cache[j] + change

                        # Use new cache for scoring
                        self.valence_cache = new_cache
                        new_score, _ = self._score_assignment(G_new, rings)

                        stats["beam_explored"] += 1

                        # Keep if improvement
                        delta = parent_score - new_score
                        if delta > 0:
                            new_history = [*parent_history, (i, j, change)]
                            candidates.append(
                                (
                                    new_score,
                                    G_new,
                                    new_cache,
                                    (i, j, change),
                                    new_history,
                                )
                            )

            if not candidates:
                self._log("  No improvements found in any beam, stopping", 2)
                break

            # Sort and keep top beam_width
            candidates.sort(key=lambda x: x[0])

            self._log(
                f"  Generated {len(candidates)} candidates, keeping top {min(self.config.beam_width, len(candidates))}",
                2,
            )

            beam = [
                (score, graph, cache, history)
                for score, graph, cache, edge, history in candidates[: self.config.beam_width]
            ]

            # Track best ever
            best_in_beam = beam[0]
            if best_in_beam[0] < best_ever_score:
                improvement = best_ever_score - best_in_beam[0]
                best_ever_score = best_in_beam[0]
                best_ever_graph = self._copy_graph_state(best_in_beam[1])
                best_ever_cache = best_in_beam[2].copy()
                stats["improvements"] += 1

                # Log improvement
                last_edge = best_in_beam[3][-1]
                si = G.nodes[last_edge[0]]["symbol"]
                sj = G.nodes[last_edge[1]]["symbol"]
                edge_label = f"{si}{last_edge[0]}-{sj}{last_edge[1]}"
                self._log(
                    f"  ✓ New best: {edge_label:<10}  Δtotal = {improvement:6.2f}  score = {best_ever_score:8.2f}",
                    2,
                )

        # Apply best solution
        self._log("\nApplying best solution to graph...", 1)
        for i, j, data in best_ever_graph.edges(data=True):
            G[i][j]["bond_order"] = data["bond_order"]

        # Restore caches
        self._restore_graph_caches(G)
        self.valence_cache = best_ever_cache

        # Final scoring
        final_score, final_formal_charges = self._score_assignment(G, rings)
        stats["final_score"] = final_score
        stats["final_formal_charges"] = final_formal_charges

        self._log("-" * 80, 0)
        self._log(
            f"Explored {stats['beam_explored']} states across {stats['iterations']} iterations",
            1,
        )
        self._log(f"Found {stats['improvements']} improvements", 1)
        self._log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self._log("-" * 80, 0)

        return stats

    # =========================================================================
    # Aromatic detection (post-optimization)
    # =========================================================================

    def detect_aromatic_rings(self, G: nx.Graph, kekule: bool = False) -> int:
        """Detect aromatic rings using Hückel rule (4n+2 π electrons).

        Only performed on 5 and 6 member rings with C, N, O, S, P atoms.
        Sets bond orders to 1.5 for aromatic rings where this does not
        introduce valence violations.
        Stores aromatic ring indices in G.graph["_aromatic_rings"].
        """
        self._log(f"\n{'=' * 80}", 0)
        self._log("AROMATIC RING DETECTION (Hückel 4n+2)", 0)
        self._log("=" * 80, 0)

        # Use cached cycles (metal-free) instead of recalculating
        cycles = G.graph.get("_rings", [])
        aromatic_count = 0
        aromatic_rings = 0
        G.graph["_aromatic_rings"] = []

        for ring_idx, cycle in enumerate(cycles):
            if len(cycle) not in (5, 6):
                continue

            ring_atoms = [f"{G.nodes[i]['symbol']}{i}" for i in cycle]

            if not all(G.nodes[i]["symbol"] in self.data.aromatic_atoms for i in cycle):
                non_aromatic = [
                    G.nodes[i]["symbol"] for i in cycle if G.nodes[i]["symbol"] not in self.data.aromatic_atoms
                ]
                self._log(f"✗ Contains non-aromatic atoms: {non_aromatic}", 2)
                continue

            is_planar = self.geometry.check_planarity(cycle, G)
            if not is_planar:
                self._log(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)
                self._log("✗ Not planar, skipping aromaticity check", 2)
                continue

            for i in cycle:
                sym = G.nodes[i]["symbol"]
                if sym == "C":
                    degree = sum(1 for nbr in G.neighbors(i) if G.nodes[nbr]["symbol"] not in self.data.metals)
                    if degree >= 4:
                        self._log(
                            f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}",
                            1,
                        )
                        self._log(
                            f"✗ Contains sp3 carbon {sym}{i} (degree={degree}), skipping aromaticity check",
                            2,
                        )
                        is_planar = False
                        break

            if not is_planar:
                continue

            self._log(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)

            # Count π electrons (simplified)
            pi_electrons = 0
            pi_breakdown = []
            contrib, label = 0, None
            for idx in cycle:
                sym = G.nodes[idx]["symbol"]
                fc = G.nodes[idx].get("formal_charge", 0)
                degree = sum(1 for nbr in G.neighbors(idx) if G.nodes[nbr]["symbol"] not in self.data.metals)

                if sym == "C":
                    contrib = max(0, 1 - fc) if fc > 0 else 1 + abs(fc)
                    label = f"{sym}{idx}:1" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"

                elif sym == "B":
                    contrib = abs(fc) if fc < 0 else 0
                    label = f"{sym}{idx}:0(empty_p)" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"

                elif sym == "N":
                    if degree == 3:
                        contrib = 1 if fc > 0 else 2
                        label = f"{sym}{idx}:2(LP)" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"
                    else:  # degree == 2
                        contrib = 2 if fc < 0 else 1
                        label = f"{sym}{idx}:1" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"

                elif sym in ("O", "S"):
                    contrib = 2
                    label = f"{sym}{idx}:2(LP)" if fc == 0 else f"{sym}{idx}:2(LP,fc={fc:+d})"

                pi_electrons += contrib
                pi_breakdown.append(label)

            self._log(f"π electrons: {pi_electrons} ({', '.join(pi_breakdown)})", 2)

            # Hückel rule: 4n+2 π electrons (n = 0, 1, 2, ...)
            is_aromatic = pi_electrons >= 2 and pi_electrons in (2, 6, 10, 14, 18)

            if is_aromatic:
                n = (pi_electrons - 2) // 4
                self._log(f"✓ AROMATIC (4n+2 rule: n={n})", 2)
                G.graph["_aromatic_rings"].append(cycle)

                if kekule:
                    continue

                # If any ring bond has order > 2 (e.g. triple bond in
                # benzyne), 1.5 cannot represent that bonding and the
                # conversion would invalidate the optimised valence/charge.
                ring_edges = [(cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))]
                high_order = next(
                    ((i, j) for i, j in ring_edges if G.has_edge(i, j) and G.edges[i, j]["bond_order"] > 2.01),
                    None,
                )
                if high_order is not None:
                    i, j = high_order
                    bo = G.edges[i, j]["bond_order"]
                    self._log(
                        f"  ✗ Bond {G.nodes[i]['symbol']}{i}-{G.nodes[j]['symbol']}{j} "
                        f"has order {bo:.1f} > 2, keeping Kekulé structure",
                        2,
                    )
                    continue

                ring_edges = [(cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))]

                bonds_set = 0
                for i, j in ring_edges:
                    if G.has_edge(i, j):
                        old_order = G.edges[i, j]["bond_order"]
                        G.edges[i, j]["bond_order"] = 1.5
                        if abs(old_order - 1.5) > 0.01:
                            bonds_set += 1
                            aromatic_count += 1

                if bonds_set > 0:
                    aromatic_rings += 1
            else:
                self._log("✗ Not aromatic (4n+2 rule violated)", 2)

        self._log(f"\n{'-' * 80}", 0)
        self._log(
            f"SUMMARY: {aromatic_rings} aromatic rings, {aromatic_count} bonds set to 1.5",
            1,
        )
        self._log(f"{'-' * 80}\n", 0)

        return aromatic_count

    # =========================================================================
    # Metal-ligand classification
    # =========================================================================

    def _get_ligand_unit_info(self, G: nx.Graph, metal_idx: int, start_atom: int) -> Tuple[int, str]:
        """Get charge and identity for a ligand unit by following linear chain.

        Returns: (charge, ligand_id)
        Handles: CO, CN⁻, SCN⁻, NO, monatomic ligands
        """
        symbols = [G.nodes[start_atom]["symbol"]]
        charge = G.nodes[start_atom].get("formal_charge", 0)
        current = start_atom
        prev = metal_idx

        # Follow linear chain
        while True:
            neighbors = [n for n in G.neighbors(current) if n != prev and G.nodes[n]["symbol"] not in self.data.metals]
            if len(neighbors) != 1:
                break  # Not linear or branch point
            next_atom = neighbors[0]
            symbols.append(G.nodes[next_atom]["symbol"])
            charge += G.nodes[next_atom].get("formal_charge", 0)
            prev, current = current, next_atom

        # Identify common ligands
        ligand_formula = "".join(symbols)
        if ligand_formula == "CO":
            ligand_id = "CO"
        elif ligand_formula == "CN":
            ligand_id = "CN"
        elif ligand_formula == "NO":
            ligand_id = "NO"
        elif ligand_formula in ("SCN", "NCS"):
            ligand_id = "SCN"
        elif len(symbols) == 1:
            ligand_id = symbols[0]
        else:
            ligand_id = ligand_formula

        return charge, ligand_id

    def classify_metal_ligands(self, G: nx.Graph, formal_charges: Optional[List[int]] = None) -> Dict[str, Any]:
        """Infer ligand types and metal oxidation state from formal charges.

        Handles: monatomic (H⁻, Cl⁻), linear chains (CO, CN⁻), rings (Cp⁻).
        """

        # Helper to get formal charge
        def get_fc(atom_idx):
            if formal_charges is not None:
                return formal_charges[atom_idx]
            return G.nodes[atom_idx].get("formal_charge", 0)

        classification: dict[str, Any] = {
            "dative_bonds": [],
            "ionic_bonds": [],
            "metal_ox_states": {},
        }

        # Get rings (metal-free)
        rings = G.graph.get("_rings", [])

        for metal_idx in G.nodes():
            if G.nodes[metal_idx]["symbol"] not in self.data.metals:
                continue

            ligand_charge_sum = 0
            processed_atoms: set = set()  # Track atoms already assigned to ligands

            # First pass: detect ring-based ligands (Cp⁻)
            metal_bonded_atoms = [n for n in G.neighbors(metal_idx) if G.nodes[n]["symbol"] not in self.data.metals]

            for ring in rings:
                # Check if entire ring bonds to this metal
                ring_set = set(ring)
                bonded_ring_atoms = [a for a in metal_bonded_atoms if a in ring_set]

                if len(bonded_ring_atoms) >= len(ring) / 2:
                    # Sum charges for entire ring
                    ring_charge = sum(get_fc(a) for a in ring)
                    ligand_charge_sum += ring_charge

                    # Mark as processed
                    processed_atoms.update(bonded_ring_atoms)

                    # Use first atom as representative
                    rep_atom = bonded_ring_atoms[0]
                    ligand_type = f"{len(ring)}-ring"

                    if ring_charge == 0:
                        classification["dative_bonds"].append((metal_idx, rep_atom, ligand_type))
                    else:
                        classification["ionic_bonds"].append((metal_idx, rep_atom, ring_charge, ligand_type))

            # Second pass: handle remaining ligands
            for donor_atom in metal_bonded_atoms:
                if donor_atom in processed_atoms:
                    continue

                donor_sym = G.nodes[donor_atom]["symbol"]

                # Check if monatomic (H, halides)
                non_metal_neighbors = [
                    n for n in G.neighbors(donor_atom) if G.nodes[n]["symbol"] not in self.data.metals
                ]

                if len(non_metal_neighbors) == 0:
                    # Monatomic ligand (H⁻, Cl⁻, etc.)
                    ligand_charge = get_fc(donor_atom)
                    ligand_type = f"{donor_sym}"
                else:
                    # Linear chain ligand (CO, CN⁻, etc.)
                    ligand_charge, ligand_type = self._get_ligand_unit_info(G, metal_idx, donor_atom)

                ligand_charge_sum += ligand_charge

                if ligand_charge == 0:
                    classification["dative_bonds"].append((metal_idx, donor_atom, ligand_type))
                else:
                    classification["ionic_bonds"].append((metal_idx, donor_atom, ligand_charge, ligand_type))

            # Infer oxidation state: opposite of ligand charge sum
            ox_state = -ligand_charge_sum
            classification["metal_ox_states"][metal_idx] = ox_state

        return classification
