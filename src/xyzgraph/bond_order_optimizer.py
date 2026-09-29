"""Bond order optimization for molecular graphs.

Assigns bond orders (1.0/1.5/2.0/3.0) and formal charges to a
connectivity graph produced by BondDetector.

Three optimization modes:
- quick: Fast heuristic valence adjustment (no formal charges)
- greedy: Greedy optimizer with formal charge minimization
- beam: Beam search optimizer (default, best quality)

Also handles:
- π-bond seeding by matching atoms that lack valence
- Post-optimization aromatic detection (Hückel 4n+2 rule)
- Formal charge computation and balancing
- Metal-ligand classification and oxidation state inference
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .data_loader import MolecularData
from .geometry import GeometryCalculator
from .parameters import OptimizerConfig, ScoringWeights
from .scoring_arrays import (
    ScoringArrays,
    aromatic_capable,
    aromatic_systems,
    huckel_aromatic,
    ring_pi_electrons,
)
from .utils import smallest_rings

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = ScoringWeights()
_DEFAULT_CONFIG = OptimizerConfig()

# Valence violation detection (check_valence_violation)
VALENCE_CHECK_LIMITS: Dict[str, float] = {"C": 4}
VALENCE_CHECK_TOLERANCE = 0.3

# Scoring valence limits (max bond order sum before hard penalty)
SCORING_VALENCE_LIMITS: Dict[str, float] = {"C": 4, "N": 4, "O": 3, "S": 6, "P": 6}
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
    def _compute_formal_charge_value(
        symbol: str,
        valence_electrons: int,
        bond_order_sum: float,
        degree: Optional[int] = None,
        allowed: Tuple[int, ...] = (),
        full_shell: bool = False,
    ) -> int:
        """Compute formal charge ``V - L - bond_sum``.

        The lone-pair count ``L`` is the neutral leftover ``V - bond_sum`` when
        that is a real lone pair (non-negative, even) and the atom has every bond
        single or sits at one of its ``allowed`` valences (a sulfoxide S at 4),
        else it completes the preferred shell ``min(8, 2*V)``.  ``degree``
        (non-metal neighbour count) gates the all-single test; ``None`` skips it.
        ``full_shell`` (a metal-bound lone-pair donor, five or more valence
        electrons) forbids a neutral reading short of the shell: an oxo is O2-,
        an imido NR2-, as the ionic convention counts them, and keeps the lone
        pair it gives the metal (a Au-P(=O)R2 phosphinito is P-, not a P+ with
        no pair to give). Groups 13-14 keep an empty p orbital instead: a
        carbene or silylene stays neutral.
        """
        target = min(8, 2 * valence_electrons)
        l_octet = max(0.0, target - 2 * bond_order_sum)
        l_neutral = valence_electrons - bond_order_sum
        all_single = degree is None or abs(bond_order_sum - degree) < 1e-9
        at_allowed = any(abs(bond_order_sum - v) < 1e-9 for v in allowed)
        sub_shell = full_shell and valence_electrons + bond_order_sum < target
        use_neutral = (
            (all_single or at_allowed)
            and not sub_shell
            and l_neutral >= 0
            and abs(l_neutral - round(l_neutral)) < 1e-9
            and round(l_neutral) % 2 == 0
        )
        L = l_neutral if use_neutral else l_octet
        if full_shell:
            L = max(L, 2)
        return round(valence_electrons - L - bond_order_sum)

    # =========================================================================
    # Public API
    # =========================================================================

    def optimize(self, G: nx.Graph, mode: str = "beam", arrays: Optional[ScoringArrays] = None) -> Dict[str, Any]:
        """Optimize bond orders.

        Parameters
        ----------
        G : nx.Graph
            Graph with initial bond_order=1.0 edges.
        mode : str
            Optimizer: "greedy" or "beam".
        arrays : ScoringArrays, optional
            Arrays built for this graph's topology, reused across calls; built here if omitted.

        Returns
        -------
        dict
            Statistics about the optimization run.
        """
        if mode == "greedy":
            return self._full_valence_optimize(G, arrays)
        if mode == "beam":
            return self._beam_search_optimize(G, arrays)
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
            non_metal_nbrs = [nbr for nbr in G.neighbors(node) if G.nodes[nbr]["symbol"] not in self.data.metals]
            bond_sum = sum(G.edges[node, nbr].get("bond_order", 1.0) for nbr in non_metal_nbrs)
            degree = len(non_metal_nbrs)

            donor = V >= 5 and len(non_metal_nbrs) < G.degree(node)  # a lone-pair donor to a metal
            allowed = tuple(sorted(self.data.valences.get(sym, ())))[: 1 if degree == 1 else None]  # see ScoringArrays
            fc = self._compute_formal_charge_value(sym, V, bond_sum, degree, allowed, full_shell=donor)
            formal.append(fc)

        metals = [i for i in G.nodes() if G.nodes[i]["symbol"] in self.data.metals]
        # The structure is scored with every atom at an octet. When that cannot reach the stated charge, the
        # carbons the octet rule read as C- are cations with an empty p orbital (tropylium, trityl): a
        # carbocation is the last resort, never a rival to an onium (an iminium N+=C over N-C+).
        if not metals:
            short = [
                i
                for i in G.nodes()
                if formal[i] == -1
                and self.data.electrons.get(G.nodes[i]["symbol"]) == 4
                and not any(G.nodes[n]["symbol"] in self.data.metals for n in G.neighbors(i))
            ]
            if short and self.charge - sum(formal) >= 2 * len(short):
                for i in short:
                    formal[i] = 1
        residual = self.charge - sum(formal)
        self._log(f"\nNon-metal formal charges sum to {sum(formal):+d} (target: {self.charge:+d})", 2)

        if metals:
            # Ionic convention: the metals carry what the stated total leaves after the ligands.
            if len(metals) == 1:
                formal[metals[0]] = residual
                sym = G.nodes[metals[0]]["symbol"]
                states = sorted({0, *self.data.valences.get(sym, [])})
                if residual not in states:
                    logger.warning(
                        "%s%d reads %+d at charge=%+d, not one of its usual oxidation states %s; check charge=",
                        sym,
                        metals[0],
                        residual,
                        self.charge,
                        states,
                    )
            else:
                self._split_metal_charge(G, formal, metals, residual)
            self._log_metal_coordination(G, formal, metals)
        elif residual:
            electrons = sum(G.nodes[i]["atomic_number"] for i in G.nodes()) - self.charge
            radical = self._radical_site(G, formal, residual) if electrons % 2 and abs(residual) == 1 else None
            if radical is not None:
                # The octet rule read the unpaired electron as a charge; put it back as a radical.
                formal[radical] += residual
                self._log(f"  Radical on {G.nodes[radical]['symbol']}{radical}", 3)
            else:
                logger.warning(
                    "Stated charge %+d, but the bond orders give %+d; formal charges follow the structure",
                    self.charge,
                    sum(formal),
                )
                self._log(f"  Stated charge {self.charge:+d} not reached; formal charges follow the structure", 3)
        else:
            charged = [f"{G.nodes[i]['symbol']}{i}:{q:+d}" for i, q in enumerate(formal) if q]
            self._log(f"  Charged atoms: {', '.join(charged) or 'none'}", 3)

        return formal

    def _split_metal_charge(self, G: nx.Graph, formal: List[int], metals: List[int], total: int) -> None:
        """Share ``total`` between several metals.

        Each takes minus its ligands' charge, then the remainder goes one unit at a time to the metal with the
        most room left: below its valence electrons when adding, above 0 when removing.
        """
        classification = self.classify_metal_ligands(G, formal)
        for m in metals:
            formal[m] = -sum(entry[2] for entry in classification["ionic_bonds"] if entry[0] == m)
        remainder = total - sum(formal[m] for m in metals)
        step = 1 if remainder > 0 else -1
        for _ in range(abs(remainder)):
            cap = {m: self.data.electrons.get(G.nodes[m]["symbol"], 0) for m in metals}
            room = {m: cap[m] - formal[m] if step > 0 else formal[m] for m in metals}
            formal[max(metals, key=lambda m: (room[m], -m))] += step

    def _radical_site(self, G: nx.Graph, formal: List[int], residual: int) -> Optional[int]:
        """Find the atom the octet rule charged although its neutral electron count is odd (an unpaired electron).

        The atom must be able to hold that electron: its bond sum no higher than its highest allowed valence
        (a four-bonded N is an ammonium, never a radical).
        """
        candidates = []
        for i in G.nodes():
            sym = G.nodes[i]["symbol"]
            if sym == "H" or sym in self.data.metals or formal[i] != -residual:
                continue
            nonmetal = [n for n in G.neighbors(i) if G.nodes[n]["symbol"] not in self.data.metals]
            bond_sum = sum(G.edges[i, n].get("bond_order", 1.0) for n in nonmetal)
            if bond_sum > max(self.data.valences.get(sym, [0])):
                continue
            if round(self.data.electrons.get(sym, 0) - bond_sum) % 2:
                candidates.append((-self.data.electronegativity.get(sym, DEFAULT_ELECTRONEGATIVITY), i))
        return min(candidates)[1] if candidates else None

    def _log_metal_coordination(self, G: nx.Graph, formal: List[int], metals: List[int]) -> None:
        classification = self.classify_metal_ligands(G, formal)
        for m in metals:
            self._log(
                f"\n[{m:>3}] {G.nodes[m]['symbol']}  oxidation_state={formal[m]:+d}  coordination={G.degree(m)}", 4
            )
            for _m, donor, chg, ligand_type in sorted(
                (e for e in classification["ionic_bonds"] if e[0] == m), key=lambda e: e[2]
            ):
                self._log(f"  • {ligand_type:>6} ({chg:+d})  [donor: {G.nodes[donor]['symbol']}{donor}]", 4)
            for _m, donor, ligand_type in (e for e in classification["dative_bonds"] if e[0] == m):
                self._log(f"  • {ligand_type:>6} ( 0)  [donor: {G.nodes[donor]['symbol']}{donor}]", 4)

    # =========================================================================
    # π seeding
    # =========================================================================

    def assign_bond_orders(self, G: nx.Graph, mode: str = "beam") -> Dict[str, Any]:
        """Seed and refine bond orders; a single metal is tried at each of its usual oxidation states.

        The ionic convention leaves a complex's ligands short by the metal's oxidation state ``s``:
        they carry ``s - charge`` extra electrons. Each state is seeded with that many electrons
        and refined, and the lowest-scoring structure is kept, so the ligands' own chemistry
        (aromaticity, charges, bond lengths) picks the state rather than a charge-blind seed. The
        matchings and scoring arrays are built once and shared by every state.
        """
        metals = [n for n in G.nodes() if G.nodes[n]["symbol"] in self.data.metals]
        if len(metals) != 1:
            self.seed_pi_bonds(G)
            return self.optimize(G, mode)

        states = sorted({0, *self.data.valences.get(G.nodes[metals[0]]["symbol"], [])})
        slots, need = self._pi_slots(G)
        tables = self._slot_tables(G, slots, need, max(0, states[-1] - self.charge))
        start = {(i, j): d["bond_order"] for i, j, d in G.edges(data=True)}
        arrays = ScoringArrays.from_graph(G, self.data, self.weights)
        best, seen = None, set()
        for state in states:
            nx.set_edge_attributes(G, start, "bond_order")
            self._apply_pairs(G, self._pick_pairs(tables, max(0, state - self.charge)))
            seed = tuple(d["bond_order"] for _, _, d in G.edges(data=True))
            if seed in seen:  # the ligands could not take more electrons: same seed, same result
                continue
            seen.add(seed)
            mark = len(self.log_buffer)
            stats = self.optimize(G, mode, arrays=arrays)
            log = self.log_buffer[mark:]
            del self.log_buffer[mark:]
            orders = {(i, j): d["bond_order"] for i, j, d in G.edges(data=True)}
            if best is None or (stats["final_score"], state) < (best[0]["final_score"], best[1]):
                best = (stats, state, orders, log)
            self._log(f"Oxidation state {state:+d}: score {stats['final_score']:.2f}", 1)
        stats, state, orders, log = best
        nx.set_edge_attributes(G, orders, "bond_order")
        self._log(f"Kept the structure seeded at oxidation state {state:+d}", 1)
        self.log_buffer.extend(log)
        return stats

    def seed_pi_bonds(self, G: nx.Graph, electrons: Optional[int] = None) -> int:
        """Seed π bonds by matching atoms that both lack valence; the beam search refines from there.

        An atom lacks ``v - d`` bonds, ``d`` its non-metal degree and ``v`` its lowest allowed valence
        of at least ``d``, and gets one matching node per missing bond, so two matches on one pair
        make a triple. Optional nodes, which pair only when a neighbour needs them, cover three cases:

        - a central atom of an element with higher allowed valences (S, P, halogens) may expand up to
          its highest valence towards a terminal neighbour (the S=O, P=O and Cl=O of an expanded
          octet); one at an allowed valence with a lone pair left may make one onium bond the same
          way (a nitro N+=O). Never towards a ring or chain atom, and two terminal atoms never expand
          into each other (an eta2-S2 stays S=S);
        - a metal-bound atom of groups 13-14 lacking two or more bonds has its sigma pair on the metal
          and a free p orbital, so its missing bonds are optional: an aryl, vinyl or acyl carbon uses
          one because its neighbour needs it, a carbene none.

        A maximum-cardinality matching leaves the fewest atoms short; among those, bonds between two
        short atoms beat optional ones, then shorter bonds (distance over the vdW sum) win.

        Extra ``electrons`` (a metal-free anion's ``-charge`` by default; a complex's from
        assign_bond_orders) are nodes too: each fills one missing bond with a lone pair, preferring
        electronegative atoms, so the matching must place them and the bond lengths choose where.
        Cations are left to the beam search. Returns the number of π bonds seeded.
        """
        if electrons is None:
            has_metal = any(G.nodes[n]["symbol"] in self.data.metals for n in G)
            electrons = -self.charge if not has_metal and self.charge < 0 else 0
        slots, need = self._pi_slots(G)
        pairs = self._pick_pairs(self._slot_tables(G, slots, need, electrons), electrons)
        bonds = self._apply_pairs(G, pairs)
        filled = sum(slot < need[atom] for pair in pairs for atom, slot in pair if atom != "e")
        self._log(f"\nπ seeding: {bonds} π bonds, {sum(need.values()) - filled} missing bonds left", 1)
        return bonds

    def _pi_slots(self, G: nx.Graph) -> Tuple[nx.Graph, Dict[int, int]]:
        """Build the graph of π slots (see seed_pi_bonds) and each atom's number of needed slots."""
        metals = self.data.metals
        need, spare, loose, degree = {}, {}, {}, {}
        for n in G.nodes():
            sym = G.nodes[n]["symbol"]
            allowed = self.data.valences.get(sym)
            if sym in metals or not allowed:
                continue
            degree[n] = sum(1 for m in G.neighbors(n) if G.nodes[m]["symbol"] not in metals)
            valence = min((v for v in allowed if v >= degree[n]), default=degree[n])
            need[n] = valence - degree[n]
            # Extra bonds a central atom can make to a terminal one: up to its highest valence (S, P,
            # halogens), or, once its valence is full, one more through a lone pair as an onium (the
            # N+=O of a nitro group; a nitroso N=O is still short and makes its bond normally).
            lone_pair_left = self.data.electrons.get(sym, 0) - valence >= 2
            onium = 1 if degree[n] in allowed and lone_pair_left else 0
            spare[n] = max(allowed) - valence if len(allowed) > 1 else onium
            if self.data.electrons.get(sym, 0) <= 4 and need[n] >= 2 and degree[n] < G.degree(n):
                loose[n], need[n] = need[n], 0  # sigma pair on the metal, p orbital free: bonds optional

        slots = nx.Graph()
        slots.add_nodes_from((n, a) for n, k in need.items() for a in range(k))  # a lone donor is a piece too
        bonus = G.number_of_nodes()  # outweighs any sum of closeness: needed pairs come first
        for i, j, data in G.edges(data=True):
            if i not in need or j not in need:
                continue
            vdw = self.data.vdw.get(G.nodes[i]["symbol"], 2.0) + self.data.vdw.get(G.nodes[j]["symbol"], 2.0)
            closeness = 1.0 - data["distance"] / vdw
            for u, v in ((i, j), (j, i)):
                terminal_spare = spare[u] if degree[u] > 1 and degree[v] == 1 else 0
                optional = loose.get(u, 0) + terminal_spare
                partner = need[v] + (loose[v] if u in loose and v in loose else 0)  # an eta2-alkyne pairs
                for a in range(need[u] + optional):
                    for b in range(partner):
                        needed = a < need[u] and b < need[v]
                        slots.add_edge((u, a), (v, b), weight=closeness + bonus * needed)
        return slots, need

    def _slot_tables(self, G: nx.Graph, slots: nx.Graph, need: Dict[int, int], electrons: int) -> List[List]:
        """Match each connected piece of the slot graph with 0..electrons extra electrons.

        The pieces share nothing but the electrons, so their matchings are independent: each row is
        (pairs, weight, matching) for that many electrons on that piece, and _pick_pairs shares them out.
        In a complex the electrons are the metal's, given to its donor atoms, so only metal-bound
        atoms take them; without a metal any atom can.
        """
        metals = self.data.metals
        has_metal = any(G.nodes[n]["symbol"] in metals for n in G)
        takes = {n for n in need if not has_metal or any(G.nodes[m]["symbol"] in metals for m in G.neighbors(n))}
        tables = []
        for piece in sorted(nx.connected_components(slots), key=min):
            sub = slots.subgraph(piece)
            fillable = [(n, a) for n, a in piece if a < need[n] and n in takes]
            rows = []
            for e in range(min(electrons, len(fillable)) + 1):
                H = nx.Graph(sub)
                for k in range(e):
                    for n, a in fillable:
                        en = self.data.electronegativity.get(G.nodes[n]["symbol"], DEFAULT_ELECTRONEGATIVITY)
                        H.add_edge(("e", k), (n, a), weight=0.1 * en)
                matched = nx.max_weight_matching(H, maxcardinality=True) if H.number_of_edges() else set()
                rows.append((len(matched), sum(H.edges[u, v]["weight"] for u, v in matched), matched))
            tables.append(rows)
        return tables

    @staticmethod
    def _pick_pairs(tables: List[List], electrons: int) -> List:
        """Share ``electrons`` among the pieces for the most pairs, then the most weight (exact knapsack)."""
        best = {0: ((0, 0.0), [])}  # electrons used -> ((pairs, weight), rows chosen)
        for rows in tables:
            grown: Dict[int, Tuple] = {}
            for used, (key, chosen) in best.items():
                for e, (count, weight, matched) in enumerate(rows):
                    if used + e > electrons:
                        break
                    candidate = ((key[0] + count, key[1] + weight), [*chosen, matched])
                    if used + e not in grown or candidate[0] > grown[used + e][0]:
                        grown[used + e] = candidate
            best = grown
        _, chosen = max(best.values(), key=lambda entry: entry[0])
        return [pair for matched in chosen for pair in matched]

    @staticmethod
    def _apply_pairs(G: nx.Graph, pairs: List) -> int:
        """Raise the bond order of every matched atom pair (electron pairs are lone pairs); return the count."""
        bonds = [(u[0], v[0]) for u, v in pairs if "e" not in (u[0], v[0])]
        for i, j in bonds:
            G.edges[i, j]["bond_order"] = min(3.0, G.edges[i, j]["bond_order"] + 1.0)
        return len(bonds)

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
    # Full mode: Greedy optimizer
    # =========================================================================

    def _full_valence_optimize(self, G: nx.Graph, arrays: Optional[ScoringArrays] = None) -> Dict[str, Any]:
        """Greedy optimizer using vectorised numpy scoring.

        Returns a stats dict containing iterations, improvements,
        initial_score, final_score, and final formal_charges.
        """
        self._log(f"\n{'=' * 80}", 0)
        self._log("FULL VALENCE OPTIMIZATION", 1)
        self._log("=" * 80, 0)

        if "_rings" not in G.graph:
            G.graph["_rings"] = smallest_rings(G)

        # Lock metal bonds
        metal_count = 0
        for _i, _j, data in G.edges(data=True):
            if data.get("metal_coord", False):
                data["bond_order"] = 1.0
                metal_count += 1
        if metal_count > 0:
            self._log(f"Locked {metal_count} metal bonds", 1)

        # Build array representation
        sa = arrays if arrays is not None else ScoringArrays.from_graph(G, self.data, self.weights)
        bo = sa.read_bond_orders(G)
        vs = sa.compute_valence_sums(bo)

        # Initial scoring
        current_score, formal_charges = sa.score(vs, bo, self.charge, self.weights)
        initial_score = current_score

        stats: dict[str, Any] = {
            "iterations": 0,
            "improvements": 0,
            "initial_score": initial_score,
            "final_score": initial_score,
            "final_formal_charges": formal_charges.tolist()
            if hasattr(formal_charges, "tolist")
            else list(formal_charges),
        }

        self._log(f"Initial score: {initial_score:.2f}", 1)

        stagnation = 0

        for iteration in range(self.config.max_iter):
            stats["iterations"] = iteration + 1
            best_delta = 0.0
            best_move: Optional[Tuple[int, float]] = None  # (edge_idx, change)

            self._log(f"\nIteration {iteration + 1}:", 1)

            # Get top candidate edges (vectorised)
            top_eidxs = sa.top_candidate_edges(bo, vs, self.config.edge_per_iter)

            for raw_eidx in top_eidxs:
                eidx = int(raw_eidx)
                old_bo = bo[eidx]

                # +2 (single->triple) only where a double can't satisfy either
                # endpoint (both deficient by >= 2); else always rejected, so skip.
                ei, ej = int(sa.edge_src[eidx]), int(sa.edge_dst[eidx])
                changes = (
                    (+1.0, -1.0, +2.0)
                    if (old_bo <= 1.0 and sa.vmax[ei] - vs[ei] >= 2.0 - 1e-9 and sa.vmax[ej] - vs[ej] >= 2.0 - 1e-9)
                    else (+1.0, -1.0)
                )
                for change in changes:
                    new_bo_val = old_bo + change
                    if new_bo_val < 1.0 or new_bo_val > 3.0:
                        continue

                    # Temporarily apply
                    bo[eidx] = new_bo_val
                    sa.update_valence_sums(vs, eidx, old_bo, new_bo_val)

                    new_score, _ = sa.score(vs, bo, self.charge, self.weights)
                    delta = current_score - new_score

                    # Rollback
                    sa.update_valence_sums(vs, eidx, new_bo_val, old_bo)
                    bo[eidx] = old_bo

                    if delta > best_delta:
                        best_delta = delta
                        best_move = (eidx, change)

            if best_move and best_delta > 1e-6:
                best_eidx, change = best_move
                old_bo = bo[best_eidx]
                new_bo_val = old_bo + change
                bo[best_eidx] = new_bo_val
                sa.update_valence_sums(vs, best_eidx, old_bo, new_bo_val)
                current_score, _ = sa.score(vs, bo, self.charge, self.weights)

                stats["improvements"] += 1
                stagnation = 0

                i, j = int(sa.edge_src[best_eidx]), int(sa.edge_dst[best_eidx])
                si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
                edge_label = f"{si}{i}-{sj}{j}"
                action = "promoted" if change > 0 else "demoted"
                self._log(
                    f"✓ {edge_label:<10}  {action}  Δscore = {best_delta:6.2f}  new_score = {current_score:8.2f}",
                    2,
                )
            else:
                stagnation += 1
                if stagnation >= MAX_STAGNATION_ITERATIONS:
                    break

        # Apply to graph
        sa.write_bond_orders_to_graph(G, bo)

        # Final scoring
        final_score, final_fc = sa.score(vs, bo, self.charge, self.weights)
        stats["final_score"] = final_score
        stats["final_formal_charges"] = final_fc.tolist() if hasattr(final_fc, "tolist") else list(final_fc)

        self._log("-" * 80, 0)
        self._log(f"Optimized: {stats['improvements']} improvements", 1)
        self._log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self._log("-" * 80, 0)

        return stats

    # =========================================================================
    # Charge-budget escape (alternating shift path)
    # =========================================================================

    def _find_kekule_shift_path(self, sa, bond_orders, valence_sums):
        """Find an alternating-BO chain (1,2,1,2,...,1) between two deficient atoms.

        Flipping every bond saturates both endpoints while leaving bond_sum
        unchanged at each interior atom (it loses 1 on one side, gains 1 on the
        other), so it escapes traps no single-edge move can.  Returns the
        edge-index list, or None.
        """
        deficit_mask = (~sa.is_metal) & (~sa.is_h) & (valence_sums < sa.vmax - 0.01)
        deficit_atoms = [int(a) for a in np.where(deficit_mask)[0]]
        if len(deficit_atoms) < 2:
            return None
        deficit_set = set(deficit_atoms)

        adj = self._atom_adjacency(sa)

        for start in deficit_atoms:
            # BFS state = (atom, expected_BO_for_next_edge); the first edge
            # must be single so flipping it elevates and saturates ``start``.
            visited = {(start, 1.0): None}
            q = deque([(start, 1.0)])
            found = None
            while q and found is None:
                atom, expect = q.popleft()
                for nb, eidx in adj[atom]:
                    bo = bond_orders[eidx]
                    if abs(bo - expect) > 0.01:
                        continue  # wrong BO for alternation
                    next_expect = 2.0 if expect < 1.5 else 1.0
                    state = (nb, next_expect)
                    if state in visited:
                        continue
                    visited[state] = (atom, expect, eidx)
                    # Endpoint: another deficit atom reached via a single bond
                    if nb in deficit_set and nb != start and expect < 1.5:
                        found = state
                        break
                    q.append(state)
            if found is None:
                continue

            path = []
            cur = found
            entry = visited[cur]
            while entry is not None:
                patom, pexpect, eidx = entry
                path.append(int(eidx))
                cur = (patom, pexpect)
                entry = visited[cur]
            path.reverse()
            return path

        return None

    @staticmethod
    def _atom_adjacency(sa):
        """Return per-atom list of (neighbour, edge_index) pairs, cached on sa."""
        adj = getattr(sa, "_atom_adj_cache", None)
        if adj is not None:
            return adj
        n = sa.n_atoms
        adj = [[] for _ in range(n)]
        for eidx in range(sa.n_edges):
            i, j = int(sa.edge_src[eidx]), int(sa.edge_dst[eidx])
            adj[i].append((j, eidx))
            adj[j].append((i, eidx))
        sa._atom_adj_cache = adj
        return adj

    # =========================================================================
    # Beam search optimizer
    # =========================================================================

    def _beam_search_optimize(self, G: nx.Graph, arrays: Optional[ScoringArrays] = None) -> Dict[str, Any]:
        """Beam search using vectorised numpy scoring.

        Each beam hypothesis is a (bond_orders, valence_sums) pair of
        numpy arrays. A move is applied in place, scored and rolled back;
        only an improving move is copied into a new hypothesis.
        """
        self._log(f"\n{'=' * 80}", 0)
        self._log(f"BEAM SEARCH OPTIMIZATION (width={self.config.beam_width})", 0)
        self._log("=" * 80, 0)

        if "_rings" not in G.graph:
            G.graph["_rings"] = smallest_rings(G)

        # Lock metal bonds
        metal_count = 0
        for _i, _j, data in G.edges(data=True):
            if data.get("metal_coord", False):
                data["bond_order"] = 1.0
                metal_count += 1
        if metal_count > 0:
            self._log(f"Locked {metal_count} metal bonds", 1)

        # Build array representation (topology is immutable after this)
        sa = arrays if arrays is not None else ScoringArrays.from_graph(G, self.data, self.weights)
        base_bo = sa.read_bond_orders(G)
        base_vs = sa.compute_valence_sums(base_bo)

        # Initial scoring
        current_score, formal_charges = sa.score(base_vs, base_bo, self.charge, self.weights)
        initial_score = current_score
        self._log(f"Initial score: {initial_score:.2f}", 1)

        # Beam: list of (score, bond_orders, valence_sums, history)
        beam: list = [(current_score, base_bo.copy(), base_vs.copy(), [])]

        stats: dict[str, Any] = {
            "iterations": 0,
            "improvements": 0,
            "initial_score": initial_score,
            "final_score": initial_score,
            "final_formal_charges": formal_charges.tolist()
            if hasattr(formal_charges, "tolist")
            else list(formal_charges),
            "beam_explored": 0,
        }

        best_ever_score = current_score
        best_ever_bo = base_bo.copy()
        best_ever_vs = base_vs.copy()

        # Cache scores by bond-order bytes; beam children recur across iterations.
        score_cache: Dict[bytes, float] = {}
        stats["score_cache_hits"] = 0

        for iteration in range(self.config.max_iter):
            stats["iterations"] = iteration + 1
            self._log(f"\nIteration {iteration + 1}:", 1)

            candidates = []

            for _beam_idx, (parent_score, parent_bo, parent_vs, parent_history) in enumerate(beam):
                # Get top candidate edges (vectorised)
                top_eidxs = sa.top_candidate_edges(parent_bo, parent_vs, self.config.edge_per_iter)

                for raw_eidx in top_eidxs:
                    eidx = int(raw_eidx)
                    old_bo = parent_bo[eidx]
                    i = int(sa.edge_src[eidx])
                    j = int(sa.edge_dst[eidx])

                    # +2 (single->triple) only where a double can't satisfy either
                    # endpoint (both deficient by >= 2).
                    changes = (
                        (+1.0, -1.0, +2.0)
                        if (
                            old_bo <= 1.0
                            and sa.vmax[i] - parent_vs[i] >= 2.0 - 1e-9
                            and sa.vmax[j] - parent_vs[j] >= 2.0 - 1e-9
                        )
                        else (+1.0, -1.0)
                    )
                    for change in changes:
                        new_bo = old_bo + change
                        if new_bo < 1.0 or new_bo > 3.0:
                            continue

                        parent_bo[eidx] = new_bo
                        sa.update_valence_sums(parent_vs, eidx, old_bo, new_bo)

                        cand_key = parent_bo.tobytes()
                        new_score = score_cache.get(cand_key)
                        if new_score is None:
                            new_score, _ = sa.score(parent_vs, parent_bo, self.charge, self.weights)
                            score_cache[cand_key] = new_score
                            stats["beam_explored"] += 1
                        else:
                            stats["score_cache_hits"] += 1

                        if new_score < parent_score:
                            move = (i, j, change)
                            candidates.append(
                                (new_score, parent_bo.copy(), parent_vs.copy(), move, [*parent_history, move])
                            )

                        sa.update_valence_sums(parent_vs, eidx, new_bo, old_bo)
                        parent_bo[eidx] = old_bo

            if not candidates:
                # A met charge means converged; a missed one needs the shift path
                # (a charge-forcing valence no single-edge move can fix).
                for parent_score, parent_bo, parent_vs, parent_history in beam:
                    if not any(sa.charge_terms(sa.formal_charges(parent_vs), self.charge)):
                        continue
                    path = self._find_kekule_shift_path(sa, parent_bo, parent_vs)
                    if not path:
                        continue
                    cand_bo = parent_bo.copy()
                    cand_vs = parent_vs.copy()
                    for eidx in path:
                        old = cand_bo[eidx]
                        new = 1.0 if old > 1.5 else 2.0
                        cand_bo[eidx] = new
                        sa.update_valence_sums(cand_vs, eidx, old, new)
                    new_score, _ = sa.score(cand_vs, cand_bo, self.charge, self.weights)
                    stats["beam_explored"] += 1
                    if new_score < parent_score:
                        ends = (int(sa.edge_src[path[0]]), int(sa.edge_dst[path[-1]]), "shift")
                        candidates.append((new_score, cand_bo, cand_vs, ends, [*parent_history, ends]))

                if not candidates:
                    self._log("  No single-edge improvement found, stopping", 2)
                    break
                self._log(f"  Stall escape: {len(candidates)} charge-fixing shift path(s)", 2)

            # Sort and keep top beam_width
            candidates.sort(key=lambda x: x[0])
            self._log(
                f"  Generated {len(candidates)} candidates, keeping top {min(self.config.beam_width, len(candidates))}",
                2,
            )

            beam = [(score, bo, vs, history) for score, bo, vs, _edge, history in candidates[: self.config.beam_width]]

            # Track best ever
            best_in_beam = beam[0]
            if best_in_beam[0] < best_ever_score:
                improvement = best_ever_score - best_in_beam[0]
                best_ever_score = best_in_beam[0]
                best_ever_bo = best_in_beam[1].copy()
                best_ever_vs = best_in_beam[2].copy()
                stats["improvements"] += 1

                last_edge = best_in_beam[3][-1]
                si = G.nodes[last_edge[0]]["symbol"]
                sj = G.nodes[last_edge[1]]["symbol"]
                edge_label = f"{si}{last_edge[0]}-{sj}{last_edge[1]}"
                self._log(
                    f"  ✓ New best: {edge_label:<10}  Δtotal = {improvement:6.2f}  score = {best_ever_score:8.2f}",
                    2,
                )

        # Apply best solution back to nx.Graph
        self._log("\nApplying best solution to graph...", 1)
        sa.write_bond_orders_to_graph(G, best_ever_bo)

        # Final scoring (use array scorer for consistency)
        final_score, final_fc = sa.score(best_ever_vs, best_ever_bo, self.charge, self.weights)
        stats["final_score"] = final_score
        stats["final_formal_charges"] = final_fc.tolist() if hasattr(final_fc, "tolist") else list(final_fc)

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
        """Mark Hückel-aromatic rings and set their bonds to 1.5; return the number of bonds changed.

        The rule is the scorer's (``aromatic_capable``, ``aromatic_systems``, ``ring_pi_electrons``,
        ``huckel_aromatic``): each atom's ring π electrons are read once from the Kekulé structure,
        and a ring is aromatic if it, or it fused with one neighbour, holds 4n+2 π electrons. A ring
        with a bond above 2 (a benzyne) keeps its Kekulé orders; ``kekule=True`` records aromatic
        rings without changing any bond. Stores aromatic rings in ``G.graph["_aromatic_rings"]``.
        """
        self._log(f"\n{'=' * 80}", 0)
        self._log("AROMATIC RING DETECTION (Hückel 4n+2)", 0)
        self._log("=" * 80, 0)

        cycles = [c for c in G.graph.get("_rings", []) if aromatic_capable(G, c, self.data)]
        pi = self._ring_pi_electrons(G)
        aromatic = set()
        for members, atoms in aromatic_systems(cycles):
            counts = np.array([pi[i] for i in atoms])
            ok = huckel_aromatic(counts, self._ring_system_charge(G, atoms))
            breakdown = ", ".join(f"{G.nodes[i]['symbol']}{i}:{pi[i]}" for i in atoms)
            self._log(f"\n{'Fused rings' if len(members) > 1 else 'Ring'} {atoms}: π = {counts.sum()} ({breakdown})", 1)
            self._log("✓ AROMATIC" if ok else "✗ Not aromatic (not 4n+2, or cross-conjugated)", 2)
            if ok:
                aromatic.update(members)

        G.graph["_aromatic_rings"] = [cycles[r] for r in sorted(aromatic)]
        changed = 0
        if not kekule:
            for cycle in G.graph["_aromatic_rings"]:
                changed += self._set_aromatic(G, [(cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))])

        self._log(f"\n{'-' * 80}", 0)
        self._log(f"SUMMARY: {len(aromatic)} aromatic rings, {changed} bonds set to 1.5", 1)
        self._log(f"{'-' * 80}\n", 0)
        return changed

    def _set_aromatic(self, G: nx.Graph, edges) -> int:
        """Set ``edges`` to 1.5 unless one is above a double bond; return how many changed."""
        if any(G.edges[i, j]["bond_order"] > 2.01 for i, j in edges if G.has_edge(i, j)):
            self._log("  ✗ A bond above 2, keeping the Kekulé structure", 2)
            return 0
        changed = 0
        for i, j in edges:
            if G.has_edge(i, j):
                changed += abs(G.edges[i, j]["bond_order"] - 1.5) > 0.01
                G.edges[i, j]["bond_order"] = 1.5
        return changed

    def _ring_pi_electrons(self, G: nx.Graph) -> Dict[int, int]:
        """Ring π electrons of every ring atom under the current (Kekulé) bond orders."""
        metals = self.data.metals
        ring_bonds = self._ring_bond_set(G)
        out: Dict[int, int] = {}
        for i in {a for cycle in G.graph.get("_rings", []) for a in cycle}:
            nbrs = [nb for nb in G.neighbors(i) if G.nodes[nb]["symbol"] not in metals]
            pi_bonds = [frozenset((i, nb)) in ring_bonds for nb in nbrs if G.edges[i, nb]["bond_order"] > 1.3]
            out[i] = int(
                ring_pi_electrons(
                    self.data.electrons.get(G.nodes[i]["symbol"], 0),
                    G.nodes[i].get("formal_charge", 0),
                    sum(G.edges[i, nb]["bond_order"] for nb in nbrs),
                    len(nbrs),
                    any(pi_bonds),
                    not all(pi_bonds),
                )
            )
        return out

    @staticmethod
    def _ring_bond_set(G: nx.Graph) -> set:
        """Bonds that lie on a ring: consecutive pairs of every cached cycle."""
        bonds: set = set()
        for cyc in G.graph.get("_rings", []):
            for k in range(len(cyc)):
                bonds.add(frozenset((cyc[k], cyc[(k + 1) % len(cyc)])))
        return bonds

    def _ring_system_charge(self, G: nx.Graph, atoms: List[int]) -> int:
        """Formal charge of a ring plus its direct non-metal substituents.

        Charges conjugated into a ring often sit one bond outside it (olate
        oxygens in croconate/squarate), so the ring atoms alone misrepresent
        the charge available to the ring π system. A metal's charge is its
        oxidation state, not the ring's.
        """
        in_rings = {i for r in G.graph.get("_rings", []) for i in r}
        substituents = {
            nb
            for i in atoms
            for nb in G.neighbors(i)
            if nb not in in_rings and G.nodes[nb]["symbol"] not in self.data.metals
        }
        return sum(G.nodes[i].get("formal_charge", 0) for i in (*atoms, *substituents))

    def _get_ligand_unit_info(self, G: nx.Graph, metal_idx: int, start_atom: int, get_fc) -> Tuple[int, str]:
        """Get charge and identity for a ligand unit by following linear chain.

        Returns: (charge, ligand_id)
        Handles: CO, CN⁻, SCN⁻, NO, monatomic ligands.  ``get_fc`` reads the
        computed charge (not yet written to the graph nodes).
        """
        symbols = [G.nodes[start_atom]["symbol"]]
        charge = get_fc(start_atom)
        current = start_atom
        prev = metal_idx

        # Follow linear chain
        while True:
            neighbors = [n for n in G.neighbors(current) if n != prev and G.nodes[n]["symbol"] not in self.data.metals]
            if len(neighbors) != 1:
                break  # Not linear or branch point
            next_atom = neighbors[0]
            symbols.append(G.nodes[next_atom]["symbol"])
            charge += get_fc(next_atom)
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
                    ligand_charge, ligand_type = self._get_ligand_unit_info(G, metal_idx, donor_atom, get_fc)

                if ligand_charge == 0:
                    classification["dative_bonds"].append((metal_idx, donor_atom, ligand_type))
                else:
                    classification["ionic_bonds"].append((metal_idx, donor_atom, ligand_charge, ligand_type))

            # Ionic convention: the oxidation state is the metal's formal charge, which conserves the total
            # (compute_formal_charges); the ligand charges are reported per bond above.
            classification["metal_ox_states"][metal_idx] = get_fc(metal_idx)

        return classification
