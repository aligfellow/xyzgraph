"""Numpy array representation of molecular graphs for vectorized scoring.

Pre-extracts graph topology and atom properties into contiguous arrays
so that scoring during bond-order optimisation avoids Python-level
iteration over NetworkX dicts.

The graph topology (nodes, edges, adjacency) is **immutable** after
construction — only ``bond_orders`` (shape ``[E]``) changes during
optimisation.  This means an entire beam hypothesis can be forked with
a single ``bond_orders.copy()`` instead of deep-copying an nx.Graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from .parameters import ScoringWeights

# Re-use the same constants as the main optimizer module.
VALENCE_CHECK_LIMITS: Dict[str, float] = {"C": 4}
VALENCE_CHECK_TOLERANCE = 0.3
SCORING_VALENCE_LIMITS: Dict[str, float] = {"C": 4, "N": 5, "O": 3, "S": 6, "P": 6}
SCORING_VALENCE_TOLERANCE = 0.1
DEFAULT_ELECTRONEGATIVITY = 2.5

# Symbol sets used in scoring (as frozensets for fast lookup)
_NOS_SYMS = frozenset(("N", "O", "S"))
_PROTONATION_HEAVY = frozenset(("N", "O"))


def _compute_formal_charge_vec(
    is_h: np.ndarray,
    valence_electrons: np.ndarray,
    bond_order_sums: np.ndarray,
) -> np.ndarray:
    """Vectorised formal-charge computation.

    Must match ``BondOrderOptimizer._compute_formal_charge_value`` exactly.
    """
    fc = np.zeros(len(is_h), dtype=np.int64)

    # H atoms: fc = V - bond_sum
    h_mask = is_h
    fc[h_mask] = valence_electrons[h_mask] - np.round(bond_order_sums[h_mask]).astype(np.int64)

    # Non-H atoms: target shell = min(8, 2*V) — octet for groups 14-17,
    # sextet for group 13 (B/Al/Ga: V=3 → target=6).  Hypervalent expansion
    # (P/S/Cl with 2*bond_sum > 8) is handled by the max(0, ...) clamp.
    nh = ~h_mask
    B = 2.0 * bond_order_sums[nh]
    target = np.minimum(8.0, 2.0 * valence_electrons[nh])
    L = np.maximum(0.0, target - B)
    fc[nh] = np.round(valence_electrons[nh] - L - B / 2.0).astype(np.int64)

    return fc


@dataclass
class ScoringArrays:
    """Immutable array representation of a molecular graph.

    Constructed once via :meth:`from_graph`, then used for all scoring
    calls during optimisation.  Only ``bond_orders`` is mutable.
    """

    # --- Node arrays (length N) ---
    n_atoms: int
    is_metal: np.ndarray  # bool [N]
    is_h: np.ndarray  # bool [N]
    non_metal: np.ndarray  # bool [N]
    valence_electrons: np.ndarray  # int [N]
    electronegativity: np.ndarray  # float [N]
    vmax: np.ndarray  # float [N], max allowed valence per atom

    # Pre-computed masks for scoring
    vi_mask: np.ndarray  # bool [N] — has_valence_info & non_metal
    is_nos: np.ndarray  # bool [N] — N, O, or S
    is_protonation_heavy: np.ndarray  # bool [N] — N or O

    # Pre-sliced valence data for vi_mask atoms only
    vi_allowed: np.ndarray  # float [N_vi, max_v]
    vi_allowed_mask: np.ndarray  # bool [N_vi, max_v]

    # Pre-computed vlim thresholds (vlim + tolerance, baked in)
    scoring_vlim_thresh: np.ndarray  # float [N]
    check_vlim_thresh: np.ndarray  # float [N]
    has_h_neighbor: np.ndarray  # bool [N]

    # --- Edge arrays (length E) ---
    n_edges: int
    edge_src: np.ndarray  # intp [E]
    edge_dst: np.ndarray  # intp [E]
    bond_orders: np.ndarray  # float [E]
    is_metal_coord: np.ndarray  # bool [E]
    edge_has_metal: np.ndarray  # bool [E]

    # --- CSR adjacency ---
    node_edge_neighbor: np.ndarray  # intp [2*E]
    csr_owners: np.ndarray  # intp [2*E]

    # --- Ring data ---
    ring_edge_indices: List[np.ndarray]
    ring_atoms: List[np.ndarray]
    conjugatable_ring_mask: List[bool]
    ring_exo_eidxs: List[np.ndarray]

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_graph(
        cls,
        G: nx.Graph,
        data,  # MolecularData
        weights: ScoringWeights,
    ) -> ScoringArrays:
        """Build array representation from an nx.Graph."""
        nodes = sorted(G.nodes())
        n = len(nodes)

        # --- Node arrays (one-time extraction from nx.Graph) ---
        sym_list = [G.nodes[i]["symbol"] for i in range(n)]
        symbol_strs = np.array(sym_list, dtype=object)
        is_metal = np.array([s in data.metals for s in sym_list])
        is_h = np.array([s == "H" for s in sym_list])
        is_nos = np.array([s in _NOS_SYMS for s in sym_list])
        is_protonation_heavy = np.array([s in _PROTONATION_HEAVY for s in sym_list])

        valence_electrons = np.array(
            [data.electrons.get(s, 0) for s in sym_list],
            dtype=np.int64,
        )
        electronegativity = np.array(
            [data.electronegativity.get(s, DEFAULT_ELECTRONEGATIVITY) for s in sym_list],
            dtype=np.float64,
        )

        # Allowed valences (padded 2-D array built without per-atom loops)
        raw_vals = [data.valences.get(s, []) for s in sym_list]
        lengths = np.array([len(v) for v in raw_vals], dtype=np.intp)
        max_v = max(int(np.max(lengths)) if n > 0 else 0, 1)

        allowed_valences = np.zeros((n, max_v), dtype=np.float64)
        allowed_valences_mask = np.zeros((n, max_v), dtype=bool)
        has_valence_info = lengths > 0
        vmax_arr = np.full(n, 4.0, dtype=np.float64)

        # Flatten all valences into a single array, then scatter into padded 2-D
        if np.any(has_valence_info):
            flat_vals = np.concatenate([np.asarray(v, dtype=np.float64) for v in raw_vals if v])
            row_idx = np.repeat(np.where(has_valence_info)[0], lengths[has_valence_info])
            col_idx = np.concatenate([np.arange(vlen) for vlen in lengths[has_valence_info]])
            allowed_valences[row_idx, col_idx] = flat_vals
            allowed_valences_mask[row_idx, col_idx] = True
            # vmax: max allowed valence per atom (only where info exists)
            vi_atoms = np.where(has_valence_info)[0]
            vmax_arr[vi_atoms] = np.array([max(raw_vals[i]) for i in vi_atoms], dtype=np.float64)

        # Per-node limits (vectorised via symbol lookup)
        scoring_vlim = np.full(n, np.inf, dtype=np.float64)
        check_vlim = np.full(n, np.inf, dtype=np.float64)
        for sym, lim in SCORING_VALENCE_LIMITS.items():
            mask = symbol_strs == sym
            scoring_vlim[mask] = lim
        for sym, lim in VALENCE_CHECK_LIMITS.items():
            mask = symbol_strs == sym
            check_vlim[mask] = lim

        # --- Edge arrays (extract from nx.Graph into numpy) ---
        raw_edges = [
            (min(ei, ej), max(ei, ej), d.get("bond_order", 1.0), bool(d.get("metal_coord", False)))
            for ei, ej, d in G.edges(data=True)
        ]
        raw_edges.sort()

        n_edges = len(raw_edges)
        if n_edges > 0:
            edge_arr = np.array([(e[0], e[1]) for e in raw_edges], dtype=np.intp)
            edge_src = edge_arr[:, 0]
            edge_dst = edge_arr[:, 1]
            bond_orders_arr = np.array([e[2] for e in raw_edges], dtype=np.float64)
            is_metal_coord = np.array([e[3] for e in raw_edges], dtype=bool)
        else:
            edge_src = np.empty(0, dtype=np.intp)
            edge_dst = np.empty(0, dtype=np.intp)
            bond_orders_arr = np.empty(0, dtype=np.float64)
            is_metal_coord = np.empty(0, dtype=bool)
        edge_has_metal = is_metal[edge_src] | is_metal[edge_dst] if n_edges > 0 else np.empty(0, dtype=bool)

        edge_index_map: Dict[Tuple[int, int], int] = {(int(edge_src[i]), int(edge_dst[i])): i for i in range(n_edges)}

        # --- CSR adjacency (built via numpy argsort, no Python loops) ---
        if n_edges > 0:
            # Each edge (src, dst) contributes two entries: src->dst and dst->src
            owners = np.concatenate([edge_src, edge_dst])  # [2*E] node that "owns" this entry
            neighbors = np.concatenate([edge_dst, edge_src])  # [2*E] the neighbor
            edge_ids = np.concatenate([np.arange(n_edges), np.arange(n_edges)])  # [2*E] edge index

            # Sort by owner node to build CSR order
            sort_order = np.argsort(owners, kind="stable")
            owners_sorted = owners[sort_order]
            node_edge_indices = edge_ids[sort_order]
            node_edge_neighbor = neighbors[sort_order]

            # Build pointer array from sorted owner counts
            node_edge_ptr = np.zeros(n + 1, dtype=np.intp)
            np.add.at(node_edge_ptr[1:], owners_sorted, 1)
            np.cumsum(node_edge_ptr, out=node_edge_ptr)
        else:
            node_edge_ptr = np.zeros(n + 1, dtype=np.intp)
            node_edge_indices = np.empty(0, dtype=np.intp)
            node_edge_neighbor = np.empty(0, dtype=np.intp)

        # CSR owner array (cached — reused in scoring hot path)
        if len(node_edge_neighbor) > 0:
            csr_owners = np.repeat(np.arange(n), np.diff(node_edge_ptr))
            # has_h_neighbor (fully vectorised via CSR scatter)
            nbr_is_h = is_h[node_edge_neighbor]  # [2*E]
            has_h_neighbor = np.zeros(n, dtype=bool)
            np.bitwise_or.at(has_h_neighbor, csr_owners, nbr_is_h)
        else:
            csr_owners = np.empty(0, dtype=np.intp)
            has_h_neighbor = np.zeros(n, dtype=bool)

        # --- Ring data ---
        rings = G.graph.get("_rings", [])
        ring_edge_indices_list: List[np.ndarray] = []
        ring_atoms_list: List[np.ndarray] = []
        in_any_ring = np.zeros(n, dtype=bool)
        conjugatable_ring_mask: List[bool] = []
        ring_exocyclic: List[List[Tuple[int, int, int]]] = []

        for ring in rings:
            ring_arr = np.array(ring, dtype=np.intp)
            ring_atoms_list.append(ring_arr)
            in_any_ring[ring_arr] = True

            r_edge_idx = []
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                key = (min(a, b), max(a, b))
                eidx = edge_index_map.get(key)
                if eidx is not None:
                    r_edge_idx.append(eidx)
            ring_edge_indices_list.append(np.array(r_edge_idx, dtype=np.intp))

            is_conj = len(ring) in (5, 6) and all(sym_list[i] in data.scoring_conjugatable_atoms for i in ring)
            conjugatable_ring_mask.append(is_conj)

            # Pre-compute exocyclic candidate bonds for this ring
            exo: List[Tuple[int, int, int]] = []
            if is_conj:
                ring_set = set(ring)
                for atom_i in ring:
                    ring_sym = sym_list[atom_i]
                    # Only C and N can have relevant exocyclic double bonds
                    if ring_sym not in ("C", "N"):
                        continue
                    s_ptr, e_ptr = node_edge_ptr[atom_i], node_edge_ptr[atom_i + 1]
                    for pos in range(s_ptr, e_ptr):
                        nbr = int(node_edge_neighbor[pos])
                        if nbr in ring_set:
                            continue
                        # in_any_ring not yet fully built, so check directly
                        # (will be set after all rings processed — but for exo
                        # detection we only need the ring_set check above)
                        nbr_sym = sym_list[nbr]
                        if nbr_sym in data.metals:
                            continue
                        # Filter by the specific sym/nbr_sym rules
                        if ring_sym == "C" and nbr_sym == "O":
                            continue
                        if ring_sym == "N" and nbr_sym not in ("C", "P", "S"):
                            continue
                        exo.append((atom_i, int(node_edge_indices[pos]), nbr))
            ring_exocyclic.append(exo)

        # Re-check fused-ring filtering for exocyclic bonds
        # (now that in_any_ring is complete)
        for r_idx in range(len(rings)):
            if not conjugatable_ring_mask[r_idx]:
                continue
            ring_set = {int(a) for a in ring_atoms_list[r_idx]}
            ring_exocyclic[r_idx] = [
                (ai, ei, ni) for ai, ei, ni in ring_exocyclic[r_idx] if not in_any_ring[ni] or ni in ring_set
            ]
            # Remove self-ring refs that slipped through
            ring_exocyclic[r_idx] = [
                (ai, ei, ni) for ai, ei, ni in ring_exocyclic[r_idx] if ni not in ring_set and not in_any_ring[ni]
            ]

        # Pre-build exocyclic edge index arrays (avoids per-call np.array construction)
        ring_exo_eidxs: List[np.ndarray] = []
        for exo_list in ring_exocyclic:
            if exo_list:
                ring_exo_eidxs.append(np.array([eidx for _, eidx, _ in exo_list], dtype=np.intp))
            else:
                ring_exo_eidxs.append(np.empty(0, dtype=np.intp))

        non_metal = ~is_metal
        vi_mask = has_valence_info & non_metal

        return cls(
            n_atoms=n,
            is_metal=is_metal,
            is_h=is_h,
            non_metal=non_metal,
            valence_electrons=valence_electrons,
            electronegativity=electronegativity,
            vmax=vmax_arr,
            vi_mask=vi_mask,
            is_nos=is_nos,
            is_protonation_heavy=is_protonation_heavy,
            vi_allowed=allowed_valences[vi_mask],
            vi_allowed_mask=allowed_valences_mask[vi_mask],
            scoring_vlim_thresh=scoring_vlim + SCORING_VALENCE_TOLERANCE,
            check_vlim_thresh=check_vlim + VALENCE_CHECK_TOLERANCE,
            has_h_neighbor=has_h_neighbor,
            n_edges=n_edges,
            edge_src=edge_src,
            edge_dst=edge_dst,
            bond_orders=bond_orders_arr,
            is_metal_coord=is_metal_coord,
            edge_has_metal=edge_has_metal,
            node_edge_neighbor=node_edge_neighbor,
            csr_owners=csr_owners,
            ring_edge_indices=ring_edge_indices_list,
            ring_atoms=ring_atoms_list,
            conjugatable_ring_mask=conjugatable_ring_mask,
            ring_exo_eidxs=ring_exo_eidxs,
        )

    # =====================================================================
    # Valence sums
    # =====================================================================

    def compute_valence_sums(self, bond_orders: np.ndarray | None = None) -> np.ndarray:
        """Compute valence sum per node, excluding metal bonds.

        Uses np.add.at for scatter-add — no Python loop over atoms.
        """
        if bond_orders is None:
            bond_orders = self.bond_orders

        # Mask: exclude metal-coord edges and edges where either endpoint is metal
        valid = ~self.is_metal_coord & ~self.edge_has_metal
        effective_bo = np.where(valid, bond_orders, 0.0)

        valence_sums = np.zeros(self.n_atoms, dtype=np.float64)
        np.add.at(valence_sums, self.edge_src, effective_bo)
        np.add.at(valence_sums, self.edge_dst, effective_bo)
        return valence_sums

    def update_valence_sums(
        self,
        valence_sums: np.ndarray,
        edge_idx: int,
        old_bo: float,
        new_bo: float,
    ) -> None:
        """Incrementally update valence sums after changing one bond order."""
        if self.is_metal_coord[edge_idx] or self.edge_has_metal[edge_idx]:
            return
        delta = new_bo - old_bo
        valence_sums[self.edge_src[edge_idx]] += delta
        valence_sums[self.edge_dst[edge_idx]] += delta

    # =====================================================================
    # Valence violation check
    # =====================================================================

    def check_valence_violation(self, valence_sums: np.ndarray) -> bool:
        """Return True if any atom exceeds its check_vlim."""
        return bool(np.any(valence_sums > self.check_vlim_thresh))

    # =====================================================================
    # Scoring
    # =====================================================================

    def score(
        self,
        valence_sums: np.ndarray,
        bond_orders: np.ndarray,
        charge: int,
        weights: ScoringWeights,
    ) -> Tuple[float, np.ndarray]:
        """Vectorised scoring — replaces ``_score_assignment``."""
        # Fast reject
        if self.check_valence_violation(valence_sums):
            return 1e9, np.zeros(self.n_atoms, dtype=np.int64)

        # Formal charges (fully vectorised)
        fc = _compute_formal_charge_vec(self.is_h, self.valence_electrons, valence_sums)
        fc[self.is_metal] = 0

        non_metal = self.non_metal
        abs_fc = np.abs(fc)
        fc_sum = float(np.sum(abs_fc[non_metal]))
        n_charged = int(np.count_nonzero(fc[non_metal]))

        # Valence error (pre-sliced arrays, no re-indexing)
        valence_err = 0.0
        vi_mask = self.vi_mask
        if vi_mask.any():
            diffs = np.abs(valence_sums[vi_mask, np.newaxis] - self.vi_allowed)
            diffs[~self.vi_allowed_mask] = np.inf
            valence_err = float(np.sum(np.min(diffs, axis=1) ** 2))

        # Scoring valence limit violations (pre-computed threshold)
        over_limit = non_metal & (valence_sums > self.scoring_vlim_thresh)
        violation = float(np.sum(over_limit)) * weights.violation_weight

        # Electronegativity penalty (vectorised over all non-metal atoms)
        # Operate on full arrays — branchless, avoids np.any guard overhead
        fc_f = fc.astype(np.float64)
        abs_fc_f = np.abs(fc_f)
        en_contrib = np.where(
            fc_f < 0,
            abs_fc_f * (3.5 - self.electronegativity) * 0.5,
            np.where(fc_f > 0, abs_fc_f * (self.electronegativity - 2.5) * 0.5, 0.0),
        )
        en_penalty = float(np.sum(en_contrib[non_metal]))

        # Bonus: protonated N/O/S with positive fc and H neighbor
        nos_pos_h = non_metal & self.is_nos & (fc > 0) & self.has_h_neighbor
        en_penalty -= 1.5 * float(np.sum(nos_pos_h))

        # Protonation penalty (fully vectorised via CSR scatter)
        protonation = 0.0
        h_neigh_neutral = non_metal & self.has_h_neighbor & (fc == 0)
        if np.any(h_neigh_neutral):
            # For each node, count how many non-H positive-fc neighbours it has
            # using the CSR adjacency: scatter a "1" from each positive non-H node
            # to all its neighbours via the CSR structure.
            is_pos_non_h = ~self.is_h & (fc > 0)  # [N]
            nbr_is_pos_non_h = is_pos_non_h[self.node_edge_neighbor]  # [2*E]
            pos_nbr_count = np.zeros(self.n_atoms, dtype=np.float64)
            np.add.at(pos_nbr_count, self.csr_owners, nbr_is_pos_non_h.astype(np.float64))

            # Apply penalty only to qualifying atoms
            qual = h_neigh_neutral & (pos_nbr_count > 0)
            if np.any(qual):
                penalty_vals = np.where(self.is_protonation_heavy[qual], 8.0, 3.0)
                protonation = float(np.sum(penalty_vals * pos_nbr_count[qual]))

        # Conjugation penalty
        conjugation = self._ring_conjugation_penalty(bond_orders, weights)

        # Total
        charge_error = abs(int(np.sum(fc)) - charge)
        total = (
            weights.violation_weight * violation
            + weights.conjugation_weight * conjugation
            + weights.protonation_weight * protonation
            + weights.formal_charge_weight * fc_sum
            + weights.charged_atoms_weight * n_charged
            + weights.charge_error_weight * charge_error
            + weights.electronegativity_weight * en_penalty
            + weights.valence_error_weight * valence_err
        )
        return total, fc

    # =====================================================================
    # Ring conjugation penalty
    # =====================================================================

    def _ring_conjugation_penalty(
        self,
        bond_orders: np.ndarray,
        weights: ScoringWeights,
    ) -> float:
        """Compute conjugation penalty using pre-indexed ring/exocyclic data."""
        penalty = 0.0

        for r_idx, ring_eidx in enumerate(self.ring_edge_indices):
            if not self.conjugatable_ring_mask[r_idx]:
                continue

            ring_size = len(self.ring_atoms[r_idx])

            # Elevated bonds in ring (vectorised)
            elevated_bonds = int(np.sum(bond_orders[ring_eidx] > 1.3))

            # Exocyclic double bonds (pre-built edge index array)
            exo_eidxs = self.ring_exo_eidxs[r_idx]
            if len(exo_eidxs) > 0:
                exocyclic_double = int(np.sum(bond_orders[exo_eidxs] >= 1.8))
            else:
                exocyclic_double = 0

            # Continuous deficit (no `expected - 1` slack): every missing
            # elevated bond up to the Kekulé ideal is penalised, giving the
            # optimiser a gradient all the way to the fully-conjugated state.
            expected = ring_size // 2
            deficit = max(0, expected - elevated_bonds)
            if deficit > 0:
                penalty += deficit * weights.conjugation_deficit_penalty
            if exocyclic_double > 0:
                penalty += exocyclic_double * weights.exocyclic_double_penalty

        return penalty

    # =====================================================================
    # Edge candidate selection
    # =====================================================================

    def eligible_edges_mask(self, bond_orders: np.ndarray) -> np.ndarray:
        """Bool mask [E] of edges eligible for bond-order changes."""
        return ~self.is_metal_coord & (bond_orders < 3.0)

    def top_candidate_edges(
        self,
        bond_orders: np.ndarray,
        valence_sums: np.ndarray,
        valences_data: Dict,
        k: int,
    ) -> np.ndarray:
        """Return indices of top-k candidate edges by valence deficit."""
        eligible = self.eligible_edges_mask(bond_orders)
        eidxs = np.where(eligible)[0]
        if len(eidxs) == 0:
            return np.empty(0, dtype=np.intp)

        # Edge scores = deficit_i + deficit_j (using pre-computed vmax)
        src = self.edge_src[eidxs]
        dst = self.edge_dst[eidxs]
        scores = (self.vmax[src] - valence_sums[src]) + (self.vmax[dst] - valence_sums[dst])

        order = np.argsort(-scores)
        return eidxs[order[:k]]

    # =====================================================================
    # Write back to nx.Graph
    # =====================================================================

    def write_bond_orders_to_graph(self, G: nx.Graph, bond_orders: np.ndarray) -> None:
        """Apply array bond orders back to the nx.Graph edge attributes."""
        for eidx in range(self.n_edges):
            i, j = int(self.edge_src[eidx]), int(self.edge_dst[eidx])
            G[i][j]["bond_order"] = float(bond_orders[eidx])
