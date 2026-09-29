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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .geometry import GeometryCalculator

if TYPE_CHECKING:
    from .parameters import ScoringWeights

VALENCE_CHECK_LIMITS: Dict[str, float] = {"C": 4}
VALENCE_CHECK_TOLERANCE = 0.3
SCORING_VALENCE_LIMITS: Dict[str, float] = {"C": 4, "N": 4, "O": 3, "S": 6, "P": 6}
SCORING_VALENCE_TOLERANCE = 0.1
DEFAULT_ELECTRONEGATIVITY = 2.5
# A single bond is about 0.40 of the vdW sum (C-C: 1.54 over 2 x 1.91 A); a π bond shortens it.
SINGLE_BOND_VDW_FRACTION = 0.40
# Neighbouring p orbitals twisted past this (degrees) have lost a quarter of their overlap (cos^2).
MAX_PI_TWIST = 30.0
# A sigma bond to a metal lies along the hybrid a carbon's other bonds leave free (projection 1); a p
# orbital is perpendicular to it (0). Halfway between the two.
SIGMA_MIN_HYBRID = 0.5

# Symbol sets used in scoring (as frozensets for fast lookup)
_NOS_SYMS = frozenset(("N", "O", "S"))
_PROTONATION_HEAVY = frozenset(("N", "O"))


def ring_pi_electrons(valence_electrons, formal_charge, bond_order_sum, degree, ring_pi, exo_pi):
    """π electrons each atom gives a ring under a Lewis structure, the same rule for every element.

    An atom in a ring π bond gives one; one whose π bond leaves the ring gives none. Otherwise its
    ``V - fc - bond_order_sum`` unshared electrons fill the sp2 hybrids its ``degree`` sigma bonds
    leave free first, and the rest (up to a pair) is its p orbital: two for a pyrrole N, a furan O
    or a Cp- carbon, none for a borane B or a carbene C. Bond sums and degrees exclude metals.
    """
    lone = np.asarray(valence_electrons) - formal_charge - bond_order_sum
    p_orbital = np.clip(lone - 2 * np.maximum(0, 3 - np.asarray(degree)), 0, 2)
    return np.rint(np.where(ring_pi, 1, np.where(exo_pi, 0, p_orbital))).astype(np.int64)


def huckel_aromatic(pi: np.ndarray, system_charge: int) -> bool:
    """Test for 4n+2 π electrons in a ring that is not cross-conjugated.

    An atom with no π electron joins the ring only through a betaine form, and a ring supports one
    such form, so two make it cross-conjugated (a quinone) unless the ring carries a negative charge,
    whose surplus electrons fund more (croconate's olates, counted in ``system_charge`` with the
    ring's direct substituents).
    """
    return int(pi.sum()) % 4 == 2 and (system_charge < 0 or int(np.count_nonzero(pi == 0)) < 2)


def aromatic_systems(rings: List[List[int]]) -> List[Tuple[Tuple[int, ...], List[int]]]:
    """Each capable ring alone, each bicycle, and each larger fused system, as (ring indices, atoms).

    A ring is aromatic if any system holding it has 4n+2 π electrons. Counted ring by ring, a bridgehead
    lone pair (indolizine's N, indenyl's C-) lands in both rings, so a bicycle (two rings fused on one
    bond, every atom on its perimeter) is judged whole: indolizine's 10π. A macrocycle (a ring beyond
    seven atoms) carries its fused rings in one π system, judged whole too: a porphyrin's 26π, of which
    a pyrrole and the macrocycle (sharing two bonds) are no bicycle. Fused small rings stay local.
    """
    sets = [set(r) for r in rings]
    fused = nx.Graph()
    fused.add_nodes_from(range(len(rings)))
    fused.add_edges_from(
        (a, b) for a in range(len(rings)) for b in range(a + 1, len(rings)) if len(sets[a] & sets[b]) >= 2
    )
    systems: List[Tuple[Tuple[int, ...], List[int]]] = [((r,), list(ring)) for r, ring in enumerate(rings)]
    systems += [((a, b), sorted(sets[a] | sets[b])) for a, b in fused.edges() if len(sets[a] & sets[b]) == 2]
    for component in nx.connected_components(fused):
        if len(component) > 2 and any(len(rings[r]) > 7 for r in component):
            systems.append((tuple(sorted(component)), sorted(set().union(*(sets[r] for r in component)))))
    return systems


def aromatic_capable(G: nx.Graph, ring, data) -> bool:
    """Test whether a ring of five or more atoms can be aromatic: every atom can hold a ring p orbital.

    Aromatic elements with three or fewer non-metal neighbours, whose neighbouring p orbitals stay
    within MAX_PI_TWIST of parallel: fixed by the geometry, so checked once.
    """
    if len(ring) < 5:
        return False
    metals = data.metals
    for i in ring:
        if G.nodes[i]["symbol"] not in data.aromatic_atoms:
            return False
        if sum(1 for nb in G.neighbors(i) if G.nodes[nb]["symbol"] not in metals) > 3:
            return False
    return GeometryCalculator.max_ring_twist(list(ring), G) <= MAX_PI_TWIST


def sigma_bound(G: nx.Graph, i: int, data) -> bool:
    """Test whether a group-14 atom is sigma-bonded to a metal, rather than through its p orbital.

    A sigma donor keeps the pair it gives the metal, so three bonds to non-metals at most (an aryl C-,
    not a neutral C with four and M-C). The bond takes the hybrid the atom's other bonds leave free,
    along minus the sum of their unit vectors (length 1 for an sp3, sp2 or sp atom one bond short;
    0 for a planar sp2 or linear sp atom with none free). A metal must lie along it, at SIGMA_MIN_HYBRID
    or more; otherwise it meets the p orbital. A pi face (a heavy neighbour on the same metal: an
    alkene, Cp, an arene) is pi-bound whatever its shape.
    """
    if data.electrons.get(G.nodes[i]["symbol"]) != 4:
        return False
    metals = [m for m in G.neighbors(i) if G.nodes[m]["symbol"] in data.metals]
    others = [x for x in G.neighbors(i) if G.nodes[x]["symbol"] not in data.metals]
    if not metals or any(G.has_edge(m, x) for m in metals for x in others if G.nodes[x]["symbol"] != "H"):
        return False
    at = np.asarray(G.nodes[i]["position"])

    def unit(k: int) -> np.ndarray:
        v = np.asarray(G.nodes[k]["position"]) - at
        return v / np.linalg.norm(v)

    free = -sum((unit(x) for x in others), np.zeros(3))
    return all(float(free @ unit(m)) >= SIGMA_MIN_HYBRID for m in metals)


def _compute_formal_charge_vec(
    valence_electrons: np.ndarray,
    bond_order_sums: np.ndarray,
    degree: np.ndarray | None = None,
    at_allowed: np.ndarray | None = None,
    full_shell: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised formal-charge computation.

    Must match ``BondOrderOptimizer._compute_formal_charge_value`` exactly;
    ``at_allowed`` marks atoms whose bond sum is one of their allowed valences,
    ``full_shell`` atoms that may not stay neutral short of their shell
    and keep the lone pair they give the metal.
    """
    bos = bond_order_sums
    V = valence_electrons.astype(np.float64)
    target = np.minimum(8.0, 2.0 * V)  # an octet, a duet for H
    l_octet = np.maximum(0.0, target - 2.0 * bos)
    l_neutral = V - bos
    all_single = True if degree is None else np.abs(bos - degree) < 1e-9
    if at_allowed is not None:
        all_single = all_single | at_allowed
    if full_shell is not None:
        all_single = all_single & ~(full_shell & (V + bos < target))
    use_neutral = (
        all_single
        & (l_neutral >= 0)
        & (np.abs(l_neutral - np.round(l_neutral)) < 1e-9)
        & (np.round(l_neutral).astype(np.int64) % 2 == 0)
    )
    L = np.where(use_neutral, l_neutral, l_octet)
    if full_shell is not None:
        L = np.where(full_shell, np.maximum(L, 2.0), L)
    return np.round(V - L - bos).astype(np.int64)


@dataclass
class ScoringArrays:
    """Immutable array representation of a molecular graph.

    Constructed once via :meth:`from_graph`, then used for all scoring
    calls during optimisation.  Only ``bond_orders`` is mutable.
    """

    # --- Node arrays (length N) ---
    n_atoms: int
    atomic_numbers: np.ndarray  # int [N]
    valences_by_z: Dict[int, np.ndarray]  # allowed valences per element, for isoelectronic lookups
    terminal: np.ndarray  # bool [N], one non-metal neighbour: only the lowest valence
    n_metals: int
    metal_cap: float  # total valence electrons of the metals: they cannot give up more
    metal_state_sums: np.ndarray  # totals the metals reach at their usual oxidation states (or 0) each
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
    has_metal_neighbor: np.ndarray  # bool [N]
    donor_full_shell: np.ndarray  # bool [N], metal-bound lone-pair donors (>= 5 valence electrons)
    non_metal_degree: np.ndarray  # int [N]

    # --- Edge arrays (length E) ---
    n_edges: int
    edge_src: np.ndarray  # intp [E]
    edge_dst: np.ndarray  # intp [E]
    bond_orders: np.ndarray  # float [E]
    is_metal_coord: np.ndarray  # bool [E]
    fixed_order: np.ndarray  # bool [E], a bond to a metal (ionic convention) or to H (one orbital): order stays 1
    edge_has_metal: np.ndarray  # bool [E]
    pi_stretch: np.ndarray  # float [E], a π-capable bond's distance / vdW sum beyond a single bond's, else 0

    # --- CSR adjacency ---
    node_edge_neighbor: np.ndarray  # intp [2*E]
    csr_owners: np.ndarray  # intp [2*E]

    # --- Ring data ---
    edge_in_ring: np.ndarray  # bool [E], the bond lies on a ring
    n_aromatic_rings: int  # rings that could be aromatic
    ring_systems: List[Tuple[Tuple[int, ...], np.ndarray, np.ndarray]]  # aromatic_systems: rings, atoms, + substituents

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_graph(
        cls,
        G: nx.Graph,
        data,  # MolecularData
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

        # Allowed valences (padded 2-D array built without per-atom loops). An expanded valence needs
        # partners to bond: an atom with one non-metal neighbour keeps its lowest (a Br is Br-C, never C#Br).
        terminal = np.array(
            [sum(1 for nb in G.neighbors(i) if sym_list[nb] not in data.metals) == 1 for i in range(n)], dtype=bool
        )
        raw_vals = [sorted(data.valences.get(s, []))[: 1 if terminal[i] else None] for i, s in enumerate(sym_list)]
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
        for i in {j for m in np.where(is_metal)[0] for j in G.neighbors(int(m))}:
            if sigma_bound(G, i, data):
                scoring_vlim[i] = min(scoring_vlim[i], 3.0)
        for sym, lim in VALENCE_CHECK_LIMITS.items():
            mask = symbol_strs == sym
            check_vlim[mask] = lim

        # --- Edge arrays (extract from nx.Graph into numpy) ---
        raw_edges = [
            (min(ei, ej), max(ei, ej), d.get("bond_order", 1.0), bool(d.get("metal_coord", False)), d.get("distance"))
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

        metal_syms = [s for s in sym_list if s in data.metals]
        state_sums = {0}
        for s in metal_syms:
            state_sums = {total + q for total in state_sums for q in {0, *data.valences.get(s, [])}}

        # Bond-length evidence for π-capable bonds (no metal, no H): distance over the vdW sum.
        vdw = np.array([data.vdw.get(s, 2.0) for s in sym_list], dtype=np.float64)
        dist = np.array([np.nan if e[4] is None else e[4] for e in raw_edges], dtype=np.float64)
        pi_capable = ~edge_has_metal & ~is_h[edge_src] & ~is_h[edge_dst] if n_edges > 0 else np.empty(0, dtype=bool)
        stretch = dist / (vdw[edge_src] + vdw[edge_dst]) - SINGLE_BOND_VDW_FRACTION
        pi_stretch = np.where(pi_capable & ~np.isnan(dist), stretch, 0.0)

        edge_index_map: Dict[Tuple[int, int], int] = {(int(edge_src[i]), int(edge_dst[i])): i for i in range(n_edges)}

        # --- CSR adjacency (built via numpy argsort, no Python loops) ---
        if n_edges > 0:
            # Each edge (src, dst) contributes two entries: src->dst and dst->src
            owners = np.concatenate([edge_src, edge_dst])  # [2*E] node that "owns" this entry
            neighbors = np.concatenate([edge_dst, edge_src])  # [2*E] the neighbor

            # Sort by owner node to build CSR order
            sort_order = np.argsort(owners, kind="stable")
            owners_sorted = owners[sort_order]
            node_edge_neighbor = neighbors[sort_order]

            # Build pointer array from sorted owner counts
            node_edge_ptr = np.zeros(n + 1, dtype=np.intp)
            np.add.at(node_edge_ptr[1:], owners_sorted, 1)
            np.cumsum(node_edge_ptr, out=node_edge_ptr)
        else:
            node_edge_ptr = np.zeros(n + 1, dtype=np.intp)
            node_edge_neighbor = np.empty(0, dtype=np.intp)

        # CSR owner array (cached — reused in scoring hot path)
        if len(node_edge_neighbor) > 0:
            csr_owners = np.repeat(np.arange(n), np.diff(node_edge_ptr))
            # has_h_neighbor (fully vectorised via CSR scatter)
            nbr_is_h = is_h[node_edge_neighbor]  # [2*E]
            has_h_neighbor = np.zeros(n, dtype=bool)
            np.bitwise_or.at(has_h_neighbor, csr_owners, nbr_is_h)
            nbr_is_metal = is_metal[node_edge_neighbor]
            has_metal_neighbor = np.zeros(n, dtype=bool)
            np.bitwise_or.at(has_metal_neighbor, csr_owners, nbr_is_metal)
            nbr_is_non_metal = (~nbr_is_metal).astype(np.int64)
            non_metal_degree = np.zeros(n, dtype=np.int64)
            np.add.at(non_metal_degree, csr_owners, nbr_is_non_metal)
        else:
            csr_owners = np.empty(0, dtype=np.intp)
            has_h_neighbor = np.zeros(n, dtype=bool)
            has_metal_neighbor = np.zeros(n, dtype=bool)
            non_metal_degree = np.zeros(n, dtype=np.int64)

        # --- Ring data ---
        rings = G.graph.get("_rings", [])
        edge_in_ring = np.zeros(n_edges, dtype=bool)
        for ring in rings:
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                eidx = edge_index_map.get((min(a, b), max(a, b)))
                if eidx is not None:
                    edge_in_ring[eidx] = True
        in_a_ring = {i for ring in rings for i in ring}
        capable = [ring for ring in rings if aromatic_capable(G, ring, data)]
        systems = []
        for members, atoms in aromatic_systems(capable):
            substituents = {
                nb for i in atoms for nb in G.neighbors(i) if nb not in in_a_ring and sym_list[nb] not in data.metals
            }
            systems.append((members, np.array(atoms, dtype=np.intp), np.array([*atoms, *sorted(substituents)])))

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
            has_metal_neighbor=has_metal_neighbor,
            donor_full_shell=has_metal_neighbor & ~is_metal & (valence_electrons >= 5),
            non_metal_degree=non_metal_degree,
            n_edges=n_edges,
            edge_src=edge_src,
            edge_dst=edge_dst,
            bond_orders=bond_orders_arr,
            is_metal_coord=is_metal_coord,
            fixed_order=is_metal_coord | is_h[edge_src] | is_h[edge_dst] if n_edges else np.empty(0, dtype=bool),
            edge_has_metal=edge_has_metal,
            node_edge_neighbor=node_edge_neighbor,
            csr_owners=csr_owners,
            edge_in_ring=edge_in_ring,
            n_aromatic_rings=len(capable),
            ring_systems=systems,
            pi_stretch=pi_stretch,
            atomic_numbers=np.array([data.s2n.get(s, 0) for s in sym_list], dtype=np.int64),
            terminal=terminal,
            valences_by_z={
                data.s2n[s]: np.array(v, dtype=np.float64) for s, v in data.valences.items() if s in data.s2n
            },
            n_metals=len(metal_syms),
            metal_cap=float(sum(data.electrons.get(s, 0) for s in metal_syms)),
            metal_state_sums=np.array(sorted(state_sums)),
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

    def _allowed_valence_gap(self, valence_sums: np.ndarray) -> np.ndarray:
        """Per atom, distance from its bond sum to the nearest allowed valence (inf without valence data)."""
        gap = np.full(self.n_atoms, np.inf)
        if self.vi_mask.any():
            diffs = np.abs(valence_sums[self.vi_mask, np.newaxis] - self.vi_allowed)
            diffs[~self.vi_allowed_mask] = np.inf
            gap[self.vi_mask] = np.min(diffs, axis=1)
        return gap

    def _valence_gap(self, valence_sums: np.ndarray, fc: np.ndarray, neutral_gap: np.ndarray) -> np.ndarray:
        """Per atom, distance to its nearest allowed valence, as the score counts it.

        A charged atom has the valences of the element it is isoelectronic with (N+ as C, O- as F).
        """
        gap = neutral_gap.copy()
        for i in np.flatnonzero((fc != 0) & self.vi_mask):
            iso = self.valences_by_z.get(int(self.atomic_numbers[i] - fc[i]))
            if iso is not None:
                iso = iso[:1] if self.terminal[i] else iso
                gap[i] = float(np.min(np.abs(iso - valence_sums[i])))
        return gap

    def charge_terms(self, fc: np.ndarray, charge: Optional[int]) -> Tuple[float, float]:
        """Impossible and soft deviations from the stated total ``charge``.

        The metals take whatever the non-metals leave: beyond their valence electrons is impossible,
        a total no combination of their usual oxidation states reaches is soft. Without metals the
        total must be met, bar the one electron an odd count leaves unpaired (the octet rule reads it
        as a charge). An unknown (None) charge of a metal-free molecule aims, softly, for the least a
        closed shell allows: 0, or 1 either way for an odd electron count.
        """
        nonmetal_charge = int(np.sum(fc[self.non_metal]))
        if charge is None:
            return 0.0, float(max(0, abs(nonmetal_charge) - int(self.atomic_numbers.sum()) % 2))
        if self.n_metals:
            metal_charge = charge - nonmetal_charge
            impossible = max(0.0, metal_charge - self.metal_cap)
            soft = 0.0 if impossible else float(np.min(np.abs(self.metal_state_sums - metal_charge)))
            return impossible, soft
        mismatch = abs(nonmetal_charge - charge)
        impossible = max(0, mismatch - int(self.atomic_numbers.sum() - charge) % 2)
        return float(impossible), float(mismatch - impossible)

    def formal_charges(self, valence_sums: np.ndarray, gap: np.ndarray | None = None) -> np.ndarray:
        """Formal charges under ``valence_sums``; a metal reads 0 here."""
        if gap is None:
            gap = self._allowed_valence_gap(valence_sums)
        fc = _compute_formal_charge_vec(
            self.valence_electrons,
            valence_sums,
            self.non_metal_degree,
            at_allowed=gap < 1e-9,
            full_shell=self.donor_full_shell,
        )
        fc[self.is_metal] = 0
        return fc

    def score(
        self,
        valence_sums: np.ndarray,
        bond_orders: np.ndarray,
        charge: Optional[int],
        weights: ScoringWeights,
    ) -> Tuple[float, np.ndarray]:
        """Vectorised scoring — replaces ``_score_assignment``."""
        # Fast reject
        if self.check_valence_violation(valence_sums):
            return 1e9, np.zeros(self.n_atoms, dtype=np.int64)

        gap = self._allowed_valence_gap(valence_sums)
        fc = self.formal_charges(valence_sums, gap)

        non_metal = self.non_metal
        abs_fc = np.abs(fc)
        # A localised charge costs q^2; a donor's charge is the bookkeeping of its bond to the metal
        # (an oxo O2-, an imido NR2-), so it costs q.
        charge_cost = np.where(self.donor_full_shell, abs_fc, abs_fc**2)
        fc_sum = float(np.sum(charge_cost[non_metal]))
        # Charged sites exclude donors, whose charge is their bond to the metal. Opposite charges across a
        # bond (N+-O-, C-#O+) write one polar bond, not two separate charges.
        site = non_metal & ~self.donor_full_shell & (fc != 0)
        polar = int(
            np.count_nonzero(site[self.edge_src] & site[self.edge_dst] & (fc[self.edge_src] * fc[self.edge_dst] < 0))
        )
        n_charged = int(np.count_nonzero(site)) - polar

        valence_err = float(np.sum(self._valence_gap(valence_sums, fc, gap)[self.vi_mask] ** 2))

        # Scoring valence limit violations (pre-computed threshold)
        over_limit = non_metal & (valence_sums > self.scoring_vlim_thresh)
        violation = float(np.sum(over_limit))

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

        # Tiebreaker: a negative charge prefers the metal-bound atom.
        metal_anion = non_metal & (fc < 0) & self.has_metal_neighbor
        en_penalty -= weights.metal_anion_bonus * float(np.sum(metal_anion))

        # Only an exchangeable Bronsted proton (N/O/S-H, ``is_nos``) can relocate
        # to relieve a positive neighbour; a C-H cannot, so it does not count.
        protonation = 0.0
        h_neigh_neutral = non_metal & self.is_nos & self.has_h_neighbor & (fc == 0)
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

        # Rings that could be aromatic and are not.
        conjugation = self._ring_conjugation_penalty(bond_orders, fc, valence_sums)

        # A π bond on a bond shorter than a single one gains, on a longer one pays.
        geometry = float(np.dot(bond_orders - 1.0, self.pi_stretch))

        impossible, charge_error = self.charge_terms(fc, charge)
        violation = (violation + impossible) * weights.violation_weight  # squared in the total: effectively hard

        total = (
            weights.violation_weight * violation
            + weights.conjugation_weight * conjugation
            + weights.protonation_weight * protonation
            + weights.formal_charge_weight * fc_sum
            + weights.charged_atoms_weight * n_charged
            + weights.charge_error_weight * charge_error
            + weights.electronegativity_weight * en_penalty
            + weights.valence_error_weight * valence_err
            + weights.geometry_weight * geometry
        )
        return total, fc

    # =====================================================================
    # Ring conjugation penalty
    # =====================================================================

    def _ring_conjugation_penalty(self, bond_orders: np.ndarray, fc: np.ndarray, valence_sums: np.ndarray) -> int:
        """Rings that could be aromatic but are not under this Lewis structure (see huckel_aromatic)."""
        if not self.n_aromatic_rings:
            return 0
        has_pi = (bond_orders > 1.3) & ~self.edge_has_metal
        ring_pi = np.zeros(self.n_atoms, dtype=bool)
        exo_pi = np.zeros(self.n_atoms, dtype=bool)
        for flags, edges in ((ring_pi, has_pi & self.edge_in_ring), (exo_pi, has_pi & ~self.edge_in_ring)):
            flags[self.edge_src[edges]] = True
            flags[self.edge_dst[edges]] = True
        pi = ring_pi_electrons(self.valence_electrons, fc, valence_sums, self.non_metal_degree, ring_pi, exo_pi)
        aromatic = {
            r
            for members, atoms, system in self.ring_systems
            if huckel_aromatic(pi[atoms], int(fc[system].sum()))
            for r in members
        }
        return self.n_aromatic_rings - len(aromatic)

    # =====================================================================
    # Edge candidate selection
    # =====================================================================

    def eligible_edges_mask(self, bond_orders: np.ndarray) -> np.ndarray:
        """Bool mask [E] of edges eligible for bond-order changes."""
        return ~self.fixed_order & (bond_orders < 3.0)

    def top_candidate_edges(self, bond_orders: np.ndarray, valence_sums: np.ndarray, k: int) -> np.ndarray:
        """Return indices of top-k candidate edges by valence-error pressure.

        Ranks by each endpoint's distance to its nearest allowed valence (what
        the scorer's valence_err term squares), emitting an edge only if an
        endpoint is off a valid valence.  Edges between two satisfied atoms are
        skipped: a flip there moves both away from target, so a solved region
        yields no candidates on its own.
        """
        eligible = self.eligible_edges_mask(bond_orders)
        eidxs = np.where(eligible)[0]
        if len(eidxs) == 0:
            return np.empty(0, dtype=np.intp)

        # Search pressure, not the score: a charged atom sits off its neutral valences, so moves that
        # could neutralise or shift its charge stay candidates. The score uses the isoelectronic gap.
        verr = self._allowed_valence_gap(valence_sums)
        verr[~self.vi_mask] = 0.0

        src = self.edge_src[eidxs]
        dst = self.edge_dst[eidxs]
        pressure = verr[src] + verr[dst]

        # Keep only edges touching an unsatisfied atom.
        keep = pressure > 0.1
        eidxs = eidxs[keep]
        pressure = pressure[keep]
        if len(eidxs) == 0:
            return np.empty(0, dtype=np.intp)

        order = np.argsort(-pressure)
        return eidxs[order[:k]]

    # =====================================================================
    # Write back to nx.Graph
    # =====================================================================

    def read_bond_orders(self, G: nx.Graph) -> np.ndarray:
        """Read the graph's current bond orders in this array's edge order."""
        return np.array(
            [G[int(i)][int(j)].get("bond_order", 1.0) for i, j in zip(self.edge_src, self.edge_dst)], dtype=np.float64
        )

    def write_bond_orders_to_graph(self, G: nx.Graph, bond_orders: np.ndarray) -> None:
        """Apply array bond orders back to the nx.Graph edge attributes."""
        for eidx in range(self.n_edges):
            i, j = int(self.edge_src[eidx]), int(self.edge_dst[eidx])
            G[i][j]["bond_order"] = float(bond_orders[eidx])
