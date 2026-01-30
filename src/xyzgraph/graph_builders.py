"""Molecular graph construction."""

import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

from .config import DEFAULT_PARAMS
from .data_loader import DATA
from .utils import read_xyz_file

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# =============================================================================
# METADATA COMPUTATION
# =============================================================================


def compute_metadata(
    method: str,
    charge: int,
    multiplicity: Optional[int],
    quick: bool,
    optimizer: str,
    max_iter: int,
    edge_per_iter: int,
    beam_width: int,
    bond: Optional[List[Tuple[int, int]]],
    unbond: Optional[List[Tuple[int, int]]],
    clean_up: bool,
    threshold: float,
    threshold_h_h: float,
    threshold_h_nonmetal: float,
    threshold_h_metal: float,
    threshold_metal_ligand: float,
    threshold_nonmetal_nonmetal: float,
    relaxed: bool,
    allow_metal_metal_bonds: bool,
    threshold_metal_metal_self: float,
    period_scaling_h_bonds: float,
    period_scaling_nonmetal_bonds: float,
) -> Dict[str, Any]:
    """
    Compute non-default parameters for metadata.

    Returns dict of parameters that differ from defaults.
    """
    non_default = {}

    if method != DEFAULT_PARAMS["method"]:
        non_default["method"] = method
    if charge != DEFAULT_PARAMS["charge"]:
        non_default["charge"] = charge
    if multiplicity != DEFAULT_PARAMS["multiplicity"]:
        non_default["multiplicity"] = multiplicity
    if quick != DEFAULT_PARAMS["quick"]:
        non_default["quick"] = quick
    if optimizer != DEFAULT_PARAMS["optimizer"]:
        non_default["optimizer"] = optimizer
    if max_iter != DEFAULT_PARAMS["max_iter"]:
        non_default["max_iter"] = max_iter
    if edge_per_iter != DEFAULT_PARAMS["edge_per_iter"]:
        non_default["edge_per_iter"] = edge_per_iter
    if beam_width != DEFAULT_PARAMS["beam_width"]:
        non_default["beam_width"] = beam_width
    if bond != DEFAULT_PARAMS["bond"]:
        non_default["bond"] = bond
    if unbond != DEFAULT_PARAMS["unbond"]:
        non_default["unbond"] = unbond
    if clean_up != DEFAULT_PARAMS["clean_up"]:
        non_default["clean_up"] = clean_up
    if threshold != DEFAULT_PARAMS["threshold"]:
        non_default["threshold"] = threshold
    if threshold_h_h != DEFAULT_PARAMS["threshold_h_h"]:
        non_default["threshold_h_h"] = threshold_h_h
    if threshold_h_nonmetal != DEFAULT_PARAMS["threshold_h_nonmetal"]:
        non_default["threshold_h_nonmetal"] = threshold_h_nonmetal
    if threshold_h_metal != DEFAULT_PARAMS["threshold_h_metal"]:
        non_default["threshold_h_metal"] = threshold_h_metal
    if threshold_metal_ligand != DEFAULT_PARAMS["threshold_metal_ligand"]:
        non_default["threshold_metal_ligand"] = threshold_metal_ligand
    if threshold_nonmetal_nonmetal != DEFAULT_PARAMS["threshold_nonmetal_nonmetal"]:
        non_default["threshold_nonmetal_nonmetal"] = threshold_nonmetal_nonmetal
    if relaxed != DEFAULT_PARAMS["relaxed"]:
        non_default["relaxed"] = relaxed
    if allow_metal_metal_bonds != DEFAULT_PARAMS["allow_metal_metal_bonds"]:
        non_default["allow_metal_metal_bonds"] = allow_metal_metal_bonds
    if threshold_metal_metal_self != DEFAULT_PARAMS["threshold_metal_metal_self"]:
        non_default["threshold_metal_metal_self"] = threshold_metal_metal_self
    if period_scaling_h_bonds != DEFAULT_PARAMS["period_scaling_h_bonds"]:
        non_default["period_scaling_h_bonds"] = period_scaling_h_bonds
    if period_scaling_nonmetal_bonds != DEFAULT_PARAMS["period_scaling_nonmetal_bonds"]:
        non_default["period_scaling_nonmetal_bonds"] = period_scaling_nonmetal_bonds

    return non_default


# =============================================================================
# GRAPH-BASED BOND CONSTRUCTION CLASS
# =============================================================================


class GraphBuilder:
    """Molecular graph construction with integrated state management.

    atoms: List of (symbol, (x, y, z)) tuples.
    """

    def __init__(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        charge: int = DEFAULT_PARAMS["charge"],
        multiplicity: Optional[int] = DEFAULT_PARAMS["multiplicity"],
        method: str = DEFAULT_PARAMS["method"],
        quick: bool = DEFAULT_PARAMS["quick"],
        optimizer: str = DEFAULT_PARAMS["optimizer"],
        max_iter: int = DEFAULT_PARAMS["max_iter"],
        edge_per_iter: int = DEFAULT_PARAMS["edge_per_iter"],
        beam_width: int = DEFAULT_PARAMS["beam_width"],
        bond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["bond"],
        unbond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["unbond"],
        clean_up: bool = DEFAULT_PARAMS["clean_up"],
        debug: bool = DEFAULT_PARAMS["debug"],
        threshold: float = DEFAULT_PARAMS["threshold"],
        threshold_h_h: float = DEFAULT_PARAMS["threshold_h_h"],
        threshold_h_nonmetal: float = DEFAULT_PARAMS["threshold_h_nonmetal"],
        threshold_h_metal: float = DEFAULT_PARAMS["threshold_h_metal"],
        threshold_metal_ligand: float = DEFAULT_PARAMS["threshold_metal_ligand"],
        threshold_nonmetal_nonmetal: float = DEFAULT_PARAMS["threshold_nonmetal_nonmetal"],
        relaxed: bool = DEFAULT_PARAMS["relaxed"],
        allow_metal_metal_bonds: bool = DEFAULT_PARAMS["allow_metal_metal_bonds"],
        threshold_metal_metal_self: float = DEFAULT_PARAMS["threshold_metal_metal_self"],
        period_scaling_h_bonds: float = DEFAULT_PARAMS["period_scaling_h_bonds"],
        period_scaling_nonmetal_bonds: float = DEFAULT_PARAMS["period_scaling_nonmetal_bonds"],
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

        if self.optimizer not in ("greedy", "beam"):
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
        self.relaxed = relaxed
        self.allow_metal_metal_bonds = allow_metal_metal_bonds
        self.threshold_metal_metal_self = threshold_metal_metal_self
        self.period_scaling_h_bonds = period_scaling_h_bonds
        self.period_scaling_nonmetal_bonds = period_scaling_nonmetal_bonds

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
        """Log message."""
        if self.debug:
            indent = "  " * level
            line = f"{indent}{msg}"
            print(line)
            self.log_buffer.append(line)

    def get_log(self) -> str:
        """Get full build log as string."""
        return "\n".join(self.log_buffer)

    # =========================================================================
    # Helper methods
    # =========================================================================

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _calculate_angle(self, atom1: int, center: int, atom2: int, G: nx.Graph) -> float:
        """Calculate angle (in degrees) between three atoms: atom1-center-atom2."""
        pos1 = np.array(G.nodes[atom1]["position"])
        pos_center = np.array(G.nodes[center]["position"])
        pos2 = np.array(G.nodes[atom2]["position"])

        v1 = pos1 - pos_center
        v2 = pos2 - pos_center

        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return 0.0

        v1 = v1 / v1_norm
        v2 = v2 / v2_norm

        # Calculate angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi

        return angle

    def _ring_angle_sum(self, ring: List[int], G: nx.Graph) -> float:
        """Calculate sum of internal angles in a ring."""
        angle_sum = 0.0
        n = len(ring)

        for i in range(n):
            prev = ring[(i - 1) % n]
            curr = ring[i]
            next = ring[(i + 1) % n]
            angle = self._calculate_angle(prev, curr, next, G)
            angle_sum += angle

        return angle_sum

    def _validate_bond_geometry(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        distance: float,
        confidence: float,
        baseline_bonds=None,
    ) -> bool:
        """Check if adding bond i-j creates geometrically valid configuration.

        Used for low-confidence (long) bonds from extended thresholds.

        Parameters
        ----------
        confidence : float
            Bond confidence score (0.0 = at threshold, 1.0 = very short).
            Used to set adaptive thresholds for diagonal detection.
        baseline_bonds : list, optional
            List of (confidence, i, j, distance, has_metal) tuples.
            Used for agostic H-M bond filtering.

        Returns
        -------
        bool
            True if bond should be added, False if it's spurious.
        """
        # If neither atom has neighbors yet, bond is valid
        if G.degree(i) == 0 and G.degree(j) == 0:
            return True

        # Get symbols to check for metals
        sym_i = G.nodes[i]["symbol"]
        sym_j = G.nodes[j]["symbol"]
        is_metal_i = sym_i in self.data.metals
        is_metal_j = sym_j in self.data.metals
        has_metal = is_metal_i or is_metal_j  # Bond involves metal at either end

        # Agostic H-M / F-M bond filtering: reject weak H-M or F-M bonds
        if has_metal and baseline_bonds is not None:
            # Identify H or F atom
            nonmetal_atom = None
            if sym_i in ("H", "F"):
                nonmetal_atom = i
                nonmetal_sym = sym_i
            elif sym_j in ("H", "F"):
                nonmetal_atom = j
                nonmetal_sym = sym_j

            if nonmetal_atom is not None:
                # Check existing strong bonds to non-metals (excluding metals)
                for X_atom in G.neighbors(nonmetal_atom):
                    X_sym = G.nodes[X_atom]["symbol"]
                    if X_sym in self.data.metals or X_sym == "H":
                        continue  # Skip metal neighbors and H-H

                    # Look up bond confidence
                    for conf, bi, bj, _, _ in baseline_bonds:
                        if nonmetal_atom in (bi, bj) and X_atom in (bi, bj):
                            # Reject weak M-nonmetal bonds if stronger bond exists
                            if conf / max(confidence, 0.01) > 2.0:
                                self.log(
                                    f"  Rejected {nonmetal_sym}-M agostic: {nonmetal_sym}-X bond stronger "
                                    f"(conf={conf:.2f} vs {confidence:.2f})",
                                    4,
                                )
                                return False
                            break  # Found relevant bond, check complete

        # Configure thresholds based on relaxed mode
        # Relaxed mode: more permissive for TS structures with strained rings
        if self.relaxed:
            # Relaxed thresholds
            acute_threshold_metal = 12.0
            acute_threshold_nonmetal = 20.0
            angle_threshold_h_ring = 115.0
            angle_threshold_base = 135.0  # Fixed value for relaxed, no Z-dependence
            diagonal_ratio_initial = 0.75
            diagonal_ratio_max = 0.85
            diagonal_ratio_hard = 0.90
            STRENGTH_RATIO = 5
            conf_threshold = 0.5
        else:
            # Strict thresholds (current default behavior)
            acute_threshold_metal = 15.0
            acute_threshold_nonmetal = 35.0
            angle_threshold_h_ring = 95.0
            angle_threshold_base = 110.0  # Will add Z-factor: 110 + (avg_z - 6) * 2
            diagonal_ratio_initial = 0.65
            diagonal_ratio_max = 0.75
            diagonal_ratio_hard = 0.80
            STRENGTH_RATIO = 20
            conf_threshold = 0.75

        if confidence < conf_threshold and not has_metal and baseline_bonds is not None:
            # Get neighbors of i and j
            neighbors_i = set(G.neighbors(i))
            neighbors_j = set(G.neighbors(j))

            # Check if any neighbor of i connects to any neighbor of j
            # This would form 4-ring: i-ni-nj-j
            forms_4ring = any(G.has_edge(ni, nj) for ni in neighbors_i for nj in neighbors_j if ni != nj)

            if forms_4ring:
                # Check if either atom would exceed valence AND all existing bonds are much stronger
                for atom in [i, j]:
                    atom_sym = G.nodes[atom]["symbol"]

                    # Skip if no valence data
                    if atom_sym not in DATA.valences:
                        continue

                    # Check current valence (excluding metals)
                    current_val = sum(
                        G[atom][nbr].get("bond_order", 1.0)
                        for nbr in G.neighbors(atom)
                        if G.nodes[nbr]["symbol"] not in self.data.metals
                    )
                    max_val = max(DATA.valences[atom_sym])

                    # Only check if atom would exceed valence
                    if current_val + 1.0 > max_val:
                        # Check if ALL existing bonds to this atom are much stronger
                        all_bonds_stronger = all(
                            conf_baseline / max(confidence, 0.001) > STRENGTH_RATIO
                            for conf_baseline, bi, bj, _, _ in baseline_bonds
                            if atom in (bi, bj)
                        )

                        # Reject only if ALL existing bonds are much stronger
                        if all_bonds_stronger:
                            self.log(
                                f"  Rejected bond {sym_i}{i}-{sym_j}{j}: weak 4-ring closure "
                                f"(conf={confidence:.2f}), ALL existing bonds stronger",
                                4,
                            )
                            return False

        # Check angles at atom i with existing neighbors
        for existing_neighbor in G.neighbors(i):
            angle = self._calculate_angle(existing_neighbor, i, j, G)

            # Variable acute angle threshold: metals more lenient vs non-metals
            acute_threshold = acute_threshold_metal if has_metal else acute_threshold_nonmetal

            if angle < acute_threshold:
                self.log(
                    f"  Rejected bond {sym_i}{i}-{sym_j}{j}: angle too acute "
                    f"({angle:.1f}°, threshold={acute_threshold:.1f}°) with {existing_neighbor}-{i}",
                    4,
                )
                return False

            # Nearly collinear - check if spurious or valid geometry
            if angle > 160.0:
                # If bond involves metal (at either end), be lenient with collinearity
                if has_metal:
                    self.log(
                        f"  Bond {i}-{j}: collinear ({angle:.1f}°) with {existing_neighbor}-{i}, "
                        f"involves metal ({sym_i}-{sym_j}) - allowed",
                        4,
                    )
                    continue

                # For non-metal bonds: distinguish same direction (spurious) vs opposite (trans/linear)
                if G.degree(i) >= 2:
                    # Calculate direction vectors
                    pos_i = np.array(G.nodes[i]["position"])
                    pos_existing = np.array(G.nodes[existing_neighbor]["position"])
                    pos_new = np.array(G.nodes[j]["position"])

                    v_existing = pos_existing - pos_i
                    v_new = pos_new - pos_i

                    # Normalize
                    v_existing = v_existing / np.linalg.norm(v_existing)
                    v_new = v_new / np.linalg.norm(v_new)

                    # Dot product: +1 = same direction, -1 = opposite
                    dot_product = np.dot(v_existing, v_new)

                    # Same direction (bond behind another) - spurious
                    if dot_product > 0.9:
                        self.log(
                            f"  Rejected bond {sym_i}{i}-{sym_j}{j}: collinear ({angle:.1f}°) "
                            f"same direction as {existing_neighbor}-{i}",
                            4,
                        )
                        return False
                    # Opposite direction (trans/linear) - valid
                    elif dot_product < -0.9:
                        self.log(
                            f"  Bond {sym_i}{i}-{sym_j}{j}: collinear ({angle:.1f}°) "
                            f"opposite direction to {existing_neighbor}-{i} - valid trans",
                            4,
                        )
                        continue
                    # Not truly collinear - allow
                    else:
                        continue

        # Check angles at atom j with existing neighbors (symmetric check)
        for existing_neighbor in G.neighbors(j):
            angle = self._calculate_angle(existing_neighbor, j, i, G)

            # Use configured threshold (same as for atom i)
            acute_threshold = acute_threshold_metal if has_metal else acute_threshold_nonmetal

            if angle < acute_threshold:
                self.log(
                    f"  Rejected bond {sym_i}{i}-{sym_j}{j}: angle too acute "
                    f"({angle:.1f}°, threshold={acute_threshold:.1f}°) with {existing_neighbor}-{j}",
                    4,
                )
                return False

            # Nearly collinear - check if spurious or valid geometry
            if angle > 160.0:
                # If bond involves metal, be lenient
                if has_metal:
                    self.log(
                        f"  Bond {sym_i}{i}-{sym_j}{j}: collinear ({angle:.1f}°) "
                        f"with {existing_neighbor}-{j}, involves metal - allowed",
                        4,
                    )
                    continue

                # For non-metal bonds: check direction
                if G.degree(j) >= 2:
                    pos_j = np.array(G.nodes[j]["position"])
                    pos_existing = np.array(G.nodes[existing_neighbor]["position"])
                    pos_new = np.array(G.nodes[i]["position"])

                    v_existing = pos_existing - pos_j
                    v_new = pos_new - pos_j

                    v_existing = v_existing / np.linalg.norm(v_existing)
                    v_new = v_new / np.linalg.norm(v_new)

                    dot_product = np.dot(v_existing, v_new)

                    if dot_product > 0.9:
                        self.log(
                            f"  Rejected bond {sym_i}{i}-{sym_j}{j}: collinear ({angle:.1f}°) "
                            f"same direction as {existing_neighbor}-{j}",
                            4,
                        )
                        return False
                    elif dot_product < -0.9:
                        self.log(
                            f"  Bond {sym_i}{i}-{sym_j}{j}: collinear ({angle:.1f}°) "
                            f"opposite direction to {existing_neighbor}-{j} - valid trans",
                            4,
                        )
                        continue
                    else:
                        continue

        # Check if i and j are ALREADY in the same ring (would create diagonal)
        current_rings = G.graph.get("_rings", [])
        for ring in current_rings:
            ring_set = set(ring)
            # If both i and j are in this ring, adding bond would create diagonal
            if i in ring_set and j in ring_set:
                # === CLUSTER BYPASS: Check if ring is homogeneous inorganic cluster ===
                ring_elements = {G.nodes[node]["symbol"] for node in ring}
                if len(ring_elements) == 1 and next(iter(ring_elements)) not in {"C", "H"}:
                    elem = next(iter(ring_elements))
                    elem_count = G.graph.get("_element_counts", {}).get(elem, 0)
                    if elem_count >= 8:
                        self.log(
                            f"  Bond {sym_i}{i}-{sym_j}{j}: diagonal in homogeneous {elem} cluster ring - allowed",
                            4,
                        )
                        continue  # Skip this ring check, continue to next ring

                # Allow for very small rings (3-4) if metal involved
                if len(ring) <= 4 and has_metal:
                    self.log(
                        f"  Bond {sym_i}{i}-{sym_j}{j}: diagonal in existing {len(ring)}-ring involves metal - allowed",
                        4,
                    )
                    continue
                # Reject diagonals in small rings (would create impossible geometry)
                if len(ring) <= 4:
                    self.log(
                        f"  Rejected bond {sym_i}{i}-{sym_j}{j}: would create diagonal in existing {len(ring)}-ring",
                        4,
                    )
                    return False
                if len(ring) >= 5:
                    self.log(
                        f"  Rejected bond {sym_i}{i}-{sym_j}{j}: would create diagonal in existing {len(ring)}-ring",
                        4,
                    )
                    return False

        # Check for diagonal bonds across existing rings (creates spurious 3-rings)
        # If i and j share a common neighbor, they form a triangle - check if it's reasonable
        common_neighbors = set(G.neighbors(i)) & set(G.neighbors(j))

        if common_neighbors:
            # Check each potential 3-ring formed via common neighbor
            for k in common_neighbors:
                # === CLUSTER BYPASS: Check if potential 3-ring is homogeneous cluster ===
                sym_k = G.nodes[k]["symbol"]
                ring_elements = {sym_i, sym_j, sym_k}
                if len(ring_elements) == 1 and next(iter(ring_elements)) not in {"C", "H"}:
                    elem = next(iter(ring_elements))
                    elem_count = G.graph.get("_element_counts", {}).get(elem, 0)
                    if elem_count >= 8:
                        self.log(
                            f"  Bond {sym_i}{i}-{sym_j}{j}: 3-ring in homogeneous {elem} cluster"
                            f" - bypassing validation",
                            4,
                        )
                        continue  # Skip validation for this 3-ring, check next common neighbor

                # === 3-RING VALIDATION ===
                is_metal_k = sym_k in self.data.metals
                if is_metal_k:
                    if "H" not in (sym_i, sym_j):
                        self.log(
                            f"  3-ring formation via {sym_k}{k} involves metal, low confidence L-L, rejected",
                            4,
                        )
                        return False

                # M-L BOND PRIORITY CHECK: Reject weak M-ligand bonds if 3-ring crosses stronger M-donor bond
                # Similar to agostic H-M filtering, but applies to any ligand forming 3-ring via metal
                has_metal_in_bond = is_metal_i or is_metal_j

                if has_metal_in_bond and baseline_bonds is not None:
                    # Identify which atom is metal, which is ligand
                    metal_atom = i if is_metal_i else j

                    # Check if k (ring vertex) is already bonded to metal with higher confidence
                    for conf, bi, bj, _, _ in baseline_bonds:
                        if metal_atom in (bi, bj) and k in (bi, bj):
                            # Found existing M-k bond - compare confidences
                            if "H" in (sym_i, sym_j, sym_k):
                                if conf / max(confidence, 0.01) > 1.5:
                                    self.log(
                                        f"  Rejected bond {sym_i}{i}-{sym_j}{j}: 3-ring via {sym_k}{k}, "
                                        f"existing M-{sym_k}{k} bond stronger (conf={conf:.2f} vs {confidence:.2f}, "
                                        f"ratio={conf / max(confidence, 0.01):.1f})",
                                        4,
                                    )
                                    return False
                            elif conf / max(confidence, 0.01) > 3.0:
                                self.log(
                                    f"  Rejected bond {sym_i}{i}-{sym_j}{j}: 3-ring diagonal, existing M-{sym_k}{k} "
                                    f"bond much stronger (conf={conf:.2f} vs {confidence:.2f}, "
                                    f"ratio={conf / max(confidence, 0.01):.1f})",
                                    4,
                                )
                                return False

                # ANGLE CHECK (metal-aware, then H-aware)
                angle_i = self._calculate_angle(k, i, j, G)
                angle_j = self._calculate_angle(k, j, i, G)
                angle_k = self._calculate_angle(i, k, j, G)
                max_angle = max(angle_i, angle_j, angle_k)

                has_H_in_ring = "H" in (sym_i, sym_j, sym_k)
                has_metal_in_ring = any(s in self.data.metals for s in (sym_i, sym_j, sym_k))

                # Priority: Metal > H > others
                if has_metal_in_ring:
                    # Metal in ring
                    angle_threshold = 135.0 if self.relaxed else 115.0
                    ring_type = "metal-containing"
                elif has_H_in_ring:
                    # Use configured H-ring threshold
                    angle_threshold = angle_threshold_h_ring
                    ring_type = "H-containing"
                else:
                    # Use configured base threshold + Z-factor
                    z_list = [
                        G.nodes[i]["atomic_number"],
                        G.nodes[j]["atomic_number"],
                        G.nodes[k]["atomic_number"],
                    ]
                    avg_z = sum(min(z, 18) for z in z_list) / 3.0
                    angle_threshold = angle_threshold_base + (avg_z - 6) * 2.0
                    ring_type = "non-H"

                if max_angle > angle_threshold:
                    self.log(
                        f"  Rejected bond {sym_i}{i}-{sym_j}{j}: 3-ring angle {max_angle:.1f}° > "
                        f"{angle_threshold:.1f}° ({ring_type})",
                        4,
                    )
                    return False

                # DISTANCE RATIO CHECK (diagonal detection)
                d_ik = G[i][k]["distance"]
                d_kj = G[k][j]["distance"]
                d_ij = distance

                # VDW-normalize distances for atom-type awareness (consistent with threshold calculation)
                norm_ik = d_ik / (DATA.vdw[sym_i] + DATA.vdw[sym_k])
                norm_kj = d_kj / (DATA.vdw[sym_k] + DATA.vdw[sym_j])
                norm_ij = d_ij / (DATA.vdw[sym_i] + DATA.vdw[sym_j])

                # Calculate ratio using normalized distances
                norm_path = norm_ik + norm_kj
                ratio = norm_ij / norm_path

                # If diagonal is nearly as long as going around, it's suspicious
                if ratio > diagonal_ratio_initial:  # Use configured initial threshold
                    # Confidence-based threshold (stricter for low confidence)
                    # √2/2 ≈ 0.707 is theoretical diagonal ratio for square
                    max_conf_for_interp = 0.7
                    diagonal_threshold = (
                        diagonal_ratio_initial
                        + min(confidence, max_conf_for_interp)
                        * (diagonal_ratio_max - diagonal_ratio_initial)
                        / max_conf_for_interp
                    )

                    self.log(
                        f"  3-ring via {sym_k}{k}: ratio={ratio:.3f}, threshold={diagonal_threshold:.3f}",
                        4,
                    )

                    if ratio > diagonal_threshold:
                        if has_metal and not has_H_in_ring:
                            self.log(
                                f"  Bond {sym_i}{i}-{sym_j}{j}: diagonal (ratio={ratio:.2f}) across 3-ring via "
                                f"{sym_k}{k}, metal bond - allowed",
                                4,
                            )
                            continue

                        # Before rejecting, check valence as fallback
                        # Real 3-rings (epoxide) have capacity, spurious diagonals don't
                        atoms_at_limit = 0
                        for atom in [i, j]:
                            atom_sym = G.nodes[atom]["symbol"]

                            # Skip metals only when checking M-L bonds with H in ring
                            # (these are handled permissively; don't reject based on metal valence)
                            if atom_sym in self.data.metals and has_metal and has_H_in_ring:
                                continue

                            if atom_sym not in DATA.valences:
                                continue

                            current_val = sum(
                                G[atom][nbr].get("bond_order", 1.0)
                                for nbr in G.neighbors(atom)
                                if G.nodes[nbr]["symbol"] not in self.data.metals
                            )
                            max_val = max(DATA.valences[atom_sym])

                            if current_val + 1.0 > max_val:
                                atoms_at_limit += 1

                        if atoms_at_limit > 1:
                            # Both at limit + bad ratio = spurious diagonal
                            self.log(
                                f"  Rejected bond {sym_i}{i}-{sym_j}{j}: diagonal across 3-ring via {sym_k}{k} "
                                f"(ratio={ratio:.2f}, threshold={diagonal_threshold:.2f}) and both atoms at "
                                f"valence limit",
                                4,
                            )
                            return False
                        elif ratio > diagonal_ratio_hard:
                            # Even with valence capacity, ratio > hard threshold is too suspicious
                            self.log(
                                f"  Rejected bond {sym_i}{i}-{sym_j}{j}: diagonal ratio too high (ratio={ratio:.2f} > "
                                f"{diagonal_ratio_hard:.2f}) even with valence capacity",
                                4,
                            )
                            return False
                        else:
                            for atom in [i, j]:
                                if G.nodes[atom]["symbol"] != "C" or G.degree(atom) <= 3:
                                    continue

                                # Carbon would become 5-coordinate - verify out-of-plane approach
                                other = j if atom == i else i
                                neighbors = list(G.neighbors(atom))[:3]  # First 3 for plane definition

                                # Position vectors relative to carbon
                                pos_C = np.array(G.nodes[atom]["position"])
                                vec_new = np.array(G.nodes[other]["position"]) - pos_C
                                vec_nb = [np.array(G.nodes[n]["position"]) - pos_C for n in neighbors]

                                # Plane normal from first 2 neighbor vectors
                                normal = np.cross(vec_nb[0], vec_nb[1])
                                norm_mag = np.linalg.norm(normal)

                                if norm_mag < 1e-6:  # Collinear neighbors - skip check
                                    continue

                                # Normalize vectors
                                normal /= norm_mag
                                vec_new /= np.linalg.norm(vec_new)

                                # Angle between new bond and plane normal
                                angle_to_normal = (
                                    np.arccos(np.clip(np.abs(np.dot(vec_new, normal)), 0, 1)) * 180 / np.pi
                                )

                                # Reject if in-plane (real TS should approach from out-of-plane)
                                if angle_to_normal < 60:
                                    self.log(
                                        f"  Rejected bond {sym_i}{i}-{sym_j}{j}: C hypervalent but in-plane "
                                        f"(angle to normal={angle_to_normal:.1f}°, need >60°)",
                                        4,
                                    )
                                    return False

                            # At least one has valence capacity + reasonable ratio = likely real 3-ring (e.g., epoxide)
                            self.log(
                                f"  Bond {sym_i}{i}-{sym_j}{j}: suspicious ratio ({ratio:.2f}) but valence allows "
                                f"- likely real 3-ring",
                                4,
                            )
                            # Continue to next common neighbor check

                # VALENCE CHECK (focus on bonding atoms i-j)
                atoms_at_limit = 0
                for atom in [i, j]:  # Only check the two atoms being bonded
                    atom_sym = G.nodes[atom]["symbol"]

                    # Skip metals only when checking M-L bonds with H in ring
                    # (these are handled permissively; don't reject based on metal valence)
                    if atom_sym in self.data.metals and has_metal and has_H_in_ring:
                        continue

                    # Skip if element not in valence dictionary (unknown chemistry)
                    if atom_sym not in DATA.valences:
                        continue

                    current_val = sum(
                        G[atom][nbr].get("bond_order", 1.0)
                        for nbr in G.neighbors(atom)
                        if G.nodes[nbr]["symbol"] not in self.data.metals
                    )
                    max_val = max(DATA.valences[atom_sym])

                    # Check if adding this bond would EXCEED max valence
                    if current_val + 1.0 > max_val:
                        atoms_at_limit += 1

                if atoms_at_limit > 1:
                    # In relaxed mode, allow modest valence overflow (≤1.0 over max)
                    if self.relaxed:
                        # Check if overflow is modest (both atoms ≤1.0 over their max valence)
                        overflow_ok = True
                        for atom in [i, j]:
                            atom_sym = G.nodes[atom]["symbol"]
                            # Skip metals only when checking M-L bonds with H in ring
                            if atom_sym in self.data.metals and has_metal and has_H_in_ring:
                                continue
                            if atom_sym not in DATA.valences:
                                continue
                            current_val = sum(
                                G[atom][nbr].get("bond_order", 1.0)
                                for nbr in G.neighbors(atom)
                                if G.nodes[nbr]["symbol"] not in self.data.metals
                            )
                            max_val = max(DATA.valences[atom_sym])
                            overflow = (current_val + 1.0) - max_val
                            if overflow > 1.0:  # More than 1.0 over max is too much
                                overflow_ok = False
                                break

                        if overflow_ok:
                            self.log(
                                f"  Bond {sym_i}{i}-{sym_j}{j}: both atoms would exceed valence but overflow ≤1.0 "
                                f"- allowed in relaxed mode",
                                4,
                            )
                        else:
                            self.log(
                                f"  Rejected bond {sym_i}{i}-{sym_j}{j}: both bonding atoms would exceed valence "
                                f"by >1.0 (even in relaxed mode)",
                                4,
                            )
                            return False
                    else:
                        # Both bonding atoms would exceed - reject
                        self.log(
                            f"  Rejected bond {sym_i}{i}-{sym_j}{j}: both bonding atoms would exceed valence",
                            4,
                        )
                        return False

        # All checks passed → allow this 3-ring bond
        return True

    def _find_new_rings_from_edge(self, G: nx.Graph, i: int, j: int) -> List[List[int]]:
        """Efficiently detect new rings formed by adding edge (i, j).

        Find shortest path from i to j. Returns list of new rings.
        """
        # Skip if either atom is a metal (no organic rings to track)
        if G.nodes[i]["symbol"] in DATA.metals or G.nodes[j]["symbol"] in DATA.metals:
            return []

        # Work on metal-free subgraph
        non_metal_nodes = [n for n in G.nodes() if G.nodes[n]["symbol"] not in DATA.metals]
        G_no_metals = G.subgraph(non_metal_nodes).copy()

        # Check if i and j are in the metal-free graph
        if i not in G_no_metals or j not in G_no_metals:
            return []

        # Temporarily remove the new edge (i,j) to find alternative path
        if G_no_metals.has_edge(i, j):
            G_no_metals.remove_edge(i, j)

        # Find shortest path from i to j
        try:
            path = nx.shortest_path(G_no_metals, source=i, target=j)
            # Path found → ring formed
            # Ring is the path + the new edge (i,j)
            return [path]
        except nx.NetworkXNoPath:
            # No path exists → no ring formed
            return []
        except nx.NodeNotFound:
            # Node not in graph
            return []

    def _get_period(self, atomic_number: int) -> int:
        """Get period (row) from atomic number."""
        if atomic_number <= 2:
            return 1
        elif atomic_number <= 10:
            return 2
        elif atomic_number <= 18:
            return 3
        elif atomic_number <= 36:
            return 4
        elif atomic_number <= 54:
            return 5
        elif atomic_number <= 86:
            return 6
        else:
            return 7

    def _get_threshold_with_period_scaling(
        self, base_threshold: float, z_i: int, z_j: int, has_hydrogen: bool = False
    ) -> float:
        """
        Apply period-dependent scaling to bond threshold.

        Heavier elements need looser thresholds because VDW/covalent
        ratio increases down the periodic table.

        Parameters
        ----------
        - base_threshold: Base threshold value
        - z_i, z_j: Atomic numbers of the two atoms
        - has_hydrogen: Whether bond involves hydrogen

        Returns scaled threshold
        """
        if has_hydrogen:
            # H bonds: use H-bond scaling factor
            if self.period_scaling_h_bonds == 0.0:
                return base_threshold
            non_h_z = z_i if z_i > 1 else z_j
            period = self._get_period(non_h_z)
            period_factor = 1.0 + (period - 2) * self.period_scaling_h_bonds
            return base_threshold * period_factor
        else:
            # Non-H bonds: check if both atoms are nonmetals
            sym_i = DATA.n2s.get(z_i, "")
            sym_j = DATA.n2s.get(z_j, "")
            both_nonmetal = sym_i not in DATA.metals and sym_j not in DATA.metals

            if both_nonmetal and self.period_scaling_nonmetal_bonds != 0.0:
                # Both nonmetals: use nonmetal scaling
                max_period = max(self._get_period(z_i), self._get_period(z_j))
                period_factor = 1.0 + (max_period - 2) * self.period_scaling_nonmetal_bonds
                return base_threshold * period_factor
            else:
                # Metal bond: no period scaling
                return base_threshold

    def _should_bond_metal(self, sym_i: str, sym_j: str) -> bool:
        """
        Chemical filter for metal bonds (called AFTER distance check).

        Returns False only for implausible metal pairings.

        Accepts:
        - Metal to donor atoms (O, N, C, P, S)
        - Metal to halides/oxo (ionic)
        - Metal to H (hydrides)
        - Metal-metal (if allow_metal_metal_bonds flag is enabled)
        """
        # Neither metal - always OK
        if sym_i not in self.data.metals and sym_j not in self.data.metals:
            return True

        # Both metals - check flag
        if sym_i in self.data.metals and sym_j in self.data.metals:
            return self.allow_metal_metal_bonds

        # One metal, one non-metal - check ligand plausibility
        other = sym_j if sym_i in self.data.metals else sym_i

        # Accept common ligands
        if other in ("O", "N", "C", "P", "S", "H"):
            return True

        # Accept halides
        if other in ("F", "Cl", "Br", "I"):
            return True

        # Accept other plausible ligands
        if other in ("B", "Si", "Se", "Te"):
            return True

        return False

    @staticmethod
    def _valence_sum(G: nx.Graph, node: int) -> float:
        """Sum bond orders around a node."""
        return sum(G.edges[node, nbr].get("bond_order", 1.0) for nbr in G.neighbors(node))

    def _compute_formal_charge_value(self, symbol: str, valence_electrons: int, bond_order_sum: float) -> int:
        """Compute formal charge for an atom."""
        if symbol == "H":
            return valence_electrons - int(bond_order_sum)

        B = 2 * bond_order_sum
        target = 8
        L = max(0, target - B)
        return round(valence_electrons - L - B / 2)

    def _compute_formal_charges(self, G: nx.Graph) -> List[int]:
        """Compute formal charges for all atoms and balance to total charge."""
        formal = []

        self.log("\n" + "=" * 80, 0)
        self.log("FORMAL CHARGE CALCULATION", 0)
        self.log("=" * 80, 0)

        for node in G.nodes():
            sym = G.nodes[node]["symbol"]

            if sym in DATA.metals:
                formal.append(0)
                continue

            V = DATA.electrons.get(sym)
            if V is None:
                formal.append(0)
                continue

            # Exclude metal bonds from ligand valence for formal charge calculation
            # Metal-ligand bonds are coordinative/dative
            bond_sum = sum(
                G.edges[node, nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(node)
                if G.nodes[nbr]["symbol"] not in DATA.metals  # Exclude metal bonds
            )

            # Special case: H bonded only to metal(s) is hydride (H⁻)
            if sym == "H" and bond_sum == 0:
                if all(G.nodes[nbr]["symbol"] in DATA.metals for nbr in G.neighbors(node)):
                    fc = -1  # Hydride
                    formal.append(fc)
                    continue

            fc = self._compute_formal_charge_value(sym, V, bond_sum)
            formal.append(fc)

        # Check if system has metals
        has_metals = any(G.nodes[i]["symbol"] in DATA.metals for i in G.nodes())

        # Log initial formal charges
        initial_sum = sum(formal)
        self.log("\nInitial formal charges:", 2)
        self.log(f"  Sum: {initial_sum:+d} (target: {self.charge:+d})", 3)

        if has_metals:
            # Show metal coordination summary
            self.log("\nMetal coordination summary:", 3)

            # Compute ligand classification inline, passing formal charges
            ligand_classification = self._classify_metal_ligands(G, formal)

            for metal_idx, ox_state in sorted(ligand_classification["metal_ox_states"].items()):
                metal_sym = G.nodes[metal_idx]["symbol"]
                coord_num = len(list(G.neighbors(metal_idx)))

                # Get ligands for this metal
                metal_dative = [entry for entry in ligand_classification["dative_bonds"] if entry[0] == metal_idx]
                metal_ionic = [entry for entry in ligand_classification["ionic_bonds"] if entry[0] == metal_idx]

                self.log(
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
                        self.log(
                            f"  • {ligand_type:>6} ({charge_str})  [donor: {d_sym}{donor}]",
                            4,
                        )

                # Display neutral ligands
                if metal_dative:
                    for entry in metal_dative:
                        _m, donor, ligand_type = entry if len(entry) == 3 else (*entry, "unknown")
                        d_sym = G.nodes[donor]["symbol"]
                        self.log(f"  • {ligand_type:>6} ( 0)  [donor: {d_sym}{donor}]", 4)
        else:
            # No metals - show traditional formal charge list
            charged_atoms = [(i, formal[i]) for i in range(len(formal)) if formal[i] != 0]
            if charged_atoms:
                self.log("  Charged atoms:", 3)
                for i, fc in charged_atoms:
                    sym = G.nodes[i]["symbol"]
                    self.log(f"    {sym}{i}: {fc:+d}", 4)
            else:
                self.log("  (no charged atoms)", 3)

        # Balance residual charge with priority-based distribution
        residual = self.charge - sum(formal)

        # Check if system has metals - if so, skip redistribution
        # (residual represents metal oxidation states, ligand charges are already correct)
        has_metals = any(G.nodes[i]["symbol"] in DATA.metals for i in G.nodes())

        if residual != 0 and not has_metals:
            self.log("\nResidual charge distribution needed:", 2)
            self.log(f"  Residual: {residual:+d}", 3)

            candidates = []
            for i in range(len(self.atoms)):
                if self._valence_sum(G, i) == 0:
                    continue

                sym = G.nodes[i]["symbol"]
                if sym in DATA.metals:
                    continue

                # Skip atoms bonded to metals (their charge is balanced by metal coordination)
                # This preserves ligand charges: Cp⁻, CO, etc.
                bonded_to_metal = any(G.nodes[nbr]["symbol"] in DATA.metals for nbr in G.neighbors(i))
                if bonded_to_metal:
                    continue

                score = 0

                # Priority: heteroatoms (more electronegative, better charge bearers)
                if sym in ("O", "N", "S", "Cl", "Br", "I", "F", "P"):
                    score += 5

                # Lower priority: already charged (can accumulate more charge)
                if abs(formal[i]) > 0:
                    score += 2

                candidates.append((score, i))

            candidates.sort(reverse=True, key=lambda x: x[0])

            self.log("  Top candidates (showing first 10):", 3)
            for score, idx in candidates[:10]:
                sym = G.nodes[idx]["symbol"]
                current_fc = formal[idx]
                self.log(f"    {sym}{idx}: score={score}, current_fc={current_fc:+d}", 4)

            # Distribute charge
            sign = 1 if residual > 0 else -1
            distributed_to = []
            for _, idx in candidates:
                if residual == 0:
                    break
                formal[idx] += sign
                residual -= sign
                distributed_to.append((G.nodes[idx]["symbol"], idx, formal[idx]))

            self.log(f"  Distributed to {len(distributed_to)} atoms:", 3)
            for sym, idx, new_fc in distributed_to:
                self.log(f"    {sym}{idx}: {new_fc:+d}", 4)
        elif residual != 0 and has_metals:
            self.log("\nMetal complex detected: ", 2)
            self.log(f"  Residual: {residual:+d} (represents metal oxidation states)", 3)
        else:
            self.log("\nNo residual charge distribution needed (sum matches target)", 2)

        return formal

    def _check_valence_violation(
        self, G: nx.Graph, limits: Optional[Dict[str, float]] = None, tol: float = 0.3
    ) -> bool:
        """Check for pentavalent carbon etc."""
        if limits is None:
            limits = {"C": 4}

        for i in G.nodes():
            sym = G.nodes[i]["symbol"]
            if sym in limits:
                # Exclude metal bonds from valence (like formal charge calculation)
                val = sum(
                    G[i][j].get("bond_order", 1.0) for j in G.neighbors(i) if G.nodes[j]["symbol"] not in DATA.metals
                )
                if val > limits[sym] + tol:
                    return True
        return False

    # =========================================================================
    # Main build method
    # =========================================================================

    def build(self) -> nx.Graph:
        """Build molecular graph using configured method."""
        mode = "QUICK" if self.quick else "FULL"
        self.log(f"\n{'=' * 80}")
        self.log(f"BUILDING GRAPH ({self.method.upper()}, {mode} MODE)")
        self.log(f"Atoms: {len(self.atoms)}, Charge: {self.charge}, Multiplicity: {self.multiplicity}")
        self.log(f"{'=' * 80}\n")

        if self.method == "cheminf":
            self.graph = self._build_cheminf()
        elif self.method == "xtb":
            self.graph = self._build_xtb()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Store build log in graph
        self.graph.graph["build_log"] = self.get_log()

        self.log(f"\n{'=' * 80}")
        self.log("GRAPH CONSTRUCTION COMPLETE")
        self.log(f"{'=' * 80}\n")

        return self.graph

    # =========================================================================
    # Cheminformatics path
    # =========================================================================

    def _build_initial_graph(self) -> nx.Graph:
        """Build initial graph with 2-phase construction.

        Step 1: Baseline bonds (DEFAULT thresholds), compute rings from baseline structure.
        Step 2: Extended bonds (CUSTOM thresholds if modified, strict validation).
        """
        G = nx.Graph()

        pos = np.array(self.positions)

        # Add nodes
        for i, atomic_num, symbol in zip(range(len(self.atoms)), self.atomic_numbers, self.symbols):
            G.add_node(i, symbol=symbol, atomic_number=atomic_num, position=tuple(pos[i]))

        self.log(f"Added {len(self.atoms)} atoms", 1)

        # Precompute element counts and chemical formula (for cluster detection and metadata)
        from collections import Counter

        element_counts = Counter(self.symbols)
        G.graph["_element_counts"] = dict(element_counts)

        # Generate chemical formula (sorted by Hill system: C, H, then alphabetical)
        formula_parts = []
        if "C" in element_counts:
            formula_parts.append(f"C{element_counts['C']}" if element_counts["C"] > 1 else "C")
        if "H" in element_counts:
            formula_parts.append(f"H{element_counts['H']}" if element_counts["H"] > 1 else "H")
        for elem in sorted(element_counts.keys()):
            if elem not in ("C", "H"):
                formula_parts.append(f"{elem}{element_counts[elem]}" if element_counts[elem] > 1 else elem)
        G.graph["formula"] = "".join(formula_parts)

        self.log(f"Chemical formula: {G.graph['formula']}", 1)

        # Check if custom thresholds are being used
        has_custom = (
            self.threshold != DEFAULT_PARAMS["threshold"]
            or self.threshold_h_h != DEFAULT_PARAMS["threshold_h_h"]
            or self.threshold_h_nonmetal != DEFAULT_PARAMS["threshold_h_nonmetal"]
            or self.threshold_h_metal != DEFAULT_PARAMS["threshold_h_metal"]
            or self.threshold_metal_ligand != DEFAULT_PARAMS["threshold_metal_ligand"]
            or self.threshold_nonmetal_nonmetal != DEFAULT_PARAMS["threshold_nonmetal_nonmetal"]
        )

        if has_custom:
            self.log("Custom thresholds detected - using 2-phase construction", 1)

        # ===== STEP 1: Baseline bonds (using DEFAULT thresholds) =====
        baseline_bonds = []

        for i in range(len(self.atoms)):
            si = self.symbols[i]
            is_metal_i = si in self.data.metals

            for j in range(i + 1, len(self.atoms)):
                sj = self.symbols[j]
                is_metal_j = sj in self.data.metals
                has_metal = is_metal_i or is_metal_j
                is_metal_metal_self = is_metal_i and is_metal_j and (si == sj)
                has_h = "H" in (si, sj)

                d = self._distance(pos[i], pos[j])
                r_sum = DATA.vdw.get(si, 2.0) + DATA.vdw.get(sj, 2.0)

                # Use DEFAULT thresholds for baseline
                if si == "H" and sj == "H":
                    baseline_threshold = DEFAULT_PARAMS["threshold_h_h"] * r_sum * DEFAULT_PARAMS["threshold"]
                elif has_h and has_metal:
                    baseline_threshold = DEFAULT_PARAMS["threshold_h_metal"] * r_sum * DEFAULT_PARAMS["threshold"]
                elif has_h and not has_metal:
                    baseline_threshold = DEFAULT_PARAMS["threshold_h_nonmetal"] * r_sum * DEFAULT_PARAMS["threshold"]
                elif is_metal_metal_self:
                    baseline_threshold = DEFAULT_PARAMS["threshold_metal_metal_self"] * r_sum
                elif has_metal:
                    baseline_threshold = DEFAULT_PARAMS["threshold_metal_ligand"] * r_sum
                else:
                    baseline_threshold = (
                        DEFAULT_PARAMS["threshold_nonmetal_nonmetal"] * r_sum * DEFAULT_PARAMS["threshold"]
                    )

                # Apply period scaling using DEFAULT scaling factors
                z_i = self.atomic_numbers[i]
                z_j = self.atomic_numbers[j]
                baseline_threshold = self._get_threshold_with_period_scaling(
                    baseline_threshold, z_i, z_j, has_hydrogen=has_h
                )

                if d < baseline_threshold:
                    confidence = 1.0 - (d / baseline_threshold)
                    baseline_bonds.append((confidence, i, j, d, has_metal))

        # Sort by confidence (most confident bonds first)
        baseline_bonds.sort(reverse=True, key=lambda x: x[0])

        self.log(
            f"Step 1: Found {len(baseline_bonds)} baseline bonds (using default thresholds)",
            1,
        )

        # Add baseline bonds with confidence-based validation
        edge_count = 0
        rejected_count = 0

        for confidence, i, j, d, has_metal in baseline_bonds:
            si, sj = self.symbols[i], self.symbols[j]
            self.log(
                f"  Evaluating bond {si}{i}-{sj}{j} (d={d:.3f} Å, conf={confidence:.2f})",
                3,
            )
            # Check metal bonding rules
            if has_metal and not self._should_bond_metal(si, sj):
                rejected_count += 1
                continue

            # High confidence: add directly
            if confidence > 0.4:
                G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=has_metal)
                edge_count += 1
                self.log("  Added high-confidence bond", 4)
            # Low confidence: validate geometry
            elif self._validate_bond_geometry(G, i, j, d, confidence, baseline_bonds):
                G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=has_metal)
                edge_count += 1
                self.log("  Added validated bond", 4)
            else:
                rejected_count += 1

        self.log(f"Step 1: {edge_count} baseline bonds added, {rejected_count} rejected", 1)

        # Compute rings from baseline structure
        non_metal_nodes = [n for n in G.nodes() if G.nodes[n]["symbol"] not in DATA.metals]
        G_no_metals = G.subgraph(non_metal_nodes).copy()
        rings = nx.cycle_basis(G_no_metals)

        G.graph["_rings"] = rings
        G.graph["_neighbors"] = {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph["_has_H"] = {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()}

        self.log(f"Found {len(rings)} rings from initial bonding (excluding metal cycles)", 1)

        # ===== STEP 2: Extended bonds (CUSTOM thresholds if modified) =====
        if has_custom:
            extended_bonds = []
            baseline_edges = set(G.edges())

            for i in range(len(self.atoms)):
                si = self.symbols[i]
                is_metal_i = si in self.data.metals

                for j in range(i + 1, len(self.atoms)):
                    # Skip if already in baseline
                    if (i, j) in baseline_edges or (j, i) in baseline_edges:
                        continue

                    sj = self.symbols[j]
                    is_metal_j = sj in self.data.metals
                    has_metal = is_metal_i or is_metal_j
                    has_h = "H" in (si, sj)

                    d = self._distance(pos[i], pos[j])
                    r_sum = DATA.vdw.get(si, 2.0) + DATA.vdw.get(sj, 2.0)

                    # Use CUSTOM thresholds for extended
                    if si == "H" and sj == "H":
                        custom_threshold = self.threshold_h_h * r_sum * self.threshold
                    elif has_h and has_metal:
                        custom_threshold = self.threshold_h_metal * r_sum * self.threshold
                    elif has_h and not has_metal:
                        custom_threshold = self.threshold_h_nonmetal * r_sum * self.threshold
                    elif has_metal:
                        custom_threshold = self.threshold_metal_ligand * r_sum
                    else:
                        custom_threshold = self.threshold_nonmetal_nonmetal * r_sum * self.threshold

                    # Apply period scaling using CUSTOM scaling factors
                    z_i = self.atomic_numbers[i]
                    z_j = self.atomic_numbers[j]
                    custom_threshold = self._get_threshold_with_period_scaling(
                        custom_threshold, z_i, z_j, has_hydrogen=has_h
                    )

                    if d < custom_threshold:
                        confidence = 1.0 - (d / custom_threshold)
                        extended_bonds.append((confidence, i, j, d, has_metal))

            # Sort by confidence (HIGHEST first)
            extended_bonds.sort(reverse=True, key=lambda x: x[0])

            self.log(
                f"Step 2: Found {len(extended_bonds)} extended bonds (custom thresholds)",
                1,
            )

            # Add extended bonds with STRICT validation and incremental ring updates
            extended_added = 0
            extended_rejected = 0
            new_rings_count = 0

            for confidence, i, j, d, has_metal in extended_bonds:
                si, sj = self.symbols[i], self.symbols[j]
                self.log(
                    f"  Evaluating extended bond {si}{i}-{sj}{j} (d={d:.3f} Å, conf={confidence:.2f})",
                    3,
                )
                # Check metal bonding rules
                if has_metal and not self._should_bond_metal(si, sj):
                    extended_rejected += 1
                    continue

                # ALL extended bonds require geometric validation
                if self._validate_bond_geometry(G, i, j, d, confidence, baseline_bonds):
                    G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=has_metal)
                    extended_added += 1

                    # Incremental ring detection (metal-free)
                    new_rings = self._find_new_rings_from_edge(G, i, j)
                    if new_rings:
                        G.graph["_rings"].extend(new_rings)
                        new_rings_count += len(new_rings)
                        ring_size = len(new_rings[0])
                        self.log(f"    Bond {si}{i}-{sj}{j} creates new {ring_size}-ring", 3)

                    # Update caches incrementally
                    G.graph["_neighbors"][i] = list(G.neighbors(i))
                    G.graph["_neighbors"][j] = list(G.neighbors(j))
                else:
                    extended_rejected += 1

            self.log(
                f"Step 2: {extended_added} extended bonds added, {extended_rejected} rejected, "
                f"{new_rings_count} new rings detected",
                1,
            )

        # Handle user-specified bonds
        if self.bond:
            for i, j in self.bond:
                if not G.has_edge(i, j):
                    d = self._distance(pos[i], pos[j])
                    G.add_edge(i, j, bond_order=1, distance=d)
                    si = self.symbols[i]
                    sj = self.symbols[j]
                    self.log(f"Added user-specified bond {si}{i}-{sj}{j} (d={d:.3f} Å)", 2)

        if self.unbond:
            for i, j in self.unbond:
                if G.has_edge(i, j):
                    G.remove_edge(i, j)
                    si = self.symbols[i]
                    sj = self.symbols[j]
                    self.log(f"Removed user-specified bond {si}{i}-{sj}{j}", 2)

        # Final ring update if extended bonds were added
        if has_custom:
            non_metal_nodes = [n for n in G.nodes() if G.nodes[n]["symbol"] not in DATA.metals]
            G_no_metals = G.subgraph(non_metal_nodes).copy()
            rings = nx.cycle_basis(G_no_metals)
            G.graph["_rings"] = rings
            self.log(f"Final: {len(rings)} rings after extended bonds", 1)

        total_bonds = G.number_of_edges()
        self.log(f"Total bonds in graph: {total_bonds}", 1)

        return G

    def _estimate_pi_electrons(self, G: nx.Graph, cycle: List[int]) -> int:
        """
        Estimate π electrons using metal-bonding as hint.

        Heuristic: 5-membered C-ring bonded to metal → likely Cp⁻ → π=6
        """
        pi_electrons = 0
        bonded_to_metal = any(any(G.nodes[nbr]["symbol"] in DATA.metals for nbr in G.neighbors(c)) for c in cycle)

        for idx in cycle:
            sym = G.nodes[idx]["symbol"]
            if sym == "C":
                pi_electrons += 1
            elif sym == "N":
                # Use TOTAL degree, not in-ring degree
                degree = sum(1 for nbr in G.neighbors(idx) if G.nodes[nbr]["symbol"] not in DATA.metals)
                if degree == 3:
                    pi_electrons += 2  # Pyrrole-like: 3 bonds, LP in π system
                elif degree == 2:
                    pi_electrons += 1  # Pyridine-like: 2 bonds, LP not in π system
            elif sym in ("O", "S"):
                pi_electrons += 2

        # 5-ring bonded to metal with 5 π electrons → assume Cp⁻ (6 total)
        if len(cycle) == 5 and bonded_to_metal and pi_electrons == 5:
            pi_electrons += 1

        return pi_electrons

    def _init_kekule_for_aromatic_rings(self, G: nx.Graph) -> int:
        """Initialize Kekulé patterns for aromatic rings.

        1) Validate rings (planarity, aromatic atoms, sp2 carbons, Huckel, Cp-like).
        2) Initialize Kekulé patterns with propagation respecting fused rings.
        """
        cycles = G.graph.get("_rings")
        if cycles is None:
            cycles = nx.cycle_basis(G)
            G.graph["_rings"] = cycles

        initialized = 0
        aromatic_atoms = {"C", "N", "O", "S", "B", "P", "Se"}

        self.log("\n" + "=" * 80, 0)
        self.log("KEKULE INITIALIZATION FOR AROMATIC RINGS", 0)
        self.log("=" * 80, 0)

        # --- Phase 0: Precompute edge info ---
        edge_to_rings = {}
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

        ring_adj = {i: set() for i in range(len(cycles))}
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

            ring_atoms = [f"{G.nodes[i]['symbol']}{i}" for i in cycle]
            self.log(f"\nRing {r_idx} ({len(cycle)}-membered): {ring_atoms}", 2)

            # Must contain only aromatic atoms
            if not all(G.nodes[i]["symbol"] in aromatic_atoms for i in cycle):
                self.log("✗ Contains non-aromatic atoms", 3)
                continue

            # Check planarity
            if not self._check_planarity(cycle, G, threshold=0.15):
                self.log("✗ Not planar", 3)
                continue

            # Check for sp3 carbon
            has_sp3 = False
            for idx in cycle:
                sym = G.nodes[idx]["symbol"]
                if sym == "C":
                    degree = sum(
                        1 for nbr in G.neighbors(idx) if G.nodes[nbr]["symbol"] not in getattr(DATA, "metals", {})
                    )
                    if degree >= 4:
                        self.log(f"✗ Contains non-sp2 carbon {sym}{idx}", 3)
                        has_sp3 = True
                        break
            if has_sp3:
                continue

            # Cp-like detection for 5-membered carbon rings
            if len(cycle) == 5 and all(G.nodes[i]["symbol"] == "C" for i in cycle):
                metal_neighbors = {}
                for c in cycle:
                    for nbr in G.neighbors(c):
                        if G.nodes[nbr]["symbol"] in getattr(DATA, "metals", {}):
                            metal_neighbors.setdefault(nbr, []).append(c)
                is_cp_like = any(len(carbons) == 5 for carbons in metal_neighbors.values())
                if is_cp_like:
                    metal_idx = next(m for m, carbons in metal_neighbors.items() if len(carbons) == 5)
                    metal_sym = G.nodes[metal_idx]["symbol"]
                    self.log(
                        f"✓ Detected Cp-like ring (all 5 C bonded to {metal_sym}{metal_idx})",
                        3,
                    )

            # Estimate π electrons and Hückel rule
            pi_electrons = self._estimate_pi_electrons(G, cycle)
            self.log(f"π electrons estimate: {pi_electrons}", 3)
            if pi_electrons not in (6, 10):
                self.log(f"✗ Hückel rule violated (π={pi_electrons})", 3)
                continue

            valid_rings.add(r_idx)

        if not valid_rings:
            self.log("No rings passed validation, skipping Kekulé init", 1)
            return 0

        self.log(f"{'-' * 80}", 0)
        self.log(f"Valid rings for Kekulé initialization: \n\t{sorted(valid_rings)}", 0)

        # --- Phase 2: Kekulé initialization ---
        MAX_VALENCE = {"H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "Se": 2}

        def max_val(n):
            return MAX_VALENCE.get(G.nodes[n].get("symbol"), 4)

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
            metal_map = {}
            for c in cycles[r_idx]:
                for nbr in G.neighbors(c):
                    if G.nodes[nbr]["symbol"] in getattr(DATA, "metals", {}):
                        metal_map.setdefault(nbr, []).append(c)
            if any(len(cs) == 5 for cs in metal_map.values()):
                # Apply alternating pattern [1,2,1,2,1] rotated to best match existing anchors
                L = 5
                base = [1.0, 2.0, 1.0, 2.0, 1.0]
                applied = False
                for rot in range(L):
                    p = base[-rot:] + base[:-rot]
                    if apply_pattern(r_idx, p):
                        initialized += 1
                        applied = True
                        self.log(f"✓ Cp-like 5-ring {r_idx} initialized (rotation {rot})", 3)
                        break
                if not applied:
                    self.log(f"✗ Cp-like 5-ring {r_idx} could not be safely applied", 3)

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
                    initialized += 1
                    hetero_initialized.add(r_idx)
                    self.log(f"✓ 5-heterocycle {r_idx} (lp {lp}) initialized", 3)
                else:
                    self.log(
                        f"✗ 5-heterocycle {r_idx} (lp {lp}) could not be safely applied",
                        3,
                    )

        # --- Priority 2b: propagate to fused rings ---
        to_propagate = set()
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
                        initialized += 1
                        success = True
                        self.log(f"✓ Propagated init to fused ring {r_idx} (6-ring)", 3)
                        break
            elif L == 5:
                base = [2.0, 1.0, 2.0, 1.0, 1.0]
                for rot in range(5):
                    p = base[-rot:] + base[:-rot]
                    if apply_pattern(r_idx, p):
                        initialized += 1
                        success = True
                        self.log(
                            f"✓ Propagated init to fused ring {r_idx} (5-ring rotation {rot})",
                            3,
                        )
                        break
            if not success:
                self.log(f"• Could not propagate safely to fused ring {r_idx}", 4)

        # --- Priority 3 & 4: fused benzene clusters + isolated 6-rings ---
        six_ring_indices = [i for i in valid_rings if len(cycles[i]) == 6]
        if six_ring_indices:
            sub_adj = {i: set() for i in six_ring_indices}
            for i in six_ring_indices:
                for j in ring_adj[i]:
                    if j in sub_adj:
                        sub_adj[i].add(j)
            seen = set()
            for start in six_ring_indices:
                if start in seen:
                    continue
                comp = set()
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
                comp = sorted(comp)

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
                    # valence check and commit
                    for r in comp:
                        patt = assigned[r]
                        for idx_edge, (i, j) in enumerate(ring_edges[r]):
                            bo = patt[idx_edge]
                            if not can_set_edge(i, j, bo):
                                if bo > 1.5 and can_set_edge(i, j, 1.0):
                                    bo = 1.0
                                else:
                                    return None
                            G.edges[i, j]["bond_order"] = bo
                    return assigned

                assigned = None
                for seed in (True, False):
                    assigned = try_component(seed, comp)
                    if assigned is not None:
                        initialized += len(comp)
                        self.log(f"✓ Initialized fused benzene block rings {comp}", 3)
                        break
                if assigned is None:
                    self.log(
                        f"✗ Could not find consistent Kekulé for fused benzene block {comp}",
                        3,
                    )

        # --- Priority 5: remaining carbon-only 5-membered rings ---
        for r_idx in valid_rings:
            if len(cycles[r_idx]) != 5:
                continue
            if any(G.nodes[i]["symbol"] != "C" for i in cycles[r_idx]):
                continue
            fused = any(len(edge_to_rings[frozenset((a, b))]) > 1 for a, b in ring_edges[r_idx])
            if fused:
                self.log(f"• Skipping fused carbon-5 ring {r_idx}", 4)
                continue
            pattern = [2.0, 1.0, 2.0, 1.0, 1.0]
            if apply_pattern(r_idx, pattern):
                initialized += 1
                self.log(f"✓ Initialized isolated carbon-5 ring {r_idx}", 3)
            else:
                self.log(f"• Could not safely init isolated carbon-5 ring {r_idx}", 4)

        self.log("\n" + "-" * 80, 0)
        self.log(f"SUMMARY: Initialized {initialized} ring(s) with Kekulé pattern", 1)
        self.log("-" * 80, 0)
        return initialized

    # =============================================================================
    # QUICK MODE: Simple heuristic valence adjustment
    # =============================================================================

    def _quick_valence_adjust(self, G: nx.Graph) -> Dict[str, int]:
        """Perform fast heuristic bond order adjustment.

        No formal charge optimization - just satisfy valences.
        """
        stats = {"iterations": 0, "promotions": 0}

        # Lock metal bonds
        for i, j in G.edges():
            if G.edges[i, j].get("metal_coord", False):
                G.edges[i, j]["bond_order"] = 1.0

        for iteration in range(3):
            stats["iterations"] = iteration + 1
            changed = False

            # Calculate deficits
            deficits = {}
            for node in G.nodes():
                sym = G.nodes[node]["symbol"]
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
                dist_ratio = data["distance"] / (DATA.vdw.get(si, 2.0) + DATA.vdw.get(sj, 2.0))
                if dist_ratio > 0.60:
                    continue

                # Promote if both atoms need more valence
                if di > 0.3 and dj > 0.3:
                    increment = min(di, dj, 3.0 - bo)
                    if increment >= 0.5:
                        data["bond_order"] = bo + increment
                        stats["promotions"] += 1
                        changed = True
            self.log(f"Iteration {iteration + 1}: Promotions={stats['promotions']}", 1)

            if not changed:
                break

        return stats

    def _edge_score(self, G: nx.Graph, i: int, j: int) -> float:
        """Check scoring of edge."""
        if not self._eligible_edge(G, i, j):
            return float("-inf")
        si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
        vmax_i = max(DATA.valences.get(si, [4]))
        vmax_j = max(DATA.valences.get(sj, [4]))
        di = vmax_i - self.valence_cache[i]
        dj = vmax_j - self.valence_cache[j]
        return di + dj

    def _eligible_edge(self, G: nx.Graph, i: int, j: int) -> bool:
        data = G[i][j]
        if data.get("metal_coord", False):
            return False
        if data.get("locked", False):
            return False
        if data.get("bond_order", 1.0) >= 3.0:
            return False
        return True

    def _ekey(self, i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _edge_likelihood(self, G: nx.Graph, *, init: bool = False, touch_nodes: Optional[set] = None):
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
        top = [e for _, e in items[: self.edge_per_iter]]
        self.edge_scores_cache = top
        return top

    # =============================================================================
    # FULL MODE: Formal charge optimization
    # =============================================================================

    def _ring_conjugation_penalty(self, G: nx.Graph, rings) -> float:
        """Assess conjugation penalties in aromatic rings (5-6 members).

        Returns a numeric penalty (larger = worse).
        """
        conjugation_penalty = 0.0
        for ring in rings:
            if len(ring) not in (5, 6):
                continue

            conjugatable = {"C", "N", "O", "S", "P"}
            if not all(G.nodes[i]["symbol"] in conjugatable for i in ring):
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

                        # Skip metal bonds - they don't disrupt aromatic π system
                        if nbr_sym in DATA.metals:
                            continue

                        bo = data.get("bond_order", 1.0)
                        if bo >= 1.8:
                            if (ring_sym == "C" and nbr_sym != "O") or (ring_sym == "N" and nbr_sym in ("C", "P", "S")):
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
        """Optimize bond orders with formal charge minimization.

        Returns a stats dict containing iterations, improvements,
        initial_score, final_score, and final formal_charges.
        """
        self.log(f"\n{'=' * 80}", 0)
        self.log("FULL VALENCE OPTIMIZATION", 1)
        self.log("=" * 80, 0)

        # --- Precompute / cache graph info ---
        rings = G.graph.get("_rings") or nx.cycle_basis(G)
        G.graph["_rings"] = rings

        neighbor_cache = G.graph.get("_neighbors") or {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph["_neighbors"] = neighbor_cache

        has_H = G.graph.get("_has_H") or {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G[n]) for n in G.nodes()}
        G.graph["_has_H"] = has_H

        # Build valence cache excluding metal bonds (coordination bonds don't count)
        valence_cache = {
            n: sum(
                G[n][nbr].get("bond_order", 1.0) for nbr in G.neighbors(n) if G.nodes[nbr]["symbol"] not in DATA.metals
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
            self.log(f"Locked {metal_count} metal bonds", 1)

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

        self.log(f"Initial score: {initial_score:.2f}", 1)

        stagnation = 0
        self.edge_scores_cache = None
        last_promoted_edge = None
        self._edge_score_map = None

        # --- Optimization loop ---
        for iteration in range(self.max_iter):
            stats["iterations"] = iteration + 1
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
                self.log(
                    f"✓ {edge_label:<10}  {action}  Δscore = {best_delta:6.2f}  new_score = {current_score:8.2f}",
                    2,
                )

            else:
                stagnation += 1
                last_promoted_edge = None
                self.edge_scores_cache = None  # force full recompute next time

                if stagnation >= 3:
                    break  # stop if no improvement

        # --- Final scoring ---
        final_formal_charges = self._score_assignment(G, rings)[1]
        stats["final_score"] = current_score
        stats["final_formal_charges"] = final_formal_charges

        self.log("-" * 80, 0)
        self.log(f"Optimized: {stats['improvements']} improvements", 1)
        self.log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self.log("-" * 80, 0)

        return stats

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
                    if G.nodes[nbr]["symbol"] not in DATA.metals
                )
                for n in G.nodes()
            }
        else:
            # Incremental update (excluding metal bonds)
            for n in nodes:
                self.valence_cache[n] = sum(
                    G[n][nbr].get("bond_order", 1.0)
                    for nbr in G.neighbors(n)
                    if G.nodes[nbr]["symbol"] not in DATA.metals
                )

    def _restore_graph_caches(self, G: nx.Graph) -> None:
        """Rebuild cached graph properties after modifications.

        Called after applying bond order changes.
        """
        # Update neighbor cache
        G.graph["_neighbors"] = {n: list(G.neighbors(n)) for n in G.nodes()}

        # Update H-neighbor cache
        G.graph["_has_H"] = {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()}

    def _copy_graph_state(self, G: nx.Graph) -> nx.Graph:
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

    def _score_assignment(self, G: nx.Graph, rings: Optional[List[List[int]]] = None) -> Tuple[float, List[int]]:
        """Scoring that uses pre-computed valence cache."""
        EN = {
            "H": 2.2,
            "C": 2.5,
            "N": 3.0,
            "O": 3.5,
            "F": 4.0,
            "P": 2.2,
            "S": 2.6,
            "Cl": 3.2,
            "Br": 3.0,
            "I": 2.7,
        }

        if self._check_valence_violation(G):
            return 1e9, [0 for _ in G.nodes()]

        # Ring cache
        if rings is None:
            rings = G.graph.get("_rings", nx.cycle_basis(G))

        # Neighbor cache
        neighbor_cache = G.graph.get("_neighbors", {n: list(G.neighbors(n)) for n in G.nodes()})

        # H-neighbor cache
        has_H = G.graph.get(
            "_has_H",
            {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()},
        )

        penalties = {
            "valence": 0.0,
            "en": 0.0,
            "violation": 0.0,
            "protonation": 0.0,
            "conjugation": 0.0,
            "fc": 0,
            "n_charged": 0,
        }

        # Conjugation penalty
        penalties["conjugation"] = self._ring_conjugation_penalty(G, rings)

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
            sym = G.nodes[node]["symbol"]

            vsum = self.valence_cache[node]

            if sym in DATA.metals:
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
            if sym in DATA.valences:
                allowed = DATA.valences[sym]
                min_error = min(abs(vsum - v) for v in allowed)
                penalties["valence"] += min_error**2

                limits = {"C": 4, "N": 5, "O": 3, "S": 6, "P": 6}
                if sym in limits and vsum > limits[sym] + 0.1:
                    penalties["violation"] += 1000.0

            # Electronegativity penalty
            en = EN.get(sym, 2.5)
            if fc != 0:
                penalties["en"] += abs(fc) * ((3.5 - en) if fc < 0 else (en - 2.5)) * 0.5

        # Total score
        charge_error = abs(sum(formal_charges) - self.charge)
        score = (
            1000.0 * penalties["violation"]
            + 12.0 * penalties["conjugation"]
            + 8.0 * penalties["protonation"]
            + 10.0 * penalties["fc"]
            + 10.0 * penalties["n_charged"]
            + 10.0 * charge_error
            + 2.0 * penalties["en"]
            + 5.0 * penalties["valence"]
        )

        return score, formal_charges

    def _beam_search_optimize(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Memory-efficient beam search with incremental valence cache updates.

        Strategy:
        - Maintain valence cache per hypothesis (small dict)
        - When promoting edge (i,j), update valence for nodes i and j
        - Score calculation uses cached valences
        """
        self.log(f"\n{'=' * 80}", 0)
        self.log(f"BEAM SEARCH OPTIMIZATION (width={self.beam_width})", 0)
        self.log("=" * 80, 0)

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
            self.log(f"Locked {metal_count} metal bonds", 1)

        # Build initial valence cache excluding metal bonds (shared starting point)
        base_valence_cache = {
            n: sum(
                G[n][nbr].get("bond_order", 1.0) for nbr in G.neighbors(n) if G.nodes[nbr]["symbol"] not in DATA.metals
            )
            for n in G.nodes()
        }

        # Initial scoring
        self.valence_cache = base_valence_cache.copy()
        current_score, formal_charges = self._score_assignment(G, rings)
        initial_score = current_score

        self.log(f"Initial score: {initial_score:.2f}", 1)

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

        for iteration in range(self.max_iter):
            stats["iterations"] = iteration + 1
            self.log(f"\nIteration {iteration + 1}:", 1)

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
                self.log("  No improvements found in any beam, stopping", 2)
                break

            # Sort and keep top beam_width
            candidates.sort(key=lambda x: x[0])

            self.log(
                f"  Generated {len(candidates)} candidates, keeping top {min(self.beam_width, len(candidates))}",
                2,
            )

            beam = [
                (score, graph, cache, history) for score, graph, cache, edge, history in candidates[: self.beam_width]
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
                self.log(
                    f"  ✓ New best: {edge_label:<10}  Δtotal = {improvement:6.2f}  score = {best_ever_score:8.2f}",
                    2,
                )

        # Apply best solution
        self.log("\nApplying best solution to graph...", 1)
        for i, j, data in best_ever_graph.edges(data=True):
            G[i][j]["bond_order"] = data["bond_order"]

        # Restore caches
        self._restore_graph_caches(G)
        self.valence_cache = best_ever_cache

        # Final scoring
        final_score, final_formal_charges = self._score_assignment(G, rings)
        stats["final_score"] = final_score
        stats["final_formal_charges"] = final_formal_charges

        self.log("-" * 80, 0)
        self.log(
            f"Explored {stats['beam_explored']} states across {stats['iterations']} iterations",
            1,
        )
        self.log(f"Found {stats['improvements']} improvements", 1)
        self.log(f"Score: {initial_score:.2f} → {stats['final_score']:.2f}", 1)
        self.log("-" * 80, 0)

        return stats

    # =============================================================================
    # AROMATIC DETECTION
    # =============================================================================

    def _detect_aromatic_rings(self, G: nx.Graph) -> int:
        """Detect aromatic rings using Hückel rule (4n+2 π electrons).

        Only performed on 5 and 6 member rings with C, N, O, S, P atoms.
        Sets bond orders to 1.5 for aromatic rings.
        """
        self.log(f"\n{'=' * 80}", 0)
        self.log("AROMATIC RING DETECTION (Hückel 4n+2)", 0)
        self.log("=" * 80, 0)

        # Use cached cycles (metal-free) instead of recalculating
        cycles = G.graph.get("_rings", [])
        aromatic_count = 0
        aromatic_rings = 0

        for ring_idx, cycle in enumerate(cycles):
            if len(cycle) not in (5, 6):
                continue

            ring_atoms = [f"{G.nodes[i]['symbol']}{i}" for i in cycle]

            aromatic_atoms = {"C", "N", "O", "S", "B"}

            if not all(G.nodes[i]["symbol"] in aromatic_atoms for i in cycle):
                non_aromatic = [G.nodes[i]["symbol"] for i in cycle if G.nodes[i]["symbol"] not in aromatic_atoms]
                self.log(f"✗ Contains non-aromatic atoms: {non_aromatic}", 2)
                continue

            is_planar = self._check_planarity(cycle, G)
            if not is_planar:
                self.log(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)
                self.log("✗ Not planar, skipping aromaticity check", 2)
                continue

            for i in cycle:
                G.nodes[i]
                if len(list(G.neighbors(i))) >= 4:
                    self.log(
                        f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}",
                        1,
                    )
                    self.log("✗ Contains sp3 character, skipping aromaticity check", 2)
                    is_planar = False
                    break

            self.log(f"\nRing {ring_idx + 1} ({len(cycle)}-membered): {ring_atoms}", 1)

            # Count π electrons (simplified)
            pi_electrons = 0
            pi_breakdown = []
            contrib, label = 0, None
            for idx in cycle:
                sym = G.nodes[idx]["symbol"]
                fc = G.nodes[idx].get("formal_charge", 0)
                degree = sum(1 for nbr in G.neighbors(idx) if G.nodes[nbr]["symbol"] not in DATA.metals)

                if sym == "C":
                    # Carbon: 1 π electron, adjusted by formal charge
                    contrib = max(0, 1 - fc) if fc > 0 else 1 + abs(fc)
                    label = f"{sym}{idx}:1" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"

                elif sym == "B":
                    # Boron: empty p-orbital contributes 0 (or |fc| if charged)
                    contrib = abs(fc) if fc < 0 else 0
                    label = f"{sym}{idx}:0(empty_p)" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"

                elif sym == "N":
                    # Nitrogen: depends on degree and formal charge
                    if degree == 3:
                        # Pyrrole-like: 2 π (LP) unless charged
                        contrib = 1 if fc > 0 else 2
                        label = f"{sym}{idx}:2(LP)" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"
                    else:  # degree == 2
                        # Pyridine-like: 1 π normally, 2 π if negative
                        contrib = 2 if fc < 0 else 1
                        label = f"{sym}{idx}:1" if fc == 0 else f"{sym}{idx}:{contrib}(fc={fc:+d})"

                elif sym in ("O", "S"):
                    # Oxygen/Sulfur: 2 π (LP) regardless of positive charge
                    contrib = 2
                    label = f"{sym}{idx}:2(LP)" if fc == 0 else f"{sym}{idx}:2(LP,fc={fc:+d})"

                pi_electrons += contrib
                pi_breakdown.append(label)

            self.log(f"π electrons: {pi_electrons} ({', '.join(pi_breakdown)})", 2)

            # Hückel rule: 4n+2 π electrons (n = 0, 1, 2, ...)
            is_aromatic = pi_electrons >= 2 and pi_electrons in (2, 6, 10, 14, 18)

            if is_aromatic:
                n = (pi_electrons - 2) // 4
                self.log(f"✓ AROMATIC (4n+2 rule: n={n})", 2)
                # Set all ring edges to 1.5
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
                self.log("✗ Not aromatic (4n+2 rule violated)", 2)

        self.log(f"\n{'-' * 80}", 0)
        self.log(
            f"SUMMARY: {aromatic_rings} aromatic rings, {aromatic_count} bonds set to 1.5",
            1,
        )
        self.log(f"{'-' * 80}\n", 0)

        return aromatic_count

    def _check_planarity(self, cycle: List[int], G: nx.Graph, threshold: float = 0.15) -> bool:
        """Check if a ring is approximately planar."""
        if len(cycle) < 3:
            return True

        coords = np.array([G.nodes[i]["position"] for i in cycle])

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

    def _get_ligand_unit_info(self, G: nx.Graph, metal_idx: int, start_atom: int) -> Tuple[int, str]:
        """
        Get charge and identity for a ligand unit by following linear chain from start_atom.

        Returns: (charge, ligand_id)
        Handles: CO, CN⁻, SCN⁻, NO, monatomic ligands
        Limitation: Won't traverse rings (like Cp) but those handled separately
        """
        symbols = [G.nodes[start_atom]["symbol"]]
        charge = G.nodes[start_atom].get("formal_charge", 0)
        current = start_atom
        prev = metal_idx

        # Follow linear chain
        while True:
            neighbors = [n for n in G.neighbors(current) if n != prev and G.nodes[n]["symbol"] not in DATA.metals]
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

    def _classify_metal_ligands(self, G: nx.Graph, formal_charges: Optional[List[int]] = None) -> Dict[str, Any]:
        """Infer ligand types and metal oxidation state from formal charges.

        Handles: monatomic (H⁻, Cl⁻), linear chains (CO, CN⁻), rings (Cp⁻).

        Parameters
        ----------
        G : nx.Graph
            Graph with molecular structure.
        formal_charges : list, optional
            List of formal charges (if not stored in nodes yet).

        Returns
        -------
        dict
            Classification with dative_bonds, ionic_bonds, and metal_ox_states.
        """

        # Helper to get formal charge
        def get_fc(atom_idx):
            if formal_charges is not None:
                return formal_charges[atom_idx]
            return G.nodes[atom_idx].get("formal_charge", 0)

        classification: dict[str, Any] = {"dative_bonds": [], "ionic_bonds": [], "metal_ox_states": {}}

        # Get rings (metal-free)
        rings = G.graph.get("_rings", [])

        for metal_idx in G.nodes():
            if G.nodes[metal_idx]["symbol"] not in DATA.metals:
                continue

            ligand_charge_sum = 0
            processed_atoms = set()  # Track atoms already assigned to ligands

            # First pass: detect ring-based ligands (Cp⁻)
            metal_bonded_atoms = [n for n in G.neighbors(metal_idx) if G.nodes[n]["symbol"] not in DATA.metals]

            for ring in rings:
                # Check if entire ring bonds to this metal
                ring_set = set(ring)
                bonded_ring_atoms = [a for a in metal_bonded_atoms if a in ring_set]

                if len(bonded_ring_atoms) >= len(ring) / 2:  # At least half the ring bonded
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
                non_metal_neighbors = [n for n in G.neighbors(donor_atom) if G.nodes[n]["symbol"] not in DATA.metals]

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

    def _compute_gasteiger_charges(self, G: nx.Graph) -> List[float]:
        """Compute Gasteiger charges using RDKit."""
        try:
            rw = Chem.RWMol()
            for i in G.nodes():
                rw.AddAtom(Chem.Atom(G.nodes[i]["symbol"]))

            for i, j, data in G.edges(data=True):
                bo = data["bond_order"]
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
            except Exception:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)

            Chem.AllChem.ComputeGasteigerCharges(mol)  # ty: ignore

            charges = []
            for atom in mol.GetAtoms():
                try:
                    c = float(atom.GetProp("_GasteigerCharge"))
                    if np.isnan(c):
                        c = 0.0
                except Exception:
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
        """Build molecular graph using cheminformatics approach."""
        if self.multiplicity is None:
            total_electrons = sum(self.atomic_numbers) - self.charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2

        # Build initial graph (with inline geometric validation)
        G = self._build_initial_graph()

        self.log(f"Initial bonds: {G.number_of_edges()}", 1)

        # Initialize Kekulé patterns for aromatic rings (gives optimizer a head start)
        self._init_kekule_for_aromatic_rings(G)
        stats = None
        # Valence adjustment
        if self.quick:
            stats = self._quick_valence_adjust(G)
        else:
            if self.optimizer == "greedy":
                stats = self._full_valence_optimize(G)
            if self.optimizer == "beam":
                stats = self._beam_search_optimize(G)

        # Compute formal charges BEFORE aromatic detection
        formal_charges = self._compute_formal_charges(G)

        # Store formal charges in nodes for aromatic detection to use
        for i, fc in enumerate(formal_charges):
            G.nodes[i]["formal_charge"] = fc

        # Aromatic detection (Hückel rule) - now can use formal charges
        self._detect_aromatic_rings(G)

        # Compute charges
        gasteiger_raw = self._compute_gasteiger_charges(G)
        raw_sum = sum(gasteiger_raw)
        delta = (self.charge - raw_sum) / len(self.atoms) if self.atoms else 0.0
        gasteiger_adj = [c + delta for c in gasteiger_raw]

        # Classify metal-ligand bonds
        ligand_classification = self._classify_metal_ligands(G)
        G.graph["ligand_classification"] = ligand_classification

        # Annotate graph
        for node in G.nodes():
            G.nodes[node]["charges"] = {
                "gasteiger_raw": gasteiger_raw[node],
                "gasteiger": gasteiger_adj[node],
            }
            G.nodes[node]["formal_charge"] = formal_charges[node]
            G.nodes[node]["valence"] = self._valence_sum(G, node)

            # Aggregate charge (add H contributions)
            agg = gasteiger_adj[node]
            for nbr in G.neighbors(node):
                if G.nodes[nbr]["symbol"] == "H":
                    agg += gasteiger_adj[nbr]
            G.nodes[node]["agg_charge"] = agg

        # Add bond types
        for i, j, data in G.edges(data=True):
            data["bond_type"] = (G.nodes[i]["symbol"], G.nodes[j]["symbol"])

        G.graph["total_charge"] = self.charge
        G.graph["multiplicity"] = self.multiplicity
        G.graph["valence_stats"] = stats
        G.graph["method"] = "cheminf-quick" if self.quick else "cheminf-full"

        return G

    def _build_xtb(self) -> nx.Graph:
        """Build graph using xTB quantum chemistry calculations."""
        if self.multiplicity is None:
            total_electrons = sum(self.atomic_numbers) - self.charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2

        work = "xtb_tmp_local"
        basename = "xtb"
        if os.system("which xtb > /dev/null 2>&1") != 0:
            raise RuntimeError("xTB not found in PATH - install xTB or use 'cheminf' method")

        os.makedirs(work, exist_ok=True)

        # Write XYZ file natively
        xyz_path = os.path.join(work, f"{basename}.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(self.atoms)}\n")
            f.write("xyzgraph generated XYZ for xTB\n")
            for symbol, (x, y, z) in self.atoms:
                f.write(f"{symbol:>2} {x:15.8f} {y:15.8f} {z:15.8f}\n")

        cmd = (
            f"cd {work} && xtb {basename}.xyz --chrg {self.charge} --uhf {self.multiplicity - 1} --gfn2 "
            f"> {basename}.out"
        )
        ret = os.system(cmd)

        if ret != 0:
            self.log(f"Warning: xTB returned non-zero exit code {ret}", 1)

        # Parse WBO
        bonds = []
        bond_orders = []
        wbo_file = os.path.join(work, f"{basename}_wbo")
        if not os.path.exists(wbo_file) and os.path.exists(os.path.join(work, "wbo")):
            os.rename(os.path.join(work, "wbo"), wbo_file)

        try:
            with open(wbo_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 3 and float(parts[2]) > 0.5:  # bonding threshold
                        bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))  # xTB uses 1-indexed
                        bond_orders.append(float(parts[2]))
            self.log(f"Parsed {len(bonds)} bonds from xTB WBO", 1)
        except FileNotFoundError:
            pass

        # Parse charges
        charges = []
        charges_file = os.path.join(work, f"{basename}_charges")
        if not os.path.exists(charges_file) and os.path.exists(os.path.join(work, "charges")):
            os.rename(os.path.join(work, "charges"), charges_file)

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
            G.add_node(
                i,
                symbol=symbol,
                atomic_number=self.atomic_numbers[i],
                position=pos[i],
                charges={"mulliken": charges[i] if i < len(charges) else 0.0},
            )

        if bonds:
            for (i, j), bo in zip(bonds, bond_orders):
                d = self._distance(pos[i], pos[j])
                si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
                G.add_edge(
                    i,
                    j,
                    bond_order=float(bo),
                    distance=d,
                    bond_type=(si, sj),
                    metal_coord=(si in DATA.metals or sj in DATA.metals),
                )
            self.log(f"Built graph with {G.number_of_edges()} bonds from xTB", 1)
        else:
            # Fallback to distance-based if xTB failed
            self.log(
                "Warning: No xTB bonds found, falling back to distance-based, try using `--method cheminf`",
                1,
            )
            G = self._build_initial_graph()

        # Add derived properties
        for node in G.nodes():
            G.nodes[node]["valence"] = self._valence_sum(G, node)
            agg = G.nodes[node]["charges"].get("mulliken", 0.0)
            for nbr in G.neighbors(node):
                if G.nodes[nbr]["symbol"] == "H":
                    agg += G.nodes[nbr]["charges"].get("mulliken", 0.0)
            G.nodes[node]["agg_charge"] = agg

        G.graph["total_charge"] = self.charge
        G.graph["multiplicity"] = self.multiplicity
        G.graph["method"] = "xtb"

        return G


def build_graph(
    atoms: List[Tuple[str, Tuple[float, float, float]]] | str,
    charge: int = DEFAULT_PARAMS["charge"],
    multiplicity: Optional[int] = DEFAULT_PARAMS["multiplicity"],
    method: str = DEFAULT_PARAMS["method"],
    quick: bool = DEFAULT_PARAMS["quick"],
    optimizer: str = DEFAULT_PARAMS["optimizer"],
    max_iter: int = DEFAULT_PARAMS["max_iter"],
    edge_per_iter: int = DEFAULT_PARAMS["edge_per_iter"],
    beam_width: int = DEFAULT_PARAMS["beam_width"],
    bond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["bond"],
    unbond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["unbond"],
    clean_up: bool = DEFAULT_PARAMS["clean_up"],
    debug: bool = DEFAULT_PARAMS["debug"],
    threshold: float = DEFAULT_PARAMS["threshold"],
    threshold_h_h: float = DEFAULT_PARAMS["threshold_h_h"],
    threshold_h_nonmetal: float = DEFAULT_PARAMS["threshold_h_nonmetal"],
    threshold_h_metal: float = DEFAULT_PARAMS["threshold_h_metal"],
    threshold_metal_ligand: float = DEFAULT_PARAMS["threshold_metal_ligand"],
    threshold_nonmetal_nonmetal: float = DEFAULT_PARAMS["threshold_nonmetal_nonmetal"],
    relaxed: bool = DEFAULT_PARAMS["relaxed"],
    allow_metal_metal_bonds: bool = DEFAULT_PARAMS["allow_metal_metal_bonds"],
    threshold_metal_metal_self: float = DEFAULT_PARAMS["threshold_metal_metal_self"],
    period_scaling_h_bonds: float = DEFAULT_PARAMS["period_scaling_h_bonds"],
    period_scaling_nonmetal_bonds: float = DEFAULT_PARAMS["period_scaling_nonmetal_bonds"],
    metadata: Optional[Dict[str, Any]] = None,
) -> nx.Graph:
    """Build molecular graph using GraphBuilder.

    atoms: Either a list of (symbol, (x,y,z)) tuples, or a filepath to read.
    metadata: Pre-computed metadata dict (for CLI to avoid duplication).
    """
    # Handle filepath input
    if isinstance(atoms, str):
        atoms = read_xyz_file(atoms)

    # Compute metadata if not provided
    if metadata is None:
        metadata = compute_metadata(
            method=method,
            charge=charge,
            multiplicity=multiplicity,
            quick=quick,
            optimizer=optimizer,
            max_iter=max_iter,
            edge_per_iter=edge_per_iter,
            beam_width=beam_width,
            bond=bond,
            unbond=unbond,
            clean_up=clean_up,
            threshold=threshold,
            threshold_h_h=threshold_h_h,
            threshold_h_nonmetal=threshold_h_nonmetal,
            threshold_h_metal=threshold_h_metal,
            threshold_metal_ligand=threshold_metal_ligand,
            threshold_nonmetal_nonmetal=threshold_nonmetal_nonmetal,
            relaxed=relaxed,
            allow_metal_metal_bonds=allow_metal_metal_bonds,
            threshold_metal_metal_self=threshold_metal_metal_self,
            period_scaling_h_bonds=period_scaling_h_bonds,
            period_scaling_nonmetal_bonds=period_scaling_nonmetal_bonds,
        )

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
        threshold_nonmetal_nonmetal=threshold_nonmetal_nonmetal,
        relaxed=relaxed,
        allow_metal_metal_bonds=allow_metal_metal_bonds,
        threshold_metal_metal_self=threshold_metal_metal_self,
        period_scaling_h_bonds=period_scaling_h_bonds,
        period_scaling_nonmetal_bonds=period_scaling_nonmetal_bonds,
    )

    G = builder.build()

    # Add metadata to graph (with version/citation info)
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "parameters": metadata,
    }

    return G


def build_graph_rdkit(
    xyz_file: str | List[Tuple[str, Tuple[float, float, float]]],
    charge: int = DEFAULT_PARAMS["charge"],
    bohr_units: bool = False,
) -> nx.Graph:
    """
    Build molecular graph using RDKit's DetermineBonds algorithm.

    Uses RDKit's distance-based bond perception with Hueckel rule for conjugation.

    Parameters
    ----------
    xyz_file : str or List[Tuple[str, Tuple[float, float, float]]]
        Either path to XYZ file or list of (symbol, (x, y, z)) tuples
    charge : int, default=0
        Total molecular charge
    bohr_units : bool, default=False
        Whether coordinates are in Bohr (only used if xyz_file is a path)

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing:
        - symbol: element symbol
        - atomic_number: atomic number
        - position: (x, y, z) coordinates
        - charges: empty dict (RDKit doesn't compute partial charges)
        - formal_charge: RDKit formal charge
        - valence: sum of bond orders

    Raises
    ------
    ValueError
        If RDKit fails to parse the structure or determine bonds

    Examples
    --------
    >>> from xyzgraph import build_graph_rdkit
    >>> G = build_graph_rdkit("structure.xyz", charge=-1)
    >>> print(f"Graph has {G.number_of_nodes()} atoms and {G.number_of_edges()} bonds")

    Notes
    -----
    RDKit has limited support for coordination complexes. For metal-containing
    systems, consider using build_graph() with method='cheminf' or
    build_graph_from_orca() instead.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    # Handle input
    if isinstance(xyz_file, str):
        atoms = read_xyz_file(xyz_file, bohr_units=bohr_units)
    else:
        atoms = xyz_file

    # Build XYZ block for RDKit
    nat = len(atoms)
    symbols = [symbol for symbol, _ in atoms]
    positions = [pos for _, pos in atoms]
    xyz_lines = [str(nat), f"Generated by xyzgraph build_graph_rdkit (charge={charge})"]
    for sym, (x, y, z) in zip(symbols, positions):
        xyz_lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    xyz_block = "\n".join(xyz_lines) + "\n"

    # Parse with RDKit
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        raise ValueError("RDKit MolFromXYZBlock failed to parse structure")

    # Determine bonds
    mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge, useHueckel=True)
    except Exception as e:
        # Check for metals
        if any(s in DATA.metals for s in symbols):
            raise ValueError(f"RDKit DetermineBonds failed (metal atoms detected): {e}") from e
        raise ValueError(f"RDKit DetermineBonds failed: {e}") from e

    if mol.GetNumBonds() == 0:
        raise ValueError("RDKit DetermineBonds produced no bonds")

    # Light sanitize
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass

    # Build NetworkX graph
    G = nx.Graph()

    # Add nodes
    for a in mol.GetAtoms():
        i = a.GetIdx()
        symbol = a.GetSymbol()
        atomic_number = DATA.s2n.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

        G.add_node(
            i,
            symbol=symbol,
            atomic_number=atomic_number,
            position=positions[i],
            charges={},  # RDKit doesn't compute partial charges
            formal_charge=a.GetFormalCharge(),
            valence=0.0,
        )

    # Add edges
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()

        # Convert RDKit bond type to numeric order
        if b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC:
            bo = 1.5
        elif b.GetBondType() == Chem.BondType.SINGLE:
            bo = 1.0
        elif b.GetBondType() == Chem.BondType.DOUBLE:
            bo = 2.0
        elif b.GetBondType() == Chem.BondType.TRIPLE:
            bo = 3.0
        else:
            bo = 1.0

        # Calculate distance
        pos_i = np.array(positions[i])
        pos_j = np.array(positions[j])
        distance = float(np.linalg.norm(pos_i - pos_j))

        si = G.nodes[i]["symbol"]
        sj = G.nodes[j]["symbol"]

        G.add_edge(
            i,
            j,
            bond_order=bo,
            distance=distance,
            bond_type=(si, sj),
            metal_coord=(si in DATA.metals or sj in DATA.metals),
        )

    # Compute valence
    for node in G.nodes():
        valence = sum(
            G[node][nbr].get("bond_order", 1.0)
            for nbr in G.neighbors(node)
            if G.nodes[nbr]["symbol"] not in DATA.metals
        )
        G.nodes[node]["valence"] = valence

        # Aggregated charge (just formal for RDKit, include H neighbors)
        agg_charge = float(G.nodes[node]["formal_charge"])
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg_charge += G.nodes[nbr]["formal_charge"]
        G.nodes[node]["agg_charge"] = agg_charge

    # Add metadata
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "rdkit",
    }
    G.graph["total_charge"] = charge
    G.graph["method"] = "rdkit"

    return G


def build_graph_rdkit_tm(
    xyz_file: str | list[tuple[str, tuple[float, float, float]]],
    charge: int = DEFAULT_PARAMS["charge"],
    bohr_units: bool = False,
) -> nx.Graph:
    """
    Build molecular graph using xyz2mol_tm.get_tmc_mol (tmQM/coordination complexes).

    This function combines:
    1. XYZ coordinates for atomic positions
    2. Connectivity from xyz2mol_tm (specialized for metal coordination)
    3. Graph matching to align RDKit atom ordering with XYZ ordering

    Strategy for mismatched connectivity:
    - Attempts perfect isomorphism first
    - Falls back to partial matching if graphs differ
    - Uses element + connectivity similarity to find best correspondence
    - Requires sufficient overlap (>75% of edges) to proceed

    Parameters
    ----------
    xyz_file : str or list of (symbol, (x, y, z)) tuples
        Path to an XYZ file or coordinates.
    charge : int
        Total molecular charge.
    bohr_units : bool
        Whether input coordinates are in Bohr (converted to Å if True).

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing symbol, atomic_number, position,
        formal_charge, valence, and charges (empty dict).
    """
    import tempfile

    import networkx as nx
    import numpy as np
    from networkx.algorithms import isomorphism
    from rdkit import Chem

    from . import BOHR_TO_ANGSTROM, DATA

    # Import xyz2mol_tm
    try:
        from xyz2mol_tm import xyz2mol_tmc  # ty: ignore
    except ImportError:
        raise ImportError(
            "xyz2mol_tm not found. Install via:\npip install git+https://github.com/jensengroup/xyz2mol_tm.git"
        ) from None

    # ===== STEP 1: Parse XYZ coordinates =====
    if isinstance(xyz_file, str):
        with open(xyz_file, "r") as f:
            lines = f.readlines()
        nat = int(lines[0].strip())
        lines[1].strip()
        atoms = []
        for line in lines[2 : 2 + nat]:
            parts = line.split()
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            if bohr_units:
                x, y, z = (
                    x * BOHR_TO_ANGSTROM,
                    y * BOHR_TO_ANGSTROM,
                    z * BOHR_TO_ANGSTROM,
                )
            atoms.append((sym, (x, y, z)))
    elif isinstance(xyz_file, list):
        atoms = xyz_file
        if bohr_units:
            atoms = [(s, (x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM, z * BOHR_TO_ANGSTROM)) for s, (x, y, z) in atoms]
    else:
        raise TypeError("xyz_file must be a path or list of (symbol, position) tuples")

    heavy_idx = [i for i, (s, _) in enumerate(atoms) if s != "H"]

    # ===== STEP 2: Get connectivity from xyz2mol_tm =====
    # xyz2mol_tm reads XYZ only from file, so create temp file
    xyz_lines = [str(len(atoms)), "Generated by build_graph_rdkit_tm"]
    xyz_lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in atoms]
    xyz_block = "\n".join(xyz_lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xyz", delete=False) as tmp:
        tmp.write(xyz_block)
        tmp.flush()
        # --- timeout protection around xyz2mol_tmc ---
        import signal

        def handler(signum, frame):
            raise TimeoutError("xyz2mol_tmc took too long")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)  # 5 seconds timeout

        try:
            mol = xyz2mol_tmc.get_tmc_mol(tmp.name, overall_charge=charge)
        except TimeoutError:
            print(f"[Warning] xyz2mol_tmc timed out for {xyz_file}. Skipping RDKit-TM graph.")
            mol = None  # gracefully skip
        except Exception as e:
            print(f"[Warning] xyz2mol_tmc failed for {xyz_file}: {e}")
            mol = None
        finally:
            signal.alarm(0)

    if mol is None:
        # Return a placeholder graph or skip
        import networkx as nx

        G = nx.Graph()
        G.graph["metadata"] = {
            "source": "rdkit_tm",
            "note": "xyz2mol_tmc failed or timed out",
        }
        return G

    # Build RDKit connectivity graph (element + bonds only)
    G_rdkit = nx.Graph()
    for i in range(mol.GetNumAtoms()):
        G_rdkit.add_node(i, symbol=mol.GetAtomWithIdx(i).GetSymbol())

    for bond in mol.GetBonds():
        G_rdkit.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # ===== STEP 3: Build XYZ heavy-atom graph =====
    G_xyz_heavy = build_graph([atoms[i] for i in heavy_idx], charge=charge, quick=True)
    mapping_to_original = {i: heavy_idx[i] for i in range(len(heavy_idx))}
    G_xyz_relabeled = nx.relabel_nodes(G_xyz_heavy, mapping_to_original)

    # ===== STEP 3a: Filter XYZ edges to match RDKit connectivity =====
    allowed_pairs = set()
    for bond in mol.GetBonds():
        sym_i = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
        sym_j = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
        allowed_pairs.add(frozenset([sym_i, sym_j]))

    edges_to_keep = [
        (i, j)
        for i, j in G_xyz_relabeled.edges()
        if frozenset([G_xyz_relabeled.nodes[i]["symbol"], G_xyz_relabeled.nodes[j]["symbol"]]) in allowed_pairs
    ]

    G_xyz_simple = nx.Graph()
    for n in G_xyz_relabeled.nodes():
        G_xyz_simple.add_node(n, symbol=G_xyz_relabeled.nodes[n]["symbol"])
    G_xyz_simple.add_edges_from(edges_to_keep)

    # ===== STEP 4: Match graphs (try perfect first, fall back to partial) =====
    nm = isomorphism.categorical_node_match("symbol", "")
    GM = isomorphism.GraphMatcher(G_rdkit, G_xyz_simple, node_match=nm)

    if GM.is_isomorphic():
        rdkit_to_xyz = GM.mapping
        print("Indexed against xyzgraph by perfect isomorphism.")
    else:
        # Graphs differ - use partial matching
        print("Warning: Graphs not perfectly isomorphic.")
        print(f"  RDKit: {G_rdkit.number_of_nodes()} nodes, {G_rdkit.number_of_edges()} edges")
        print(f"  XYZ:   {G_xyz_simple.number_of_nodes()} nodes, {G_xyz_simple.number_of_edges()} edges")
        print("  Attempting partial matching based on connectivity similarity...")

        rdkit_to_xyz = _partial_graph_matching(G_rdkit, G_xyz_simple)

        # Validate mapping quality
        mapped_edges = 0
        total_rdkit_edges = G_rdkit.number_of_edges()
        for i, j in G_rdkit.edges():
            xyz_i = rdkit_to_xyz.get(i)
            xyz_j = rdkit_to_xyz.get(j)
            if xyz_i and xyz_j and G_xyz_simple.has_edge(xyz_i, xyz_j):
                mapped_edges += 1

        overlap = mapped_edges / total_rdkit_edges if total_rdkit_edges > 0 else 0
        print(f"  Mapping quality: {mapped_edges}/{total_rdkit_edges} edges match ({overlap * 100:.1f}%)")

        if overlap < 0.75:
            raise ValueError(
                f"Insufficient graph overlap ({overlap * 100:.1f}%). "
                f"xyz2mol_tm and geometric methods disagree too much on connectivity."
            )

    # ===== STEP 5: Build final graph with XYZ ordering =====
    G = nx.Graph()
    # Add all atoms (heavy + H) with original XYZ indices
    for idx, (sym, pos) in enumerate(atoms):
        G.add_node(
            idx,
            symbol=sym,
            atomic_number=Chem.GetPeriodicTable().GetAtomicNumber(sym),
            position=pos,
            formal_charge=0,
            valence=0.0,
            charges={},
        )

    # Add heavy-heavy edges from RDKit, mapped to XYZ indices
    for bond in mol.GetBonds():
        i_xyz = rdkit_to_xyz[bond.GetBeginAtomIdx()]
        j_xyz = rdkit_to_xyz[bond.GetEndAtomIdx()]

        bt = bond.GetBondType()
        # Extract bond order from RDKit
        bo = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5,
        }.get(bt, 1.0)

        # Calculate distance from XYZ coordinates
        pos_i = np.array(G.nodes[i_xyz]["position"])
        pos_j = np.array(G.nodes[j_xyz]["position"])

        G.add_edge(
            i_xyz,
            j_xyz,
            bond_order=bo,
            distance=float(np.linalg.norm(pos_i - pos_j)),
            bond_type=(G.nodes[i_xyz]["symbol"], G.nodes[j_xyz]["symbol"]),
            metal_coord=(G.nodes[i_xyz]["symbol"] in DATA.metals or G.nodes[j_xyz]["symbol"] in DATA.metals),
        )

    # Connect hydrogens to nearest heavy atom (geometrically)
    for idx, (sym, pos) in enumerate(atoms):
        if sym == "H":
            pos_arr = np.array(pos)
            dists = [np.linalg.norm(pos_arr - np.array(G.nodes[i]["position"])) for i in heavy_idx]
            nearest = heavy_idx[int(np.argmin(dists))]
            G.add_edge(
                idx,
                nearest,
                bond_order=1.0,
                distance=float(min(dists)),
                bond_type=("H", G.nodes[nearest]["symbol"]),
                metal_coord=(G.nodes[nearest]["symbol"] in DATA.metals),
            )

    # --- Update valences and formal charges ---
    for node in G.nodes():
        G.nodes[node]["valence"] = sum(G.edges[node, nbr]["bond_order"] for nbr in G.neighbors(node))

    for rdkit_idx, xyz_idx in rdkit_to_xyz.items():
        G.nodes[xyz_idx]["formal_charge"] = mol.GetAtomWithIdx(rdkit_idx).GetFormalCharge()

    # --- Metadata ---
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "rdkit_tm",
    }
    G.graph["total_charge"] = charge
    G.graph["method"] = "rdkit_tm"

    return G


def _partial_graph_matching(G_rdkit: nx.Graph, G_xyz: nx.Graph) -> dict:
    """
    Graph-distance + neighbor-symbol similarity based partial matching for non-isomorphic graphs.

    Parameters
    ----------
    G_rdkit : nx.Graph
        RDKit molecular graph (nodes with 'symbol')
    G_xyz : nx.Graph
        XYZ-based molecular graph (nodes with 'symbol')

    Returns
    -------
    dict
        Mapping {rdkit_node -> xyz_node}
    """
    from collections import defaultdict

    import networkx as nx
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment  # ty: ignore
    except ImportError:
        raise ImportError("scipy not found. Install via:\npip install scipy") from None

    print("  Starting graph-distance + neighbor-symbol partial matching...")

    # Group nodes by element
    rdkit_by_elem = defaultdict(list)
    xyz_by_elem = defaultdict(list)
    for n in G_rdkit.nodes():
        rdkit_by_elem[G_rdkit.nodes[n]["symbol"]].append(n)
    for n in G_xyz.nodes():
        xyz_by_elem[G_xyz.nodes[n]["symbol"]].append(n)

    # Check element counts
    for elem in rdkit_by_elem:
        rdkit_count = len(rdkit_by_elem[elem])
        xyz_count = len(xyz_by_elem.get(elem, []))
        if rdkit_count != xyz_count:
            raise ValueError(
                f"Cannot perform partial matching: element '{elem}' count mismatch. RDKit has {rdkit_count}, "
                f"XYZ has {xyz_count}. This could be bimetallic and not handled by xyz2mol_tm."
            )

    # Compute shortest-path distance matrices
    print("   Computing all-pairs shortest-path distance matrices...")
    D_rdkit = np.asarray(nx.floyd_warshall_numpy(G_rdkit))
    D_xyz = np.asarray(nx.floyd_warshall_numpy(G_xyz))

    rdkit_nodes = list(G_rdkit.nodes())
    xyz_nodes = list(G_xyz.nodes())
    rdkit_index = {n: i for i, n in enumerate(rdkit_nodes)}
    xyz_index = {n: i for i, n in enumerate(xyz_nodes)}

    rdkit_to_xyz = {}

    # Match nodes per element
    for elem, rdkit_list in rdkit_by_elem.items():
        if elem not in xyz_by_elem:
            raise ValueError(f"Element {elem} in RDKit but not in XYZ")
        xyz_list = xyz_by_elem[elem]

        n_r, n_x = len(rdkit_list), len(xyz_list)
        min_count = min(n_r, n_x)
        if n_r != n_x:
            print(f"   Warning: Element {elem} count mismatch: RDKit={n_r}, XYZ={n_x}. Matching {min_count} atoms.")

        # Build score matrix
        scores = np.zeros((n_r, n_x))
        for i, r_node in enumerate(rdkit_list):
            d_r = D_rdkit[rdkit_index[r_node], :]
            r_neighs = set(G_rdkit.neighbors(r_node))
            r_symbols = {G_rdkit.nodes[n]["symbol"] for n in r_neighs}

            for j, x_node in enumerate(xyz_list):
                d_x = D_xyz[xyz_index[x_node], :]
                x_neighs = set(G_xyz.neighbors(x_node))
                x_symbols = {G_xyz.nodes[n]["symbol"] for n in x_neighs}

                # 1) Graph-distance similarity
                dist_diff = np.sum(np.abs(d_r - d_x))
                score = -dist_diff  # negative = more similar

                # 2) Neighbor symbol overlap bonus
                common_symbols = len(r_symbols & x_symbols)
                score += common_symbols * 5  # adjust weight if needed

                scores[i, j] = score

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-scores)
        for i, j in zip(row_ind[:min_count], col_ind[:min_count]):
            r_node = rdkit_list[i]
            x_node = xyz_list[j]
            score = scores[i, j]
            rdkit_to_xyz[r_node] = x_node
            print(f"   Matched {r_node} → {x_node} (score={score:.2f})")

    print(f"Finished partial matching. {len(rdkit_to_xyz)} atoms mapped.")
    return rdkit_to_xyz


def build_graph_orca(
    orca_file: str,
    bond_threshold: float = DEFAULT_PARAMS["orca_bond_threshold"],
    debug: bool = DEFAULT_PARAMS["debug"],
) -> nx.Graph:
    """
    Build molecular graph from ORCA quantum chemistry output file.

    Uses Mayer bond orders and Mulliken charges from ORCA calculations.
    Coordinates, charge, and multiplicity are read from the output file.

    Parameters
    ----------
    orca_file : str
        Path to ORCA output file
    bond_threshold : float, default=0.5
        Minimum Mayer bond order to consider as a bond
    debug : bool, default=False
        Enable debug logging

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing:
        - symbol: element symbol
        - atomic_number: atomic number
        - position: (x, y, z) coordinates in Angstrom
        - charges: dict with 'mulliken' key
        - formal_charge: computed formal charge
        - valence: sum of bond orders
        - agg_charge: aggregated charge (including H neighbors)

    Raises
    ------
    OrcaParseError
        If ORCA output cannot be parsed or required data is missing

    Examples
    --------
    >>> from xyzgraph import build_graph_from_orca
    >>> G = build_graph_from_orca("structure.out")
    >>> print(f"Graph has {G.number_of_nodes()} atoms and {G.number_of_edges()} bonds")
    """
    from .orca_parser import OrcaParseError, parse_orca_output

    # Parse ORCA output
    try:
        orca_data = parse_orca_output(orca_file)
    except OrcaParseError as e:
        raise OrcaParseError(f"Failed to parse ORCA output: {e}") from e

    atoms = orca_data["atoms"]
    bonds = orca_data["bonds"]
    mulliken_charges = orca_data["charges"]
    charge = orca_data["charge"]
    multiplicity = orca_data["multiplicity"]

    if debug:
        print(f"Parsed ORCA output: {len(atoms)} atoms, {len(bonds)} bonds (before threshold)")
        print(f"Charge: {charge}, Multiplicity: {multiplicity}")

    # Build graph
    G = nx.Graph()

    # Add nodes
    for i, (symbol, pos) in enumerate(atoms):
        atomic_number = DATA.s2n.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

        G.add_node(
            i,
            symbol=symbol,
            atomic_number=atomic_number,
            position=pos,
            charges={"mulliken": mulliken_charges[i]},
        )

    # Add edges (filter by threshold)
    bonds_added = 0
    for i, j, mayer_bo in bonds:
        if mayer_bo >= bond_threshold:
            # Calculate distance
            pos_i = np.array(atoms[i][1])
            pos_j = np.array(atoms[j][1])
            distance = float(np.linalg.norm(pos_i - pos_j))

            si = G.nodes[i]["symbol"]
            sj = G.nodes[j]["symbol"]

            G.add_edge(
                i,
                j,
                bond_order=float(mayer_bo),
                distance=distance,
                bond_type=(si, sj),
                metal_coord=(si in DATA.metals or sj in DATA.metals),
            )
            bonds_added += 1

    if debug:
        print(f"Added {bonds_added} bonds (threshold={bond_threshold})")

    # Compute derived properties
    for node in G.nodes():
        # Valence (sum of bond orders, excluding metal bonds for consistency)
        valence = sum(
            G[node][nbr].get("bond_order", 1.0)
            for nbr in G.neighbors(node)
            if G.nodes[nbr]["symbol"] not in DATA.metals
        )
        G.nodes[node]["valence"] = valence

        # Compute formal charge using existing logic
        sym = G.nodes[node]["symbol"]
        if sym in DATA.metals:
            formal_charge = 0
        else:
            V = DATA.electrons.get(sym, 0)
            if V == 0:
                formal_charge = 0
            # Use simple formal charge calculation
            elif sym == "H":
                formal_charge = int(V - valence)
            else:
                B = 2 * valence
                target = 8
                L = max(0, target - B)
                formal_charge = round(V - L - B / 2)

        G.nodes[node]["formal_charge"] = formal_charge

        # Aggregated charge (include H neighbors like other methods)
        agg_charge = mulliken_charges[node]
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg_charge += mulliken_charges[nbr]
        G.nodes[node]["agg_charge"] = agg_charge

    # Add graph metadata
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "orca",
        "source_file": orca_file,
        "bond_threshold": bond_threshold,
    }
    G.graph["total_charge"] = charge
    G.graph["multiplicity"] = multiplicity
    G.graph["method"] = "orca"

    if debug:
        print(f"\nFinal graph: {G.number_of_nodes()} atoms, {G.number_of_edges()} bonds")

    return G
