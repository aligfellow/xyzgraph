"""Geometric checks for bond validity.

Filters spurious bonds based on acute angles, ring closure geometry,
agostic bond filtering, collinearity checks, and diagonal detection.
"""

import logging
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .data_loader import DATA, MolecularData
from .geometry import GeometryCalculator
from .parameters import GeometryThresholds

logger = logging.getLogger(__name__)


class BondGeometryChecker:
    """Checks whether a proposed bond is geometrically valid.

    Uses GeometryCalculator for pure math and GeometryThresholds
    for all configurable parameters (no magic numbers).
    """

    # Indentation to nest under GraphBuilder's "Evaluating bond" (level 5 = 10 spaces)
    LOG_INDENT = "  " * 5

    def __init__(
        self,
        geometry: GeometryCalculator,
        thresholds: GeometryThresholds,
        data: MolecularData,
    ):
        self.geometry = geometry
        self.thresholds = thresholds
        self.data = data

    def _log(self, msg: str, *args):
        """Log with indentation matching the calling context."""
        logger.debug(self.LOG_INDENT + msg, *args)

    def _calculate_angle(self, atom1: int, center: int, atom2: int, G: nx.Graph) -> float:
        """Calculate angle (degrees) between three atoms: atom1-center-atom2."""
        pos1 = G.nodes[atom1]["position"]
        pos_center = G.nodes[center]["position"]
        pos2 = G.nodes[atom2]["position"]
        return self.geometry.angle(pos1, pos_center, pos2)

    def check(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        distance: float,
        confidence: float,
        baseline_bonds: Optional[List[Tuple[float, int, int, float, bool]]] = None,
    ) -> bool:
        """Check if adding bond i-j creates geometrically valid configuration.

        Used for low-confidence (long) bonds from extended thresholds.

        Parameters
        ----------
        G : nx.Graph
            Current molecular graph
        i, j : int
            Atom indices for the proposed bond
        distance : float
            Distance between atoms i and j
        confidence : float
            Bond confidence score (0.0 = at threshold, 1.0 = very short).
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
        has_metal = is_metal_i or is_metal_j

        # Agostic H-M / F-M bond filtering: reject weak H-M or F-M bonds
        if has_metal and baseline_bonds is not None:
            if self._check_agostic_rejection(G, i, j, sym_i, sym_j, confidence, baseline_bonds):
                return False

        # Use thresholds from config
        t = self.thresholds
        relaxed = not t.apply_z_adjustment  # relaxed mode has no Z-adjustment

        # 4-ring closure check for low-confidence non-metal bonds
        if confidence < t.confidence_threshold and not has_metal and baseline_bonds is not None:
            if self._check_4ring_rejection(G, i, j, sym_i, sym_j, confidence, baseline_bonds):
                return False

        # Check angles at both atoms
        if not self._check_angles_at_atom(G, i, j, sym_i, sym_j, has_metal):
            return False
        if not self._check_angles_at_atom(G, j, i, sym_i, sym_j, has_metal):
            return False

        # Check diagonal in existing rings
        if not self._check_ring_diagonals(G, i, j, sym_i, sym_j, has_metal):
            return False

        # Check 3-ring formation via common neighbors
        if not self._check_common_neighbor_rings(
            G,
            i,
            j,
            sym_i,
            sym_j,
            is_metal_i,
            is_metal_j,
            has_metal,
            distance,
            confidence,
            baseline_bonds,
            relaxed,
        ):
            return False

        return True

    def _check_agostic_rejection(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        sym_i: str,
        sym_j: str,
        confidence: float,
        baseline_bonds: List[Tuple[float, int, int, float, bool]],
    ) -> bool:
        """Return True if bond should be rejected due to agostic filtering."""
        nonmetal_atom = None
        nonmetal_sym = None
        if sym_i in ("H", "F"):
            nonmetal_atom = i
            nonmetal_sym = sym_i
        elif sym_j in ("H", "F"):
            nonmetal_atom = j
            nonmetal_sym = sym_j

        if nonmetal_atom is None:
            return False

        for X_atom in G.neighbors(nonmetal_atom):
            X_sym = G.nodes[X_atom]["symbol"]
            if X_sym in self.data.metals or X_sym == "H":
                continue

            for conf, bi, bj, _, _ in baseline_bonds:
                if nonmetal_atom in (bi, bj) and X_atom in (bi, bj):
                    if conf / max(confidence, 0.01) > 2.0:
                        self._log(
                            "Rejected %s-M agostic: %s-X bond stronger (conf=%.2f vs %.2f)",
                            nonmetal_sym,
                            nonmetal_sym,
                            conf,
                            confidence,
                        )
                        return True
                    break
        return False

    def _check_4ring_rejection(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        sym_i: str,
        sym_j: str,
        confidence: float,
        baseline_bonds: List[Tuple[float, int, int, float, bool]],
    ) -> bool:
        """Return True if bond should be rejected due to weak 4-ring closure."""
        t = self.thresholds
        neighbors_i = set(G.neighbors(i))
        neighbors_j = set(G.neighbors(j))

        forms_4ring = any(G.has_edge(ni, nj) for ni in neighbors_i for nj in neighbors_j if ni != nj)
        if not forms_4ring:
            return False

        for atom in [i, j]:
            atom_sym = G.nodes[atom]["symbol"]
            if atom_sym not in DATA.valences:
                continue

            current_val = sum(
                G[atom][nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(atom)
                if G.nodes[nbr]["symbol"] not in self.data.metals
            )
            max_val = max(DATA.valences[atom_sym])

            if current_val + 1.0 > max_val:
                all_bonds_stronger = all(
                    conf_baseline / max(confidence, 0.001) > t.strength_ratio
                    for conf_baseline, bi, bj, _, _ in baseline_bonds
                    if atom in (bi, bj)
                )

                if all_bonds_stronger:
                    self._log(
                        "Rejected bond %s%d-%s%d: weak 4-ring closure (conf=%.2f), ALL existing bonds stronger",
                        sym_i,
                        i,
                        sym_j,
                        j,
                        confidence,
                    )
                    return True
        return False

    def _check_angles_at_atom(
        self,
        G: nx.Graph,
        center: int,
        other: int,
        sym_i: str,
        sym_j: str,
        has_metal: bool,
    ) -> bool:
        """Check angle constraints at center atom. Return False if bond rejected."""
        t = self.thresholds

        for existing_neighbor in G.neighbors(center):
            angle = self._calculate_angle(existing_neighbor, center, other, G)

            acute_threshold = t.acute_threshold_metal if has_metal else t.acute_threshold_nonmetal

            if angle < acute_threshold:
                self._log(
                    "Rejected bond %s%d-%s%d: angle too acute (%.1f, threshold=%.1f) with %d-%d",
                    sym_i,
                    center,
                    sym_j,
                    other,
                    angle,
                    acute_threshold,
                    existing_neighbor,
                    center,
                )
                return False

            # Nearly collinear
            if angle > t.collinearity_angle:
                if has_metal:
                    self._log(
                        "Bond %d-%d: collinear (%.1f) with %d-%d, involves metal (%s-%s) - allowed",
                        center,
                        other,
                        angle,
                        existing_neighbor,
                        center,
                        sym_i,
                        sym_j,
                    )
                    continue

                if G.degree(center) >= 2:
                    pos_center = np.array(G.nodes[center]["position"])
                    pos_existing = np.array(G.nodes[existing_neighbor]["position"])
                    pos_new = np.array(G.nodes[other]["position"])

                    v_existing = pos_existing - pos_center
                    v_new = pos_new - pos_center

                    v_existing = v_existing / np.linalg.norm(v_existing)
                    v_new = v_new / np.linalg.norm(v_new)

                    dot_product = np.dot(v_existing, v_new)

                    if dot_product > t.collinearity_dot_threshold:
                        self._log(
                            "Rejected bond %s%d-%s%d: collinear (%.1f) same direction as %d-%d",
                            sym_i,
                            center,
                            sym_j,
                            other,
                            angle,
                            existing_neighbor,
                            center,
                        )
                        return False
                    elif dot_product < -t.collinearity_dot_threshold:
                        self._log(
                            "Bond %s%d-%s%d: collinear (%.1f) opposite direction to %d-%d - valid trans",
                            sym_i,
                            center,
                            sym_j,
                            other,
                            angle,
                            existing_neighbor,
                            center,
                        )
                        continue
                    else:
                        continue

        return True

    def _check_ring_diagonals(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        sym_i: str,
        sym_j: str,
        has_metal: bool,
    ) -> bool:
        """Check if bond would create diagonal in existing ring. Return False if rejected."""
        current_rings = G.graph.get("_rings", [])
        for ring in current_rings:
            ring_set = set(ring)
            if i not in ring_set or j not in ring_set:
                continue

            # Cluster bypass: homogeneous inorganic cluster
            ring_elements = {G.nodes[node]["symbol"] for node in ring}
            if len(ring_elements) == 1 and next(iter(ring_elements)) not in {"C", "H"}:
                elem = next(iter(ring_elements))
                elem_count = G.graph.get("_element_counts", {}).get(elem, 0)
                if elem_count >= 8:
                    self._log(
                        "Bond %s%d-%s%d: diagonal in homogeneous %s cluster ring - allowed",
                        sym_i,
                        i,
                        sym_j,
                        j,
                        elem,
                    )
                    continue

            if len(ring) <= 4 and has_metal:
                self._log(
                    "Bond %s%d-%s%d: diagonal in existing %d-ring involves metal - allowed",
                    sym_i,
                    i,
                    sym_j,
                    j,
                    len(ring),
                )
                continue

            self._log(
                "Rejected bond %s%d-%s%d: would create diagonal in existing %d-ring",
                sym_i,
                i,
                sym_j,
                j,
                len(ring),
            )
            return False

        return True

    def _check_common_neighbor_rings(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        sym_i: str,
        sym_j: str,
        is_metal_i: bool,
        is_metal_j: bool,
        has_metal: bool,
        distance: float,
        confidence: float,
        baseline_bonds: Optional[List[Tuple[float, int, int, float, bool]]],
        relaxed: bool,
    ) -> bool:
        """Check 3-ring formation via common neighbors. Return False if rejected."""
        t = self.thresholds
        common_neighbors = set(G.neighbors(i)) & set(G.neighbors(j))
        if not common_neighbors:
            return True

        for k in common_neighbors:
            sym_k = G.nodes[k]["symbol"]

            # Cluster bypass
            ring_elements = {sym_i, sym_j, sym_k}
            if len(ring_elements) == 1 and next(iter(ring_elements)) not in {"C", "H"}:
                elem = next(iter(ring_elements))
                elem_count = G.graph.get("_element_counts", {}).get(elem, 0)
                if elem_count >= 8:
                    self._log(
                        "Bond %s%d-%s%d: 3-ring in homogeneous %s cluster - bypassing",
                        sym_i,
                        i,
                        sym_j,
                        j,
                        elem,
                    )
                    continue

            # 3-ring validation
            is_metal_k = sym_k in self.data.metals
            if is_metal_k:
                if "H" not in (sym_i, sym_j):
                    self._log(
                        "3-ring formation via %s%d involves metal, low confidence L-L, rejected",
                        sym_k,
                        k,
                    )
                    return False

            # M-L bond priority check
            has_metal_in_bond = is_metal_i or is_metal_j

            if has_metal_in_bond and baseline_bonds is not None:
                metal_atom = i if is_metal_i else j

                for conf, bi, bj, _, _ in baseline_bonds:
                    if metal_atom in (bi, bj) and k in (bi, bj):
                        if "H" in (sym_i, sym_j, sym_k):
                            if conf / max(confidence, 0.01) > 1.5:
                                self._log(
                                    "Rejected bond %s%d-%s%d: 3-ring via %s%d, "
                                    "existing M-%s%d bond stronger (conf=%.2f vs %.2f)",
                                    sym_i,
                                    i,
                                    sym_j,
                                    j,
                                    sym_k,
                                    k,
                                    sym_k,
                                    k,
                                    conf,
                                    confidence,
                                )
                                return False
                        elif conf / max(confidence, 0.01) > 3.0:
                            self._log(
                                "Rejected bond %s%d-%s%d: 3-ring diagonal, existing M-%s%d "
                                "bond much stronger (conf=%.2f vs %.2f)",
                                sym_i,
                                i,
                                sym_j,
                                j,
                                sym_k,
                                k,
                                conf,
                                confidence,
                            )
                            return False

            # Angle check
            angle_i = self._calculate_angle(k, i, j, G)
            angle_j = self._calculate_angle(k, j, i, G)
            angle_k = self._calculate_angle(i, k, j, G)
            max_angle = max(angle_i, angle_j, angle_k)

            has_H_in_ring = "H" in (sym_i, sym_j, sym_k)
            has_metal_in_ring = any(s in self.data.metals for s in (sym_i, sym_j, sym_k))

            if has_metal_in_ring:
                angle_threshold = 135.0 if relaxed else 115.0
                ring_type = "metal-containing"
            elif has_H_in_ring:
                angle_threshold = t.angle_threshold_h_ring
                ring_type = "H-containing"
            else:
                z_list = [
                    G.nodes[i]["atomic_number"],
                    G.nodes[j]["atomic_number"],
                    G.nodes[k]["atomic_number"],
                ]
                avg_z = sum(min(z, 18) for z in z_list) / 3.0
                if t.apply_z_adjustment:
                    angle_threshold = t.angle_threshold_base + (avg_z - 6) * 2.0
                else:
                    angle_threshold = t.angle_threshold_base
                ring_type = "non-H"

            if max_angle > angle_threshold:
                self._log(
                    "Rejected bond %s%d-%s%d: 3-ring angle %.1f > %.1f (%s)",
                    sym_i,
                    i,
                    sym_j,
                    j,
                    max_angle,
                    angle_threshold,
                    ring_type,
                )
                return False

            # Distance ratio check (diagonal detection)
            if not self._check_diagonal_ratio(
                G,
                i,
                j,
                k,
                sym_i,
                sym_j,
                sym_k,
                distance,
                confidence,
                has_metal,
                has_H_in_ring,
            ):
                return False

            # Valence check
            if not self._check_3ring_valence(
                G,
                i,
                j,
                sym_i,
                sym_j,
                has_metal,
                has_H_in_ring,
                relaxed,
            ):
                return False

        return True

    def _check_diagonal_ratio(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        k: int,
        sym_i: str,
        sym_j: str,
        sym_k: str,
        distance: float,
        confidence: float,
        has_metal: bool,
        has_H_in_ring: bool,
    ) -> bool:
        """Check diagonal ratio for 3-ring. Return False if rejected."""
        t = self.thresholds

        d_ik = G[i][k]["distance"]
        d_kj = G[k][j]["distance"]
        d_ij = distance

        norm_ik = d_ik / (DATA.vdw[sym_i] + DATA.vdw[sym_k])
        norm_kj = d_kj / (DATA.vdw[sym_k] + DATA.vdw[sym_j])
        norm_ij = d_ij / (DATA.vdw[sym_i] + DATA.vdw[sym_j])

        norm_path = norm_ik + norm_kj
        ratio = norm_ij / norm_path

        if ratio <= t.diagonal_ratio_initial:
            return True

        max_conf_for_interp = 0.7
        diagonal_threshold = (
            t.diagonal_ratio_initial
            + min(confidence, max_conf_for_interp)
            * (t.diagonal_ratio_max - t.diagonal_ratio_initial)
            / max_conf_for_interp
        )

        self._log(
            "3-ring via %s%d: ratio=%.3f, threshold=%.3f",
            sym_k,
            k,
            ratio,
            diagonal_threshold,
        )

        if ratio <= diagonal_threshold:
            return True

        if has_metal and not has_H_in_ring:
            self._log(
                "Bond %s%d-%s%d: diagonal (ratio=%.2f) across 3-ring via %s%d, metal bond - allowed",
                sym_i,
                i,
                sym_j,
                j,
                ratio,
                sym_k,
                k,
            )
            return True

        # Valence fallback check
        atoms_at_limit = 0
        for atom in [i, j]:
            atom_sym = G.nodes[atom]["symbol"]

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
            self._log(
                "Rejected bond %s%d-%s%d: diagonal across 3-ring via %s%d "
                "(ratio=%.2f, threshold=%.2f) and both atoms at valence limit",
                sym_i,
                i,
                sym_j,
                j,
                sym_k,
                k,
                ratio,
                diagonal_threshold,
            )
            return False

        if ratio > t.diagonal_ratio_hard:
            self._log(
                "Rejected bond %s%d-%s%d: diagonal ratio too high (ratio=%.2f > %.2f) even with valence capacity",
                sym_i,
                i,
                sym_j,
                j,
                ratio,
                t.diagonal_ratio_hard,
            )
            return False

        # Check for hypervalent carbon in-plane approach
        for atom in [i, j]:
            if G.nodes[atom]["symbol"] != "C" or G.degree(atom) <= 3:
                continue

            other = j if atom == i else i
            neighbors = list(G.neighbors(atom))[:3]

            pos_C = np.array(G.nodes[atom]["position"])
            vec_new = np.array(G.nodes[other]["position"]) - pos_C
            vec_nb = [np.array(G.nodes[n]["position"]) - pos_C for n in neighbors]

            normal = np.cross(vec_nb[0], vec_nb[1])
            norm_mag = np.linalg.norm(normal)

            if norm_mag < 1e-6:
                continue

            normal /= norm_mag
            vec_new /= np.linalg.norm(vec_new)

            angle_to_normal = np.arccos(np.clip(np.abs(np.dot(vec_new, normal)), 0, 1)) * 180 / np.pi

            if angle_to_normal < 60:
                self._log(
                    "Rejected bond %s%d-%s%d: C hypervalent but in-plane (angle to normal=%.1f, need >60)",
                    sym_i,
                    i,
                    sym_j,
                    j,
                    angle_to_normal,
                )
                return False

        self._log(
            "Bond %s%d-%s%d: suspicious ratio (%.2f) but valence allows - likely real 3-ring",
            sym_i,
            i,
            sym_j,
            j,
            ratio,
        )
        return True

    def _check_3ring_valence(
        self,
        G: nx.Graph,
        i: int,
        j: int,
        sym_i: str,
        sym_j: str,
        has_metal: bool,
        has_H_in_ring: bool,
        relaxed: bool,
    ) -> bool:
        """Valence check for 3-ring bonding atoms. Return False if rejected."""
        atoms_at_limit = 0
        for atom in [i, j]:
            atom_sym = G.nodes[atom]["symbol"]

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

        if atoms_at_limit <= 1:
            return True

        if relaxed:
            overflow_ok = True
            for atom in [i, j]:
                atom_sym = G.nodes[atom]["symbol"]
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
                if overflow > 1.0:
                    overflow_ok = False
                    break

            if overflow_ok:
                self._log(
                    "Bond %s%d-%s%d: both atoms exceed valence but overflow <=1.0 - allowed in relaxed mode",
                    sym_i,
                    i,
                    sym_j,
                    j,
                )
                return True

            self._log(
                "Rejected bond %s%d-%s%d: both bonding atoms exceed valence by >1.0 (even in relaxed mode)",
                sym_i,
                i,
                sym_j,
                j,
            )
            return False

        self._log(
            "Rejected bond %s%d-%s%d: both bonding atoms would exceed valence",
            sym_i,
            i,
            sym_j,
            j,
        )
        return False
