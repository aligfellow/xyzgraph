"""Geometric checks for bond validity.

Filters spurious bonds based on acute angles, ring closure geometry,
agostic bond filtering, collinearity checks, and diagonal detection.
"""

import logging
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .data_loader import MolecularData
from .geometry import GeometryCalculator
from .parameters import GeometryThresholds

logger = logging.getLogger(__name__)


def has_lone_pair(G: nx.Graph, a: int, data: MolecularData) -> bool:
    """Test whether non-metal ``a`` keeps a lone pair: valence electrons exceeding its sigma bonds to non-metals by two.

    A bond to a metal is not counted, so an amido N on a metal still has the pair it could give a proton.
    """
    sigma = sum(1 for n in G.neighbors(a) if G.nodes[n]["symbol"] not in data.metals)
    return data.electrons.get(G.nodes[a]["symbol"], 0) - sigma >= 2


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

        # Agostic H-M / F-M bond filtering: reject weak H-M or F-M bonds. An equilibrium convention: in a
        # transition state the H may be in flight to the metal.
        if has_metal and baseline_bonds is not None and not self.thresholds.transition_state:
            if self._check_agostic_rejection(G, i, j, sym_i, sym_j, confidence, baseline_bonds):
                return False

        # Use thresholds from config
        t = self.thresholds
        relaxed = t.transition_state

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
        """Return True if bond should be rejected due to agostic filtering.

        A C-H near a metal is agostic, not bonded to it. An H on an atom less electronegative than H
        (B-H, Si-H) is hydridic and bridges the metal as a hydride (M-H-B), so it is kept.
        """
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
            if nonmetal_sym == "H" and self.data.electronegativity.get(X_sym, 2.5) < self.data.electronegativity["H"]:
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
            if self._valence_overflow(G, atom) > 0:
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

                # angle > 160° means vectors are opposite (cos > 160° < -0.94),
                # so this is always a valid trans arrangement.
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

            # A ligand bond closing a triangle through a metal (eta2-S2, a P-P ring) is real when it is short
            # against the two metal legs; a 1,3 contact between two donors of the metal is not.
            if sym_k in self.data.metals and not has_metal and "H" not in (sym_i, sym_j):
                ratio = self._ring_ratio(G, i, j, k, distance)
                if ratio > t.diagonal_ratio_initial:
                    self._log(
                        "Rejected bond %s%d-%s%d: 3-ring via %s%d, ratio %.2f > %.2f",
                        sym_i,
                        i,
                        sym_j,
                        j,
                        sym_k,
                        k,
                        ratio,
                        t.diagonal_ratio_initial,
                    )
                    return False

            # M-L bond priority check
            has_metal_in_bond = is_metal_i or is_metal_j

            # A hydridic H (on B or Si) bridges the metal itself: its M-H does not yield to an M-B contact.
            partner = j if is_metal_i else i
            bridging_hydride = (
                G.nodes[partner]["symbol"] == "H"
                and self.data.electronegativity.get(sym_k, 2.5) < self.data.electronegativity["H"]
            )
            if has_metal_in_bond and baseline_bonds is not None and not bridging_hydride:
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
                angle_threshold = t.angle_threshold_base
                ring_type = "non-H"

            if max_angle >= angle_threshold:
                self._log(
                    "Rejected bond %s%d-%s%d: 3-ring angle %.1f >= %.1f (%s)",
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
            if not self._check_3ring_valence(G, i, j, sym_i, sym_j, relaxed):
                return False

        return True

    def _ring_ratio(self, G: nx.Graph, i: int, j: int, k: int, distance: float) -> float:
        """Length of i-j over the i-k-j path, each normalised by its vdW sum (a bond is short against the path)."""
        vdw = {n: self.data.vdw[G.nodes[n]["symbol"]] for n in (i, j, k)}
        path = G[i][k]["distance"] / (vdw[i] + vdw[k]) + G[k][j]["distance"] / (vdw[k] + vdw[j])
        return distance / (vdw[i] + vdw[j]) / path

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
        """Check diagonal ratio for 3-ring. Return False if rejected.

        A real 3-ring bond is short against the path around it (cyclopropane, an epoxide: 0.5); a
        square's diagonal reaches 0.71. Outside a transition state a non-metal closure must meet
        diagonal_ratio_initial outright; the confidence and valence allowances below are for the
        stretched rings of a transition state (and metal bonds, judged on the finished graph).
        """
        t = self.thresholds
        ratio = self._ring_ratio(G, i, j, k, distance)

        if ratio <= t.diagonal_ratio_initial:
            return True
        if not t.transition_state and not has_metal:
            self._log(
                "Rejected bond %s%d-%s%d: 3-ring via %s%d, ratio %.2f > %.2f",
                sym_i,
                i,
                sym_j,
                j,
                sym_k,
                k,
                ratio,
                t.diagonal_ratio_initial,
            )
            return False

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
        if all(self._valence_overflow(G, atom) > 0 for atom in (i, j)):
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

    def _valence_overflow(self, G: nx.Graph, atom: int) -> float:
        """How far one more bond takes a non-metal past its highest valence (> 0: over the limit).

        A metal's oxidation states do not cap its coordination, so a metal is never at a limit.
        """
        sym = G.nodes[atom]["symbol"]
        if sym in self.data.metals or sym not in self.data.valences:
            return float("-inf")
        bonds = sum(
            G[atom][nbr].get("bond_order", 1.0)
            for nbr in G.neighbors(atom)
            if G.nodes[nbr]["symbol"] not in self.data.metals
        )
        return bonds + 1.0 - max(self.data.valences[sym])

    def metal_bond_blocked(self, G: nx.Graph, m: int, x: int) -> Optional[str]:
        """Why metal ``m`` cannot bond non-metal ``x``, judged on the finished graph; None if it can.

        A metal has no valence limit (its oxidation states do not cap its coordination), so the bond
        is judged on x and its partners, the atoms bonded to both x and m:

        - an x less electronegative than H, or saturated (four sigma bonds, no lone pair), reaches m
          through its H partner when it has one: a hydridic B-H or Si-H bridges the metal (B-H-M),
          an agostic C-H touches it, and the H is the bond, not M-x;
        - a saturated x (four sigma bonds, no lone pair) bonds m only side-on through one partner
          (the B-B of a diborane); lying beyond its partners, it is instead the diagonal of a ring
          when it bridges two (a metallacycle's SiR2) or sits past a lone-pair donor (an amine's
          alpha CH2, an amide's SiMe3);
        - any other x is the diagonal of a chelate ring when it bridges two or more lone-pair donors
          and lies beyond them (a carboxylate C, a dithiocarbamate C). A face binds through every
          atom instead: x keeps its bond when a ring holds it and those donors (Cp, cyclo-P5), and
          an eta3 face's centre lies nearer.

        Lone pairs as in has_lone_pair.
        """
        metals = self.data.metals

        def sigma(a: int) -> int:
            return sum(1 for nbr in G.neighbors(a) if G.nodes[nbr]["symbol"] not in metals)

        def lone_pair(a: int) -> bool:
            return has_lone_pair(G, a, self.data)

        partners = [d for d in G.neighbors(x) if d != m and G.has_edge(d, m) and G.nodes[d]["symbol"] not in metals]
        if not partners or G.nodes[x]["symbol"] == "H":
            return None
        hydridic = self.data.electronegativity.get(G.nodes[x]["symbol"], 2.5) < self.data.electronegativity["H"]
        saturated = sigma(x) >= 4 and not lone_pair(x)
        if (hydridic or saturated) and any(G.nodes[d]["symbol"] == "H" for d in partners):
            return "reaches the metal through its bridging H"
        beyond = all(G.edges[m, x]["distance"] >= G.edges[m, d]["distance"] for d in partners)
        if saturated:
            if beyond and (len(partners) >= 2 or lone_pair(partners[0])):
                return "saturated: a ring diagonal beyond " + ", ".join(f"{G.nodes[d]['symbol']}{d}" for d in partners)
            return None
        if any({x, *partners} <= set(ring) for ring in G.graph.get("_rings", [])):
            return None
        if len(partners) >= 2 and beyond and all(lone_pair(d) for d in partners):
            return "chelate ring diagonal beyond lone-pair donors " + ", ".join(
                f"{G.nodes[d]['symbol']}{d}" for d in partners
            )
        return None

    def _check_3ring_valence(self, G: nx.Graph, i: int, j: int, sym_i: str, sym_j: str, relaxed: bool) -> bool:
        """Valence check for 3-ring bonding atoms. Return False if rejected.

        A non-metal pair is rejected when both atoms would exceed their valence; a metal bond is
        judged later, on the finished graph (metal_bond_blocked).
        """
        if sym_i in self.data.metals or sym_j in self.data.metals:
            return True  # judged once the ligand skeleton is complete (metal_bond_blocked)

        overflow = [self._valence_overflow(G, atom) for atom in (i, j)]
        if min(overflow) <= 0:
            return True

        if relaxed:
            if max(overflow) <= 1.0:
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
