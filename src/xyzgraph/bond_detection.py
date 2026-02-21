"""Bond detection from 3D atomic coordinates.

Detects which atoms are bonded using distance-based heuristics and
geometric validation. Produces a connectivity graph with all bond_order=1.0.

Two-phase construction:
1. Baseline bonds using default thresholds, compute rings
2. Extended bonds using custom thresholds (if different from defaults)
"""

import logging
from collections import Counter
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .bond_geometry_check import BondGeometryChecker
from .data_loader import MolecularData
from .geometry import GeometryCalculator
from .parameters import BondThresholds

logger = logging.getLogger(__name__)

# Default thresholds for baseline bond detection
_DEFAULT_THRESHOLDS = BondThresholds()

# Bonds above this confidence are added without geometric validation
HIGH_CONFIDENCE_THRESHOLD = 0.4


class BondDetector:
    """Detects bonds from 3D atomic coordinates using distance-based heuristics.

    Uses VDW radii and configurable distance thresholds to identify
    covalent bonds, then validates geometry to reject spurious bonds.
    """

    def __init__(
        self,
        geometry: GeometryCalculator,
        bond_checker: BondGeometryChecker,
        thresholds: BondThresholds,
        data: MolecularData,
    ):
        self.geometry = geometry
        self.bond_checker = bond_checker
        self.thresholds = thresholds
        self.data = data
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

    @staticmethod
    def _get_period(atomic_number: int) -> int:
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
        """Apply period-dependent scaling to bond threshold.

        Heavier elements need looser thresholds because VDW/covalent
        ratio increases down the periodic table.
        """
        if has_hydrogen:
            if self.thresholds.period_scaling_h_bonds == 0.0:
                return base_threshold
            non_h_z = z_i if z_i > 1 else z_j
            period = self._get_period(non_h_z)
            period_factor = 1.0 + (period - 2) * self.thresholds.period_scaling_h_bonds
            return base_threshold * period_factor
        else:
            sym_i = self.data.n2s.get(z_i, "")
            sym_j = self.data.n2s.get(z_j, "")
            both_nonmetal = sym_i not in self.data.metals and sym_j not in self.data.metals

            if both_nonmetal and self.thresholds.period_scaling_nonmetal_bonds != 0.0:
                max_period = max(self._get_period(z_i), self._get_period(z_j))
                period_factor = 1.0 + (max_period - 2) * self.thresholds.period_scaling_nonmetal_bonds
                return base_threshold * period_factor
            else:
                return base_threshold

    def _should_bond_metal(self, sym_i: str, sym_j: str) -> bool:
        """Chemical filter for metal bonds (called AFTER distance check).

        Returns False only for implausible metal pairings.
        """
        if sym_i not in self.data.metals and sym_j not in self.data.metals:
            return True

        if sym_i in self.data.metals and sym_j in self.data.metals:
            return self.thresholds.allow_metal_metal_bonds

        other = sym_j if sym_i in self.data.metals else sym_i

        if other in ("O", "N", "C", "P", "S", "H"):
            return True
        if other in ("F", "Cl", "Br", "I"):
            return True
        if other in ("B", "Si", "Se", "Te"):
            return True

        return False

    def _find_new_rings_from_edge(self, G: nx.Graph, i: int, j: int) -> List[List[int]]:
        """Detect new rings formed by adding edge (i, j).

        Finds shortest path from i to j in metal-free subgraph.
        """
        if G.nodes[i]["symbol"] in self.data.metals or G.nodes[j]["symbol"] in self.data.metals:
            return []

        non_metal_nodes = [n for n in G.nodes() if G.nodes[n]["symbol"] not in self.data.metals]
        G_no_metals = G.subgraph(non_metal_nodes).copy()

        if i not in G_no_metals or j not in G_no_metals:
            return []

        if G_no_metals.has_edge(i, j):
            G_no_metals.remove_edge(i, j)

        try:
            path = nx.shortest_path(G_no_metals, source=i, target=j)
            return [path]
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    @staticmethod
    def _compute_threshold(
        thresholds: BondThresholds,
        si: str,
        sj: str,
        has_h: bool,
        has_metal: bool,
        r_sum: float,
        is_metal_metal_self: bool = False,
    ) -> float:
        """Compute distance threshold for a pair of atoms."""
        if si == "H" and sj == "H":
            return thresholds.threshold_h_h * r_sum * thresholds.threshold
        elif has_h and has_metal:
            return thresholds.threshold_h_metal * r_sum * thresholds.threshold
        elif has_h and not has_metal:
            return thresholds.threshold_h_nonmetal * r_sum * thresholds.threshold
        elif is_metal_metal_self:
            return thresholds.threshold_metal_metal_self * r_sum
        elif has_metal:
            return thresholds.threshold_metal_ligand * r_sum
        else:
            return thresholds.threshold_nonmetal_nonmetal * r_sum * thresholds.threshold

    def detect(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        bond: Optional[List[Tuple[int, int]]] = None,
        unbond: Optional[List[Tuple[int, int]]] = None,
    ) -> nx.Graph:
        """Detect bonds and return graph with all bond_order=1.0.

        Parameters
        ----------
        atoms : list of (symbol, (x, y, z))
            Atomic coordinates.
        bond : list of (i, j), optional
            User-specified bonds to force.
        unbond : list of (i, j), optional
            User-specified bonds to remove.

        Returns
        -------
        nx.Graph
            Connectivity graph with bond_order=1.0 on all edges.
        """
        self.log_buffer = []  # Reset log buffer

        G = nx.Graph()

        # Pre-compute atom properties
        symbols = [symbol for symbol, _ in atoms]
        atomic_numbers = [self.data.s2n[symbol] for symbol, _ in atoms]
        positions = [(x, y, z) for _, (x, y, z) in atoms]
        pos = np.array(positions)

        # Add nodes
        for i, (atomic_num, symbol) in enumerate(zip(atomic_numbers, symbols)):
            G.add_node(i, symbol=symbol, atomic_number=atomic_num, position=tuple(pos[i]))

        self._log(f"Added {len(atoms)} atoms", 1)

        # Element counts (needed during geometry checks)
        element_counts = Counter(symbols)
        G.graph["_element_counts"] = dict(element_counts)

        # Check for custom thresholds
        has_custom = (
            self.thresholds.threshold != _DEFAULT_THRESHOLDS.threshold
            or self.thresholds.threshold_h_h != _DEFAULT_THRESHOLDS.threshold_h_h
            or self.thresholds.threshold_h_nonmetal != _DEFAULT_THRESHOLDS.threshold_h_nonmetal
            or self.thresholds.threshold_h_metal != _DEFAULT_THRESHOLDS.threshold_h_metal
            or self.thresholds.threshold_metal_ligand != _DEFAULT_THRESHOLDS.threshold_metal_ligand
            or self.thresholds.threshold_nonmetal_nonmetal != _DEFAULT_THRESHOLDS.threshold_nonmetal_nonmetal
        )

        if has_custom:
            self._log("Custom thresholds detected - using 2-phase construction", 1)

        # ===== STEP 1: Baseline bonds (DEFAULT thresholds) =====
        baseline_bonds = []

        for i in range(len(atoms)):
            si = symbols[i]
            is_metal_i = si in self.data.metals

            for j in range(i + 1, len(atoms)):
                sj = symbols[j]
                is_metal_j = sj in self.data.metals
                has_metal = is_metal_i or is_metal_j
                is_metal_metal_self = is_metal_i and is_metal_j and (si == sj)
                has_h = "H" in (si, sj)

                d = GeometryCalculator.distance(tuple(pos[i]), tuple(pos[j]))
                r_sum = self.data.vdw.get(si, 2.0) + self.data.vdw.get(sj, 2.0)

                baseline_threshold = self._compute_threshold(
                    _DEFAULT_THRESHOLDS, si, sj, has_h, has_metal, r_sum, is_metal_metal_self
                )

                z_i = atomic_numbers[i]
                z_j = atomic_numbers[j]
                baseline_threshold = self._get_threshold_with_period_scaling(
                    baseline_threshold, z_i, z_j, has_hydrogen=has_h
                )

                if d < baseline_threshold:
                    confidence = 1.0 - (d / baseline_threshold)
                    baseline_bonds.append((confidence, i, j, d, has_metal))

        baseline_bonds.sort(reverse=True, key=lambda x: x[0])

        self._log(
            f"Step 1: Found {len(baseline_bonds)} baseline bonds (using default thresholds)",
            1,
        )

        # Add baseline bonds with confidence-based validation
        edge_count = 0
        rejected_count = 0

        for confidence, i, j, d, has_metal in baseline_bonds:
            si, sj = symbols[i], symbols[j]
            self._log(
                f"  Evaluating bond {si}{i}-{sj}{j} (d={d:.3f} Å, conf={confidence:.2f})",
                3,
            )
            if has_metal and not self._should_bond_metal(si, sj):
                rejected_count += 1
                continue

            if confidence > HIGH_CONFIDENCE_THRESHOLD:
                G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=has_metal)
                edge_count += 1
                self._log("  Added high-confidence bond", 4)
            elif self.bond_checker.check(G, i, j, d, confidence, baseline_bonds):
                G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=has_metal)
                edge_count += 1
                self._log("  Added validated bond", 4)
            else:
                rejected_count += 1

        self._log(f"Step 1: {edge_count} baseline bonds added, {rejected_count} rejected", 1)

        # Compute rings from baseline structure
        non_metal_nodes = [n for n in G.nodes() if G.nodes[n]["symbol"] not in self.data.metals]
        G_no_metals = G.subgraph(non_metal_nodes).copy()
        rings = nx.cycle_basis(G_no_metals)

        G.graph["_rings"] = rings
        G.graph["_neighbors"] = {n: list(G.neighbors(n)) for n in G.nodes()}
        G.graph["_has_H"] = {n: any(G.nodes[nbr]["symbol"] == "H" for nbr in G.neighbors(n)) for n in G.nodes()}

        self._log(f"Found {len(rings)} rings from initial bonding (excluding metal cycles)", 1)

        # ===== STEP 2: Extended bonds (CUSTOM thresholds if modified) =====
        if has_custom:
            extended_bonds = []
            baseline_edges = set(G.edges())

            for i in range(len(atoms)):
                si = symbols[i]
                is_metal_i = si in self.data.metals

                for j in range(i + 1, len(atoms)):
                    if (i, j) in baseline_edges or (j, i) in baseline_edges:
                        continue

                    sj = symbols[j]
                    is_metal_j = sj in self.data.metals
                    has_metal = is_metal_i or is_metal_j
                    has_h = "H" in (si, sj)

                    d = GeometryCalculator.distance(tuple(pos[i]), tuple(pos[j]))
                    r_sum = self.data.vdw.get(si, 2.0) + self.data.vdw.get(sj, 2.0)

                    # Custom phase: no is_metal_metal_self check (preserves original behavior)
                    custom_threshold = self._compute_threshold(self.thresholds, si, sj, has_h, has_metal, r_sum)

                    z_i = atomic_numbers[i]
                    z_j = atomic_numbers[j]
                    custom_threshold = self._get_threshold_with_period_scaling(
                        custom_threshold, z_i, z_j, has_hydrogen=has_h
                    )

                    if d < custom_threshold:
                        confidence = 1.0 - (d / custom_threshold)
                        extended_bonds.append((confidence, i, j, d, has_metal))

            extended_bonds.sort(reverse=True, key=lambda x: x[0])

            self._log(
                f"Step 2: Found {len(extended_bonds)} extended bonds (custom thresholds)",
                1,
            )

            # Add extended bonds with STRICT validation and incremental ring updates
            extended_added = 0
            extended_rejected = 0
            new_rings_count = 0

            for confidence, i, j, d, has_metal in extended_bonds:
                si, sj = symbols[i], symbols[j]
                self._log(
                    f"  Evaluating extended bond {si}{i}-{sj}{j} (d={d:.3f} Å, conf={confidence:.2f})",
                    3,
                )
                if has_metal and not self._should_bond_metal(si, sj):
                    extended_rejected += 1
                    continue

                if self.bond_checker.check(G, i, j, d, confidence, baseline_bonds):
                    G.add_edge(i, j, bond_order=1.0, distance=d, metal_coord=has_metal)
                    extended_added += 1

                    new_rings = self._find_new_rings_from_edge(G, i, j)
                    if new_rings:
                        G.graph["_rings"].extend(new_rings)
                        new_rings_count += len(new_rings)
                        ring_size = len(new_rings[0])
                        self._log(f"    Bond {si}{i}-{sj}{j} creates new {ring_size}-ring", 3)

                    G.graph["_neighbors"][i] = list(G.neighbors(i))
                    G.graph["_neighbors"][j] = list(G.neighbors(j))
                else:
                    extended_rejected += 1

            self._log(
                f"Step 2: {extended_added} extended bonds added, {extended_rejected} rejected, "
                f"{new_rings_count} new rings detected",
                1,
            )

        # Handle user-specified bonds
        if bond:
            for i, j in bond:
                if not G.has_edge(i, j):
                    d = GeometryCalculator.distance(tuple(pos[i]), tuple(pos[j]))
                    G.add_edge(i, j, bond_order=1, distance=d)
                    si = symbols[i]
                    sj = symbols[j]
                    self._log(f"Added user-specified bond {si}{i}-{sj}{j} (d={d:.3f} Å)", 2)

        if unbond:
            for i, j in unbond:
                if G.has_edge(i, j):
                    G.remove_edge(i, j)
                    si = symbols[i]
                    sj = symbols[j]
                    self._log(f"Removed user-specified bond {si}{i}-{sj}{j}", 2)

        # Final ring update if graph topology was modified
        if has_custom or bond or unbond:
            non_metal_nodes = [n for n in G.nodes() if G.nodes[n]["symbol"] not in self.data.metals]
            G_no_metals = G.subgraph(non_metal_nodes).copy()
            rings = nx.cycle_basis(G_no_metals)
            G.graph["_rings"] = rings
            self._log(f"Final: {len(rings)} rings after bond modifications", 1)

        total_bonds = G.number_of_edges()
        self._log(f"Total bonds in graph: {total_bonds}", 1)

        return G
