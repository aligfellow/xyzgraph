"""Pure geometric calculations for molecular structures.

All methods are stateless and operate on coordinate data.
"""

import math
from typing import List, Tuple

import networkx as nx
import numpy as np


class GeometryCalculator:
    """Stateless utility for molecular geometry calculations.

    All methods are static - no mutable state, can be shared across components.
    """

    @staticmethod
    def distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Euclidean distance between two 3D points. Pure-math inline."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def angle(
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
        pos3: Tuple[float, float, float],
    ) -> float:
        """Angle at pos2 formed by pos1-pos2-pos3 (in degrees)."""
        v1x = pos1[0] - pos2[0]
        v1y = pos1[1] - pos2[1]
        v1z = pos1[2] - pos2[2]
        v2x = pos3[0] - pos2[0]
        v2y = pos3[1] - pos2[1]
        v2z = pos3[2] - pos2[2]

        v1_norm = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        v2_norm = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return 0.0

        cos_angle = (v1x * v2x + v1y * v2y + v1z * v2z) / (v1_norm * v2_norm)
        if cos_angle > 1.0:
            cos_angle = 1.0
        elif cos_angle < -1.0:
            cos_angle = -1.0
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def ring_angle_sum(ring: List[int], graph: nx.Graph) -> float:
        """Sum of internal angles in a ring."""
        if len(ring) < 3:
            return 0.0

        angle_sum = 0.0
        n = len(ring)

        for i in range(n):
            prev = ring[(i - 1) % n]
            curr = ring[i]
            next_node = ring[(i + 1) % n]

            pos_prev = graph.nodes[prev]["position"]
            pos_curr = graph.nodes[curr]["position"]
            pos_next = graph.nodes[next_node]["position"]

            angle_sum += GeometryCalculator.angle(pos_prev, pos_curr, pos_next)

        return angle_sum

    @staticmethod
    def check_planarity(ring: List[int], graph: nx.Graph, tolerance: float = 0.15) -> bool:
        """Check if ring atoms lie approximately in a plane.

        Uses SVD to find best-fit plane and measures deviations.

        Parameters
        ----------
        ring : List[int]
            Node indices forming the ring
        graph : nx.Graph
            Graph containing node positions
        tolerance : float
            Maximum allowed deviation from plane (Angstroms)

        Returns
        -------
        bool
            True if all atoms within tolerance of best-fit plane
        """
        if len(ring) < 3:
            return True  # 3-rings always planar

        coords = np.array([graph.nodes[i]["position"] for i in ring])

        # Fit plane using SVD
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        # Plane normal is smallest singular vector
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]

        # Check distance of each point to plane
        distances = np.abs(centered @ normal)
        max_deviation = distances.max()

        return max_deviation < tolerance

    @staticmethod
    def is_collinear(
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
        pos3: Tuple[float, float, float],
        angle_threshold: float = 160.0,
    ) -> bool:
        """Check if three points are nearly collinear.

        Parameters
        ----------
        pos1, pos2, pos3 : Tuple[float, float, float]
            3D coordinates
        angle_threshold : float
            Angle in degrees above which points are considered collinear

        Returns
        -------
        bool
            True if angle > threshold or angle < (180 - threshold)
        """
        angle_deg = GeometryCalculator.angle(pos1, pos2, pos3)
        return angle_deg > angle_threshold or angle_deg < (180.0 - angle_threshold)

    @staticmethod
    def dot_product_normalized(
        pos1: Tuple[float, float, float],
        center: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        """Calculate normalized dot product of vectors (center→pos1) · (center→pos2).

        Returns
        -------
        float
            Dot product in range [-1, 1]. Returns 0.0 if either vector is zero.
        """
        v1 = np.array(pos1) - np.array(center)
        v2 = np.array(pos2) - np.array(center)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))
