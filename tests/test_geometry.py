"""Basic tests for geometry calculations.

Verifies extraction from GraphBuilder works correctly.
"""

import networkx as nx
import numpy as np
import pytest

from xyzgraph.geometry import GeometryCalculator


def test_distance_basic():
    """Distance calculation works."""
    assert GeometryCalculator.distance((0, 0, 0), (1, 0, 0)) == pytest.approx(1.0)


def test_angle_molecular_context():
    """Angle calculation for H-C-H tetrahedral geometry."""
    # Tetrahedral carbon with two hydrogens
    c_pos = (0, 0, 0)
    h1_pos = (1, 1, 1)
    h2_pos = (-1, -1, 1)
    angle = GeometryCalculator.angle(h1_pos, c_pos, h2_pos)
    assert angle == pytest.approx(109.5, abs=0.1)  # Tetrahedral angle


def test_ring_angle_sum_benzene():
    """Ring angle sum for hexagon (benzene-like)."""
    G = nx.Graph()
    for i in range(6):
        angle = 2 * np.pi * i / 6
        G.add_node(i, position=(np.cos(angle), np.sin(angle), 0))
    # Hexagon sum = 720°
    assert GeometryCalculator.ring_angle_sum(list(range(6)), G) == pytest.approx(720.0, abs=1)


def test_planarity_benzene():
    """Planarity check for flat benzene ring."""
    G = nx.Graph()
    for i in range(6):
        angle = 2 * np.pi * i / 6
        G.add_node(i, position=(np.cos(angle), np.sin(angle), 0))
    assert GeometryCalculator.check_planarity(list(range(6)), G)


def test_collinearity_linear_molecule():
    """Collinearity for linear molecules (CO2-like)."""
    # O=C=O linear arrangement
    assert GeometryCalculator.is_collinear((-1, 0, 0), (0, 0, 0), (1, 0, 0))


def test_dot_product_bond_vectors():
    """Dot product for bond angle analysis."""
    # Two perpendicular bonds from origin
    dot = GeometryCalculator.dot_product_normalized((1, 0, 0), (0, 0, 0), (0, 1, 0))
    assert dot == pytest.approx(0.0)  # 90° angle
