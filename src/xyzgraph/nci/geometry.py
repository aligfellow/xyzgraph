"""Geometry primitives for NCI detection."""

from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector, or [1,0,0] for zero-length input."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees."""
    c = np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def plane_normal(points: np.ndarray) -> np.ndarray:
    """Plane normal via SVD. Robust to noise."""
    pts = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    normal = vh[-1]
    n = np.linalg.norm(normal)
    return normal / n if n > 1e-12 else normal


def point_plane_distance(point: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> float:
    """Unsigned distance from point to plane defined by origin and normal."""
    return float(abs(np.dot(point - origin, unit(normal))))
