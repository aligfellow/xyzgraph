"""Shared contour extraction infrastructure for surface modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from xyzrender.cube import BOHR_TO_ANG, CubeData

# --- Contour processing ---
# 3D lobe filtering (physical units — scales with cube grid spacing)
MIN_LOBE_VOLUME_BOHR3 = 0.1  # discard 3D orbital components smaller than this (Bohr^3)

# 2D projected-grid properties (grid-cell units, not related to cube spacing)
UPSAMPLE_FACTOR = 3  # 80x80 -> 400x400 -- smooth enough for publication
BLUR_SIGMA = 0.8  # Gaussian sigma in 2D grid cells before upsampling
MIN_LOOP_PERIMETER = 15.0  # upsampled grid units — discard tiny contour fragments


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Lobe3D:
    """A spatially connected 3D orbital lobe (connected component)."""

    flat_indices: np.ndarray  # indices into flattened grid/position arrays
    phase: str  # "pos" or "neg"


@dataclass
class LobeContour2D:
    """Contour loops for one 3D lobe projected to 2D."""

    loops: list[np.ndarray]  # each (M, 2) array of [row, col] points
    phase: str  # "pos" or "neg"
    z_depth: float  # average z-coordinate (for front/back ordering)
    centroid_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)  # for pairing
    lobe_color: str | None = None  # per-lobe color override (NCI avg coloring)


@runtime_checkable
class ContourGrid(Protocol):
    """Protocol for 2D projection grid metadata used by SVG path converters.

    Both :class:`SurfaceContours` and :class:`~xyzrender.nci.NCIContours`
    satisfy this protocol, allowing :func:`combined_path_d` to accept
    either type without inheritance coupling.
    """

    resolution: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class SurfaceContours:
    """Pre-computed MO or density contour data ready for SVG rendering."""

    lobes: list[LobeContour2D] = field(default_factory=list)  # sorted by z_depth
    resolution: int = 0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    pos_color: str = "#2554A5"
    neg_color: str = "#851639"
    # Tight Angstrom extent of actual lobe contours (for canvas fitting)
    lobe_x_min: float | None = None
    lobe_x_max: float | None = None
    lobe_y_min: float | None = None
    lobe_y_max: float | None = None


# Backward-compatible alias (used by renderer and other callers)
MOContours = SurfaceContours


# ---------------------------------------------------------------------------
# Grid/projection functions
# ---------------------------------------------------------------------------


def cube_corners_ang(cube: CubeData) -> np.ndarray:
    """Compute the 8 corner positions of the cube grid in Angstrom."""
    n1, n2, n3 = cube.grid_shape
    corners = np.empty((8, 3))
    idx = 0
    for i in (0, n1 - 1):
        for j in (0, n2 - 1):
            for k in (0, n3 - 1):
                corners[idx] = cube.origin + i * cube.steps[0] + j * cube.steps[1] + k * cube.steps[2]
                idx += 1
    return corners * BOHR_TO_ANG


def compute_grid_positions(cube: CubeData) -> np.ndarray:
    """Compute all grid positions in Angstrom (flattened). Cached for reuse."""
    n1, n2, n3 = cube.grid_shape
    ii, jj, kk = np.mgrid[0:n1, 0:n2, 0:n3]
    positions = (
        cube.origin + ii[..., None] * cube.steps[0] + jj[..., None] * cube.steps[1] + kk[..., None] * cube.steps[2]
    )
    return positions.reshape(-1, 3) * BOHR_TO_ANG


# ---------------------------------------------------------------------------
# Marching squares
# ---------------------------------------------------------------------------

# Lookup table: for each 4-bit case index, list of (edge_a, edge_b) pairs
# Corners: 0=top-left(i,j), 1=top-right(i,j+1), 2=bottom-right(i+1,j+1), 3=bottom-left(i+1,j)
# Edges: 0=top, 1=right, 2=bottom, 3=left
_MS_TABLE: dict[int, list[tuple[int, int]]] = {
    0: [],
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    5: [(3, 0), (1, 2)],  # saddle — resolved below
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(2, 3)],
    9: [(2, 0)],
    10: [(0, 3), (2, 1)],  # saddle — resolved below
    11: [(2, 1)],
    12: [(1, 3)],
    13: [(1, 0)],
    14: [(0, 3)],
    15: [],
}


def marching_squares(
    grid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Extract contour line segments from a 2D scalar field.

    Returns (N, 4) array where each row is [row1, col1, row2, col2].
    """
    ny, nx = grid.shape
    _empty = np.empty((0, 4))
    if ny < 2 or nx < 2:
        return _empty

    # Corner values for all (ny-1) x (nx-1) cells
    v0 = grid[:-1, :-1]  # top-left
    v1 = grid[:-1, 1:]  # top-right
    v2 = grid[1:, 1:]  # bottom-right
    v3 = grid[1:, :-1]  # bottom-left

    # 4-bit case index per cell
    case = (
        (v0 >= threshold).view(np.uint8)
        | ((v1 >= threshold).view(np.uint8) << 1)
        | ((v2 >= threshold).view(np.uint8) << 2)
        | ((v3 >= threshold).view(np.uint8) << 3)
    )

    # Early exit: no contour crossings
    if not np.any(case & (case != 15)):
        return _empty

    # Cell row/col index grids
    ri, ci = np.indices((ny - 1, nx - 1), dtype=float)

    # Interpolation parameter t on each edge, clamped to [0, 1]
    def _t(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        dv = vb - va
        safe_dv = np.where(np.abs(dv) > 1e-12, dv, 1.0)
        t = np.where(np.abs(dv) > 1e-12, (threshold - va) / safe_dv, 0.5)
        return np.clip(t, 0.0, 1.0)

    t01, t12, t23, t30 = _t(v0, v1), _t(v1, v2), _t(v2, v3), _t(v3, v0)

    # Edge crossing positions (row, col) for each of the 4 edges:
    er = [ri, ri + t12, ri + 1, ri + 1 - t30]
    ec = [ci + t01, ci + 1, ci + 1 - t23, ci]

    # Saddle-point centre value (only used for cases 5 and 10)
    center = (v0 + v1 + v2 + v3) * 0.25

    # Gather segments per case (14 iterations, not ny*nx)
    seg_r1, seg_c1, seg_r2, seg_c2 = [], [], [], []

    def _gather(mask: np.ndarray, ea: int, eb: int) -> None:
        seg_r1.append(er[ea][mask])
        seg_c1.append(ec[ea][mask])
        seg_r2.append(er[eb][mask])
        seg_c2.append(ec[eb][mask])

    for cv in range(1, 15):
        mask = case == cv
        if not mask.any():
            continue

        if cv == 5:
            alt = mask & (center >= threshold)
            std = mask & ~alt
            if std.any():
                _gather(std, 3, 0)
                _gather(std, 1, 2)
            if alt.any():
                _gather(alt, 3, 2)
                _gather(alt, 1, 0)
        elif cv == 10:
            alt = mask & (center >= threshold)
            std = mask & ~alt
            if std.any():
                _gather(std, 0, 3)
                _gather(std, 2, 1)
            if alt.any():
                _gather(alt, 0, 1)
                _gather(alt, 2, 3)
        else:
            for ea, eb in _MS_TABLE[cv]:
                _gather(mask, ea, eb)

    if not seg_r1:
        return _empty

    return np.column_stack(
        [
            np.concatenate(seg_r1),
            np.concatenate(seg_c1),
            np.concatenate(seg_r2),
            np.concatenate(seg_c2),
        ]
    )


# ---------------------------------------------------------------------------
# Segment chaining into closed loops
# ---------------------------------------------------------------------------


def chain_segments(
    segments: np.ndarray,
    decimals: int = 4,
) -> list[np.ndarray]:
    """Connect line segments into closed contour loops."""
    n_seg = len(segments)
    if n_seg == 0:
        return []

    # 2*n_seg endpoints: endpoint 2i = start of segment i, 2i+1 = end
    endpoints = np.empty((2 * n_seg, 2))
    endpoints[0::2] = segments[:, :2]
    endpoints[1::2] = segments[:, 2:]

    # Integer keys for fast pair matching via sort
    kscale = 10.0**decimals
    ikeys = np.rint(endpoints * kscale).astype(np.int64)
    ikeys[:, 0] -= ikeys[:, 0].min()
    ikeys[:, 1] -= ikeys[:, 1].min()
    max_col = int(ikeys[:, 1].max()) + 1
    combined = ikeys[:, 0] * max_col + ikeys[:, 1]

    # Sort and pair-match consecutive equal keys
    order = np.argsort(combined, kind="mergesort")
    sorted_keys = combined[order]

    match = np.full(2 * n_seg, -1, dtype=np.intp)
    i = 0
    while i < 2 * n_seg - 1:
        if sorted_keys[i] == sorted_keys[i + 1]:
            a, b = int(order[i]), int(order[i + 1])
            match[a] = b
            match[b] = a
            i += 2
        else:
            i += 1

    # Walk chains using array indexing
    used = np.zeros(n_seg, dtype=bool)
    loops: list[np.ndarray] = []

    for start_seg in range(n_seg):
        if used[start_seg]:
            continue
        used[start_seg] = True
        chain_pts = [endpoints[2 * start_seg], endpoints[2 * start_seg + 1]]
        cur = 2 * start_seg + 1

        while True:
            partner = match[cur]
            if partner < 0:
                break
            seg = partner >> 1
            if used[seg]:
                break
            used[seg] = True
            exit_ep = partner ^ 1
            chain_pts.append(endpoints[exit_ep])
            cur = exit_ep

        if len(chain_pts) >= 3:
            loops.append(np.array(chain_pts))

    return loops


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def resample_loop(
    loop: np.ndarray,
    target_spacing: float = 1.5,
) -> np.ndarray:
    """Resample a closed contour loop at uniform arc-length intervals."""
    n = len(loop)
    if n < 3:
        return loop

    closed = np.vstack([loop, loop[:1]])
    diffs = np.diff(closed, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(dists.sum())
    if total_len < 1e-6:
        return loop

    n_pts = max(int(total_len / target_spacing + 0.5), 8)

    cum = np.empty(n + 1)
    cum[0] = 0.0
    np.cumsum(dists, out=cum[1:])

    targets = np.linspace(0, total_len, n_pts, endpoint=False)
    seg_idx = np.searchsorted(cum[1:], targets, side="right")
    seg_idx = np.clip(seg_idx, 0, n - 1)

    seg_len = dists[seg_idx]
    safe_len = np.where(seg_len > 1e-12, seg_len, 1.0)
    t = np.where(seg_len > 1e-12, (targets - cum[seg_idx]) / safe_len, 0.0)

    p0 = closed[seg_idx]
    p1 = closed[seg_idx + 1]
    return p0 + t[:, np.newaxis] * (p1 - p0)


# ---------------------------------------------------------------------------
# Gaussian smoothing + bilinear upsampling
# ---------------------------------------------------------------------------


def loop_perimeter(loop: np.ndarray) -> float:
    """Sum of segment lengths around a contour loop."""
    diffs = np.diff(np.vstack([loop, loop[:1]]), axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def gaussian_blur_2d(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable Gaussian blur to 2D grid (vectorized, pure numpy)."""
    size = int(4 * sigma + 0.5) * 2 + 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    pad = size // 2
    ny, nx = grid.shape

    # Horizontal pass: convolve each row via matrix multiply
    padded = np.pad(grid, ((0, 0), (pad, pad)), mode="edge")
    idx = np.arange(nx)[:, None] + np.arange(size)[None, :]
    temp = padded[:, idx] @ kernel  # (ny, nx)

    # Vertical pass: convolve each column via matrix multiply
    padded = np.pad(temp, ((pad, pad), (0, 0)), mode="edge")
    idx = np.arange(ny)[:, None] + np.arange(size)[None, :]
    return padded[idx, :].transpose(0, 2, 1) @ kernel  # (ny, nx)


def upsample_2d(grid: np.ndarray, factor: int) -> np.ndarray:
    """Upsample 2D array by integer factor using bilinear interpolation."""
    ny, nx = grid.shape
    x_old = np.arange(nx)
    x_new = np.linspace(0, nx - 1, nx * factor)
    # Interpolate along columns first
    temp = np.array([np.interp(x_new, x_old, grid[i]) for i in range(ny)])
    # Then along rows
    y_old = np.arange(ny)
    y_new = np.linspace(0, ny - 1, ny * factor)
    return np.array([np.interp(y_new, y_old, temp[:, j]) for j in range(nx * factor)]).T


# ---------------------------------------------------------------------------
# SVG path conversion
# ---------------------------------------------------------------------------


def loop_to_path_d(
    loop: np.ndarray,
    grid: ContourGrid,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> str | None:
    """Convert a contour loop to a smooth SVG path (Catmull-Rom to cubic Bezier)."""
    if len(loop) < 3:
        return None
    res = max(grid.resolution - 1, 1)

    # Vectorized grid → SVG coordinate transform
    x_ang = grid.x_min + (loop[:, 1] / res) * (grid.x_max - grid.x_min)
    y_ang = grid.y_min + (loop[:, 0] / res) * (grid.y_max - grid.y_min)
    sx = canvas_w / 2 + scale * (x_ang - cx)
    sy = canvas_h / 2 - scale * (y_ang - cy)

    # Catmull-Rom control points via rolled arrays
    p0x, p0y = np.roll(sx, 1), np.roll(sy, 1)
    p2x, p2y = np.roll(sx, -1), np.roll(sy, -1)
    p3x, p3y = np.roll(sx, -2), np.roll(sy, -2)

    cp1x = sx + (p2x - p0x) / 6
    cp1y = sy + (p2y - p0y) / 6
    cp2x = p2x - (p3x - sx) / 6
    cp2y = p2y - (p3y - sy) / 6

    # Build SVG path string
    coords = np.column_stack([cp1x, cp1y, cp2x, cp2y, p2x, p2y])
    cmds = [f"C {a:.1f} {b:.1f} {c:.1f} {d:.1f} {e:.1f} {f:.1f}" for a, b, c, d, e, f in coords.tolist()]
    return f"M {sx[0]:.1f} {sy[0]:.1f} " + " ".join(cmds) + " Z"


def combined_path_d(
    loops: list[np.ndarray],
    grid: ContourGrid,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> str | None:
    """Combine all contour loops of one phase into a single SVG path d-string.

    Uses fill-rule="evenodd" so inner loops become holes automatically.
    """
    parts = []
    for loop in loops:
        d = loop_to_path_d(loop, grid, scale, cx, cy, canvas_w, canvas_h)
        if d:
            parts.append(d)
    return " ".join(parts) if parts else None
