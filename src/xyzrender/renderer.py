"""SVG renderer for molecular structures."""

from __future__ import annotations

import itertools
import logging

import numpy as np
from xyzgraph import DATA

from xyzrender.colors import _FOG_NEAR, WHITE, blend_fog, cmap_viridis, get_color, get_gradient_colors
from xyzrender.dens import dens_layers_svg
from xyzrender.mo import (
    classify_mo_lobes,
    mo_back_lobes_svg,
    mo_front_lobes_svg,
    mo_gradient_defs_svg,
)
from xyzrender.types import BondStyle, Color, RenderConfig

logger = logging.getLogger(__name__)

_RADIUS_SCALE = 0.075  # VdW → display radius
_REF_SPAN = 6.0  # reference molecular span (Å) for proportional bond/stroke scaling
_REF_CANVAS = 800  # reference canvas size (px) — bond/label widths are defined at this size
_CENTROID_VDW = 0.5  # VdW radius (Å) for NCI pi-system centroid dummy nodes
_H_ATOM_SCALE = 0.6  # display-radius shrink factor for H atoms (ball-and-stick)
_H_VDW_SCALE = 0.8  # VdW-sphere shrink factor for H atoms


def render_svg(graph, config: RenderConfig | None = None, *, _log: bool = True) -> str:
    """Render molecular graph to SVG string."""
    cfg = config or RenderConfig()
    node_ids = list(graph.nodes())
    n = len(node_ids)
    symbols = [graph.nodes[i]["symbol"] for i in node_ids]
    pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    a_nums = [DATA.s2n.get(s, 0) for s in symbols]  # 0 for NCI centroid nodes ("*")

    if cfg.auto_orient and n > 1:
        # Collect TS bond pairs to prioritize in orientation
        ts_pairs = list(cfg.ts_bonds) if cfg.ts_bonds else []
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False) or d.get("bond_type", "") == "TS":
                ts_pairs.append((i, j))
        # Exclude NCI centroid dummy nodes from PCA fitting
        atom_mask = np.array([s != "*" for s in symbols])
        fit_mask = atom_mask if not atom_mask.all() else None
        from xyzrender.utils import pca_orient

        pos = pca_orient(pos, ts_pairs or None, fit_mask=fit_mask)

    raw_vdw = np.array(
        [_CENTROID_VDW if s == "*" else DATA.vdw.get(s, 1.5) * (_H_ATOM_SCALE if s == "H" else 1.0) for s in symbols]
    )
    radii = raw_vdw * cfg.atom_scale * _RADIUS_SCALE

    # VdW sphere radii use a separate (larger) H scaling
    raw_vdw_sphere = np.array(
        [_CENTROID_VDW if s == "*" else DATA.vdw.get(s, 1.5) * (_H_VDW_SCALE if s == "H" else 1.0) for s in symbols]
    )

    # Use VdW radii for canvas fitting when VdW spheres are active
    if cfg.vdw_indices is not None:
        vdw_active = set(range(n)) if len(cfg.vdw_indices) == 0 else set(cfg.vdw_indices)
        fit_radii = np.array([raw_vdw_sphere[i] * cfg.vdw_scale if i in vdw_active else radii[i] for i in range(n)])
    else:
        fit_radii = radii

    # Expand canvas for surface bounds (MO / density / ESP are mutually exclusive)
    extra_lo = extra_hi = None
    if cfg.mo_contours is not None:
        mo = cfg.mo_contours
        if mo.lobe_x_min is not None:
            extra_lo = np.array([mo.lobe_x_min, mo.lobe_y_min])
            extra_hi = np.array([mo.lobe_x_max, mo.lobe_y_max])
    elif cfg.dens_contours is not None:
        extra_lo = np.array([cfg.dens_contours.x_min, cfg.dens_contours.y_min])
        extra_hi = np.array([cfg.dens_contours.x_max, cfg.dens_contours.y_max])
    if cfg.esp_surface is not None:
        extra_lo = np.array([cfg.esp_surface.x_min, cfg.esp_surface.y_min])
        extra_hi = np.array([cfg.esp_surface.x_max, cfg.esp_surface.y_max])
    scale, cx, cy, canvas_w, canvas_h = _fit_canvas(pos, fit_radii, cfg, extra_lo=extra_lo, extra_hi=extra_hi)

    # scale_ratio: encodes both molecule complexity AND canvas size so that
    # bond/label widths defined at _REF_CANVAS grow proportionally on larger canvases.
    ref_scale = (_REF_CANVAS - 2 * cfg.padding) / _REF_SPAN
    scale_ratio = scale / ref_scale
    bw = cfg.bond_width * scale_ratio
    sw = cfg.atom_stroke_width * scale_ratio
    fs_label = cfg.label_font_size * scale_ratio

    if _log:
        logger.debug(
            "Render: %d atoms, %d bonds, scale=%.2f, center=(%.2f, %.2f)", n, graph.number_of_edges(), scale, cx, cy
        )
    z_order = np.argsort(pos[:, 2])

    # Atom base colors — CPK by default, Viridis cmap when --cmap is active
    if cfg.atom_cmap is not None:
        cmap_vals = cfg.atom_cmap
        if cfg.cmap_range is not None:
            vmin, vmax = cfg.cmap_range
        else:
            vmin = min(cmap_vals.values())
            vmax = max(cmap_vals.values())
        vrange = max(vmax - vmin, 1e-10)
        unlabeled = Color.from_hex(cfg.cmap_unlabeled)
        colors = [cmap_viridis((cmap_vals[ai] - vmin) / vrange) if ai in cmap_vals else unlabeled for ai in range(n)]
    else:
        colors = [get_color(a, cfg.color_overrides) for a in a_nums]

    # Bond lookup: (bond_order, style)
    bonds: dict[tuple[int, int], tuple[float, BondStyle]] = {}
    for i, j, d in graph.edges(data=True):
        bo = d.get("bond_order", 1.0) if cfg.bond_orders else 1.0
        bt = d.get("bond_type", "")
        if bt == "TS" or d.get("TS", False):
            style = BondStyle.DASHED
        elif bt == "NCI" or d.get("NCI", False):
            style = BondStyle.DOTTED
        else:
            style = BondStyle.SOLID
        bonds[(i, j)] = bonds[(j, i)] = (bo, style)
    # Manual overrides (add or restyle)
    for i, j in cfg.ts_bonds:
        bonds[(i, j)] = bonds[(j, i)] = (bonds.get((i, j), (1.0, BondStyle.SOLID))[0], BondStyle.DASHED)
    for i, j in cfg.nci_bonds:
        bonds[(i, j)] = bonds[(j, i)] = (bonds.get((i, j), (1.0, BondStyle.SOLID))[0], BondStyle.DOTTED)

    # Only hide C-H hydrogens (not O-H, N-H, etc.)
    hidden = set()
    if cfg.hide_h:
        show = set(cfg.show_h_indices)
        for ai in range(n):
            if symbols[ai] == "H" and ai not in show:
                if all(symbols[nb] == "C" for nb in graph.neighbors(ai)):
                    hidden.add(ai)

    aromatic_rings = [set(r) for r in graph.graph.get("aromatic_rings", [])]

    # Ensure all aromatic bonds are covered by ring data — auto-detect missing rings
    aromatic_ring_edges = set()
    for ring in aromatic_rings:
        rl = list(ring)
        for ii in range(len(rl)):
            for jj in range(ii + 1, len(rl)):
                if (rl[ii], rl[jj]) in bonds or (rl[jj], rl[ii]) in bonds:
                    aromatic_ring_edges.add((min(rl[ii], rl[jj]), max(rl[ii], rl[jj])))
    missing = False
    for (i, j), (bo, _style) in bonds.items():
        if i < j and 1.3 < bo < 1.7 and (i, j) not in aromatic_ring_edges:
            missing = True
            break
    if missing:
        import networkx as nx

        arom_g = nx.Graph()
        for (i, j), (bo, _style) in bonds.items():
            if i < j and 1.3 < bo < 1.7:
                arom_g.add_edge(i, j)
        if arom_g.number_of_edges() > 0:
            aromatic_rings = [set(c) for c in nx.minimum_cycle_basis(arom_g)]

    # Fog factors — normalized across depth range, with a dead-zone near the front
    fog_f = np.zeros(n)
    fog_rgb = np.array([255, 255, 255])
    if cfg.fog:
        zr = max(pos[:, 2].max() - pos[:, 2].min(), 1e-6)
        depth = pos[:, 2].max() - pos[:, 2]  # distance from front atom
        fog_f = cfg.fog_strength * np.clip((depth - _FOG_NEAR) / zr, 0.0, 1.0)

    # --- Build SVG ---
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{canvas_w}" height="{canvas_h}">'
    ]
    svg.append(f'  <rect width="100%" height="100%" fill="{cfg.background}"/>')

    use_grad = cfg.gradient
    # Cmap gives each atom a unique color — must use per-atom gradient defs (like fog mode)
    use_per_atom_grad = cfg.fog or cfg.atom_cmap is not None
    if use_grad:
        svg.append("  <defs>")
        if use_per_atom_grad:
            # Per-atom gradient defs
            for ai in range(n):
                if ai in hidden:
                    continue
                hi, lo = get_gradient_colors(colors[ai], cfg.gradient_strength)
                if cfg.fog:
                    t = min(fog_f[ai] ** 2 * 0.7, 0.70)
                    hi, lo = hi.blend(WHITE, t), lo.blend(WHITE, t)
                    fs = blend_fog(cfg.atom_stroke_color, fog_rgb, fog_f[ai])
                else:
                    fs = cfg.atom_stroke_color
                r = radii[ai] * scale
                sa = f' stroke="{fs}" stroke-width="{sw:.1f}"'
                svg.append(
                    f'    <g id="a{ai}"><radialGradient id="g{ai}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f'</radialGradient><circle cx="0" cy="0" r="{r:.1f}" fill="url(#g{ai})"{sa}/></g>'
                )
        else:
            # Shared gradient defs keyed by atomic number (no fog, no cmap)
            seen = set()
            for ai in range(n):
                an = a_nums[ai]
                if an in seen or ai in hidden:
                    continue
                seen.add(an)
                hi, lo = get_gradient_colors(colors[ai], cfg.gradient_strength)
                r = radii[ai] * scale
                sa = f' stroke="{cfg.atom_stroke_color}" stroke-width="{sw:.1f}"'
                svg.append(
                    f'    <g id="a{an}"><radialGradient id="g{an}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f'</radialGradient><circle cx="0" cy="0" r="{r:.1f}" fill="url(#g{an})"{sa}/></g>'
                )
        svg.append("  </defs>")

    # VdW surface defs
    vdw_set = None
    if cfg.vdw_indices is not None:
        vdw_set = set(range(n)) if len(cfg.vdw_indices) == 0 else set(cfg.vdw_indices)
        svg.append("  <defs>")
        seen_vdw = set()
        for ai in z_order:
            if ai not in vdw_set:
                continue
            an = a_nums[ai]
            if an not in seen_vdw:
                seen_vdw.add(an)
                hi = colors[ai]  # true atom color at center
                lo = colors[ai].darken(0.845 * cfg.vdw_gradient_strength)
                svg.append(
                    f'    <radialGradient id="vg{an}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f"</radialGradient>"
                )
        svg.append("  </defs>")

    # MO lobe gradient defs + front/back classification
    mo_is_front = None
    if cfg.mo_contours is not None:
        mo = cfg.mo_contours
        mo_is_front = classify_mo_lobes(mo.lobes, float(pos[:, 2].mean()))
        svg.append("  <defs>")
        svg.extend(mo_gradient_defs_svg(mo))
        svg.append("  </defs>")

    # --- Back MO orbital lobes (behind molecule) — flat faded fill ---
    if cfg.mo_contours is not None:
        assert mo_is_front is not None
        svg.extend(
            mo_back_lobes_svg(cfg.mo_contours, mo_is_front, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h)
        )

    # Interleaved z-order: for each atom, render it then its bonds to deeper atoms
    gap = cfg.bond_gap * bw  # pixel gap scales with bond width

    def add_bond(ai, aj, bo, style):
        """Render bond — closure captures shared rendering state."""
        rij = pos[aj] - pos[ai]
        dist = np.linalg.norm(rij)
        if dist < 1e-6:
            return
        d = rij / dist

        start = pos[ai] + d * radii[ai] * 0.9
        end = pos[aj] - d * radii[aj] * 0.9
        if np.dot(end - start, d) <= 0:
            return

        x1, y1 = _proj(start, scale, cx, cy, canvas_w, canvas_h)
        x2, y2 = _proj(end, scale, cx, cy, canvas_w, canvas_h)
        dx, dy = x2 - x1, y2 - y1
        ln = (dx * dx + dy * dy) ** 0.5
        if ln < 1:
            return
        px, py = -dy / ln, dx / ln

        color = cfg.bond_color
        if cfg.fog:
            avg_fog = (fog_f[ai] + fog_f[aj]) / 2 * 0.75  # bonds fog less than atoms
            color = blend_fog(color, fog_rgb, avg_fog)

        # TS/NCI override: single line with dash pattern (scaled to bond width)
        if style == BondStyle.DASHED:
            d, g = bw * 1.2, bw * 2.2
            w = bw * 1.2
            svg.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" '
                f'stroke-dasharray="{d:.1f},{g:.1f}"/>'
            )
            return
        if style == BondStyle.DOTTED:
            d, g = bw * 0.08, bw * 2
            svg.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{bw:.1f}" stroke-linecap="round" stroke-dasharray="{d:.1f},{g:.1f}"/>'
            )
            return

        is_aromatic = 1.3 < bo < 1.7
        if is_aromatic:
            # Solid + dashed parallel lines, dashed toward ring center
            side = _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h)
            w = bw * 0.7
            for ib in [-1, 1]:
                ox, oy = px * ib * gap, py * ib * gap
                dash = f' stroke-dasharray="{w * 1.0:.1f},{w * 2.0:.1f}"' if ib == side else ""
                svg.append(
                    f'  <line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" '
                    f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"{dash}/>'
                )
        else:
            nb = max(1, round(bo))
            w = bw if nb == 1 else bw * 0.7
            for ib in range(-nb + 1, nb, 2):
                ox, oy = px * ib * gap, py * ib * gap
                svg.append(
                    f'  <line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" '
                    f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"/>'
                )

    for idx, ai in enumerate(z_order):
        if ai in hidden:
            continue
        xi, yi = _proj(pos[ai], scale, cx, cy, canvas_w, canvas_h)

        # Atom
        if use_grad:
            ref = f"#a{ai}" if use_per_atom_grad else f"#a{a_nums[ai]}"
            svg.append(f'  <use x="{xi:.1f}" y="{yi:.1f}" xlink:href="{ref}"/>')
        else:
            fill, stroke = colors[ai].hex, cfg.atom_stroke_color
            if cfg.fog:
                fill = blend_fog(fill, fog_rgb, fog_f[ai])
                stroke = blend_fog(stroke, fog_rgb, fog_f[ai])
            svg.append(
                f'  <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{radii[ai] * scale:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
            )

        # Atom index label — depth-sorted with atom so nearer atoms occlude it
        if cfg.show_indices:
            fmt = cfg.idx_format
            sym = symbols[ai]
            if fmt == "sn":
                idx_text = f"{sym}{ai + 1}"
            elif fmt == "s":
                idx_text = sym
            else:  # "n"
                idx_text = str(ai + 1)
            svg.append(_text_svg(xi, yi, idx_text, fs_label, cfg.label_color, halo=False))

        # Bonds to deeper atoms
        for aj in z_order[idx + 1 :]:
            if aj in hidden or (ai, aj) not in bonds:
                continue
            bo, style = bonds[(ai, aj)]
            add_bond(ai, aj, bo, style)

    # --- Front MO orbital lobes (on top of molecule) ---
    if cfg.mo_contours is not None:
        assert mo_is_front is not None
        svg.extend(
            mo_front_lobes_svg(cfg.mo_contours, mo_is_front, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h)
        )

    # --- Density surface (stacked z-layers on top of molecule) ---
    if cfg.dens_contours is not None:
        svg.extend(dens_layers_svg(cfg.dens_contours, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h))

    # --- ESP surface (embedded heatmap on top of molecule) ---
    if cfg.esp_surface is not None:
        from xyzrender.esp import esp_surface_svg

        svg.extend(esp_surface_svg(cfg.esp_surface, scale, cx, cy, canvas_w, canvas_h, cfg.surface_opacity))

    # VdW surface overlay — on top of molecule, group opacity for proper occlusion
    if vdw_set is not None:
        svg.append(f'  <g opacity="{cfg.vdw_opacity}">')
        for ai in z_order:
            if ai in vdw_set:
                vr = raw_vdw_sphere[ai] * cfg.vdw_scale * scale
                xi, yi = _proj(pos[ai], scale, cx, cy, canvas_w, canvas_h)
                svg.append(f'    <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{vr:.1f}" fill="url(#vg{a_nums[ai]})"/>')
        svg.append("  </g>")

    # --- Annotations (bond/angle/dihedral/custom labels, always on top) ---
    has_annotations = bool(cfg.annotations)
    if has_annotations:
        svg.extend(
            _annotations_svg(
                graph, cfg, pos, hidden, scale, cx, cy, canvas_w, canvas_h, fog_f, fog_rgb, bw, fs_label, radii
            )
        )

    svg.append("</svg>")
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_canvas(pos, radii, cfg, extra_lo=None, extra_hi=None):
    """Scale + center so molecule fits canvas with tight aspect ratio."""
    pad = radii.max() if len(radii) else 0
    lo = pos[:, :2].min(axis=0) - pad
    hi = pos[:, :2].max(axis=0) + pad
    if extra_lo is not None:
        lo = np.minimum(lo, extra_lo)
    if extra_hi is not None:
        hi = np.maximum(hi, extra_hi)
    spans = hi - lo  # [x_span, y_span]
    if cfg.fixed_span is not None:
        max_span = cfg.fixed_span
    else:
        max_span = max(spans.max(), 1e-6)
    scale = (cfg.canvas_size - 2 * cfg.padding) / max_span
    if cfg.fixed_span is not None:
        # GIF mode: keep canvas square for consistent framing
        w = h = cfg.canvas_size
    else:
        # Static: crop to molecule aspect ratio
        w = int(spans[0] * scale + 2 * cfg.padding)
        h = int(spans[1] * scale + 2 * cfg.padding)
    if cfg.fixed_center is not None:
        return scale, cfg.fixed_center[0], cfg.fixed_center[1], w, h
    center = (lo + hi) / 2
    return scale, center[0], center[1], w, h


def _proj(p, scale, cx, cy, cw, ch):
    """3D position → 2D pixel coordinates (y-flipped for SVG)."""
    return cw / 2 + scale * (p[0] - cx), ch / 2 - scale * (p[1] - cy)


def _text_svg(x: float, y: float, text: str, font_size: float, color: str, *, halo: bool = True) -> str:
    """SVG <text> element, bold, with optional white halo for legibility over bond lines."""
    base = (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-family="monospace" font-size="{font_size}px" '
        f'font-weight="bold" fill="{color}" text-anchor="middle" dominant-baseline="central"'
    )
    if halo:
        sw = font_size * 0.35
        style = f"paint-order:stroke;stroke:#ffffff;stroke-width:{sw:.1f}px;stroke-linejoin:round"
        return base + f' style="{style}">{text}</text>'
    return base + f">{text}</text>"


# Palette for dihedral path segments — distinct, never white
_DIHEDRAL_PALETTE = ["#984ea3", "#458f41", "#3177b0", "#a72d2f", "#A46424"]


def _annotations_svg(
    graph,
    cfg: RenderConfig,
    pos: np.ndarray,
    hidden: set,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    fog_f: np.ndarray,
    fog_rgb: np.ndarray,
    bw: float,
    fs: float,
    radii: np.ndarray,
) -> list[str]:
    """Render all annotation elements as a flat list of SVG strings."""
    from xyzrender.annotations import AngleLabel, AtomValueLabel, BondLabel, DihedralLabel

    svg: list[str] = []
    col = cfg.label_color

    # Separate passes for each annotation type
    dihedral_idx = 0
    for ann in cfg.annotations:
        if isinstance(ann, AtomValueLabel):
            xi, yi = _proj(pos[ann.index], scale, cx, cy, canvas_w, canvas_h)
            svg.append(_text_svg(xi, yi + fs * cfg.label_offset, ann.text, fs, col))

        elif isinstance(ann, BondLabel):
            mi = (pos[ann.i] + pos[ann.j]) / 2
            mx, my = _proj(mi, scale, cx, cy, canvas_w, canvas_h)
            # Perpendicular offset so label doesn't overlap bond line
            xi, yi = _proj(pos[ann.i], scale, cx, cy, canvas_w, canvas_h)
            xj, yj = _proj(pos[ann.j], scale, cx, cy, canvas_w, canvas_h)
            dx, dy = xj - xi, yj - yi
            ln = (dx * dx + dy * dy) ** 0.5
            bl_off = fs * cfg.label_offset
            if ln > 1e-3:
                px_off, py_off = dy / ln * bl_off, -dx / ln * bl_off
            else:
                px_off, py_off = 0.0, bl_off
            svg.append(_text_svg(mx + px_off, my + py_off, ann.text, fs, col))

        elif isinstance(ann, AngleLabel):
            xi, yi = _proj(pos[ann.i], scale, cx, cy, canvas_w, canvas_h)
            xj, yj = _proj(pos[ann.j], scale, cx, cy, canvas_w, canvas_h)
            xk, yk = _proj(pos[ann.k], scale, cx, cy, canvas_w, canvas_h)

            # 2D vectors from center j toward i and k
            vi = np.array([xi - xj, yi - yj])
            vk = np.array([xk - xj, yk - yj])
            li, lk = np.linalg.norm(vi), np.linalg.norm(vk)
            if li < 1e-3 or lk < 1e-3:
                continue
            vi_hat = vi / li
            vk_hat = vk / lk

            arc_r = radii[ann.j] * scale * 1.5  # scaled with the vertex atom radius

            # Arc endpoints on the unit circle around j
            sx = xj + arc_r * vi_hat[0]
            sy = yj + arc_r * vi_hat[1]
            ex = xj + arc_r * vk_hat[0]
            ey = yj + arc_r * vk_hat[1]

            # Sweep direction: go from vi to vk the short way (inside of angle)
            cross = vi_hat[0] * vk_hat[1] - vi_hat[1] * vk_hat[0]
            sweep = 1 if cross > 0 else 0

            arc = f"M {sx:.1f},{sy:.1f} A {arc_r:.1f},{arc_r:.1f} 0 0,{sweep} {ex:.1f},{ey:.1f}"
            svg.append(
                f'  <path d="{arc}" fill="none" stroke="{col}"'
                f' stroke-width="{bw * 0.5:.1f}"'
                f' stroke-dasharray="{bw * 0.8:.1f},{bw * 1.0:.1f}" stroke-linecap="round"/>'
            )

            # Text at bisector, beyond the arc; distance scales with label_offset
            mid = vi_hat + vk_hat
            mid_len = np.linalg.norm(mid)
            if mid_len > 1e-6:
                mid_hat = mid / mid_len
            else:
                mid_hat = np.array([-vi_hat[1], vi_hat[0]])
            tx = xj + (arc_r + fs * cfg.label_offset * 0.5) * mid_hat[0]
            ty = yj + (arc_r + fs * cfg.label_offset * 0.75) * mid_hat[1]
            svg.append(_text_svg(tx, ty, ann.text, fs, col))

        elif isinstance(ann, DihedralLabel):
            seg_color = _DIHEDRAL_PALETTE[dihedral_idx % len(_DIHEDRAL_PALETTE)]
            dihedral_idx += 1

            # Draw 3 segments: i-j, j-k, k-m, each fog-blended by segment midpoint depth
            atoms_seq = [ann.i, ann.j, ann.k, ann.m]
            for seg_a, seg_b in itertools.pairwise(atoms_seq):
                xa, ya = _proj(pos[seg_a], scale, cx, cy, canvas_w, canvas_h)
                xb, yb = _proj(pos[seg_b], scale, cx, cy, canvas_w, canvas_h)
                seg_col = seg_color
                if cfg.fog:
                    avg_fog = (fog_f[seg_a] + fog_f[seg_b]) / 2 * 0.75
                    seg_col = blend_fog(seg_color, fog_rgb, avg_fog)
                svg.append(
                    f'  <line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
                    f'stroke="{seg_col}" stroke-width="{bw * 0.5:.1f}" stroke-linecap="round" '
                    f'stroke-dasharray="{bw * 1.0:.1f},{bw * 1.25:.1f}"/>'
                )

            # Text near j-k midpoint, perpendicular offset opposite to BondLabel
            mid_jk = (pos[ann.j] + pos[ann.k]) / 2
            mx, my = _proj(mid_jk, scale, cx, cy, canvas_w, canvas_h)
            xj2, yj2 = _proj(pos[ann.j], scale, cx, cy, canvas_w, canvas_h)
            xk2, yk2 = _proj(pos[ann.k], scale, cx, cy, canvas_w, canvas_h)
            ddx, ddy = xk2 - xj2, yk2 - yj2
            dln = (ddx * ddx + ddy * ddy) ** 0.5
            doff = fs * cfg.label_offset * 0.5
            if dln > 1e-3:
                dpx, dpy = -ddy / dln * doff, ddx / dln * doff
            else:
                dpx, dpy = 0.0, -doff
            svg.append(_text_svg(mx + dpx, my + dpy, ann.text, fs, col))

    return svg


def _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h):
    """Which perpendicular side (+1/-1) of the bond faces the aromatic ring center."""
    for ring in aromatic_rings:
        if ai in ring and aj in ring:
            centroid = pos[list(ring)].mean(axis=0)
            rcx, rcy = _proj(centroid, scale, cx, cy, canvas_w, canvas_h)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            return 1 if px * (rcx - mx) + py * (rcy - my) > 0 else -1
    return 1
