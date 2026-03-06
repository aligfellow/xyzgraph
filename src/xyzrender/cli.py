"""Command-line interface for xyzrender."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from xyzrender.config import build_render_config, load_config

if TYPE_CHECKING:
    from xyzrender.cube import CubeData
    from xyzrender.types import RenderConfig

from xyzrender.io import (
    detect_nci,
    load_molecule,
    load_stdin,
    load_trajectory_frames,
    load_ts_molecule,
    rotate_with_viewer,
)
from xyzrender.renderer import render_svg

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {"svg", "png", "pdf"}


def _basename(input_path: str | None, from_stdin: bool) -> str:
    """Derive output basename from input file or fallback for stdin."""
    if from_stdin or not input_path:
        return "graphic"
    return Path(input_path).stem


def _write_output(svg: str, output: str, cfg: RenderConfig, parser: argparse.ArgumentParser) -> None:
    """Write SVG to file, converting to PNG/PDF based on extension."""
    ext = output.rsplit(".", 1)[-1].lower()
    if ext == "svg":
        with open(output, "w") as f:
            f.write(svg)
    elif ext == "png":
        from xyzrender.export import svg_to_png

        svg_to_png(svg, output, size=cfg.canvas_size, dpi=cfg.dpi)
    elif ext == "pdf":
        from xyzrender.export import svg_to_pdf

        svg_to_pdf(svg, output)
    else:
        supported = ", ".join("." + e for e in sorted(_SUPPORTED_EXTENSIONS))
        parser.error(f"Unsupported output format: .{ext} (use {supported})")


def _parse_pairs(s: str) -> list[tuple[int, int]]:
    """Parse '1-6,3-4' → [(0,5), (2,3)] (1-indexed input → 0-indexed)."""
    if not s.strip():
        return []
    pairs = []
    for part in s.split(","):
        a, b = part.split("-")
        pairs.append((int(a) - 1, int(b) - 1))
    return pairs


def _parse_indices(s: str) -> list[int]:
    """Parse '1-20,25,30' → [0..19, 24, 29] (1-indexed input → 0-indexed)."""
    if not s.strip():
        return []
    indices = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            indices.extend(range(int(a) - 1, int(b)))
        else:
            indices.append(int(part) - 1)
    return indices


def main() -> None:
    """Entry point for the CLI."""
    p = argparse.ArgumentParser(
        prog="xyzrender", description="Publication-quality molecular graphics from the command line."
    )

    # --- Input / Output ---
    io_g = p.add_argument_group("input/output")
    io_g.add_argument("input", nargs="?", help="XYZ file (reads stdin if omitted)")
    io_g.add_argument("-o", "--output", help="Output file (.svg, .png, .pdf)")
    io_g.add_argument("-c", "--charge", type=int, default=0)
    io_g.add_argument("-m", "--multiplicity", type=int, default=None)
    io_g.add_argument("-d", "--debug", action="store_true", help="Debug output")

    # --- Styling ---
    style_g = p.add_argument_group("styling")
    style_g.add_argument("--config", default=None, help="Config preset or JSON path (default, flat, custom)")
    style_g.add_argument("-S", "--canvas-size", type=int, default=None)
    style_g.add_argument("-a", "--atom-scale", type=float, default=None)
    style_g.add_argument("-b", "--bond-width", type=float, default=None)
    style_g.add_argument("-s", "--atom-stroke-width", type=float, default=None)
    style_g.add_argument("--bond-color", default=None, help="Bond color (hex or named)")
    style_g.add_argument("-B", "--background", default=None)
    style_g.add_argument("-t", "--transparent", action="store_true", help="Transparent background")
    style_g.add_argument("-G", "--gradient-strength", type=float, default=None, help="Gradient contrast")
    style_g.add_argument("--grad", action=argparse.BooleanOptionalAction, default=None, help="Radial gradients")
    style_g.add_argument("-F", "--fog-strength", type=float, default=None, help="Fog strength")
    style_g.add_argument("--vdw-opacity", type=float, default=None, help="VdW sphere opacity")
    style_g.add_argument("--vdw-scale", type=float, default=None, help="VdW sphere radius scale")
    style_g.add_argument("--vdw-gradient", type=float, default=None, help="VdW sphere gradient strength")

    # --- Display ---
    disp_g = p.add_argument_group("display")
    disp_g.add_argument("--hy", nargs="*", type=int, default=None, help="Show H atoms (no args=all, or 1-indexed)")
    disp_g.add_argument("--no-hy", action="store_true", default=False, help="Hide all H atoms")
    disp_g.add_argument("--bo", action=argparse.BooleanOptionalAction, default=None, help="Bond orders")
    disp_g.add_argument(
        "-k", "--kekule", action="store_true", default=False, help="Use Kekule bond orders (no aromatic 1.5)"
    )
    disp_g.add_argument("--fog", action=argparse.BooleanOptionalAction, default=None, help="Depth fog")
    disp_g.add_argument("--vdw", nargs="?", const="", default=None, help='VdW spheres (no args=all, or "1-20,25")')

    # --- Surfaces (MO / density / ESP) ---
    surf_g = p.add_argument_group("surfaces")
    surf_g.add_argument("--mo", action="store_true", default=False, help="Render MO lobes from .cube input")
    surf_g.add_argument(
        "--mo-colors",
        nargs=2,
        default=None,
        metavar=("POS", "NEG"),
        help="MO lobe colors as hex or named color (default: steelblue maroon)",
    )
    surf_g.add_argument("--dens", action="store_true", default=False, help="Render density isosurface from .cube input")
    surf_g.add_argument("--dens-color", default=None, help="Density surface color (hex or named, default: steelblue)")
    surf_g.add_argument(
        "--esp", default=None, metavar="CUBE", help="ESP cube file for potential coloring (implies --dens)"
    )
    surf_g.add_argument(
        "--iso", type=float, default=None, help="Isosurface threshold (MO default: 0.05, density/ESP default: 0.001)"
    )
    surf_g.add_argument("--opacity", type=float, default=None, help="Surface opacity (default: 1.0, >1 boosts)")

    # --- Orientation ---
    orient_g = p.add_argument_group("orientation")
    orient_g.add_argument(
        "--orient", action=argparse.BooleanOptionalAction, default=None, help="Auto-orientation (default: on)"
    )
    orient_g.add_argument("-I", "--interactive", action="store_true", help="Open in v viewer for interactive rotation")

    # --- TS / NCI ---
    ts_g = p.add_argument_group("transition state / NCI")
    ts_g.add_argument("--ts", action="store_true", dest="ts_detect", help="Auto-detect TS bonds via graphRC")
    ts_g.add_argument("--ts-frame", type=int, default=0, help="TS reference frame for graphRC (0-indexed)")
    ts_g.add_argument("--ts-bond", default="", help='Manual TS bond pair(s), 1-indexed: "1-6,3-4"')
    ts_g.add_argument("--nci", action="store_true", help="Auto-detect NCI interactions via xyzgraph")
    ts_g.add_argument("--nci-bond", default="", help='Manual NCI bond pair(s), 1-indexed: "1-5,2-8"')

    # --- GIF animation ---
    gif_g = p.add_argument_group("GIF animation")
    gif_g.add_argument("--gif-ts", action="store_true", help="TS vibration GIF (via graphRC)")
    gif_g.add_argument("--gif-trj", action="store_true", help="Trajectory/optimization GIF (multi-frame input)")
    gif_g.add_argument(
        "--gif-rot",
        nargs="?",
        const="y",
        default=None,
        help="Rotation GIF (default axis: y). Combinable with --gif-ts.",
    )
    gif_g.add_argument("-go", "--gif-output", default=None, help="GIF output path")
    gif_g.add_argument("--gif-fps", type=int, default=10, help="GIF frames per second (default: 10)")
    gif_g.add_argument("--rot-frames", type=int, default=120, help="Rotation frames (default: 120)")

    # --- Measurements & annotations ---
    annot_g = p.add_argument_group("measurements & annotations")
    annot_g.add_argument(
        "--label-size", type=float, default=None, metavar="PT", help="Label font size (overrides preset)"
    )
    annot_g.add_argument(
        "--measure",
        nargs="*",
        default=None,
        metavar="TYPE",
        help="Print bond measurements to stdout: d (distances), a (angles), t (dihedrals). "
        "Combine: --measure d a. Omit types for all.",
    )
    annot_g.add_argument(
        "--idx",
        nargs="?",
        const="sn",
        default=None,
        metavar="FMT",
        help="Label all atoms with index in SVG: sn (C1, default), s (C only), n (number only)",
    )
    annot_g.add_argument(
        "-l",
        dest="label_specs",
        nargs="+",
        action="append",
        default=None,
        metavar="TOKEN",
        help=(
            "Annotate SVG (repeatable): "
            '"-l 1 2 d" (bond distance), "-l 2 a" (all angles at atom 2), '
            '"-l 1 2 3 4 t" (dihedral polyline), "-l 1 +0.5" (custom atom label), '
            '"-l 1 2 NBO" (custom bond label). Indices 1-based.'
        ),
    )
    annot_g.add_argument(
        "--label",
        default=None,
        metavar="FILE",
        help="Annotation file (same spec syntax as -l, one per line, # comments, comma or space separated)",
    )
    annot_g.add_argument(
        "--cmap",
        default=None,
        metavar="FILE",
        help="Atom property colormap file: two columns (1-indexed atom index, value); header lines are skipped",
    )
    annot_g.add_argument(
        "--cmap-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="Explicit colormap range (default: auto from file values)",
    )

    # --- Crystal / periodic structures ---
    crystal_g = p.add_argument_group("crystal / periodic structures")
    crystal_g.add_argument(
        "--crystal",
        nargs="?",
        const="auto",
        default=None,
        metavar="{vasp,qe}",
        help=(
            "Load as a periodic crystal structure via phonopy (requires xyzrender[crystal]). "
            "Enables unit cell box, crystallographic axes (a/b/c), and image atoms. "
            "Optionally specify the phonopy interface: vasp, qe (auto-detected from filename "
            "if omitted). Examples: --crystal, --crystal vasp, --crystal qe"
        ),
    )
    crystal_g.add_argument(
        "--cell",
        action="store_true",
        default=False,
        help=(
            "Draw the unit cell box from an extXYZ Lattice= header. "
            "No crystallographic axes or image atoms — just the box. "
            "Does not require phonopy."
        ),
    )
    crystal_g.add_argument("--no-cell", action="store_true", default=False, help="Hide the unit cell box (--crystal)")
    crystal_g.add_argument(
        "--no-ghosts", action="store_true", default=False, help="Hide image atoms outside the cell (--crystal)"
    )
    crystal_g.add_argument(
        "--axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show/hide crystallographic axis arrows a/b/c (--crystal)",
    )
    crystal_g.add_argument("--cell-color", default=None, help="Unit cell box color (hex or named, default: #333333)")
    crystal_g.add_argument("--cell-width", type=float, default=None, help="Unit cell box line width (default: 1.5)")
    crystal_g.add_argument(
        "--ghost-opacity",
        type=float,
        default=None,
        help="Opacity for ghost (periodic image) atoms/bonds (default: 0.5)",
    )
    crystal_g.add_argument(
        "--axis",
        default=None,
        metavar="HKL",
        help=(
            "Orient crystal looking down a crystallographic direction, e.g. --axis 111. "
            "Each digit is one Miller index (0-9). Requires --crystal."
        ),
    )

    args = p.parse_args()
    from xyzrender import configure_logging

    configure_logging(verbose=True, debug=args.debug)

    # Build config: preset/JSON base + CLI overrides
    config_data = load_config(args.config or "default")

    cli_overrides: dict = {}
    for attr, key in [
        ("canvas_size", "canvas_size"),
        ("atom_scale", "atom_scale"),
        ("bond_width", "bond_width"),
        ("atom_stroke_width", "atom_stroke_width"),
        ("bond_color", "bond_color"),
        ("gradient_strength", "gradient_strength"),
        ("transparent", "transparent"),
        ("fog_strength", "fog_strength"),
        ("background", "background"),
        ("vdw_opacity", "vdw_opacity"),
        ("vdw_scale", "vdw_scale"),
        ("label_size", "label_font_size"),
    ]:
        val = getattr(args, attr)
        if val is not None:
            cli_overrides[key] = val
    if args.vdw_gradient is not None:
        cli_overrides["vdw_gradient_strength"] = args.vdw_gradient
    if args.grad is not None:
        cli_overrides["gradient"] = args.grad
    if args.fog is not None:
        cli_overrides["fog"] = args.fog
    if args.bo is not None:
        cli_overrides["bond_orders"] = args.bo

    cfg = build_render_config(config_data, cli_overrides)

    # Per-molecule settings (always from CLI)
    if args.no_hy:
        cfg.hide_h = True  # --no-hy: hide all H
    elif args.hy is None:
        cfg.hide_h = True  # default: hide C-H
    elif len(args.hy) == 0:
        cfg.hide_h = False  # --hy with no args: show all
    else:
        cfg.hide_h = True  # --hy 1 3 5: show specific only
        cfg.show_h_indices = [i - 1 for i in args.hy]
    cfg.ts_bonds = _parse_pairs(args.ts_bond)
    cfg.nci_bonds = _parse_pairs(args.nci_bond)
    cfg.vdw_indices = (
        _parse_indices(args.vdw) if args.vdw is not None and args.vdw != "" else ([] if args.vdw == "" else None)
    )
    # Auto-orient: on by default, off for interactive/stdin
    from_stdin = not args.input and not sys.stdin.isatty()
    if args.orient is not None:
        cfg.auto_orient = args.orient
    elif args.interactive or from_stdin:
        cfg.auto_orient = False
    else:
        cfg.auto_orient = True

    # Output path defaults and validation
    base = _basename(args.input, from_stdin)
    if not args.output:
        args.output = f"{base}.svg"

    static_ext = args.output.rsplit(".", 1)[-1].lower()
    if static_ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join("." + e for e in sorted(_SUPPORTED_EXTENSIONS))
        p.error(f"Unsupported static output format: .{static_ext} (use {supported})")

    wants_gif = args.gif_ts or args.gif_rot or args.gif_trj

    # Warn when annotation flags (static-SVG only) are combined with GIF output
    annotation_flags_used = args.idx is not None or args.label_specs or args.label
    if annotation_flags_used and wants_gif:
        print(
            "Warning: --idx, -l and --label apply to static SVG output only and will not appear in the GIF.",
            file=sys.stderr,
        )

    if args.rot_frames != 120 and not args.gif_rot:
        logger.warning("--rot-frames ignored without --gif-rot")
    if args.gif_ts and args.gif_trj:
        p.error(
            "--gif-ts and --gif-trj are mutually exclusive. "
            "Use --gif-trj with --ts if you want TS bonds shown in the trj gif"
        )

    if args.transparent and args.background is not None:
        logger.warning("--transparent and --background are mutually exclusive; using transparent")
        args.background = None

    is_cube = args.input and args.input.endswith(".cube")

    # --- Crystal / periodic structure detection ---
    # interface_mode is non-None iff phonopy crystal loading is requested (--crystal flag).
    # --cell is a separate, lighter path: extXYZ box only, no phonopy, no image atoms.
    interface_mode: str | None = None
    if args.crystal is not None:
        if args.input is None:
            p.error("--crystal requires an input file")
        if args.crystal in {"vasp", "qe"}:
            interface_mode = args.crystal
        else:
            # "auto" (bare --crystal) or unrecognised value: detect from filename
            p_in = Path(args.input)
            stem = p_in.stem.upper()
            ext = p_in.suffix.lower()
            if ext == ".vasp" or stem in {"POSCAR", "CONTCAR"}:
                interface_mode = "vasp"
            elif ext == ".in":
                interface_mode = "qe"
            else:
                p.error(
                    f"Cannot auto-detect crystal interface from {args.input!r}. "
                    "Specify explicitly: --crystal vasp or --crystal qe"
                )

    # MO mutual exclusivity
    if args.mo and args.vdw is not None:
        p.error("--mo and --vdw are mutually exclusive")
    if args.mo and args.gif_ts:
        p.error("--mo and --gif-ts are mutually exclusive")
    if args.mo and args.gif_trj:
        p.error("--mo and --gif-trj are mutually exclusive")
    if args.mo and not is_cube:
        p.error("--mo requires a .cube input file")
    if args.dens and args.mo:
        p.error("--dens and --mo are mutually exclusive")
    if args.dens and args.vdw is not None:
        p.error("--dens and --vdw are mutually exclusive")
    if args.dens and not is_cube:
        p.error("--dens requires a .cube input file")
    if args.esp and args.mo:
        p.error("--esp and --mo are mutually exclusive")
    if args.esp and args.vdw is not None:
        p.error("--esp and --vdw are mutually exclusive")
    if args.esp and args.dens:
        p.error("--esp and --dens are mutually exclusive (--esp implies density rendering)")
    if args.esp and not is_cube:
        p.error("--esp requires a .cube density input file")
    if args.esp and wants_gif:
        p.error("--esp does not support GIF rotation")
    if wants_gif:
        gif_path = args.gif_output or f"{base}.gif"
        gif_ext = gif_path.rsplit(".", 1)[-1].lower()
        if gif_ext != "gif":
            p.error(f"GIF output must have .gif extension, got: .{gif_ext}")

    # Load molecule (--gif-ts implies TS detection)
    cube_data = None
    crystal_data = None
    needs_ts = args.ts_detect or args.gif_ts
    if is_cube and needs_ts:
        print(
            "Warning: --ts/--gif-ts has no effect with cube files (single geometry, no frequency data). "
            "Use --ts-bond to manually specify TS bonds."
        )
    if is_cube:
        from xyzrender.io import load_cube

        graph, cube_data = load_cube(args.input, charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
    elif interface_mode is not None:
        from xyzrender.crystal import add_crystal_images, load_crystal

        graph, crystal_data = load_crystal(args.input, interface_mode)
        if not args.no_ghosts:
            add_crystal_images(graph, crystal_data)
    elif needs_ts and args.input:
        graph, _ts_frames = load_ts_molecule(
            args.input,
            charge=args.charge,
            multiplicity=args.multiplicity,
            ts_frame=args.ts_frame,
            kekule=args.kekule,
        )
    elif args.input:
        graph = load_molecule(args.input, charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
    elif not sys.stdin.isatty():
        graph = load_stdin(charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
    else:
        p.error("No input file and stdin is a terminal")

    # Post-load analysis
    if args.nci:
        graph = detect_nci(graph)

    # --- Crystal config: phonopy path (--crystal) ---
    if crystal_data is not None:
        cfg.crystal_data = crystal_data
        cfg.show_cell = not args.no_cell
        cfg.show_crystal_axes = args.axes
        # PCA auto-orient makes no sense for full periodic crystals
        if args.orient is None:
            cfg.auto_orient = False
        if args.cell_color is not None:
            from xyzrender.types import resolve_color

            cfg.cell_color = resolve_color(args.cell_color)
        if args.cell_width is not None:
            cfg.cell_line_width = args.cell_width
        if args.ghost_opacity is not None:
            cfg.periodic_image_opacity = args.ghost_opacity

        # --axis HKL: orient so that [hkl] points along the viewing (+z) axis
        if args.axis is not None:
            import numpy as np

            hkl = args.axis.lstrip("-")
            if not (hkl.isdigit() and len(hkl) >= 3):
                p.error(f"--axis: expected a 3-digit Miller index string (e.g. 111, 100), got {args.axis!r}")
            h, k_idx, l_idx = int(hkl[0]), int(hkl[1]), int(hkl[2])
            lat = crystal_data.lattice
            v = h * lat[0] + k_idx * lat[1] + l_idx * lat[2]
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-10:
                p.error(f"--axis [{hkl}] has zero length")
            v = v / v_norm
            # Rodrigues rotation: map v → (0,0,1) (looking down the hkl direction)
            z = np.array([0.0, 0.0, 1.0])
            cos_a = float(np.clip(np.dot(v, z), -1.0, 1.0))
            if abs(cos_a - 1.0) < 1e-9:
                rot_view: np.ndarray = np.eye(3)
            elif abs(cos_a + 1.0) < 1e-9:
                rot_view = np.diag([1.0, -1.0, -1.0])
            else:
                ax = np.cross(v, z)
                ax /= np.linalg.norm(ax)
                s_a = float(np.sqrt(max(0.0, 1.0 - cos_a**2)))
                ax_cross = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                rot_view = cos_a * np.eye(3) + s_a * ax_cross + (1 - cos_a) * np.outer(ax, ax)
            node_ids_v = list(graph.nodes())
            pos_v = np.array([graph.nodes[i]["position"] for i in node_ids_v], dtype=float)
            centroid_v = pos_v.mean(axis=0)
            pos_rotated_v = (rot_view @ (pos_v - centroid_v).T).T + centroid_v
            for idx, nid in enumerate(node_ids_v):
                graph.nodes[nid]["position"] = tuple(pos_rotated_v[idx].tolist())
            crystal_data.lattice = (rot_view @ crystal_data.lattice.T).T
            crystal_data.cell_origin = rot_view @ (crystal_data.cell_origin - centroid_v) + centroid_v
            cfg.auto_orient = False

    # --- Cell box: extXYZ Lattice= path (--cell, no phonopy) ---
    elif args.cell:
        if "lattice" not in graph.graph:
            logger.warning("--cell: no Lattice= found in input file, cell box will not be drawn")
        else:
            import numpy as np

            from xyzrender.types import CrystalData

            cfg.crystal_data = CrystalData(
                lattice=np.array(graph.graph["lattice"], dtype=float),
                cell_origin=np.array(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float),
            )
            cfg.show_cell = True
            cfg.show_crystal_axes = False  # --cell = box only; axes are shown for --crystal only
            if args.cell_color is not None:
                from xyzrender.types import resolve_color

                cfg.cell_color = resolve_color(args.cell_color)
            if args.cell_width is not None:
                cfg.cell_line_width = args.cell_width
    elif "lattice" in graph.graph:
        logger.info("Lattice= found in file; use --cell to draw the unit cell box")

    # Default --no-bo for any periodic input (phonopy crystal or extXYZ cell box).
    # Bond orders from xyzgraph are not PBC-aware so they are unreliable for periodic structures.
    if cfg.crystal_data is not None and args.bo is None:
        cfg.bond_orders = False
        logger.warning("Periodic structure: bond orders disabled by default (use --bo to override)")

    # Measurements (terminal only — no SVG effect)
    if args.measure is not None:
        from xyzrender.measure import print_measurements

        modes = args.measure if args.measure else ["all"]  # bare --measure means all
        valid_modes = {"all", "d", "a", "t", "tor", "dih"}
        for m in modes:
            if m.lower() not in valid_modes:
                p.error(f"--measure: unknown type {m!r} (valid: d, a, t, or omit for all)")
        print_measurements(graph, modes)

    # Atom index labels in SVG
    if args.idx is not None:
        valid_fmts = {"sn", "s", "n"}
        if args.idx not in valid_fmts:
            p.error(f"--idx: unknown format {args.idx!r} (valid: sn, s, n)")
        cfg.show_indices = True
        cfg.idx_format = args.idx

    # SVG annotations (-l / --label)
    if args.label_specs or args.label:
        from xyzrender.annotations import parse_annotations

        try:
            cfg.annotations = parse_annotations(
                inline_specs=args.label_specs or [],
                file_path=args.label,
                graph=graph,
            )
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))

    # Atom property colormap
    if args.cmap:
        from xyzrender.annotations import load_cmap

        try:
            cfg.atom_cmap = load_cmap(args.cmap, graph)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))
        if args.cmap_range is not None:
            cfg.cmap_range = tuple(args.cmap_range)

    # Orientation
    if args.interactive:
        rotate_with_viewer(graph)

    # Surface opacity (shared across MO / density / ESP)
    if args.opacity is not None:
        cfg.surface_opacity = args.opacity

    # MO: resolve colors once (used for both static render and gif-rot)
    mo_colors = None
    if args.mo:
        from xyzrender.types import resolve_color

        mo_colors = args.mo_colors or [
            config_data.get("mo_pos_color", "steelblue"),
            config_data.get("mo_neg_color", "maroon"),
        ]
        mo_colors = [resolve_color(c) for c in mo_colors]

    # MO contour computation (PCA must happen here so rot is available for the grid)
    if args.mo and cube_data is not None:
        from typing import cast

        import numpy as np

        from xyzrender.mo import build_mo_contours
        from xyzrender.utils import kabsch_rotation, pca_orient

        assert mo_colors is not None  # set when args.mo is True
        cube = cast("CubeData", cube_data)
        rot = None
        if cfg.auto_orient:
            node_ids = list(graph.nodes())
            pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
            oriented, rot = pca_orient(pos, return_matrix=True)

            # Tilt -30° around x-axis so orbital lobes above/below the
            # molecular plane are clearly separated in the projection.
            tilt = np.radians(-30)
            rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(tilt), -np.sin(tilt)],
                    [0, np.sin(tilt), np.cos(tilt)],
                ]
            )
            rot = rx @ rot
            oriented = oriented @ rx.T

            for idx, nid in enumerate(node_ids):
                graph.nodes[nid]["position"] = tuple(oriented[idx].tolist())
            cfg.auto_orient = False  # already applied, don't re-apply in render_svg

        # If positions were modified (e.g. by interactive viewer) but PCA was not
        # applied, compute R from the correspondence between original cube atom
        # positions and current graph positions (Kabsch algorithm).
        if rot is None:
            orig = np.array([p for _, p in cube.atoms], dtype=float)
            curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
            if not np.allclose(orig, curr, atol=1e-6):
                rot = kabsch_rotation(orig, curr)

        # Centroids for aligning the orbital grid with atom positions
        atom_centroid = np.array([p for _, p in cube.atoms], dtype=float).mean(axis=0)
        node_ids_mo = list(graph.nodes())
        curr_centroid = np.array(
            [graph.nodes[i]["position"] for i in node_ids_mo],
            dtype=float,
        ).mean(axis=0)

        cfg.mo_contours = build_mo_contours(
            cube,
            rot=rot,
            isovalue=args.iso if args.iso is not None else 0.05,
            pos_color=mo_colors[0],
            neg_color=mo_colors[1],
            atom_centroid=atom_centroid,
            target_centroid=curr_centroid,
        )

    # Density isosurface computation
    if args.dens and cube_data is not None:
        from typing import cast

        import numpy as np

        from xyzrender.dens import build_density_contours
        from xyzrender.types import resolve_color
        from xyzrender.utils import kabsch_rotation, pca_orient

        cube = cast("CubeData", cube_data)
        dens_color = resolve_color(args.dens_color or config_data.get("dens_color", "steelblue"))
        dens_isovalue = args.iso if args.iso is not None else config_data.get("dens_iso", 0.001)

        rot = None
        if cfg.auto_orient:
            node_ids = list(graph.nodes())
            pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
            oriented, rot = pca_orient(pos, return_matrix=True)
            # No tilt for density (unlike MO which tilts -30 degrees)
            for idx, nid in enumerate(node_ids):
                graph.nodes[nid]["position"] = tuple(oriented[idx].tolist())
            cfg.auto_orient = False

        if rot is None:
            orig = np.array([p for _, p in cube.atoms], dtype=float)
            curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
            if not np.allclose(orig, curr, atol=1e-6):
                rot = kabsch_rotation(orig, curr)

        atom_centroid = np.array([p for _, p in cube.atoms], dtype=float).mean(axis=0)
        node_ids_dens = list(graph.nodes())
        curr_centroid = np.array(
            [graph.nodes[i]["position"] for i in node_ids_dens],
            dtype=float,
        ).mean(axis=0)

        cfg.dens_contours = build_density_contours(
            cube,
            isovalue=dens_isovalue,
            color=dens_color,
            rot=rot,
            atom_centroid=atom_centroid,
            target_centroid=curr_centroid,
        )

    # ESP surface computation
    esp_cube_data = None
    if args.esp and cube_data is not None:
        from typing import cast

        import numpy as np

        from xyzrender.cube import parse_cube
        from xyzrender.esp import _compute_normals_phys, build_esp_surface
        from xyzrender.utils import kabsch_rotation, pca_orient

        cube = cast("CubeData", cube_data)
        esp_cube_data = parse_cube(args.esp)

        # Verify grid shapes match
        if cube.grid_shape != esp_cube_data.grid_shape:
            p.error(
                f"Grid shape mismatch: density cube {cube.grid_shape} vs ESP cube {esp_cube_data.grid_shape}. "
                "Both cubes must come from the same calculation."
            )

        esp_isovalue = args.iso if args.iso is not None else config_data.get("dens_iso", 0.001)

        rot = None
        if cfg.auto_orient:
            node_ids = list(graph.nodes())
            pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
            oriented, rot = pca_orient(pos, return_matrix=True)
            # No tilt for ESP (same as density)
            for idx, nid in enumerate(node_ids):
                graph.nodes[nid]["position"] = tuple(oriented[idx].tolist())
            cfg.auto_orient = False

        if rot is None:
            orig = np.array([p for _, p in cube.atoms], dtype=float)
            curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
            if not np.allclose(orig, curr, atol=1e-6):
                rot = kabsch_rotation(orig, curr)

        atom_centroid = np.array([p for _, p in cube.atoms], dtype=float).mean(axis=0)
        node_ids_esp = list(graph.nodes())
        curr_centroid = np.array(
            [graph.nodes[i]["position"] for i in node_ids_esp],
            dtype=float,
        ).mean(axis=0)

        normals_phys = _compute_normals_phys(cube)
        cfg.esp_surface = build_esp_surface(
            cube,
            esp_cube_data,
            isovalue=esp_isovalue,
            rot=rot,
            atom_centroid=atom_centroid,
            target_centroid=curr_centroid,
            normals_phys=normals_phys,
        )

    # Render static output
    svg = render_svg(graph, cfg)
    _write_output(svg, args.output, cfg, p)

    # GIF output
    if wants_gif:
        from xyzrender.gif import (
            ROTATION_AXES,
            render_rotation_gif,
            render_trajectory_gif,
            render_vibration_gif,
            render_vibration_rotation_gif,
        )

        if args.gif_rot and args.gif_rot not in ROTATION_AXES:
            # Allow crystallographic hkl format (3+ digit string) for crystal inputs
            _test_ax = args.gif_rot.lstrip("-")
            if not (_test_ax.isdigit() and len(_test_ax) >= 3 and cfg.crystal_data is not None):
                p.error(
                    f"Invalid rotation axis: {args.gif_rot!r} "
                    f"(valid: {', '.join(ROTATION_AXES)}, or 3-digit hkl for crystal inputs)"
                )

        if args.gif_ts and args.gif_rot:
            if not args.input:
                p.error("--gif-ts requires an input file")
            render_vibration_rotation_gif(
                args.input,
                cfg,
                gif_path,
                charge=args.charge,
                multiplicity=args.multiplicity,
                ts_frame=args.ts_frame,
                fps=args.gif_fps,
                axis=args.gif_rot,
                n_frames=args.rot_frames,
                reference_graph=graph,
                detect_nci=args.nci,
            )
        elif args.gif_ts:
            if not args.input:
                p.error("--gif-ts requires an input file")
            render_vibration_gif(
                args.input,
                cfg,
                gif_path,
                charge=args.charge,
                multiplicity=args.multiplicity,
                ts_frame=args.ts_frame,
                fps=args.gif_fps,
                reference_graph=graph,
                detect_nci=args.nci,
            )
        elif args.gif_trj:
            if not args.input:
                p.error("--gif-trj requires an input file")
            frames = load_trajectory_frames(args.input)
            if len(frames) < 2:
                p.error("--gif-trj requires multi-frame input")
            render_trajectory_gif(
                frames,
                cfg,
                gif_path,
                charge=args.charge,
                multiplicity=args.multiplicity,
                fps=args.gif_fps,
                reference_graph=graph,
                detect_nci=args.nci,
                axis=args.gif_rot or None,
                kekule=args.kekule,
            )
        elif args.gif_rot:
            mo_data = None
            if args.mo and cube_data is not None:
                assert mo_colors is not None  # set when args.mo is True
                mo_data = {
                    "cube_data": cube_data,
                    "isovalue": args.iso if args.iso is not None else 0.05,
                    "pos_color": mo_colors[0],
                    "neg_color": mo_colors[1],
                    "surface_opacity": cfg.surface_opacity,
                }
            dens_data = None
            if args.dens and cube_data is not None:
                from xyzrender.types import resolve_color

                dens_data = {
                    "cube_data": cube_data,
                    "isovalue": args.iso if args.iso is not None else config_data.get("dens_iso", 0.001),
                    "color": resolve_color(args.dens_color or config_data.get("dens_color", "steelblue")),
                    "surface_opacity": cfg.surface_opacity,
                }
            render_rotation_gif(
                graph,
                cfg,
                gif_path,
                n_frames=args.rot_frames,
                fps=args.gif_fps,
                axis=args.gif_rot,
                mo_data=mo_data,
                dens_data=dens_data,
            )


if __name__ == "__main__":
    main()
