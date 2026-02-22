"""Command-line interface."""

import argparse

from . import (
    __citation__,
    __version__,
    build_graph,
    build_graph_orca,
    build_graph_rdkit,
    build_graph_rdkit_tm,
    compare_with_rdkit,
    graph_debug_report,
    graph_to_ascii,
    read_xyz_file,
)
from .config import DEFAULT_PARAMS
from .graph_builders import compute_metadata
from .utils import _parse_pairs, configure_debug_logging, graph_to_dict


def print_header(input_file, params_used, frame_info=None):
    """Print formatted header with version, citation, and parameter information."""
    import os
    import textwrap

    print("=" * 80)
    print(" " * 31 + "XYZGRAPH")
    print(" " * 10 + "Molecular Graph Construction from Cartesian Coordinates")
    print(" " * 26 + "A. S. Goodfellow, 2025")
    print("=" * 80)
    print()
    print(f"Version:        xyzgraph v{__version__}")

    wrapped_citation = textwrap.fill(
        __citation__,
        width=80,
        initial_indent="Citation:       ",
        subsequent_indent="                ",
    )
    print(wrapped_citation)

    input_str = os.path.basename(input_file)
    if frame_info is not None:
        input_str += f" (frame {frame_info})"
    print(f"Input:          {input_str}")

    if params_used:
        params_str = ", ".join(f"{k}={v}" for k, v in params_used.items())
        wrapped_params = textwrap.fill(
            params_str,
            width=80,
            initial_indent="Parameters:     ",
            subsequent_indent="                ",
        )
        print(wrapped_params)

    print()
    print("=" * 80)
    print()


def display_graph(G, args, show_h_indices, label=""):
    """Display graph with debug report and/or ASCII visualization."""
    if args.debug:
        if label:
            print(f"\n{'=' * 80}")
            print(f"# {label.upper()} GRAPH DETAILS")
            print("=" * 80)
        print(graph_debug_report(G, include_h=args.show_h, show_h_indices=show_h_indices))

    # Determine if ASCII should be shown
    has_explicit_output = args.ascii or args.compare_rdkit or args.compare_rdkit_tm or args.orca_out
    show_ascii = (args.ascii or not has_explicit_output) and not args.nci

    if show_ascii:
        title = f"# ASCII Depiction ({label})" if label else "# ASCII Depiction"
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")
        ascii_out, _ = graph_to_ascii(
            G,
            scale=max(0.2, args.ascii_scale),
            include_h=args.show_h,
            show_h_indices=show_h_indices,
        )
        print(ascii_out)

    return show_ascii


def display_ncis(G, args, show_h_indices):
    """Run NCI detection, print summary, and render ASCII with NCI dotted lines."""
    from .nci import detect_ncis
    from .nci.display import format_nci_table, render_nci_ascii

    ncis = detect_ncis(G)
    print(format_nci_table(G, ncis, debug=args.debug))
    print(render_nci_ascii(
        G, ncis,
        scale=max(0.2, args.ascii_scale),
        include_h=args.show_h,
        show_h_indices=show_h_indices,
    ))


def compare_graphs(G1, G2, label1, label2):
    """Compare two graphs and print diff summary."""
    print(f"\n{'=' * 80}")
    print(f"# GRAPH COMPARISON: {label1} vs {label2}")
    print("=" * 80)

    edges1 = {tuple(sorted((i, j))) for i, j in G1.edges()}
    edges2 = {tuple(sorted((i, j))) for i, j in G2.edges()}

    only_1 = edges1 - edges2
    only_2 = edges2 - edges1
    shared = edges1 & edges2

    print(f"# {label1}: {G1.number_of_nodes()} atoms, {G1.number_of_edges()} bonds")
    print(f"# {label2}: {G2.number_of_nodes()} atoms, {G2.number_of_edges()} bonds")
    print(f"# Shared bonds: {len(shared)}")
    print(f"# Only in {label1}: {len(only_1)}")
    print(f"# Only in {label2}: {len(only_2)}")

    if only_1:
        print(f"\n# Bonds only in {label1} (first 20):")
        for i, j in sorted(only_1)[:20]:
            si, sj = G1.nodes[i]["symbol"], G1.nodes[j]["symbol"]
            bo = G1[i][j]["bond_order"]
            print(f"#   {si}{i}-{sj}{j} (BO={bo:.2f})")

    if only_2:
        print(f"\n# Bonds only in {label2} (first 20):")
        for i, j in sorted(only_2)[:20]:
            si, sj = G2.nodes[i]["symbol"], G2.nodes[j]["symbol"]
            bo = G2[i][j]["bond_order"]
            print(f"#   {si}{i}-{sj}{j} (BO={bo:.2f})")


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Build molecular graph from XYZ or ORCA output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_file", nargs="?", help="Input file (XYZ or ORCA .out)")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    p.add_argument("--citation", action="store_true", help="Print citation and exit")

    # --- Common Options ---
    common = p.add_argument_group("Common Options")
    common.add_argument(
        "--method",
        choices=["cheminf", "xtb"],
        default=DEFAULT_PARAMS["method"],
        help=f"Graph construction method (default: {DEFAULT_PARAMS['method']})",
    )
    common.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep temporary xTB files (only for --method xtb)",
    )
    common.add_argument(
        "-c",
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge (default: 0)",
    )
    common.add_argument(
        "-m",
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity (default: auto estimation)",
    )
    common.add_argument(
        "-q",
        "--quick",
        action="store_true",
        default=DEFAULT_PARAMS["quick"],
        help="Quick mode: connectivity only, no formal charge optimization",
    )
    common.add_argument(
        "--relaxed",
        action="store_true",
        default=DEFAULT_PARAMS["relaxed"],
        help="Relaxed geometric validation (for transition states)",
    )
    common.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1.0,
        help="Global scaling for bond thresholds (default: 1.0)",
    )

    # --- Output Options ---
    output = p.add_argument_group("Output Options")
    output.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
    output.add_argument("-a", "--ascii", action="store_true", help="Show 2D ASCII depiction")
    output.add_argument(
        "--json",
        action="store_true",
        help="Output graph as JSON (for generating test fixtures)",
    )
    output.add_argument(
        "-as",
        "--ascii-scale",
        type=float,
        default=2.5,
        help="ASCII scaling factor (default: 2.5)",
    )
    output.add_argument(
        "--nci",
        action="store_true",
        help="Detect and report non-covalent interactions",
    )
    output.add_argument(
        "-H",
        "--show-h",
        action="store_true",
        help="Include hydrogens in visualizations",
    )
    output.add_argument(
        "--show-h-idx",
        type=str,
        help="Show specific H atoms (comma-separated indices)",
    )

    # --- Input Options ---
    input_options = p.add_argument_group("Input Options")
    input_options.add_argument(
        "-b",
        "--bohr",
        action="store_true",
        default=False,
        help="XYZ file in Bohr units (default: Angstrom)",
    )
    input_options.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index for trajectory files, 0-indexed (default: 0)",
    )
    input_options.add_argument(
        "--all-frames",
        action="store_true",
        default=False,
        help="Process all frames in trajectory",
    )

    # --- Comparison Options ---
    comparison = p.add_argument_group("Comparison Options")
    comparison.add_argument("--compare-rdkit", action="store_true", help="Compare with RDKit graph")
    comparison.add_argument(
        "--compare-rdkit-tm",
        action="store_true",
        help="Compare with RDKit xyz2mol_tm graph",
    )
    comparison.add_argument("--orca-out", type=str, help="ORCA output file for comparison")
    comparison.add_argument(
        "--orca-threshold",
        type=float,
        default=DEFAULT_PARAMS["orca_bond_threshold"],
        help=f"Min Mayer bond order for ORCA (default: {DEFAULT_PARAMS['orca_bond_threshold']})",
    )

    # --- Optimizer Options ---
    optimizer = p.add_argument_group("Optimizer Options")
    optimizer.add_argument(
        "-o",
        "--optimizer",
        choices=["greedy", "beam"],
        default=DEFAULT_PARAMS["optimizer"],
        help=f"Algorithm (default: {DEFAULT_PARAMS['optimizer']})",
    )
    optimizer.add_argument(
        "-bw",
        "--beam-width",
        type=int,
        default=DEFAULT_PARAMS["beam_width"],
        help=f"Beam width (default: {DEFAULT_PARAMS['beam_width']})",
    )
    optimizer.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_PARAMS["max_iter"],
        help=f"Max iterations (default: {DEFAULT_PARAMS['max_iter']})",
    )
    optimizer.add_argument(
        "--edge-per-iter",
        type=int,
        default=DEFAULT_PARAMS["edge_per_iter"],
        help=f"Edges per iteration (default: {DEFAULT_PARAMS['edge_per_iter']})",
    )

    # --- Constraints ---
    constraints = p.add_argument_group("Bond Constraints")
    constraints.add_argument("--bond", type=str, help="Force bonds (e.g., --bond 0,1 2,3)")
    constraints.add_argument("--unbond", type=str, help="Prevent bonds (e.g., --unbond 0,1)")

    # --- Advanced Thresholds ---
    advanced = p.add_argument_group("Advanced Thresholds")
    advanced.add_argument(
        "--threshold-h-h",
        type=float,
        default=DEFAULT_PARAMS["threshold_h_h"],
        help=f"H-H vdW threshold (default: {DEFAULT_PARAMS['threshold_h_h']})",
    )
    advanced.add_argument(
        "--threshold-h-nonmetal",
        type=float,
        default=DEFAULT_PARAMS["threshold_h_nonmetal"],
        help=f"H-nonmetal vdW threshold (default: {DEFAULT_PARAMS['threshold_h_nonmetal']})",
    )
    advanced.add_argument(
        "--threshold-h-metal",
        type=float,
        default=DEFAULT_PARAMS["threshold_h_metal"],
        help=f"H-metal vdW threshold (default: {DEFAULT_PARAMS['threshold_h_metal']})",
    )
    advanced.add_argument(
        "--threshold-metal-ligand",
        type=float,
        default=DEFAULT_PARAMS["threshold_metal_ligand"],
        help=f"Metal-ligand vdW threshold (default: {DEFAULT_PARAMS['threshold_metal_ligand']})",
    )
    advanced.add_argument(
        "--threshold-nonmetal",
        type=float,
        default=DEFAULT_PARAMS["threshold_nonmetal_nonmetal"],
        help=f"Nonmetal-nonmetal vdW threshold (default: {DEFAULT_PARAMS['threshold_nonmetal_nonmetal']})",
    )
    advanced.add_argument(
        "--allow-metal-metal-bonds",
        action="store_true",
        default=DEFAULT_PARAMS["allow_metal_metal_bonds"],
        help=f"Allow metal-metal bonds (default: {DEFAULT_PARAMS['allow_metal_metal_bonds']})",
    )
    advanced.add_argument(
        "--threshold-metal-metal-self",
        type=float,
        default=DEFAULT_PARAMS["threshold_metal_metal_self"],
        help=f"Metal-metal vdW threshold (default: {DEFAULT_PARAMS['threshold_metal_metal_self']})",
    )
    advanced.add_argument(
        "--period-scaling-h-bonds",
        type=float,
        default=DEFAULT_PARAMS["period_scaling_h_bonds"],
        help=f"Period scaling for H bonds (default: {DEFAULT_PARAMS['period_scaling_h_bonds']})",
    )
    advanced.add_argument(
        "--period-scaling-nonmetal-bonds",
        type=float,
        default=DEFAULT_PARAMS["period_scaling_nonmetal_bonds"],
        help=f"Period scaling for nonmetal bonds (default: {DEFAULT_PARAMS['period_scaling_nonmetal_bonds']})",
    )

    args = p.parse_args()

    # Handle --version / --citation flags
    if args.version:
        print(f"xyzgraph v{__version__}")
        return
    if args.citation:
        print(__citation__)
        return

    # Configure logging for debug mode
    if args.debug:
        configure_debug_logging()

    # Require input file
    if not args.input_file:
        p.error("the following arguments are required: input_file")

    # Parse constraints
    bond = _parse_pairs(args.bond) if args.bond else None
    unbond = _parse_pairs(args.unbond) if args.unbond else None

    # Parse show_h_idx
    show_h_indices = None
    if args.show_h_idx:
        try:
            show_h_indices = [int(idx.strip()) for idx in args.show_h_idx.split(",")]
        except ValueError:
            print(f"Error: Invalid hydrogen indices in --show-h-idx: {args.show_h_idx}")
            return

    # Determine mode
    is_orca_file = args.input_file.endswith(".out")

    # MODE 1: ORCA-only (input is .out file, no --orca-out flag)
    if is_orca_file and not args.orca_out:
        try:
            G = build_graph_orca(args.input_file, bond_threshold=args.orca_threshold, debug=args.debug)
        except Exception as e:
            print(f"Error parsing ORCA output: {e}")
            return

        print_header(args.input_file, {"method": "orca", "bond_threshold": args.orca_threshold})
        display_graph(G, args, show_h_indices, label="ORCA")
        return

    # MODE 2 & 3: XYZ-based (with optional ORCA comparison)
    metadata = compute_metadata(
        method=args.method,
        charge=args.charge,
        multiplicity=args.multiplicity,
        quick=args.quick,
        optimizer=args.optimizer,
        max_iter=args.max_iter,
        edge_per_iter=args.edge_per_iter,
        beam_width=args.beam_width,
        bond=bond,
        unbond=unbond,
        clean_up=not args.no_clean,
        threshold=args.threshold,
        threshold_h_h=args.threshold_h_h,
        threshold_h_nonmetal=args.threshold_h_nonmetal,
        threshold_h_metal=args.threshold_h_metal,
        threshold_metal_ligand=args.threshold_metal_ligand,
        threshold_nonmetal_nonmetal=args.threshold_nonmetal,
        relaxed=args.relaxed,
        allow_metal_metal_bonds=args.allow_metal_metal_bonds,
        threshold_metal_metal_self=args.threshold_metal_metal_self,
        period_scaling_h_bonds=args.period_scaling_h_bonds,
        period_scaling_nonmetal_bonds=args.period_scaling_nonmetal_bonds,
    )

    # Determine frames to process
    from .utils import count_frames_and_atoms

    num_frames, _ = count_frames_and_atoms(args.input_file)

    if args.all_frames:
        frames_to_process = list(range(num_frames))
        print_header(args.input_file, metadata, frame_info=f"0-{num_frames - 1}")
        print(f"# Processing all {num_frames} frames from trajectory file...\n")
    else:
        # Show frame info if multi-frame file
        frame_info = args.frame if num_frames > 1 else None
        print_header(args.input_file, metadata, frame_info=frame_info)
        frames_to_process = [args.frame]

    # Process each frame
    for frame_idx in frames_to_process:
        if args.all_frames and frame_idx > 0:
            print(f"\n{'=' * 80}")
            print(f"Frame {frame_idx}")
            print(f"{'=' * 80}\n")

        # Build primary graph (cheminf or xtb)
        atoms = read_xyz_file(args.input_file, bohr_units=args.bohr, frame=frame_idx)
        print(f"# Building {args.method} graph from {args.input_file}...")
        G_primary = build_graph(
            atoms=atoms,
            method=args.method,
            charge=args.charge,
            multiplicity=args.multiplicity,
            quick=args.quick,
            optimizer=args.optimizer,
            max_iter=args.max_iter,
            edge_per_iter=args.edge_per_iter,
            beam_width=args.beam_width,
            bond=bond,
            unbond=unbond,
            clean_up=not args.no_clean,
            debug=args.debug,
            threshold=args.threshold,
            threshold_h_h=args.threshold_h_h,
            threshold_h_nonmetal=args.threshold_h_nonmetal,
            threshold_h_metal=args.threshold_h_metal,
            threshold_metal_ligand=args.threshold_metal_ligand,
            threshold_nonmetal_nonmetal=args.threshold_nonmetal,
            relaxed=args.relaxed,
            allow_metal_metal_bonds=args.allow_metal_metal_bonds,
            threshold_metal_metal_self=args.threshold_metal_metal_self,
            period_scaling_h_bonds=args.period_scaling_h_bonds,
            period_scaling_nonmetal_bonds=args.period_scaling_nonmetal_bonds,
            metadata=metadata,
        )
        print(f"Constructed graph with chemical formula: {G_primary.graph['formula']}")

        # JSON output mode - print and skip other output
        if args.json:
            import json

            print(json.dumps(graph_to_dict(G_primary), indent=2))
            continue  # Skip to next frame or exit

        # Build comparison graphs if requested
        G_orca = None
        G_rdkit = None
        G_rdkit_tm = None

        if args.orca_out:
            print(f"# Building ORCA graph from {args.orca_out}...")
            try:
                G_orca = build_graph_orca(args.orca_out, bond_threshold=args.orca_threshold, debug=args.debug)
            except Exception as e:
                print(f"Error parsing ORCA output: {e}")

        if args.compare_rdkit:
            print(f"# Building RDKit graph from {args.input_file}...")
            try:
                G_rdkit = build_graph_rdkit(args.input_file, charge=args.charge, bohr_units=args.bohr)
            except ValueError as e:
                print(f"# Failed to build RDKit graph: {e}")

        if args.compare_rdkit_tm:
            print(f"# Building RDKit-TM graph from {args.input_file}...")
            try:
                G_rdkit_tm = build_graph_rdkit_tm(args.input_file, charge=args.charge, bohr_units=args.bohr)
            except (ValueError, ImportError) as e:
                print(f"# Failed to build RDKit-TM graph: {e}")

        # Display primary graph
        show_ascii = display_graph(G_primary, args, show_h_indices, label=args.method)

        # NCI detection
        if args.nci:
            display_ncis(G_primary, args, show_h_indices)

        # Compare with ORCA if available
        if G_orca:
            compare_graphs(G_primary, G_orca, args.method, "ORCA")

            if args.debug:
                print(f"\n{'=' * 80}")
                print("# ORCA GRAPH DETAILS")
                print("=" * 80)
                print(graph_debug_report(G_orca, include_h=args.show_h, show_h_indices=show_h_indices))

            if show_ascii:
                print(f"\n{'=' * 80}\n# ASCII Depiction (ORCA, aligned)\n{'=' * 80}\n")
                _, layout = graph_to_ascii(
                    G_primary,
                    scale=max(0.2, args.ascii_scale),
                    include_h=args.show_h,
                    show_h_indices=show_h_indices,
                )
                ascii_orca, _ = graph_to_ascii(
                    G_orca,
                    scale=max(0.2, args.ascii_scale),
                    include_h=args.show_h,
                    show_h_indices=show_h_indices,
                    reference_layout=layout,
                )
                print(ascii_orca)

        # Compare with RDKit if available
        if G_rdkit:
            print(
                compare_with_rdkit(
                    reference_graph=G_primary,
                    rdkit_graph=G_rdkit,
                    verbose=args.debug,
                    ascii=show_ascii,
                    ascii_scale=args.ascii_scale,
                    ascii_include_h=args.show_h,
                ).rstrip()
            )

        # Compare with RDKit-TM if available
        if G_rdkit_tm:
            compare_graphs(G_primary, G_rdkit_tm, args.method, "RDKit-TM")

            if args.debug:
                print(f"\n{'=' * 80}")
                print("# RDKIT-TM GRAPH DETAILS")
                print("=" * 80)
                print(graph_debug_report(G_rdkit_tm, include_h=args.show_h, show_h_indices=show_h_indices))

            if show_ascii:
                print(f"\n{'=' * 80}\n# ASCII Depiction (RDKit-TM, aligned)\n{'=' * 80}\n")
                _, layout = graph_to_ascii(
                    G_primary,
                    scale=max(0.2, args.ascii_scale),
                    include_h=args.show_h,
                    show_h_indices=show_h_indices,
                )
                ascii_rdkit_tm, _ = graph_to_ascii(
                    G_rdkit_tm,
                    scale=max(0.2, args.ascii_scale),
                    include_h=args.show_h,
                    show_h_indices=show_h_indices,
                    reference_layout=layout,
                )
                print(ascii_rdkit_tm)


if __name__ == "__main__":
    main()
