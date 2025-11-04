import argparse
from . import (
    build_graph,
    graph_debug_report,
    graph_to_ascii,
    read_xyz_file,
    xyz2mol_compare,
    DEFAULT_PARAMS,
    BOHR_TO_ANGSTROM,
)

from .utils import _parse_pairs

def main():
    p = argparse.ArgumentParser(description="Build molecular graph from XYZ.")
    p.add_argument("xyz", help="Input XYZ file")
    
    # Method and quality
    p.add_argument("--method", choices=["cheminf", "xtb"], default=DEFAULT_PARAMS['method'],
                    help=f"Graph construction method (default: {DEFAULT_PARAMS['method']}) (xtb requires xTB binary installed and available in PATH)")
    p.add_argument("-q", "--quick", action="store_true", default=DEFAULT_PARAMS['quick'],
                    help="Quick mode: fast heuristics, less accuracy (NOT recommended)")
    p.add_argument("--max-iter", type=int, default=DEFAULT_PARAMS['max_iter'],
                    help=f"Maximum iterations for bond order optimization (default: {DEFAULT_PARAMS['max_iter']}, cheminf only)")

    p.add_argument("-t", "--threshold", type=float, default=1.0,
                    help="Scaling factor for bond detection thresholds (default: 1.0)")
    p.add_argument("--relaxed", action="store_true", default=DEFAULT_PARAMS['relaxed'],
                    help="Relaxed mode: use more permissive geometric validation for transition states and strained rings, more likely to produce spurious structures")
    p.add_argument("--edge-per-iter", type=int, default=DEFAULT_PARAMS['edge_per_iter'],
                    help=f"Number of edges to adjust per iteration (default: {DEFAULT_PARAMS['edge_per_iter']}, cheminf only)")
    p.add_argument("-o", "--optimizer", choices=["greedy", "beam"], default=DEFAULT_PARAMS['optimizer'],
                    help=f"Optimization algorithm (default: {DEFAULT_PARAMS['optimizer']}, cheminf , BEAM recommended)")

    p.add_argument("-bw", "--beam-width", type=int, default=DEFAULT_PARAMS['beam_width'],
                    help=f"Beam width for beam search (default: {DEFAULT_PARAMS['beam_width']}). i.e. number of candidate graphs to retain per iteration")
    p.add_argument("--bond", type=str,
                    help="Specify atoms that must be bonded in the graph construction. Example: --bond 0,1 2,3")
    p.add_argument("--unbond", type=str,
                   help="Specify that two atoms indices are NOT bonded in the graph construction. Example: --unbond 0,1 1,2")
    
    # Molecular properties
    p.add_argument("-c", "--charge", type=int, default=0,
                    help="Total molecular charge (default: 0)")
    p.add_argument("-m", "--multiplicity", type=int, default=None,
                    help="Spin multiplicity (auto-detected if not specified)")
    p.add_argument("-b", "--bohr", action="store_true", default=False,
                    help="XYZ file provided in units bohr (default is Angstrom)")
    
    # Output control
    p.add_argument("-d", "--debug", action="store_true",
                    help="Enable debug output (construction details + graph report)")
    p.add_argument("-a", "--ascii", action="store_true",
                    help="Show 2D ASCII depiction (auto-enabled if no other output)")
    p.add_argument("-as", "--ascii-scale", type=float, default=3.0,
                    help="ASCII scaling factor (default: 3.0)")
    p.add_argument("-H", "--show-h", action="store_true",
                    help="Include hydrogens in visualizations (hidden by default)")
    p.add_argument("--show-h-idx", type=str,
                    help="Show specific hydrogen atoms by index (comma-separated, e.g., '3,7,12')")
    
    # Comparison
    p.add_argument("--compare-rdkit", action="store_true",
                    help="Compare with xyz2mol output (uses rdkit implementation)")
    
    # xTB specific
    p.add_argument("--no-clean", action="store_true",
                    help="Keep temporary xTB files (only for --method xtb)")
    
    # Advanced bond detection thresholds (VDW radii multipliers)
    p.add_argument("--threshold-h-h", type=float, default=DEFAULT_PARAMS['threshold_h_h'],
                    help=f"ADVANCED: vdW threshold for H-H bonds (default: {DEFAULT_PARAMS['threshold_h_h']})")
    p.add_argument("--threshold-h-nonmetal", type=float, default=DEFAULT_PARAMS['threshold_h_nonmetal'],
                    help=f"ADVANCED: vdW threshold for H-nonmetal bonds (default: {DEFAULT_PARAMS['threshold_h_nonmetal']})")
    p.add_argument("--threshold-h-metal", type=float, default=DEFAULT_PARAMS['threshold_h_metal'],
                    help=f"ADVANCED: vdW threshold for H-metal bonds (default: {DEFAULT_PARAMS['threshold_h_metal']})")
    p.add_argument("--threshold-metal-ligand", type=float, default=DEFAULT_PARAMS['threshold_metal_ligand'],
                    help=f"ADVANCED: vdW threshold for metal-ligand bonds (default: {DEFAULT_PARAMS['threshold_metal_ligand']})")
    p.add_argument("--threshold-nonmetal", type=float, default=DEFAULT_PARAMS['threshold_nonmetal_nonmetal'],
                    help=f"ADVANCED: vdW threshold for nonmetal-nonmetal bonds (default: {DEFAULT_PARAMS['threshold_nonmetal_nonmetal']})")
    
    args = p.parse_args()
    
    # Parse forced_bonds: "0,5 3,7" → [(0, 5), (3, 7)]
    bond = _parse_pairs(args.bond) if args.bond else None
    unbond = _parse_pairs(args.unbond) if args.unbond else None
    
    # Parse show_h_idx: "3,7,12" → [3, 7, 12]
    show_h_indices = None
    if args.show_h_idx:
        try:
            show_h_indices = [int(idx.strip()) for idx in args.show_h_idx.split(',')]
        except ValueError:
            print(f"Error: Invalid hydrogen indices in --show-h-idx: {args.show_h_idx}")
            return

    # Read structure (now as list of (atomic_number, (x,y,z)))
    atoms = read_xyz_file(args.xyz, bohr_units=args.bohr)

    # Create analyzer with all parameters
    G = build_graph(
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
            relaxed=args.relaxed
        )

    # Determine what to show
    has_explicit_output = args.debug or args.ascii or args.compare_rdkit
    show_ascii = args.ascii or not has_explicit_output
    
    if not args.ascii and not has_explicit_output:
        print("\n# (Auto-enabled ASCII output - use --help for more options)\n")
    
    if args.debug:
        print(graph_debug_report(G, include_h=args.show_h, show_h_indices=show_h_indices))

    if show_ascii:
        print(f"\n{'=' * 60}\n# ASCII Depiction\n{'=' * 60}")
        print(graph_to_ascii(G, scale=max(0.2, args.ascii_scale), include_h=args.show_h, show_h_indices=show_h_indices))


    if args.compare_rdkit:
        print(xyz2mol_compare(
            atoms,
            charge=args.charge,
            verbose=args.debug,
            ascii=show_ascii,
            ascii_scale=args.ascii_scale,
            ascii_include_h=args.show_h,
            reference_graph=G if len(G) == len(atoms) else None
        ).rstrip())

if __name__ == "__main__":
    main()
