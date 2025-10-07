import argparse
from ase.io import read as read_xyz
from .graph_builders import build_graph, set_debug
from .ascii_renderer import graph_debug_report, graph_to_ascii
from .compare import xyz2mol_compare

def main():
    p = argparse.ArgumentParser(description="Build molecular graph from XYZ.")
    p.add_argument("xyz", help="Input XYZ file")
    
    # Method and quality
    p.add_argument("--method", choices=["cheminf", "xtb"], default="cheminf",
                   help="Graph construction method (default: cheminf) (xtb requires xTB binary installed and available in PATH)")
    p.add_argument("-q", "--quick", action="store_true", default=False,
                   help="Quick mode: fast heuristics, less accuracy")
    
    # Molecular properties
    p.add_argument("-c", "--charge", type=int, default=0,
                   help="Total molecular charge (default: 0)")
    p.add_argument("-m", "--multiplicity", type=int, default=None,
                   help="Spin multiplicity (auto-detected if not specified)")
    
    # Output control
    p.add_argument("-d", "--debug", action="store_true",
                   help="Enable debug output (construction details + graph report)")
    p.add_argument("-a", "--ascii", action="store_true",
                   help="Show 2D ASCII depiction (auto-enabled if no other output)")
    p.add_argument("-as", "--ascii-scale", type=float, default=4.0,
                   help="ASCII scaling factor (default: 4.0)")
    p.add_argument("-H", "--show-h", action="store_true",
                   help="Include hydrogens in visualizations (hidden by default)")
    
    # Comparison
    p.add_argument("--compare-xyz2mol", action="store_true",
                   help="Compare with xyz2mol output (requires xyz2mol installed)")
    
    # xTB specific
    p.add_argument("--no-clean", action="store_true",
                   help="Keep temporary xTB files (only for --method xtb)")
    
    args = p.parse_args()
    
    # Enable debug mode globally if requested
    if args.debug:
        set_debug(True)
    
    # Read structure
    atoms = read_xyz(args.xyz)
    
    # Build graph
    G = build_graph(
        atoms,
        method=args.method,
        charge=args.charge,
        multiplicity=args.multiplicity,
        quick=args.quick,
        clean_up=not args.no_clean
    )
    
    # Determine what to show
    has_explicit_output = args.debug or args.ascii or args.compare_xyz2mol
    show_ascii = args.ascii or not has_explicit_output
    
    # Debug report (if requested)
    if args.debug:
        print("\n" + "=" * 70)
        print("GRAPH DEBUG REPORT")
        print("=" * 70)
        print(graph_debug_report(G, include_h=args.show_h))
    
    # ASCII visualization
    if show_ascii:
        if not args.ascii and not has_explicit_output:
            print("\n# (Auto-enabled ASCII output - use --help for more options)\n")
        
        mode = "QUICK" if args.quick else "FULL"
        print(f"\n# 2D Structure ({args.method.upper()}, {mode} mode)") # add hydrogen hidden if hidden in this line
        if not args.show_h:
            print("# (Hydrogens hidden - use -H to show)")
        print(f"# Charge: {args.charge}, Multiplicity: {G.graph.get('multiplicity', 'N/A')}")
        
        print(graph_to_ascii(G,
                            scale=max(0.2, args.ascii_scale),
                            include_h=args.show_h))
    
    # xyz2mol comparison
    if args.compare_xyz2mol:
        print("\n" + "=" * 70)
        print("XYZ2MOL COMPARISON")
        print("=" * 70)
        
        print(xyz2mol_compare(
            atoms,
            charge=args.charge,
            verbose=args.debug,
            ascii=show_ascii,
            ascii_scale=max(0.2, args.ascii_scale),
            ascii_include_h=args.show_h,
            reference_graph=G
        ).rstrip())

if __name__ == "__main__":
    main()