import argparse
from ase.io import read as read_xyz
from . import BOHR_TO_ANGSTROM
from .analyser import MolecularAnalyzer

def main():
    p = argparse.ArgumentParser(description="Build molecular graph from XYZ.")
    p.add_argument("xyz", help="Input XYZ file")
    
    # Method and quality
    p.add_argument("--method", choices=["cheminf", "xtb"], default="cheminf",
                    help="Graph construction method (default: cheminf) (xtb requires xTB binary installed and available in PATH)")
    p.add_argument("-q", "--quick", action="store_true", default=False,
                    help="Quick mode: fast heuristics, less accuracy")
    p.add_argument("--max-iter", type=int, default=50,
                    help="Maximum iterations for bond order optimization (default: 50, cheminf only)")
    p.add_argument("--edge-per-iter", type=int, default=6,
                    help="Number of edges to adjust per iteration (default: 6, cheminf only)")

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
    
    # Comparison
    p.add_argument("--compare-xyz2mol", action="store_true",
                    help="Compare with xyz2mol output (uses rdkit implementation)")
    
    # xTB specific
    p.add_argument("--no-clean", action="store_true",
                    help="Keep temporary xTB files (only for --method xtb)")
    
    args = p.parse_args()
    
    # Read structure
    atoms = read_xyz(args.xyz)

    if args.bohr:
        atoms.positions *= BOHR_TO_ANGSTROM

 # Create analyzer with all parameters
    analyzer = MolecularAnalyzer(
        atoms=atoms,
        method=args.method,
        charge=args.charge,
        multiplicity=args.multiplicity,
        quick=args.quick,
        max_iter=args.max_iter,
        edge_per_iter=args.edge_per_iter,
        clean_up=not args.no_clean,
        debug=args.debug
    )

    G = analyzer.build()
    
    # Determine what to show
    has_explicit_output = args.debug or args.ascii or args.compare_xyz2mol
    show_ascii = args.ascii or not has_explicit_output
    
    if not args.ascii and not has_explicit_output:
        print("\n# (Auto-enabled ASCII output - use --help for more options)\n")
    
    # Print outputs
    analyzer.print_summary(
        show_report=args.debug,
        show_ascii=show_ascii,
        show_xyz2mol=args.compare_xyz2mol,
        include_h=args.show_h,
        ascii_scale=max(0.2, args.ascii_scale)
    )

if __name__ == "__main__":
    main()