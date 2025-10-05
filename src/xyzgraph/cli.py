import argparse
from ase.io import read as read_xyz
from .graph_builders import build_graph
from .ascii_renderer import graph_debug_report, graph_to_ascii
from .compare import xyz2mol_compare

def main():
    p = argparse.ArgumentParser(description="Build molecular graph from XYZ.")
    p.add_argument("xyz", help="Input XYZ file")
    p.add_argument("--method", choices=["cheminf","xtb"], default="cheminf")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=None)
    p.add_argument("--no-clean", action="store_true", help="Do not remove tmp XTB folder")
    p.add_argument("--ascii", action="store_true", help="Show 2D ASCII depiction")
    p.add_argument("--ascii-scale", type=float, default=3,
                   help="Scaling factor (>1 enlarges) for 2D ASCII.")
    p.add_argument("--show-h", action="store_true",
                   help="Include hydrogens (and C–H bonds) in outputs (hidden by default).")
    p.add_argument("--compare-xyz2mol", action="store_true",
                   help="Also print xyz2mol-derived graph (if installed).")
    p.add_argument("--debug-graph","-dg", action="store_true", default=False,
                   help="Print detailed graph debug report (bonds / valences / charges).")
    args = p.parse_args()

    atoms = read_xyz(args.xyz)
    G = build_graph(atoms,
                    method=args.method,
                    charge=args.charge,
                    multiplicity=args.multiplicity,
                    clean_up=not args.no_clean)

    # Determine if user supplied any visibility flags explicitly
    user_flag = any([args.ascii, args.debug_graph, args.compare_xyz2mol])
    auto_ascii = False

    # Default: if ONLY xyz provided (no flags) -> show ASCII automatically
    if not user_flag:
        auto_ascii = True

    # If comparison requested and no debug/ascii, enable ASCII automatically
    if args.compare_xyz2mol and not (args.debug_graph or args.ascii):
        auto_ascii = True

    ascii_active = args.ascii or auto_ascii

    # Debug report only when requested
    if args.debug_graph:
        print(graph_debug_report(G, include_h=args.show_h))

    if ascii_active:
        if auto_ascii and not args.ascii:
            print("# (auto-enabled ASCII output)\n")
        print("\n# 2D ASCII depiction (scale={:.2f}, hydrogens={})\n".format(
            args.ascii_scale, "shown" if args.show_h else "hidden"))
        print(graph_to_ascii(G,
                             scale=max(0.2, args.ascii_scale),
                             include_h=args.show_h))

    if args.compare_xyz2mol:
        print("\n# xyz2mol comparison\n")
        print(xyz2mol_compare(atoms,
                              charge=args.charge,
                              verbose=args.debug_graph,          # map debug to verbose
                              ascii=ascii_active,
                              ascii_scale=max(0.2, args.ascii_scale),
                              ascii_include_h=args.show_h,
                            #   reference_graph=G   # NEW: align xyz2mol ASCII to reference
                              ).rstrip())

if __name__ == "__main__":
    main()
