# cli.py
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


# ---------- helpers (local, single-file) ----------

def _looks_like_coord(line: str) -> bool:
    parts = line.split()
    if len(parts) < 4:
        return False
    if not parts[0] or not parts[0][0].isalpha():
        return False
    try:
        float(parts[1]); float(parts[2]); float(parts[3])
    except Exception:
        return False
    return True


def _count_frames_in_file(path: str) -> int:
    """
    Count true XYZ blocks: natoms header, comment line(s), then natoms coord lines.
    Robust against blank/energy/comment lines between header and first coordinate.
    """
    count = 0
    with open(path, "r") as fh:
        while True:
            hdr = fh.readline()
            if not hdr:
                break
            hdr = hdr.strip()
            if not hdr:
                continue
            try:
                nat = int(hdr)
            except ValueError:
                continue
            if nat < 1 or nat > 10000:
                continue
            # One mandatory comment/title line
            _ = fh.readline()
            # Skip any further non-coordinate lines until the first coordinate
            first_is_coord = False
            for _skip in range(10):
                probe = fh.readline()
                if not probe:
                    break
                if _looks_like_coord(probe):
                    first_is_coord = True
                    break
            if not first_is_coord:
                continue
            # Skip the remaining nat-1 coordinate lines
            for _ in range(max(0, nat - 1)):
                _ = fh.readline()
            count += 1
    return count


def _iter_frames(path: str, bohr_units: bool):
    """
    Yield (file_index_1based, atoms) for each valid XYZ block in the file.
    atoms is [(symbol, (x, y, z)), ...] in Å, matching read_xyz_file format.
    """
    scale = BOHR_TO_ANGSTROM if bohr_units else 1.0
    file_idx = 0
    with open(path, "r") as fh:
        while True:
            hdr = fh.readline()
            if not hdr:
                break
            hdr = hdr.strip()
            if not hdr:
                continue
            try:
                nat = int(hdr)
            except ValueError:
                continue
            if nat < 1 or nat > 10000:
                continue

            # One comment line, then skip any non-coordinate lines
            _ = fh.readline()
            first_coord = None
            for _skip in range(10):
                probe = fh.readline()
                if not probe:
                    break
                if _looks_like_coord(probe):
                    first_coord = probe
                    break
            if first_coord is None:
                continue

            atoms = []
            def _add(line: str) -> bool:
                parts = line.split()
                try:
                    sym = parts[0]
                    x, y, z = map(float, parts[1:4])
                except Exception:
                    return False
                atoms.append((sym, (x * scale, y * scale, z * scale)))
                return True

            if not _add(first_coord):
                continue
            ok = True
            for _ in range(nat - 1):
                line = fh.readline()
                if not line or not _looks_like_coord(line) or not _add(line):
                    ok = False
                    break
            if not ok or len(atoms) != nat:
                continue

            file_idx += 1
            yield file_idx, atoms


def _parse_pairs(arg_value: str):
    """
    Parse '--bond "i,j a,b"' or '--unbond "i,j a,b"' into [(i,j), (a,b)].
    """
    pairs = []
    for pair_str in arg_value.split():
        i_str, j_str = pair_str.split(",")
        pairs.append((int(i_str), int(j_str)))
    return pairs


# ---------- CLI entry ----------

def main():
    p = argparse.ArgumentParser(description="Build molecular graph from XYZ.")

    p.add_argument("xyz", help="Input XYZ file")

    # Method and quality
    p.add_argument("--method", choices=["cheminf", "xtb"], default=DEFAULT_PARAMS["method"],
                   help=f"Graph construction method (default: {DEFAULT_PARAMS['method']}) (xtb requires xTB binary in PATH)")
    p.add_argument("-q", "--quick", action="store_true", default=DEFAULT_PARAMS["quick"],
                   help="Quick mode: faster, less accurate (NOT recommended)")
    p.add_argument("--max-iter", type=int, default=DEFAULT_PARAMS["max_iter"],
                   help=f"Max iterations for bond-order optimization (default: {DEFAULT_PARAMS['max_iter']}, cheminf only)")
    p.add_argument("--edge-per-iter", type=int, default=DEFAULT_PARAMS["edge_per_iter"],
                   help=f"Number of edges to adjust per iteration (default: {DEFAULT_PARAMS['edge_per_iter']}, cheminf only)")
    p.add_argument("-o", "--optimizer", choices=["greedy", "beam"], default=DEFAULT_PARAMS["optimizer"],
                   help=f"Optimization algorithm (default: {DEFAULT_PARAMS['optimizer']}; beam recommended)")
    p.add_argument("-bw", "--beam-width", type=int, default=DEFAULT_PARAMS["beam_width"],
                   help=f"Beam width for beam search (default: {DEFAULT_PARAMS['beam_width']})")
    p.add_argument("--bond", type=str,
                   help='Force bonds: e.g. --bond "0,1 2,3" (0-based indices)')
    p.add_argument("--unbond", type=str,
                   help='Force non-bonds: e.g. --unbond "0,1 1,2" (0-based indices)')

    # Molecular properties
    p.add_argument("-c", "--charge", type=int, default=0, help="Total charge (default: 0)")
    p.add_argument("-m", "--multiplicity", type=int, default=None, help="Spin multiplicity (auto if omitted)")
    p.add_argument("-b", "--bohr", action="store_true", default=False, help="XYZ in bohr units (default: Å)")

    # Output control
    p.add_argument("-d", "--debug", action="store_true", help="Show graph construction diagnostics")
    p.add_argument("-a", "--ascii", action="store_true", help="Show ASCII depiction")
    p.add_argument("-as", "--ascii-scale", type=float, default=3.0, help="ASCII scaling (default: 3.0)")
    p.add_argument("-H", "--show-h", action="store_true", help="Include hydrogens in ASCII")

    # Comparison
    p.add_argument("--compare-rdkit", action="store_true", help="Compare with RDKit DetermineBonds (xyz2mol-like)")

    # xTB specific
    p.add_argument("--no-clean", action="store_true", help="Keep temporary xTB files (method=xtb)")

    # Multi-XYZ control
    p.add_argument("--all-frames", action="store_true", help="Render all frames if file contains multiple XYZ blocks")
    p.add_argument("--frame", type=int, action="append", help="Render specific 1-based frame(s); may repeat")
    p.add_argument("--start", type=int, default=0, help="0-based slice start (with --stop/--stride)")
    p.add_argument("--stop", type=int, default=None, help="0-based slice stop (exclusive)")
    p.add_argument("--stride", type=int, default=1, help="Slice stride")

    args = p.parse_args()

    # Parse forced bond constraints
    bond = _parse_pairs(args.bond) if args.bond else None
    unbond = _parse_pairs(args.unbond) if args.unbond else None

    # Detect total frames; if 0/1, fall back to single-structure behavior
    total_in_file = _count_frames_in_file(args.xyz)
    multi_requested = bool(
        args.all_frames or args.frame or args.stop is not None or args.start != 0 or args.stride != 1
    )

    if total_in_file <= 1 and not multi_requested:
        # Single XYZ
        atoms = read_xyz_file(args.xyz, bohr_units=args.bohr)
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
        )

        has_explicit_output = args.debug or args.ascii or args.compare_rdkit
        show_ascii = args.ascii or not has_explicit_output
        if not args.ascii and not has_explicit_output:
            print("\n# (Auto-enabled ASCII output - use --help for more options)\n")

        if args.debug:
            print(graph_debug_report(G))

        if show_ascii:
            print(f"\n{'=' * 60}\n# ASCII Depiction\n{'=' * 60}")
            print(graph_to_ascii(G, scale=max(0.2, args.ascii_scale), include_h=args.show_h))

        if args.compare_rdkit:
            print(
                xyz2mol_compare(
                    atoms,
                    charge=args.charge,
                    verbose=args.debug,
                    ascii=show_ascii,
                    ascii_scale=args.ascii_scale,
                    ascii_include_h=args.show_h,
                    reference_graph=G if len(G) == len(atoms) else None,
                ).rstrip()
            )
        return

    # Multi-XYZ: build selection (1-based indices)
    if args.frame:
        selected = sorted({i for i in args.frame if 1 <= i <= max(1, total_in_file)})
    else:
        start0 = max(0, args.start)
        stop0 = total_in_file if args.stop is None else max(0, args.stop)
        stride = max(1, args.stride)
        all_idx = list(range(1, total_in_file + 1))
        if args.all_frames or args.stop is not None or args.start != 0 or args.stride != 1:
            selected = all_idx[start0:stop0:stride]
        else:
            selected = all_idx if args.all_frames else all_idx  # default: all frames

    if not selected:
        # Nothing matched filters; behave like single
        atoms = read_xyz_file(args.xyz, bohr_units=args.bohr)
        print("\n# Multi-XYZ not detected (or filters excluded all frames); rendering single frame\n")
        print(f"\n{'=' * 60}\n# ASCII Depiction\n{'=' * 60}")
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
        )
        print(graph_to_ascii(G, scale=max(0.2, args.ascii_scale), include_h=args.show_h))
        return

    print("\n# Multi-XYZ detected: rendering frames\n")

    # Render frames, aligning all ASCII layouts to the first frame’s layout
    ref_layout = None
    first_atoms = None
    rendered = 0
    total_selected = len(selected)
    want = set(selected)

    for file_idx, atoms in _iter_frames(args.xyz, args.bohr):
        if file_idx not in want:
            continue

        rendered += 1
        print(f"\n{'=' * 60}\n# Frame {rendered}/{total_selected} (file #{file_idx} of {total_in_file})\n{'=' * 60}")

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
        )

        if args.debug:
            print(graph_debug_report(G))

        if ref_layout is None:
            ascii_txt, ref_layout = graph_to_ascii(
                G, scale=max(0.2, args.ascii_scale), include_h=args.show_h, return_layout=True
            )
            first_atoms = atoms
        else:
            ascii_txt = graph_to_ascii(
                G, scale=max(0.2, args.ascii_scale), include_h=args.show_h, reference_layout=ref_layout
            )
        print(ascii_txt)

    # Optional comparison on first rendered frame only (keeps output concise)
    if args.compare_rdkit and first_atoms is not None:
        print("\n# --compare-rdkit: comparing first rendered frame only\n")
        G0 = build_graph(
            atoms=first_atoms,
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
        )
        print(
            xyz2mol_compare(
                first_atoms,
                charge=args.charge,
                verbose=args.debug,
                ascii=False,
                ascii_scale=args.ascii_scale,
                ascii_include_h=args.show_h,
                reference_graph=G0 if len(G0) == len(first_atoms) else None,
            ).rstrip()
        )


if __name__ == "__main__":
    main()
