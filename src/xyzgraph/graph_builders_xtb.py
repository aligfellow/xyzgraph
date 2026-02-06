"""xTB-based molecular graph construction."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import networkx as nx

from .bond_order_optimizer import BondOrderOptimizer
from .data_loader import DATA
from .geometry import GeometryCalculator

logger = logging.getLogger(__name__)

XTB_WBO_THRESHOLD = 0.5
"""Minimum Wiberg bond order from xTB to count as a bond."""


def build_graph_xtb(
    atoms: List[Tuple[str, Tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: Optional[int] = None,
    xtb_dir: Optional[str] = None,
    basename: str = "xtb",
    clean_up: bool = True,
    debug: bool = False,
) -> nx.Graph:
    """Build molecular graph from xTB Wiberg bond orders and Mulliken charges.

    Parameters
    ----------
    atoms : list of (symbol, (x, y, z))
        Atom symbols and Cartesian coordinates.
    charge : int
        Molecular charge.
    multiplicity : int or None
        Spin multiplicity.  Inferred from electron count if None.
    xtb_dir : str or None
        Path to a directory containing existing xTB output.  When provided
        the xTB calculation is skipped and results are read directly.
        When *None* (default), xTB is executed in a temporary directory.
    basename : str
        Base name for output files.  Files will be named ``{basename}.xyz``,
        ``{basename}.out``, ``{basename}.wbo``, ``{basename}.charges``.
        Also used to find existing output when *xtb_dir* is provided.
    clean_up : bool
        Remove temporary files after a fresh xTB run.
        Ignored when *xtb_dir* is provided.
    debug : bool
        Enable debug logging.

    Returns
    -------
    nx.Graph
        Molecular graph with xTB-derived bond orders and charges.
    """
    if debug:
        from .utils import configure_debug_logging

        configure_debug_logging()

    # Derive arrays from atoms
    positions = [pos for _, pos in atoms]
    atomic_numbers = [DATA.s2n.get(sym, 0) for sym, _ in atoms]

    if multiplicity is None:
        total_electrons = sum(atomic_numbers) - charge
        multiplicity = 1 if total_electrons % 2 == 0 else 2

    # ------------------------------------------------------------------
    # Run or read xTB
    # ------------------------------------------------------------------
    if xtb_dir is not None:
        work = xtb_dir
        ran_xtb = False
    else:
        work = "xtb_tmp_local"

        if os.system("which xtb > /dev/null 2>&1") != 0:
            raise RuntimeError("xTB not found in PATH - install xTB or use 'cheminf' method")

        os.makedirs(work, exist_ok=True)

        xyz_path = os.path.join(work, f"{basename}.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(atoms)}\n")
            f.write("xyzgraph generated XYZ for xTB\n")
            for symbol, (x, y, z) in atoms:
                f.write(f"{symbol:>2} {x:15.8f} {y:15.8f} {z:15.8f}\n")

        cmd = f"cd {work} && xtb {basename}.xyz --chrg {charge} --uhf {multiplicity - 1} --gfn2 > {basename}.out"
        ret = os.system(cmd)
        if ret != 0:
            logger.warning("xTB returned non-zero exit code %d", ret)

        # Rename xTB output files to use basename for organization
        for bare, ext in [("wbo", "wbo"), ("charges", "charges")]:
            bare_path = os.path.join(work, bare)
            new_path = os.path.join(work, f"{basename}.{ext}")
            if os.path.exists(bare_path) and not os.path.exists(new_path):
                os.rename(bare_path, new_path)
                logger.debug("Renamed %s -> %s", bare, f"{basename}.{ext}")

        ran_xtb = True

    # ------------------------------------------------------------------
    # Parse WBO
    # ------------------------------------------------------------------
    bonds: list[tuple[int, int]] = []
    bond_orders: list[float] = []

    wbo_path = _find_file(work, "wbo", basename)
    if wbo_path is not None:
        with open(wbo_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3 and float(parts[2]) > XTB_WBO_THRESHOLD:
                    bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
                    bond_orders.append(float(parts[2]))
        logger.debug("Parsed %d bonds from xTB WBO", len(bonds))

    # ------------------------------------------------------------------
    # Parse Mulliken charges
    # ------------------------------------------------------------------
    charges: list[float] = []

    charges_path = _find_file(work, "charges", basename)
    if charges_path is not None:
        with open(charges_path) as f:
            for line in f:
                charges.append(float(line.split()[0]))
        logger.debug("Parsed %d Mulliken charges from xTB", len(charges))
    else:
        charges = [0.0] * len(atoms)

    # ------------------------------------------------------------------
    # Clean up temp files (only when we ran the calculation ourselves)
    # ------------------------------------------------------------------
    if ran_xtb and clean_up:
        try:
            for fname in os.listdir(work):
                os.remove(os.path.join(work, fname))
            os.rmdir(work)
        except Exception as e:
            logger.warning("Could not clean up temp files: %s", e)

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    G = nx.Graph()

    for i, (symbol, _) in enumerate(atoms):
        G.add_node(
            i,
            symbol=symbol,
            atomic_number=atomic_numbers[i],
            position=positions[i],
            charges={"mulliken": charges[i] if i < len(charges) else 0.0},
        )

    if bonds:
        for (i, j), bo in zip(bonds, bond_orders):
            d = GeometryCalculator.distance(tuple(positions[i]), tuple(positions[j]))
            si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
            G.add_edge(
                i,
                j,
                bond_order=float(bo),
                distance=d,
                bond_type=(si, sj),
                metal_coord=(si in DATA.metals or sj in DATA.metals),
            )
        logger.debug("Built graph with %d bonds from xTB", G.number_of_edges())
    else:
        logger.warning("No xTB bonds found, falling back to distance-based (try --method cheminf)")
        from .graph_builders import build_graph

        return build_graph(atoms, charge=charge, quick=True)

    # Derived properties
    for node in G.nodes():
        G.nodes[node]["valence"] = BondOrderOptimizer.valence_sum(G, node)
        agg = G.nodes[node]["charges"].get("mulliken", 0.0)
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg += G.nodes[nbr]["charges"].get("mulliken", 0.0)
        G.nodes[node]["agg_charge"] = agg

    G.graph["total_charge"] = charge
    G.graph["multiplicity"] = multiplicity
    G.graph["method"] = "xtb"

    return G


def _find_file(directory: str, name: str, basename: str = "xtb") -> Optional[str]:
    """Find an xTB output file.

    Checks in order:
    1. {basename}.{name} (e.g., xtb.wbo) — our renamed format
    2. {name} (e.g., wbo) — raw xTB binary output
    3. *.{name} (e.g., xtb_water.wbo) — user-provided files
    """
    # Our renamed format (what build_graph_xtb produces)
    preferred = os.path.join(directory, f"{basename}.{name}")
    if os.path.exists(preferred):
        return preferred

    # Raw xTB binary output (bare name)
    bare = os.path.join(directory, name)
    if os.path.exists(bare):
        return bare

    # User-provided files with .{name} extension
    for fname in os.listdir(directory):
        if fname.endswith(f".{name}"):
            return os.path.join(directory, fname)

    return None
