"""Crystal structure support (optional — requires xyzrender[crystal] / phonopy).

This module contains all phonopy-dependent functionality for loading periodic
crystal structures and generating periodic image atoms for rendering.  It is
intentionally separated from ``io.py`` so that the optional ``phonopy``
dependency is not imported at all unless crystal loading is actually requested.

Public API
----------
load_crystal
    Load a VASP/QE/... crystal structure file and return a molecular graph
    together with its ``CrystalData`` (lattice matrix + cell origin).
add_crystal_images
    Populate a crystal graph with ghost atoms from the 26 neighbouring unit
    cells so that bonds crossing cell boundaries are visible.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
from xyzgraph import DATA, build_graph

from xyzrender.types import CrystalData

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

logger = logging.getLogger(__name__)

# Covalent radii in Å for common elements. Used to detect PBC image bonds.
# Source: Alvarez (2008), DOI:10.1039/b801115j (single-bond radii)
# TODO: change to share xyzgraph bond detection logic
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.50,
    "Fe": 1.42,
    "Co": 1.38,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.40,
    "At": 1.50,
    "Rn": 1.50,
}


def _cov_r(sym: str) -> float:
    """Covalent radius (Å) from the Alvarez (2008) table; falls back to 0.55 x VdW."""
    return _COVALENT_RADII.get(sym, DATA.vdw.get(sym, 1.5) * 0.55)


def load_crystal(
    path: str | Path,
    interface_mode: str,
) -> tuple[nx.Graph, CrystalData]:
    """Load a periodic crystal structure using phonopy.

    Parameters
    ----------
    path:
        Path to the crystal structure input file (POSCAR/CONTCAR for VASP,
        ``*.in`` / ``pw.in`` for Quantum ESPRESSO, etc.).
    interface_mode:
        Phonopy interface identifier: ``"vasp"``, ``"qe"``, ``"abinit"``, etc.

    Returns
    -------
    tuple[nx.Graph, CrystalData]
        Molecular graph with atoms as nodes and ``CrystalData`` containing the
        3x3 lattice matrix (rows = a, b, c in Å).
    """
    try:
        from phonopy.interface.calculator import get_calculator_physical_units, read_crystal_structure
    except ImportError:
        msg = "Crystal structure loading requires phonopy: pip install 'xyzrender[crystal]'"
        raise ImportError(msg) from None

    unitcell, _ = read_crystal_structure(str(path), interface_mode=interface_mode)
    if unitcell is None:
        msg = f"Failed to read crystal structure from {path!r} (interface_mode={interface_mode!r})"
        raise ValueError(msg)
    # Convert native units → Angstrom.
    factor: float = get_calculator_physical_units(interface_mode).distance_to_A
    symbols: list[str] = list(unitcell.symbols)
    positions = unitcell.positions * factor  # ndarray, shape (N, 3), in Å
    lattice = np.array(unitcell.cell) * factor  # shape (3, 3), rows = a, b, c in Å

    atoms: list[tuple[str, tuple[float, float, float]]] = [
        (sym, (float(pos[0]), float(pos[1]), float(pos[2]))) for sym, pos in zip(symbols, positions, strict=True)
    ]
    graph = build_graph(atoms, charge=0, multiplicity=None, kekule=False, quick=True)
    logger.info(
        "Crystal graph: %d atoms, %d bonds, lattice=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        lattice.diagonal().round(3),
    )
    return graph, CrystalData(lattice=lattice)


def add_crystal_images(graph: nx.Graph, crystal_data: CrystalData) -> int:
    """Add periodic image atoms that are bonded to cell atoms.

    For each of the 26 neighbouring unit cells, adds image copies of cell
    atoms that form at least one bond with an atom inside the cell.  Image
    nodes carry ``image=True`` and ``source=<cell_atom_id>`` attributes;
    image bonds carry ``image_bond=True``.

    Returns the number of image atoms added.
    """
    lattice = crystal_data.lattice  # (3, 3)
    a, b, c = lattice[0], lattice[1], lattice[2]

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    cell_syms = {i: graph.nodes[i]["symbol"] for i in cell_ids}
    cell_pos = {i: np.array(graph.nodes[i]["position"]) for i in cell_ids}

    next_id = max(cell_ids) + 1
    n_added = 0

    shifts = [(dx, dy, dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    for dx, dy, dz in shifts:
        offset = dx * a + dy * b + dz * c
        for src_id in cell_ids:
            sym_i = cell_syms[src_id]
            img_pos = cell_pos[src_id] + offset
            ri = _cov_r(sym_i)

            # TODO: ghost bond detection uses its own covalent-radii at 1.2x
            # tolerance, defined above, independent of xyzgraph.
            # Ideally the ghost detection would reuse the graph's existing bond
            # logic, should be able to acess the distance detection from xyzgraph.
            bonded_to: list[int] = [
                j for j in cell_ids if float(np.linalg.norm(img_pos - cell_pos[j])) < (ri + _cov_r(cell_syms[j])) * 1.2
            ]

            if not bonded_to:
                continue

            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=sym_i,
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=src_id,
            )
            for j in bonded_to:
                graph.add_edge(img_id, j, bond_order=1.0, image_bond=True)

    logger.info("Added %d image atoms", n_added)
    return n_added
