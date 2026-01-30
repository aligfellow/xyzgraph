"""
Parser for ORCA quantum chemistry output files.

Extracts:
- Molecular charge and multiplicity
- Cartesian coordinates
- Mayer bond orders
- Mulliken population charges
"""

import re
from typing import Any, Dict, List, Tuple


class OrcaParseError(Exception):
    """Raised when ORCA output cannot be parsed."""


def parse_orca_output(filepath: str) -> Dict[str, Any]:
    """
    Parse ORCA output file for molecular structure and bond information.

    Parameters
    ----------
    filepath : str
        Path to ORCA output file

    Returns
    -------
    dict
        Dictionary containing:
        - 'atoms': List[(symbol, (x, y, z))]
        - 'bonds': List[(i, j, bond_order)]
        - 'charges': List[mulliken_charge]
        - 'charge': int (molecular charge)
        - 'multiplicity': int

    Raises
    ------
    OrcaParseError
        If required sections are missing or malformed
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Parse charge and multiplicity
    charge, multiplicity = _parse_charge_multiplicity(content)

    # Parse coordinates
    atoms = _parse_coordinates(content)

    # Parse Mayer bond orders
    bonds = _parse_mayer_bonds(content)

    # Parse Mulliken charges
    mulliken_charges = _parse_mulliken_charges(content)

    # Validate
    if not atoms:
        raise OrcaParseError("No coordinates found in ORCA output")

    if not bonds:
        raise OrcaParseError("No Mayer bond orders found in ORCA output")

    if len(mulliken_charges) != len(atoms):
        raise OrcaParseError(f"Mismatch: {len(atoms)} atoms but {len(mulliken_charges)} charges")

    return {
        "atoms": atoms,
        "bonds": bonds,
        "charges": mulliken_charges,
        "charge": charge,
        "multiplicity": multiplicity,
    }


def _parse_charge_multiplicity(content: str) -> Tuple[int, int]:
    """
    Extract molecular charge and multiplicity from General Settings section.

    Example:
        Total Charge           Charge          ....   -1
        Multiplicity           Mult            ....    1
    """
    charge_match = re.search(r"Total Charge\s+Charge\s+\.\.\.\.\s+(-?\d+)", content)
    mult_match = re.search(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)", content)

    if not charge_match:
        raise OrcaParseError("Could not find 'Total Charge' in ORCA output")

    if not mult_match:
        raise OrcaParseError("Could not find 'Multiplicity' in ORCA output")

    charge = int(charge_match.group(1))
    multiplicity = int(mult_match.group(1))

    return charge, multiplicity


def _parse_coordinates(content: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    """
    Extract Cartesian coordinates from ORCA output.

    Example section:
        ---------------------------------
        CARTESIAN COORDINATES (ANGSTROEM)
        ---------------------------------
          H     -0.637681    1.729953   -3.147280
          O     -0.794237    0.887459   -2.691626
    """
    # Find the CARTESIAN COORDINATES (ANGSTROEM) section
    coord_section = re.search(
        r"-+\s*CARTESIAN COORDINATES \(ANGSTROEM\)\s*-+\s*(.*?)(?:\n\s*\n|---)",
        content,
        re.DOTALL,
    )

    if not coord_section:
        raise OrcaParseError("Could not find 'CARTESIAN COORDINATES (ANGSTROEM)' section")

    atoms = []
    for line in coord_section.group(1).strip().split("\n"):
        data = line.strip()
        if not data:
            continue

        parts = data.split()
        if len(parts) < 4:
            continue

        try:
            symbol = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            atoms.append((symbol, (x, y, z)))
        except (ValueError, IndexError):
            continue

    return atoms


def _parse_mayer_bonds(content: str) -> List[Tuple[int, int, float]]:
    """
    Extract Mayer bond orders from ORCA output.

    Example section:
        Mayer bond orders larger than 0.100000
        B(  0-H ,  1-O ) :   0.7879 B(  1-O ,  2-C ) :   0.9724

    Returns list of (atom_i, atom_j, bond_order)
    """
    # Find the Mayer bond orders section
    mayer_section = re.search(
        r"Mayer bond orders larger than\s+[\d.]+\s*(.*?)(?:\n\s*\n|---)",
        content,
        re.DOTALL,
    )

    if not mayer_section:
        raise OrcaParseError("Could not find 'Mayer bond orders' section")

    bonds = []

    # Pattern: B(  0-H ,  1-O ) :   0.7879
    # Captures: atom_i, symbol_i, atom_j, symbol_j, bond_order
    pattern = r"B\(\s*(\d+)-\w+\s*,\s*(\d+)-\w+\s*\)\s*:\s*([\d.]+)"

    for match in re.finditer(pattern, mayer_section.group(1)):
        i = int(match.group(1))
        j = int(match.group(2))
        bo = float(match.group(3))
        bonds.append((i, j, bo))

    return bonds


def _parse_mulliken_charges(content: str) -> List[float]:
    """
    Extract Mulliken charges from MAYER POPULATION ANALYSIS section.

    Example section:
        * MAYER POPULATION ANALYSIS *

        ATOM       NA         ZA         QA         VA         BVA        FA
        0 H      0.5779     1.0000     0.4221     0.7984     0.7984     0.0000
        1 O      8.6366     8.0000    -0.6366     1.8554     1.8554     0.0000

    NA = Mulliken gross atomic population
    We compute charge as: ZA - NA
    """
    # Find the MAYER POPULATION ANALYSIS section
    mayer_pop = re.search(
        r"\*\s*MAYER POPULATION ANALYSIS\s*\*.*?ATOM\s+NA\s+ZA\s+QA\s+VA\s+BVA\s+FA\s*(.*?)\
            (?:\n\s*\n|Mayer bond orders)",
        content,
        re.DOTALL,
    )

    if not mayer_pop:
        raise OrcaParseError("Could not find 'MAYER POPULATION ANALYSIS' section")

    charges = []

    for line in mayer_pop.group(1).strip().split("\n"):
        data = line.strip()
        if not data:
            continue

        parts = data.split()
        if len(parts) < 7:
            continue

        try:
            # Skip if first column is not a number (header remnants)
            int(parts[0])

            # Column indices: 0=atom_idx, 1=symbol, 2=NA, 3=ZA, 4=QA
            # QA is the Mulliken charge (already computed by ORCA as ZA - NA)
            qa = float(parts[4])
            charges.append(qa)
        except (ValueError, IndexError):
            continue

    return charges
