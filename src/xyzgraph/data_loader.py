"""Molecular reference data loader."""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Dict, FrozenSet, List, Set

BOHR_TO_ANGSTROM = 0.5291772105


@dataclass
class MolecularData:
    """Container for molecular reference data.

    Loaded once on import and accessible globally.
    """

    vdw: Dict[str, float]
    valences: Dict[str, List[int]]
    electrons: Dict[str, int]
    metals: Set[str]
    s2n: Dict[str, int]
    n2s: Dict[int, str]
    electronegativity: Dict[str, float]
    conjugatable_atoms: FrozenSet[str]
    aromatic_atoms: FrozenSet[str]
    scoring_conjugatable_atoms: FrozenSet[str]
    max_aromatic_valence: Dict[str, int]

    _instance = None

    @classmethod
    def get_instance(cls) -> "MolecularData":
        """Get or create instance."""
        if cls._instance is None:
            cls._instance = cls._load_data()
        return cls._instance

    @classmethod
    def _load_data(cls) -> "MolecularData":
        """Load all molecular data from package resources."""
        data_path = resources.files("xyzgraph.data")

        # Load VDW radii
        vdw_file = data_path / "vdw_radii.json"
        with vdw_file.open("r") as f:
            vdw_data = json.load(f)
        vdw_radii = {
            element: vdw_data[element]["vdw_radius"] * BOHR_TO_ANGSTROM
            for element in vdw_data
            if not element.startswith("_")
        }

        # Load expected valences
        try:
            valence_file = data_path / "expected_valences.json"
            with valence_file.open("r") as f:
                valence_data = json.load(f)
            expected_valences = {
                element: valence_data[element] for element in valence_data if not element.startswith("_")
            }
        except Exception as e:
            print(f"Warning: Could not load valences file: {e}, using fallback")
            expected_valences = {
                "H": [1],
                "C": [4],
                "N": [3, 5],
                "O": [2],
                "F": [1],
                "P": [3, 5],
                "S": [2, 4, 6],
                "Cl": [1],
                "Br": [1],
                "I": [1, 2, 3],
                "Li": [1],
                "Na": [1],
                "K": [1],
                "Mg": [2],
                "Ca": [2],
                "Zn": [2],
                "Fe": [2, 3],
                "Cu": [1, 2],
                "Mn": [2, 3, 4],
                "Ru": [2, 3, 4],
            }

        # Load valence electrons
        try:
            ve_file = data_path / "valence_electrons.json"
            with ve_file.open("r") as f:
                ve_data = json.load(f)
            valence_electrons = {el: ve_data[el] for el in ve_data if not el.startswith("_")}
        except Exception as e:
            print(f"Warning: Could not load valence electrons file: {e}, using fallback")
            valence_electrons = {
                "H": 1,
                "C": 4,
                "N": 5,
                "O": 6,
                "F": 7,
                "P": 5,
                "S": 6,
                "Cl": 7,
                "Br": 7,
                "I": 7,
            }

        # Define metals set
        metals = {
            "Li",
            "Na",
            "K",
            "Mg",
            "Ca",
            "Zn",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Al",
            "Ga",
            "In",
            "Sn",
            "Pb",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        }

        # Load element mappings
        element_file = data_path / "atom_symbols.json"
        with element_file.open("r") as f:
            s2n = json.load(f)
        n2s = {v: k for k, v in s2n.items()}

        # Pauling electronegativity (determines which atom carries charge)
        electronegativity = {
            "H": 2.2,
            "C": 2.5,
            "N": 3.0,
            "O": 3.5,
            "F": 4.0,
            "P": 2.2,
            "S": 2.6,
            "Cl": 3.2,
            "Br": 3.0,
            "I": 2.7,
        }

        # Atoms eligible for Kekulé pattern initialization
        conjugatable_atoms = frozenset({"C", "N", "O", "S", "B", "P", "Se"})

        # Atoms that receive aromatic bond order 1.5
        aromatic_atoms = frozenset({"C", "N", "O", "S", "B"})

        # Atoms checked for conjugation penalty in scoring
        scoring_conjugatable_atoms = frozenset({"C", "N", "O", "S", "P"})

        # Max bond order sum for aromatic ring members
        max_aromatic_valence = {"H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "Se": 2}

        return cls(
            vdw=vdw_radii,
            valences=expected_valences,
            electrons=valence_electrons,
            metals=metals,
            s2n=s2n,
            n2s=n2s,
            electronegativity=electronegativity,
            conjugatable_atoms=conjugatable_atoms,
            aromatic_atoms=aromatic_atoms,
            scoring_conjugatable_atoms=scoring_conjugatable_atoms,
            max_aromatic_valence=max_aromatic_valence,
        )


# Eagerly instantiate on module import
DATA = MolecularData.get_instance()
