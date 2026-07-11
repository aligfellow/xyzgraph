"""Aromaticity perception tests.

One integration case per rule branch of the aromaticity model, which asks a
single question per ring atom: is its p-orbital available to the ring
(contributing 1 electron from a ring π bond or 2 from a lone pair), or is it
committed elsewhere / empty (contributing 0)?

  * a carbon with an exocyclic double bond contributes 0 (masked cation);
  * a trigonal carbon with only single bonds contributes 0 (empty p);
  * a lone-pair donor (N, O, S) with an exocyclic double bond contributes 0;
  * >= 2 zero-contribution carbons on a non-anionic ring system means
    cross-conjugation, never aromaticity, whatever the Hückel total —
    exactly one is allowed (betaine resonance: pyridones, pyranones);
  * anionic ring systems (charge summed over ring + direct substituents,
    e.g. croconate's exocyclic olate oxygens) are left to the Hückel count.

This is the classical model — caffeine's pyrimidinedione ring is NOT
aromatic — and deliberately disagrees with RDKit's default on the
>= 2 exocyclic-π cases (uracil, quinones, xanthines, ...).

Geometries are embedded from SMILES at test time (RDKit, fixed seed, MMFF
with UFF fallback). Perception was verified identical across three embedding
seeds per molecule, so the result does not hinge on the exact conformer.

Each case asserts:
  arom - sorted element compositions of the aromatic rings ("C4N2" = a
         six-membered ring of 4 C + 2 N; [] = no aromatic ring)
  n15  - number of edges at bond order 1.5
  co2  - number of localised C=O / C=S double bonds (order 2.0), or None
         to skip the check
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from xyzgraph import build_graph
from xyzgraph.utils import graph_to_dict

if TYPE_CHECKING:
    from pathlib import Path

# fmt: off
CASES = [
    # plain aromatics and a substituent control (C=O one bond off the ring)
    ("benzene",             "c1ccccc1",                          ["C6"],           6,  None),
    ("imidazole",           "c1c[nH]cn1",                        ["C3N2"],         5,  None),
    ("benzaldehyde",        "O=Cc1ccccc1",                       ["C6"],           6,  1),
    # fused systems: SSSR path (naphthalene) and perimeter path (azulene 5-7)
    ("naphthalene",         "c1ccc2ccccc2c1",                    ["C6", "C6"],     11, None),
    ("azulene",             "c1ccc2cccc2cc1",                    ["C5", "C7"],     11, None),
    # exactly one exocyclic-π carbon: betaine form allowed, ring aromatic
    # (behaviour change vs v1.6.10, which called the pyridones non-aromatic)
    ("2-pyridone",          "O=c1cccc[nH]1",                     ["C5N1"],         6,  1),
    ("2H-pyran-2-one",      "O=c1cccco1",                        ["C5O1"],         6,  1),
    ("cytosine",            "Nc1cc[nH]c(=O)n1",                  ["C4N2"],         6,  1),
    ("guanine",             "Nc1nc2[nH]cnc2c(=O)[nH]1",          ["C3N2", "C4N2"], 10, 1),
    # one exocyclic-π carbon but the Hückel count fails
    ("cyclopentadienone",   "O=C1C=CC=C1",                       [],               0,  1),
    ("fulvene",             "C=C1C=CC=C1",                       [],               0,  None),
    # >= 2 exocyclic-π carbons: cross-conjugated, never aromatic; the
    # carbonyls stay localised double bonds
    ("p-benzoquinone",      "O=C1C=CC(=O)C=C1",                  [],               0,  2),
    ("uracil",              "O=c1cc[nH]c(=O)[nH]1",              [],               0,  2),
    ("2-thiouracil",        "S=c1cc[nH]c(=O)[nH]1",              [],               0,  2),  # C=S counts too
    # nitro N+ (a direct substituent charge) doesn't lift the guard
    ("5-nitrouracil",       "O=c1[nH]cc(c(=O)[nH]1)[N+](=O)[O-]", [],              0,  2),
    ("caffeine",            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",        ["C3N2"],         5,  2),  # imidazole ring survives
    # fused: benzo rings keep aromaticity, the dione ring is excluded;
    # acenaphthenequinone exercises the perimeter-path guard, indigo the
    # inter-ring C=C (partner atom lives in another ring)
    ("anthraquinone",       "O=C1c2ccccc2C(=O)c2ccccc21",        ["C6", "C6"],     12, 2),
    ("acenaphthenequinone", "O=C1C(=O)c2cccc3cccc1c23",          ["C6", "C6"],     11, 2),
    ("indigo",              "O=C1Nc2ccccc2/C1=C1\\C(=O)Nc2ccccc21", ["C6", "C6"],  12, 2),
    # exocyclic π partner inside its own ring (cyclopropylidene): still no
    # p-electron for the pyranone ring — must not aromatise (and must not
    # produce valence-5 carbons)
    ("biscyclopropylidene-pyranone", "O=C1OC(=C2CC2)C=CC1=C3CC3", [],              0,  1),
    # lone-pair donor with its p-orbital committed exocyclically: sulfonyl S
    # donates nothing, so only the benzo ring is aromatic. (co2 unchecked:
    # the optimizer picks the pre-existing zwitterionic C=N+/O- Kekulé.)
    ("saccharin",           "O=C1NS(=O)(=O)c2ccccc21",           ["C6"],           6,  None),
    # charged rings: empty-p carbon counted for pyrylium (aromatic cation);
    # anionic ring systems escape the cross-conjugation guard
    ("pyrylium",            "c1cc[o+]cc1",                       ["C5O1"],         6,  None),
    ("cyclopentadienide",   "[cH-]1cccc1",                       ["C5"],           5,  None),
    ("croconate_dianion",   "[O-]C1=C([O-])C(=O)C(=O)C1=O",      ["C5"],           5,  3),
]
# fmt: on


def _xyz_from_smiles(smiles: str, name: str, directory: Path) -> tuple[str, int]:
    """Embed a 3D geometry from SMILES, write it as .xyz, return (path, charge)."""
    parsed = Chem.MolFromSmiles(smiles)
    assert parsed is not None, f"bad SMILES: {name}"
    mol = Chem.AddHs(parsed)
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=42) == 0, f"embedding failed: {name}"
    try:
        converged = rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    except ValueError:
        converged = -1
    if converged != 0:
        rdForceFieldHelpers.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), name]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    path = directory / f"{name}.xyz"
    path.write_text("\n".join(lines) + "\n")
    return str(path), Chem.GetFormalCharge(mol)


def _ring_composition(nodes: list[dict], ring: list[int]) -> str:
    counts = Counter(nodes[i]["symbol"] for i in ring)
    return "".join(f"{sym}{n}" for sym, n in sorted(counts.items()))


@pytest.mark.parametrize(("name", "smiles", "arom", "n15", "co2"), CASES)
def test_aromaticity(name, smiles, arom, n15, co2, tmp_path):
    path, charge = _xyz_from_smiles(smiles, name, tmp_path)
    G = build_graph(path, charge=charge)
    result = graph_to_dict(G)
    nodes = result["nodes"]

    rings = result["graph"].get("aromatic_rings") or []
    assert sorted(_ring_composition(nodes, r) for r in rings) == arom

    delocalised = [e for e in result["edges"] if e["bond_order"] == pytest.approx(1.5)]
    assert len(delocalised) == n15

    if co2 is not None:
        cx = [
            e
            for e in result["edges"]
            if {nodes[e["idx1"]]["symbol"], nodes[e["idx2"]]["symbol"]} in ({"C", "O"}, {"C", "S"})
            and e["bond_order"] == pytest.approx(2.0)
        ]
        assert len(cx) == co2
