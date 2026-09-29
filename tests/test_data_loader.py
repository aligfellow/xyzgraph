"""Tests for molecular reference data loader."""

from xyzgraph.data_loader import DATA, MolecularData


def test_singleton():
    """DATA is loaded once and reused."""
    assert MolecularData.get_instance() is DATA


def test_core_data_present():
    """Essential fields are populated."""
    assert "C" in DATA.vdw
    assert "C" in DATA.valences
    assert "Fe" in DATA.metals
    assert "C" not in DATA.metals
    assert DATA.electronegativity["O"] > DATA.electronegativity["C"]
    # The radius table is keyed by real element symbols (Gd and Ho were once misspelt).
    assert set(DATA.vdw) <= set(DATA.s2n)
    # A metal's usual oxidation states never exceed the electrons it has to give.
    assert all(max(DATA.valences.get(m, [0])) <= DATA.electrons[m] for m in DATA.metals)
