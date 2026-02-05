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
    assert DATA.aromatic_atoms < DATA.conjugatable_atoms
