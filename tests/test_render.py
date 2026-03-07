"""Integration tests for end-to-end rendering."""

from pathlib import Path

import pytest

from xyzrender import load_molecule, render_svg
from xyzrender.types import RenderConfig

EXAMPLES = Path(__file__).parent.parent / "examples" / "structures"


def test_caffeine_renders():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<circle" in svg


def test_caffeine_gradient():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph, RenderConfig(gradient=True))
    assert "<use" in svg
    assert "<defs>" in svg


def test_ethanol_fog_mode():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(fog=True))
    assert "<circle" in svg
    assert "<use" not in svg


def test_gradient_and_fog():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph, RenderConfig(gradient=True, fog=True))
    assert "<use" in svg
    assert "<defs>" in svg


def test_hide_h():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    svg_show = render_svg(graph, RenderConfig(hide_h=False))
    svg_hide = render_svg(graph, RenderConfig(hide_h=True))
    assert svg_hide.count("<circle") < svg_show.count("<circle")


def test_bond_orders_off():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph, RenderConfig(bond_orders=False))
    assert "<svg" in svg
    assert "</svg>" in svg


def test_auto_orient():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    svg_orient = render_svg(graph, RenderConfig(auto_orient=True))
    svg_raw = render_svg(graph, RenderConfig(auto_orient=False))
    assert "<svg" in svg_orient
    assert "<svg" in svg_raw


def test_custom_canvas_size():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(canvas_size=400))
    # Canvas fits to molecule aspect ratio; larger dimension ≤ canvas_size
    assert 'width="' in svg
    assert 'height="' in svg
    import re

    m_w = re.search(r'width="(\d+)"', svg)
    m_h = re.search(r'height="(\d+)"', svg)
    assert m_w is not None
    assert m_h is not None
    w = int(m_w.group(1))
    h = int(m_h.group(1))
    assert max(w, h) <= 400
    assert min(w, h) > 0


def test_custom_background():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(background="#000000"))
    assert "#000000" in svg


def test_color_overrides():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(color_overrides={"O": "#00ff00"}))
    assert "#00ff00" in svg


def test_vdw_spheres():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(vdw_indices=[]))
    assert "vg" in svg


def test_log_suppression():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, _log=False)
    assert "<svg" in svg


def test_benzene_aromatic():
    graph, _ = load_molecule(EXAMPLES / "benzene.xyz")
    svg = render_svg(graph, RenderConfig(bond_orders=True, hide_h=False))
    assert "<line" in svg
    assert "<svg" in svg


def test_asparagine_renders():
    graph, _ = load_molecule(EXAMPLES / "asparagine.xyz")
    svg = render_svg(graph)
    assert "<svg" in svg
    assert "</svg>" in svg


# ---------------------------------------------------------------------------
# Cheminformatics format render tests
# ---------------------------------------------------------------------------


def test_caffeine_sdf_renders():
    pytest.importorskip("rdkit", reason="rdkit required for SDF rendering test")
    graph, _ = load_molecule(EXAMPLES / "caffeine_sdf.sdf")
    svg = render_svg(graph)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<circle" in svg


def test_water_mol2_renders():
    graph, _ = load_molecule(EXAMPLES / "water_mol2.mol2")
    svg = render_svg(graph)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<circle" in svg


def test_ala_phe_ala_pdb_renders():
    graph, crystal = load_molecule(EXAMPLES / "ala_phe_ala.pdb")
    svg = render_svg(graph)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<circle" in svg
    assert crystal is None


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
def test_caffeine_cif_renders():
    pytest.importorskip("ase", reason="ase required for CIF rendering test")
    from xyzrender.types import CrystalData

    graph, crystal = load_molecule(EXAMPLES / "caffeine_cif.cif")
    assert isinstance(crystal, CrystalData)
    svg = render_svg(graph)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<circle" in svg
