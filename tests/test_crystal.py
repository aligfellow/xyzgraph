"""Tests for crystal structure loading and rendering."""

import copy
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "structures"
VASP_FILE = EXAMPLES / "NV63.vasp"
QE_FILE = EXAMPLES / "NV63.in"
EXTXYZ_FILE = EXAMPLES / "caffeine_cell.xyz"


@pytest.fixture(scope="module")
def vasp_crystal():
    from xyzrender.crystal import load_crystal

    return load_crystal(VASP_FILE, "vasp")


@pytest.fixture(scope="module")
def qe_crystal():
    from xyzrender.crystal import load_crystal

    return load_crystal(QE_FILE, "qe")


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


def test_load_crystal_vasp(vasp_crystal):
    graph, crystal_data = vasp_crystal
    assert graph.number_of_nodes() == 63
    assert crystal_data.lattice.shape == (3, 3)


def test_load_crystal_qe(qe_crystal):
    graph, crystal_data = qe_crystal
    assert graph.number_of_nodes() == 63
    assert crystal_data.lattice.shape == (3, 3)


def test_load_crystal_vasp_qe_same_lattice(vasp_crystal, qe_crystal):
    """VASP and QE files describe the same structure — lattices must match."""
    _, cd_vasp = vasp_crystal
    _, cd_qe = qe_crystal
    np.testing.assert_allclose(cd_vasp.lattice, cd_qe.lattice, atol=1e-3)


def test_crystal_images(vasp_crystal):
    """add_crystal_images produces image nodes each bonded to ≥1 cell atom."""
    from xyzrender.crystal import add_crystal_images

    graph, crystal_data = copy.deepcopy(vasp_crystal)
    n_cell = graph.number_of_nodes()
    n_added = add_crystal_images(graph, crystal_data)

    assert n_added > 0, "Expected at least some image atoms"
    cell_ids = set(range(n_cell))

    for node_id, attrs in graph.nodes(data=True):
        if not attrs.get("image", False):
            continue
        # Every image atom must have at least one bond to a cell atom
        neighbors = list(graph.neighbors(node_id))
        cell_neighbors = [nb for nb in neighbors if nb in cell_ids]
        assert cell_neighbors, f"Image node {node_id} (sym={attrs['symbol']}) has no bond to a cell atom"


def test_crystal_images_no_orphans(vasp_crystal):
    """No image node may exist without at least one image_bond=True edge to a cell atom."""
    from xyzrender.crystal import add_crystal_images

    graph, crystal_data = copy.deepcopy(vasp_crystal)
    n_cell = graph.number_of_nodes()
    add_crystal_images(graph, crystal_data)

    cell_ids = set(range(n_cell))
    for node_id, attrs in graph.nodes(data=True):
        if not attrs.get("image", False):
            continue
        image_bonds_to_cell = [
            nb
            for nb in graph.neighbors(node_id)
            if nb in cell_ids and graph.edges[node_id, nb].get("image_bond", False)
        ]
        assert image_bonds_to_cell, f"Image node {node_id} has no image_bond edge to a cell atom"


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


def test_render_crystal_cell_box(vasp_crystal):
    """render_svg with crystal_data + show_cell=True produces exactly 12 cell edges."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = vasp_crystal
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=True)
    svg = render_svg(graph, cfg)

    # Count lines tagged as cell edges
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12, f"Expected 12 cell-box lines, got {len(cell_lines)}"


def test_render_no_cell(vasp_crystal):
    """render_svg with show_cell=False produces no cell-edge lines."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = vasp_crystal
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=False)
    svg = render_svg(graph, cfg)

    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 0


def test_render_crystal_no_crystal_data(vasp_crystal):
    """Crystal-specific SVG elements are absent when crystal_data is None."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, _crystal_data = vasp_crystal
    cfg = RenderConfig()
    svg = render_svg(graph, cfg)
    assert 'class="cell-edge"' not in svg


def test_render_crystal_with_images(vasp_crystal):
    """Image atoms render with opacity and produce a valid SVG."""
    from xyzrender.crystal import add_crystal_images
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = copy.deepcopy(vasp_crystal)
    add_crystal_images(graph, crystal_data)
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=True, periodic_image_opacity=0.5)
    svg = render_svg(graph, cfg)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert 'opacity="0.50"' in svg


def test_render_crystal_no_images(vasp_crystal):
    """Without add_crystal_images, no opacity attributes appear in atoms/bonds."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = vasp_crystal
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=True, periodic_image_opacity=0.5)
    svg = render_svg(graph, cfg)
    assert 'opacity="0.50"' not in svg


# ---------------------------------------------------------------------------
# extXYZ Lattice= tests (--cell path, no phonopy)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extxyz_graph():
    from xyzrender.io import load_molecule

    return load_molecule(EXTXYZ_FILE)


def test_extxyz_lattice_parsed(extxyz_graph):
    """extXYZ file with Lattice= stores a (3, 3) lattice on graph.graph."""
    lat = np.array(extxyz_graph.graph["lattice"])
    assert lat.shape == (3, 3)


def test_extxyz_lattice_values(extxyz_graph):
    """Lattice= row-major values are parsed correctly."""
    lat = np.array(extxyz_graph.graph["lattice"])
    # caffeine_cell.xyz: Lattice="14.8 0.0 0.0  0.0 16.7 0.0  -0.484 0.0 3.940"
    np.testing.assert_allclose(lat[0, 0], 14.8, atol=1e-3)
    np.testing.assert_allclose(lat[1, 1], 16.7, atol=1e-3)
    np.testing.assert_allclose(lat[2, 2], 3.940, atol=1e-3)


def test_extxyz_cell_box_renders(extxyz_graph):
    """extXYZ --cell path: CrystalData from graph.graph produces 12 cell edges."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import CrystalData, RenderConfig

    cfg = RenderConfig(
        crystal_data=CrystalData(lattice=np.array(extxyz_graph.graph["lattice"], dtype=float)),
        show_cell=True,
        show_crystal_axes=False,
    )
    svg = render_svg(extxyz_graph, cfg)
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12
