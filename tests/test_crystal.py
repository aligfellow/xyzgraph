"""Tests for crystal structure loading and rendering."""

from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "structures"
VASP_FILE = EXAMPLES / "NV63.vasp"
QE_FILE = EXAMPLES / "NV63.in"

# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


def test_load_crystal_vasp():
    from xyzrender.io import load_crystal

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
    assert graph.number_of_nodes() == 63
    assert crystal_data.lattice.shape == (3, 3)


def test_load_crystal_qe():
    from xyzrender.io import load_crystal

    graph, crystal_data = load_crystal(QE_FILE, "qe")
    assert graph.number_of_nodes() == 63
    assert crystal_data.lattice.shape == (3, 3)


def test_load_crystal_vasp_qe_same_lattice():
    """VASP and QE files describe the same structure — lattices must match."""
    from xyzrender.io import load_crystal

    _, cd_vasp = load_crystal(VASP_FILE, "vasp")
    _, cd_qe = load_crystal(QE_FILE, "qe")
    np.testing.assert_allclose(cd_vasp.lattice, cd_qe.lattice, atol=1e-3)


def test_crystal_images():
    """add_crystal_images produces image nodes each bonded to ≥1 cell atom."""
    from xyzrender.io import add_crystal_images, load_crystal

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
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
        assert cell_neighbors, (
            f"Image node {node_id} (sym={attrs['symbol']}) has no bond to a cell atom"
        )


def test_crystal_images_no_orphans():
    """No image node may exist without at least one image_bond=True edge to a cell atom."""
    from xyzrender.io import add_crystal_images, load_crystal

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
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
        assert image_bonds_to_cell, (
            f"Image node {node_id} has no image_bond edge to a cell atom"
        )


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


def test_render_crystal_cell_box():
    """render_svg with crystal_data + show_cell=True produces exactly 12 cell edges."""
    from xyzrender.io import load_crystal
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=True)
    svg = render_svg(graph, cfg)

    # Count lines tagged as cell edges
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12, f"Expected 12 cell-box lines, got {len(cell_lines)}"


def test_render_no_cell():
    """render_svg with show_cell=False produces no cell-edge lines."""
    from xyzrender.io import load_crystal
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=False)
    svg = render_svg(graph, cfg)

    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 0


def test_render_crystal_no_crystal_data():
    """Crystal-specific SVG elements are absent when crystal_data is None."""
    from xyzrender.io import load_crystal
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, _crystal_data = load_crystal(VASP_FILE, "vasp")
    # Deliberately omit crystal_data
    cfg = RenderConfig()
    svg = render_svg(graph, cfg)
    assert 'class="cell-edge"' not in svg


def test_render_crystal_with_images():
    """Image atoms render with opacity and produce a valid SVG."""
    from xyzrender.io import add_crystal_images, load_crystal
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
    add_crystal_images(graph, crystal_data)
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=True, periodic_image_opacity=0.5)
    svg = render_svg(graph, cfg)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    # At least one opacity attribute should appear (from image atoms/bonds)
    assert 'opacity="0.50"' in svg


def test_render_crystal_no_images():
    """Without add_crystal_images, no opacity attributes appear in atoms/bonds."""
    from xyzrender.io import load_crystal
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, crystal_data = load_crystal(VASP_FILE, "vasp")
    # Don't add images
    cfg = RenderConfig(crystal_data=crystal_data, show_cell=True, periodic_image_opacity=0.5)
    svg = render_svg(graph, cfg)
    assert 'opacity="0.50"' not in svg
