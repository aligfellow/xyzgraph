"""Basic tests for bond detection.

Verifies extraction from GraphBuilder works correctly.
Tests use molecular contexts: methane, water, diatomic H2.
"""

from itertools import combinations

import pytest

from xyzgraph.bond_detection import BondDetector
from xyzgraph.bond_geometry_check import BondGeometryChecker
from xyzgraph.data_loader import DATA
from xyzgraph.geometry import GeometryCalculator
from xyzgraph.parameters import BondThresholds, GeometryThresholds


@pytest.fixture
def detector():
    """BondDetector with default thresholds."""
    geometry = GeometryCalculator()
    bond_checker = BondGeometryChecker(
        geometry=geometry,
        thresholds=GeometryThresholds.strict(),
        data=DATA,
    )
    return BondDetector(
        geometry=geometry,
        bond_checker=bond_checker,
        thresholds=BondThresholds(),
        data=DATA,
    )


def test_methane_connectivity(detector):
    """Methane: 5 atoms, 4 C-H bonds."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    # All bonds are C-H
    for i, j in G.edges():
        syms = {G.nodes[i]["symbol"], G.nodes[j]["symbol"]}
        assert syms == {"C", "H"}


def _dimer(metal, half, bridge, h):
    """Two ``metal`` atoms at +-half on x and two ``bridge`` atoms at +-h on y (atoms 0, 1 and 2, 3)."""
    return [(metal, (half, 0.0, 0.0)), (metal, (-half, 0.0, 0.0)), (bridge, (0.0, h, 0.0)), (bridge, (0.0, -h, 0.0))]


@pytest.mark.parametrize(
    ("atoms", "spanned", "bridges"),
    [
        pytest.param(  # diborane, gas-phase geometry: B-Hb 1.33, B-Ht 1.19, B...B 1.77 A
            [*_dimer("B", 0.885, "H", 0.99), *[("H", (x, 0.0, z)) for x in (1.47, -1.47) for z in (1.04, -1.04)]],
            (0, 1),
            (2, 3),
            id="B2H6",
        ),
        pytest.param(  # HMg(mu-H)2MgH: Mg-Hb 1.79, Mg-Ht 1.70, Mg...Mg 2.50 A
            [*_dimer("Mg", 1.25, "H", 1.28), ("H", (2.95, 0.0, 0.0)), ("H", (-2.95, 0.0, 0.0))],
            (0, 1),
            (2, 3),
            id="Mg2H4",
        ),
        pytest.param(  # HZn(mu-H)2ZnH: Zn-Hb 1.70, Zn-Ht 1.55, Zn...Zn 2.40 A
            [*_dimer("Zn", 1.2, "H", 1.2), ("H", (2.75, 0.0, 0.0)), ("H", (-2.75, 0.0, 0.0))],
            (0, 1),
            (2, 3),
            id="Zn2H4",
        ),
        pytest.param(  # [H3B(mu-H)BH3]-, wide and asymmetric: B-Hb 1.24/1.47, B-H-B 96 deg, B...B 2.02 A
            [
                ("B", (1.24, 0.0, 0.0)),
                ("B", (-0.154, 1.462, 0.0)),
                ("H", (0.0, 0.0, 0.0)),
                ("H", (1.644, -0.953, 0.627)),
                ("H", (1.644, -0.067, -1.139)),
                ("H", (1.644, 1.019, 0.512)),
                ("H", (0.893, 1.781, -0.493)),
                ("H", (-1.134, 1.568, -0.684)),
                ("H", (-0.285, 1.658, 1.177)),
            ],
            (0, 1),
            (2,),
            id="B2H7-wide",
        ),
        pytest.param(  # HCa(mu-H)2CaH: Ca-Hb 2.30 A, an ionic hydride as long as a Ca-O bond
            [*_dimer("Ca", 1.696, "H", 1.554), ("H", (3.676, 0.0, 0.0)), ("H", (-3.676, 0.0, 0.0))],
            (0, 1),
            (2, 3),
            id="Ca2H4",
        ),
        pytest.param(  # HU(mu-H)2UH: U-Hb 2.15 A
            [*_dimer("U", 1.647, "H", 1.382), ("H", (3.582, 0.0, 0.0)), ("H", (-3.582, 0.0, 0.0))],
            (0, 1),
            (2, 3),
            id="U2H4",
        ),
        pytest.param(  # Cu3(mu3-H)Cl3: Cu-H 1.75, Cu...Cu 2.6 A
            [
                ("Cu", (1.5, 0.0, 0.0)),
                ("Cu", (-0.75, 1.3, 0.0)),
                ("Cu", (-0.75, -1.3, 0.0)),
                ("H", (0.0, 0.0, 0.9)),
                ("Cl", (3.5, 0.0, 0.0)),
                ("Cl", (-1.75, 3.03, 0.0)),
                ("Cl", (-1.75, -3.03, 0.0)),
            ],
            (0, 1, 2),
            (3,),
            id="Cu3-mu3H",
        ),
        pytest.param(  # H2Al(mu-CH3)2AlH2: Al-Cb 2.12, Al...Al 2.60 A; each methyl shares one pair, as diborane's H
            [
                *_dimer("Al", 1.3, "C", 1.66),
                ("H", (0.956, 2.024, -0.375)),
                ("H", (-0.803, 2.024, -0.641)),
                ("H", (-0.153, 2.024, 1.016)),
                ("H", (-0.956, -2.024, 0.375)),
                ("H", (0.153, -2.024, -1.016)),
                ("H", (0.803, -2.024, 0.641)),
                *[("H", (x, 0.0, z)) for x in (2.09, -2.09) for z in (1.368, -1.368)],
            ],
            (0, 1),
            (2, 3),
            id="Al2Me2H4",
        ),
    ],
)
def test_one_pair_bridge_bonds_every_leg_and_no_diagonal(detector, atoms, spanned, bridges):
    """A bridge sharing one pair (a hydride, a mu-CH3 between metals) bonds each atom it spans; none bond each other."""
    G = detector.detect(atoms)
    assert not any(G.has_edge(a, b) for a, b in combinations(spanned, 2))
    assert all(G.has_edge(x, a) for x in bridges for a in spanned)


@pytest.mark.parametrize(
    "atoms",
    [
        pytest.param(  # Rh2(mu-CH2)Cl4: Rh-C 2.03, Rh-Rh 2.67 A
            [
                ("Rh", (1.335, 0.0, 0.0)),
                ("Rh", (-1.335, 0.0, 0.0)),
                ("C", (0.0, 1.529, 0.0)),
                ("H", (0.153, 1.893, -1.016)),
                ("H", (0.803, 1.893, 0.641)),
                ("Cl", (2.935, -1.0, 1.3)),
                ("Cl", (2.935, -1.0, -1.3)),
                ("Cl", (-2.935, -1.0, 1.3)),
                ("Cl", (-2.935, -1.0, -1.3)),
            ],
            id="mu-CH2",
        ),
        pytest.param(  # Pd2(mu-Cp)Cl2: one ring C on both Pd (2.20 A), its ring neighbours on one each (2.31 A)
            [
                ("Pd", (1.3, 0.0, 0.0)),
                ("Pd", (-1.3, 0.0, 0.0)),
                ("C", (0.0, 0.0, 1.775)),
                ("C", (1.149, 0.723, 2.192)),
                ("C", (0.71, 1.892, 2.868)),
                ("C", (-0.71, 1.892, 2.868)),
                ("C", (-1.149, 0.723, 2.192)),
                ("H", (0.0, -0.935, 1.235)),
                ("H", (2.176, 0.434, 2.025)),
                ("H", (1.345, 2.649, 3.304)),
                ("H", (-1.345, 2.649, 3.304)),
                ("H", (-2.176, 0.434, 2.025)),
                ("Cl", (3.6, 0.0, 0.0)),
                ("Cl", (-3.6, 0.0, 0.0)),
            ],
            id="mu-Cp",
        ),
    ],
)
def test_bridge_giving_more_than_one_pair_keeps_the_metal_bond(detector, atoms):
    """A mu-CH2 has a pair for each metal and a Cp face gives its ring's pi electrons: the M-M bond under them stays."""
    G = detector.detect(atoms)
    assert G.has_edge(0, 1)
    assert G.has_edge(2, 0)
    assert G.has_edge(2, 1)


def test_water_connectivity(detector):
    """Water: 3 atoms, 2 O-H bonds."""
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.757, 0.586, 0.0)),
        ("H", (-0.757, 0.586, 0.0)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2


def test_h2_diatomic(detector):
    """H2: 2 atoms, 1 H-H bond."""
    atoms = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.74, 0.0, 0.0)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_formula_hill_system(detector):
    """Chemical formula follows Hill system (C, H, then alphabetical)."""
    from xyzgraph.utils import compute_formula

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("C", (1.2, 0.0, 0.0)),
        ("H", (1.8, 0.9, 0.0)),
    ]
    G = detector.detect(atoms)
    compute_formula(G)
    assert G.graph["formula"] == "CHO"


def test_user_specified_bond(detector):
    """User-specified bonds are added."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (5.0, 0.0, 0.0)),  # Too far for automatic detection
    ]
    G = detector.detect(atoms, bond=[(0, 1)])
    assert G.has_edge(0, 1)


def test_user_specified_unbond(detector):
    """User-specified unbonds remove detected bonds."""
    atoms = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629)),
    ]
    G = detector.detect(atoms, unbond=[(0, 1)])
    assert not G.has_edge(0, 1)
    assert G.number_of_edges() == 3  # 4 - 1


def test_benzene_ring_detection(detector):
    """Benzene: 6 C-C bonds form a ring."""
    atoms = [
        ("C", (1.396, 0.000, 0.000)),
        ("C", (0.698, 1.209, 0.000)),
        ("C", (-0.698, 1.209, 0.000)),
        ("C", (-1.396, 0.000, 0.000)),
        ("C", (-0.698, -1.209, 0.000)),
        ("C", (0.698, -1.209, 0.000)),
        ("H", (2.479, 0.000, 0.000)),
        ("H", (1.240, 2.147, 0.000)),
        ("H", (-1.240, 2.147, 0.000)),
        ("H", (-2.479, 0.000, 0.000)),
        ("H", (-1.240, -2.147, 0.000)),
        ("H", (1.240, -2.147, 0.000)),
    ]
    G = detector.detect(atoms)
    assert G.number_of_nodes() == 12
    assert G.number_of_edges() == 12  # 6 C-C + 6 C-H
    assert len(G.graph["_rings"]) >= 1  # At least one ring detected


def test_unbond_invalidates_ring_cache(detector):
    """Ring cache must not reference edges removed by unbond.

    Regression test: unbond removing a ring edge left a stale _rings cache,
    causing KeyError in downstream bond-order optimization when iterating
    cached ring edges that no longer exist in the graph.
    """
    # Benzene geometry — has a 6-membered ring
    atoms = [
        ("C", (1.396, 0.000, 0.000)),
        ("C", (0.698, 1.209, 0.000)),
        ("C", (-0.698, 1.209, 0.000)),
        ("C", (-1.396, 0.000, 0.000)),
        ("C", (-0.698, -1.209, 0.000)),
        ("C", (0.698, -1.209, 0.000)),
        ("H", (2.479, 0.000, 0.000)),
        ("H", (1.240, 2.147, 0.000)),
        ("H", (-1.240, 2.147, 0.000)),
        ("H", (-2.479, 0.000, 0.000)),
        ("H", (-1.240, -2.147, 0.000)),
        ("H", (1.240, -2.147, 0.000)),
    ]
    # Remove edge (0,1) which is part of the ring
    G = detector.detect(atoms, unbond=[(0, 1)])
    assert not G.has_edge(0, 1)

    # Ring cache must not contain any ring that references the removed edge
    for ring in G.graph["_rings"]:
        edges = list(zip(ring, [*ring[1:], ring[0]]))
        for i, j in edges:
            assert G.has_edge(i, j) or G.has_edge(j, i), (
                f"Stale ring cache: ring {ring} references removed edge ({i}, {j})"
            )
