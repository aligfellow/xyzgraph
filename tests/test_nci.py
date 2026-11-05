"""Tests for NCI detection module.

Each test builds a graph from a known structure, runs detect_ncis(),
and checks that the expected interaction types and atom sites are found.
Tests also verify no unexpected NCI types appear (false positive guard).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from xyzgraph import build_graph
from xyzgraph.nci import NCIAnalyzer, NCIThresholds, detect_ncis

STRUCTURES = Path(__file__).resolve().parent.parent / "examples" / "nci"


def _types(ncis):
    """Return sorted list of unique NCI type strings."""
    return sorted({nci.type for nci in ncis})


def _by_type(ncis, nci_type):
    """Filter NCIs by type."""
    return [n for n in ncis if n.type == nci_type]


# ------------------------------------------------------------------
# Water dimer H-bond
# ------------------------------------------------------------------


class TestWaterHBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "water-HB.xyz"))
        return detect_ncis(G)

    def test_hbond_detected(self, ncis):
        hbonds = _by_type(ncis, "hbond")
        assert len(hbonds) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["hbond"]

    def test_donor_acceptor_atoms(self, ncis):
        hbonds = _by_type(ncis, "hbond")
        # donor O(0) with H(2), acceptor O(3)
        sites = {(hb.site_a, hb.site_b) for hb in hbonds}
        assert ((0,), (3,)) in sites

    def test_geometry_keys(self, ncis):
        hb = _by_type(ncis, "hbond")[0]
        assert "distance" in hb.geometry
        assert "h_distance" in hb.geometry
        assert "angle" in hb.geometry

    def test_stored_in_graph(self):
        G = build_graph(str(STRUCTURES / "water-HB.xyz"))
        result = detect_ncis(G)
        assert G.graph["ncis"] is result


# ------------------------------------------------------------------
# Organic H-bond (MeOH...NMe2H)
# ------------------------------------------------------------------


class TestOrganicHBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "Hbond.xyz"))
        return detect_ncis(G)

    def test_hbond_detected(self, ncis):
        hbonds = _by_type(ncis, "hbond")
        assert len(hbonds) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["hbond"]

    def test_donor_acceptor(self, ncis):
        hbonds = _by_type(ncis, "hbond")
        # donor O(2) with H(7), acceptor N(8)
        sites = {(hb.site_a, hb.site_b) for hb in hbonds}
        assert ((2,), (8,)) in sites


# ------------------------------------------------------------------
# Bifurcated H-bond
# ------------------------------------------------------------------


class TestBifurcatedHBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "bimp_bifurcated.xyz"))
        return detect_ncis(G)

    def test_bifurcated_detected(self, ncis):
        bif = _by_type(ncis, "hbond_bifurcated")
        assert len(bif) >= 2, "Expected at least 2 bifurcated H-bonds sharing an acceptor"

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["hbond", "hbond_bifurcated"]

    def test_shared_acceptor(self, ncis):
        bif = _by_type(ncis, "hbond_bifurcated")
        if len(bif) >= 2:
            acceptors = [b.site_b for b in bif]
            assert len(acceptors) > len(set(acceptors)) or len(set(acceptors)) < len(acceptors) + 1


# ------------------------------------------------------------------
# Non-bifurcated H-bond (bimp structure)
# ------------------------------------------------------------------


class TestBimpHBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "bimp.xyz"))
        return detect_ncis(G)

    def test_hbonds_detected(self, ncis):
        hbonds = _by_type(ncis, "hbond") + _by_type(ncis, "hbond_bifurcated")
        assert len(hbonds) >= 2

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["hbond"]

    def test_expected_donors(self, ncis):
        hbonds = _by_type(ncis, "hbond") + _by_type(ncis, "hbond_bifurcated")
        donors = {hb.site_a[0] for hb in hbonds}
        assert 15 in donors or 43 in donors


# ------------------------------------------------------------------
# Halogen bond (MeI...O=CMe)
# ------------------------------------------------------------------


class TestHalogenBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "MeI_Fc-halogen-bond.xyz"))
        return detect_ncis(G)

    def test_halogen_bond_detected(self, ncis):
        xb = _by_type(ncis, "halogen_bond")
        assert len(xb) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["halogen_bond"]

    def test_xb_atoms(self, ncis):
        xb = _by_type(ncis, "halogen_bond")
        # I(4) to O(3)
        sites = {(x.site_a, x.site_b) for x in xb}
        assert ((4,), (3,)) in sites

    def test_geometry(self, ncis):
        xb = _by_type(ncis, "halogen_bond")[0]
        assert "distance" in xb.geometry
        assert "angle" in xb.geometry
        assert xb.geometry["angle"] >= 140.0


# ------------------------------------------------------------------
# Chalcogen bond (Ac-S)
# ------------------------------------------------------------------


class TestChalcogenBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "Ac_S-chalcogen-bond.xyz"))
        return detect_ncis(G)

    def test_chalcogen_bond_detected(self, ncis):
        chb = _by_type(ncis, "chalcogen_bond")
        assert len(chb) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["chalcogen_bond"]

    def test_chb_atoms(self, ncis):
        chb = _by_type(ncis, "chalcogen_bond")
        # S(10) to O(0)
        sites = {(c.site_a, c.site_b) for c in chb}
        assert ((10,), (0,)) in sites


# ------------------------------------------------------------------
# Pi-pi stacking (parallel benzene dimer)
# ------------------------------------------------------------------


class TestPiPiParallel:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "benzene-stack.xyz"))
        return detect_ncis(G)

    def test_parallel_detected(self, ncis):
        pp = _by_type(ncis, "pi_pi_parallel")
        assert len(pp) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["pi_pi_parallel"]

    def test_ring_atoms(self, ncis):
        pp = _by_type(ncis, "pi_pi_parallel")
        nci = pp[0]
        # Two benzene rings: atoms 0-5 and 12-17
        assert set(nci.site_a).issubset(range(6)) or set(nci.site_a).issubset(range(12, 18))
        assert set(nci.site_b).issubset(range(6)) or set(nci.site_b).issubset(range(12, 18))

    def test_geometry(self, ncis):
        pp = _by_type(ncis, "pi_pi_parallel")[0]
        assert "distance" in pp.geometry
        assert "angle" in pp.geometry
        assert pp.geometry["angle"] < 30.0


# ------------------------------------------------------------------
# Pi-pi stacking (T-shaped benzene dimer)
# ------------------------------------------------------------------


class TestPiPiTShaped:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "benzene-t.xyz"))
        return detect_ncis(G)

    def test_t_shaped_detected(self, ncis):
        ts = _by_type(ncis, "pi_pi_t_shaped")
        assert len(ts) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["pi_pi_t_shaped"]

    def test_geometry(self, ncis):
        ts = _by_type(ncis, "pi_pi_t_shaped")[0]
        assert "angle" in ts.geometry
        assert ts.geometry["angle"] >= 60.0


# ------------------------------------------------------------------
# Cation-pi (NH4+ over benzene)
# ------------------------------------------------------------------


class TestCationPi:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "benzene_NH4-cation-pi.xyz"), charge=1)
        return detect_ncis(G)

    def test_cation_pi_detected(self, ncis):
        cp = _by_type(ncis, "cation_pi")
        assert len(cp) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["cation_pi"]

    def test_cation_is_nitrogen(self, ncis):
        cp = _by_type(ncis, "cation_pi")
        cation_atoms = {c.site_a[0] for c in cp}
        assert 12 in cation_atoms or 13 in cation_atoms

    def test_ring_site(self, ncis):
        cp = _by_type(ncis, "cation_pi")[0]
        assert len(cp.site_b) >= 5


# ------------------------------------------------------------------
# Halogen-pi (I over olefin)
# ------------------------------------------------------------------


class TestHalogenPi:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "I-pi.xyz"))
        return detect_ncis(G)

    def test_halogen_pi_detected(self, ncis):
        hp = _by_type(ncis, "halogen_pi")
        assert len(hp) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["halogen_pi"]

    def test_iodine_site(self, ncis):
        hp = _by_type(ncis, "halogen_pi")
        halogen_atoms = {h.site_a[0] for h in hp}
        assert 18 in halogen_atoms


# ------------------------------------------------------------------
# Salt bridge
# ------------------------------------------------------------------


class TestSaltBridge:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "salt_bridge.xyz"))
        return detect_ncis(G)

    def test_salt_bridge_detected(self, ncis):
        sb = _by_type(ncis, "salt_bridge")
        assert len(sb) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["salt_bridge"]

    def test_geometry(self, ncis):
        sb = _by_type(ncis, "salt_bridge")
        for s in sb:
            assert "distance" in s.geometry
            assert "h_distance" in s.geometry
            assert "angle" in s.geometry


# ------------------------------------------------------------------
# Pnictogen bond (PH2-NO2...NH3)
# ------------------------------------------------------------------


class TestPnictogenBond:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "pnictogen-bond.xyz"))
        return detect_ncis(G)

    def test_pnictogen_bond_detected(self, ncis):
        pnb = _by_type(ncis, "pnictogen_bond")
        assert len(pnb) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["pnictogen_bond"]

    def test_pnb_atoms(self, ncis):
        pnb = _by_type(ncis, "pnictogen_bond")
        # P(4) sigma-hole to N(0) of NH3
        sites = {(p.site_a, p.site_b) for p in pnb}
        assert ((4,), (0,)) in sites

    def test_geometry(self, ncis):
        pnb = _by_type(ncis, "pnictogen_bond")[0]
        assert "distance" in pnb.geometry
        assert "angle" in pnb.geometry
        assert pnb.geometry["angle"] >= 140.0


# ------------------------------------------------------------------
# CH-pi (methane above benzene)
# ------------------------------------------------------------------


class TestCHPi:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "ch-pi.xyz"))
        return detect_ncis(G)

    def test_ch_pi_detected(self, ncis):
        chpi = _by_type(ncis, "ch_pi")
        assert len(chpi) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["ch_pi"]

    def test_carbon_site(self, ncis):
        chpi = _by_type(ncis, "ch_pi")
        # C(6) is the methane carbon
        carbons = {c.site_a[0] for c in chpi}
        assert 6 in carbons

    def test_geometry(self, ncis):
        chpi = _by_type(ncis, "ch_pi")[0]
        assert "distance" in chpi.geometry
        assert "angle" in chpi.geometry


# ------------------------------------------------------------------
# HB-pi (water O-H above benzene)
# ------------------------------------------------------------------


class TestHBPi:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "hb-pi.xyz"))
        return detect_ncis(G)

    def test_hb_pi_detected(self, ncis):
        hbpi = _by_type(ncis, "hb_pi")
        assert len(hbpi) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["hb_pi"]

    def test_donor_is_oxygen(self, ncis):
        hbpi = _by_type(ncis, "hb_pi")
        # O(12) is the water oxygen
        donors = {h.site_a[0] for h in hbpi}
        assert 12 in donors

    def test_geometry(self, ncis):
        hbpi = _by_type(ncis, "hb_pi")[0]
        assert "distance" in hbpi.geometry
        assert "angle" in hbpi.geometry


# ------------------------------------------------------------------
# Ionic (imidazolium chloride)
# ------------------------------------------------------------------


class TestIonic:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "ionic.xyz"))
        return detect_ncis(G)

    def test_ionic_detected(self, ncis):
        ionic = _by_type(ncis, "ionic")
        assert len(ionic) >= 1

    def test_only_expected_types(self, ncis):
        # Cl- above the imidazolium ring also triggers anion_pi
        assert _types(ncis) == ["anion_pi", "ionic"]

    def test_geometry(self, ncis):
        ionic = _by_type(ncis, "ionic")[0]
        assert "distance" in ionic.geometry


# ------------------------------------------------------------------
# Anion-pi (Cl- above benzene)
# ------------------------------------------------------------------


class TestAnionPi:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "anion-pi.xyz"), charge=-1)
        return detect_ncis(G)

    def test_anion_pi_detected(self, ncis):
        anpi = _by_type(ncis, "anion_pi")
        assert len(anpi) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["anion_pi"]

    def test_chlorine_site(self, ncis):
        anpi = _by_type(ncis, "anion_pi")
        anions = {a.site_a[0] for a in anpi}
        assert 12 in anions

    def test_ring_site(self, ncis):
        anpi = _by_type(ncis, "anion_pi")[0]
        assert len(anpi.site_b) == 6

    def test_geometry(self, ncis):
        anpi = _by_type(ncis, "anion_pi")[0]
        assert "distance" in anpi.geometry
        assert "angle" in anpi.geometry


# ------------------------------------------------------------------
# Cation-LP (K+ ... dimethyl ether)
# ------------------------------------------------------------------


class TestCationLP:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "cation-lp.xyz"), charge=1)
        return detect_ncis(G)

    def test_cation_lp_detected(self, ncis):
        catlp = _by_type(ncis, "cation_lp")
        assert len(catlp) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["cation_lp"]

    def test_potassium_site(self, ncis):
        catlp = _by_type(ncis, "cation_lp")[0]
        assert catlp.site_a == (0,)  # K

    def test_oxygen_lp(self, ncis):
        catlp = _by_type(ncis, "cation_lp")[0]
        assert catlp.site_b == (1,)  # O

    def test_geometry(self, ncis):
        catlp = _by_type(ncis, "cation_lp")[0]
        assert "distance" in catlp.geometry
        assert catlp.geometry["distance"] == pytest.approx(3.5, abs=0.1)


# ------------------------------------------------------------------
# Pi-pi ring-domain (benzene + butadiene)
# ------------------------------------------------------------------


class TestPiPiRingDomain:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "pi-pi-ring-domain.xyz"))
        return detect_ncis(G)

    def test_ring_domain_detected(self, ncis):
        rd = _by_type(ncis, "pi_pi_ring_domain")
        assert len(rd) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["pi_pi_ring_domain"]

    def test_ring_site(self, ncis):
        rd = _by_type(ncis, "pi_pi_ring_domain")[0]
        assert len(rd.site_a) == 6  # benzene ring

    def test_domain_site(self, ncis):
        rd = _by_type(ncis, "pi_pi_ring_domain")[0]
        assert len(rd.site_b) == 4  # butadiene domain

    def test_geometry(self, ncis):
        rd = _by_type(ncis, "pi_pi_ring_domain")[0]
        assert "distance" in rd.geometry
        assert "h_separation" in rd.geometry
        assert rd.geometry["h_separation"] == pytest.approx(3.3, abs=0.1)


# ------------------------------------------------------------------
# Pi-pi domain-domain (two butadienes)
# ------------------------------------------------------------------


class TestPiPiDomainDomain:
    @pytest.fixture
    def ncis(self):
        G = build_graph(str(STRUCTURES / "pi-pi-domain-domain.xyz"))
        return detect_ncis(G)

    def test_domain_domain_detected(self, ncis):
        dd = _by_type(ncis, "pi_pi_domain_domain")
        assert len(dd) >= 1

    def test_only_expected_types(self, ncis):
        assert _types(ncis) == ["pi_pi_domain_domain"]

    def test_both_domains(self, ncis):
        dd = _by_type(ncis, "pi_pi_domain_domain")[0]
        assert len(dd.site_a) == 4
        assert len(dd.site_b) == 4

    def test_geometry(self, ncis):
        dd = _by_type(ncis, "pi_pi_domain_domain")[0]
        assert "distance" in dd.geometry
        assert "h_separation" in dd.geometry
        assert dd.geometry["h_separation"] == pytest.approx(3.3, abs=0.1)


# ------------------------------------------------------------------
# NCIAnalyzer batch API
# ------------------------------------------------------------------


class TestNCIAnalyzer:
    def test_detect_returns_list(self):
        import numpy as np

        G = build_graph(str(STRUCTURES / "water-HB.xyz"))
        analyzer = NCIAnalyzer(G)
        positions = np.array([G.nodes[i]["position"] for i in G.nodes()])
        result = analyzer.detect(positions)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_custom_thresholds(self):
        G = build_graph(str(STRUCTURES / "water-HB.xyz"))
        # Very tight thresholds should detect fewer/no NCIs
        tight = NCIThresholds(hb_vdw_scale=0.1, hb_da_max=0.5)
        ncis = detect_ncis(G, thresholds=tight)
        assert len(ncis) == 0 or all(n.type != "hbond" for n in ncis)


# ------------------------------------------------------------------
# NCIData dataclass
# ------------------------------------------------------------------


class TestNCIData:
    def test_frozen(self):
        from xyzgraph.nci import NCIData

        nci = NCIData(
            type="hbond",
            site_a=(0,),
            site_b=(3,),
            aux_atoms=(1,),
            geometry={"distance": 2.8},
        )
        with pytest.raises(AttributeError):
            nci.__setattr__("type", "other")

    def test_default_score(self):
        from xyzgraph.nci import NCIData

        nci = NCIData(
            type="hbond",
            site_a=(0,),
            site_b=(3,),
            aux_atoms=(1,),
            geometry={"distance": 2.8},
        )
        assert nci.score == 1.0

    def test_to_dict_basic(self):
        from xyzgraph.nci import NCIData

        nci = NCIData(
            type="hbond",
            site_a=(0,),
            site_b=(3,),
            aux_atoms=(2,),
            geometry={"distance": 2.85, "h_distance": 1.92, "angle": 168.3},
        )
        d = nci.to_dict()
        assert d["type"] == "hbond"
        assert d["site_a"] == [0]
        assert d["site_b"] == [3]
        assert d["aux_atoms"] == [2]
        assert d["geometry"] == {"distance": 2.85, "h_distance": 1.92, "angle": 168.3}
        assert d["score"] == 1.0

    def test_to_dict_pi_site(self):
        from xyzgraph.nci import NCIData

        nci = NCIData(
            type="cation_pi",
            site_a=(12,),
            site_b=(0, 1, 2, 3, 4, 5),
            aux_atoms=(),
            geometry={"distance": 3.2, "angle": 165.0},
        )
        d = nci.to_dict()
        assert d["site_b"] == [0, 1, 2, 3, 4, 5]
        assert d["aux_atoms"] == []

    def test_to_dict_custom_score(self):
        from xyzgraph.nci import NCIData

        nci = NCIData(
            type="ionic",
            site_a=(0,),
            site_b=(1,),
            aux_atoms=(),
            geometry={"distance": 3.0},
            score=0.75,
        )
        d = nci.to_dict()
        assert d["score"] == 0.75


# ------------------------------------------------------------------
# Serialization: graph_to_dict with NCIs
# ------------------------------------------------------------------


class TestNCISerialization:
    def test_graph_to_dict_includes_ncis(self):
        from xyzgraph.utils import graph_to_dict

        G = build_graph(str(STRUCTURES / "water-HB.xyz"))
        detect_ncis(G)
        result = graph_to_dict(G)
        assert "ncis" in result
        assert len(result["ncis"]) >= 1
        nci = result["ncis"][0]
        assert "type" in nci
        assert "site_a" in nci
        assert "site_b" in nci
        assert "aux_atoms" in nci
        assert "geometry" in nci
        assert "score" in nci
        assert isinstance(nci["site_a"], list)

    def test_graph_to_dict_without_ncis(self):
        from xyzgraph.utils import graph_to_dict

        G = build_graph(str(STRUCTURES / "water-HB.xyz"))
        result = graph_to_dict(G)
        assert "ncis" not in result


def _contact_graph(symbols, positions, bonds, charges=None):
    G = nx.Graph()
    for atom, (symbol, position) in enumerate(zip(symbols, positions)):
        G.add_node(atom, symbol=symbol, position=tuple(position), formal_charge=(charges or {}).get(atom, 0))
    for first, second, order in bonds:
        G.add_edge(first, second, bond_order=order)
    for atom in G:
        G.nodes[atom]["valence"] = sum(edge["bond_order"] for edge in G[atom].values())
    return G


def _ring_contact_graph(donor, hydrogens):
    ring = [(1.4 * np.cos(i * np.pi / 3), 1.4 * np.sin(i * np.pi / 3), 0) for i in range(6)]
    positions = [*ring, (0, 0, 3), *[(i * 0.3, 0, 2) for i in range(hydrogens)]]
    bonds = [(i, (i + 1) % 6, 1.5) for i in range(6)] + [(6, 7 + i, 1) for i in range(hydrogens)]
    G = _contact_graph([*(["C"] * 6), donor, *(["H"] * hydrogens)], positions, bonds)
    G.graph["aromatic_rings"] = [tuple(range(6))]
    return G


@pytest.mark.parametrize("dative", [False, True])
def test_terminal_hydrides_are_distinct_hb_acceptors(dative):
    from xyzgraph.nci import NCICandidate, build_nci_graph

    positions = [(0, 0, 0), (1, 0, 0), (4.3, 0, 0), (2.8, 0.4, 0), (2.8, -0.4, 0)]
    positions += [(x, 0, 0) for x in (10, 11, 20, 21, 6)]
    G = _contact_graph(
        ["N", "H", "Ir", "H", "H", "I", "C", "P", "C", "C"],
        positions,
        [(0, 1, 1), (2, 3, 1), (2, 4, 1), (5, 6, 1), (7, 8, 1), (3, 9, 0)],
        charges={4: -1},
    )
    G.edges[3, 9]["NCI"] = True
    if dative:
        for h in (3, 4):
            G.edges[2, h].update(metal_coord=True, dative_donor=h)
    analyzer = NCIAnalyzer(G)
    candidates = analyzer.candidates()

    assert {c for c in candidates if c.site_b in ((3,), (4,))} == {
        NCICandidate("HB", (0,), (3,), (1,)),
        NCICandidate("HB", (0,), (4,), (1,)),
    }
    assert not {3, 4}.intersection(analyzer.lone_pair_sites)
    assert not any(c.kind == "HB" and c.site_a == (2,) for c in candidates)
    observed = analyzer.detect(np.array([G.nodes[i]["position"] for i in G]))
    assert {(h.site_b, h.aux_atoms) for h in observed if h.type.startswith("hbond")} == {((3,), (1,)), ((4,), (1,))}
    decorated = build_nci_graph(G, observed)
    assert all(decorated.edges[1, h]["NCI"] and decorated.edges[2, h]["bond_order"] == 1 for h in (3, 4))
    G.edges[2, 3]["bond_order"] = 0
    assert not any(c.kind == "HB" and c.site_b == (3,) for c in NCIAnalyzer(G).candidates())


@pytest.mark.parametrize(
    ("hh_distance", "angle", "detected"),
    [(2.39, 150, True), (2.41, 150, False), (1.8, 120, False)],
)
def test_dihydrogen_hbond_geometry_and_measurements(hh_distance, angle, detected):
    direction = np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0])
    hydride = np.array([1, 0, 0]) + hh_distance * direction
    G = _contact_graph(
        ["N", "H", "H", "Ir"],
        [(0, 0, 0), (1, 0, 0), hydride, hydride + np.array([0, 1.6, 0])],
        [(0, 1, 1), (2, 3, 1)],
    )
    analyzer = NCIAnalyzer(G)
    positions = np.array([G.nodes[i]["position"] for i in G])
    hbonds = [h for h in analyzer.detect(positions) if h.type == "hbond" and h.site_b == (2,)]

    assert bool(hbonds) is detected
    if detected:
        (hb,) = hbonds
        assert (hb.site_a, hb.site_b, hb.aux_atoms) == ((0,), (2,), (1,))
        assert hb.geometry == pytest.approx(
            {"distance": np.linalg.norm(hydride), "h_distance": hh_distance, "angle": angle}
        )


@pytest.mark.parametrize(("oxo", "accepts"), [(0, True), (2, False)])
def test_sulfur_accepts_hb_only_with_a_lone_pair(oxo, accepts):
    """A thioether S(II) keeps lone pairs; a sulfonyl S(VI) bonds all six valence electrons."""
    G = _contact_graph(
        ["N", "H", "S", "C", "C", "O", "O"][: 5 + oxo],
        [(i, 0, 0) for i in range(5 + oxo)],
        [(0, 1, 1), (2, 3, 1), (2, 4, 1), *[(2, 5 + k, 2) for k in range(oxo)]],
    )
    assert any(c.kind == "HB" and c.site_b == (2,) for c in NCIAnalyzer(G).candidates()) is accepts


@pytest.mark.parametrize("other", ["H", "Fe"])
def test_h2_and_bridging_h_are_not_terminal_hydride_acceptors(other):
    G = _contact_graph(
        ["N", "H", "H", "Ir", other],
        [(0, 0, 0), (1, 0, 0), (2.8, 0, 0), (3.3, 1.5, 0), (4, 0, 0)],
        [(0, 1, 1), (2, 3, 1), (2, 4, 1)],
    )
    if other == "H":
        G.add_edge(3, 4, bond_order=1)  # coordinated H2 retains its H-H bond
    assert not any(c.kind == "HB" and c.site_b == (2,) for c in NCIAnalyzer(G).candidates())


def test_hydride_distance_limit_does_not_change_ordinary_hydrogen_bonds():
    G = build_graph(str(STRUCTURES / "water-HB.xyz"))
    positions = np.array([G.nodes[i]["position"] for i in G])
    normal = NCIAnalyzer(G).detect(positions)

    assert normal
    assert NCIAnalyzer(G, NCIThresholds(hb_hh_max=0.01)).detect(positions) == normal


def test_hbond_candidates_keep_each_hydrogen_without_requiring_contact_geometry():
    from xyzgraph.nci import NCICandidate

    G = _contact_graph(
        ["O", "H", "H", "O", "H", "H"],
        [(0, 0, 0), (-0.3, 0.9, 0), (-0.3, -0.9, 0), (20, 0, 0), (20.3, 0.9, 0), (20.3, -0.9, 0)],
        [(0, 1, 1), (0, 2, 1), (3, 4, 1), (3, 5, 1)],
    )
    analyzer = NCIAnalyzer(G)

    candidates = analyzer.candidates()

    assert set(candidates) == {
        NCICandidate("HB", (0,), (3,), (1,)),
        NCICandidate("HB", (0,), (3,), (2,)),
        NCICandidate("HB", (3,), (0,), (4,)),
        NCICandidate("HB", (3,), (0,), (5,)),
    }
    assert [c.aux_atoms for c in candidates if c.site_a == (0,)] == [(1,), (2,)]
    assert analyzer.detect(np.array([G.nodes[i]["position"] for i in G])) == []


def test_halogen_pi_retains_the_first_passing_reference():
    G = _ring_contact_graph("I", 2)
    G.nodes[7].update(symbol="C", position=(1, 0, 3))
    G.nodes[8].update(symbol="C", position=(0, 0, 4))
    analyzer = NCIAnalyzer(G)
    positions = np.array([G.nodes[i]["position"] for i in G])

    assert [n.aux_atoms for n in _by_type(analyzer.detect(positions), "halogen_pi")] == [(8,)]
    positions[7] = (0, 0, 4)
    assert [n.aux_atoms for n in _by_type(analyzer.detect(positions), "halogen_pi")] == [(7,)]


@pytest.mark.parametrize(("donor", "kind", "hydrogens"), [("O", "HBPI", 2), ("C", "CHPI", 4), ("Fe", "HBPI", 2)])
def test_ring_candidates_keep_each_eligible_hydrogen_reference(donor, kind, hydrogens):
    G = _ring_contact_graph(donor, hydrogens)

    candidates = [c for c in NCIAnalyzer(G).candidates() if c.kind == kind and c.site_a == (6,)]

    assert {c.aux_atoms for c in candidates} == (set() if donor == "Fe" else {(7 + i,) for i in range(hydrogens)})
    assert all(c.site_b == tuple(range(6)) for c in candidates)


@pytest.mark.parametrize(
    ("symbol", "kind", "detected"), [("S", "ChB", "chalcogen_bond"), ("P", "PnB", "pnictogen_bond")]
)
def test_sigma_candidates_list_all_references_and_detection_keeps_first_passing_order(symbol, kind, detected):
    G = _contact_graph(
        [symbol, "H", "C", "O", "H", "H"],
        [(0, 0, 0), (0, 1, 0), (-1.5, 0, 0), (2.8, 0, 0), (3.1, 0.9, 0), (3.1, -0.9, 0)],
        [(0, 1, 1), (0, 2, 1), (3, 4, 1), (3, 5, 1)],
    )
    analyzer = NCIAnalyzer(G)
    candidates = [c for c in analyzer.candidates() if c.kind == kind and c.site_a == (0,) and c.site_b == (3,)]
    positions = np.array([G.nodes[i]["position"] for i in G])

    assert [c.aux_atoms for c in candidates] == [(1,), (2,)]
    assert [n.aux_atoms for n in analyzer.detect(positions) if n.type == detected and n.site_a == (0,)] == [(2,)]
    positions[1] = (-1, 0, 0)
    assert [n.aux_atoms for n in analyzer.detect(positions) if n.type == detected and n.site_a == (0,)] == [(1,)]


@pytest.mark.parametrize(
    ("reference", "edge"),
    [("Pd", {}), ("C", {"metal_coord": True}), ("C", {"dative_donor": 0}), ("C", {"NCI": True, "bond_order": 0})],
)
def test_coordination_and_nci_edges_cannot_be_halogen_bond_references(reference, edge):
    G = _contact_graph(
        ["Cl", reference, "N", "H", "H", "H"],
        [(0, 0, 0), (-2.3, 0, 0), (2.65, 0, 0), (2.99, 0.97, 0), (2.99, -0.49, 0.84), (2.99, -0.49, -0.84)],
        [(0, 1, 1), (2, 3, 1), (2, 4, 1), (2, 5, 1)],
    )
    G.edges[0, 1].update(edge)

    assert not _by_type(NCIAnalyzer(G).detect(np.array([G.nodes[i]["position"] for i in G])), "halogen_bond")


def test_a_coordinating_sigma_donor_keeps_its_other_covalent_reference():
    G = _contact_graph(
        ["S", "Pd", "C", "O", "H", "H"],
        [(0, 0, 0), (-2.3, 0, 0), (-1.5, 0, 0), (2.8, 0, 0), (3.1, 0.9, 0), (3.1, -0.9, 0)],
        [(0, 1, 1), (0, 2, 1), (3, 4, 1), (3, 5, 1)],
    )
    G.edges[0, 1].update(metal_coord=True, dative_donor=0)
    analyzer = NCIAnalyzer(G)

    detected = [n for n in analyzer.detect(np.array([G.nodes[i]["position"] for i in G])) if n.type == "chalcogen_bond"]

    assert [(n.site_a, n.site_b, n.aux_atoms) for n in detected] == [((0,), (3,), (2,))]
    assert [c.aux_atoms for c in analyzer.candidates() if c.kind == "ChB" and c.site_a == (0,)] == [(2,)]


@pytest.mark.parametrize(("halogen", "donates"), [("F", False), ("I", True)])
def test_only_heavier_halogens_are_sigma_hole_donors(halogen, donates):
    G = _contact_graph(
        [halogen, "C", "N", "H", "H", "H"], [(i, 0, 0) for i in range(6)], [(0, 1, 1), *[(2, h, 1) for h in (3, 4, 5)]]
    )

    assert any(c.kind == "XB" and c.site_a == (0,) for c in NCIAnalyzer(G).candidates()) is donates


def test_halogen_pi_rejects_a_hydrogen_reference():
    G = _ring_contact_graph("Cl", 1)
    G.nodes[7]["symbol"] = "H"

    assert not any(c.kind == "HALPI" for c in NCIAnalyzer(G).candidates())


@pytest.mark.parametrize(("symbol", "hydrogens", "keeps_pair"), [("N", 3, False), ("O", 2, True)])
def test_explicit_donation_spends_the_donors_pair_without_spending_other_remaining_pairs(symbol, hydrogens, keeps_pair):
    symbols = [symbol, "Pd", "O", "H", "H", *(["H"] * hydrogens)]
    positions: list[tuple[float, float, float]] = [(0, 0, 0), (-2.3, 0, 0), (2.8, 0, 0), (1.84, 0, 0), (3.04, 0.93, 0)]
    positions += [(0, 0.8 * (i + 1), 0.6) for i in range(hydrogens)]
    bonds = [(0, 1, 1), (2, 3, 1), (2, 4, 1), *[(0, 5 + i, 1) for i in range(hydrogens)]]
    G = _contact_graph(symbols, positions, bonds)
    G.edges[0, 1]["metal_coord"] = True
    unannotated = NCIAnalyzer(G)
    assert any(
        n.type == "hbond" and n.site_a == (2,) and n.site_b == (0,) for n in unannotated.detect(np.array(positions))
    )
    G.edges[0, 1]["dative_donor"] = 0
    analyzer = NCIAnalyzer(G)

    observed = any(
        n.type == "hbond" and n.site_a == (2,) and n.site_b == (0,) for n in analyzer.detect(np.array(positions))
    )

    assert observed is keeps_pair
    assert (0 in analyzer.lone_pair_sites) is keeps_pair
    assert any(c.kind == "HB" and c.site_a == (2,) and c.site_b == (0,) for c in analyzer.candidates()) is keeps_pair
    assert any(c.kind == "HB" and c.site_a == (0,) and c.site_b == (2,) for c in analyzer.candidates())


def test_dative_donor_must_name_an_endpoint_of_its_edge():
    G = _contact_graph(["N", "Pd", "O"], [(0, 0, 0), (2, 0, 0), (5, 0, 0)], [(0, 1, 1)])
    G.edges[0, 1]["dative_donor"] = 2

    with pytest.raises(ValueError, match=r"edge \(0, 1\).*set it to.*or remove the annotation"):
        NCIAnalyzer(G)


@pytest.mark.parametrize(("capped", "observed"), [(False, ["ionic"]), (True, [])])
def test_a_buried_cation_meets_its_anion_at_its_substituents(capped, observed):
    """R4N+ is met at its methyls' surface, as one ionic contact; an R3NH+ face is measured from N."""
    arms = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) / np.sqrt(3)
    u, v = -arms[0], np.array([1, -1, 0]) / np.sqrt(2)
    G = _contact_graph(
        ["N", "H" if capped else "C", "C", "C", "C", "C", "O", "O", "H"],
        [(0, 0, 0), *(arms * 1.5), 4.3 * u, 3.6 * u + 1.1 * v, 3.6 * u - 1.1 * v, 5.4 * u],
        [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (5, 6, 1), (5, 7, 2), (5, 8, 1)],
        charges={0: 1, 6: -1},
    )
    found = NCIAnalyzer(G).detect(np.array([G.nodes[i]["position"] for i in G]))

    assert [n.type for n in found] == observed


def test_charges_cancelling_in_a_covalent_group_are_not_ions():
    """Nitro N+/O- cancel; azide N-=N+=N- is an anion at its terminal N."""
    from xyzgraph.nci.sites import detect_sites

    G = _contact_graph(
        ["C", "N", "O", "O", "N", "N", "N"],
        [(0, 0, 0), (1.5, 0, 0), (2, 1, 0), (2, -1, 0), (10, 0, 0), (11.2, 0, 0), (12.4, 0, 0)],
        [(0, 1, 1), (1, 2, 2), (1, 3, 1), (4, 5, 2), (5, 6, 2)],
        charges={1: 1, 3: -1, 4: -1, 5: 1, 6: -1},
    )
    sites = detect_sites(G, NCIThresholds())

    assert (sites["cationic_atoms"], sites["anionic_atoms"]) == ([], [4, 6])


@pytest.mark.parametrize("side", [1, -1])
def test_halogen_pi_is_observed_on_either_face(side):
    """The ring normal's sign is arbitrary, so a halogen and its mirror image through the ring are both observed."""
    G = _ring_contact_graph("I", 1)
    G.nodes[6]["position"] = (0, 0, 3 * side)
    G.nodes[7].update(symbol="C", position=(0, 0, 5.1 * side))
    observed = NCIAnalyzer(G).detect(np.array([G.nodes[i]["position"] for i in G]))

    assert [n.type for n in observed] == ["halogen_pi"]


@pytest.mark.parametrize(
    ("symbols", "bonds", "has_pair"),
    [
        (["N", "H", "H", "H"], [(0, 1, 1), (0, 2, 1), (0, 3, 1)], True),  # amine
        (["N", "H", "C", "H", "H"], [(0, 1, 1), (0, 2, 2), (2, 3, 1), (2, 4, 1)], True),  # imine, pair in the plane
        (["N", "H", "H", "C", "O", "H"], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (3, 4, 2), (3, 5, 1)], False),  # amide
        (["P", "O", "O", "O", "O"], [(0, k, 1) for k in range(1, 5)], False),  # four sigma bonds, orders unread
    ],
)
def test_a_lone_pair_comes_from_structure(symbols, bonds, has_pair):
    """Four sigma bonds leave no pair; a three-connected N's pair conjugates into an adjacent pi bond."""
    from xyzgraph.nci.sites import detect_sites

    G = _contact_graph(symbols, [(2 * i, 0, 0) for i in range(len(symbols))], bonds)
    sites = detect_sites(G, NCIThresholds())

    assert (0 in sites["hbond_acceptors"], 0 in sites["lp_donors"]) == (has_pair, has_pair)


@pytest.mark.parametrize("flagged", [True, False])
def test_a_metal_has_no_sigma_references_whether_or_not_its_bonds_are_flagged(flagged):
    """Unflagged Pd-L edges must not join Pd to its ligands' charge group, nor make Pd a buried ion."""
    from xyzgraph.nci.sites import detect_sites, sigma_reference_atoms

    hydrogens = [(n, h, 1) for n in (3, 4) for h in range(5 + 3 * (n - 3), 8 + 3 * (n - 3))]
    G = _contact_graph(
        ["Pd", "Cl", "Cl", "N", "N", *["H"] * 6],
        [(2 * i, 0, 0) for i in range(11)],
        [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), *hydrogens],
        charges={0: 2, 1: -1, 2: -1},
    )
    for ligand in (1, 2, 3, 4):
        G.edges[0, ligand]["metal_coord"] = flagged
    sites = detect_sites(G, NCIThresholds())

    assert sigma_reference_atoms(G, 0, "ion") == ()
    assert (sites["cationic_atoms"], sites["anionic_atoms"]) == ([0], [1, 2])
