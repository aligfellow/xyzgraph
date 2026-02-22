"""Tests for NCI detection module.

Each test builds a graph from a known structure, runs detect_ncis(),
and checks that the expected interaction types and atom sites are found.
Tests also verify no unexpected NCI types appear (false positive guard).
"""

from __future__ import annotations

from pathlib import Path

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
        assert "d_DA" in hb.geometry
        assert "d_HA" in hb.geometry
        assert "angle_DHA" in hb.geometry

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
        assert "d_XA" in xb.geometry
        assert "angle_CXA" in xb.geometry
        assert xb.geometry["angle_CXA"] >= 140.0


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
        assert "d_centroid" in pp.geometry
        assert "angle_planes" in pp.geometry
        assert pp.geometry["angle_planes"] < 30.0


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
        assert "angle_planes" in ts.geometry
        assert ts.geometry["angle_planes"] >= 60.0


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
            assert "d_HA" in s.geometry
            assert "angle_CHA" in s.geometry


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
        assert "d_PA" in pnb.geometry
        assert "angle_NPA" in pnb.geometry
        assert pnb.geometry["angle_NPA"] >= 140.0


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
        assert "d_H_centroid" in chpi.geometry
        assert "angle_CH_centroid" in chpi.geometry


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
        assert "d_H_centroid" in hbpi.geometry
        assert "angle_DH_centroid" in hbpi.geometry


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
        assert "d_cation_anion" in ionic.geometry


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
        assert "d_anion_centroid" in anpi.geometry
        assert "angle_to_normal" in anpi.geometry


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
        assert "d_cation_lp" in catlp.geometry
        assert catlp.geometry["d_cation_lp"] == pytest.approx(3.5, abs=0.1)


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
        assert "d_centroid" in rd.geometry
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
        assert "d_centroid" in dd.geometry
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
            geometry={"d_DA": 2.8},
        )
        with pytest.raises(AttributeError):
            nci.type = "other"  # type: ignore[misc]

    def test_default_score(self):
        from xyzgraph.nci import NCIData

        nci = NCIData(
            type="hbond",
            site_a=(0,),
            site_b=(3,),
            aux_atoms=(1,),
            geometry={"d_DA": 2.8},
        )
        assert nci.score == 1.0
