"""Per-frame NCI geometry validation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from . import geometry as geom
from .interaction import NCIData

if TYPE_CHECKING:
    import networkx as nx

    from .thresholds import NCIThresholds

logger = logging.getLogger(__name__)


class NCIDetector:
    """Geometry-based NCI detector operating on pre-enumerated pairs.

    Parameters
    ----------
    graph : nx.Graph
        Molecular graph (used for neighbor lookups).
    symbols : list[str]
        Element symbols indexed by atom number.
    pi_rings, pi_domains : list[tuple[int, ...]]
        Pre-identified pi-systems.
    vdw : dict[str, float]
        Van der Waals radii by element symbol.
    thresholds : NCIThresholds
        Detection thresholds.
    """

    def __init__(
        self,
        graph: nx.Graph,
        symbols: list[str],
        pi_rings: list[tuple[int, ...]],
        pi_domains: list[tuple[int, ...]],
        vdw: dict[str, float],
        thresholds: NCIThresholds,
    ) -> None:
        self._G = graph
        self._sym = symbols
        self._rings = pi_rings
        self._domains = pi_domains
        self._vdw = vdw
        self._thr = thresholds

    def detect(self, pos: np.ndarray, pairs: dict[str, list]) -> list[NCIData]:
        """Run all detection methods on one frame."""
        ring_normals = self._compute_normals(pos, self._rings)
        domain_normals = self._compute_domain_normals(pos)
        pi_eff = self._compute_pi_radii(pos)

        results: list[NCIData] = []
        results.extend(self._detect_hbonds(pos, pairs["HB"]))
        results.extend(self._detect_hb_pi(pos, pairs["HBPI"], ring_normals, domain_normals))
        results.extend(self._detect_halogen_bonds(pos, pairs["XB"]))
        results.extend(self._detect_halogen_pi(pos, pairs["HALPI"], ring_normals, domain_normals, pi_eff))
        results.extend(self._detect_chalcogen_bonds(pos, pairs["ChB"]))
        results.extend(self._detect_pnictogen_bonds(pos, pairs["PnB"]))
        results.extend(self._detect_pi_stacking(pos, pairs["PIPI"], ring_normals, domain_normals))
        results.extend(self._detect_cation_pi(pos, pairs["CATPI"], ring_normals, domain_normals, pi_eff))
        results.extend(self._detect_anion_pi(pos, pairs["ANPI"], ring_normals, domain_normals, pi_eff))
        results.extend(self._detect_ch_pi(pos, pairs["CHPI"], ring_normals, domain_normals))
        results.extend(self._detect_cation_lp(pos, pairs["CATLP"]))

        ionic = list(self._detect_ionic(pos, pairs["IONIC"]))
        hbonds_for_sb = [r for r in results if r.type == "hbond"]
        salt_bridges, remove_ionic, remove_hb = self._detect_salt_bridges(pos, ionic, hbonds_for_sb)
        # Remove ionic/hbond entries that were reclassified as salt bridges
        results = [r for r in results if r not in remove_hb]
        ionic = [r for r in ionic if r not in remove_ionic]
        results.extend(ionic)
        results.extend(salt_bridges)

        if self._thr.report_bifurcated:
            results = self._detect_bifurcated(results)

        logger.debug("NCI detection complete: %d interactions found", len(results))
        return results

    # ------------------------------------------------------------------
    # H-bonds
    # ------------------------------------------------------------------

    def _detect_hbonds(self, pos: np.ndarray, hb_pairs: list) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for d, a in hb_pairs:
            D_sym, A_sym = self._sym[d], self._sym[a]
            rcut = thr.hb_vdw_scale * (self._vdw[D_sym] + self._vdw[A_sym])
            dist_da = float(np.linalg.norm(pos[d] - pos[a]))
            if dist_da > rcut:
                continue
            for n in self._G.neighbors(d):
                if self._sym[n] != "H":
                    continue
                h = n
                v_dh = pos[d] - pos[h]
                v_ha = pos[a] - pos[h]
                ang = geom.angle_deg(v_dh, v_ha)
                if ang >= thr.hb_h_angle_min:
                    dist_ha = float(np.linalg.norm(pos[h] - pos[a]))
                    results.append(
                        NCIData(
                            type="hbond",
                            site_a=(d,),
                            site_b=(a,),
                            aux_atoms=(h,),
                            geometry={"distance": dist_da, "h_distance": dist_ha, "angle": ang},
                        )
                    )
                    logger.debug("  hbond: D=%d A=%d H=%d dist=%.2f angle=%.1f", d, a, h, dist_da, ang)
        logger.debug("H-bonds: %d detected from %d pairs", len(results), len(hb_pairs))
        return results

    # ------------------------------------------------------------------
    # HB-pi
    # ------------------------------------------------------------------

    def _detect_hb_pi(self, pos: np.ndarray, hbpi_pairs: list, rn: dict, dn: dict) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for (d, h), site in hbpi_pairs:
            centroid, normal = self._pi_centroid_normal(site, pos, rn, dn)
            if normal is None:
                continue
            d_centroid = float(np.linalg.norm(pos[h] - centroid))
            if d_centroid > thr.hbpi_centroid_max:
                continue
            v_hd = pos[d] - pos[h]
            v_hc = centroid - pos[h]
            angle_dhc = geom.angle_deg(v_hd, v_hc)
            if angle_dhc < thr.hbpi_dh_to_centroid_angle_min:
                continue
            cos_align = abs(np.dot(normal, geom.unit(v_hc)))
            if cos_align < thr.hbpi_plane_alignment_min:
                continue
            results.append(
                NCIData(
                    type="hb_pi",
                    site_a=(d,),
                    site_b=tuple(site),
                    aux_atoms=(h,),
                    geometry={
                        "distance": d_centroid,
                        "angle": angle_dhc,
                        "plane_alignment": float(cos_align),
                    },
                )
            )
            logger.debug("  hb_pi: D=%d H=%d dist=%.2f angle=%.1f", d, h, d_centroid, angle_dhc)
        logger.debug("HB-pi: %d detected from %d pairs", len(results), len(hbpi_pairs))
        return results

    # ------------------------------------------------------------------
    # Sigma-hole bonds (halogen, chalcogen, pnictogen)
    # ------------------------------------------------------------------

    def _detect_halogen_bonds(self, pos: np.ndarray, xb_pairs: list) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for x, a in xb_pairs:
            X_sym, A_sym = self._sym[x], self._sym[a]
            rcut = thr.vdw_scale * (self._vdw[X_sym] + self._vdw[A_sym])
            dist = float(np.linalg.norm(pos[x] - pos[a]))
            if dist > rcut:
                continue
            for n in self._G.neighbors(x):
                if self._sym[n] == "H":
                    continue
                ang = geom.angle_deg(pos[n] - pos[x], pos[a] - pos[x])
                if ang >= thr.sigma_linear_min:
                    results.append(
                        NCIData(
                            type="halogen_bond",
                            site_a=(x,),
                            site_b=(a,),
                            aux_atoms=(n,),
                            geometry={"distance": dist, "angle": ang},
                        )
                    )
                    logger.debug("  halogen_bond: X=%d A=%d dist=%.2f angle=%.1f", x, a, dist, ang)
                    break
        logger.debug("Halogen bonds: %d detected from %d pairs", len(results), len(xb_pairs))
        return results

    def _detect_chalcogen_bonds(self, pos: np.ndarray, chb_pairs: list) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for y, a in chb_pairs:
            Y_sym, A_sym = self._sym[y], self._sym[a]
            rcut = thr.vdw_scale * (self._vdw[Y_sym] + self._vdw[A_sym])
            dist = float(np.linalg.norm(pos[y] - pos[a]))
            if dist > rcut:
                continue
            for n in self._G.neighbors(y):
                ang = geom.angle_deg(pos[n] - pos[y], pos[a] - pos[y])
                if ang >= thr.sigma_linear_min:
                    results.append(
                        NCIData(
                            type="chalcogen_bond",
                            site_a=(y,),
                            site_b=(a,),
                            aux_atoms=(n,),
                            geometry={"distance": dist, "angle": ang},
                        )
                    )
                    logger.debug("  chalcogen_bond: Y=%d A=%d dist=%.2f angle=%.1f", y, a, dist, ang)
                    break
        logger.debug("Chalcogen bonds: %d detected from %d pairs", len(results), len(chb_pairs))
        return results

    def _detect_pnictogen_bonds(self, pos: np.ndarray, pnb_pairs: list) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for p, a in pnb_pairs:
            P_sym, A_sym = self._sym[p], self._sym[a]
            rcut = thr.pn_vdw_scale * (self._vdw[P_sym] + self._vdw[A_sym])
            dist = float(np.linalg.norm(pos[p] - pos[a]))
            if dist > rcut:
                continue
            for n in self._G.neighbors(p):
                ang = geom.angle_deg(pos[n] - pos[p], pos[a] - pos[p])
                if ang >= thr.sigma_linear_min:
                    results.append(
                        NCIData(
                            type="pnictogen_bond",
                            site_a=(p,),
                            site_b=(a,),
                            aux_atoms=(n,),
                            geometry={"distance": dist, "angle": ang},
                        )
                    )
                    logger.debug("  pnictogen_bond: Pn=%d A=%d dist=%.2f angle=%.1f", p, a, dist, ang)
                    break
        logger.debug("Pnictogen bonds: %d detected from %d pairs", len(results), len(pnb_pairs))
        return results

    # ------------------------------------------------------------------
    # Halogen-pi
    # ------------------------------------------------------------------

    def _detect_halogen_pi(
        self,
        pos: np.ndarray,
        halpi_pairs: list,
        rn: dict,
        dn: dict,
        pi_eff: dict,
    ) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for x, site in halpi_pairs:
            centroid, normal = self._pi_centroid_normal(site, pos, rn, dn)
            if normal is None:
                continue
            dist = float(np.linalg.norm(pos[x] - centroid))
            rcut = thr.halpi_vdw_scale * (self._vdw[self._sym[x]] + pi_eff.get(frozenset(site), 1.7))
            if dist > rcut:
                continue
            # Check sigma-hole directionality
            ok = False
            for n in self._G.neighbors(x):
                if self._sym[n] == "H":
                    continue
                ang_xn = geom.angle_deg(pos[n] - pos[x], centroid - pos[x])
                if ang_xn >= thr.sigma_linear_min:
                    ok = True
                    break
            if not ok:
                continue
            ang = geom.angle_deg(normal, centroid - pos[x])
            if thr.halpi_axis_angle_min <= ang <= thr.halpi_axis_angle_max:
                results.append(
                    NCIData(
                        type="halogen_pi",
                        site_a=(x,),
                        site_b=tuple(site),
                        aux_atoms=(),
                        geometry={"distance": dist, "angle": ang},
                    )
                )
                logger.debug("  halogen_pi: X=%d dist=%.2f angle=%.1f", x, dist, ang)
        logger.debug("Halogen-pi: %d detected from %d pairs", len(results), len(halpi_pairs))
        return results

    # ------------------------------------------------------------------
    # Pi-pi stacking
    # ------------------------------------------------------------------

    def _detect_pi_stacking(
        self,
        pos: np.ndarray,
        pipi_pairs: list,
        rn: dict,
        dn: dict,
    ) -> list[NCIData]:
        all_results: list[NCIData] = []
        for pi1, pi2 in pipi_pairs:
            nci = self._classify_pi_pair(pi1, pi2, pos, rn, dn)
            if nci is not None:
                all_results.append(nci)

        # Fused-ring dedup: for each pair of fused groups, keep the
        # interaction with the shortest centroid-centroid distance.
        results = self._dedup_fused_pipi(all_results)
        for nci in results:
            logger.debug("  %s: dist=%.2f angle=%.1f", nci.type, nci.geometry["distance"], nci.geometry["angle"])
        logger.debug(
            "Pi-pi stacking: %d detected (%d before fused dedup) from %d pairs",
            len(results),
            len(all_results),
            len(pipi_pairs),
        )
        return results

    def _dedup_fused_pipi(self, ncis: list[NCIData]) -> list[NCIData]:
        """Keep only the best interaction per fused-group pair."""
        if not ncis:
            return ncis

        # Build fused groups from all known pi systems via union-find
        all_pi = [tuple(sorted(r)) for r in self._rings] + [tuple(sorted(d)) for d in self._domains]
        parent: dict[tuple[int, ...], tuple[int, ...]] = {s: s for s in all_pi}

        def find(x: tuple[int, ...]) -> tuple[int, ...]:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, s1 in enumerate(all_pi):
            for s2 in all_pi[i + 1 :]:
                if not set(s1).isdisjoint(set(s2)):
                    r1, r2 = find(s1), find(s2)
                    if r1 != r2:
                        parent[r2] = r1

        # For each pair of fused groups, keep shortest distance
        best: dict[tuple, NCIData] = {}
        for nci in ncis:
            ka = find(tuple(sorted(nci.site_a)))
            kb = find(tuple(sorted(nci.site_b)))
            gkey = (min(ka, kb), max(ka, kb))
            dist = nci.geometry["distance"]
            if gkey not in best or dist < best[gkey].geometry["distance"]:
                best[gkey] = nci
        return list(best.values())

    def _classify_pi_pair(
        self,
        site1: tuple[int, ...],
        site2: tuple[int, ...],
        pos: np.ndarray,
        rn: dict,
        dn: dict,
    ) -> NCIData | None:
        if len(site1) < 3 or len(site2) < 3:
            return None
        c1, n1 = self._pi_centroid_normal(site1, pos, rn, dn)
        c2, n2 = self._pi_centroid_normal(site2, pos, rn, dn)
        if n1 is None or n2 is None:
            return None

        thr = self._thr
        v12 = c2 - c1
        dcc = float(np.linalg.norm(v12))
        ang = geom.angle_deg(n1, n2)
        ang = min(ang, 180.0 - ang)

        h1 = geom.point_plane_distance(c2, c1, n1)
        h2 = geom.point_plane_distance(c1, c2, n2)
        h = 0.5 * (h1 + h2)
        v_inplane = v12 - np.dot(v12, n1) * n1
        d_lat = float(np.linalg.norm(v_inplane))

        is_ring1 = site1 in self._rings
        is_ring2 = site2 in self._rings

        # Parallel check
        if (
            h <= thr.pii_parallel_rmax
            and dcc <= thr.pii_parallel_centroid_max
            and ang <= thr.pii_parallel_angle_max
            and d_lat <= thr.pii_parallel_lateral_max
        ):
            nci_type = self._pi_subtype("parallel", is_ring1, is_ring2)
            return NCIData(
                type=nci_type,
                site_a=tuple(site1),
                site_b=tuple(site2),
                aux_atoms=(),
                geometry={"distance": dcc, "h_separation": h, "angle": ang, "lateral_disp": d_lat},
            )

        # T-shaped check
        if (
            thr.pii_t_rmin <= dcc <= thr.pii_t_rmax
            and thr.pii_t_angle_min <= ang <= thr.pii_t_angle_max
            and (h1 <= thr.pii_t_hmax or h2 <= thr.pii_t_hmax)
        ):
            approach1 = geom.angle_deg(v12, n1)
            approach2 = geom.angle_deg(-v12, n2)
            dev1, dev2 = abs(90.0 - approach1), abs(90.0 - approach2)
            if not any(d <= thr.pii_t_approach_angle_max for d in (dev1, dev2)):
                return None
            h_result = self._find_t_shaped_h(site1, site2, pos)
            if thr.require_h_for_t_shaped and h_result is None:
                return None
            nci_type = self._pi_subtype("t_shaped", is_ring1, is_ring2)
            aux = (h_result[0],) if h_result else ()
            # Order so site_a is the edge ring (with H) and site_b is the face ring
            if h_result and tuple(h_result[1]) == tuple(site1):
                edge, face = site2, site1
            else:
                edge, face = site1, site2
            return NCIData(
                type=nci_type,
                site_a=tuple(edge),
                site_b=tuple(face),
                aux_atoms=aux,
                geometry={"distance": dcc, "h_separation": h, "angle": ang},
            )
        return None

    @staticmethod
    def _pi_subtype(mode: str, is_ring1: bool, is_ring2: bool) -> str:
        if is_ring1 and is_ring2:
            return f"pi_pi_{mode}"
        if is_ring1 or is_ring2:
            return "pi_pi_ring_domain"
        return "pi_pi_domain_domain"

    def _find_t_shaped_h(
        self, site1: tuple[int, ...], site2: tuple[int, ...], pos: np.ndarray
    ) -> tuple[int, tuple[int, ...]] | None:
        """Find the H atom mediating a T-shaped pi-pi interaction.

        Returns ``(h_index, face_site)`` where *face_site* is the ring
        that the H points into, or ``None`` if no suitable H is found.
        """
        min_dist = float("inf")
        closest = None
        for a1 in site1:
            for a2 in site2:
                d = float(np.linalg.norm(pos[a1] - pos[a2]))
                if d < min_dist:
                    min_dist = d
                    closest = (a1, a2)
        if closest is None:
            return None

        for atom, other_site in [(closest[0], site2), (closest[1], site1)]:
            other_centroid = pos[list(other_site)].mean(axis=0)
            for nbr in self._G.neighbors(atom):
                if self._sym[nbr] != "H":
                    continue
                v_hc = other_centroid - pos[nbr]
                h_dist = float(np.linalg.norm(v_hc))
                if h_dist >= self._thr.pii_t_rmax:
                    continue
                v_ch = pos[nbr] - pos[atom]
                cos_a = np.dot(v_ch, v_hc) / (np.linalg.norm(v_ch) * h_dist)
                if cos_a > 0.15:
                    return (nbr, other_site)
        return None

    # ------------------------------------------------------------------
    # Ion-pi interactions
    # ------------------------------------------------------------------

    def _detect_cation_pi(
        self,
        pos: np.ndarray,
        catpi_pairs: list,
        rn: dict,
        dn: dict,
        pi_eff: dict,
    ) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for c, site in catpi_pairs:
            centroid, normal = self._pi_centroid_normal(site, pos, rn, dn)
            if normal is None:
                continue
            d = float(np.linalg.norm(pos[c] - centroid))
            rcut = thr.catpi_vdw_scale * (self._vdw[self._sym[c]] + pi_eff.get(frozenset(site), 1.7))
            if d > rcut:
                continue
            ang_raw = geom.angle_deg(normal, centroid - pos[c])
            ang = max(ang_raw, 180.0 - ang_raw)
            if ang > thr.catpi_axis_angle_min:
                results.append(
                    NCIData(
                        type="cation_pi",
                        site_a=(c,),
                        site_b=tuple(site),
                        aux_atoms=(),
                        geometry={"distance": d, "angle": ang},
                    )
                )
                logger.debug("  cation_pi: C=%d dist=%.2f angle=%.1f", c, d, ang)
        logger.debug("Cation-pi: %d detected from %d pairs", len(results), len(catpi_pairs))
        return results

    def _detect_anion_pi(
        self,
        pos: np.ndarray,
        anpi_pairs: list,
        rn: dict,
        dn: dict,
        pi_eff: dict,
    ) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for a, site in anpi_pairs:
            centroid, normal = self._pi_centroid_normal(site, pos, rn, dn)
            if normal is None:
                continue
            d = float(np.linalg.norm(pos[a] - centroid))
            rcut = thr.anpi_vdw_scale * (self._vdw[self._sym[a]] + pi_eff.get(frozenset(site), 1.7))
            if d > rcut:
                continue
            ang_raw = geom.angle_deg(normal, centroid - pos[a])
            ang = max(ang_raw, 180.0 - ang_raw)
            if ang > thr.catpi_axis_angle_min:
                results.append(
                    NCIData(
                        type="anion_pi",
                        site_a=(a,),
                        site_b=tuple(site),
                        aux_atoms=(),
                        geometry={"distance": d, "angle": ang},
                    )
                )
                logger.debug("  anion_pi: A=%d dist=%.2f angle=%.1f", a, d, ang)
        logger.debug("Anion-pi: %d detected from %d pairs", len(results), len(anpi_pairs))
        return results

    # ------------------------------------------------------------------
    # CH-pi
    # ------------------------------------------------------------------

    def _detect_ch_pi(self, pos: np.ndarray, chpi_pairs: list, rn: dict, dn: dict) -> list[NCIData]:
        thr = self._thr
        results: list[NCIData] = []
        for (c, h), site in chpi_pairs:
            centroid, normal = self._pi_centroid_normal(site, pos, rn, dn)
            if normal is None:
                continue
            d_centroid = float(np.linalg.norm(pos[h] - centroid))
            if d_centroid > thr.chpi_centroid_max:
                continue
            v_ch = pos[h] - pos[c]
            v_hc = centroid - pos[h]
            ang = geom.angle_deg(v_ch, v_hc)
            if ang > thr.chpi_ch_to_centroid_angle_max:
                continue
            cos_theta = abs(np.dot(geom.unit(centroid - pos[h]), normal))
            if cos_theta < thr.chpi_plane_alignment_min:
                continue
            results.append(
                NCIData(
                    type="ch_pi",
                    site_a=(c,),
                    site_b=tuple(site),
                    aux_atoms=(h,),
                    geometry={
                        "distance": d_centroid,
                        "angle": ang,
                        "plane_alignment": float(cos_theta),
                    },
                )
            )
            logger.debug("  ch_pi: C=%d H=%d dist=%.2f angle=%.1f", c, h, d_centroid, ang)
        logger.debug("CH-pi: %d detected from %d pairs", len(results), len(chpi_pairs))
        return results

    # ------------------------------------------------------------------
    # Cation-lone pair
    # ------------------------------------------------------------------

    def _detect_cation_lp(self, pos: np.ndarray, catlp_pairs: list) -> list[NCIData]:
        results: list[NCIData] = []
        for c, lp in catlp_pairs:
            dist = float(np.linalg.norm(pos[c] - pos[lp]))
            rcut = self._thr.catlp_vdw_scale * (self._vdw[self._sym[c]] + self._vdw[self._sym[lp]])
            if dist <= rcut:
                results.append(
                    NCIData(
                        type="cation_lp",
                        site_a=(c,),
                        site_b=(lp,),
                        aux_atoms=(),
                        geometry={"distance": dist},
                    )
                )
                logger.debug("  cation_lp: C=%d LP=%d dist=%.2f", c, lp, dist)
        logger.debug("Cation-LP: %d detected from %d pairs", len(results), len(catlp_pairs))
        return results

    # ------------------------------------------------------------------
    # Ionic / salt bridges
    # ------------------------------------------------------------------

    def _detect_ionic(self, pos: np.ndarray, ionic_pairs: list) -> list[NCIData]:
        results: list[NCIData] = []
        for c, a in ionic_pairs:
            dist = float(np.linalg.norm(pos[c] - pos[a]))
            rcut = self._thr.ionic_vdw_scale * (self._vdw[self._sym[c]] + self._vdw[self._sym[a]])
            if dist <= rcut:
                results.append(
                    NCIData(
                        type="ionic",
                        site_a=(c,),
                        site_b=(a,),
                        aux_atoms=(),
                        geometry={"distance": dist},
                    )
                )
                logger.debug("  ionic: C=%d A=%d dist=%.2f", c, a, dist)
        logger.debug("Ionic: %d detected from %d pairs", len(results), len(ionic_pairs))
        return results

    def _detect_salt_bridges(
        self,
        pos: np.ndarray,
        ionic: list[NCIData],
        hbonds: list[NCIData],
    ) -> tuple[list[NCIData], list[NCIData], list[NCIData]]:
        """Detect salt bridges (H-mediated ionic). Returns (bridges, ionic_to_remove, hbonds_to_remove)."""
        thr = self._thr
        bridges: list[NCIData] = []
        remove_ionic: list[NCIData] = []
        remove_hb: list[NCIData] = []

        for ion in ionic:
            c, a = ion.site_a[0], ion.site_b[0]
            dist_ca = float(np.linalg.norm(pos[c] - pos[a]))
            rcut = thr.sb_vdw_scale * (self._vdw[self._sym[c]] + self._vdw[self._sym[a]])
            if dist_ca > rcut:
                continue
            best_h, best_dist, best_ang = None, float("inf"), 0.0
            for n in self._G.neighbors(c):
                if self._sym[n] != "H":
                    continue
                h = n
                v_ch = pos[c] - pos[h]
                v_ha = pos[a] - pos[h]
                ang = geom.angle_deg(v_ch, v_ha)
                if ang >= thr.salt_bridge_angle_min:
                    dist_ha = float(np.linalg.norm(pos[h] - pos[a]))
                    if dist_ha < best_dist or (abs(dist_ha - best_dist) <= 0.2 and ang > best_ang):
                        best_h, best_dist, best_ang = h, dist_ha, ang

            if best_h is not None:
                bridges.append(
                    NCIData(
                        type="salt_bridge",
                        site_a=(c,),
                        site_b=(a,),
                        aux_atoms=(best_h,),
                        geometry={"distance": dist_ca, "h_distance": best_dist, "angle": best_ang},
                    )
                )
                remove_ionic.append(ion)
                # Remove corresponding hbond if it exists
                for hb in hbonds:
                    if hb.site_a == (c,) and hb.site_b == (a,) and best_h in hb.aux_atoms:
                        remove_hb.append(hb)

        return bridges, remove_ionic, remove_hb

    # ------------------------------------------------------------------
    # Bifurcated H-bonds (post-processing)
    # ------------------------------------------------------------------

    def _detect_bifurcated(self, results: list[NCIData]) -> list[NCIData]:
        """Relabel H-bonds sharing an acceptor as bifurcated."""
        hbonds = [r for r in results if r.type == "hbond"]
        other = [r for r in results if r.type != "hbond"]

        # Group by acceptor
        by_acceptor: dict[tuple[int, ...], list[NCIData]] = {}
        for hb in hbonds:
            by_acceptor.setdefault(hb.site_b, []).append(hb)

        kept: list[NCIData] = []
        for _acc, group in by_acceptor.items():
            if len(group) >= 2 and len({hb.site_a for hb in group}) >= 2:
                # Different donors to same acceptor = bifurcated
                for hb in group:
                    kept.append(
                        NCIData(
                            type="hbond_bifurcated",
                            site_a=hb.site_a,
                            site_b=hb.site_b,
                            aux_atoms=hb.aux_atoms,
                            geometry=hb.geometry,
                        )
                    )
            else:
                kept.extend(group)

        return other + kept

    # ------------------------------------------------------------------
    # Pi-system helpers
    # ------------------------------------------------------------------

    def _compute_normals(self, pos: np.ndarray, sites: list[tuple[int, ...]]) -> dict[tuple[int, ...], np.ndarray]:
        """Compute plane normals for a list of atom-index tuples."""
        normals: dict[tuple[int, ...], np.ndarray] = {}
        for site in sites:
            pts = pos[list(site)]
            normals[site] = geom.plane_normal(pts)
        return normals

    def _compute_domain_normals(self, pos: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        """Compute normals for non-ring pi-domains (may need neighbor extension for 2-atom domains)."""
        normals: dict[tuple[int, ...], np.ndarray] = {}
        for domain in self._domains:
            if len(domain) == 2:
                i, j = domain
                pts = [pos[i], pos[j]]
                for atom, other in [(i, j), (j, i)]:
                    for n in self._G.neighbors(atom):
                        if n != other:
                            pts.append(pos[n])
                            break
                if len(pts) >= 3:
                    normals[domain] = geom.plane_normal(np.array(pts))
            else:
                normals[domain] = geom.plane_normal(pos[list(domain)])
        return normals

    def _compute_pi_radii(self, pos: np.ndarray) -> dict[frozenset[int], float]:
        """Mean vdW radius for each pi-site."""
        eff: dict[frozenset[int], float] = {}
        for site in list(self._rings) + list(self._domains):
            syms = [self._sym[i] for i in site]
            eff[frozenset(site)] = float(np.mean([self._vdw.get(s, 1.7) for s in syms]))
        return eff

    def _pi_centroid_normal(
        self,
        site: tuple[int, ...],
        pos: np.ndarray,
        rn: dict[tuple[int, ...], np.ndarray],
        dn: dict[tuple[int, ...], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Get centroid and normal for a pi-site."""
        centroid = pos[list(site)].mean(axis=0)
        normal = rn.get(site)
        if normal is None:
            normal = dn.get(site)
        return centroid, normal
