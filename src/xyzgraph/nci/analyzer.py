"""NCIAnalyzer and detect_ncis convenience function."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .interaction import NCICandidate
from .sites import sigma_reference_atoms

if TYPE_CHECKING:
    import networkx as nx

    from .interaction import NCIData
    from .thresholds import NCIThresholds

logger = logging.getLogger(__name__)


def detect_ncis(G: nx.Graph, thresholds: NCIThresholds | None = None) -> list[NCIData]:
    """Detect NCIs in a single structure. Stores result in ``G.graph["ncis"]``."""
    analyzer = NCIAnalyzer(G, thresholds=thresholds)
    positions = np.array([G.nodes[i]["position"] for i in G.nodes()])
    ncis = analyzer.detect(positions)
    G.graph["ncis"] = ncis
    return ncis


class NCIAnalyzer:
    """Batch-efficient NCI detection. Build once from topology, evaluate many frames.

    Parameters
    ----------
    G : nx.Graph
        Graph produced by any ``build_graph*`` function.
    thresholds : NCIThresholds, optional
        Custom detection thresholds.
    """

    def __init__(self, G: nx.Graph, thresholds: NCIThresholds | None = None) -> None:
        from xyzgraph.data_loader import DATA

        from .detector import NCIDetector
        from .pairs import enumerate_pairs
        from .pi_systems import analyse_pi_systems
        from .sites import detect_sites
        from .thresholds import NCIThresholds as _Thr

        for first, second, edge in G.edges(data=True):
            if "dative_donor" in edge and edge["dative_donor"] not in (first, second):
                raise ValueError(
                    f"edge ({first}, {second}) has dative_donor={edge['dative_donor']!r}; "
                    f"set it to the donating endpoint {first} or {second}, or remove the annotation"
                )

        self._thr = thresholds or _Thr()
        self._graph = G
        self._vdw: dict[str, float] = DATA.vdw

        positions = np.array([G.nodes[i]["position"] for i in G.nodes()])
        symbols = [G.nodes[i]["symbol"] for i in G.nodes()]

        logger.debug("\n" + "=" * 80)
        logger.debug("NCI TOPOLOGY ANALYSIS")
        logger.debug("=" * 80)

        self._pi_rings, self._pi_domains = analyse_pi_systems(G, positions)
        self._sites = detect_sites(G, self._thr)
        self._pairs = enumerate_pairs(G, self._sites, self._pi_rings, self._pi_domains, self._thr)
        self._detector = NCIDetector(
            graph=G,
            symbols=symbols,
            pi_rings=self._pi_rings,
            pi_domains=self._pi_domains,
            vdw=self._vdw,
            thresholds=self._thr,
        )

    def detect(self, positions: np.ndarray) -> list[NCIData]:
        """Evaluate geometry for pre-enumerated pairs."""
        return self._detector.detect(positions, self._pairs)

    @property
    def lone_pair_sites(self) -> tuple[int, ...]:
        """Return atoms with an available lone pair."""
        return tuple(self._sites["lp_donors"])

    def candidates(self) -> tuple[NCICandidate, ...]:
        """Return topology candidates in native order, one per H or sigma reference."""
        candidates = []
        for kind, pairs in self._pairs.items():
            for first, second in pairs:
                aux = ()
                if kind in ("CHPI", "HBPI"):
                    donor, hydrogen = first
                    site_a, site_b, aux = (donor,), tuple(second), (hydrogen,)
                elif kind == "PIPI":
                    site_a, site_b = tuple(first), tuple(second)
                elif kind in ("CATPI", "ANPI", "HALPI"):
                    site_a, site_b = (first,), tuple(second)
                else:
                    site_a, site_b = (first,), (second,)
                if kind in ("XB", "ChB", "PnB", "HALPI"):
                    for reference in sigma_reference_atoms(self._graph, first, kind):
                        candidates.append(NCICandidate(kind, site_a, site_b, (reference,)))
                elif kind == "HB":
                    for hydrogen in self._graph.neighbors(first):
                        if self._graph.nodes[hydrogen]["symbol"] == "H":
                            candidates.append(NCICandidate(kind, site_a, site_b, (hydrogen,)))
                else:
                    candidates.append(NCICandidate(kind, site_a, site_b, aux))
        return tuple(candidates)

    def cutoff(self, candidate: NCICandidate) -> float:
        """Return the distance past which ``detect()`` never reports ``candidate``; its other tests apply within it."""
        return self._detector.cutoff(candidate.kind, candidate.site_a, candidate.site_b)
