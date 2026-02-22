"""NCIAnalyzer and detect_ncis convenience function."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

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
