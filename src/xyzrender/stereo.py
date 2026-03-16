"""Stereochemistry labeling wrapper (R/S and E/Z) using xyzgraph."""

from __future__ import annotations

import logging

from xyzgraph.stereo import assign_ez, assign_rs

from xyzrender.annotations import AtomValueLabel, BondLabel, Annotation

logger = logging.getLogger(__name__)


def build_stereo_annotations(
    graph,
    *,
    include_rs: bool = True,
    include_ez: bool = True,
    rs_style: str = "label",
) -> list[Annotation]:
    """Generate stereochemistry labels from a molecular graph."""
    if rs_style not in {"label", "atom"}:
        raise ValueError("rs_style must be 'label' or 'atom'")

    if graph.graph.get("stereo_hint", False):
        logger.warning(
            "Input contains stereochemistry annotations; --stereo uses 3D geometry only and may disagree."
        )

    annotations: list[Annotation] = []

    if include_rs:
        rs = assign_rs(graph)
        for idx, label in rs.items():
            annotations.append(AtomValueLabel(idx, label, on_atom=(rs_style == "atom")))
    if include_ez:
        ez = assign_ez(graph)
        for (i, j), label in ez.items():
            annotations.append(BondLabel(i, j, label))
    return annotations
