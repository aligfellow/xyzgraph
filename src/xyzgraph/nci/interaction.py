"""NCI data model."""

from __future__ import annotations

from dataclasses import dataclass

NCI_TYPES = [
    "hbond",
    "hbond_bifurcated",
    "halogen_bond",
    "chalcogen_bond",
    "pnictogen_bond",
    "pi_pi_parallel",
    "pi_pi_t_shaped",
    "pi_pi_ring_domain",
    "pi_pi_domain_domain",
    "cation_pi",
    "anion_pi",
    "halogen_pi",
    "ch_pi",
    "hb_pi",
    "cation_lp",
    "ionic",
    "salt_bridge",
]


@dataclass(frozen=True)
class NCIData:
    """A single detected non-covalent interaction."""

    type: str  # one of NCI_TYPES
    site_a: tuple[int, ...]  # atom indices (single atom or group)
    site_b: tuple[int, ...]  # atom indices (single atom or group)
    aux_atoms: tuple[int, ...]  # e.g. H in D-H···A
    geometry: dict[str, float]  # measured distances/angles
    score: float = 1.0  # 1.0 = binary detected, later 0.0-1.0 decay
