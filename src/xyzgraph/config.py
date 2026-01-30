"""Default configuration parameters."""

from typing import TypedDict


class DefaultParamsType(TypedDict):
    """Type definition for DEFAULT_PARAMS."""

    method: str
    charge: int
    multiplicity: int | None
    quick: bool
    optimizer: str
    max_iter: int
    edge_per_iter: int
    beam_width: int
    bond: list[tuple[int, int]] | None
    unbond: list[tuple[int, int]] | None
    clean_up: bool
    debug: bool
    threshold: float
    threshold_h_h: float
    threshold_h_nonmetal: float
    threshold_h_metal: float
    threshold_metal_ligand: float
    threshold_nonmetal_nonmetal: float
    threshold_metal_metal_self: float
    relaxed: bool
    allow_metal_metal_bonds: bool
    period_scaling_h_bonds: float
    period_scaling_nonmetal_bonds: float
    orca_bond_threshold: float


DEFAULT_PARAMS: DefaultParamsType = {
    "method": "cheminf",
    "charge": 0,
    "multiplicity": None,
    "quick": False,
    "optimizer": "beam",
    "max_iter": 50,
    "edge_per_iter": 10,
    "beam_width": 5,
    "bond": None,
    "unbond": None,
    "clean_up": True,
    "debug": False,
    "threshold": 1.0,
    # Advanced bonding thresholds:
    "threshold_h_h": 0.38,
    "threshold_h_nonmetal": 0.42,
    "threshold_h_metal": 0.45,
    "threshold_metal_ligand": 0.65,
    "threshold_nonmetal_nonmetal": 0.55,
    "threshold_metal_metal_self": 0.7,
    "relaxed": False,
    # Heavy element and metal bonding:
    "allow_metal_metal_bonds": True,
    "period_scaling_h_bonds": 0.05,
    "period_scaling_nonmetal_bonds": 0.00,
    # ORCA-specific parameters:
    "orca_bond_threshold": 0.25,
}
