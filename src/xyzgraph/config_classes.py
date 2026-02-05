"""Type-safe configuration dataclasses for graph building.

All parameters empirically tuned on test molecules and CSD structures.
Inline docs explain what each parameter controls and typical ranges.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class GeometryThresholds:
    """Thresholds for geometric validation (used in BondValidator).

    Default values are strict mode (for stable molecules).
    Use relaxed() for transition states with strained geometries.
    """

    # Acute angle rejection (degrees)
    acute_threshold_metal: float = 15.0
    """Min angle at metal center. Standard coordination geometries."""

    acute_threshold_nonmetal: float = 35.0
    """Min angle at nonmetal. Allows cyclopropane (60°), rejects spurious bonds."""

    # Ring angle thresholds (degrees)
    angle_threshold_h_ring: float = 95.0
    """Min angle for H in rings. Based on tetrahedral geometry."""

    angle_threshold_base: float = 110.0
    """Base threshold for ring closure. Z-adjusted for heavier elements if enabled."""

    apply_z_adjustment: bool = True
    """Add (avg_Z - 6) * 2.0° to thresholds for heavier elements."""

    # 4-ring diagonal validation
    diagonal_ratio_initial: float = 0.65
    """Initial diagonal/bond ratio for 4-rings. Square has ratio ~1.41."""

    diagonal_ratio_max: float = 0.75
    """Max ratio after confidence adjustment."""

    diagonal_ratio_hard: float = 0.80
    """Absolute cutoff regardless of other factors."""

    # Agostic bond filtering
    strength_ratio: float = 20.0
    """Reject M-H if existing_conf/new_conf > threshold. Filters spurious agostic bonds."""

    confidence_threshold: float = 0.75
    """Only validate bonds with confidence < threshold."""

    # Planarity and collinearity
    planarity_tolerance: float = 0.15
    """Max deviation (Å) from plane for aromatic rings."""

    collinearity_angle: float = 160.0
    """Angles > 160° or < 20° are collinear."""

    collinearity_dot_threshold: float = 0.9
    """Dot product threshold for parallel vectors. cos(26°) ≈ 0.9."""

    @classmethod
    def relaxed(cls) -> "GeometryThresholds":
        """Permissive thresholds for transition states."""
        return cls(
            acute_threshold_metal=12.0,
            acute_threshold_nonmetal=20.0,
            angle_threshold_h_ring=115.0,
            angle_threshold_base=135.0,
            apply_z_adjustment=False,
            diagonal_ratio_initial=0.75,
            diagonal_ratio_max=0.85,
            diagonal_ratio_hard=0.90,
            strength_ratio=5.0,
            confidence_threshold=0.5,
        )

    @classmethod
    def strict(cls) -> "GeometryThresholds":
        """Strict thresholds (same as default). For explicit intent."""
        return cls()


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for bond order assignment scoring.

    Lower score = better assignment. Empirically tuned on test molecules.
    """

    # Primary penalties
    violation_weight: float = 1000.0
    """Valence violations (e.g., 5-coordinate C). Highest priority."""

    conjugation_weight: float = 12.0
    """Disrupted aromatic conjugation. Tuned on benzene, naphthalene."""

    protonation_weight: float = 8.0
    """Incorrect protonation states for N, O, S."""

    formal_charge_weight: float = 10.0
    """Magnitude of formal charges. Prefer neutral."""

    charged_atoms_weight: float = 10.0
    """Number of charged atoms. Prefer localized charges."""

    charge_error_weight: float = 10.0
    """Deviation from target molecular charge."""

    electronegativity_weight: float = 2.0
    """Charge on wrong atoms (e.g., negative on C)."""

    valence_error_weight: float = 5.0
    """Non-standard valences. Soft constraint."""

    # Ring conjugation
    exocyclic_double_penalty: float = 12.0
    """Double bond outside aromatic ring disrupts aromaticity."""

    conjugation_deficit_penalty: float = 5.0
    """Non-aromatic π-electron count. Hückel rule: 4n+2."""

    # Electronegativity (Pauling scale)
    electronegativity: Dict[str, float] = field(
        default_factory=lambda: {
            "H": 2.2,
            "C": 2.5,
            "N": 3.0,
            "O": 3.5,
            "F": 4.0,
            "P": 2.2,
            "S": 2.6,
            "Cl": 3.2,
            "Br": 3.0,
            "I": 2.7,
        }
    )
    """Pauling EN. Determines which atom carries charge."""

    invalid_score: float = 1e6
    """Infinite penalty for impossible states."""


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for bond order optimization."""

    max_iter: int = 50
    """Max iterations. Most molecules converge < 30."""

    edge_per_iter: int = 10
    """Edges to modify per iteration. Trade-off speed vs quality."""

    beam_width: int = 5
    """Beam search paths. Balance between quality and cost."""

    min_bond_order: float = 1.0
    """Min bond order. Cannot delete bonds."""

    max_bond_order: float = 3.0
    """Max bond order. No quadruple bonds."""

    convergence_tolerance: float = 1e-6
    """Floating-point equality threshold."""


@dataclass(frozen=True)
class BondThresholds:
    """Distance thresholds for bond detection.

    Format: bond_detected = distance < element_threshold x (VDW_i + VDW_j) x threshold
    Tuned on CSD organic and coordination complexes.
    """

    threshold: float = 1.0
    """Global scaling factor applied to element-specific thresholds.

    1.0 = use element-specific thresholds as-is.
    > 1.0 = more permissive (detect more bonds).
    """

    threshold_h_h: float = 0.38
    """H-H bonds. Tighter than other thresholds (H is small)."""

    threshold_h_nonmetal: float = 0.42
    """H to C, N, O, S. C-H ~1.09 Å, VDW sum 2.9 Å → 0.42 x 2.9 = 1.22 Å."""

    threshold_h_metal: float = 0.45
    """H to metals. M-H longer than nonmetal-H."""

    threshold_metal_ligand: float = 0.65
    """Metal-ligand dative bonds. Longer than covalent."""

    threshold_nonmetal_nonmetal: float = 0.55
    """C-C, C-N, C-O. Baseline for organic chemistry."""

    threshold_metal_metal_self: float = 0.7
    """M-M in clusters. Rare, need permissive threshold."""

    period_scaling_h_bonds: float = 0.05
    """Add per period for H-X. H-Si = 0.42+0.05, H-Ge = 0.42 + 0.10."""

    period_scaling_nonmetal_bonds: float = 0.0
    """No scaling for nonmetal-nonmetal. VDW radii already account for size."""

    allow_metal_metal_bonds: bool = True
    """Enable M-M bond detection."""
