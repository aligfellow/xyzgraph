"""Algorithm parameters for graph building.

All parameters empirically tuned on test molecules and CSD structures.
Inline docs explain what each parameter controls and typical ranges.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryThresholds:
    """Thresholds for geometric validation (used in BondValidator).

    Default values are strict mode (for stable molecules). transition() keeps partial bonds with strict
    limits; relaxed() adds permissive limits for strained transition-state geometries.
    """

    # Acute angle rejection (degrees)
    acute_threshold_metal: float = 15.0
    """Min angle at metal center. Standard coordination geometries."""

    acute_threshold_nonmetal: float = 35.0
    """Min angle at nonmetal. Allows cyclopropane (60°), rejects spurious bonds."""

    # Ring angle thresholds (degrees)
    angle_threshold_h_ring: float = 95.0
    """Min angle for H in rings. Based on tetrahedral geometry."""

    angle_threshold_base: float = 90.0
    """Largest angle of a non-H, non-metal 3-ring. Every side of a real 3-ring is a bond, so every
    angle is acute; a right or obtuse apex is a 1,3 contact (a 4-ring diagonal, an azolate C...C)."""

    transition_state: bool = False
    """Keep partial bonds: the equilibrium conventions (hydrogen-bond legs, the agostic filter, the
    metal-bond post-pass, the strict 3-ring ratio) are off. Set by transition() and relaxed()."""

    # 3-ring closure validation
    diagonal_ratio_initial: float = 0.65
    """A bond closing a 3-ring must be short against the two-bond path around it (each length over
    its vdW sum; an equilateral ring gives 0.5). Accepted outright up to this ratio; through a metal
    apex (an eta2-S2, a P-P ring) no further."""

    diagonal_ratio_max: float = 0.75
    """Upper ratio for a confident bond, interpolated by confidence from diagonal_ratio_initial."""

    diagonal_ratio_hard: float = 0.80
    """Absolute cutoff regardless of other factors."""

    # Agostic bond filtering
    strength_ratio: float = 20.0
    """Reject M-H if existing_conf/new_conf > threshold. Filters spurious agostic bonds."""

    confidence_threshold: float = 0.75
    """Only validate bonds with confidence < threshold."""

    # Collinearity
    collinearity_angle: float = 160.0
    """Angles > 160° or < 20° are collinear."""

    collinearity_dot_threshold: float = 0.9
    """Dot product threshold for parallel vectors. cos(26°) ≈ 0.9."""

    @classmethod
    def transition(cls) -> "GeometryThresholds":
        """Strict limits that keep partial bonds (a stretched cutoff, threshold > 1).

        A forming bond may close a 3-ring at an obtuse apex.
        """
        return cls(transition_state=True, angle_threshold_base=135.0)

    @classmethod
    def relaxed(cls) -> "GeometryThresholds":
        """Permissive thresholds for transition states."""
        return cls(
            acute_threshold_metal=12.0,
            acute_threshold_nonmetal=20.0,
            angle_threshold_h_ring=115.0,
            angle_threshold_base=135.0,
            transition_state=True,
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

    conjugation_weight: float = 144.0
    """Per ring that could be aromatic (planar, sp2) and is not 4n+2 under the Lewis structure."""

    protonation_weight: float = 8.0
    """Incorrect protonation states for N, O, S."""

    formal_charge_weight: float = 10.0
    """Squared formal charges: prefer neutral, and two unit charges over one double charge
    (a CO ligand as C≡O, not C2-=O). A metal-bound lone-pair donor counts its charge linearly:
    an oxo O2- or imido NR2- is the ionic convention, not a localised charge."""

    charged_atoms_weight: float = 10.0
    """Number of charged sites: charged atoms, a pair of opposite charges across a bond (the N+-O- of a
    nitro, the C-#O+ of CO) counting once, as the polar bond it writes. A metal-bound lone-pair donor is
    no site: its charge is the ionic bookkeeping of its bond to the metal."""

    charge_error_weight: float = 10.0
    """Deviation from target molecular charge."""

    electronegativity_weight: float = 2.0
    """Charge on wrong atoms (e.g., negative on C)."""

    metal_anion_bonus: float = 3.0
    """Bonus when fc<0 sits on a metal-bonded ligand atom. Tiebreaks carboxylate placement."""

    valence_error_weight: float = 5.0
    """Non-standard valences. Soft constraint."""

    geometry_weight: float = 10.0
    """Per π bond, how far its bond's distance over the vdW sum is from a single bond's 0.40: a
    shortened bond (C=C 0.35, C#O 0.30) gains, a single-length one pays, so π goes on the shorter
    bonds (the N+=C of a thiazolium over C=S+) without costing π bonds as such."""

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
    """D-block metal-ligand dative bonds. Longer than covalent."""

    threshold_sblock_ligand: float = 0.55
    """S-block metal-ligand bonds. Lower than d-block."""

    threshold_nonmetal_nonmetal: float = 0.55
    """C-C, C-N, C-O. Baseline for organic chemistry."""

    threshold_metal_metal_self: float = 0.7
    """M-M in clusters. Rare, need permissive threshold."""

    period_scaling_h_bonds: float = 0.05
    """Add per period for H-X. H-Si = 0.42+0.05, H-Ge = 0.42 + 0.10."""

    period_scaling_nonmetal_bonds: float = 0.05
    """Per-period scaling for a nonmetal pair, by its lighter atom: a bond between two heavy atoms
    (As-As, S-S, Cl-S) is long against the vdW sum, while a light partner keeps it short (so an
    S...O chalcogen contact stays an NCI)."""

    period_scaling_sblock_bonds: float = 0.05
    """Per-period scaling for s-block M-L. Heavier s-block metals bond longer."""

    allow_metal_metal_bonds: bool = True
    """Enable M-M bond detection."""
