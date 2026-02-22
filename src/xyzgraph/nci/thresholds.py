"""Tunable geometric thresholds for NCI detection."""

from dataclasses import dataclass


@dataclass
class NCIThresholds:
    """Geometric thresholds for all NCI types.

    All distances in Angstroms, all angles in degrees.
    """

    # Pi-pi stacking (parallel)
    pii_parallel_rmax: float = 4.5  # max ring-plane separation
    pii_parallel_centroid_max: float = 5.0  # max centroid-centroid distance
    pii_parallel_angle_max: float = 30.0  # max angle between ring normals
    pii_parallel_lateral_max: float = 2.0  # max lateral displacement

    # Pi-pi stacking (T-shaped)
    pii_t_rmin: float = 4.0
    pii_t_rmax: float = 5.0
    pii_t_angle_min: float = 80.0  # angle between ring normals
    pii_t_angle_max: float = 100.0
    pii_t_hmax: float = 2.0  # distance from centroid to other ring plane
    pii_t_approach_angle_max: float = 10.0  # deviation from perpendicular approach
    require_h_for_t_shaped: bool = True

    # Cation-pi / Anion-pi
    catpi_vdw_scale: float = 0.9
    anpi_vdw_scale: float = 0.9
    catpi_axis_angle_min: float = 120.0  # angle(ring normal, cation->centroid)
    ionic_min_charge: float = 0.55
    partial_charge_threshold: float = 0.15

    # Cation-lone pair
    catlp_vdw_scale: float = 0.9

    # CH-pi
    chpi_h_plane_max: float = 3.0
    chpi_centroid_max: float = 3.5
    chpi_ch_to_centroid_angle_max: float = 30.0
    chpi_plane_alignment_min: float = 0.7
    chpi_min_pi_atoms: int = 2
    chpi_detailed_mode: bool = False

    # H-bond
    hb_da_max: float = 3.4
    hb_h_angle_min: float = 130.0
    hb_vdw_scale: float = 1.15
    # HB-pi
    hbpi_centroid_max: float = 4.0
    hbpi_dh_to_centroid_angle_min: float = 120.0
    hbpi_plane_alignment_min: float = 0.3
    hbpi_min_pi_atoms: int = 2

    # Sigma-hole bonds (halogen, chalcogen, pnictogen)
    vdw_scale: float = 0.9
    sigma_linear_min: float = 140.0
    pn_vdw_scale: float = 0.8
    # Halogen-pi
    halpi_vdw_scale: float = 0.8
    halpi_axis_angle_min: float = 120.0
    halpi_axis_angle_max: float = 180.0

    # Ionic / salt bridge
    ionic_dmax: float = 4.0
    ionic_vdw_scale: float = 0.9
    sb_vdw_scale: float = 0.8
    salt_bridge_angle_min: float = 100.0

    report_bifurcated: bool = True
