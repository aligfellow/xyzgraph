# Centralized default parameters for graph building
DEFAULT_PARAMS = {
    'method': 'cheminf',
    'charge': 0,
    'multiplicity': None,
    'quick': False,
    'optimizer': 'beam',
    'max_iter': 50,
    'edge_per_iter': 10,
    'beam_width': 5,
    'bond': None,
    'unbond': None,
    'clean_up': True,
    'debug': False,
    'threshold': 1.0,
    
    # Advanced bonding thresholds:
    'threshold_h_h': 0.38,
    'threshold_h_nonmetal': 0.42,
    'threshold_h_metal': 0.48,
    'threshold_metal_ligand': 0.65,
    'threshold_nonmetal_nonmetal': 0.55,
    'threshold_metal_metal_self': 0.7,
    'relaxed': False,
    
    # Heavy element and metal bonding:
    'allow_metal_metal_bonds': True,
    'period_scaling_h_bonds': 0.05,
    'period_scaling_nonmetal_bonds': 0.00,
    
    # ORCA-specific parameters:
    'orca_bond_threshold': 0.25,
}
