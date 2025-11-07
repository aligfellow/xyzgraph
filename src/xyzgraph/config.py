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
    'threshold_metal_ligand': 0.6,
    'threshold_nonmetal_nonmetal': 0.55,
    'relaxed': False,
    
    # ORCA-specific parameters:
    'orca_bond_threshold': 0.5,
}
