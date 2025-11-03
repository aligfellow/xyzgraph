"""
xyzgraph - Molecular graph construction from XYZ geometries
"""

# Eagerly load data singleton
from .data_loader import DATA, BOHR_TO_ANGSTROM

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
}

# Main interfaces (imported after DEFAULT_PARAMS to avoid circular import)
from .graph_builders import GraphBuilder, build_graph

# Utilities
from .ascii_renderer import graph_to_ascii
from .utils import graph_debug_report, read_xyz_file
from .compare import xyz2mol_compare

__all__ = [
    # Main interfaces
    'GraphBuilder',
    'build_graph',
    
    # Visualization
    'graph_to_ascii',
    'graph_debug_report',
    
    # Utilities
    'read_xyz_file',
    'xyz2mol_compare',

    # Configuration
    'DEFAULT_PARAMS',
    
    # Data access
    'DATA',                 # Access as DATA.vdw, DATA.metals, etc.
    'BOHR_TO_ANGSTROM',
]
