"""
xyzgraph - Molecular graph construction from XYZ geometries
"""

# Eagerly load data singleton
from .data_loader import DATA, BOHR_TO_ANGSTROM

# Main interfaces
from .graph_builders import GraphBuilder
from .analyser import MolecularAnalyzer, analyze_molecule

# Utilities
from .ascii_renderer import graph_to_ascii
from .utils import graph_debug_report

__all__ = [
    # Main interfaces
    'MolecularAnalyzer',
    'GraphBuilder',
    'analyze_molecule',
    
    # Visualization
    'graph_to_ascii',
    'graph_debug_report',
    
    # Data access
    'DATA',                 # Access as DATA.vdw, DATA.metals, etc.
    'BOHR_TO_ANGSTROM',
]