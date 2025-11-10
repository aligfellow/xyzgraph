from importlib.metadata import version
__version__ = version("xyzgraph")
__citation__ = f"A. S. Goodfellow, xyzgraph: Molecular Graph Construction from Cartesian Coordinates, v{__version__}, 2025, https://github.com/aligfellow/xyzgraph.git."

# Eagerly load data 
from .data_loader import DATA, BOHR_TO_ANGSTROM

# Import default parameters from config
from .config import DEFAULT_PARAMS

# Main interfaces (imported after DEFAULT_PARAMS to avoid circular import)
from .graph_builders import GraphBuilder, build_graph, build_graph_rdkit, build_graph_rdkit_tm, build_graph_orca

# ORCA parser
from .orca_parser import parse_orca_output, OrcaParseError

# Utilities
from .ascii_renderer import graph_to_ascii
from .utils import graph_debug_report, read_xyz_file
from .compare import compare_with_rdkit

__all__ = [
    # Main interfaces
    'GraphBuilder',
    'build_graph',
    'build_graph_rdkit',
    'build_graph_rdkit_tm',
    'build_graph_orca',
    
    # ORCA support
    'parse_orca_output',
    'OrcaParseError',
    
    # Visualization
    'graph_to_ascii',
    'graph_debug_report',
    
    # Utilities
    'read_xyz_file',
    'compare_with_rdkit',

    # Configuration
    'DEFAULT_PARAMS',
    
    # Data access
    'DATA',                 # Access as DATA.vdw, DATA.metals, etc.
    'BOHR_TO_ANGSTROM',
]
