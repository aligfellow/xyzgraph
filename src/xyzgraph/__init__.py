from importlib.metadata import version
__version__ = version("xyzgraph")
__citation__ = f"A. S. Goodfellow, xyzgraph: Molecular Graph Construction from Cartesian Coordinates, v{__version__}, 2025, https://github.com/aligfellow/xyzgraph.git."

# Eagerly load data 
from .data_loader import DATA, BOHR_TO_ANGSTROM

# Import default parameters from config
from .config import DEFAULT_PARAMS

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
