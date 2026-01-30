"""XYZGraph package."""

from importlib.metadata import version

__version__ = version("xyzgraph")
__citation__ = (
    f"A. S. Goodfellow, xyzgraph: Molecular Graph Construction from Cartesian "
    f"Coordinates, v{__version__}, 2025, https://github.com/aligfellow/xyzgraph.git."
)

# Eagerly load data
# Utilities
from .ascii_renderer import graph_to_ascii
from .compare import compare_with_rdkit

# Import default parameters from config
from .config import DEFAULT_PARAMS
from .data_loader import BOHR_TO_ANGSTROM, DATA

# Main interfaces (imported after DEFAULT_PARAMS to avoid circular import)
from .graph_builders import (
    GraphBuilder,
    build_graph,
    build_graph_orca,
    build_graph_rdkit,
    build_graph_rdkit_tm,
)

# ORCA parser
from .orca_parser import OrcaParseError, parse_orca_output
from .utils import graph_debug_report, read_xyz_file

__all__ = [
    "BOHR_TO_ANGSTROM",
    # Data access
    "DATA",  # Access as DATA.vdw, DATA.metals, etc.
    # Configuration
    "DEFAULT_PARAMS",
    # Main interfaces
    "GraphBuilder",
    "OrcaParseError",
    "build_graph",
    "build_graph_orca",
    "build_graph_rdkit",
    "build_graph_rdkit_tm",
    "compare_with_rdkit",
    "graph_debug_report",
    # Visualization
    "graph_to_ascii",
    # ORCA support
    "parse_orca_output",
    # Utilities
    "read_xyz_file",
]
