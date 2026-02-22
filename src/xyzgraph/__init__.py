"""XYZGraph package."""

from importlib.metadata import version

__version__ = version("xyzgraph")
__citation__ = (
    f"A. S. Goodfellow, xyzgraph: Molecular Graph Construction from Cartesian "
    f"Coordinates, v{__version__}, 2025, https://github.com/aligfellow/xyzgraph.git."
)

# Data access
# Comparison & visualization
from .ascii_renderer import graph_to_ascii
from .compare import compare_with_rdkit
from .data_loader import DATA

# Featurisers
from .featurisers import compute_gasteiger_charges

# Main API
from .graph_builders import build_graph
from .graph_builders_orca import build_graph_orca
from .graph_builders_rdkit import build_graph_rdkit
from .graph_builders_rdkit_tm import build_graph_rdkit_tm
from .graph_builders_xtb import build_graph_xtb

# NCI detection
from .nci import NCIAnalyzer, NCIThresholds, detect_ncis

# ORCA support
from .orca_parser import OrcaParseError, parse_orca_output
from .utils import count_frames_and_atoms, graph_debug_report, graph_to_dict, read_xyz_file

__all__ = [
    # Data access
    "DATA",
    # NCI detection
    "NCIAnalyzer",
    "NCIThresholds",
    # ORCA support
    "OrcaParseError",
    # Main API
    "build_graph",
    "build_graph_orca",
    "build_graph_rdkit",
    "build_graph_rdkit_tm",
    "build_graph_xtb",
    # Comparison & visualization
    "compare_with_rdkit",
    # Featurisers
    "compute_gasteiger_charges",
    # Utilities
    "count_frames_and_atoms",
    "detect_ncis",
    "graph_debug_report",
    "graph_to_ascii",
    "graph_to_dict",
    "parse_orca_output",
    "read_xyz_file",
]
