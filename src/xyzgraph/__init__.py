"""XYZGraph package."""

from importlib.metadata import version

__version__ = version("xyzgraph")
__citation__ = (
    f"A. S. Goodfellow, xyzgraph: Molecular Graph Construction from Cartesian "
    f"Coordinates, v{__version__}, 2025, https://github.com/aligfellow/xyzgraph.git."
)

from .ascii_renderer import graph_to_ascii
from .compare import compare_with_rdkit
from .data_loader import DATA
from .featurisers import compute_gasteiger_charges
from .graph_builders import build_graph
from .graph_builders_orca import build_graph_orca
from .graph_builders_rdkit import build_graph_rdkit
from .graph_builders_rdkit_tm import build_graph_rdkit_tm
from .graph_builders_xtb import build_graph_xtb
from .nci import NCIAnalyzer, NCIThresholds, detect_ncis
from .orca_parser import OrcaParseError, parse_orca_output
from .stereo import annotate_stereo, assign_axial, assign_ez, assign_planar, assign_rs
from .utils import count_frames_and_atoms, graph_debug_report, graph_to_dict, read_xyz_file

__all__ = [
    "annotate_stereo",
    "assign_axial",
    "assign_ez",
    "assign_planar",
    "assign_rs",
    "build_graph",
    "build_graph_orca",
    "build_graph_rdkit",
    "build_graph_rdkit_tm",
    "build_graph_xtb",
    "compare_with_rdkit",
    "compute_gasteiger_charges",
    "count_frames_and_atoms",
    "DATA",
    "detect_ncis",
    "graph_debug_report",
    "graph_to_ascii",
    "graph_to_dict",
    "NCIAnalyzer",
    "NCIThresholds",
    "OrcaParseError",
    "parse_orca_output",
    "read_xyz_file",
]
