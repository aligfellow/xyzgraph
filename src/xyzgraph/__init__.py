from .graph_builders import (
    build_graph_cheminf,
    build_graph_xtb,
    build_graph,
    VDW,
    VALENCES,
    VALENCE_ELECTRONS,
    get_vdw,
    get_expected_valences,
    get_valence_electrons
)
from .compare import xyz2mol_compare
from .ascii_renderer import (
    graph_debug_report,
    graph_to_ascii,
    GraphToASCII
)

# Eager-load caches so `from xyzgraph import VDW` returns a dict (not None).
if VDW is None:
    VDW = get_vdw()
if VALENCES is None:
    VALENCES = get_expected_valences()
if VALENCE_ELECTRONS is None:
    VALENCE_ELECTRONS = get_valence_electrons()

__all__ = [
    "build_graph_cheminf",
    "build_graph_xtb",
    "build_graph",
    "xyz2mol_compare",
    "graph_debug_report",
    "graph_to_ascii",
    "GraphToASCII",
    # Data caches (lazy; may be None until a getter or builder is invoked)
    "VDW",
    "VALENCES",
    "VALENCE_ELECTRONS",
    # Accessors
    "get_vdw",
    "get_expected_valences",
    "get_valence_electrons",
]
