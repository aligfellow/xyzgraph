"""
xyzgraph - Molecular graph construction from XYZ geometries
"""

from .graph_builders import (
    build_graph,
    build_graph_cheminf,
    build_graph_xtb,
    set_debug,
    get_vdw,
    get_expected_valences,
    get_valence_electrons,
    METALS,
)

from .ascii_renderer import (
    graph_to_ascii,
)

# Eager-load caches so `from xyzgraph import VDW` returns a dict (not None).
VDW = get_vdw()
VALENCES = get_expected_valences()
VALENCE_ELECTRONS = get_valence_electrons()

__all__ = [
    # Main API
    'build_graph',
    'build_graph_cheminf',
    'build_graph_xtb',
    'set_debug',
    
    # Data accessors
    'get_vdw',
    'get_expected_valences',
    'get_valence_electrons',
    'METALS',
    
    # Visualization
    'graph_to_ascii',
    'graph_debug_report',
]