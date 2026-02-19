"""Molecular graph construction."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .bond_detection import BondDetector
from .bond_geometry_check import BondGeometryChecker
from .bond_order_optimizer import BondOrderOptimizer
from .config import DEFAULT_PARAMS
from .data_loader import DATA
from .geometry import GeometryCalculator
from .parameters import BondThresholds, GeometryThresholds, OptimizerConfig, ScoringWeights
from .utils import configure_debug_logging, read_xyz_file

logger = logging.getLogger(__name__)


# =============================================================================
# METADATA COMPUTATION
# =============================================================================


def compute_metadata(
    method: str,
    charge: int,
    multiplicity: Optional[int],
    quick: bool,
    optimizer: str,
    max_iter: int,
    edge_per_iter: int,
    beam_width: int,
    bond: Optional[List[Tuple[int, int]]],
    unbond: Optional[List[Tuple[int, int]]],
    clean_up: bool,
    threshold: float,
    threshold_h_h: float,
    threshold_h_nonmetal: float,
    threshold_h_metal: float,
    threshold_metal_ligand: float,
    threshold_nonmetal_nonmetal: float,
    relaxed: bool,
    allow_metal_metal_bonds: bool,
    threshold_metal_metal_self: float,
    period_scaling_h_bonds: float,
    period_scaling_nonmetal_bonds: float,
) -> Dict[str, Any]:
    """
    Compute non-default parameters for metadata.

    Returns dict of parameters that differ from defaults.
    """
    non_default = {}

    if method != DEFAULT_PARAMS["method"]:
        non_default["method"] = method
    if charge != DEFAULT_PARAMS["charge"]:
        non_default["charge"] = charge
    if multiplicity != DEFAULT_PARAMS["multiplicity"]:
        non_default["multiplicity"] = multiplicity
    if quick != DEFAULT_PARAMS["quick"]:
        non_default["quick"] = quick
    if optimizer != DEFAULT_PARAMS["optimizer"]:
        non_default["optimizer"] = optimizer
    if max_iter != DEFAULT_PARAMS["max_iter"]:
        non_default["max_iter"] = max_iter
    if edge_per_iter != DEFAULT_PARAMS["edge_per_iter"]:
        non_default["edge_per_iter"] = edge_per_iter
    if beam_width != DEFAULT_PARAMS["beam_width"]:
        non_default["beam_width"] = beam_width
    if bond != DEFAULT_PARAMS["bond"]:
        non_default["bond"] = bond
    if unbond != DEFAULT_PARAMS["unbond"]:
        non_default["unbond"] = unbond
    if clean_up != DEFAULT_PARAMS["clean_up"]:
        non_default["clean_up"] = clean_up
    if threshold != DEFAULT_PARAMS["threshold"]:
        non_default["threshold"] = threshold
    if threshold_h_h != DEFAULT_PARAMS["threshold_h_h"]:
        non_default["threshold_h_h"] = threshold_h_h
    if threshold_h_nonmetal != DEFAULT_PARAMS["threshold_h_nonmetal"]:
        non_default["threshold_h_nonmetal"] = threshold_h_nonmetal
    if threshold_h_metal != DEFAULT_PARAMS["threshold_h_metal"]:
        non_default["threshold_h_metal"] = threshold_h_metal
    if threshold_metal_ligand != DEFAULT_PARAMS["threshold_metal_ligand"]:
        non_default["threshold_metal_ligand"] = threshold_metal_ligand
    if threshold_nonmetal_nonmetal != DEFAULT_PARAMS["threshold_nonmetal_nonmetal"]:
        non_default["threshold_nonmetal_nonmetal"] = threshold_nonmetal_nonmetal
    if relaxed != DEFAULT_PARAMS["relaxed"]:
        non_default["relaxed"] = relaxed
    if allow_metal_metal_bonds != DEFAULT_PARAMS["allow_metal_metal_bonds"]:
        non_default["allow_metal_metal_bonds"] = allow_metal_metal_bonds
    if threshold_metal_metal_self != DEFAULT_PARAMS["threshold_metal_metal_self"]:
        non_default["threshold_metal_metal_self"] = threshold_metal_metal_self
    if period_scaling_h_bonds != DEFAULT_PARAMS["period_scaling_h_bonds"]:
        non_default["period_scaling_h_bonds"] = period_scaling_h_bonds
    if period_scaling_nonmetal_bonds != DEFAULT_PARAMS["period_scaling_nonmetal_bonds"]:
        non_default["period_scaling_nonmetal_bonds"] = period_scaling_nonmetal_bonds

    return non_default


# =============================================================================
# GRAPH-BASED BOND CONSTRUCTION CLASS
# =============================================================================


class GraphBuilder:
    """Molecular graph construction with integrated state management.

    atoms: List of (symbol, (x, y, z)) tuples.
    """

    def __init__(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        charge: int = DEFAULT_PARAMS["charge"],
        multiplicity: Optional[int] = DEFAULT_PARAMS["multiplicity"],
        method: str = DEFAULT_PARAMS["method"],
        quick: bool = DEFAULT_PARAMS["quick"],
        optimizer: str = DEFAULT_PARAMS["optimizer"],
        max_iter: int = DEFAULT_PARAMS["max_iter"],
        edge_per_iter: int = DEFAULT_PARAMS["edge_per_iter"],
        beam_width: int = DEFAULT_PARAMS["beam_width"],
        bond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["bond"],
        unbond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["unbond"],
        clean_up: bool = DEFAULT_PARAMS["clean_up"],
        debug: bool = DEFAULT_PARAMS["debug"],
        threshold: float = DEFAULT_PARAMS["threshold"],
        threshold_h_h: float = DEFAULT_PARAMS["threshold_h_h"],
        threshold_h_nonmetal: float = DEFAULT_PARAMS["threshold_h_nonmetal"],
        threshold_h_metal: float = DEFAULT_PARAMS["threshold_h_metal"],
        threshold_metal_ligand: float = DEFAULT_PARAMS["threshold_metal_ligand"],
        threshold_nonmetal_nonmetal: float = DEFAULT_PARAMS["threshold_nonmetal_nonmetal"],
        relaxed: bool = DEFAULT_PARAMS["relaxed"],
        allow_metal_metal_bonds: bool = DEFAULT_PARAMS["allow_metal_metal_bonds"],
        threshold_metal_metal_self: float = DEFAULT_PARAMS["threshold_metal_metal_self"],
        period_scaling_h_bonds: float = DEFAULT_PARAMS["period_scaling_h_bonds"],
        period_scaling_nonmetal_bonds: float = DEFAULT_PARAMS["period_scaling_nonmetal_bonds"],
    ):
        self.atoms = atoms  # List of (symbol, (x,y,z))
        self.charge = charge
        self.method = method
        self.optimizer = optimizer.lower()
        self.quick = quick
        self.max_iter = max_iter
        self.edge_per_iter = edge_per_iter
        self.beam_width = beam_width
        self.bond = bond
        self.unbond = unbond
        self.clean_up = clean_up

        if self.optimizer not in ("greedy", "beam"):
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Choose from: 'greedy', 'beam'")

        # Auto-detect multiplicity
        if multiplicity is None:
            total_electrons = sum(DATA.s2n[symbol] for symbol, _ in atoms) - charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2
        else:
            self.multiplicity = multiplicity

        self.threshold = threshold
        self.threshold_h_h = threshold_h_h
        self.threshold_h_nonmetal = threshold_h_nonmetal
        self.threshold_h_metal = threshold_h_metal
        self.threshold_metal_ligand = threshold_metal_ligand
        self.threshold_nonmetal_nonmetal = threshold_nonmetal_nonmetal
        self.relaxed = relaxed
        self.allow_metal_metal_bonds = allow_metal_metal_bonds
        self.threshold_metal_metal_self = threshold_metal_metal_self
        self.period_scaling_h_bonds = period_scaling_h_bonds
        self.period_scaling_nonmetal_bonds = period_scaling_nonmetal_bonds

        # Reference to global data
        self.data = DATA

        # Pre-compute atom properties from tuples
        self.symbols = [symbol for symbol, _ in self.atoms]
        self.atomic_numbers = [DATA.s2n[symbol] for symbol, _ in self.atoms]
        self.positions = [(x, y, z) for _, (x, y, z) in self.atoms]

        # State
        self.graph: Optional[nx.Graph] = None
        self.log_buffer: List[str] = []

        # Geometry calculator (stateless, shared)
        self._geometry = GeometryCalculator()

        # Geometry thresholds and bond validator
        self._geom_thresholds = GeometryThresholds.relaxed() if relaxed else GeometryThresholds.strict()
        self._bond_checker = BondGeometryChecker(
            geometry=self._geometry,
            thresholds=self._geom_thresholds,
            data=DATA,
        )

        # Bond detection
        self._bond_thresholds = BondThresholds(
            threshold=threshold,
            threshold_h_h=threshold_h_h,
            threshold_h_nonmetal=threshold_h_nonmetal,
            threshold_h_metal=threshold_h_metal,
            threshold_metal_ligand=threshold_metal_ligand,
            threshold_nonmetal_nonmetal=threshold_nonmetal_nonmetal,
            threshold_metal_metal_self=threshold_metal_metal_self,
            period_scaling_h_bonds=period_scaling_h_bonds,
            period_scaling_nonmetal_bonds=period_scaling_nonmetal_bonds,
            allow_metal_metal_bonds=allow_metal_metal_bonds,
        )
        self._bond_detector = BondDetector(
            geometry=self._geometry,
            bond_checker=self._bond_checker,
            thresholds=self._bond_thresholds,
            data=DATA,
        )

        # Bond order optimizer
        self._optimizer_config = OptimizerConfig(
            max_iter=max_iter,
            edge_per_iter=edge_per_iter,
            beam_width=beam_width,
        )
        self._optimizer = BondOrderOptimizer(
            geometry=self._geometry,
            data=DATA,
            charge=charge,
            weights=ScoringWeights(),
            config=self._optimizer_config,
        )

    def log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        indent = "  " * level
        line = f"{indent}{msg}"
        logger.debug(line)
        self.log_buffer.append(line)

    def get_log(self) -> str:
        """Get full build log as string."""
        return "\n".join(self.log_buffer)

    # =========================================================================
    # Main build method
    # =========================================================================

    def build(self) -> nx.Graph:
        """Build molecular graph using configured method."""
        mode = "QUICK" if self.quick else "FULL"
        self.log(f"\n{'=' * 80}")
        self.log(f"BUILDING GRAPH ({self.method.upper()}, {mode} MODE)")
        self.log(f"Atoms: {len(self.atoms)}, Charge: {self.charge}, Multiplicity: {self.multiplicity}")
        self.log(f"{'=' * 80}\n")

        if self.method == "cheminf":
            self.graph = self._build_cheminf()
        elif self.method == "xtb":
            self.graph = self._build_xtb()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute formula
        from .utils import compute_formula

        compute_formula(self.graph)

        # Rename useful computed data for clean output (no _ prefix)
        if "_rings" in self.graph.graph:
            self.graph.graph["rings"] = self.graph.graph.pop("_rings")
        if "_element_counts" in self.graph.graph:
            self.graph.graph["element_counts"] = self.graph.graph.pop("_element_counts")
        if "_aromatic_rings" in self.graph.graph:
            self.graph.graph["aromatic_rings"] = self.graph.graph.pop("_aromatic_rings")

        # Store build log in graph
        self.graph.graph["build_log"] = self.get_log()

        self.log(f"\n{'=' * 80}")
        self.log("GRAPH CONSTRUCTION COMPLETE")
        self.log(f"{'=' * 80}\n")

        return self.graph

    # =========================================================================
    # Cheminformatics path
    # =========================================================================

    def _build_initial_graph(self) -> nx.Graph:
        """Build initial graph with 2-phase construction.

        Delegates to BondDetector.detect() for distance-based bond detection
        with geometric validation.
        """
        G = self._bond_detector.detect(self.atoms, bond=self.bond, unbond=self.unbond)
        self.log_buffer.extend(self._bond_detector.get_log())
        return G

    # =============================================================================
    # MAIN BUILD FUNCTIONS
    # =============================================================================

    def _build_cheminf(self) -> nx.Graph:
        """Build molecular graph using cheminformatics approach."""
        if self.multiplicity is None:
            total_electrons = sum(self.atomic_numbers) - self.charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2

        # Build initial graph (with inline geometric validation)
        G = self._build_initial_graph()

        self.log(f"Initial bonds: {G.number_of_edges()}", 1)

        # Bond order optimization (delegates to BondOrderOptimizer)
        self._optimizer.log_buffer.clear()

        # Initialize Kekulé patterns for aromatic rings (gives optimizer a head start)
        self._optimizer.init_kekule(G)

        # Valence adjustment
        self._optimizer.optimize(G, mode=self.optimizer, quick=self.quick)

        # Compute formal charges BEFORE aromatic detection
        formal_charges = self._optimizer.compute_formal_charges(G)

        # Store formal charges in nodes for aromatic detection to use
        for i, fc in enumerate(formal_charges):
            G.nodes[i]["formal_charge"] = fc

        # Aromatic detection (Hückel rule) - now can use formal charges
        self._optimizer.detect_aromatic_rings(G)

        # Collect optimizer logs
        self.log_buffer.extend(self._optimizer.get_log())

        # Classify metal-ligand bonds
        self._optimizer.log_buffer.clear()
        ligand_classification = self._optimizer.classify_metal_ligands(G)
        self.log_buffer.extend(self._optimizer.get_log())
        G.graph["ligand_classification"] = ligand_classification

        # Store oxidation states on metal nodes
        for metal_idx, ox_state in ligand_classification.get("metal_ox_states", {}).items():
            G.nodes[metal_idx]["oxidation_state"] = ox_state

        # Annotate graph (charges only set by featurisers like compute_gasteiger_charges())
        for node in G.nodes():
            G.nodes[node]["formal_charge"] = formal_charges[node]
            # Split valence: organic (excludes metal bonds) and metal (coordination bonds)
            organic_val = sum(
                G.edges[node, nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(node)
                if G.nodes[nbr]["symbol"] not in self.data.metals
            )
            metal_val = sum(
                G.edges[node, nbr].get("bond_order", 1.0)
                for nbr in G.neighbors(node)
                if G.nodes[nbr]["symbol"] in self.data.metals
            )
            G.nodes[node]["valence"] = organic_val
            G.nodes[node]["metal_valence"] = metal_val

        # Add bond types
        for i, j, data in G.edges(data=True):
            data["bond_type"] = (G.nodes[i]["symbol"], G.nodes[j]["symbol"])

        G.graph["total_charge"] = self.charge
        G.graph["multiplicity"] = self.multiplicity
        G.graph["method"] = "cheminf-quick" if self.quick else "cheminf-full"

        return G

    def _build_xtb(self) -> nx.Graph:
        """Build graph using xTB — delegates to standalone module."""
        from .graph_builders_xtb import build_graph_xtb

        return build_graph_xtb(
            atoms=self.atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
            clean_up=self.clean_up,
        )


def build_graph(
    atoms: List[Tuple[str, Tuple[float, float, float]]] | str,
    charge: int = DEFAULT_PARAMS["charge"],
    multiplicity: Optional[int] = DEFAULT_PARAMS["multiplicity"],
    method: str = DEFAULT_PARAMS["method"],
    quick: bool = DEFAULT_PARAMS["quick"],
    optimizer: str = DEFAULT_PARAMS["optimizer"],
    max_iter: int = DEFAULT_PARAMS["max_iter"],
    edge_per_iter: int = DEFAULT_PARAMS["edge_per_iter"],
    beam_width: int = DEFAULT_PARAMS["beam_width"],
    bond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["bond"],
    unbond: Optional[List[Tuple[int, int]]] = DEFAULT_PARAMS["unbond"],
    clean_up: bool = DEFAULT_PARAMS["clean_up"],
    debug: bool = DEFAULT_PARAMS["debug"],
    threshold: float = DEFAULT_PARAMS["threshold"],
    threshold_h_h: float = DEFAULT_PARAMS["threshold_h_h"],
    threshold_h_nonmetal: float = DEFAULT_PARAMS["threshold_h_nonmetal"],
    threshold_h_metal: float = DEFAULT_PARAMS["threshold_h_metal"],
    threshold_metal_ligand: float = DEFAULT_PARAMS["threshold_metal_ligand"],
    threshold_nonmetal_nonmetal: float = DEFAULT_PARAMS["threshold_nonmetal_nonmetal"],
    relaxed: bool = DEFAULT_PARAMS["relaxed"],
    allow_metal_metal_bonds: bool = DEFAULT_PARAMS["allow_metal_metal_bonds"],
    threshold_metal_metal_self: float = DEFAULT_PARAMS["threshold_metal_metal_self"],
    period_scaling_h_bonds: float = DEFAULT_PARAMS["period_scaling_h_bonds"],
    period_scaling_nonmetal_bonds: float = DEFAULT_PARAMS["period_scaling_nonmetal_bonds"],
    metadata: Optional[Dict[str, Any]] = None,
) -> nx.Graph:
    """Build molecular graph using GraphBuilder.

    atoms: Either a list of (symbol, (x,y,z)) tuples, or a filepath to read.
    metadata: Pre-computed metadata dict (for CLI to avoid duplication).
    """
    # Configure logging for debug mode (API backward compat)
    if debug:
        configure_debug_logging()

    # Handle filepath input
    if isinstance(atoms, str):
        atoms = read_xyz_file(atoms)

    # Compute metadata if not provided
    if metadata is None:
        metadata = compute_metadata(
            method=method,
            charge=charge,
            multiplicity=multiplicity,
            quick=quick,
            optimizer=optimizer,
            max_iter=max_iter,
            edge_per_iter=edge_per_iter,
            beam_width=beam_width,
            bond=bond,
            unbond=unbond,
            clean_up=clean_up,
            threshold=threshold,
            threshold_h_h=threshold_h_h,
            threshold_h_nonmetal=threshold_h_nonmetal,
            threshold_h_metal=threshold_h_metal,
            threshold_metal_ligand=threshold_metal_ligand,
            threshold_nonmetal_nonmetal=threshold_nonmetal_nonmetal,
            relaxed=relaxed,
            allow_metal_metal_bonds=allow_metal_metal_bonds,
            threshold_metal_metal_self=threshold_metal_metal_self,
            period_scaling_h_bonds=period_scaling_h_bonds,
            period_scaling_nonmetal_bonds=period_scaling_nonmetal_bonds,
        )

    builder = GraphBuilder(
        atoms=atoms,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        quick=quick,
        optimizer=optimizer,
        max_iter=max_iter,
        edge_per_iter=edge_per_iter,
        beam_width=beam_width,
        bond=bond,
        unbond=unbond,
        clean_up=clean_up,
        debug=debug,
        threshold=threshold,
        threshold_h_h=threshold_h_h,
        threshold_h_nonmetal=threshold_h_nonmetal,
        threshold_h_metal=threshold_h_metal,
        threshold_metal_ligand=threshold_metal_ligand,
        threshold_nonmetal_nonmetal=threshold_nonmetal_nonmetal,
        relaxed=relaxed,
        allow_metal_metal_bonds=allow_metal_metal_bonds,
        threshold_metal_metal_self=threshold_metal_metal_self,
        period_scaling_h_bonds=period_scaling_h_bonds,
        period_scaling_nonmetal_bonds=period_scaling_nonmetal_bonds,
    )

    G = builder.build()

    # Add metadata to graph (with version/citation info)
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "parameters": metadata,
    }

    return G
