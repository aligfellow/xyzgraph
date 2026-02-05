"""Molecular graph construction."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

from .bond_detection import BondDetector
from .bond_geometry_check import BondGeometryChecker
from .bond_order_optimizer import BondOrderOptimizer
from .config import DEFAULT_PARAMS
from .data_loader import DATA
from .geometry import GeometryCalculator
from .parameters import BondThresholds, GeometryThresholds, OptimizerConfig, ScoringWeights
from .utils import configure_debug_logging, read_xyz_file

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

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

    def _compute_gasteiger_charges(self, G: nx.Graph) -> List[float]:
        """Compute Gasteiger charges using RDKit."""
        try:
            rw = Chem.RWMol()
            for i in G.nodes():
                rw.AddAtom(Chem.Atom(G.nodes[i]["symbol"]))

            for i, j, data in G.edges(data=True):
                bo = data["bond_order"]
                if bo >= 2.5:
                    bt = Chem.BondType.TRIPLE
                elif bo >= 1.75:
                    bt = Chem.BondType.DOUBLE
                elif bo >= 1.25:
                    bt = Chem.BondType.AROMATIC
                else:
                    bt = Chem.BondType.SINGLE
                rw.AddBond(int(i), int(j), bt)

            mol = rw.GetMol()

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)

            Chem.AllChem.ComputeGasteigerCharges(mol)  # ty: ignore

            charges = []
            for atom in mol.GetAtoms():
                try:
                    c = float(atom.GetProp("_GasteigerCharge"))
                    if np.isnan(c):
                        c = 0.0
                except Exception:
                    c = 0.0
                charges.append(c)

            return charges

        except Exception as e:
            self.log(f"Gasteiger charge calculation failed: {e}", 2)
            return [0.0] * len(self.atoms)

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
        stats = self._optimizer.optimize(G, mode=self.optimizer, quick=self.quick)

        # Compute formal charges BEFORE aromatic detection
        formal_charges = self._optimizer.compute_formal_charges(G)

        # Store formal charges in nodes for aromatic detection to use
        for i, fc in enumerate(formal_charges):
            G.nodes[i]["formal_charge"] = fc

        # Aromatic detection (Hückel rule) - now can use formal charges
        self._optimizer.detect_aromatic_rings(G)

        # Collect optimizer logs
        self.log_buffer.extend(self._optimizer.get_log())

        # Compute charges (RDKit Gasteiger - stays in GraphBuilder)
        gasteiger_raw = self._compute_gasteiger_charges(G)
        raw_sum = sum(gasteiger_raw)
        delta = (self.charge - raw_sum) / len(self.atoms) if self.atoms else 0.0
        gasteiger_adj = [c + delta for c in gasteiger_raw]

        # Classify metal-ligand bonds
        self._optimizer.log_buffer.clear()
        ligand_classification = self._optimizer.classify_metal_ligands(G)
        self.log_buffer.extend(self._optimizer.get_log())
        G.graph["ligand_classification"] = ligand_classification

        # Annotate graph
        for node in G.nodes():
            G.nodes[node]["charges"] = {
                "gasteiger_raw": gasteiger_raw[node],
                "gasteiger": gasteiger_adj[node],
            }
            G.nodes[node]["formal_charge"] = formal_charges[node]
            G.nodes[node]["valence"] = BondOrderOptimizer.valence_sum(G, node)

            # Aggregate charge (add H contributions)
            agg = gasteiger_adj[node]
            for nbr in G.neighbors(node):
                if G.nodes[nbr]["symbol"] == "H":
                    agg += gasteiger_adj[nbr]
            G.nodes[node]["agg_charge"] = agg

        # Add bond types
        for i, j, data in G.edges(data=True):
            data["bond_type"] = (G.nodes[i]["symbol"], G.nodes[j]["symbol"])

        G.graph["total_charge"] = self.charge
        G.graph["multiplicity"] = self.multiplicity
        G.graph["valence_stats"] = stats
        G.graph["method"] = "cheminf-quick" if self.quick else "cheminf-full"

        return G

    def _build_xtb(self) -> nx.Graph:
        """Build graph using xTB quantum chemistry calculations."""
        if self.multiplicity is None:
            total_electrons = sum(self.atomic_numbers) - self.charge
            self.multiplicity = 1 if total_electrons % 2 == 0 else 2

        work = "xtb_tmp_local"
        basename = "xtb"
        if os.system("which xtb > /dev/null 2>&1") != 0:
            raise RuntimeError("xTB not found in PATH - install xTB or use 'cheminf' method")

        os.makedirs(work, exist_ok=True)

        # Write XYZ file natively
        xyz_path = os.path.join(work, f"{basename}.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(self.atoms)}\n")
            f.write("xyzgraph generated XYZ for xTB\n")
            for symbol, (x, y, z) in self.atoms:
                f.write(f"{symbol:>2} {x:15.8f} {y:15.8f} {z:15.8f}\n")

        cmd = (
            f"cd {work} && xtb {basename}.xyz --chrg {self.charge} --uhf {self.multiplicity - 1} --gfn2 "
            f"> {basename}.out"
        )
        ret = os.system(cmd)

        if ret != 0:
            self.log(f"Warning: xTB returned non-zero exit code {ret}", 1)

        # Parse WBO
        bonds = []
        bond_orders = []
        wbo_file = os.path.join(work, f"{basename}_wbo")
        if not os.path.exists(wbo_file) and os.path.exists(os.path.join(work, "wbo")):
            os.rename(os.path.join(work, "wbo"), wbo_file)

        try:
            with open(wbo_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 3 and float(parts[2]) > 0.5:  # bonding threshold
                        bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))  # xTB uses 1-indexed
                        bond_orders.append(float(parts[2]))
            self.log(f"Parsed {len(bonds)} bonds from xTB WBO", 1)
        except FileNotFoundError:
            pass

        # Parse charges
        charges = []
        charges_file = os.path.join(work, f"{basename}_charges")
        if not os.path.exists(charges_file) and os.path.exists(os.path.join(work, "charges")):
            os.rename(os.path.join(work, "charges"), charges_file)

        try:
            with open(charges_file) as f:
                for line in f:
                    charges.append(float(line.split()[0]))
            self.log(f"Parsed {len(charges)} Mulliken charges from xTB", 1)
        except FileNotFoundError:
            charges = [0.0] * len(self.atoms)

        if self.clean_up:
            try:
                for f in os.listdir(work):
                    os.remove(os.path.join(work, f))
                os.rmdir(work)
            except Exception as e:
                self.log(f"Warning: Could not clean up temp files: {e}", 1)

        # Build graph
        G = nx.Graph()
        pos = self.positions  # Use pre-computed positions

        for i, (symbol, _) in enumerate(self.atoms):
            G.add_node(
                i,
                symbol=symbol,
                atomic_number=self.atomic_numbers[i],
                position=pos[i],
                charges={"mulliken": charges[i] if i < len(charges) else 0.0},
            )

        if bonds:
            for (i, j), bo in zip(bonds, bond_orders):
                d = GeometryCalculator.distance(tuple(pos[i]), tuple(pos[j]))
                si, sj = G.nodes[i]["symbol"], G.nodes[j]["symbol"]
                G.add_edge(
                    i,
                    j,
                    bond_order=float(bo),
                    distance=d,
                    bond_type=(si, sj),
                    metal_coord=(si in DATA.metals or sj in DATA.metals),
                )
            self.log(f"Built graph with {G.number_of_edges()} bonds from xTB", 1)
        else:
            # Fallback to distance-based if xTB failed
            self.log(
                "Warning: No xTB bonds found, falling back to distance-based, try using `--method cheminf`",
                1,
            )
            G = self._build_initial_graph()

        # Add derived properties
        for node in G.nodes():
            G.nodes[node]["valence"] = BondOrderOptimizer.valence_sum(G, node)
            agg = G.nodes[node]["charges"].get("mulliken", 0.0)
            for nbr in G.neighbors(node):
                if G.nodes[nbr]["symbol"] == "H":
                    agg += G.nodes[nbr]["charges"].get("mulliken", 0.0)
            G.nodes[node]["agg_charge"] = agg

        G.graph["total_charge"] = self.charge
        G.graph["multiplicity"] = self.multiplicity
        G.graph["method"] = "xtb"

        return G


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


def build_graph_rdkit(
    xyz_file: str | List[Tuple[str, Tuple[float, float, float]]],
    charge: int = DEFAULT_PARAMS["charge"],
    bohr_units: bool = False,
) -> nx.Graph:
    """
    Build molecular graph using RDKit's DetermineBonds algorithm.

    Uses RDKit's distance-based bond perception with Hueckel rule for conjugation.

    Parameters
    ----------
    xyz_file : str or List[Tuple[str, Tuple[float, float, float]]]
        Either path to XYZ file or list of (symbol, (x, y, z)) tuples
    charge : int, default=0
        Total molecular charge
    bohr_units : bool, default=False
        Whether coordinates are in Bohr (only used if xyz_file is a path)

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing:
        - symbol: element symbol
        - atomic_number: atomic number
        - position: (x, y, z) coordinates
        - charges: empty dict (RDKit doesn't compute partial charges)
        - formal_charge: RDKit formal charge
        - valence: sum of bond orders

    Raises
    ------
    ValueError
        If RDKit fails to parse the structure or determine bonds

    Examples
    --------
    >>> from xyzgraph import build_graph_rdkit
    >>> G = build_graph_rdkit("structure.xyz", charge=-1)
    >>> print(f"Graph has {G.number_of_nodes()} atoms and {G.number_of_edges()} bonds")

    Notes
    -----
    RDKit has limited support for coordination complexes. For metal-containing
    systems, consider using build_graph() with method='cheminf' or
    build_graph_from_orca() instead.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    # Handle input
    if isinstance(xyz_file, str):
        atoms = read_xyz_file(xyz_file, bohr_units=bohr_units)
    else:
        atoms = xyz_file

    # Build XYZ block for RDKit
    nat = len(atoms)
    symbols = [symbol for symbol, _ in atoms]
    positions = [pos for _, pos in atoms]
    xyz_lines = [str(nat), f"Generated by xyzgraph build_graph_rdkit (charge={charge})"]
    for sym, (x, y, z) in zip(symbols, positions):
        xyz_lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    xyz_block = "\n".join(xyz_lines) + "\n"

    # Parse with RDKit
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        raise ValueError("RDKit MolFromXYZBlock failed to parse structure")

    # Determine bonds
    mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge, useHueckel=True)
    except Exception as e:
        # Check for metals
        if any(s in DATA.metals for s in symbols):
            raise ValueError(f"RDKit DetermineBonds failed (metal atoms detected): {e}") from e
        raise ValueError(f"RDKit DetermineBonds failed: {e}") from e

    if mol.GetNumBonds() == 0:
        raise ValueError("RDKit DetermineBonds produced no bonds")

    # Light sanitize
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass

    # Build NetworkX graph
    G = nx.Graph()

    # Add nodes
    for a in mol.GetAtoms():
        i = a.GetIdx()
        symbol = a.GetSymbol()
        atomic_number = DATA.s2n.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

        G.add_node(
            i,
            symbol=symbol,
            atomic_number=atomic_number,
            position=positions[i],
            charges={},  # RDKit doesn't compute partial charges
            formal_charge=a.GetFormalCharge(),
            valence=0.0,
        )

    # Add edges
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()

        # Convert RDKit bond type to numeric order
        if b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC:
            bo = 1.5
        elif b.GetBondType() == Chem.BondType.SINGLE:
            bo = 1.0
        elif b.GetBondType() == Chem.BondType.DOUBLE:
            bo = 2.0
        elif b.GetBondType() == Chem.BondType.TRIPLE:
            bo = 3.0
        else:
            bo = 1.0

        # Calculate distance
        pos_i = np.array(positions[i])
        pos_j = np.array(positions[j])
        distance = float(np.linalg.norm(pos_i - pos_j))

        si = G.nodes[i]["symbol"]
        sj = G.nodes[j]["symbol"]

        G.add_edge(
            i,
            j,
            bond_order=bo,
            distance=distance,
            bond_type=(si, sj),
            metal_coord=(si in DATA.metals or sj in DATA.metals),
        )

    # Compute valence
    for node in G.nodes():
        valence = sum(
            G[node][nbr].get("bond_order", 1.0)
            for nbr in G.neighbors(node)
            if G.nodes[nbr]["symbol"] not in DATA.metals
        )
        G.nodes[node]["valence"] = valence

        # Aggregated charge (just formal for RDKit, include H neighbors)
        agg_charge = float(G.nodes[node]["formal_charge"])
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg_charge += G.nodes[nbr]["formal_charge"]
        G.nodes[node]["agg_charge"] = agg_charge

    # Add metadata
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "rdkit",
    }
    G.graph["total_charge"] = charge
    G.graph["method"] = "rdkit"

    return G


def build_graph_rdkit_tm(
    xyz_file: str | list[tuple[str, tuple[float, float, float]]],
    charge: int = DEFAULT_PARAMS["charge"],
    bohr_units: bool = False,
) -> nx.Graph:
    """
    Build molecular graph using xyz2mol_tm.get_tmc_mol (tmQM/coordination complexes).

    This function combines:
    1. XYZ coordinates for atomic positions
    2. Connectivity from xyz2mol_tm (specialized for metal coordination)
    3. Graph matching to align RDKit atom ordering with XYZ ordering

    Strategy for mismatched connectivity:
    - Attempts perfect isomorphism first
    - Falls back to partial matching if graphs differ
    - Uses element + connectivity similarity to find best correspondence
    - Requires sufficient overlap (>75% of edges) to proceed

    Parameters
    ----------
    xyz_file : str or list of (symbol, (x, y, z)) tuples
        Path to an XYZ file or coordinates.
    charge : int
        Total molecular charge.
    bohr_units : bool
        Whether input coordinates are in Bohr (converted to Å if True).

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing symbol, atomic_number, position,
        formal_charge, valence, and charges (empty dict).
    """
    import tempfile

    import networkx as nx
    import numpy as np
    from networkx.algorithms import isomorphism
    from rdkit import Chem

    from . import BOHR_TO_ANGSTROM, DATA

    # Import xyz2mol_tm
    try:
        from xyz2mol_tm import xyz2mol_tmc  # ty: ignore
    except ImportError:
        raise ImportError(
            "xyz2mol_tm not found. Install via:\npip install git+https://github.com/jensengroup/xyz2mol_tm.git"
        ) from None

    # ===== STEP 1: Parse XYZ coordinates =====
    if isinstance(xyz_file, str):
        with open(xyz_file, "r") as f:
            lines = f.readlines()
        nat = int(lines[0].strip())
        lines[1].strip()
        atoms = []
        for line in lines[2 : 2 + nat]:
            parts = line.split()
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            if bohr_units:
                x, y, z = (
                    x * BOHR_TO_ANGSTROM,
                    y * BOHR_TO_ANGSTROM,
                    z * BOHR_TO_ANGSTROM,
                )
            atoms.append((sym, (x, y, z)))
    elif isinstance(xyz_file, list):
        atoms = xyz_file
        if bohr_units:
            atoms = [(s, (x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM, z * BOHR_TO_ANGSTROM)) for s, (x, y, z) in atoms]
    else:
        raise TypeError("xyz_file must be a path or list of (symbol, position) tuples")

    heavy_idx = [i for i, (s, _) in enumerate(atoms) if s != "H"]

    # ===== STEP 2: Get connectivity from xyz2mol_tm =====
    # xyz2mol_tm reads XYZ only from file, so create temp file
    xyz_lines = [str(len(atoms)), "Generated by build_graph_rdkit_tm"]
    xyz_lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in atoms]
    xyz_block = "\n".join(xyz_lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xyz", delete=False) as tmp:
        tmp.write(xyz_block)
        tmp.flush()
        # --- timeout protection around xyz2mol_tmc ---
        import signal

        def handler(signum, frame):
            raise TimeoutError("xyz2mol_tmc took too long")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)  # 5 seconds timeout

        try:
            mol = xyz2mol_tmc.get_tmc_mol(tmp.name, overall_charge=charge)
        except TimeoutError:
            logger.warning("xyz2mol_tmc timed out for %s. Skipping RDKit-TM graph.", xyz_file)
            mol = None  # gracefully skip
        except Exception as e:
            logger.warning("xyz2mol_tmc failed for %s: %s", xyz_file, e)
            mol = None
        finally:
            signal.alarm(0)

    if mol is None:
        # Return a placeholder graph or skip
        import networkx as nx

        G = nx.Graph()
        G.graph["metadata"] = {
            "source": "rdkit_tm",
            "note": "xyz2mol_tmc failed or timed out",
        }
        return G

    # Build RDKit connectivity graph (element + bonds only)
    G_rdkit = nx.Graph()
    for i in range(mol.GetNumAtoms()):
        G_rdkit.add_node(i, symbol=mol.GetAtomWithIdx(i).GetSymbol())

    for bond in mol.GetBonds():
        G_rdkit.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # ===== STEP 3: Build XYZ heavy-atom graph =====
    G_xyz_heavy = build_graph([atoms[i] for i in heavy_idx], charge=charge, quick=True)
    mapping_to_original = {i: heavy_idx[i] for i in range(len(heavy_idx))}
    G_xyz_relabeled = nx.relabel_nodes(G_xyz_heavy, mapping_to_original)

    # ===== STEP 3a: Filter XYZ edges to match RDKit connectivity =====
    allowed_pairs = set()
    for bond in mol.GetBonds():
        sym_i = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
        sym_j = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
        allowed_pairs.add(frozenset([sym_i, sym_j]))

    edges_to_keep = [
        (i, j)
        for i, j in G_xyz_relabeled.edges()
        if frozenset([G_xyz_relabeled.nodes[i]["symbol"], G_xyz_relabeled.nodes[j]["symbol"]]) in allowed_pairs
    ]

    G_xyz_simple = nx.Graph()
    for n in G_xyz_relabeled.nodes():
        G_xyz_simple.add_node(n, symbol=G_xyz_relabeled.nodes[n]["symbol"])
    G_xyz_simple.add_edges_from(edges_to_keep)

    # ===== STEP 4: Match graphs (try perfect first, fall back to partial) =====
    nm = isomorphism.categorical_node_match("symbol", "")
    GM = isomorphism.GraphMatcher(G_rdkit, G_xyz_simple, node_match=nm)

    if GM.is_isomorphic():
        rdkit_to_xyz = GM.mapping
        print("Indexed against xyzgraph by perfect isomorphism.")
    else:
        # Graphs differ - use partial matching
        print("Warning: Graphs not perfectly isomorphic.")
        print(f"  RDKit: {G_rdkit.number_of_nodes()} nodes, {G_rdkit.number_of_edges()} edges")
        print(f"  XYZ:   {G_xyz_simple.number_of_nodes()} nodes, {G_xyz_simple.number_of_edges()} edges")
        print("  Attempting partial matching based on connectivity similarity...")

        rdkit_to_xyz = _partial_graph_matching(G_rdkit, G_xyz_simple)

        # Validate mapping quality
        mapped_edges = 0
        total_rdkit_edges = G_rdkit.number_of_edges()
        for i, j in G_rdkit.edges():
            xyz_i = rdkit_to_xyz.get(i)
            xyz_j = rdkit_to_xyz.get(j)
            if xyz_i and xyz_j and G_xyz_simple.has_edge(xyz_i, xyz_j):
                mapped_edges += 1

        overlap = mapped_edges / total_rdkit_edges if total_rdkit_edges > 0 else 0
        print(f"  Mapping quality: {mapped_edges}/{total_rdkit_edges} edges match ({overlap * 100:.1f}%)")

        if overlap < 0.75:
            raise ValueError(
                f"Insufficient graph overlap ({overlap * 100:.1f}%). "
                f"xyz2mol_tm and geometric methods disagree too much on connectivity."
            )

    # ===== STEP 5: Build final graph with XYZ ordering =====
    G = nx.Graph()
    # Add all atoms (heavy + H) with original XYZ indices
    for idx, (sym, pos) in enumerate(atoms):
        G.add_node(
            idx,
            symbol=sym,
            atomic_number=Chem.GetPeriodicTable().GetAtomicNumber(sym),
            position=pos,
            formal_charge=0,
            valence=0.0,
            charges={},
        )

    # Add heavy-heavy edges from RDKit, mapped to XYZ indices
    for bond in mol.GetBonds():
        i_xyz = rdkit_to_xyz[bond.GetBeginAtomIdx()]
        j_xyz = rdkit_to_xyz[bond.GetEndAtomIdx()]

        bt = bond.GetBondType()
        # Extract bond order from RDKit
        bo = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5,
        }.get(bt, 1.0)

        # Calculate distance from XYZ coordinates
        pos_i = np.array(G.nodes[i_xyz]["position"])
        pos_j = np.array(G.nodes[j_xyz]["position"])

        G.add_edge(
            i_xyz,
            j_xyz,
            bond_order=bo,
            distance=float(np.linalg.norm(pos_i - pos_j)),
            bond_type=(G.nodes[i_xyz]["symbol"], G.nodes[j_xyz]["symbol"]),
            metal_coord=(G.nodes[i_xyz]["symbol"] in DATA.metals or G.nodes[j_xyz]["symbol"] in DATA.metals),
        )

    # Connect hydrogens to nearest heavy atom (geometrically)
    for idx, (sym, pos) in enumerate(atoms):
        if sym == "H":
            pos_arr = np.array(pos)
            dists = [np.linalg.norm(pos_arr - np.array(G.nodes[i]["position"])) for i in heavy_idx]
            nearest = heavy_idx[int(np.argmin(dists))]
            G.add_edge(
                idx,
                nearest,
                bond_order=1.0,
                distance=float(min(dists)),
                bond_type=("H", G.nodes[nearest]["symbol"]),
                metal_coord=(G.nodes[nearest]["symbol"] in DATA.metals),
            )

    # --- Update valences and formal charges ---
    for node in G.nodes():
        G.nodes[node]["valence"] = sum(G.edges[node, nbr]["bond_order"] for nbr in G.neighbors(node))

    for rdkit_idx, xyz_idx in rdkit_to_xyz.items():
        G.nodes[xyz_idx]["formal_charge"] = mol.GetAtomWithIdx(rdkit_idx).GetFormalCharge()

    # --- Metadata ---
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "rdkit_tm",
    }
    G.graph["total_charge"] = charge
    G.graph["method"] = "rdkit_tm"

    return G


def _partial_graph_matching(G_rdkit: nx.Graph, G_xyz: nx.Graph) -> dict:
    """
    Graph-distance + neighbor-symbol similarity based partial matching for non-isomorphic graphs.

    Parameters
    ----------
    G_rdkit : nx.Graph
        RDKit molecular graph (nodes with 'symbol')
    G_xyz : nx.Graph
        XYZ-based molecular graph (nodes with 'symbol')

    Returns
    -------
    dict
        Mapping {rdkit_node -> xyz_node}
    """
    from collections import defaultdict

    import networkx as nx
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment  # ty: ignore
    except ImportError:
        raise ImportError("scipy not found. Install via:\npip install scipy") from None

    print("  Starting graph-distance + neighbor-symbol partial matching...")

    # Group nodes by element
    rdkit_by_elem = defaultdict(list)
    xyz_by_elem = defaultdict(list)
    for n in G_rdkit.nodes():
        rdkit_by_elem[G_rdkit.nodes[n]["symbol"]].append(n)
    for n in G_xyz.nodes():
        xyz_by_elem[G_xyz.nodes[n]["symbol"]].append(n)

    # Check element counts
    for elem in rdkit_by_elem:
        rdkit_count = len(rdkit_by_elem[elem])
        xyz_count = len(xyz_by_elem.get(elem, []))
        if rdkit_count != xyz_count:
            raise ValueError(
                f"Cannot perform partial matching: element '{elem}' count mismatch. RDKit has {rdkit_count}, "
                f"XYZ has {xyz_count}. This could be bimetallic and not handled by xyz2mol_tm."
            )

    # Compute shortest-path distance matrices
    print("   Computing all-pairs shortest-path distance matrices...")
    D_rdkit = np.asarray(nx.floyd_warshall_numpy(G_rdkit))
    D_xyz = np.asarray(nx.floyd_warshall_numpy(G_xyz))

    rdkit_nodes = list(G_rdkit.nodes())
    xyz_nodes = list(G_xyz.nodes())
    rdkit_index = {n: i for i, n in enumerate(rdkit_nodes)}
    xyz_index = {n: i for i, n in enumerate(xyz_nodes)}

    rdkit_to_xyz = {}

    # Match nodes per element
    for elem, rdkit_list in rdkit_by_elem.items():
        if elem not in xyz_by_elem:
            raise ValueError(f"Element {elem} in RDKit but not in XYZ")
        xyz_list = xyz_by_elem[elem]

        n_r, n_x = len(rdkit_list), len(xyz_list)
        min_count = min(n_r, n_x)
        if n_r != n_x:
            print(f"   Warning: Element {elem} count mismatch: RDKit={n_r}, XYZ={n_x}. Matching {min_count} atoms.")

        # Build score matrix
        scores = np.zeros((n_r, n_x))
        for i, r_node in enumerate(rdkit_list):
            d_r = D_rdkit[rdkit_index[r_node], :]
            r_neighs = set(G_rdkit.neighbors(r_node))
            r_symbols = {G_rdkit.nodes[n]["symbol"] for n in r_neighs}

            for j, x_node in enumerate(xyz_list):
                d_x = D_xyz[xyz_index[x_node], :]
                x_neighs = set(G_xyz.neighbors(x_node))
                x_symbols = {G_xyz.nodes[n]["symbol"] for n in x_neighs}

                # 1) Graph-distance similarity
                dist_diff = np.sum(np.abs(d_r - d_x))
                score = -dist_diff  # negative = more similar

                # 2) Neighbor symbol overlap bonus
                common_symbols = len(r_symbols & x_symbols)
                score += common_symbols * 5  # adjust weight if needed

                scores[i, j] = score

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-scores)
        for i, j in zip(row_ind[:min_count], col_ind[:min_count]):
            r_node = rdkit_list[i]
            x_node = xyz_list[j]
            score = scores[i, j]
            rdkit_to_xyz[r_node] = x_node
            print(f"   Matched {r_node} → {x_node} (score={score:.2f})")

    print(f"Finished partial matching. {len(rdkit_to_xyz)} atoms mapped.")
    return rdkit_to_xyz


def build_graph_orca(
    orca_file: str,
    bond_threshold: float = DEFAULT_PARAMS["orca_bond_threshold"],
    debug: bool = DEFAULT_PARAMS["debug"],
) -> nx.Graph:
    """
    Build molecular graph from ORCA quantum chemistry output file.

    Uses Mayer bond orders and Mulliken charges from ORCA calculations.
    Coordinates, charge, and multiplicity are read from the output file.

    Parameters
    ----------
    orca_file : str
        Path to ORCA output file
    bond_threshold : float, default=0.5
        Minimum Mayer bond order to consider as a bond
    debug : bool, default=False
        Enable debug logging

    Returns
    -------
    nx.Graph
        Molecular graph with nodes containing:
        - symbol: element symbol
        - atomic_number: atomic number
        - position: (x, y, z) coordinates in Angstrom
        - charges: dict with 'mulliken' key
        - formal_charge: computed formal charge
        - valence: sum of bond orders
        - agg_charge: aggregated charge (including H neighbors)

    Raises
    ------
    OrcaParseError
        If ORCA output cannot be parsed or required data is missing

    Examples
    --------
    >>> from xyzgraph import build_graph_from_orca
    >>> G = build_graph_from_orca("structure.out")
    >>> print(f"Graph has {G.number_of_nodes()} atoms and {G.number_of_edges()} bonds")
    """
    # Configure logging for debug mode (API backward compat)
    if debug:
        configure_debug_logging()

    from .orca_parser import OrcaParseError, parse_orca_output

    # Parse ORCA output
    try:
        orca_data = parse_orca_output(orca_file)
    except OrcaParseError as e:
        raise OrcaParseError(f"Failed to parse ORCA output: {e}") from e

    atoms = orca_data["atoms"]
    bonds = orca_data["bonds"]
    mulliken_charges = orca_data["charges"]
    charge = orca_data["charge"]
    multiplicity = orca_data["multiplicity"]

    logger.debug("Parsed ORCA output: %d atoms, %d bonds (before threshold)", len(atoms), len(bonds))
    logger.debug("Charge: %d, Multiplicity: %d", charge, multiplicity)

    # Build graph
    G = nx.Graph()

    # Add nodes
    for i, (symbol, pos) in enumerate(atoms):
        atomic_number = DATA.s2n.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

        G.add_node(
            i,
            symbol=symbol,
            atomic_number=atomic_number,
            position=pos,
            charges={"mulliken": mulliken_charges[i]},
        )

    # Add edges (filter by threshold)
    bonds_added = 0
    for i, j, mayer_bo in bonds:
        if mayer_bo >= bond_threshold:
            # Calculate distance
            pos_i = np.array(atoms[i][1])
            pos_j = np.array(atoms[j][1])
            distance = float(np.linalg.norm(pos_i - pos_j))

            si = G.nodes[i]["symbol"]
            sj = G.nodes[j]["symbol"]

            G.add_edge(
                i,
                j,
                bond_order=float(mayer_bo),
                distance=distance,
                bond_type=(si, sj),
                metal_coord=(si in DATA.metals or sj in DATA.metals),
            )
            bonds_added += 1

    logger.debug("Added %d bonds (threshold=%s)", bonds_added, bond_threshold)

    # Compute derived properties
    for node in G.nodes():
        # Valence (sum of bond orders, excluding metal bonds for consistency)
        valence = sum(
            G[node][nbr].get("bond_order", 1.0)
            for nbr in G.neighbors(node)
            if G.nodes[nbr]["symbol"] not in DATA.metals
        )
        G.nodes[node]["valence"] = valence

        # Compute formal charge using existing logic
        sym = G.nodes[node]["symbol"]
        if sym in DATA.metals:
            formal_charge = 0
        else:
            V = DATA.electrons.get(sym, 0)
            if V == 0:
                formal_charge = 0
            # Use simple formal charge calculation
            elif sym == "H":
                formal_charge = int(V - valence)
            else:
                B = 2 * valence
                target = 8
                L = max(0, target - B)
                formal_charge = round(V - L - B / 2)

        G.nodes[node]["formal_charge"] = formal_charge

        # Aggregated charge (include H neighbors like other methods)
        agg_charge = mulliken_charges[node]
        for nbr in G.neighbors(node):
            if G.nodes[nbr]["symbol"] == "H":
                agg_charge += mulliken_charges[nbr]
        G.nodes[node]["agg_charge"] = agg_charge

    # Add graph metadata
    from . import __citation__, __version__

    G.graph["metadata"] = {
        "version": __version__,
        "citation": __citation__,
        "source": "orca",
        "source_file": orca_file,
        "bond_threshold": bond_threshold,
    }
    G.graph["total_charge"] = charge
    G.graph["multiplicity"] = multiplicity
    G.graph["method"] = "orca"

    logger.debug("\nFinal graph: %d atoms, %d bonds", G.number_of_nodes(), G.number_of_edges())

    return G
