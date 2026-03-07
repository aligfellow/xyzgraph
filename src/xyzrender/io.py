"""Molecular input parsing."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from xyzgraph import DATA, build_graph, read_xyz_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import CrystalData

_Atoms: TypeAlias = list[tuple[str, tuple[float, float, float]]]


def load_molecule(
    path: str | Path,
    frame: int = 0,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
    rebuild: bool = False,
) -> tuple[nx.Graph, CrystalData | None]:
    """Read a molecular structure file and build a graph.

    Dispatches on file extension.  Always returns ``(graph, crystal)`` where
    *crystal* is ``None`` for non-periodic structures.
    """
    import xyzrender.formats as fmt
    from xyzrender.types import CrystalData

    p = str(path)
    logger.info("Loading %s", p)
    crystal: CrystalData | None = None

    if p.endswith(".cube"):
        graph, _cube = load_cube(p, charge=charge, multiplicity=multiplicity, kekule=kekule)
    elif p.endswith(".xyz"):
        graph = build_graph(read_xyz_file(p), charge=charge, multiplicity=multiplicity, kekule=kekule)
        try:
            with open(p) as _f:
                _f.readline()
                _comment = _f.readline()
            _lattice = _parse_extxyz_lattice(_comment)
            if _lattice is not None:
                graph.graph["lattice"] = _lattice
                logger.debug(f"extXYZ Lattice parsed:\n{_lattice}")
                _origin = _parse_extxyz_origin(_comment)
                if _origin is not None:
                    graph.graph["lattice_origin"] = _origin
        except OSError:
            pass
    elif p.endswith((".mol", ".sdf", ".mol2")):
        data = fmt.parse(p, frame=frame)
        graph = graph_from_moldata(data, charge=charge, multiplicity=multiplicity, kekule=kekule, rebuild=rebuild)
    elif p.endswith(".pdb"):
        data = fmt.parse_pdb(p)
        graph = graph_from_moldata(data, charge=charge, multiplicity=multiplicity, kekule=kekule, rebuild=rebuild)
        if data.pbc_cell is not None:
            # Position cell so it's centred on the molecular centroid.
            # PDB atoms are in Cartesian coords and needn't be near the origin,
            # so without this adjustment the cell box appears disconnected.
            centroid = np.array([pos for _, pos in data.atoms], dtype=float).mean(axis=0)
            cell_origin = centroid - 0.5 * data.pbc_cell.sum(axis=0)
            crystal = CrystalData(lattice=data.pbc_cell, cell_origin=cell_origin)
    elif p.endswith(".smi"):
        smi = Path(p).read_text(encoding="utf-8").splitlines()[0].strip()
        data = fmt.parse_smiles(smi, kekule=kekule)
        graph = graph_from_moldata(data, charge=charge, multiplicity=multiplicity, kekule=kekule, rebuild=rebuild)
    elif p.endswith(".cif"):
        data = fmt.parse_cif(p)
        graph = build_graph(data.atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)
        assert data.pbc_cell is not None
        crystal = CrystalData(lattice=data.pbc_cell)
    else:
        atoms, file_charge, file_mult = _parse_qm_output(p)
        c = charge if charge != 0 else file_charge
        m = multiplicity if multiplicity is not None else file_mult
        graph = build_graph(atoms, charge=c, multiplicity=m, kekule=kekule)

    logger.info("Built graph: %d atoms, %d bonds", graph.number_of_nodes(), graph.number_of_edges())
    return graph, crystal


def load_cube(
    path: str | Path,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
) -> tuple[nx.Graph, CubeData]:
    """Load molecular structure and orbital data from a Gaussian cube file.

    Returns both the molecular graph and the CubeData for orbital rendering.
    """
    from xyzrender.cube import parse_cube

    cube = parse_cube(path)
    graph = build_graph(cube.atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)
    logger.info(
        "Cube graph: %d atoms, %d bonds, MO %s", graph.number_of_nodes(), graph.number_of_edges(), cube.mo_index
    )
    return graph, cube


def graph_from_moldata(
    data: object,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
    rebuild: bool = False,
) -> nx.Graph:
    """Build a graph from MolData, using file bonds or xyzgraph detection."""
    import networkx as nx

    from xyzrender.formats import MolData

    assert isinstance(data, MolData)

    if not rebuild and data.bonds is not None:
        graph: nx.Graph = nx.Graph()
        for i, (sym, pos) in enumerate(data.atoms):
            graph.add_node(i, symbol=sym, position=pos)
        for i, j, order in data.bonds:
            graph.add_edge(i, j, bond_order=order)
        isolated = sum(1 for n in graph.nodes if graph.degree(n) == 0)
        if isolated > 0:
            logger.warning(
                "%d/%d atoms have no bonds from file connectivity — use --rebuild to re-detect with xyzgraph",
                isolated,
                graph.number_of_nodes(),
            )
        else:
            logger.info(
                "Graph from file connectivity: %d atoms, %d bonds",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
        return graph

    # Fall back to xyzgraph distance-based detection
    c = charge if charge != 0 else data.charge
    graph = build_graph(data.atoms, charge=c, multiplicity=multiplicity, kekule=kekule)
    logger.info(
        "Graph rebuilt via xyzgraph: %d atoms, %d bonds",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def detect_nci(graph: nx.Graph) -> nx.Graph:
    """Detect non-covalent interactions and return decorated graph.

    Uses xyzgraph's NCI detection.  Returns a new graph with ``NCI=True``
    edges for each detected interaction.  Pi-system interactions use
    centroid dummy nodes (``symbol="*"``).
    """
    from xyzgraph import detect_ncis
    from xyzgraph.nci import build_nci_graph

    logger.info("Detecting NCI interactions")
    detect_ncis(graph)
    nci_graph = build_nci_graph(graph)
    n_nci = sum(1 for _, _, d in nci_graph.edges(data=True) if d.get("NCI"))
    logger.info("Detected %d NCI interactions", n_nci)
    return nci_graph


def load_ts_molecule(
    path: str | Path,
    charge: int = 0,
    multiplicity: int | None = None,
    mode: int = 0,
    ts_frame: int = 0,
    kekule: bool = False,
) -> tuple[nx.Graph, list[dict]]:
    """Load TS and detect forming/breaking bonds via graphRC.

    Accepts QM output files or multi-frame XYZ trajectories (e.g. IRC paths).
    Returns the TS graph (with ``TS=True`` edges) and the trajectory frames.
    """
    try:
        from graphrc import run_vib_analysis
    except ImportError:
        msg = "TS detection requires graphrc: pip install 'xyzrender[ts]'"
        raise ImportError(msg) from None

    logger.info("Running graphRC analysis on %s (ts_frame=%d)", path, ts_frame)
    results = run_vib_analysis(
        input_file=str(path),
        mode=mode,
        ts_frame=ts_frame,
        enable_graph=True,
        charge=charge,
        multiplicity=multiplicity,
        print_output=False,
    )

    graph = results["graph"]["ts_graph"]
    frames = results["trajectory"]["frames"]

    # Rebuild graph with Kekule bond orders if requested, copying TS attributes
    if kekule:
        ts_frame_data = frames[ts_frame]
        atoms = list(zip(ts_frame_data["symbols"], [tuple(p) for p in ts_frame_data["positions"]], strict=True))
        kekule_graph = build_graph(atoms, charge=charge, multiplicity=multiplicity, kekule=True)
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False):
                if kekule_graph.has_edge(i, j):
                    kekule_graph[i][j].update({k: v for k, v in d.items() if k.startswith(("TS", "vib"))})
                else:
                    kekule_graph.add_edge(i, j, **{k: v for k, v in d.items() if k.startswith(("TS", "vib"))})
        graph = kekule_graph

    logger.info(
        "TS graph: %d atoms, %d bonds, %d frames", graph.number_of_nodes(), graph.number_of_edges(), len(frames)
    )
    return graph, frames


def rotate_with_viewer(graph: nx.Graph) -> None:
    """Open graph in v viewer for interactive rotation, update positions in-place.

    Writes a temp XYZ from current positions, launches v, and reads back
    the rotated coordinates.  All edge attributes (TS labels, bond orders, etc.)
    are preserved.  If the graph has a lattice, it is rotated by the same
    transformation and the cell origin is updated accordingly.
    """
    viewer = _find_viewer()
    logger.info("Opening viewer: %s", viewer)
    n = graph.number_of_nodes()
    atoms: _Atoms = [(graph.nodes[i]["symbol"], graph.nodes[i]["position"]) for i in range(n)]
    orig_pos = np.array([graph.nodes[i]["position"] for i in range(n)], dtype=float)
    lattice = graph.graph.get("lattice")

    rotated_text = _run_viewer_with_atoms(viewer, atoms, lattice=lattice)

    if not rotated_text.strip():
        sys.exit("No output from viewer — press 'z' in v to output coordinates before closing.")

    rotated_atoms = _parse_auto(rotated_text)
    if not rotated_atoms or len(rotated_atoms) != n:
        sys.exit("Could not parse viewer output.")

    for i, (_sym, pos) in enumerate(rotated_atoms):
        graph.nodes[i]["position"] = pos

    if lattice is not None:
        from xyzrender.utils import kabsch_rotation

        new_pos = np.array([graph.nodes[i]["position"] for i in range(n)], dtype=float)
        rot = kabsch_rotation(orig_pos, new_pos)
        c1 = orig_pos.mean(axis=0)
        c2 = new_pos.mean(axis=0)
        lat = np.array(lattice, dtype=float)
        origin = np.array(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
        graph.graph["lattice"] = (rot @ lat.T).T
        graph.graph["lattice_origin"] = rot @ (origin - c1) + c2


def _find_viewer() -> str:
    """Locate the v molecular viewer binary."""
    # Check PATH first (works if user has a symlink or v in PATH)
    v = shutil.which("v")
    if v:
        return v

    # Search common unix install paths for v.* (e.g. v.2.2) — picks highest version
    import glob
    from pathlib import Path

    search_dirs = [Path.home() / "bin", Path.home() / ".local" / "bin", Path("/usr/local/bin"), Path("/opt/")]

    candidates = []
    for dir in search_dirs:
        candidates.extend(glob.glob(str(dir / "v.[0-9]*")))
        candidates.extend(glob.glob(str(dir / "v")))

    if candidates:
        # sorting gives the latest versions
        return sorted(candidates)[-1]

    sys.exit(
        "Error: Cannot find 'v' viewer."
        "Add it to your $PATH environment variable or install in one of the following directories:"
        f"{', '.join(str(dir) for dir in search_dirs)}"
    )


def _run_viewer(viewer: str, xyz_path: str, extra_args: list[str] | None = None) -> str:
    """Launch v on an XYZ file and capture stdout."""
    result = subprocess.run([viewer, xyz_path, *(extra_args or [])], capture_output=True, text=True, check=False)
    return result.stdout


def _run_viewer_with_atoms(viewer: str, atoms: _Atoms, lattice: np.ndarray | None = None) -> str:
    """Write atoms to temp XYZ, launch v, capture stdout.

    If *lattice* is a diagonal (orthogonal) box, passes `cell:b{a},{b},{c}`
    to v so the cell frame is shown in the viewer too.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(f"{len(atoms)}\n\n")
        for sym, (x, y, z) in atoms:
            f.write(f"{sym}  {x: .6f}  {y: .6f}  {z: .6f}\n")
        tmp = f.name
    extra: list[str] = []
    if lattice is not None:
        # v accepts the 3x3 matrix as 9 comma-separated values
        flat = lattice.flatten()
        extra.append("cell:" + ",".join(f"{v:.6f}" for v in flat))
    try:
        return _run_viewer(viewer, tmp, extra)
    finally:
        os.unlink(tmp)


def _apply_rot_to_lattice(graph: nx.Graph, rot: np.ndarray, centroid: np.ndarray) -> None:
    """Rotate the lattice vectors and cell origin stored on *graph* by *rot*.

    The origin (if present) rotates around *centroid* like any atom would.
    When absent, render_svg defaults to (0, 0, 0) which needs no rotation.
    """
    if "lattice" not in graph.graph:
        return
    lat = np.array(graph.graph["lattice"], dtype=float)
    graph.graph["lattice"] = (rot @ lat.T).T
    if "lattice_origin" in graph.graph:
        origin = np.array(graph.graph["lattice_origin"], dtype=float)
        graph.graph["lattice_origin"] = rot @ (origin - centroid) + centroid


def apply_rotation(graph: nx.Graph, rx: float, ry: float, rz: float) -> None:
    """Rotate all atom positions in-place by Euler angles (degrees).

    Rotation is around the molecular centroid so the molecule stays centered.
    """
    nodes = list(graph.nodes())
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # Rz @ Ry @ Rx
    rot = np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ]
    )
    positions = np.array([graph.nodes[n]["position"] for n in nodes])
    centroid = positions.mean(axis=0)
    rotated = (rot @ (positions - centroid).T).T + centroid
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())
    _apply_rot_to_lattice(graph, rot, centroid)


def apply_axis_angle_rotation(graph: nx.Graph, axis: np.ndarray, angle: float) -> None:
    """Rotate all atom positions in-place around an arbitrary axis (degrees).

    Uses Rodrigues' rotation formula for a clean rotation around a single
    axis vector. Rotation is around the molecular centroid.
    """
    nodes = list(graph.nodes())
    theta = np.radians(angle)
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    k_cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = c * np.eye(3) + s * k_cross + (1 - c) * np.outer(k, k)

    positions = np.array([graph.nodes[n]["position"] for n in nodes])
    centroid = positions.mean(axis=0)
    rotated = (rot @ (positions - centroid).T).T + centroid
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())
    _apply_rot_to_lattice(graph, rot, centroid)


def load_trajectory_frames(path: str | Path) -> list[dict]:
    """Load all frames from a multi-frame XYZ or QM output (cclib).

    Returns list of `{"symbols": [...], "positions": [[x,y,z], ...]}`
    matching the graphRC frame format.
    """
    p = str(path)
    logger.info("Loading trajectory from %s", p)
    frames = _load_xyz_frames(p) if p.endswith(".xyz") else _load_qm_frames(p)
    logger.info("Loaded %d frames", len(frames))
    return frames


def load_stdin(charge: int = 0, multiplicity: int | None = None, kekule: bool = False) -> nx.Graph:
    """Read atoms from stdin — auto-detects XYZ and line-by-line formats."""
    return build_graph(_parse_auto(sys.stdin.read()), charge=charge, multiplicity=multiplicity, kekule=kekule)


def _parse_auto(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    """Auto-detect format: standard XYZ or line-by-line (symbol/Z x y z)."""
    lines = text.strip().splitlines()
    if not lines:
        return []
    # Standard XYZ: first line is atom count
    try:
        n = int(lines[0].strip())
        if n > 0 and len(lines) >= n + 2:
            return _parse_xyz(text)
    except ValueError:
        pass
    # Line-by-line: "symbol x y z" or "Z x y z" (e.g. v pipe output)
    return _parse_lines(lines)


def _parse_xyz(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    lines = text.strip().splitlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + n]:
        s, x, y, z = line.split()[:4]
        atoms.append((s, (float(x), float(y), float(z))))
    return atoms


def _parse_lines(lines: list[str]) -> list[tuple[str, tuple[float, float, float]]]:
    """Parse line-by-line atom format: 'symbol x y z' or 'Z x y z'."""
    atoms = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except (ValueError, IndexError):
            continue
        # First field: element symbol or atomic number
        try:
            sym = DATA.n2s[int(parts[0])]
        except (ValueError, KeyError):
            sym = parts[0]
        atoms.append((sym, (x, y, z)))
    return atoms


def _parse_qm_output(path: str) -> tuple[_Atoms, int, int | None]:
    """Extract coordinates from any QM output file via cclib."""
    try:
        import cclib
    except ImportError:
        msg = "QM output parsing requires cclib"
        raise ImportError(msg) from None

    logging.getLogger("cclib").setLevel(logging.CRITICAL)
    parser = cclib.io.ccopen(path, loglevel=logging.CRITICAL)
    try:
        data = parser.parse()
    except Exception:
        # cclib may crash mid-parse but still have extracted coordinates
        logger.debug("cclib raised an error; using partial data")
        data = parser

    if not hasattr(data, "atomcoords") or not hasattr(data, "atomnos") or len(data.atomcoords) == 0:
        msg = f"No coordinates found in {path}"
        raise ValueError(msg)

    atoms: _Atoms = []
    for z, (x, y, zc) in zip(data.atomnos, data.atomcoords[-1], strict=True):
        atoms.append((DATA.n2s[int(z)], (float(x), float(y), float(zc))))

    return atoms, getattr(data, "charge", 0), getattr(data, "mult", None)


def _load_xyz_frames(path: str) -> list[dict]:
    """Read all frames from a multi-frame XYZ file."""
    from xyzgraph import count_frames_and_atoms

    n_frames, n_atoms = count_frames_and_atoms(path)
    logger.debug("XYZ file: %d frames, %d atoms per frame", n_frames, n_atoms)
    frames = []
    for i in range(n_frames):
        atoms = read_xyz_file(path, frame=i)
        frames.append(
            {
                "symbols": [a[0] for a in atoms],
                "positions": [list(a[1]) for a in atoms],
            }
        )
    return frames


def _parse_extxyz_lattice(comment: str) -> np.ndarray | None:
    """Extract Lattice matrix from an XYZ comment line.

    Handles two formats:

    - extXYZ: `Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"`
    - Bare 9-float: comment line is exactly 9 space-separated floats

    Returns a (3, 3) float array (row vectors a, b, c) or None.
    """
    import re

    m = re.search(r'Lattice\s*=\s*"([^"]+)"', comment, re.IGNORECASE)
    if m:
        vals_str = m.group(1)
        try:
            vals = [float(x) for x in vals_str.split()]
        except ValueError:
            logger.warning("extXYZ Lattice= found but content is not numeric: %r", vals_str)
            return None
        if len(vals) != 9:
            logger.warning("extXYZ Lattice= found but expected 9 values, got %d: %r", len(vals), vals_str)
            return None
        return np.array(vals, dtype=float).reshape(3, 3)

    # Bare 9-float fallback (no Lattice= key — comment is exactly 9 floats)
    stripped = comment.strip()
    try:
        vals = [float(x) for x in stripped.split()]
    except ValueError:
        return None
    if len(vals) != 9:
        return None
    return np.array(vals, dtype=float).reshape(3, 3)


def _parse_extxyz_origin(comment: str) -> np.ndarray | None:
    """Extract cell Origin from an extXYZ comment line.

    Looks for Origin="ox oy oz" and returns a (3,) float array or
    None if the key is absent.
    """
    import re

    m = re.search(r'Origin\s*=\s*"([^"]+)"', comment, re.IGNORECASE)
    if not m:
        return None
    try:
        vals = [float(x) for x in m.group(1).split()]
    except ValueError:
        return None
    if len(vals) != 3:
        return None
    return np.array(vals, dtype=float)


def _load_qm_frames(path: str) -> list[dict]:
    """Extract all optimization steps from QM output via cclib."""
    try:
        import cclib
    except ImportError:
        msg = "QM output parsing requires cclib"
        raise ImportError(msg) from None

    logging.getLogger("cclib").setLevel(logging.CRITICAL)
    parser = cclib.io.ccopen(path, loglevel=logging.CRITICAL)
    try:
        data = parser.parse()
    except Exception:
        logger.debug("cclib raised an error; using partial data")
        data = parser
    symbols = [DATA.n2s[int(z)] for z in data.atomnos]
    coords = np.array(data.atomcoords)
    logger.debug("cclib trajectory: %d steps, %d atoms", len(coords), len(symbols))

    return [{"symbols": symbols, "positions": step.tolist()} for step in coords]
