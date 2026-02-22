# NCI Detection Module

Non-covalent interaction (NCI) detection for molecular graphs built by xyzgraph.

## Quick Start

```python
from xyzgraph import build_graph, detect_ncis

G = build_graph("molecule.xyz")
ncis = detect_ncis(G)

for nci in ncis:
    print(nci.type, nci.site_a, nci.site_b, nci.geometry)
```

## Accessing Results

`detect_ncis()` both returns the list and stores it on the graph as `G.graph["ncis"]`:

```python
# Direct return value
ncis = detect_ncis(G)

# Or retrieve later from the graph
ncis = G.graph["ncis"]
```

Each element is an `NCIData` with `type`, `site_a`, `site_b`, `aux_atoms`, and `geometry`:

```python
for nci in G.graph["ncis"]:
    print(nci.type)        # e.g. "hbond"
    print(nci.site_a)      # donor atom indices, e.g. (0,)
    print(nci.site_b)      # acceptor atom indices, e.g. (3,)
    print(nci.aux_atoms)   # bridging atoms, e.g. (2,) for the H
    print(nci.geometry)    # {"distance": 2.85, "h_distance": 1.92, "angle": 168.3}
```

Filter by interaction type:

```python
hbonds = [nci for nci in G.graph["ncis"] if nci.type == "hbond"]
pi_stack = [nci for nci in G.graph["ncis"] if nci.type.startswith("pi_pi")]
```

Get atom symbols for a site:

```python
for nci in G.graph["ncis"]:
    syms_a = [G.nodes[i]["symbol"] for i in nci.site_a]
    syms_b = [G.nodes[i]["symbol"] for i in nci.site_b]
    print(f"{nci.type}: {syms_a} ... {syms_b}")
```

## Geometry Keys

All interactions use standardized geometry keys. The `type` field identifies
the interaction, so keys are generic rather than type-specific:

| Key | Meaning | Present in |
|---|---|---|
| `distance` | Primary distance (D-A, centroid-centroid, ion-ion, etc.) | All types |
| `angle` | Primary angle (D-H-A, C-X-A, plane-plane, etc.) | Most types |
| `h_distance` | H-to-acceptor distance | `hbond`, `salt_bridge` |
| `h_separation` | Inter-plane separation | `pi_pi_parallel`, `pi_pi_t_shaped`, `pi_pi_ring_domain`, `pi_pi_domain_domain` |
| `lateral_disp` | Lateral displacement of centroids | `pi_pi_parallel` |
| `plane_alignment` | Cosine of H-centroid vector to plane normal | `ch_pi`, `hb_pi` |

## Serialization

`NCIData.to_dict()` converts each interaction to a JSON-compatible dictionary:

```python
nci = ncis[0]
d = nci.to_dict()
# {"type": "hbond", "site_a": [0], "site_b": [3], "aux_atoms": [2],
#  "geometry": {"distance": 2.85, "h_distance": 1.92, "angle": 168.3},
#  "score": 1.0}
```

`graph_to_dict()` automatically includes NCIs when present:

```python
from xyzgraph.utils import graph_to_dict

detect_ncis(G)
d = graph_to_dict(G)
d["ncis"]  # list of NCI dicts
```

CLI: `xyzgraph molecule.xyz --nci --json` outputs the full graph with NCIs as JSON.

## NCI Graph

`build_nci_graph()` returns a copy of the molecular graph decorated with NCI
edges and pi-system centroid nodes. This is useful for renderers or downstream
tools that work with graph objects:

```python
from xyzgraph import build_graph, detect_ncis
from xyzgraph.nci import build_nci_graph

G = build_graph("molecule.xyz")
detect_ncis(G)
nci_G = build_nci_graph(G)

# NCI edges have bond_order=0.0, NCI=True, nci_type="hbond" etc.
for i, j, d in nci_G.edges(data=True):
    if d.get("NCI"):
        print(f"NCI edge {i}-{j}: {d['nci_type']}")

# Pi-system centroids are nodes with symbol="*"
for cid in nci_G.graph["nci_centroid"]:
    print(f"Centroid node {cid} at {nci_G.nodes[cid]['position']}")

# Or find centroids by symbol
centroids = [n for n, d in nci_G.nodes(data=True) if d.get("symbol") == "*"]
```

The original graph `G` is never modified. The returned graph can be passed
directly to any renderer that understands `NCI=True` edges (e.g. as dotted
lines) and `symbol="*"` nodes (e.g. as centroid markers).

## Batch / Trajectory Analysis

For multiple frames sharing the same topology, use `NCIAnalyzer` to avoid
repeating topology work (site classification, pair enumeration, pi-system
detection) on every frame:

```python
import numpy as np
from xyzgraph import build_graph
from xyzgraph.nci import NCIAnalyzer

G = build_graph("frame0.xyz")
analyzer = NCIAnalyzer(G)  # topology work done once

for positions in trajectory_frames:
    ncis = analyzer.detect(positions)  # returns list[NCIData]
    for nci in ncis:
        d = nci.to_dict()  # clean dict for aggregation
```

Each call to `analyzer.detect(positions)` returns a plain `list[NCIData]` --
no graph objects per frame, just lightweight data suitable for building
DataFrames or time-series statistics.

## Supported Interaction Types

| Type | Description |
|---|---|
| `hbond` | Classical hydrogen bond (D-H...A) |
| `hbond_bifurcated` | Two donors sharing the same acceptor |
| `halogen_bond` | Sigma-hole bond via halogen (X...A) |
| `chalcogen_bond` | Sigma-hole bond via S, Se, Te |
| `pnictogen_bond` | Sigma-hole bond via P, As, Sb, Bi |
| `pi_pi_parallel` | Parallel-displaced pi-stacking (ring-ring) |
| `pi_pi_t_shaped` | T-shaped (edge-to-face) pi-stacking |
| `pi_pi_ring_domain` | Pi-stacking between ring and non-ring domain |
| `pi_pi_domain_domain` | Pi-stacking between two non-ring domains |
| `cation_pi` | Cation above aromatic ring |
| `anion_pi` | Anion above aromatic ring |
| `halogen_pi` | Halogen sigma-hole to pi-system |
| `ch_pi` | C-H...pi interaction |
| `hb_pi` | H-bond donor to pi-system |
| `cation_lp` | Cation to lone pair donor |
| `ionic` | Electrostatic cation-anion |
| `salt_bridge` | H-mediated ionic (cation-H...anion) |

## Data Model

Each detected interaction is an `NCIData` (frozen dataclass):

```python
@dataclass(frozen=True)
class NCIData:
    type: str                    # one of the types above
    site_a: tuple[int, ...]      # atom indices (single atom or group)
    site_b: tuple[int, ...]      # atom indices (single atom or group)
    aux_atoms: tuple[int, ...]   # bridging atoms (e.g. H in D-H...A)
    geometry: dict[str, float]   # measured distances and angles
    score: float = 1.0           # 1.0 = binary; reserved for future decay
```

For group interactions (pi-stacking, cation-pi, etc.), `site_a` or `site_b`
contain multiple atom indices representing the pi-system.

## Custom Thresholds

All geometric cutoffs are configurable via `NCIThresholds`:

```python
from xyzgraph.nci import NCIThresholds, detect_ncis

thr = NCIThresholds(
    hb_da_max=3.0,           # tighter D-A distance
    pii_parallel_rmax=4.0,   # tighter pi-stacking
)
ncis = detect_ncis(G, thresholds=thr)
```

See `thresholds.py` for the full list of parameters.

## Architecture

```
nci/
  __init__.py       # exports only
  interaction.py    # NCIData dataclass, NCI_TYPES list
  thresholds.py     # NCIThresholds dataclass
  analyzer.py       # NCIAnalyzer class, detect_ncis() convenience function
  geometry.py       # vector math primitives (unit, angle, plane normal)
  pi_systems.py     # aromatic ring and conjugated domain detection
  sites.py          # atom classification (donors, acceptors, ions, etc.)
  pairs.py          # candidate pair enumeration
  detector.py       # per-frame geometry validation for all NCI types
  graph.py          # build_nci_graph: decorate graph with NCI edges/centroids
  display.py        # format_nci_table, render_nci_ascii (CLI output)
```

**Pipeline**: `build_graph` (once) -> `NCIAnalyzer` topology setup (once) ->
`detect(positions)` geometry checks (per frame).

**For rendering**: `detect_ncis(G)` -> `build_nci_graph(G)` -> pass decorated
graph to any renderer.

## Site Detection

Sites are classified using `formal_charge` from the bond order optimizer.

## Dependencies

This module uses only `numpy` and `networkx`, both already required by xyzgraph.
No additional dependencies are introduced.
