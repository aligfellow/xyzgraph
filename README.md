# xyzgraph molecular graph generation

Usage (CLI):
  xyzgraph-build molecule.xyz --method cheminf
  xyzgraph-build molecule.xyz --method xtb --charge 0
  xyzgraph-build molecule.xyz --method cheminf --ascii   # ASCII output

Features:
  - Distance based bonding heuristic (vdw \[Tkatchenko, 10.1021/acs.jctc.4c00784])
  - Iterative valence-driven bond order refinement
  - Aromatic cycle detection (5/6 membered hetero/carbon rings)
  - Charge computation (Gasteiger for cheminf, Mulliken from XTB)
  - Post-build sanitation (valence recheck, hydrogen charge aggregation)
  - ASCII graph renderer (alignment support)
  - Optional xyz2mol comparison (diagnostic only)

## Quick CLI Usage

Minimal:
```
xyzgraph-build molecule.xyz
```

With method / charge / spin:
```
xyzgraph-build molecule.xyz --method xtb --charge -1 --multiplicity 2
```

ASCII depiction (larger, show hydrogens):
```
xyzgraph-build molecule.xyz --ascii --ascii-scale 2.5 --show-h
```

Compare with xyz2mol (if installed):
```
xyzgraph-build molecule.xyz --compare-xyz2mol
```

Key flags:
- --method {cheminf|xtb}  (default cheminf)
- --charge INT
- --multiplicity INT (inferred if omitted)
- --ascii / --ascii-scale
- --show-h
- --debug-graph / -dg
- --compare-xyz2mol
- --no-clean

Behavior:
- If only an XYZ filename is provided (no flags), a 2D ASCII depiction is shown by default.
- Without --debug-graph the detailed bond/valence/charge table is suppressed.
- --compare-xyz2mol auto-enables ASCII if neither --ascii nor -dg is set.
- xyz2mol output honors:
  * -dg (verbose atom neighbor listing + bond list)
  * --ascii / --ascii-scale / --show-h (for its own ASCII block)
  * When both --compare-xyz2mol and ASCII output are active, the xyz2mol ASCII is layout-aligned to the primary graph (same atom ordering).

## Python API Quick Start

Minimal cheminformatics build:
```python
from ase.io import read
from xyzgraph import build_graph, graph_debug_report, graph_to_ascii

atoms = read("mol.xyz")
G = build_graph(atoms)  # defaults: method='cheminf', charge=0
print(graph_debug_report(G, include_h=False))
print(graph_to_ascii(G, scale=2.0, include_h=False))
```

XTB backend (requires xtb in PATH):
```python
G_xtb = build_graph(atoms, method="xtb", charge=-1, multiplicity=2)
```

## Advanced Usage

Aligned ASCII (second graph drawn in first graph’s layout):
```python
G_ref = build_graph(atoms, method="cheminf")
G_alt = build_graph(atoms, method="xtb")
ascii_ref = graph_to_ascii(G_ref, scale=2.0, include_h=False)
ascii_alt = graph_to_ascii(G_alt, scale=2.0, include_h=False, reference=G_ref)
print(ascii_ref)
print(ascii_alt)
```

Retrieve and reuse layout explicitly:
```python
ascii_ref, layout = graph_to_ascii(G_ref, scale=2.0, include_h=False, return_layout=True)
ascii_alt = graph_to_ascii(G_alt, scale=2.0, include_h=False, reference_layout=layout)
```

Tune sanitation iterations:
```python
G_fast = build_graph(atoms, sanitize_iterations=2)
```

Access formal charges / aggregated charges:
```python
for n,data in G_ref.nodes(data=True):
    print(n, data['symbol'], data['formal_charge'], data['agg_charge'])
```

Use:
```python
print(graph_debug_report(G))
print(graph_to_ascii(G, scale=2.0))
```
And for alignment:
```python
ascii_ref, layout = graph_to_ascii(G_ref, return_layout=True)
ascii_alt = graph_to_ascii(G_alt, reference_layout=layout)
```

## Build Step Log
(Feature removed; per-step internal metrics no longer exposed.)

## xyz2mol Comparison Enhancements
Using:
```
xyzgraph-build mol.xyz --compare-xyz2mol
```
Now also prints:
- Edge connectivity differences (edges only in xyzgraph or only in xyz2mol).
- Bond order differences (|Δ| ≥ 0.25) for shared edges.
If ASCII output is active, xyz2mol ASCII is layout-aligned to the xyzgraph depiction.

Example diff section:
```
# edge_diff: only_in_xyzgraph=1 only_in_xyz2mol=0 bond_order_diffs=2
#   only_in_xyzgraph: 4-7
#   bond_order_diffs (Δ≥0.25):
#     1-2: xyzgraph=1.50 xyz2mol=1.00 Δ=+0.50
#     2-3: xyzgraph=2.00 xyz2mol=1.50 Δ=+0.50
```

## Methodology (Cheminformatics Path)

1. Input / Preload
   - Read XYZ (ASE Atoms).
   - Load data tables: van der Waals radii, expected valences, valence electron counts.

2. Initial Bond Guess
   - For each atom pair compute distance d and sum of VDW radii R = r_i + r_j.
   - Apply scaled thresholds:
       H–nonmetal: 0.45 R
       H–metal:    0.60 R
       nonmetal–nonmetal: 0.55 R
       metal–(O/N/C/P/S/halide): 0.65 R (kept only if _should_bond_metal passes).
   - Result: untyped single bonds (bond_order = 1.0).

3. Proto-Aromatic Tag
   - Detect simple 5/6 cycles with atoms in {C,N,O,S}; mark cycle edges → candidate aromatic (pre-seeding).

4. Valence-Driven Bond Order Refinement
   - Iterate (default up to 5):
       a. Compute current fractional valence per atom (sum of bond orders).
       b. Determine closest allowed valence from expected_valences.json.
       c. Compute deficit/excess; metals locked at 1.0 and skipped.
       d. Consider bond order adjustments (±0.5 or more) when:
           - Distance compact (normalized d / (r_i + r_j) ≤ 0.60).
           - Deficits complementary (patterns: ++, +−, −+).
           - Avoid over-increasing beyond 3.0 (cap).
       e. Allow one unpaired “slack” (spin) if multiplicity > 1.

5. RDKit Aromatic Perception Pass
   - Build RDKit molecule with single bonds; sanitize → aromatic flags.
   - Upgrade any aromatic bond to at least 1.5 (retain higher if already promoted).

6. Formal Charge Assignment
   - Load per-element valence electron count (valence_electrons.json).
   - For each non-metal atom: B = 2 * Σ bond_orders; L = max(0, target − B) with target=2 (H) else 8.
   - formal = V − (L + B/2). Metals forced to 0 (coordination treated graphically).
   - Adjust (distribute residual) to enforce total system charge.

7. Partial Charges
   - Gasteiger charges computed from RDKit bond typing (aromatic bonds set to aromatic type).

8. Sanitation / Recheck
   - Re-evaluate valence deviations (>0.6 from nearest allowed); if present run a limited refinement cycle.
   - Aggregate hydrogen charges onto heavy neighbors (agg_charge) while retaining per-atom charges.

9. Output Graph
   - Nodes: symbol, atomic_number, charges{...}, agg_charge, formal_charge, valence (post-sanitize).
   - Edges: bond_order (float), bond_type (symbol pair), metal_coord (bool).

## Notes / Design Choices
- Metals kept at single order to avoid over-formalization without d-electron chemistry modeling.
- Aromaticity only escalates minimum order (never reduces a prior double/triple).
- Formal charges rely on octet/duet heuristic; metals default 0 to avoid misleading oxidation states.
- Data-driven: all per-element constants externalized (radii, valences, valence electrons) for extensibility.

Unified API:
```python
from ase.io import read
from xyzgraph import build_graph, graph_to_ascii

atoms = read("mol.xyz")
G = build_graph(atoms, method="cheminf", charge=0)
print(graph_to_ascii(G, include_h=False))
```

xyz2mol comparison:
```
from xyzgraph import xyz2mol_compare
print(xyz2mol_compare(atoms, charge=0))
```

ASCII snippet example:
```
# Molecular Graph: 12 atoms, 12 bonds
[ 0] C  val=4.00  chg=+0.012  agg=+0.045 | 1(1.50) 2(1.50) 5(1.00) 6(1.00)
...
```


## References
Tckatenko, Jensen (plus ref), Rdkit?, Andrew White, ascii codes?, xyz2graph code?

## To Do
- remove compare with xyz2mol?
- supply graph and generate ascii (check working in python api)
- allow alignment of graph based on another reference? that way I could draw two with different connectivity with the same orientations for ease of viewing?