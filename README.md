# xyzgraph: Molecular Graph Construction from Cartesian Coordinates

**xyzgraph** is a Python toolkit for building molecular graphs (bond connectivity, bond orders, formal charges, and partial charges) directly from 3D atomic coordinates in XYZ format. It provides both **cheminformatics-based** and **quantum chemistry-based** (xTB) workflows with extensive customization and diagnostic tools.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Methodology Overview](#methodology-overview)
5. [Detailed Workflow](#detailed-workflow)
6. [Workflow Comparison](#workflow-comparison)
7. [CLI Reference](#cli-reference)
8. [Python API](#python-api)
9. [Visualization](#visualization)
10. [Performance Tuning](#performance-tuning)
11. [Known Limitations & Failure Modes](#known-limitations--failure-modes)
12. [Comparison with xyz2mol & RDKit](#comparison-with-xyz2mol--rdkit)
13. [References](#references)

---

## Key Features

- **Distance-based initial bonding** using van der Waals radii with metal-aware thresholds
- **Two construction methods**:
  - `cheminf`: Pure cheminformatics with iterative bond order optimization
  - `xtb`: Quantum chemistry via xTB Wiberg bond orders and Mulliken charges
- **Two quality modes** (cheminf only):
  - `--quick`: Fast heuristic-based valence adjustment (~10x speedup)
  - Full optimization: Formal charge minimization with conjugation penalties
- **Aromatic detection**: Hückel 4n+2 rule for 5/6-membered rings + RDKit refinement
- **Charge computation**: Gasteiger (cheminf) or Mulliken (xTB) partial charges
- **ASCII 2D depiction** with layout alignment for method comparison
- **xyz2mol comparison** for diagnostic validation against RDKit's bond perception

---

## Installation

### From PyPI (coming soon)
```bash
pip install xyzgraph
```

### From Source
```bash
git clone https://github.com/aligfellow/xyzgraph
cd xyzgraph
pip install -e .
```

### Dependencies
- **Core**: `numpy`, `networkx`, `rdkit`
- **Optional**: [xTB binary](https://github.com/grimme-lab/xtb) (for `--method xtb`)

To install xTB (Linux/macOS):
```bash
conda install -c conda-forge xtb
# or download from GitHub releases
```

---

## Quick Start

### CLI Examples

**Minimal usage** (auto-displays ASCII depiction):
```bash
xyzgraph molecule.xyz
```

**Specify charge and method**:
```bash
xyzgraph molecule.xyz --method xtb --charge -1 --multiplicity 2
```

**Fast mode for large molecules**:
```bash
xyzgraph molecule.xyz --quick
```

**Detailed debug output**:
```bash
xyzgraph molecule.xyz --debug
```

**Compare with RDKit**:
```bash
xyzgraph molecule.xyz --compare-xyz2mol
```

### Python API

**Basic usage**:
```python
from ase.io import read
from xyzgraph import build_graph, graph_to_ascii

atoms = read("molecule.xyz")
G = build_graph(atoms, method='cheminf', charge=0)

# Print ASCII structure
print(graph_to_ascii(G, scale=3.0, include_h=False))
```

**Using MolecularAnalyzer**:
```python
from xyzgraph import MolecularAnalyzer

analyzer = MolecularAnalyzer(
    atoms="molecule.xyz",
    method='cheminf',
    charge=-1,
    quick=False,
    debug=True
)

G = analyzer.build()
analyzer.print_summary(show_ascii=True, show_report=True)
```

---

## Methodology Overview

### Design Philosophy

xyzgraph offers two distinct pathways for molecular graph construction:

1. **Cheminformatics Path** (`method='cheminf'`): 
   - Pure graph-based approach using chemical heuristics
   - No external quantum chemistry calls
   - Fast and suitable for most organic molecules
   - Two quality modes: `quick` (heuristic) and `full` (optimized)

2. **Quantum Chemistry Path** (`method='xtb'`):
   - Uses xTB (extended tight-binding) calculations
   - Provides Wiberg bond orders and Mulliken charges
   - More accurate for unusual bonding situations
   - Requires xTB binary installation

### Cheminformatics Workflow (method='cheminf')

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Input Processing                                             │
│    • Parse XYZ file (ASE Atoms object)                          │
│    • Load reference data (VDW radii, valences, electrons)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 2. Initial Bond Graph (Distance-Based)                          │
│    • Compute pairwise distances                                 │
│    • Apply scaled VDW thresholds:                               │
│      - H-nonmetal: 0.45 × (r₁ + r₂)                             │
│      - H-metal: 0.50 × (r₁ + r₂)                                │
│      - Nonmetal-nonmetal: 0.55 × (r₁ + r₂)                      │
│      - Metal-ligand: 0.65 × (r₁ + r₂)                           │
│    • Create graph with single bonds (order = 1.0)               │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 3. Ring Pruning                                                 │
│    • Detect cycles (NetworkX cycle_basis)                       │
│    • Remove geometrically distorted small rings (3,4-membered)  │
│    • Threshold: max/min distance ratio > 1.18 (triangles)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 3.5 Kekulé Initialization for Aromatic Rings                    │
│    • Find 6-membered planar rings with C/N/O/S/B                │
│    • Initialize alternating bond orders: 2-1-2-1-2-1            │
│    • Handle fused rings (naphthalene, anthracene):              │
│      - Detect shared edges from previous rings                  │
│      - Validate consistency across fusion points                │
│      - Skip ring if conflicts detected                          │
│    • Gives optimizer excellent starting point                   │
│    • Reduces iterations needed for aromatic systems             │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
┌─────────▼──────────┐  ┌───────▼────────────────────────────────┐
│ 4a. Quick Mode     │  │ 4b. Full Optimization                  │
│  • Lock metal bonds│  │  • Lock metal bonds at 1.0             │
│  • 3 iterations    │  │  • Kekulé patterns already set         │
│  • Promote bonds   │  │  • Iterative BIDIRECTIONAL search:     │
│    where both atoms│  │    - Test both +1 AND -1 changes       │
│    need valence    │  │    - Allows Kekulé structure swaps     │
│    need valence    │  │  • Score = f(valence_error,            │
│  • Distance check  │  │             formal_charges,            │
│    (ratio < 0.60)  │  │             electronegativity,         │
│                    │  │             conjugation_penalty)       │
│                    │  │  • Optimizer choice:                   │
│                    │  │    - Greedy: best single change        │
│                    │  │    - Beam: parallel hypotheses         │
│                    │  │  • Cache valence sums for speed        │
│                    │  │  • Top-k edge candidate selection      │
└─────────┬──────────┘  └──────────┬─────────────────────────────┘
          └────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 5. Aromatic Detection (Hückel 4n+2)                             │
│    • Find 5/6-membered rings with C/N/O/S/P                     │
│    • Count π electrons (sp² carbons → 1e, N/O/S LP → 2e)        │
│    • Apply Hückel rule: 4n+2 π electrons                        │
│    • Set aromatic bonds to 1.5                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 6. RDKit Aromatic Refinement                                    │
│    • Build RDKit molecule from graph                            │
│    • Run RDKit's aromatic perception                            │
│    • Upgrade additional aromatic bonds to 1.5                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 7. Formal Charge Assignment                                     │
│    • For each non-metal atom:                                   │
│      - B = 2 × Σ(bond_orders)                                   │
│      - L = max(0, target - B)  [target: 2 for H, 8 otherwise]  │
│      - formal = V_electrons - (L + B/2)                         │
│    • Balance total to match system charge                       │
│    • Metals forced to 0 (coordination not oxidation state)      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 8. Gasteiger Partial Charges                                    │
│    • Convert bond orders to RDKit bond types                    │
│    • Compute Gasteiger charges                                  │
│    • Adjust for total charge conservation                       │
│    • Aggregate H charges onto heavy atoms                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 9. Output Graph                                                 │
│    Nodes: symbol, formal_charge, charges{}, agg_charge, valence │
│    Edges: bond_order, bond_type, metal_coord                    │
└─────────────────────────────────────────────────────────────────┘
```

**Suggested figure**: Flowchart showing the branching at step 4 (quick vs full)

### xTB Workflow (method='xtb')

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Input Processing                                             │
│    • Write XYZ to temporary directory                           │
│    • Set up xTB calculation parameters                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 2. Run xTB Calculation                                          │
│    Command: xtb <file>.xyz --chrg <charge> --uhf <unpaired>     │
│    • GFN2-xTB Hamiltonian                                       │
│    • Electronic structure optimization                          │
│    • Wiberg bond order analysis                                 │
│    • Mulliken population analysis                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 3. Parse xTB Output                                             │
│    • Read wbo file (Wiberg bond orders)                         │
│    • Read charges file (Mulliken atomic charges)                │
│    • Threshold: bond_order > 0.5 → create edge                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 4. Build Graph from xTB Data                                    │
│    • Create nodes with Mulliken charges                         │
│    • Create edges with Wiberg bond orders (as floats)           │
│    • Calculate aggregate charges (H → heavy atom)               │
│    • No further optimization needed                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 5. Cleanup (optional)                                           │
│    • Remove temporary xTB files (unless --no-clean)             │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 6. Output Graph                                                 │
│    Nodes: symbol, charges{'mulliken': ...}, agg_charge, valence │
│    Edges: bond_order (Wiberg), bond_type, metal_coord           │
└─────────────────────────────────────────────────────────────────┘
```

**Suggested figure**: Simple linear flowchart with xTB calculation as centerpiece

---

## Workflow Comparison

| Feature | cheminf (quick) | cheminf (full) | xtb |
|---------|----------------|----------------|-----|
| **Speed** | ⚡⚡⚡ Very Fast | ⚡⚡ Fast | ⚡ Moderate |
| **Accuracy** | Good for simple molecules | Very good for organics | Excellent (QM-based) |
| **External deps** | None | None | Requires xTB binary |
| **Bond orders** | Heuristic (integer-like) | Optimized (can be fractional) | Wiberg (fractional) |
| **Charges** | Gasteiger | Gasteiger | Mulliken |
| **Metal complexes** | Limited | Limited | Better (still simplified) |
| **Conjugated systems** | Basic | Good (conjugation penalty) | Excellent |
| **Best for** | Quick checks, large molecules | General organic chemistry | Unusual bonding, validation |

### When to Use Each Method

**Use `--method cheminf` (default)**:
- General organic molecules
- When speed matters
- No xTB installation available
- Batch processing many structures

**Use `--method cheminf --quick`**:
- Large molecules (>50 atoms)
- Initial rapid screening
- When approximate bond orders suffice

**Use `--method xtb`**:
- Validation of cheminf results
- Transition metal complexes
- Unusual electronic structures
- Publication-quality bond orders needed

### Optimizer Algorithms (cheminf full mode only)

**Greedy Optimizer** (`--optimizer greedy`, default in code):
- Tests all candidate edges, picks single best change per iteration
- Bidirectional: tests both +1 and -1 bond order changes
- Fast and effective for most molecules
- Can get stuck in local minima
- Best for: straightforward organic molecules

**Beam Search Optimizer** (`--optimizer beam`, default in CLI, `--beam-width 3`):
- Explores multiple optimization paths in parallel
- Maintains top-k hypotheses at each iteration
- Bidirectional: tests both +1 and -1 for each hypothesis
- More robust against local minima
- ~2-3x slower than greedy but better convergence
- Best for: complex conjugated systems, when greedy fails

**Example usage**:
```bash
# Greedy (fastest)
xyzgraph molecule.xyz --optimizer greedy

# Beam search with custom width (more thorough)
xyzgraph molecule.xyz --optimizer beam --beam-width 5
```

---

## CLI Reference

### Command Syntax
```bash
xyzgraph <xyz_file> [options]
```

### Options

#### Method & Quality
```bash
--method {cheminf,xtb}  # Construction method (default: cheminf)
-q, --quick             # Fast heuristic mode (cheminf only)
--max-iter INT          # Max optimization iterations (default: 50, cheminf only)
--edge-per-iter INT     # Edges evaluated per iteration (default: 6, cheminf only)
```

#### Molecular Properties
```bash
-c, --charge INT        # Total charge (default: 0)
-m, --multiplicity INT  # Spin multiplicity (auto-detected if omitted)
-b, --bohr              # Input coordinates in Bohr (default: Angstrom)
```

#### Output Control
```bash
-d, --debug             # Show detailed construction log + graph report
-a, --ascii             # Show ASCII 2D depiction (auto-enabled by default)
-as, --ascii-scale FLOAT # ASCII scaling factor (default: 3.0)
-H, --show-h            # Include hydrogens in visualizations
--compare-xyz2mol       # Compare with RDKit's xyz2mol
```

#### xTB Specific
```bash
--no-clean              # Keep temporary xTB files for inspection
```

### Examples

**Basic usage** (shows ASCII by default):
```bash
xyzgraph caffeine.xyz
```

**Charged molecule with debug info**:
```bash
xyzgraph acetate.xyz --charge -1 --debug
```

**Fast mode for large system**:
```bash
xyzgraph protein_ligand.xyz --quick --edge-per-iter 15
```

**Metal complex with xTB**:
```bash
xyzgraph ferrocene.xyz --method xtb --charge 0
```

**Method comparison**:
```bash
xyzgraph molecule.xyz --debug > cheminf.txt
xyzgraph molecule.xyz --method xtb --debug > xtb.txt
diff cheminf.txt xtb.txt
```

**Validate against RDKit**:
```bash
xyzgraph molecule.xyz --compare-xyz2mol --ascii-scale 4.0 -H
```

---

## Python API

### High-Level Interface: MolecularAnalyzer

The `MolecularAnalyzer` class provides a stateful interface with caching and comprehensive analysis tools.

```python
from xyzgraph import MolecularAnalyzer

# Create analyzer (accepts file path or ASE Atoms)
analyzer = MolecularAnalyzer(
    atoms="molecule.xyz",           # or ASE Atoms object
    method='cheminf',               # or 'xtb'
    charge=0,
    multiplicity=None,              # auto-detect
    quick=False,                    # use full optimization
    max_iter=50,                    # cheminf only
    edge_per_iter=6,                # cheminf only
    clean_up=True,                  # cleanup xTB temp files
    debug=False                     # show construction log
)

# Build graph (cached)
G = analyzer.build()

# Generate outputs
ascii_art = analyzer.ascii(scale=3.0, include_h=False)
report = analyzer.report(include_h=False)
comparison = analyzer.compare_xyz2mol(ascii=True, verbose=False)

# Print comprehensive summary
analyzer.print_summary(
    show_report=True,
    show_ascii=True,
    show_xyz2mol=False,
    include_h=False,
    ascii_scale=3.0
)
```

### Low-Level Interface: build_graph

For direct graph construction without analyzer overhead:

```python
from ase.io import read
from xyzgraph import build_graph

atoms = read("molecule.xyz")

# Cheminf full optimization
G_full = build_graph(
    atoms,
    method='cheminf',
    charge=0,
    quick=False,
    max_iter=50,
    edge_per_iter=6,
    debug=False
)

# Cheminf quick mode
G_quick = build_graph(atoms, method='cheminf', quick=True)

# xTB method
G_xtb = build_graph(atoms, method='xtb', charge=-1, multiplicity=2)
```

### Graph Structure

The returned `networkx.Graph` contains:

**Node attributes** (per atom):
```python
G.nodes[i] = {
    'symbol': str,              # Element symbol ('C', 'H', etc.)
    'atomic_number': int,       # Atomic number
    'formal_charge': int,       # Formal charge (-1, 0, +1, etc.)
    'valence': float,           # Sum of bond orders
    'agg_charge': float,        # Aggregated partial charge (includes H)
    'charges': {                # Method-specific charges
        'gasteiger': float,     # (cheminf) Gasteiger charge
        'gasteiger_raw': float, # (cheminf) Pre-adjustment Gasteiger
        'mulliken': float       # (xtb) Mulliken charge
    },
    'position': ndarray         # 3D coordinates
}
```

**Edge attributes** (per bond):
```python
G.edges[i, j] = {
    'bond_order': float,        # 1.0, 1.5 (aromatic), 2.0, 3.0, etc.
    'bond_type': tuple,         # (symbol_i, symbol_j)
    'distance': float,          # Bond length in Angstrom
    'metal_coord': bool         # True if metal-ligand bond
}
```

**Graph-level metadata**:
```python
G.graph = {
    'total_charge': int,        # System charge
    'multiplicity': int,        # Spin multiplicity
    'method': str,              # 'cheminf-quick', 'cheminf-full', or 'xtb'
    'valence_stats': dict,      # Optimization statistics
    'build_log': str            # Construction log (if debug=True)
}
```

### Accessing Graph Data

```python
# Iterate over atoms
for node, data in G.nodes(data=True):
    print(f"Atom {node}: {data['symbol']}, "
          f"formal charge: {data['formal_charge']}, "
          f"valence: {data['valence']:.2f}")

# Iterate over bonds
for i, j, data in G.edges(data=True):
    print(f"Bond {i}-{j}: order={data['bond_order']:.2f}, "
          f"length={data['distance']:.3f} Å")

# Query specific atom
carbon_idx = 5
carbon_data = G.nodes[carbon_idx]
neighbors = list(G.neighbors(carbon_idx))
carbon_valence = sum(G.edges[carbon_idx, n]['bond_order'] for n in neighbors)

# Find aromatic bonds
aromatic_bonds = [(i, j) for i, j, d in G.edges(data=True)
                  if 1.4 < d['bond_order'] < 1.6]

# Get formal charge distribution
charges = [G.nodes[i]['formal_charge'] for i in G.nodes()]
total_formal_charge = sum(charges)
```

---

## Visualization

### ASCII Depiction

xyzgraph includes a built-in ASCII renderer for 2D molecular structures:

```python
from xyzgraph import graph_to_ascii

# Basic rendering
ascii_art = graph_to_ascii(G, scale=3.0, include_h=False)
print(ascii_art)
```

**Output example** (benzene):
```
       C
      / \
     /   \
    C     C
    ‖     ‖
    C     C
     \   /
      \ /
       C
```
**Output example** (acyl isothiouronium):
```
                                             C
                                              \
                                              \
                                               C-------C
                                            ///
              ---C-               /C-------C
          C---     ---          //          \           /C----
         /            -C------N\            \          /      ---C
        C             /        \\           /C-------C/           \\
         \\          /          \\        //          \             C
           \\    ---C-          -C\-----N/            \           //
             C---     ----   ---         \             C---     //
                          -S-             \                ----C
                                          /C===
                                        // =======O
                                      C\       ====
                                       \\
                                        \\
                                        /C\
                                      //
                                    C/
```

**Features**:
- Single bonds: `-`, `|`, `/`, `\`
- Double bonds: `=`, `‖` (parallel lines)
- Triple bonds: `#`
- Aromatic: 1.5 bond orders shown as single
- Special edges: `*` (TS), `.` (NCI) if `G.edges[i,j]['TS']=True`

### Layout Alignment

Compare methods by aligning their ASCII depictions:

```python
from xyzgraph import build_graph, graph_to_ascii

# Build with both methods
G_cheminf = build_graph(atoms, method='cheminf')
G_xtb = build_graph(atoms, method='xtb')

# Generate aligned depictions
ascii_ref, layout = graph_to_ascii(G_cheminf, scale=3.0, 
                                    include_h=False, 
                                    return_layout=True)

ascii_xtb = graph_to_ascii(G_xtb, scale=3.0, 
                            include_h=False, 
                            reference_layout=layout)

print("Cheminf:\n", ascii_ref)
print("\nxTB:\n", ascii_xtb)
```

### Debug Report

Tabular listing of all atoms and bonds:

```python
from xyzgraph import graph_debug_report

report = graph_debug_report(G, include_h=False)
print(report)
```

**Output example**:
```
# Molecular Graph: 12 atoms, 12 bonds
# total_charge=0  multiplicity=1  sum(gasteiger)=+0.000
# (C–H hydrogens hidden; heteroatom-bound hydrogens shown)
# [idx] Sym  val=.. formal=.. chg=.. agg=.. | neighbors: idx(order)

[  0] C  val=4.00  formal=0  chg=+0.012  agg=+0.045 | 1(1.50) 2(1.50) 5(1.00)
[  1] C  val=4.00  formal=0  chg=-0.008  agg=-0.028 | 0(1.50) 3(1.50) 6(1.00)
...

# Bonds (i-j: order)
[ 0- 1]: 1.50
[ 0- 2]: 1.50
...
```

---

## Performance Tuning

### For Large Molecules (>50 atoms)

**Quick mode** provides ~10x speedup:
```bash
xyzgraph large.xyz --quick
```

**Increase edges per iteration** (trades iterations for evaluation time):
```bash
xyzgraph large.xyz --edge-per-iter 15 --max-iter 30
```

**Python equivalent**:
```python
G = build_graph(atoms, quick=True)
# or
G = build_graph(atoms, edge_per_iter=15, max_iter=30)
```

### Typical Performance

| Atoms | Method | Mode | Time |
|-------|--------|------|------|
| 20 | cheminf | quick | <0.1s |
| 20 | cheminf | full | 0.3s |
| 20 | xtb | - | 1-2s |
| 50 | cheminf | quick | 0.2s |
| 50 | cheminf | full | 2-5s |
| 50 | xtb | - | 3-5s |
| 100 | cheminf | quick | 0.5s |
| 100 | cheminf | full | 10-30s |
| 100 | xtb | - | 10-20s |

*Times approximate on standard laptop CPU*

---

## Known Limitations & Future Work

### Current Limitations

1. **Metal Complexes**
   - Bond orders locked at 1.0 (no d-orbital chemistry)
   - Formal charges set to 0 (coordination, not oxidation state)
   - Metal-metal bonds not supported
   - **Future**: Multi-reference methods or ligand field parameters

2. **Radicals & Open-Shell Systems**
   - Requires manual multiplicity specification
   - Single unpaired electron "slack" allowed in full mode
   - May not converge correctly for polyradicals
   - **Future**: UHF analysis or spin density integration

3. **Zwitterions**
   - Formal charge distribution may not match chemical intuition
   - Relies on electronegativity penalties
   - **Future**: pKa-aware charge state prediction

4. **Large Conjugated Systems**
   - May need >50 iterations for convergence
   - Conjugation penalty heuristic (not full π-MO analysis)
   - **Future**: Integrate Hückel MO solver

5. **Dependency on ASE**
   - Currently requires ASE for XYZ I/O
   - **Future**: Native XYZ parser to remove dependency

6. **Aromatic Detection**
   - Limited to 5/6-membered rings with C/N/O/S/P
   - No 7+ membered aromatics (e.g., tropylium)
   - **Future**: Extended Hückel rules + NICS validation

7. **Charged Aromatics**
   - Hückel electron counting simplified (doesn't account for ionic charge)
   - **Future**: Incorporate formal charges in π-electron count

### Common Failure Modes & Troubleshooting

**🔧 Problem**: Optimizer doesn't converge after 50 iterations
- **Likely cause**: Complex conjugated system or zwitterionic molecule
- **Solution**: Increase `--max-iter 100` or try beam search `--optimizer beam --beam-width 5`

**🔧 Problem**: Wrong bond orders (your chemical intuition says double bond should be single)
- **Likely cause**: Kekulé initialization started with incorrect pattern
- **Solution**: Try with `--quick` mode (no optimizer) or `--optimizer beam` with higher beam width

**🔧 Problem**: Metal coordination changes bond orders inappropriately
- **Likely cause**: Metal bonds are locked; valence donor atoms redistributed
- **Solution**: Use `--method xtb` for quantum chemistry treatment

**🔧 Problem**: xTB fails with "xtb not found"
- **Likely cause**: xTB binary not installed or not in PATH
- **Solution**: Install xTB via conda: `conda install -c conda-forge xtb`

**🔧 Problem**: Strange bonding for unusual elements
- **Likely cause**: Heuristics only trained on common organic/organometallic chemistry
- **Solution**: Use `--method xtb` for QM validation, or manually specify expected bonding

**🔧 Problem**: ASCII depiction looks wrong
- **Likely cause**: Graph build succeeded but layout failed for unusual geometry
- **Solution**: Check 3D coordinates are reasonable, try `--ascii-scale` adjustment

**🔧 Problem**: Significant differences vs xyz2mol comparison
- **Likely cause**: Different bonding heuristics (common and usually valid)
- **Solution**: Prefer xyzgraph for specific research needs, prefer xyz2mol for general RDKit compatibility

### Room for Improvement

- **Speed**: Cython/numba acceleration of scoring function
- **Accuracy**: Machine learning bond order predictor trained on DFT
- **Usability**: GUI for interactive refinement
- **Validation**: Benchmark suite against crystallographic data
- **Features**: Support for periodic systems (polymers, surfaces)

---

## Comparison with xyz2mol

[xyz2mol](https://github.com/jensengroup/xyz2mol) is a widely-used tool (part of RDKit ecosystem) that also constructs molecular graphs from XYZ coordinates. Here's how xyzgraph differs:

| Feature | xyz2mol (RDKit) | xyzgraph |
|---------|----------------|----------|
| **Method** | RDKit DetermineBonds | Cheminf or xTB |
| **Bond order optimization** | Internal RDKit heuristics | Explicit valence/charge minimization |
| **Customization** | Limited | Extensive (quick/full, tunable) |
| **Metal support** | Very limited | Basic (coordination geometry) |
| **Charges** | Not computed | Gasteiger or Mulliken |
| **Aromatic detection** | RDKit Kekulé search | Hückel + RDKit refinement |
| **Debug output** | Minimal | Extensive logs + ASCII |
| **External QM** | No | Optional (xTB) |

### When to Use xyz2mol Instead

- When you need maximum compatibility with RDKit workflows
- For molecules that RDKit handles well (simple organics)
- When speed is critical and defaults suffice

### When to Use xyzgraph Instead

- When you need detailed control over bond assignment
- For metal complexes or unusual bonding
- When you want QM-validated bond orders (xTB)
- For debugging/analysis with ASCII depictions and logs
- When formal/partial charges are needed

### Built-in Comparison

xyzgraph can directly compare its output to xyz2mol:

```bash
xyzgraph molecule.xyz --compare-xyz2mol
```

**Output includes**:
- Edge differences (bonds only in one method)
- Bond order differences (Δ ≥ 0.25)
- Layout-aligned ASCII depictions

**Example**:
```
# Bond differences: only_in_native=1   only_in_rdkit=0   bond_order_diffs=2
#   only_in_native: 4-7
#   bond_order_diffs (Δ≥0.25):
#     1-2   native=1.50   rdkit=1.00   Δ=+0.50
#     2-3   native=2.00   rdkit=1.50   Δ=+0.50
```

---

## References

1. **van der Waals Radii**: Tkatchenko & Scheffler, *Phys. Rev. Lett.* 2009, 102, 073005. Data from Fedorov *et al.*, *J. Chem. Theory Comput.* 2024, 20, 17, 7409–7423. [DOI: 10.1021/acs.jctc.4c00784](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00784)

2. **xTB (Extended Tight Binding)**: Bannwarth *et al.*, *J. Chem. Theory Comput.* 2019, 15, 3, 1652–1671. [DOI: 10.1021/acs.jctc.8b01176](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176)

3. **Gasteiger Charges**: Gasteiger & Marsili, *Tetrahedron* 1980, 36, 3219-3228. [DOI: 10.1016/0040-4020(80)80168-2](https://doi.org/10.1016/0040-4020(80)80168-2)

4. **RDKit**: RDKit: Open-source cheminformatics. [https://www.rdkit.org](https://www.rdkit.org)

5. **xyz2mol**: Kromann *et al.*, GitHub repository. [https://github.com/jensengroup/xyz2mol](https://github.com/jensengroup/xyz2mol). Now integrated into RDKit as `Chem.rdDetermineBonds.DetermineBonds()`.

6. **Hückel Rule**: Hückel, E., *Z. Phys.* 1931, 70, 204-286. [DOI: 10.1007/BF01339530](https://doi.org/10.1007/BF01339530)

7. **NetworkX**: Hagberg, Schult & Swart, "Exploring network structure, dynamics, and function using NetworkX", *Proc. 7th Python in Science Conf.* 2008, 11-15.

---

## License

MIT License - see LICENSE file for details.

## Citation

If you use xyzgraph in your research, please cite:

```bibtex
@software{xyzgraph2025,
  author = {Ali G. Fellow},
  title = {xyzgraph: Molecular Graph Construction from Cartesian Coordinates},
  year = {2025},
  url = {https://github.com/aligfellow/xyzgraph}
}
```

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.

## Contact

For questions or bug reports, please use the GitHub issue tracker at [https://github.com/aligfellow/xyzgraph/issues](https://github.com/aligfellow/xyzgraph/issues)
