# In a new file: molecular_analyzer.py

from typing import Optional, Union
from pathlib import Path
import networkx as nx
from ase import Atoms
from ase.io import read as read_xyz

from .graph_builders import GraphBuilder
from .ascii_renderer import graph_to_ascii
from .utils import graph_debug_report
from .compare import xyz2mol_compare


class MolecularAnalyzer:
    """
    Unified interface for molecular graph construction and analysis.
    Handles debug state, caching, and provides consistent API for both CLI and Python.
    """
    def __init__(
        self,
        atoms: Union[Atoms, str, Path],
        method: str = 'cheminf',
        charge: int = 0,
        multiplicity: Optional[int] = None,
        quick: bool = False,
        max_iter: int = 50,
        edge_per_iter: int = 6,
        clean_up: bool = True,
        debug: bool = False
    ):
        """
        Parameters
        ----------
        atoms : Atoms, str, or Path
            ASE Atoms object or path to XYZ file
        method : str
            'cheminf' or 'xtb'
        charge : int
            Total molecular charge
        multiplicity : int, optional
            Spin multiplicity (auto-detected if None)
        quick : bool
            Fast heuristic mode (cheminf only)
        max_iter : int
            Maximum optimization iterations (cheminf only)
        edge_per_iter : int
            Edges to evaluate per iteration (cheminf only)
        debug : bool
            Enable detailed construction logging
        """
        # Load atoms if needed
        if isinstance(atoms, (str, Path)):
            self.atoms = read_xyz(str(atoms))
            self.source = str(atoms)
        else:
            self.atoms = atoms
            self.source = "in-memory"
        
        # Store parameters
        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity
        self.quick = quick
        self.max_iter = max_iter
        self.edge_per_iter = edge_per_iter
        self.clean_up = clean_up
        self.debug = debug
        
        # State
        self.graph: Optional[nx.Graph] = None
        self.builder: Optional[GraphBuilder] = None
    
    def build(self) -> nx.Graph:
        """Build the molecular graph using GraphBuilder"""
        # Create GraphBuilder instance
        self.builder = GraphBuilder(
            atoms=self.atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
            method=self.method,
            quick=self.quick,
            max_iter=self.max_iter,
            edge_per_iter=self.edge_per_iter,
            debug=self.debug
        )
        # Build graph
        self.graph = self.builder.build()
        
        return self.graph
    
    def report(self, include_h: bool = False) -> str:
        """
        Generate detailed graph report.
        
        Parameters
        ----------
        include_h : bool
            Include C-H hydrogens in output
            
        Returns
        -------
        report : str
            Formatted text report
        """
        G = self.graph if self.graph is not None else self.build()
        return graph_debug_report(G, include_h=include_h)
    
    def ascii(
        self,
        scale: float = 3.0,
        include_h: bool = False,
        reference: Optional[nx.Graph] = None
    ) -> str:
        """
        Generate ASCII 2D depiction.
        
        Parameters
        ----------
        scale : float
            Scaling factor for output size
        include_h : bool
            Show C-H hydrogens
        reference : nx.Graph, optional
            Reference graph for layout alignment
            
        Returns
        -------
        ascii_art : str
            ASCII representation of molecule
        """
        G = self.graph if self.graph is not None else self.build()
        return graph_to_ascii(G, scale=scale, include_h=include_h, reference=reference)
    
    def compare_xyz2mol(
        self,
        verbose: bool = False,
        ascii: bool = True,
        ascii_scale: float = 3.0,
        ascii_include_h: bool = False
    ) -> str:
        """
        Compare with RDKit's xyz2mol implementation.
        
        Parameters
        ----------
        verbose : bool
            Include detailed atom listings
        ascii : bool
            Include ASCII depiction
        ascii_scale : float
            Scale for ASCII output
        ascii_include_h : bool
            Show hydrogens in ASCII
            
        Returns
        -------
        comparison : str
            Formatted comparison report
        """
        G = self.graph if self.graph is not None else self.build()
        return xyz2mol_compare(
            self.atoms,
            charge=self.charge,
            verbose=verbose,
            ascii=ascii,
            ascii_scale=ascii_scale,
            ascii_include_h=ascii_include_h,
            reference_graph=G
        )
    
    def print_summary(
        self,
        show_report: bool = True,
        show_ascii: bool = True,
        show_xyz2mol: bool = False,
        include_h: bool = False,
        ascii_scale: float = 3.0
    ):
        """
        Print comprehensive analysis to stdout.
        This is what the CLI uses.
        
        Parameters
        ----------
        show_report : bool
            Print detailed debug report
        show_ascii : bool
            Print ASCII depiction
        show_xyz2mol : bool
            Print xyz2mol comparison
        include_h : bool
            Include hydrogens in visualizations
        ascii_scale : float
            ASCII scaling factor
        """
        G = self.graph if self.graph is not None else self.build()
        
        if show_report:
            print("\n" + "=" * 70)
            print("GRAPH DEBUG REPORT")
            print("=" * 70)
            print(self.report(include_h=include_h))
        
        if show_ascii:
            mode = "QUICK" if self.quick else "FULL"
            print(f"\n# 2D Structure ({self.method.upper()}, {mode} mode)")
            if not include_h:
                print("# (Hydrogens hidden - use -H to show)")
            print(f"# Charge: {self.charge}, Multiplicity: {G.graph.get('multiplicity', 'N/A')}")
            print(self.ascii(scale=ascii_scale, include_h=include_h))
        
        if show_xyz2mol:
            print("\n" + "=" * 70)
            print("XYZ2MOL COMPARISON")
            print("=" * 70)
            print(self.compare_xyz2mol(
                verbose=show_report,
                ascii=show_ascii,
                ascii_scale=ascii_scale,
                ascii_include_h=include_h
            ).rstrip())


def analyze_molecule(
    atoms_or_file,
    method='cheminf',
    charge=0,
    multiplicity=None,
    quick=False,
    clean_up=True, 
    debug=False,
    show_ascii=False,
    show_report=False,
    show_xyz2mol=False,
    include_h=False,
    ascii_scale=3.0,
    max_iter=50,
    edge_per_iter=6,
    return_analyzer=False
):
    """Quick molecule analysis convenience function."""
    analyzer = MolecularAnalyzer(
        atoms=atoms_or_file,
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        quick=quick,
        max_iter=max_iter,
        edge_per_iter=edge_per_iter,
        clean_up=clean_up,
        debug=debug
    )
    G = analyzer.build()

    # Show outputs if requested
    if show_report or show_ascii or show_xyz2mol:
        analyzer.print_summary(
            show_report=show_report,
            show_ascii=show_ascii,
            show_xyz2mol=show_xyz2mol,
            include_h=include_h,
            ascii_scale=ascii_scale
        )
    
    # Return analyzer or graph
    if return_analyzer:
        return analyzer
    return G
