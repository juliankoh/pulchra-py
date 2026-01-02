"""
Main PULCHRA class for protein structure reconstruction.

Source: pulchra.c lines 3474-3687 (main)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np

from pulchra.core.structures import Molecule
from pulchra.io.pdb_parser import read_pdb_file, read_coords_from_array
from pulchra.io.pdb_writer import write_pdb


@dataclass
class PulchraConfig:
    """Configuration for PULCHRA reconstruction."""

    # Optimization options
    ca_optimize: bool = True
    ca_iterations: int = 100
    ca_max_shift: float = 0.5

    # Reconstruction options
    rebuild_backbone: bool = True
    rebuild_sidechains: bool = True
    optimize_xvol: bool = True
    xvol_iterations: int = 3

    # Refinement options
    check_chirality: bool = True
    optimize_hbonds: bool = False

    # Input/output options
    use_pdbsg: bool = False
    detect_cispro: bool = False
    random_start: bool = False
    center_chain: bool = False
    rearrange_backbone: bool = False
    include_hydrogens: bool = False

    # Behavior
    verbose: bool = False


class Pulchra:
    """
    Main class for PULCHRA protein structure reconstruction.

    PULCHRA (PowerfUL CHain Restoration Algorithm) reconstructs
    full-atom protein models from reduced C-alpha representations.

    Example usage:
        >>> p = Pulchra()
        >>> molecule = p.reconstruct("input_ca.pdb", "output.pdb")

        >>> p = Pulchra(verbose=True, optimize_hbonds=True)
        >>> molecule = p.reconstruct("input.pdb")

        >>> # From raw coordinates
        >>> ca_coords = np.array([[...], [...], ...])
        >>> molecule = p.reconstruct_coords(ca_coords, "MKWVTF...")
    """

    def __init__(
        self,
        verbose: bool = False,
        ca_optimize: bool = True,
        rebuild_backbone: bool = True,
        rebuild_sidechains: bool = True,
        optimize_xvol: bool = True,
        check_chirality: bool = True,
        optimize_hbonds: bool = False,
        use_pdbsg: bool = False,
        detect_cispro: bool = False,
        random_start: bool = False,
        center_chain: bool = False,
        ca_max_shift: float = 0.5,
        ca_iterations: int = 100,
        **kwargs,
    ):
        """
        Initialize PULCHRA with configuration options.

        Args:
            verbose: Print progress messages
            ca_optimize: Optimize C-alpha positions
            rebuild_backbone: Rebuild backbone atoms (N, C, O)
            rebuild_sidechains: Rebuild sidechain atoms
            optimize_xvol: Optimize excluded volume (fix clashes)
            check_chirality: Check and fix D-amino acids
            optimize_hbonds: Optimize backbone H-bonds
            use_pdbsg: Input uses PDB-SG format (CA + SC center)
            detect_cispro: Auto-detect cis-prolines
            random_start: Start from random CA chain
            center_chain: Center chain at origin
            ca_max_shift: Maximum CA shift from initial coords
            ca_iterations: Maximum CA optimization iterations
        """
        self.config = PulchraConfig(
            verbose=verbose,
            ca_optimize=ca_optimize,
            rebuild_backbone=rebuild_backbone,
            rebuild_sidechains=rebuild_sidechains,
            optimize_xvol=optimize_xvol,
            check_chirality=check_chirality,
            optimize_hbonds=optimize_hbonds,
            use_pdbsg=use_pdbsg,
            detect_cispro=detect_cispro,
            random_start=random_start,
            center_chain=center_chain,
            ca_max_shift=ca_max_shift,
            ca_iterations=ca_iterations,
        )

        # Lazy load data
        self._nco_stat = None
        self._nco_stat_pro = None
        self._rot_library = None

    def _load_data(self):
        """Load reconstruction data (lazy loading)."""
        if self._nco_stat is None:
            from pulchra.data.loader import load_nco_stats, load_rotamer_library
            from pulchra.reconstruction.sidechains import RotamerLibrary

            try:
                self._nco_stat, self._nco_stat_pro = load_nco_stats()
                rot_coords, rot_idx = load_rotamer_library()
                self._rot_library = RotamerLibrary(rot_coords, rot_idx)
            except FileNotFoundError as e:
                raise RuntimeError(
                    "PULCHRA data files not found. "
                    "Please run the data conversion script first:\n"
                    "  uv run python scripts/convert_data.py"
                ) from e

    def reconstruct(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Molecule:
        """
        Reconstruct a protein structure from a PDB file.

        Args:
            input_path: Path to input PDB file (CA-only or full structure)
            output_path: Optional path for output PDB file

        Returns:
            Reconstructed Molecule object
        """
        if self.config.verbose:
            print(f"Reading {input_path}...")

        # Read input structure
        molecule = read_pdb_file(input_path, use_pdbsg=self.config.use_pdbsg)

        # Run reconstruction pipeline
        molecule = self._reconstruct_molecule(molecule)

        # Write output if path provided
        if output_path is not None:
            if self.config.verbose:
                print(f"Writing {output_path}...")
            write_pdb(
                output_path,
                molecule,
                rearrange_backbone=self.config.rearrange_backbone,
                include_hydrogens=self.config.include_hydrogens,
            )

        return molecule

    def reconstruct_coords(
        self,
        ca_coords: np.ndarray,
        sequence: str,
        chain_id: str = "A",
    ) -> Molecule:
        """
        Reconstruct from raw C-alpha coordinates.

        Args:
            ca_coords: C-alpha coordinates, shape (N, 3)
            sequence: Amino acid sequence (1-letter codes)
            chain_id: Chain identifier

        Returns:
            Reconstructed Molecule object
        """
        molecule = read_coords_from_array(ca_coords, sequence, chain_id)
        return self._reconstruct_molecule(molecule)

    def _reconstruct_molecule(self, molecule: Molecule) -> Molecule:
        """
        Run the full reconstruction pipeline on a molecule.

        Args:
            molecule: Input molecule (modified in place)

        Returns:
            Reconstructed molecule
        """
        # Load data if needed
        self._load_data()

        # Center chain if requested
        if self.config.center_chain:
            if self.config.verbose:
                print("Centering chain...")
            molecule.center_to_origin()

        # CA optimization
        if self.config.ca_optimize:
            from pulchra.optimization.ca_optimizer import CAOptimizer, CAOptimizerConfig

            config = CAOptimizerConfig(
                max_iter=self.config.ca_iterations,
                detect_cispro=self.config.detect_cispro,
                random_start=self.config.random_start,
                verbose=self.config.verbose,
            )
            optimizer = CAOptimizer(config)
            optimizer.optimize(molecule)

        # Backbone reconstruction
        if self.config.rebuild_backbone:
            from pulchra.reconstruction.backbone import rebuild_backbone

            rebuild_backbone(
                molecule,
                self._nco_stat,
                self._nco_stat_pro,
                verbose=self.config.verbose,
            )

        # Sidechain reconstruction
        if self.config.rebuild_sidechains:
            from pulchra.reconstruction.sidechains import rebuild_sidechains

            rebuild_sidechains(
                molecule,
                self._rot_library,
                use_pdbsg=self.config.use_pdbsg,
                verbose=self.config.verbose,
            )

        # Excluded volume optimization
        if self.config.optimize_xvol:
            from pulchra.reconstruction.excluded_volume import optimize_exvol

            optimize_exvol(
                molecule,
                self._rot_library,
                max_iter=self.config.xvol_iterations,
                verbose=self.config.verbose,
            )

        # Chirality check
        if self.config.check_chirality:
            from pulchra.reconstruction.chirality import chirality_check

            chirality_check(molecule, verbose=self.config.verbose)

        # H-bond optimization
        if self.config.optimize_hbonds:
            from pulchra.reconstruction.hydrogen_bonds import optimize_backbone

            optimize_backbone(molecule, verbose=self.config.verbose)

        return molecule


def reconstruct(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Molecule:
    """
    Convenience function for quick reconstruction.

    Args:
        input_path: Path to input PDB file
        output_path: Optional path for output
        **kwargs: Additional options passed to Pulchra

    Returns:
        Reconstructed Molecule
    """
    p = Pulchra(**kwargs)
    return p.reconstruct(input_path, output_path)
