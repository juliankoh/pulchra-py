"""
Excluded volume optimization and steric clash detection.

Source: pulchra.c lines 2555-3041 (get_conflicts, optimize_exvol)
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np

from pulchra.core.structures import Molecule, Residue, Atom
from pulchra.core.constants import SG_XVOL_DIST, GRID_RES, FLAG_SIDECHAIN


class SpatialGrid:
    """
    Grid-based spatial indexing for fast collision detection.

    Source: pulchra.c lines 2727-2810 (allocate_grid)

    Uses a 3D grid to enable O(n) clash detection instead of O(n^2)
    all-pairs comparisons.
    """

    def __init__(self, resolution: float = GRID_RES):
        """
        Initialize the spatial grid.

        Args:
            resolution: Grid cell size in Angstroms (default 6.0)
        """
        self.resolution = resolution
        self.grid: Dict[Tuple[int, int, int], List[Atom]] = {}
        self.atom_to_residue: Dict[int, Residue] = {}

    def build(self, molecule: Molecule) -> None:
        """
        Build the spatial grid from a molecule.

        Args:
            molecule: Molecule to index
        """
        self.grid.clear()
        self.atom_to_residue.clear()

        for residue in molecule.residues:
            for atom in residue.atoms:
                cell = self._get_cell(atom.coords)
                if cell not in self.grid:
                    self.grid[cell] = []
                self.grid[cell].append(atom)
                self.atom_to_residue[id(atom)] = residue

    def _get_cell(self, coords: np.ndarray) -> Tuple[int, int, int]:
        """Get grid cell for coordinates."""
        return (
            int(coords[0] / self.resolution),
            int(coords[1] / self.resolution),
            int(coords[2] / self.resolution),
        )

    def get_neighbors(
        self,
        coords: np.ndarray,
        radius: float = None,
    ) -> List[Tuple[Atom, Residue]]:
        """
        Get all atoms within radius of given coordinates.

        Args:
            coords: Query coordinates
            radius: Search radius (default: grid resolution)

        Returns:
            List of (atom, residue) tuples
        """
        if radius is None:
            radius = self.resolution

        cell = self._get_cell(coords)
        neighbors = []

        # Search in neighboring cells
        n_cells = int(np.ceil(radius / self.resolution))
        for dx in range(-n_cells, n_cells + 1):
            for dy in range(-n_cells, n_cells + 1):
                for dz in range(-n_cells, n_cells + 1):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if neighbor_cell in self.grid:
                        for atom in self.grid[neighbor_cell]:
                            dist = np.linalg.norm(atom.coords - coords)
                            if dist <= radius:
                                residue = self.atom_to_residue.get(id(atom))
                                neighbors.append((atom, residue))

        return neighbors


def get_conflicts(
    residue: Residue,
    molecule: Molecule,
    grid: Optional[SpatialGrid] = None,
    threshold: float = SG_XVOL_DIST,
) -> int:
    """
    Count steric conflicts for a residue's sidechain atoms.

    Source: pulchra.c lines 2555-2638

    Args:
        residue: Residue to check
        molecule: Full molecule for context
        grid: Optional spatial grid (will create if not provided)
        threshold: Distance threshold for clash (default 1.6 A)

    Returns:
        Number of clashing atom pairs
    """
    if grid is None:
        grid = SpatialGrid()
        grid.build(molecule)

    conflicts = 0
    threshold_sq = threshold * threshold

    # Check each sidechain atom
    for atom in residue.atoms:
        if not atom.is_sidechain:
            continue

        # Get nearby atoms
        neighbors = grid.get_neighbors(atom.coords, radius=threshold * 2)

        for other_atom, other_residue in neighbors:
            # Skip self
            if other_residue is residue:
                continue

            # Skip if same atom
            if id(atom) == id(other_atom):
                continue

            # Check distance
            dist_sq = np.sum((atom.coords - other_atom.coords) ** 2)
            if dist_sq < threshold_sq:
                conflicts += 1

    return conflicts


def optimize_exvol(
    molecule: Molecule,
    rot_library=None,
    max_iter: int = 3,
    verbose: bool = False,
) -> int:
    """
    Optimize excluded volume by replacing clashing sidechains.

    Source: pulchra.c lines 2812-3041

    This iteratively identifies residues with steric clashes and
    attempts to replace their sidechains with alternative rotamers.

    Args:
        molecule: Molecule to optimize (modified in place)
        rot_library: Rotamer library for alternative conformations
        max_iter: Maximum optimization iterations
        verbose: Whether to print progress

    Returns:
        Remaining conflict count after optimization
    """
    if verbose:
        print("Optimizing excluded volume...")

    # Build spatial grid
    grid = SpatialGrid()
    grid.build(molecule)

    # Import here to avoid circular dependency
    from pulchra.reconstruction.sidechains import (
        RotamerLibrary,
        build_local_frame,
        _place_sidechain_atoms,
    )
    from pulchra.reconstruction.backbone import prepare_rbins

    # Load rotamer library if not provided
    if rot_library is None:
        from pulchra.data.loader import load_rotamer_library

        rot_coords, rot_idx = load_rotamer_library()
        rot_library = RotamerLibrary(rot_coords, rot_idx)

    ca_coords = molecule.get_ca_coords()
    rbins = prepare_rbins(ca_coords)
    n = len(molecule.residues)

    total_conflicts = 0

    for iteration in range(max_iter):
        conflicts_fixed = 0

        for i, residue in enumerate(molecule.residues):
            # Skip glycine
            if residue.res_type == 0:
                continue

            # Count conflicts for this residue
            conflicts = get_conflicts(residue, molecule, grid)

            if conflicts == 0:
                continue

            total_conflicts += conflicts

            # Try alternative rotamers
            if i >= len(rbins):
                continue

            bin13_1, bin13_2, bin14 = rbins[i]
            rotamers = rot_library.get_rotamers(
                residue.res_type, bin13_1, bin13_2, bin14, max_rotamers=10
            )

            if len(rotamers) <= 1:
                continue

            # Build local frame
            if i == 0:
                ca_prev = 2 * ca_coords[0] - ca_coords[1]
            else:
                ca_prev = ca_coords[i - 1]

            if i == n - 1:
                ca_next = 2 * ca_coords[-1] - ca_coords[-2]
            else:
                ca_next = ca_coords[i + 1]

            ca_curr = ca_coords[i]
            origin, v1, v2, v3 = build_local_frame(ca_prev, ca_curr, ca_next)

            # Try each rotamer and find one with fewer conflicts
            best_conflicts = conflicts
            best_rotamer_idx = 0

            # Save current sidechain
            original_atoms = [a.copy() for a in residue.atoms if a.is_sidechain]

            for rot_idx, rotamer in enumerate(rotamers[1:], 1):
                # Place this rotamer
                _place_sidechain_atoms(residue, rotamer, origin, v1, v2, v3)

                # Rebuild grid locally
                grid.build(molecule)

                # Count new conflicts
                new_conflicts = get_conflicts(residue, molecule, grid)

                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_rotamer_idx = rot_idx

            # Apply best rotamer
            if best_rotamer_idx > 0:
                _place_sidechain_atoms(
                    residue, rotamers[best_rotamer_idx], origin, v1, v2, v3
                )
                conflicts_fixed += 1
            else:
                # Restore original
                residue.remove_sidechain()
                for atom in original_atoms:
                    residue.add_atom(atom)

        # Rebuild grid for next iteration
        grid.build(molecule)

        if verbose:
            print(f"  Iteration {iteration + 1}: fixed {conflicts_fixed} clashes")

        if conflicts_fixed == 0:
            break

    # Final conflict count
    final_conflicts = sum(
        get_conflicts(res, molecule, grid)
        for res in molecule.residues
        if res.res_type != 0
    )

    if verbose:
        print(f"Remaining conflicts: {final_conflicts}")

    return final_conflicts
