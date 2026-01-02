"""
Hydrogen bond optimization for backbone refinement.

Source: pulchra.c lines 3188-3472 (hb_energy, optimize_backbone)
"""

from typing import Optional, Tuple
import numpy as np

from pulchra.core.structures import Molecule, Residue, Atom
from pulchra.core.geometry import calc_distance, rotate_point_around_axis
from pulchra.reconstruction.excluded_volume import SpatialGrid


# DSSP hydrogen bond energy parameters
HB_Q = 332.0  # Electrostatic constant
HB_CUTOFF = -0.5  # Energy cutoff for hydrogen bond


def calculate_h_position(
    n_atom: Atom,
    c_prev: Atom,
    o_prev: Atom,
) -> np.ndarray:
    """
    Calculate approximate hydrogen position on backbone nitrogen.

    The H atom is placed along the N-C(prev) direction, opposite to
    the carbonyl.

    Args:
        n_atom: Nitrogen atom
        c_prev: Previous carbonyl carbon
        o_prev: Previous carbonyl oxygen

    Returns:
        Approximate H coordinates
    """
    # Direction from C to N
    direction = n_atom.coords - c_prev.coords
    direction = direction / np.linalg.norm(direction)

    # H is about 1.0 A from N along this direction
    h_coords = n_atom.coords + 1.0 * direction

    return h_coords


def hb_energy(
    residue: Residue,
    molecule: Molecule,
    grid: Optional[SpatialGrid] = None,
) -> float:
    """
    Calculate DSSP-style hydrogen bond energy for a residue.

    Source: pulchra.c lines 3188-3318

    The DSSP energy function is:
    E = Q * (1/d_ON + 1/d_CH - 1/d_OH - 1/d_CN)

    where Q = 332 kcal/mol * A (electrostatic constant)

    Args:
        residue: Residue to evaluate
        molecule: Full molecule for context
        grid: Optional spatial grid for neighbor search

    Returns:
        Hydrogen bond energy (negative = favorable)
    """
    # Get backbone atoms
    n_atom = residue.n
    c_atom = residue.c
    o_atom = residue.o

    if n_atom is None or c_atom is None or o_atom is None:
        return 0.0

    if grid is None:
        grid = SpatialGrid()
        grid.build(molecule)

    total_energy = 0.0

    # Find potential H-bond partners
    # N-H...O=C pattern
    neighbors = grid.get_neighbors(n_atom.coords, radius=5.0)

    for other_atom, other_residue in neighbors:
        if other_residue is residue:
            continue

        # Check if this is a carbonyl oxygen
        if other_atom.name.strip() != "O":
            continue

        other_c = other_residue.c
        if other_c is None:
            continue

        # Calculate H position
        # Find previous residue to get C for H positioning
        res_idx = None
        for i, res in enumerate(molecule.residues):
            if res is residue:
                res_idx = i
                break

        if res_idx is None or res_idx == 0:
            continue

        prev_res = molecule.residues[res_idx - 1]
        if prev_res.c is None or prev_res.o is None:
            continue

        h_coords = calculate_h_position(n_atom, prev_res.c, prev_res.o)

        # Calculate distances
        d_on = calc_distance(other_atom.coords, n_atom.coords)
        d_ch = calc_distance(other_c.coords, h_coords)
        d_oh = calc_distance(other_atom.coords, h_coords)
        d_cn = calc_distance(other_c.coords, n_atom.coords)

        # Avoid division by zero
        if d_on < 0.1 or d_ch < 0.1 or d_oh < 0.1 or d_cn < 0.1:
            continue

        # DSSP energy
        energy = HB_Q * (1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn)

        # Only count if it's a favorable interaction
        if energy < HB_CUTOFF:
            total_energy += energy

    return total_energy


def optimize_backbone(
    molecule: Molecule,
    max_iter: int = 10,
    verbose: bool = False,
) -> float:
    """
    Optimize backbone H-bonds by rotating peptide planes.

    Source: pulchra.c lines 3409-3472

    This adjusts the orientation of peptide planes (C=O and N-H)
    to improve hydrogen bonding patterns.

    Args:
        molecule: Molecule to optimize (modified in place)
        max_iter: Maximum optimization iterations
        verbose: Whether to print progress

    Returns:
        Energy improvement achieved
    """
    if verbose:
        print("Optimizing backbone H-bonds...")

    grid = SpatialGrid()
    grid.build(molecule)

    # Calculate initial energy
    initial_energy = sum(hb_energy(res, molecule, grid) for res in molecule.residues)

    # Peptide plane rotation optimization
    for iteration in range(max_iter):
        improved = False

        for i, residue in enumerate(molecule.residues):
            if i == 0:
                continue

            # Get atoms involved in peptide bond
            c_prev = molecule.residues[i - 1].c
            o_prev = molecule.residues[i - 1].o
            n_curr = residue.n
            ca_curr = residue.ca

            if any(a is None for a in [c_prev, o_prev, n_curr, ca_curr]):
                continue

            # Current energy
            current_energy = hb_energy(residue, molecule, grid)
            current_energy += hb_energy(molecule.residues[i - 1], molecule, grid)

            # Try small rotations around CA-C axis
            for angle in [-5.0, -2.0, 2.0, 5.0]:
                angle_rad = np.radians(angle)

                # Rotate O around C-N axis
                axis_point = c_prev.coords
                axis_dir = n_curr.coords - c_prev.coords

                # Save original position
                orig_o = o_prev.coords.copy()

                # Rotate
                o_prev.coords = rotate_point_around_axis(
                    o_prev.coords, axis_point, axis_dir, angle_rad
                )

                # Rebuild grid
                grid.build(molecule)

                # Calculate new energy
                new_energy = hb_energy(residue, molecule, grid)
                new_energy += hb_energy(molecule.residues[i - 1], molecule, grid)

                if new_energy < current_energy - 0.1:
                    current_energy = new_energy
                    improved = True
                else:
                    # Restore
                    o_prev.coords = orig_o

        grid.build(molecule)

        if not improved:
            break

    # Calculate final energy
    final_energy = sum(hb_energy(res, molecule, grid) for res in molecule.residues)

    improvement = initial_energy - final_energy

    if verbose:
        print(f"H-bond energy: {initial_energy:.2f} -> {final_energy:.2f}")
        print(f"Improvement: {improvement:.2f}")

    return improvement
