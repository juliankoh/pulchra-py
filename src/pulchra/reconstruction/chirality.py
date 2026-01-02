"""
Chirality checking and D-amino acid correction.

Source: pulchra.c lines 3099-3182 (chirality_check)
"""

from typing import List
import numpy as np

from pulchra.core.structures import Molecule, Residue, Atom
from pulchra.core.geometry import calc_torsion, normalize
from pulchra.core.constants import FLAG_SIDECHAIN


def chirality_check(
    molecule: Molecule,
    verbose: bool = False,
) -> List[int]:
    """
    Check and fix D-amino acids in the structure.

    Source: pulchra.c lines 3099-3182

    D-amino acids have a negative CA-N-C-CB torsion angle (around -34 deg)
    while L-amino acids have positive (~34 deg). This function detects
    D-amino acids and reflects their sidechains to L configuration.

    Args:
        molecule: Molecule to check and fix (modified in place)
        verbose: Whether to print warnings

    Returns:
        List of residue indices that were fixed
    """
    if verbose:
        print("Checking chirality...")

    fixed_residues = []

    for i, residue in enumerate(molecule.residues):
        # Get required atoms
        ca = residue.ca
        n = residue.n
        c = residue.c
        cb = residue.cb

        # Skip if missing atoms or glycine
        if ca is None or n is None or c is None or cb is None:
            continue

        # Calculate CA-N-C-CB torsion angle
        angle = calc_torsion(ca.coords, n.coords, c.coords, cb.coords)

        # D-amino acid if angle is significantly negative
        # L-amino acids typically have ~+34 deg (+0.59 rad), D-amino acids ~-34 deg
        # Use -0.1 rad (~-5.7 deg) threshold to avoid false positives from numerical noise
        if angle < -0.1:
            if verbose:
                print(
                    f"WARNING: D-aa detected at {residue.name} {residue.num}: {angle:.2f}"
                )

            # Reflect sidechain across backbone plane
            _reflect_sidechain(residue, ca, n, c)

            # Verify fix
            new_angle = calc_torsion(ca.coords, n.coords, c.coords, cb.coords)
            if verbose:
                print(f", fixed: {new_angle:.2f}")

            fixed_residues.append(i)

    return fixed_residues


def _reflect_sidechain(
    residue: Residue,
    ca: Atom,
    n: Atom,
    c: Atom,
) -> None:
    """
    Reflect sidechain atoms across the backbone plane.

    This is done by rotating 180 degrees around the axis
    bisecting the N-CA-C angle.

    Args:
        residue: Residue to modify
        ca, n, c: Backbone atoms defining the plane
    """
    # Vectors from CA to N and C
    v_n = ca.coords - n.coords
    v_c = c.coords - ca.coords

    # Axis of rotation (bisector of N-CA-C angle)
    # Actually we use the vector difference for 180 degree rotation
    axis = v_n - v_c
    axis = normalize(axis)

    # For 180 degree rotation (reflection)
    cos_theta = -1.0
    sin_theta = 0.0

    # Apply rotation to each sidechain atom
    for atom in residue.atoms:
        if not atom.is_sidechain:
            continue

        # Vector from CA to atom
        p = atom.coords - ca.coords

        # Rodrigues' rotation formula for 180 degrees simplifies to:
        # p' = 2 * (axis . p) * axis - p
        # Which is a reflection across the plane perpendicular to the axis
        dot = np.dot(axis, p)
        new_p = 2 * dot * axis - p

        # Or use the full Rodrigues formula:
        # p' = p*cos(theta) + (axis x p)*sin(theta) + axis*(axis.p)*(1-cos(theta))
        # For theta = pi: cos = -1, sin = 0
        # p' = -p + 2*axis*(axis.p)
        q = (
            p * cos_theta
            + np.cross(axis, p) * sin_theta
            + axis * np.dot(axis, p) * (1 - cos_theta)
        )

        atom.coords = q + ca.coords


def check_chirality_simple(
    molecule: Molecule,
    verbose: bool = False,
) -> int:
    """
    Check and fix chirality, returning count of fixed residues.

    Args:
        molecule: Molecule to check
        verbose: Whether to print warnings

    Returns:
        Number of residues fixed
    """
    fixed = chirality_check(molecule, verbose)
    return len(fixed)
