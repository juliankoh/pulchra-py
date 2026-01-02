"""
Backbone atom reconstruction from C-alpha trace.

Source: pulchra.c lines 1884-2177 (prepare_rbins, rebuild_backbone)
"""

from typing import Optional, Tuple
import numpy as np

from pulchra.core.structures import Molecule, Residue, Atom
from pulchra.core.constants import FLAG_BACKBONE
from pulchra.core.geometry import calc_distance, calc_r14, superimpose


def prepare_rbins(ca_coords: np.ndarray) -> np.ndarray:
    """
    Prepare geometry bins for backbone reconstruction.

    This computes binned geometric descriptors for each peptide bond:
    - r13_1: Distance from CA[i-2] to CA[i]
    - r13_2: Distance from CA[i-1] to CA[i+1]
    - r14: Chiral distance from CA[i-2] to CA[i+1]

    Source: pulchra.c lines 1884-2013

    Args:
        ca_coords: C-alpha coordinates, shape (N, 3)

    Returns:
        Bin indices, shape (N+1, 3) where each row is [bin13_1, bin13_2, bin14]
    """
    n = len(ca_coords)

    # Extend CA coordinates at both ends for terminal handling
    # Add 2 extrapolated positions at start and 2 at end
    extended = np.zeros((n + 4, 3), dtype=np.float64)
    extended[2 : n + 2] = ca_coords

    # Extrapolate at N-terminus (positions 0, 1)
    # Use superposition of first 3 CA to extrapolate backwards
    if n >= 3:
        # Mirror first 3 CAs to extrapolate
        v1 = ca_coords[1] - ca_coords[0]
        v2 = ca_coords[2] - ca_coords[1]
        extended[1] = ca_coords[0] - v1
        extended[0] = extended[1] - v1

        # Extrapolate at C-terminus
        v1 = ca_coords[-1] - ca_coords[-2]
        extended[n + 2] = ca_coords[-1] + v1
        extended[n + 3] = extended[n + 2] + v1

    # C_ALPHA equivalent starts at index 2
    c_alpha = extended[2:]

    # Calculate bins for each position
    rbins = np.zeros((n + 1, 3), dtype=np.int32)

    for i in range(n + 1):
        # Get 4 consecutive CA positions (with offset -2)
        p1 = c_alpha[i - 2] if i >= 2 else extended[i]
        p2 = c_alpha[i - 1] if i >= 1 else extended[i + 1]
        p3 = c_alpha[i] if i < n else extended[i + 2]
        p4 = c_alpha[i + 1] if i + 1 < n else extended[i + 3]

        # Calculate distances
        r13_1 = calc_distance(p1, p3)
        r13_2 = calc_distance(p2, p4)
        r14 = calc_r14(p1, p2, p3, p4)

        # Calculate bins (same formula as C code)
        bin13_1 = int((r13_1 - 4.6) / 0.3)
        bin13_2 = int((r13_2 - 4.6) / 0.3)
        bin14 = int((r14 + 11.0) / 0.3)

        # Clamp to valid range
        bin13_1 = max(0, min(9, bin13_1))
        bin13_2 = max(0, min(9, bin13_2))
        bin14 = max(0, min(73, bin14))

        rbins[i] = [bin13_1, bin13_2, bin14]

    return rbins


def find_best_nco_template(
    nco_stat: np.ndarray,
    bin13_1: int,
    bin13_2: int,
    bin14: int,
) -> Optional[np.ndarray]:
    """
    Find the best matching NCO template for given geometry bins.

    Args:
        nco_stat: NCO statistics array, shape (M, 27)
                 Each row: [bin13_1, bin13_2, bin14, ...8x3 coords...]
        bin13_1, bin13_2, bin14: Target geometry bins

    Returns:
        Template data (8 atoms x 3 coords) or None if not found
    """
    if len(nco_stat) == 0:
        return None

    best_hit = 1000.0
    best_pos = 0

    for j, entry in enumerate(nco_stat):
        # Check for end marker (negative bin value)
        if entry[0] < 0:
            break

        # Calculate distance to target bins
        hit = (
            abs(entry[0] - bin13_1)
            + abs(entry[1] - bin13_2)
            + 0.2 * abs(entry[2] - bin14)
        )

        if hit < best_hit:
            best_hit = hit
            best_pos = j

        # Early exit if perfect match
        if hit < 1e-3:
            break

    # Extract template coordinates (8 atoms x 3 coords)
    # Layout: [bin13_1, bin13_2, bin14, CA1xyz, CA2xyz, CA3xyz, CA4xyz, Cxyz, Oxyz, Nxyz, ...]
    template = nco_stat[best_pos, 3:].reshape(8, 3)
    return template


def rebuild_backbone(
    molecule: Molecule,
    nco_stat: np.ndarray,
    nco_stat_pro: np.ndarray,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Rebuild backbone atoms (N, C, O) from C-alpha trace.

    Source: pulchra.c lines 2017-2177

    This uses pre-computed NCO geometry statistics to place
    backbone atoms based on local CA geometry.

    Args:
        molecule: Molecule to rebuild (modified in place)
        nco_stat: General backbone statistics
        nco_stat_pro: Proline-specific backbone statistics
        verbose: Whether to print progress

    Returns:
        Tuple of (average_rmsd, max_rmsd) for the reconstruction
    """
    if verbose:
        print("Rebuilding backbone...")

    n = len(molecule.residues)
    if n < 2:
        return 0.0, 0.0

    # Get CA coordinates and prepare geometry bins
    ca_coords = molecule.get_ca_coords()
    rbins = prepare_rbins(ca_coords)

    # Extended CA array for terminal handling
    extended_ca = np.zeros((n + 4, 3), dtype=np.float64)
    extended_ca[2 : n + 2] = ca_coords

    # Extrapolate terminals
    if n >= 3:
        v1 = ca_coords[1] - ca_coords[0]
        extended_ca[1] = ca_coords[0] - v1
        extended_ca[0] = extended_ca[1] - v1

        v1 = ca_coords[-1] - ca_coords[-2]
        extended_ca[n + 2] = ca_coords[-1] + v1
        extended_ca[n + 3] = extended_ca[n + 2] + v1

    c_alpha = extended_ca[2:]

    total_rmsd = 0.0
    max_rmsd = 0.0

    for i in range(n + 1):
        # Get 4 consecutive CA positions
        ca_local = np.zeros((4, 3))
        for j in range(4):
            idx = i - 2 + j
            if 0 <= idx < n:
                ca_local[j] = c_alpha[idx]
            elif idx < 0:
                ca_local[j] = extended_ca[idx + 2]
            else:
                ca_local[j] = extended_ca[idx + 2]

        # Get geometry bins
        bin13_1, bin13_2, bin14 = rbins[i]

        # Select appropriate NCO statistics (proline-specific if previous residue is PRO)
        prev_res = molecule.residues[i - 1] if 0 < i <= n else None
        use_pro = prev_res is not None and prev_res.name == "PRO"

        if use_pro and len(nco_stat_pro) > 0:
            template = find_best_nco_template(nco_stat_pro, bin13_1, bin13_2, bin14)
        else:
            template = find_best_nco_template(nco_stat, bin13_1, bin13_2, bin14)

        if template is None:
            continue

        # Template layout:
        # [0-3]: CA positions (for superposition)
        # [4]: C atom
        # [5]: O atom
        # [6]: N atom (or OXT for terminal)
        # [7]: (additional reference)

        template_ca = template[:4].copy()
        template_all = template.copy()

        # Superimpose template onto actual CA positions
        try:
            rmsd, _, rotation, translation, transformed = superimpose(
                ca_local, template_ca, template_all
            )
        except Exception:
            continue

        total_rmsd += rmsd
        if rmsd > max_rmsd:
            max_rmsd = rmsd

        # Place backbone atoms
        # C and O go on previous residue
        if prev_res is not None:
            _add_or_replace_atom(prev_res, "C  ", transformed[4], FLAG_BACKBONE)
            _add_or_replace_atom(prev_res, "O  ", transformed[5], FLAG_BACKBONE)

        # N goes on current residue (or OXT on terminal)
        curr_res = molecule.residues[i] if i < n else None
        if curr_res is not None:
            _add_or_replace_atom(curr_res, "N  ", transformed[6], FLAG_BACKBONE)
        elif prev_res is not None:
            # Terminal oxygen instead of nitrogen
            _add_or_replace_atom(prev_res, "OXT", transformed[6], FLAG_BACKBONE)

    avg_rmsd = total_rmsd / n if n > 0 else 0.0

    if verbose:
        print(f"Backbone rebuilding deviation: average = {avg_rmsd:.3f}, max = {max_rmsd:.3f}")

    return avg_rmsd, max_rmsd


def _add_or_replace_atom(
    residue: Residue,
    atom_name: str,
    coords: np.ndarray,
    flag: int,
) -> None:
    """
    Add or replace an atom in a residue.

    Args:
        residue: Residue to modify
        atom_name: Atom name (3 characters)
        coords: New coordinates
        flag: Atom flags
    """
    # Try to find existing atom
    existing = residue.get_atom(atom_name)

    if existing is not None:
        existing.coords = coords.copy()
        existing.flag |= flag
    else:
        # Create new atom
        atom = Atom(
            coords=coords.copy(),
            name=atom_name.ljust(3)[:3],
            num=0,  # Will be assigned during output
            flag=flag,
        )
        residue.add_atom(atom)


def rebuild_backbone_simple(
    molecule: Molecule,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Rebuild backbone using default NCO statistics.

    This is a convenience function that loads the statistics
    from the package data directory.

    Args:
        molecule: Molecule to rebuild
        verbose: Whether to print progress

    Returns:
        Tuple of (average_rmsd, max_rmsd)
    """
    from pulchra.data.loader import load_nco_stats

    nco_stat, nco_stat_pro = load_nco_stats()
    return rebuild_backbone(molecule, nco_stat, nco_stat_pro, verbose)
