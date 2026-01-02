"""
Sidechain reconstruction using rotamer library.

Source: pulchra.c lines 2308-2546 (rebuild_sidechains)
"""

from typing import Optional, List, Tuple
import numpy as np

from pulchra.core.structures import Molecule, Residue, Atom
from pulchra.core.constants import FLAG_SIDECHAIN, NHEAVY, HEAVY_ATOMS
from pulchra.core.geometry import normalize, superimpose
from pulchra.reconstruction.backbone import prepare_rbins


class RotamerLibrary:
    """
    Manages rotamer coordinate library for sidechain reconstruction.

    The rotamer library contains pre-computed sidechain conformations
    indexed by residue type and local CA geometry.
    """

    def __init__(self, rot_coords: np.ndarray, rot_idx: np.ndarray):
        """
        Initialize the rotamer library.

        Args:
            rot_coords: Rotamer coordinates, shape (N, 3)
            rot_idx: Rotamer index table, shape (M, 6)
                    Each row: [aa_type, bin13_1, bin13_2, bin14, count, offset]
        """
        self.coords = rot_coords
        self.idx = rot_idx

        # Build lookup index for faster access
        self._build_index()

    def _build_index(self):
        """Build a dictionary index for fast lookup."""
        self._index = {}
        for i, entry in enumerate(self.idx):
            aa_type = int(entry[0])
            key = (aa_type, int(entry[1]), int(entry[2]), int(entry[3]))
            if key not in self._index:
                self._index[key] = []
            self._index[key].append(i)

    def get_rotamers(
        self,
        res_type: int,
        bin13_1: int,
        bin13_2: int,
        bin14: int,
        max_rotamers: int = 10,
    ) -> List[np.ndarray]:
        """
        Get rotamer conformations matching the geometry.

        The rotamer data stores nheavy+1 coords per entry, where the first
        is a reference CA position and the rest are sidechain atoms.
        This method returns only the sidechain atoms (skipping the CA ref).

        Args:
            res_type: Amino acid type (0-19)
            bin13_1, bin13_2, bin14: Geometry bin indices
            max_rotamers: Maximum rotamers to return

        Returns:
            List of rotamer coordinate arrays, each shape (n_atoms, 3)
        """
        if res_type >= len(NHEAVY):
            return []

        n_atoms = NHEAVY[res_type]
        if n_atoms == 0:
            return []

        # Data stores n_atoms + 1 coords (CA ref + sidechain atoms)
        nsc = n_atoms + 1
        rotamers = []

        # Try exact match first
        key = (res_type, bin13_1, bin13_2, bin14)
        if key in self._index:
            for idx_pos in self._index[key]:
                entry = self.idx[idx_pos]
                count = int(entry[4])
                offset = int(entry[5])

                for i in range(min(count, max_rotamers - len(rotamers))):
                    start = offset + i * nsc
                    end = start + nsc
                    if end <= len(self.coords):
                        # Skip first coord (CA ref), take only sidechain atoms
                        rotamers.append(self.coords[start + 1 : end].copy())

                    if len(rotamers) >= max_rotamers:
                        break

        # If not enough rotamers, search nearby bins
        if len(rotamers) < max_rotamers:
            for d1 in range(-1, 2):
                for d2 in range(-1, 2):
                    for d3 in range(-2, 3):
                        if d1 == 0 and d2 == 0 and d3 == 0:
                            continue

                        key = (
                            res_type,
                            max(0, min(9, bin13_1 + d1)),
                            max(0, min(9, bin13_2 + d2)),
                            max(0, min(73, bin14 + d3)),
                        )

                        if key in self._index:
                            for idx_pos in self._index[key]:
                                entry = self.idx[idx_pos]
                                count = int(entry[4])
                                offset = int(entry[5])

                                for i in range(min(count, max_rotamers - len(rotamers))):
                                    start = offset + i * nsc
                                    end = start + nsc
                                    if end <= len(self.coords):
                                        # Skip first coord (CA ref)
                                        rotamers.append(self.coords[start + 1 : end].copy())

                                    if len(rotamers) >= max_rotamers:
                                        return rotamers

        return rotamers


def build_local_frame(
    ca_prev: np.ndarray,
    ca_curr: np.ndarray,
    ca_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a local coordinate frame from three consecutive CA atoms.

    Source: pulchra.c lines 2340-2400

    Args:
        ca_prev: Previous CA position
        ca_curr: Current CA position
        ca_next: Next CA position

    Returns:
        Tuple of (origin, v1, v2, v3) defining the local frame
    """
    # v1: direction along backbone
    v1 = ca_next - ca_prev
    v1 = normalize(v1)

    # v2: perpendicular to backbone plane
    va = ca_next - ca_curr
    vb = ca_curr - ca_prev
    v2 = np.cross(va, vb)
    v2 = normalize(v2)

    # v3: perpendicular to both
    v3 = np.cross(v1, v2)
    v3 = normalize(v3)

    return ca_curr, v1, v2, v3


def rebuild_sidechains(
    molecule: Molecule,
    rot_library: RotamerLibrary,
    use_pdbsg: bool = False,
    verbose: bool = False,
) -> int:
    """
    Rebuild sidechain atoms using rotamer library.

    Source: pulchra.c lines 2308-2546

    Args:
        molecule: Molecule to rebuild (modified in place)
        rot_library: Rotamer library for sidechain conformations
        use_pdbsg: If True, use input sidechain centers to select rotamers
        verbose: Whether to print progress

    Returns:
        Number of residues with sidechains rebuilt
    """
    if verbose:
        print("Rebuilding side chains...")

    n = len(molecule.residues)
    if n < 3:
        return 0

    # Get CA coordinates and geometry bins
    ca_coords = molecule.get_ca_coords()
    rbins = prepare_rbins(ca_coords)

    rebuilt_count = 0

    for i, residue in enumerate(molecule.residues):
        # Skip glycine (no sidechain)
        if residue.res_type == 0 or residue.res_type >= len(NHEAVY):
            continue

        n_atoms = NHEAVY[residue.res_type]
        if n_atoms == 0:
            continue

        # Get geometry bins for this position
        if i >= len(rbins):
            continue
        bin13_1, bin13_2, bin14 = rbins[i]

        # Get rotamer candidates
        rotamers = rot_library.get_rotamers(
            residue.res_type, bin13_1, bin13_2, bin14, max_rotamers=10
        )

        if not rotamers:
            continue

        # Build local coordinate frame
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

        # Select best rotamer
        if use_pdbsg and residue.sg_center is not None:
            # Select rotamer closest to input sidechain center
            best_rotamer = _select_rotamer_by_center(
                rotamers, origin, v1, v2, v3, residue.sg_center
            )
        else:
            # Use first (most common) rotamer
            best_rotamer = rotamers[0]

        # Transform rotamer to molecule frame and place atoms
        _place_sidechain_atoms(residue, best_rotamer, origin, v1, v2, v3)
        rebuilt_count += 1

    if verbose:
        print(f"Rebuilt {rebuilt_count} sidechains")

    return rebuilt_count


def _select_rotamer_by_center(
    rotamers: List[np.ndarray],
    origin: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    target_center: np.ndarray,
) -> np.ndarray:
    """
    Select the rotamer whose center of mass is closest to target.

    Args:
        rotamers: List of rotamer coordinates
        origin, v1, v2, v3: Local coordinate frame
        target_center: Target sidechain center of mass

    Returns:
        Best matching rotamer coordinates
    """
    best_rotamer = rotamers[0]
    best_distance = float("inf")

    # Rotation matrix from local to global
    rot_matrix = np.column_stack([v1, v2, v3])

    for rotamer in rotamers:
        # Transform to global frame
        transformed = rotamer @ rot_matrix.T + origin

        # Calculate center of mass
        center = transformed.mean(axis=0)

        # Distance to target
        dist = np.linalg.norm(center - target_center)

        if dist < best_distance:
            best_distance = dist
            best_rotamer = rotamer

    return best_rotamer


def _place_sidechain_atoms(
    residue: Residue,
    rotamer: np.ndarray,
    origin: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
) -> None:
    """
    Place sidechain atoms from rotamer coordinates.

    Args:
        residue: Residue to modify
        rotamer: Rotamer coordinates in local frame
        origin, v1, v2, v3: Local coordinate frame
    """
    # Remove existing sidechain atoms
    residue.remove_sidechain()

    # Get atom names for this residue type
    if residue.res_type >= len(HEAVY_ATOMS):
        return

    atom_names = HEAVY_ATOMS.get(residue.res_type, [])

    # Rotation matrix from local to global
    rot_matrix = np.column_stack([v1, v2, v3])

    # Transform and place atoms
    for i, coords in enumerate(rotamer):
        if i >= len(atom_names):
            break

        # Transform to global frame
        global_coords = coords @ rot_matrix.T + origin

        atom = Atom(
            coords=global_coords,
            name=atom_names[i],
            num=0,
            flag=FLAG_SIDECHAIN,
        )
        residue.add_atom(atom)


def rebuild_sidechains_simple(
    molecule: Molecule,
    use_pdbsg: bool = False,
    verbose: bool = False,
) -> int:
    """
    Rebuild sidechains using default rotamer library.

    Args:
        molecule: Molecule to rebuild
        use_pdbsg: Whether to use input sidechain centers
        verbose: Whether to print progress

    Returns:
        Number of residues rebuilt
    """
    from pulchra.data.loader import load_rotamer_library

    rot_coords, rot_idx = load_rotamer_library()
    rot_library = RotamerLibrary(rot_coords, rot_idx)
    return rebuild_sidechains(molecule, rot_library, use_pdbsg, verbose)
