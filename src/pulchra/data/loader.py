"""
Data loading utilities for statistical tables.

Loads precomputed backbone geometry and rotamer data from NPY files.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

# Cache for loaded data
_DATA_CACHE = {}


def get_data_path(filename: str) -> Path:
    """
    Get the path to a package data file.

    Args:
        filename: Name of the data file

    Returns:
        Path to the data file
    """
    try:
        # Python 3.9+
        from importlib.resources import files

        return files("pulchra.data") / filename
    except (ImportError, TypeError):
        # Fallback for older Python or when running from source
        return Path(__file__).parent / filename


def load_nco_stats() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load backbone geometry statistics (NCO templates).

    The NCO data contains pre-computed N, C, O positions for different
    CA-CA-CA geometry bins.

    Returns:
        Tuple of:
        - nco_stat: General backbone statistics, shape (N, bins+coords)
        - nco_stat_pro: Proline-specific statistics, shape (M, bins+coords)
    """
    if "nco_stat" in _DATA_CACHE:
        return _DATA_CACHE["nco_stat"], _DATA_CACHE["nco_stat_pro"]

    nco_path = get_data_path("nco_stat.npy")
    nco_pro_path = get_data_path("nco_stat_pro.npy")

    try:
        nco_stat = np.load(nco_path)
        nco_stat_pro = np.load(nco_pro_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"NCO data files not found. Please run the data conversion script first. "
            f"Expected files: {nco_path}, {nco_pro_path}"
        )

    _DATA_CACHE["nco_stat"] = nco_stat
    _DATA_CACHE["nco_stat_pro"] = nco_stat_pro

    return nco_stat, nco_stat_pro


def load_rotamer_library() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rotamer coordinate library.

    The rotamer data contains pre-computed sidechain conformations
    indexed by residue type and CA geometry bins.

    Returns:
        Tuple of:
        - rot_coords: Rotamer coordinates, shape (N, 3)
        - rot_idx: Rotamer index table, shape (M, 6)
                   Each row: [aa_type, bin13_1, bin13_2, bin14, count, offset]
    """
    if "rot_coords" in _DATA_CACHE:
        return _DATA_CACHE["rot_coords"], _DATA_CACHE["rot_idx"]

    coords_path = get_data_path("rot_stat_coords.npy")
    idx_path = get_data_path("rot_stat_idx.npy")

    try:
        rot_coords = np.load(coords_path)
        rot_idx = np.load(idx_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Rotamer data files not found. Please run the data conversion script first. "
            f"Expected files: {coords_path}, {idx_path}"
        )

    _DATA_CACHE["rot_coords"] = rot_coords
    _DATA_CACHE["rot_idx"] = rot_idx

    return rot_coords, rot_idx


def clear_cache():
    """Clear the data cache to free memory."""
    _DATA_CACHE.clear()


def get_nco_template(
    nco_stat: np.ndarray,
    bin13_1: int,
    bin13_2: int,
    bin14: int,
) -> Optional[np.ndarray]:
    """
    Look up an NCO template by geometry bins.

    Args:
        nco_stat: NCO statistics array
        bin13_1: First r13 distance bin
        bin13_2: Second r13 distance bin
        bin14: r14 (chiral distance) bin

    Returns:
        Template coordinates (8 atoms x 3 coords) or None if not found
    """
    # NCO stat structure: [bin13_1, bin13_2, bin14, ...coords...]
    # Find matching entry
    for entry in nco_stat:
        if (
            int(entry[0]) == bin13_1
            and int(entry[1]) == bin13_2
            and int(entry[2]) == bin14
        ):
            # Return the coordinate data (reshape to 8x3)
            return entry[3:].reshape(8, 3)

    return None


def get_rotamers_for_residue(
    rot_idx: np.ndarray,
    rot_coords: np.ndarray,
    res_type: int,
    bin13_1: int,
    bin13_2: int,
    bin14: int,
    max_rotamers: int = 10,
) -> list:
    """
    Get rotamer conformations for a residue type and geometry bins.

    Args:
        rot_idx: Rotamer index table
        rot_coords: Rotamer coordinates
        res_type: Amino acid type (0-19)
        bin13_1: First r13 distance bin
        bin13_2: Second r13 distance bin
        bin14: r14 bin
        max_rotamers: Maximum number of rotamers to return

    Returns:
        List of rotamer coordinate arrays
    """
    from pulchra.core.constants import NHEAVY

    rotamers = []
    n_atoms = NHEAVY[res_type] if res_type < len(NHEAVY) else 0

    if n_atoms == 0:
        return rotamers

    # Search for matching entries in rot_idx
    for entry in rot_idx:
        if (
            int(entry[0]) == res_type
            and int(entry[1]) == bin13_1
            and int(entry[2]) == bin13_2
            and int(entry[3]) == bin14
        ):
            count = int(entry[4])
            offset = int(entry[5])

            # Extract rotamer coordinates
            for i in range(min(count, max_rotamers)):
                start_idx = offset + i * n_atoms
                end_idx = start_idx + n_atoms
                if end_idx <= len(rot_coords):
                    rot_atoms = rot_coords[start_idx:end_idx].copy()
                    rotamers.append(rot_atoms)

            break

    return rotamers


def data_files_exist() -> bool:
    """
    Check if all required data files exist.

    Returns:
        True if all data files exist, False otherwise
    """
    required_files = [
        "nco_stat.npy",
        "nco_stat_pro.npy",
        "rot_stat_coords.npy",
        "rot_stat_idx.npy",
    ]

    for filename in required_files:
        path = get_data_path(filename)
        if not Path(path).exists():
            return False

    return True
