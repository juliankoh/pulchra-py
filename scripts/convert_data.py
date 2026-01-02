#!/usr/bin/env python3
"""
Convert PULCHRA C header data files to NumPy binary format.

This script parses the C header files containing backbone geometry
and rotamer statistics and converts them to efficient NPY format.

Usage:
    uv run python scripts/convert_data.py [--source-dir PATH] [--output-dir PATH]

Input files (from C source):
    - nco_data.h: Backbone geometry statistics (~22K lines)
    - rot_data_coords.h: Rotamer coordinates (~108K lines)
    - rot_data_idx.h: Rotamer indices (~9.9K lines)

Output files:
    - nco_stat.npy: General backbone statistics
    - nco_stat_pro.npy: Proline backbone statistics
    - rot_stat_coords.npy: Rotamer coordinates
    - rot_stat_idx.npy: Rotamer index table
"""

import argparse
import re
from pathlib import Path
import numpy as np


def parse_nco_data(header_path: Path) -> tuple:
    """
    Parse nco_struct arrays from nco_data.h.

    nco_struct has:
        int bins[3];      // r13_1, r13_2, r14 bin indices
        float data[8][3]; // 8 atoms x 3 coords (stored as flat array of 24 floats)

    Returns:
        Tuple of (nco_stat, nco_stat_pro) as numpy arrays
    """
    print(f"Parsing NCO data from {header_path}...")

    content = header_path.read_text()

    def parse_section(section_name: str) -> np.ndarray:
        """Parse a single nco_struct array section."""
        # Find the array declaration
        pattern = rf"nco_struct\s+{section_name}\s*\[\s*\]\s*=\s*\{{(.*?)\}}\s*;"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            return np.array([])

        section_content = match.group(1)

        # Pattern to match each struct entry:
        # {{bin1, bin2, bin3}, { float1, float2, ..., float24, }}
        # The floats are a flat array of 24 values
        entry_pattern = re.compile(
            r"\{\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*,\s*\{"
            r"\s*([-\d.,\s\neE+]+)"
            r"\s*\}\s*\}",
            re.MULTILINE
        )

        entries = []
        for match in entry_pattern.finditer(section_content):
            bins = [int(match.group(1)), int(match.group(2)), int(match.group(3))]

            # Parse the flat float array
            float_str = match.group(4)
            floats = re.findall(r'(-?[\d.]+(?:[eE][+-]?\d+)?)', float_str)
            coords = [float(f) for f in floats[:24]]  # Take first 24 values

            if len(coords) == 24:
                entry = bins + coords
                entries.append(entry)

        return np.array(entries, dtype=np.float32) if entries else np.array([])

    nco_stat = parse_section("nco_stat")
    print(f"  Parsed {len(nco_stat)} entries for nco_stat")

    nco_stat_pro = parse_section("nco_stat_pro")
    print(f"  Parsed {len(nco_stat_pro)} entries for nco_stat_pro")

    return nco_stat, nco_stat_pro


def parse_rot_coords(header_path: Path) -> np.ndarray:
    """
    Parse rot_stat_coords from rot_data_coords.h.

    Format: float rot_stat_coords[][3] = {{x, y, z}, ...};

    Returns:
        Numpy array of shape (N, 3)
    """
    print(f"Parsing rotamer coordinates from {header_path}...")

    content = header_path.read_text()

    # Pattern to match {x, y, z} entries
    coord_pattern = re.compile(
        r"\{\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\}"
    )

    coords = []
    for match in coord_pattern.finditer(content):
        x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
        coords.append([x, y, z])

    result = np.array(coords, dtype=np.float32)
    print(f"  Parsed {len(result)} coordinate entries")
    return result


def parse_rot_idx(header_path: Path) -> np.ndarray:
    """
    Parse rot_stat_idx from rot_data_idx.h.

    Format: int rot_stat_idx[][6] = {{aa_type, bin13_1, bin13_2, bin14, count, offset}, ...};

    Returns:
        Numpy array of shape (N, 6)
    """
    print(f"Parsing rotamer index from {header_path}...")

    content = header_path.read_text()

    # Pattern to match {int, int, int, int, int, int} entries
    idx_pattern = re.compile(
        r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}"
    )

    indices = []
    for match in idx_pattern.finditer(content):
        entry = [int(match.group(i)) for i in range(1, 7)]
        indices.append(entry)

    result = np.array(indices, dtype=np.int32)
    print(f"  Parsed {len(result)} index entries")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert PULCHRA C header data to NumPy format"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Directory containing C header files (default: project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "src" / "pulchra" / "data",
        help="Output directory for NPY files (default: src/pulchra/data)",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Parse NCO data
    nco_path = source_dir / "nco_data.h"
    if nco_path.exists():
        nco_stat, nco_stat_pro = parse_nco_data(nco_path)

        if len(nco_stat) > 0:
            out_path = output_dir / "nco_stat.npy"
            np.save(out_path, nco_stat)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        if len(nco_stat_pro) > 0:
            out_path = output_dir / "nco_stat_pro.npy"
            np.save(out_path, nco_stat_pro)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"WARNING: {nco_path} not found")

    print()

    # Parse rotamer coordinates
    rot_coords_path = source_dir / "rot_data_coords.h"
    if rot_coords_path.exists():
        rot_coords = parse_rot_coords(rot_coords_path)

        if len(rot_coords) > 0:
            out_path = output_dir / "rot_stat_coords.npy"
            np.save(out_path, rot_coords)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"WARNING: {rot_coords_path} not found")

    print()

    # Parse rotamer indices
    rot_idx_path = source_dir / "rot_data_idx.h"
    if rot_idx_path.exists():
        rot_idx = parse_rot_idx(rot_idx_path)

        if len(rot_idx) > 0:
            out_path = output_dir / "rot_stat_idx.npy"
            np.save(out_path, rot_idx)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"WARNING: {rot_idx_path} not found")

    print()
    print("Data conversion complete!")
    print()
    print("Summary of output files:")
    for npy_file in output_dir.glob("*.npy"):
        size_kb = npy_file.stat().st_size / 1024
        print(f"  {npy_file.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
