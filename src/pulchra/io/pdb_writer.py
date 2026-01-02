"""
PDB file writer.

Source: pulchra.c lines 1463-1568 (write_pdb, write_pdb_sg)
"""

from pathlib import Path
from typing import Union

from pulchra.core.structures import Molecule, Residue, Atom
from pulchra.core.constants import FLAG_BACKBONE, FLAG_SIDECHAIN, BACKBONE_ATOMS


def write_pdb(
    filename: Union[str, Path],
    molecule: Molecule,
    rearrange_backbone: bool = False,
    include_hydrogens: bool = False,
) -> None:
    """
    Write a Molecule to PDB format.

    Source: pulchra.c lines 1463-1534

    Args:
        filename: Output file path
        molecule: Molecule to write
        rearrange_backbone: If True, write backbone atoms in AMBER order (N, CA, C, O)
        include_hydrogens: If True, include hydrogen atoms (if present)
    """
    with open(filename, "w") as f:
        atom_num = 0

        for residue in molecule.residues:
            atoms_to_write = []

            if rearrange_backbone:
                # AMBER order: N, CA, C, O, then sidechains
                for bb_name in ["N  ", "CA ", "C  ", "O  "]:
                    atom = residue.get_atom(bb_name)
                    if atom is not None:
                        atoms_to_write.append(atom)

                # Add sidechain atoms
                for atom in residue.atoms:
                    if atom.is_sidechain:
                        atoms_to_write.append(atom)

                # Add OXT if present (terminal oxygen)
                oxt = residue.get_atom("OXT")
                if oxt is not None:
                    atoms_to_write.append(oxt)
            else:
                # Write atoms in order they appear
                atoms_to_write = residue.atoms

            for atom in atoms_to_write:
                # Skip hydrogens if not requested
                if not include_hydrogens and atom.name.strip().startswith("H"):
                    continue

                atom_num += 1

                line = _format_atom_line(
                    atom_num=atom_num,
                    atom_name=atom.name,
                    res_name=residue.name,
                    chain=residue.chain,
                    res_num=residue.num,
                    x=atom.x,
                    y=atom.y,
                    z=atom.z,
                    insertion_code=residue.insertion_code,
                )
                f.write(line + "\n")

        f.write("END\n")


def write_pdb_sg(
    filename: Union[str, Path],
    molecule: Molecule,
) -> None:
    """
    Write a Molecule to PDB-SG format (CA + sidechain center of mass).

    Source: pulchra.c lines 1536-1568

    Args:
        filename: Output file path
        molecule: Molecule to write
    """
    with open(filename, "w") as f:
        atom_num = 0

        for residue in molecule.residues:
            # Write CA atom
            ca = residue.ca
            if ca is not None:
                atom_num += 1
                line = _format_atom_line(
                    atom_num=atom_num,
                    atom_name="CA ",
                    res_name=residue.name,
                    chain=residue.chain,
                    res_num=residue.num,
                    x=ca.x,
                    y=ca.y,
                    z=ca.z,
                )
                f.write(line + "\n")

            # Calculate and write sidechain center of mass
            sg_center = residue.calculate_sidechain_center()
            if sg_center is not None:
                atom_num += 1
                line = _format_atom_line(
                    atom_num=atom_num,
                    atom_name="SC ",
                    res_name=residue.name,
                    chain=residue.chain,
                    res_num=residue.num,
                    x=sg_center[0],
                    y=sg_center[1],
                    z=sg_center[2],
                )
                f.write(line + "\n")

        f.write("END\n")


def _format_atom_line(
    atom_num: int,
    atom_name: str,
    res_name: str,
    chain: str,
    res_num: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.0,
    temp_factor: float = 0.0,
    element: str = "",
    insertion_code: str = "",
) -> str:
    """
    Format a single ATOM line in strict PDB format.

    Uses exact column positions per PDB specification for compatibility
    with ProDy, ProteinMPNN, and other strict PDB parsers.

    PDB format specification:
    COLUMNS        DATA TYPE       CONTENTS
    --------------------------------------------------------------------------------
     1 -  6        Record name     "ATOM  "
     7 - 11        Integer         Atom serial number
    13 - 16        Atom            Atom name
    17             Character       Alternate location indicator
    18 - 20        Residue name    Residue name
    22             Character       Chain identifier
    23 - 26        Integer         Residue sequence number
    27             AChar           Code for insertion of residues
    31 - 38        Real(8.3)       X coordinate
    39 - 46        Real(8.3)       Y coordinate
    47 - 54        Real(8.3)       Z coordinate
    55 - 60        Real(6.2)       Occupancy
    61 - 66        Real(6.2)       Temperature factor
    77 - 78        LString(2)      Element symbol
    79 - 80        LString(2)      Charge on the atom
    """
    # Strip and format atom name for cols 13-16 (4 chars total)
    name = atom_name.strip()
    if len(name) < 4:
        # Standard atoms (N, CA, CB, etc.): leading space, left-justify rest
        atom_name_fmt = f" {name:<3}"
    else:
        # 4-char names (e.g., some hydrogens): left-justify
        atom_name_fmt = f"{name:<4}"

    # Strip and format residue name for cols 18-20
    res_name_fmt = res_name.strip()

    # Determine element from atom name if not provided
    if not element:
        element = name[0] if name else "X"

    # Build line with exact column positions
    # fmt: off
    line = (
        f"ATOM  "                              # 1-6:   Record name
        f"{atom_num:>5d} "                     # 7-11:  Serial number + col 12 space
        f"{atom_name_fmt}"                     # 13-16: Atom name
        f" "                                   # 17:    AltLoc (blank)
        f"{res_name_fmt:<3} "                  # 18-20: ResName + col 21 space
        f"{chain:1}"                           # 22:    Chain ID
        f"{res_num:>4d}"                       # 23-26: Residue sequence number
        f"{insertion_code or ' ':1}"           # 27:    Insertion code
        f"   "                                 # 28-30: Blank
        f"{x:>8.3f}"                           # 31-38: X coordinate
        f"{y:>8.3f}"                           # 39-46: Y coordinate
        f"{z:>8.3f}"                           # 47-54: Z coordinate
        f"{occupancy:>6.2f}"                   # 55-60: Occupancy
        f"{temp_factor:>6.2f}"                 # 61-66: Temperature factor
        f"          "                          # 67-76: Blank
        f"{element:>2}"                        # 77-78: Element symbol
    )
    # fmt: on

    return line


def molecule_to_pdb_string(
    molecule: Molecule,
    rearrange_backbone: bool = False,
    include_hydrogens: bool = False,
) -> str:
    """
    Convert a Molecule to a PDB format string.

    Args:
        molecule: Molecule to convert
        rearrange_backbone: If True, write backbone atoms in AMBER order
        include_hydrogens: If True, include hydrogen atoms

    Returns:
        PDB format string
    """
    lines = []
    atom_num = 0

    for residue in molecule.residues:
        atoms_to_write = []

        if rearrange_backbone:
            for bb_name in ["N  ", "CA ", "C  ", "O  "]:
                atom = residue.get_atom(bb_name)
                if atom is not None:
                    atoms_to_write.append(atom)
            for atom in residue.atoms:
                if atom.is_sidechain:
                    atoms_to_write.append(atom)
            oxt = residue.get_atom("OXT")
            if oxt is not None:
                atoms_to_write.append(oxt)
        else:
            atoms_to_write = residue.atoms

        for atom in atoms_to_write:
            if not include_hydrogens and atom.name.strip().startswith("H"):
                continue

            atom_num += 1
            line = _format_atom_line(
                atom_num=atom_num,
                atom_name=atom.name,
                res_name=residue.name,
                chain=residue.chain,
                res_num=residue.num,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                insertion_code=residue.insertion_code,
            )
            lines.append(line)

    lines.append("END")
    return "\n".join(lines)
