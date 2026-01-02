"""
PDB file parser using BioPython.

Source: pulchra.c lines 669-846 (read_pdb_file)
"""

from pathlib import Path
from typing import Union
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure

from pulchra.core.structures import Atom, Residue, Molecule
from pulchra.core.constants import (
    FLAG_BACKBONE,
    FLAG_CALPHA,
    FLAG_SIDECHAIN,
    FLAG_SCM,
    FLAG_INITIAL,
    BACKBONE_ATOMS,
    get_residue_type,
    MODIFIED_RESIDUES,
)


def read_pdb_file(
    filename: Union[str, Path],
    use_pdbsg: bool = False,
    model_id: int = 0,
) -> Molecule:
    """
    Read a PDB file and convert to internal Molecule representation.

    Source: pulchra.c lines 669-846

    Args:
        filename: Path to PDB file
        use_pdbsg: If True, expect PDB-SG format (CA + SC center pseudo-atoms)
        model_id: Which model to read (0 = first model)

    Returns:
        Molecule object containing the structure
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(filename))

    return _convert_biopython_structure(structure, use_pdbsg, model_id)


def _convert_biopython_structure(
    structure: Structure,
    use_pdbsg: bool = False,
    model_id: int = 0,
) -> Molecule:
    """
    Convert BioPython Structure to internal Molecule class.

    Args:
        structure: BioPython Structure object
        use_pdbsg: If True, look for SC/CM pseudo-atoms for sidechain centers
        model_id: Which model to use

    Returns:
        Molecule object
    """
    mol = Molecule(name=structure.id)

    # Get the specified model
    models = list(structure.get_models())
    if model_id >= len(models):
        model_id = 0
    model = models[model_id]

    atom_num = 0

    for chain in model:
        for bio_residue in chain:
            # Skip hetero atoms (water, ligands) unless they're modified residues
            hetfield = bio_residue.get_id()[0]
            if hetfield.startswith("H_") or hetfield == "W":
                # Check if it's a known modified residue
                res_name = bio_residue.get_resname().strip()
                if res_name not in MODIFIED_RESIDUES:
                    continue

            res_name = bio_residue.get_resname().strip()
            res_num = bio_residue.get_id()[1]
            insertion_code = bio_residue.get_id()[2].strip()
            chain_id = chain.get_id()

            # Get residue type
            res_type = get_residue_type(res_name)

            # Create residue
            residue = Residue(
                num=res_num,
                res_type=res_type,
                name=res_name,
                chain=chain_id,
                insertion_code=insertion_code,
                protein=True,
            )

            sg_center = None

            # Process atoms
            for bio_atom in bio_residue:
                atom_name = bio_atom.get_name()
                # Pad atom name to 3 characters
                atom_name_padded = atom_name.ljust(3)[:3]

                # Handle alternate conformations - take 'A' or first
                altloc = bio_atom.get_altloc()
                if altloc and altloc not in (" ", "A"):
                    continue

                coords = bio_atom.get_coord()

                # Determine atom flags
                flag = FLAG_INITIAL

                if atom_name_padded in BACKBONE_ATOMS:
                    flag |= FLAG_BACKBONE
                    if atom_name_padded.strip() == "CA":
                        flag |= FLAG_CALPHA
                else:
                    # Check for SC/CM pseudo-atom (PDB-SG format)
                    if use_pdbsg and atom_name.strip() in ("SC", "CM"):
                        sg_center = coords.copy()
                        flag |= FLAG_SCM
                        residue.pdbsg = True
                    else:
                        flag |= FLAG_SIDECHAIN

                atom_num += 1
                atom = Atom(
                    coords=np.array(coords, dtype=np.float64),
                    name=atom_name_padded,
                    num=atom_num,
                    flag=flag,
                )

                residue.add_atom(atom)

            # Store sidechain center if found
            if sg_center is not None:
                residue.sg_center = np.array(sg_center, dtype=np.float64)

            # Only add residue if it has a CA atom (valid protein residue)
            if residue.ca is not None:
                mol.add_residue(residue)

    # Build sequence array
    mol.build_sequence_array()

    return mol


def read_ca_only(
    filename: Union[str, Path],
    sequence: str = None,
) -> Molecule:
    """
    Read a PDB file containing only CA atoms.

    This is useful for reading coarse-grained models or
    C-alpha traces that need reconstruction.

    Args:
        filename: Path to PDB file
        sequence: Optional sequence string. If not provided, will use
                 residue names from PDB.

    Returns:
        Molecule object with CA atoms only
    """
    mol = read_pdb_file(filename, use_pdbsg=False)

    # If sequence is provided, update residue types
    if sequence:
        from pulchra.core.constants import SHORT_TO_INDEX

        for i, res in enumerate(mol.residues):
            if i < len(sequence):
                letter = sequence[i].upper()
                res.res_type = SHORT_TO_INDEX.get(letter, 20)

        mol.build_sequence_array()

    return mol


def read_coords_from_array(
    ca_coords: np.ndarray,
    sequence: str,
    chain_id: str = "A",
) -> Molecule:
    """
    Create a Molecule from raw CA coordinates and sequence.

    Args:
        ca_coords: Array of CA coordinates, shape (N, 3)
        sequence: Amino acid sequence (1-letter codes)
        chain_id: Chain identifier

    Returns:
        Molecule object with CA atoms
    """
    from pulchra.core.constants import SHORT_TO_INDEX, AA_NAMES

    mol = Molecule(name="from_coords")

    for i, (coord, aa) in enumerate(zip(ca_coords, sequence)):
        res_type = SHORT_TO_INDEX.get(aa.upper(), 20)
        res_name = AA_NAMES[res_type]

        residue = Residue(
            num=i + 1,
            res_type=res_type,
            name=res_name,
            chain=chain_id,
            protein=True,
        )

        atom = Atom(
            coords=np.array(coord, dtype=np.float64),
            name="CA ",
            num=i + 1,
            flag=FLAG_BACKBONE | FLAG_CALPHA | FLAG_INITIAL,
        )

        residue.add_atom(atom)
        mol.add_residue(residue)

    mol.build_sequence_array()

    return mol
