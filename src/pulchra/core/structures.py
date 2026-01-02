"""
Core data structures for molecular representation.

Source: pulchra.c lines 334-372 (atom_type, res_type, mol_type structs)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from pulchra.core.constants import FLAG_BACKBONE, FLAG_CALPHA, FLAG_SIDECHAIN


@dataclass
class Atom:
    """
    Represents a single atom in a protein structure.

    Replaces atom_type struct from pulchra.c (lines 336-346).
    """

    coords: np.ndarray  # [x, y, z] coordinates
    name: str  # Atom name (e.g., "CA ", "N  ", "CB ")
    num: int = 0  # Atom serial number
    locnum: int = 0  # Local atom number within residue
    flag: int = 0  # Atom type flags (FLAG_BACKBONE, FLAG_CALPHA, etc.)
    cispro: bool = False  # True if preceding a cis-proline
    grid_pos: Optional[tuple] = None  # (gx, gy, gz) grid position for clash detection

    def __post_init__(self):
        """Ensure coords is a numpy array."""
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords, dtype=np.float64)

    @property
    def x(self) -> float:
        return self.coords[0]

    @x.setter
    def x(self, value: float):
        self.coords[0] = value

    @property
    def y(self) -> float:
        return self.coords[1]

    @y.setter
    def y(self, value: float):
        self.coords[1] = value

    @property
    def z(self) -> float:
        return self.coords[2]

    @z.setter
    def z(self, value: float):
        self.coords[2] = value

    @property
    def is_backbone(self) -> bool:
        return bool(self.flag & FLAG_BACKBONE)

    @property
    def is_calpha(self) -> bool:
        return bool(self.flag & FLAG_CALPHA)

    @property
    def is_sidechain(self) -> bool:
        return bool(self.flag & FLAG_SIDECHAIN)

    def distance_to(self, other: "Atom") -> float:
        """Calculate distance to another atom."""
        return np.linalg.norm(self.coords - other.coords)

    def copy(self) -> "Atom":
        """Create a copy of this atom."""
        return Atom(
            coords=self.coords.copy(),
            name=self.name,
            num=self.num,
            locnum=self.locnum,
            flag=self.flag,
            cispro=self.cispro,
            grid_pos=self.grid_pos,
        )


@dataclass
class Residue:
    """
    Represents a single amino acid residue.

    Replaces res_type struct from pulchra.c (lines 348-360).
    """

    atoms: list = field(default_factory=list)  # List of Atom objects
    num: int = 0  # PDB residue number
    locnum: int = 0  # Sequential index (0-based)
    res_type: int = 20  # Residue type index (0-19 for standard, 20 for unknown)
    name: str = "UNK"  # 3-letter residue name
    chain: str = "A"  # Chain identifier
    sg_center: Optional[np.ndarray] = None  # Sidechain center of mass [sgx, sgy, sgz]
    cm: Optional[np.ndarray] = None  # Full residue center of mass [cmx, cmy, cmz]
    protein: bool = True  # Is this a protein residue?
    pdbsg: bool = False  # Input had PDB-SG format (CA + SC center)
    insertion_code: str = ""  # PDB insertion code

    def __post_init__(self):
        """Ensure sg_center and cm are numpy arrays if provided."""
        if self.sg_center is not None and not isinstance(self.sg_center, np.ndarray):
            self.sg_center = np.array(self.sg_center, dtype=np.float64)
        if self.cm is not None and not isinstance(self.cm, np.ndarray):
            self.cm = np.array(self.cm, dtype=np.float64)

    @property
    def natoms(self) -> int:
        """Number of atoms in this residue."""
        return len(self.atoms)

    def get_atom(self, name: str) -> Optional[Atom]:
        """
        Get an atom by name.

        Args:
            name: Atom name (e.g., "CA ", "N  ")

        Returns:
            Atom object or None if not found
        """
        # Pad name to 3 characters for comparison
        padded = name.ljust(3)[:3]
        for atom in self.atoms:
            if atom.name.strip() == padded.strip():
                return atom
        return None

    @property
    def ca(self) -> Optional[Atom]:
        """Get the C-alpha atom."""
        return self.get_atom("CA")

    @property
    def n(self) -> Optional[Atom]:
        """Get the nitrogen atom."""
        return self.get_atom("N")

    @property
    def c(self) -> Optional[Atom]:
        """Get the carbonyl carbon atom."""
        return self.get_atom("C")

    @property
    def o(self) -> Optional[Atom]:
        """Get the carbonyl oxygen atom."""
        return self.get_atom("O")

    @property
    def cb(self) -> Optional[Atom]:
        """Get the C-beta atom."""
        return self.get_atom("CB")

    def get_backbone_atoms(self) -> list:
        """Get all backbone atoms (N, CA, C, O)."""
        return [a for a in self.atoms if a.is_backbone]

    def get_sidechain_atoms(self) -> list:
        """Get all sidechain atoms."""
        return [a for a in self.atoms if a.is_sidechain]

    def add_atom(self, atom: Atom):
        """Add an atom to this residue."""
        atom.locnum = len(self.atoms)
        self.atoms.append(atom)

    def remove_sidechain(self):
        """Remove all sidechain atoms."""
        self.atoms = [a for a in self.atoms if not a.is_sidechain]

    def calculate_center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of all atoms."""
        if not self.atoms:
            return np.zeros(3)
        coords = np.array([a.coords for a in self.atoms])
        return coords.mean(axis=0)

    def calculate_sidechain_center(self) -> Optional[np.ndarray]:
        """Calculate the center of mass of sidechain atoms."""
        sc_atoms = self.get_sidechain_atoms()
        if not sc_atoms:
            return None
        coords = np.array([a.coords for a in sc_atoms])
        return coords.mean(axis=0)


@dataclass
class Molecule:
    """
    Represents a protein molecule (chain).

    Replaces mol_type struct from pulchra.c (lines 362-372).
    """

    residues: list = field(default_factory=list)  # List of Residue objects
    name: str = ""  # Molecule/chain name
    sequence: Optional[np.ndarray] = None  # Array of residue type indices

    @property
    def nres(self) -> int:
        """Number of residues."""
        return len(self.residues)

    @property
    def natoms(self) -> int:
        """Total number of atoms."""
        return sum(r.natoms for r in self.residues)

    def get_residue(self, num: int, chain: str = None) -> Optional[Residue]:
        """
        Get a residue by PDB number.

        Args:
            num: PDB residue number
            chain: Optional chain ID filter

        Returns:
            Residue object or None if not found
        """
        for res in self.residues:
            if res.num == num:
                if chain is None or res.chain == chain:
                    return res
        return None

    def get_ca_coords(self) -> np.ndarray:
        """
        Get array of all C-alpha coordinates.

        Returns:
            (N, 3) numpy array of CA coordinates
        """
        coords = []
        for res in self.residues:
            ca = res.ca
            if ca is not None:
                coords.append(ca.coords)
        return np.array(coords)

    def set_ca_coords(self, coords: np.ndarray):
        """
        Set C-alpha coordinates from array.

        Args:
            coords: (N, 3) numpy array of CA coordinates
        """
        idx = 0
        for res in self.residues:
            ca = res.ca
            if ca is not None and idx < len(coords):
                ca.coords = coords[idx].copy()
                idx += 1

    def get_sequence_string(self) -> str:
        """Get the amino acid sequence as a string."""
        from pulchra.core.constants import SHORT_AA_NAMES

        seq = []
        for res in self.residues:
            if res.res_type < len(SHORT_AA_NAMES):
                seq.append(SHORT_AA_NAMES[res.res_type])
            else:
                seq.append("X")
        return "".join(seq)

    def build_sequence_array(self):
        """Build the sequence array from residues."""
        self.sequence = np.array([r.res_type for r in self.residues], dtype=np.int32)

    def add_residue(self, residue: Residue):
        """Add a residue to this molecule."""
        residue.locnum = len(self.residues)
        self.residues.append(residue)

    def get_all_atoms(self) -> list:
        """Get a flat list of all atoms."""
        atoms = []
        for res in self.residues:
            atoms.extend(res.atoms)
        return atoms

    def center_to_origin(self):
        """Translate the molecule so its center of mass is at the origin."""
        all_coords = []
        for res in self.residues:
            for atom in res.atoms:
                all_coords.append(atom.coords)

        if not all_coords:
            return

        center = np.mean(all_coords, axis=0)

        for res in self.residues:
            for atom in res.atoms:
                atom.coords -= center

    def copy(self) -> "Molecule":
        """Create a deep copy of this molecule."""
        new_mol = Molecule(name=self.name)
        for res in self.residues:
            new_res = Residue(
                num=res.num,
                locnum=res.locnum,
                res_type=res.res_type,
                name=res.name,
                chain=res.chain,
                sg_center=res.sg_center.copy() if res.sg_center is not None else None,
                cm=res.cm.copy() if res.cm is not None else None,
                protein=res.protein,
                pdbsg=res.pdbsg,
            )
            for atom in res.atoms:
                new_res.atoms.append(atom.copy())
            new_mol.residues.append(new_res)

        if self.sequence is not None:
            new_mol.sequence = self.sequence.copy()

        return new_mol
