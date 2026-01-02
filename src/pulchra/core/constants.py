"""
Constants and definitions ported from pulchra.c.

Source: pulchra.c lines 45-330
"""

import math

# Version
PULCHRA_VERSION = "4.0.0"

# Flags for atom types (lines 45-49)
FLAG_BACKBONE = 1
FLAG_CALPHA = 2
FLAG_SIDECHAIN = 4
FLAG_SCM = 8
FLAG_INITIAL = 16

# Molecule type flags (lines 51-54)
FLAG_PROTEIN = 1
FLAG_DNA = 2
FLAG_RNA = 4
FLAG_CHYDRO = 8

# Angle conversions (lines 56-57)
RADDEG = 180.0 / math.pi
DEGRAD = math.pi / 180.0

# Force field constants (lines 86-95)
CA_K = 10.0           # C-alpha bond force constant
CA_ANGLE_K = 20.0     # C-alpha angle force constant
CA_START_K = 0.01     # Restraint to initial coords
CA_XVOL_K = 10.0      # Excluded volume force constant

CA_DIST = 3.8         # Ideal CA-CA distance (Angstroms)
CA_DIST_TOL = 0.1     # Tolerance
CA_DIST_CISPRO = 2.9  # CA-CA distance for cis-proline
CA_DIST_CISPRO_TOL = 0.1

# Distance thresholds (lines 77-79)
CA_START_DIST = 3.0
CA_XVOL_DIST = 3.5
SG_XVOL_DIST = 1.6

# Grid resolution for clash detection (line 114)
GRID_RES = 6.0

# Energy epsilon (line 95)
E_EPS = 1e-10

# Default iteration counts
CA_ITER = 100
XVOL_ITER = 3

# Amino acid names (lines 118-120)
AA_NAMES = [
    "GLY", "ALA", "SER", "CYS", "VAL", "THR", "ILE",
    "PRO", "MET", "ASP", "ASN", "LEU", "LYS", "GLU",
    "GLN", "ARG", "HIS", "PHE", "TYR", "TRP", "UNK"
]

# Single letter codes (line 122)
SHORT_AA_NAMES = "GASCVTIPMDNLKEQRHFYWX"

# Number of heavy sidechain atoms per residue type (line 126)
NHEAVY = [0, 1, 2, 2, 3, 3, 4, 3, 4, 4, 4, 4, 5, 5, 5, 7, 6, 7, 8, 10]

# Backbone atom names (line 128)
BACKBONE_ATOMS = ["N  ", "CA ", "C  ", "O  "]

# Heavy sidechain atoms per residue type (lines 130-330)
# Indexed by residue type (0-19), each entry is a list of atom names
HEAVY_ATOMS = {
    0: [],  # GLY - no sidechain
    1: ["CB "],  # ALA
    2: ["CB ", "OG "],  # SER
    3: ["CB ", "SG "],  # CYS
    4: ["CB ", "CG1", "CG2"],  # VAL
    5: ["CB ", "OG1", "CG2"],  # THR
    6: ["CB ", "CG1", "CG2", "CD1"],  # ILE
    7: ["CB ", "CG ", "CD "],  # PRO
    8: ["CB ", "CG ", "SD ", "CE "],  # MET
    9: ["CB ", "CG ", "OD1", "OD2"],  # ASP
    10: ["CB ", "CG ", "OD1", "ND2"],  # ASN
    11: ["CB ", "CG ", "CD1", "CD2"],  # LEU
    12: ["CB ", "CG ", "CD ", "CE ", "NZ "],  # LYS
    13: ["CB ", "CG ", "CD ", "OE1", "OE2"],  # GLU
    14: ["CB ", "CG ", "CD ", "OE1", "NE2"],  # GLN
    15: ["CB ", "CG ", "CD ", "NE ", "CZ ", "NH1", "NH2"],  # ARG
    16: ["CB ", "CG ", "ND1", "CD2", "CE1", "NE2"],  # HIS
    17: ["CB ", "CG ", "CD1", "CD2", "CE1", "CE2", "CZ "],  # PHE
    18: ["CB ", "CG ", "CD1", "CD2", "CE1", "CE2", "CZ ", "OH "],  # TYR
    19: ["CB ", "CG ", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # TRP
}

# Mapping from 3-letter code to type index
AA_TO_INDEX = {name: i for i, name in enumerate(AA_NAMES)}

# Mapping from single letter to type index
SHORT_TO_INDEX = {letter: i for i, letter in enumerate(SHORT_AA_NAMES)}

# Common modified residues to standard residue mapping
MODIFIED_RESIDUES = {
    "MSE": "MET",  # Selenomethionine
    "TPO": "THR",  # Phosphothreonine
    "SEP": "SER",  # Phosphoserine
    "PTR": "TYR",  # Phosphotyrosine
    "CSO": "CYS",  # S-hydroxycysteine
    "HYP": "PRO",  # Hydroxyproline
    "MLY": "LYS",  # N-dimethyl-lysine
    "M3L": "LYS",  # N-trimethyl-lysine
}


def get_residue_type(name: str) -> int:
    """
    Get the residue type index for a given 3-letter residue name.

    Args:
        name: 3-letter residue code (e.g., "ALA", "GLY")

    Returns:
        Residue type index (0-19), or 20 for unknown
    """
    name = name.strip().upper()

    # Check for modified residues
    if name in MODIFIED_RESIDUES:
        name = MODIFIED_RESIDUES[name]

    return AA_TO_INDEX.get(name, 20)


def get_one_letter(three_letter: str) -> str:
    """Convert 3-letter amino acid code to 1-letter code."""
    idx = get_residue_type(three_letter)
    if idx < len(SHORT_AA_NAMES):
        return SHORT_AA_NAMES[idx]
    return "X"


def get_three_letter(one_letter: str) -> str:
    """Convert 1-letter amino acid code to 3-letter code."""
    idx = SHORT_TO_INDEX.get(one_letter.upper(), 20)
    return AA_NAMES[idx]
