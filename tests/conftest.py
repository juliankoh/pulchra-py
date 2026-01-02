"""Pytest configuration and fixtures for PULCHRA tests."""

import numpy as np
import pytest

from pulchra.core.structures import Atom, Residue, Molecule
from pulchra.core.constants import FLAG_BACKBONE, FLAG_CALPHA, FLAG_SIDECHAIN


@pytest.fixture
def simple_ca_coords():
    """Simple alpha helix-like CA coordinates."""
    # Approximate alpha helix geometry
    coords = []
    for i in range(10):
        x = 1.5 * np.cos(i * 100 * np.pi / 180)
        y = 1.5 * np.sin(i * 100 * np.pi / 180)
        z = 1.5 * i
        coords.append([x, y, z])
    return np.array(coords)


@pytest.fixture
def simple_molecule(simple_ca_coords):
    """Create a simple molecule with CA atoms only."""
    mol = Molecule(name="test")
    sequence = "AAAAAAAAAA"  # 10 alanines

    for i, (coord, aa) in enumerate(zip(simple_ca_coords, sequence)):
        res = Residue(
            num=i + 1,
            res_type=1,  # ALA
            name="ALA",
            chain="A",
        )
        atom = Atom(
            coords=np.array(coord),
            name="CA ",
            num=i + 1,
            flag=FLAG_BACKBONE | FLAG_CALPHA,
        )
        res.add_atom(atom)
        mol.add_residue(res)

    mol.build_sequence_array()
    return mol


@pytest.fixture
def alanine_dipeptide():
    """Create an alanine dipeptide with full backbone."""
    mol = Molecule(name="ala2")

    # Residue 1 (N-terminus)
    res1 = Residue(num=1, res_type=1, name="ALA", chain="A")
    res1.add_atom(Atom(coords=np.array([0.0, 0.0, 0.0]), name="N  ", flag=FLAG_BACKBONE))
    res1.add_atom(Atom(coords=np.array([1.5, 0.0, 0.0]), name="CA ", flag=FLAG_BACKBONE | FLAG_CALPHA))
    res1.add_atom(Atom(coords=np.array([2.0, 1.4, 0.0]), name="C  ", flag=FLAG_BACKBONE))
    res1.add_atom(Atom(coords=np.array([1.3, 2.4, 0.0]), name="O  ", flag=FLAG_BACKBONE))
    res1.add_atom(Atom(coords=np.array([2.0, -1.0, 0.5]), name="CB ", flag=FLAG_SIDECHAIN))
    mol.add_residue(res1)

    # Residue 2 (C-terminus)
    res2 = Residue(num=2, res_type=1, name="ALA", chain="A")
    res2.add_atom(Atom(coords=np.array([3.3, 1.4, 0.0]), name="N  ", flag=FLAG_BACKBONE))
    res2.add_atom(Atom(coords=np.array([4.0, 2.7, 0.0]), name="CA ", flag=FLAG_BACKBONE | FLAG_CALPHA))
    res2.add_atom(Atom(coords=np.array([5.5, 2.7, 0.0]), name="C  ", flag=FLAG_BACKBONE))
    res2.add_atom(Atom(coords=np.array([6.2, 1.7, 0.0]), name="O  ", flag=FLAG_BACKBONE))
    res2.add_atom(Atom(coords=np.array([3.5, 3.5, 1.2]), name="CB ", flag=FLAG_SIDECHAIN))
    mol.add_residue(res2)

    mol.build_sequence_array()
    return mol
