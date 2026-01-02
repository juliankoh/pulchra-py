"""Tests for core module."""

import numpy as np
import pytest

from pulchra.core.structures import Atom, Residue, Molecule
from pulchra.core.constants import (
    AA_NAMES,
    SHORT_AA_NAMES,
    NHEAVY,
    get_residue_type,
    get_one_letter,
)
from pulchra.core.geometry import (
    calc_distance,
    calc_r14,
    normalize,
    calc_torsion,
    calc_angle,
)


class TestConstants:
    """Test constants module."""

    def test_aa_names_count(self):
        assert len(AA_NAMES) == 21  # 20 standard + UNK

    def test_short_aa_names_count(self):
        assert len(SHORT_AA_NAMES) == 21

    def test_nheavy_count(self):
        assert len(NHEAVY) == 20

    def test_get_residue_type(self):
        assert get_residue_type("ALA") == 1
        assert get_residue_type("GLY") == 0
        assert get_residue_type("TRP") == 19
        assert get_residue_type("XYZ") == 20  # Unknown

    def test_get_one_letter(self):
        assert get_one_letter("ALA") == "A"
        assert get_one_letter("GLY") == "G"

    def test_modified_residue_mapping(self):
        assert get_residue_type("MSE") == get_residue_type("MET")


class TestAtom:
    """Test Atom class."""

    def test_creation(self):
        coords = np.array([1.0, 2.0, 3.0])
        atom = Atom(coords=coords, name="CA ")
        assert np.allclose(atom.coords, coords)
        assert atom.name == "CA "

    def test_xyz_properties(self):
        atom = Atom(coords=np.array([1.0, 2.0, 3.0]), name="CA ")
        assert atom.x == 1.0
        assert atom.y == 2.0
        assert atom.z == 3.0

    def test_xyz_setters(self):
        atom = Atom(coords=np.array([1.0, 2.0, 3.0]), name="CA ")
        atom.x = 5.0
        assert atom.x == 5.0
        assert atom.coords[0] == 5.0

    def test_distance_to(self):
        a1 = Atom(coords=np.array([0.0, 0.0, 0.0]), name="CA ")
        a2 = Atom(coords=np.array([3.0, 4.0, 0.0]), name="CA ")
        assert a1.distance_to(a2) == pytest.approx(5.0)

    def test_copy(self):
        atom = Atom(coords=np.array([1.0, 2.0, 3.0]), name="CA ", num=10)
        copied = atom.copy()
        assert np.allclose(copied.coords, atom.coords)
        assert copied.name == atom.name
        # Verify it's a true copy
        copied.coords[0] = 999
        assert atom.coords[0] == 1.0


class TestResidue:
    """Test Residue class."""

    def test_creation(self):
        res = Residue(num=1, name="ALA", res_type=1)
        assert res.num == 1
        assert res.name == "ALA"

    def test_add_atom(self):
        res = Residue(num=1, name="ALA")
        atom = Atom(coords=np.array([0, 0, 0]), name="CA ")
        res.add_atom(atom)
        assert res.natoms == 1
        assert atom.locnum == 0

    def test_get_atom(self):
        res = Residue(num=1, name="ALA")
        ca = Atom(coords=np.array([0, 0, 0]), name="CA ")
        n = Atom(coords=np.array([1, 0, 0]), name="N  ")
        res.add_atom(ca)
        res.add_atom(n)

        assert res.get_atom("CA") is ca
        assert res.get_atom("N") is n
        assert res.get_atom("XYZ") is None

    def test_ca_property(self):
        res = Residue(num=1, name="ALA")
        ca = Atom(coords=np.array([0, 0, 0]), name="CA ")
        res.add_atom(ca)
        assert res.ca is ca


class TestMolecule:
    """Test Molecule class."""

    def test_creation(self):
        mol = Molecule(name="test")
        assert mol.name == "test"
        assert mol.nres == 0

    def test_add_residue(self, simple_molecule):
        assert simple_molecule.nres == 10

    def test_get_ca_coords(self, simple_molecule, simple_ca_coords):
        coords = simple_molecule.get_ca_coords()
        assert coords.shape == simple_ca_coords.shape
        assert np.allclose(coords, simple_ca_coords)

    def test_set_ca_coords(self, simple_molecule):
        new_coords = np.random.randn(10, 3)
        simple_molecule.set_ca_coords(new_coords)
        result = simple_molecule.get_ca_coords()
        assert np.allclose(result, new_coords)

    def test_get_sequence_string(self, simple_molecule):
        seq = simple_molecule.get_sequence_string()
        assert seq == "AAAAAAAAAA"  # 10 alanines

    def test_center_to_origin(self, simple_molecule):
        simple_molecule.center_to_origin()
        coords = simple_molecule.get_ca_coords()
        center = coords.mean(axis=0)
        assert np.allclose(center, [0, 0, 0], atol=1e-10)


class TestGeometry:
    """Test geometry functions."""

    def test_calc_distance(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        assert calc_distance(p1, p2) == pytest.approx(5.0)

    def test_calc_distance_same_point(self):
        p = np.array([1.0, 2.0, 3.0])
        assert calc_distance(p, p) == 0.0

    def test_normalize(self):
        v = np.array([3.0, 0.0, 0.0])
        result = normalize(v)
        assert np.allclose(result, [1.0, 0.0, 0.0])

    def test_normalize_unit_length(self):
        v = np.array([1.0, 2.0, 3.0])
        result = normalize(v)
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_calc_r14_handedness(self):
        # Right-handed configuration
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.8, 0.0, 0.0])
        p3 = np.array([5.7, 3.3, 0.0])
        p4 = np.array([9.5, 3.3, 0.0])

        r14 = calc_r14(p1, p2, p3, p4)
        assert r14 > 0  # Right-handed should be positive

    def test_calc_angle(self):
        # 90 degree angle
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        angle = calc_angle(p1, p2, p3)
        assert angle == pytest.approx(np.pi / 2)

    def test_calc_angle_180(self):
        # 180 degree angle (straight line)
        p1 = np.array([-1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])

        angle = calc_angle(p1, p2, p3)
        assert angle == pytest.approx(np.pi)
