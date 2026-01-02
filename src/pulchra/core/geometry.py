"""
Geometric utility functions for molecular calculations.

Source: pulchra.c lines 1570-1820, 2258-2271, 3050-3094
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from pulchra.core.constants import RADDEG


def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Source: pulchra.c lines 1571-1583

    Args:
        p1: First point [x, y, z]
        p2: Second point [x, y, z]

    Returns:
        Distance between points
    """
    diff = p1 - p2
    dist_sq = np.dot(diff, diff)
    if dist_sq > 0:
        return np.sqrt(dist_sq)
    return 0.0


def calc_r14(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate chiral r14 distance with handedness sign.

    This computes the distance from p1 to p4, with sign indicating
    the handedness of the p1-p2-p3-p4 chain.

    Source: pulchra.c lines 1586-1615

    Args:
        p1: First point (e.g., CA[i-2])
        p2: Second point (e.g., CA[i-1])
        p3: Third point (e.g., CA[i])
        p4: Fourth point (e.g., CA[i+1])

    Returns:
        Signed distance (negative for left-handed, positive for right-handed)
    """
    # Distance from p1 to p4
    d = p4 - p1
    r = np.linalg.norm(d)

    # Vectors along the chain
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3

    # Calculate handedness using scalar triple product
    # hand = (v1 x v2) . v3
    cross_v1_v2 = np.cross(v1, v2)
    hand = np.dot(cross_v1_v2, v3)

    if hand < 0:
        r = -r

    return r


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Source: pulchra.c lines 2264-2271

    Args:
        v: Input vector

    Returns:
        Normalized vector (unit length)
    """
    d = np.linalg.norm(v)
    if d > 0:
        return v / d
    return v.copy()


def superimpose(
    coords1: np.ndarray,
    coords2: np.ndarray,
    transform_points: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Superimpose coords2 onto coords1 using Kabsch algorithm.

    This finds the optimal rotation and translation to minimize RMSD
    between coords1 (target) and coords2 (mobile).

    Source: pulchra.c lines 1620-1820 (reimplemented with scipy)

    Args:
        coords1: Target coordinates, shape (N, 3)
        coords2: Mobile coordinates to be superimposed, shape (N, 3)
        transform_points: Optional additional points to transform, shape (M, 3)

    Returns:
        Tuple of:
        - rmsd: Root mean square deviation after superposition
        - transformed: Transformed coords2 after superposition
        - rotation_matrix: 3x3 rotation matrix
        - translation: Translation vector
    """
    assert len(coords1) == len(coords2), "Coordinate arrays must have same length"

    # Center both coordinate sets
    center1 = coords1.mean(axis=0)
    center2 = coords2.mean(axis=0)

    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2

    # Use scipy's Rotation.align_vectors for robust SVD-based alignment
    # This implements the Kabsch algorithm
    rotation, rmsd = Rotation.align_vectors(coords1_centered, coords2_centered)
    rotation_matrix = rotation.as_matrix()

    # Transform coords2
    transformed = coords2_centered @ rotation_matrix.T + center1

    # Calculate RMSD
    diff = coords1 - transformed
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    # Translation vector (from centered coords2 to final position)
    translation = center1 - center2 @ rotation_matrix.T

    # Transform additional points if provided
    if transform_points is not None:
        transform_points_centered = transform_points - center2
        transformed_points = transform_points_centered @ rotation_matrix.T + center1
        return rmsd, transformed, rotation_matrix, translation, transformed_points

    return rmsd, transformed, rotation_matrix, translation


def superimpose_and_transform(
    template_coords: np.ndarray,
    target_coords: np.ndarray,
    points_to_transform: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Superimpose template onto target and apply same transformation to other points.

    This is the common use case: align a template to target CA positions,
    then apply the same transformation to place backbone/sidechain atoms.

    Args:
        template_coords: Template reference coordinates (e.g., template CA positions)
        target_coords: Target coordinates to align to (e.g., actual CA positions)
        points_to_transform: Additional points to transform (e.g., NCO atoms)

    Returns:
        Tuple of:
        - rmsd: RMSD of the alignment
        - transformed_points: Transformed points_to_transform
    """
    result = superimpose(target_coords, template_coords, points_to_transform)
    return result[0], result[4]


def calc_torsion(
    a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, a4: np.ndarray
) -> float:
    """
    Calculate dihedral/torsion angle for four points.

    Source: pulchra.c lines 3054-3094

    Args:
        a1, a2, a3, a4: Four atom positions defining the torsion angle

    Returns:
        Torsion angle in degrees (-180 to 180)
    """
    # Vectors
    v12 = a1 - a2
    v43 = a4 - a3
    z = a2 - a3

    # Cross products
    p = np.cross(z, v12)
    x = np.cross(z, v43)
    y = np.cross(z, x)

    # Dot products
    u = np.dot(x, x)
    v = np.dot(y, y)

    if u < 0 or v < 0:
        return 360.0

    u_norm = np.sqrt(u)
    v_norm = np.sqrt(v)

    if u_norm < 1e-10 or v_norm < 1e-10:
        return 360.0

    u_val = np.dot(p, x) / u_norm
    v_val = np.dot(p, y) / v_norm

    if u_val != 0.0 or v_val != 0.0:
        angle = np.arctan2(v_val, u_val) * RADDEG
    else:
        angle = 360.0

    return angle


def calc_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: Three points

    Returns:
        Angle in radians
    """
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-10 or v2_norm < 1e-10:
        return 0.0

    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def build_local_frame(
    ca_prev: np.ndarray, ca_curr: np.ndarray, ca_next: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a local coordinate frame from three consecutive CA atoms.

    Used for sidechain reconstruction.

    Args:
        ca_prev: Previous CA position
        ca_curr: Current CA position
        ca_next: Next CA position

    Returns:
        Tuple of three orthonormal vectors (v1, v2, v3)
    """
    # v1: direction from prev to next (backbone direction)
    v1 = ca_next - ca_prev
    v1 = normalize(v1)

    # v2: perpendicular to backbone plane
    v_a = ca_next - ca_curr
    v_b = ca_curr - ca_prev
    v2 = np.cross(v_a, v_b)
    v2 = normalize(v2)

    # v3: perpendicular to both
    v3 = np.cross(v1, v2)
    v3 = normalize(v3)

    return v1, v2, v3


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    Calculate all pairwise distances between points.

    Args:
        coords: Array of coordinates, shape (N, 3)

    Returns:
        Distance matrix, shape (N, N)
    """
    from scipy.spatial.distance import cdist

    return cdist(coords, coords)


def rotate_point_around_axis(
    point: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray, angle: float
) -> np.ndarray:
    """
    Rotate a point around an axis.

    Args:
        point: Point to rotate
        axis_point: A point on the rotation axis
        axis_dir: Direction vector of the rotation axis
        angle: Rotation angle in radians

    Returns:
        Rotated point
    """
    # Normalize axis
    axis = normalize(axis_dir)

    # Translate point to origin
    p = point - axis_point

    # Rodrigues' rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotated = (
        p * cos_a + np.cross(axis, p) * sin_a + axis * np.dot(axis, p) * (1 - cos_a)
    )

    return rotated + axis_point


def reflect_point(
    point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """
    Reflect a point across a plane.

    Args:
        point: Point to reflect
        plane_point: A point on the reflection plane
        plane_normal: Normal vector to the plane (will be normalized)

    Returns:
        Reflected point
    """
    normal = normalize(plane_normal)

    # Vector from plane point to the point
    v = point - plane_point

    # Distance to plane (signed)
    d = np.dot(v, normal)

    # Reflect
    return point - 2 * d * normal
