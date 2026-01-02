"""
C-alpha energy calculations for structure optimization.

Source: pulchra.c lines 849-1020 (calc_ca_energy)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from pulchra.core.constants import (
    CA_K,
    CA_ANGLE_K,
    CA_START_K,
    CA_XVOL_K,
    CA_DIST,
    CA_DIST_CISPRO,
    CA_START_DIST,
    CA_XVOL_DIST,
    DEGRAD,
    RADDEG,
)


@dataclass
class EnergyComponents:
    """Container for energy components."""

    bond: float = 0.0  # CA-CA bond energy
    restraint: float = 0.0  # Restraint to initial coords
    angle: float = 0.0  # CA-CA-CA angle energy
    xvol: float = 0.0  # Excluded volume energy

    @property
    def total(self) -> float:
        return self.bond + self.restraint + self.angle + self.xvol


def calc_ca_energy(
    ca_coords: np.ndarray,
    init_coords: np.ndarray,
    gradient: Optional[np.ndarray] = None,
    alpha: float = 0.0,
    cispro_flags: Optional[np.ndarray] = None,
    calc_gradient: bool = True,
    ca_start_dist: float = CA_START_DIST,
    ca_xvol_dist: float = CA_XVOL_DIST,
) -> Tuple[float, EnergyComponents, np.ndarray]:
    """
    Calculate C-alpha chain energy and gradients.

    Source: pulchra.c lines 849-1020

    This computes:
    - Bond energy: penalizes deviations from ideal CA-CA distance (3.8 A)
    - Angle energy: penalizes CA-CA-CA angles outside 80-150 degrees
    - Restraint energy: penalizes large deviations from initial coordinates
    - Excluded volume: penalizes CA atoms that are too close to each other

    Args:
        ca_coords: Current CA coordinates, shape (N, 3)
        init_coords: Initial CA coordinates, shape (N, 3)
        gradient: Gradient array to accumulate into, shape (N, 3). If None, created.
        alpha: Step size for line search (coords += alpha * gradient)
        cispro_flags: Boolean array indicating cis-proline positions
        calc_gradient: If True, compute gradients
        ca_start_dist: Distance threshold for restraint energy
        ca_xvol_dist: Distance threshold for excluded volume

    Returns:
        Tuple of:
        - total_energy: Total energy value
        - components: EnergyComponents breakdown
        - gradient: Updated gradient array
    """
    n = len(ca_coords)

    if gradient is None:
        gradient = np.zeros((n, 3), dtype=np.float64)

    if cispro_flags is None:
        cispro_flags = np.zeros(n, dtype=bool)

    # Apply step
    new_coords = ca_coords + alpha * gradient

    # Initialize energy components
    components = EnergyComponents()

    # Reset gradient if calculating
    if calc_gradient and alpha == 0.0:
        gradient.fill(0.0)

    # === Restraint energy (to initial coordinates) ===
    diff = new_coords - init_coords
    distances = np.linalg.norm(diff, axis=1)

    # Only apply restraint if distance > threshold
    mask = distances > ca_start_dist
    if np.any(mask):
        dist_sq = distances[mask] ** 2
        components.restraint = CA_START_K * np.sum(dist_sq)

        if calc_gradient:
            # Gradient: d/dx [k*r^2] = 2*k*r * (dr/dx) = 2*k*r * (x/r) = 2*k*x
            # But we have -ddist for direction, so:
            # grad = -2 * CA_START_K * (x, y, z) when dist > threshold
            for i in np.where(mask)[0]:
                grad_factor = -2.0 * CA_START_K
                gradient[i] -= grad_factor * diff[i]

    # === Bond energy (CA-CA distances) ===
    for i in range(1, n):
        dx = new_coords[i] - new_coords[i - 1]
        dist = np.linalg.norm(dx)

        # Target distance depends on cis-proline
        target_dist = CA_DIST_CISPRO if cispro_flags[i] else CA_DIST
        ddist = target_dist - dist
        ddist2 = ddist * ddist

        components.bond += CA_K * ddist2

        if calc_gradient and dist > 1e-10:
            grad = ddist * (-2.0 * CA_K) / dist
            gradient[i] -= grad * dx
            gradient[i - 1] += grad * dx

    # === Excluded volume energy ===
    for i in range(n):
        for j in range(i):
            if abs(i - j) > 2:  # Skip nearby residues
                dx = new_coords[i] - new_coords[j]
                dist = np.linalg.norm(dx)

                if dist < ca_xvol_dist:
                    ddist2 = dist * dist
                    components.xvol += CA_XVOL_K * ddist2

                    if calc_gradient and dist > 1e-10:
                        # Gradient pushes atoms apart
                        ddist = dist - ca_xvol_dist
                        grad = ddist * (8.0 * CA_XVOL_K) / dist
                        gradient[i] -= grad * dx
                        gradient[j] += grad * dx

    # === Angle energy (CA-CA-CA angles) ===
    for i in range(1, n - 1):
        r12 = new_coords[i - 1] - new_coords[i]
        r32 = new_coords[i + 1] - new_coords[i]

        d12 = np.linalg.norm(r12)
        d32 = np.linalg.norm(r32)

        if d12 < 1e-10 or d32 < 1e-10:
            continue

        cos_theta = np.dot(r12, r32) / (d12 * d32)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        theta = np.arccos(cos_theta)

        # Angle constraint: 80 - 150 degrees
        theta_deg = theta * RADDEG
        if theta_deg < 80.0:
            diff_angle = theta - 80.0 * DEGRAD
        elif theta_deg > 150.0:
            diff_angle = theta - 150.0 * DEGRAD
        else:
            diff_angle = 0.0

        components.angle += CA_ANGLE_K * diff_angle * diff_angle

        if calc_gradient and abs(diff_angle) > 1e-10 and sin_theta > 1e-10:
            d12inv = 1.0 / d12
            d32inv = 1.0 / d32

            diff_scaled = diff_angle * (-2.0 * CA_ANGLE_K) / sin_theta
            c1 = diff_scaled * d12inv
            c2 = diff_scaled * d32inv

            f1 = c1 * (r12 * (d12inv * cos_theta) - r32 * d32inv)
            f3 = c2 * (r32 * (d32inv * cos_theta) - r12 * d12inv)
            f2 = -f1 - f3

            gradient[i - 1] += f1
            gradient[i] += f2
            gradient[i + 1] += f3

    return components.total, components, gradient


def calc_ca_energy_vectorized(
    ca_coords: np.ndarray,
    init_coords: np.ndarray,
    cispro_flags: Optional[np.ndarray] = None,
    ca_start_dist: float = CA_START_DIST,
    ca_xvol_dist: float = CA_XVOL_DIST,
) -> Tuple[float, EnergyComponents]:
    """
    Calculate C-alpha chain energy using vectorized operations (no gradients).

    This is faster than calc_ca_energy when gradients are not needed.

    Args:
        ca_coords: Current CA coordinates, shape (N, 3)
        init_coords: Initial CA coordinates, shape (N, 3)
        cispro_flags: Boolean array indicating cis-proline positions
        ca_start_dist: Distance threshold for restraint energy
        ca_xvol_dist: Distance threshold for excluded volume

    Returns:
        Tuple of:
        - total_energy: Total energy value
        - components: EnergyComponents breakdown
    """
    n = len(ca_coords)

    if cispro_flags is None:
        cispro_flags = np.zeros(n, dtype=bool)

    components = EnergyComponents()

    # Restraint energy
    diff = ca_coords - init_coords
    distances = np.linalg.norm(diff, axis=1)
    mask = distances > ca_start_dist
    if np.any(mask):
        components.restraint = CA_START_K * np.sum(distances[mask] ** 2)

    # Bond energy (vectorized)
    bond_vectors = np.diff(ca_coords, axis=0)
    bond_lengths = np.linalg.norm(bond_vectors, axis=1)
    target_dists = np.where(cispro_flags[1:], CA_DIST_CISPRO, CA_DIST)
    bond_deviations = target_dists - bond_lengths
    components.bond = CA_K * np.sum(bond_deviations**2)

    # Angle energy (vectorized)
    if n >= 3:
        v1 = ca_coords[:-2] - ca_coords[1:-1]
        v2 = ca_coords[2:] - ca_coords[1:-1]
        d1 = np.linalg.norm(v1, axis=1)
        d2 = np.linalg.norm(v2, axis=1)

        valid = (d1 > 1e-10) & (d2 > 1e-10)
        cos_theta = np.zeros(len(v1))
        cos_theta[valid] = (
            np.sum(v1[valid] * v2[valid], axis=1) / (d1[valid] * d2[valid])
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta_deg = theta * RADDEG

        # Apply angle constraints
        diff_angle = np.zeros_like(theta)
        low_mask = theta_deg < 80.0
        high_mask = theta_deg > 150.0
        diff_angle[low_mask] = theta[low_mask] - 80.0 * DEGRAD
        diff_angle[high_mask] = theta[high_mask] - 150.0 * DEGRAD

        components.angle = CA_ANGLE_K * np.sum(diff_angle**2)

    # Excluded volume (using scipy for efficiency)
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(ca_coords))

    # Mask for non-neighboring pairs (|i-j| > 2)
    idx = np.arange(n)
    neighbor_mask = np.abs(idx[:, None] - idx[None, :]) <= 2
    np.fill_diagonal(neighbor_mask, True)

    # Find clashing pairs
    clash_mask = (dist_matrix < ca_xvol_dist) & ~neighbor_mask
    clash_distances = dist_matrix[clash_mask]

    if len(clash_distances) > 0:
        # Each pair is counted twice in symmetric matrix
        components.xvol = CA_XVOL_K * np.sum(clash_distances**2) / 2

    return components.total, components
