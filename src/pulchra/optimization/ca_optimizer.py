"""
C-alpha position optimization using steepest descent.

Source: pulchra.c lines 1026-1415 (ca_optimize)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from pulchra.core.structures import Molecule
from pulchra.core.constants import (
    CA_ITER,
    CA_DIST,
    CA_DIST_CISPRO,
    CA_DIST_CISPRO_TOL,
)
from pulchra.optimization.energy import calc_ca_energy, EnergyComponents


@dataclass
class OptimizationResult:
    """Result of CA optimization."""

    final_energy: float
    initial_energy: float
    iterations: int
    converged: bool
    energy_components: EnergyComponents
    trajectory: Optional[List[np.ndarray]] = None


@dataclass
class CAOptimizerConfig:
    """Configuration for CA optimizer."""

    max_iter: int = CA_ITER
    convergence_threshold: float = 0.001
    detect_cispro: bool = False
    random_start: bool = False
    save_trajectory: bool = False
    verbose: bool = False


class CAOptimizer:
    """
    Steepest descent optimizer for C-alpha positions.

    Source: pulchra.c lines 1026-1415

    This optimizer adjusts CA positions to satisfy:
    - Ideal CA-CA distances (3.8 A, or 2.9 A for cis-proline)
    - Reasonable CA-CA-CA angles (80-150 degrees)
    - No excluded volume violations
    - Stay close to initial coordinates
    """

    def __init__(self, config: Optional[CAOptimizerConfig] = None):
        """
        Initialize the optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config or CAOptimizerConfig()

    def optimize(
        self,
        molecule: Molecule,
        initial_coords: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Optimize C-alpha positions in the molecule.

        Args:
            molecule: Molecule to optimize (modified in place)
            initial_coords: Optional initial coordinates for restraints.
                          If None, uses current coordinates.

        Returns:
            OptimizationResult with optimization details
        """
        # Get current CA coordinates
        ca_coords = molecule.get_ca_coords().copy()
        n = len(ca_coords)

        if n < 2:
            return OptimizationResult(
                final_energy=0.0,
                initial_energy=0.0,
                iterations=0,
                converged=True,
                energy_components=EnergyComponents(),
            )

        # Store initial coordinates for restraints
        if initial_coords is None:
            init_coords = ca_coords.copy()
        else:
            init_coords = initial_coords.copy()

        # Detect cis-prolines if requested
        cispro_flags = np.zeros(n, dtype=bool)
        if self.config.detect_cispro:
            cispro_flags = self._detect_cis_prolines(molecule, ca_coords)

        # Random start if requested
        if self.config.random_start:
            ca_coords = self._generate_random_chain(n)

        # Initialize trajectory storage
        trajectory = [] if self.config.save_trajectory else None

        # Calculate initial energy
        gradient = np.zeros((n, 3), dtype=np.float64)
        initial_energy, initial_components, _ = calc_ca_energy(
            ca_coords, init_coords, gradient, 0.0, cispro_flags, calc_gradient=True
        )

        if self.config.verbose:
            print(
                f"Initial energy: bond={initial_components.bond:.5f} "
                f"angle={initial_components.angle:.5f} "
                f"restraints={initial_components.restraint:.5f} "
                f"xvol={initial_components.xvol:.5f} "
                f"total={initial_energy:.5f}"
            )

        # Optimization loop
        converged = False
        iteration = 0
        last_gnorm = 1000.0

        for iteration in range(self.config.max_iter):
            # Save trajectory if requested
            if trajectory is not None:
                trajectory.append(ca_coords.copy())

            # Reset gradient
            gradient.fill(0.0)

            # Calculate energy and gradient
            e_pot, components, gradient = calc_ca_energy(
                ca_coords, init_coords, gradient, 0.0, cispro_flags, calc_gradient=True
            )

            # Line search
            alpha = self._line_search(
                ca_coords, init_coords, gradient, cispro_flags
            )

            # Update coordinates
            ca_coords += alpha * gradient

            # Check convergence
            gnorm = np.linalg.norm(gradient)
            if abs(gnorm - last_gnorm) < self.config.convergence_threshold:
                converged = True
                break

            last_gnorm = gnorm

        # Calculate final energy
        gradient.fill(0.0)
        final_energy, final_components, _ = calc_ca_energy(
            ca_coords, init_coords, gradient, 0.0, cispro_flags, calc_gradient=False
        )

        if self.config.verbose:
            print(
                f"Final energy: bond={final_components.bond:.5f} "
                f"angle={final_components.angle:.5f} "
                f"restraints={final_components.restraint:.5f} "
                f"xvol={final_components.xvol:.5f} "
                f"total={final_energy:.5f} "
                f"(iterations={iteration + 1})"
            )

        # Update molecule with optimized coordinates
        molecule.set_ca_coords(ca_coords)

        return OptimizationResult(
            final_energy=final_energy,
            initial_energy=initial_energy,
            iterations=iteration + 1,
            converged=converged,
            energy_components=final_components,
            trajectory=trajectory,
        )

    def _line_search(
        self,
        ca_coords: np.ndarray,
        init_coords: np.ndarray,
        gradient: np.ndarray,
        cispro_flags: np.ndarray,
    ) -> float:
        """
        Golden section line search to find optimal step size.

        Source: pulchra.c lines 1220-1290

        Args:
            ca_coords: Current CA coordinates
            init_coords: Initial coordinates for restraints
            gradient: Current gradient
            cispro_flags: Cis-proline flags

        Returns:
            Optimal step size (alpha)
        """
        # Initial bracket
        alpha1 = -1.0
        alpha2 = 0.0
        alpha3 = 1.0

        def energy_at_alpha(alpha):
            e, _, _ = calc_ca_energy(
                ca_coords, init_coords, gradient.copy(), alpha, cispro_flags,
                calc_gradient=False
            )
            return e

        ene1 = energy_at_alpha(alpha1)
        ene2 = energy_at_alpha(alpha2)
        ene3 = energy_at_alpha(alpha3)

        # Expand bracket if needed
        max_expand = self.config.max_iter
        expand_steps = 0
        while ene2 > min(ene1, ene3) and expand_steps < max_expand:
            expand_steps += 1
            alpha1 *= 2.0
            alpha3 *= 2.0
            ene1 = energy_at_alpha(alpha1)
            ene3 = energy_at_alpha(alpha3)

        # Golden section search
        golden_ratio = 0.618034

        for _ in range(20):  # Max iterations for line search
            if alpha3 - alpha2 > alpha2 - alpha1:
                a0 = 0.5 * (alpha2 + alpha3)
                e0 = energy_at_alpha(a0)

                if e0 < ene2:
                    alpha1 = alpha2
                    ene1 = ene2
                    alpha2 = a0
                    ene2 = e0
                else:
                    alpha3 = a0
                    ene3 = e0
            else:
                a0 = 0.5 * (alpha1 + alpha2)
                e0 = energy_at_alpha(a0)

                if e0 < ene2:
                    alpha3 = alpha2
                    ene3 = ene2
                    alpha2 = a0
                    ene2 = e0
                else:
                    alpha1 = a0
                    ene1 = e0

            # Check convergence
            if abs(alpha3 - alpha1) < 0.001:
                break

        return alpha2

    def _detect_cis_prolines(
        self,
        molecule: Molecule,
        ca_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Detect probable cis-proline positions based on CA-CA distance.

        Source: pulchra.c lines 1104-1119

        Args:
            molecule: Molecule to analyze
            ca_coords: CA coordinates

        Returns:
            Boolean array of cis-proline flags
        """
        n = len(ca_coords)
        cispro_flags = np.zeros(n, dtype=bool)

        for i in range(1, n):
            res = molecule.residues[i]
            if res.name == "PRO":
                dx = ca_coords[i] - ca_coords[i - 1]
                dist = np.linalg.norm(dx)

                # Check if distance is close to cis-proline distance
                lower = CA_DIST_CISPRO - 5 * CA_DIST_CISPRO_TOL
                upper = CA_DIST_CISPRO + 5 * CA_DIST_CISPRO_TOL

                if lower < dist < upper:
                    cispro_flags[i] = True
                    if self.config.verbose:
                        print(f"Probable cis-proline found at position {res.num}")

        return cispro_flags

    def _generate_random_chain(self, n: int) -> np.ndarray:
        """
        Generate random CA coordinates as a self-avoiding walk.

        Source: pulchra.c lines 1121-1139

        Args:
            n: Number of CA atoms

        Returns:
            Random CA coordinates, shape (n, 3)
        """
        coords = np.zeros((n, 3), dtype=np.float64)

        for i in range(1, n):
            # Random unit vector
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)

            # Scale to CA-CA distance
            coords[i] = coords[i - 1] + CA_DIST * direction

        return coords


def optimize_ca_positions(
    molecule: Molecule,
    max_iter: int = CA_ITER,
    detect_cispro: bool = False,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Convenience function to optimize CA positions.

    Args:
        molecule: Molecule to optimize (modified in place)
        max_iter: Maximum iterations
        detect_cispro: Whether to detect cis-prolines
        verbose: Whether to print progress

    Returns:
        OptimizationResult with optimization details
    """
    config = CAOptimizerConfig(
        max_iter=max_iter,
        detect_cispro=detect_cispro,
        verbose=verbose,
    )
    optimizer = CAOptimizer(config)
    return optimizer.optimize(molecule)
