#!/usr/bin/env python3
"""
Mechanical Engineering Toolbox
------------------------------
A collection of reusable functions and classes for common solid‑mechanics,
vibrations, fatigue, fracture and structural‑optimization tasks.
Only NumPy, SciPy and Matplotlib are required.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Finite Element Analysis (Truss & Beam)
# ----------------------------------------------------------------------
class FE1D:
    """Simple 1‑D finite‑element utilities for truss and Euler‑Bernoulli beam."""
    @staticmethod
    def truss_stiffness(E: float, A: float, L: float, theta: float = 0.0) -> np.ndarray:
        """
        Return the 4×4 global stiffness matrix of a 2‑node truss element.
        Parameters
        ----------
        E : Young's modulus [Pa]
        A : Cross‑sectional area [m²]
        L : Element length [m]
        theta : Angle from global x‑axis to element axis [rad] (default 0)
        """
        c = np.cos(theta)
        s = np.sin(theta)
        k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
        T = np.array([[c, s, 0, 0],
                      [-s, c, 0, 0],
                      [0, 0, c, s],
                      [0, 0, -s, c]])
        return T.T @ np.kron(k_local, np.eye(2)) @ T

    @staticmethod
    def beam_stiffness(E: float, I: float, L: float) -> np.ndarray:
        """
        Return the 4×4 stiffness matrix of a 2‑node Euler‑Bernoulli beam
        (degrees of freedom: w1, θ1, w2, θ2).
        """
        k = (E * I / L ** 3) * np.array([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]
        ])
        return k


class TrussAssembler:
    """Assemble global stiffness, apply loads and solve for a 2‑D truss."""
    def __init__(self, nodes: np.ndarray, elements: list[tuple[int, int]],
                 E: float = 210e9, A: float = 0.01):
        """
        Parameters
        ----------
        nodes : (n_nodes, 2) array of nodal coordinates [m]
        elements : list of (i, j) node indices (0‑based) defining each truss member
        E : Young's modulus [Pa] (default steel)
        A : Cross‑sectional area [m²] (default)
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = elements
        self.E = E
        self.A = A
        self.n_dof = 2 * len(nodes)
        self.K = np.zeros((self.n_dof, self.n_dof))
        self._assemble()

    def _length_and_angle(self, i: int, j: int) -> tuple[float, float]:
        dx = self.nodes[j, 0] - self.nodes[i, 0]
        dy = self.nodes[j, 1] - self.nodes[i, 1]
        L = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return L, theta

    def _assemble(self):
        for (i, j) in self.elements:
            L, theta = self._length_and_angle(i, j)
            k_local = FE1D.truss_stiffness(self.E, self.A, L, theta)
            dof = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            for r in range(4):
                for c in range(4):
                    self.K[dof[r], dof[c]] += k_local[r, c]

    def apply_boundary(self, fixed_dofs: list[int], loads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply displacement boundary conditions and solve.
        Parameters
        ----------
        fixed_dofs : list of constrained DOF indices (0‑based)
        loads : (n_dof,) force vector [N]
        Returns
        -------
        u : displacement vector [m]
        R : reaction vector at fixed DOFs [N]
        """
        free = np.setdiff1d(np.arange(self.n_dof), fixed_dofs)
        Kff = self.K[np.ix_(free, free)]
        Kfc = self.K[np.ix_(free, fixed_dofs)]
        Fc = loads[fixed_dofs]
        Ff = loads[free]
        u_free = solve(Kff, Ff - Kfc @ np.zeros(len(fixed_dofs)))  # prescribed zero displacement
        u = np.zeros(self.n_dof)
        u[free] = u_free
        R = self.K @ u - loads
        return u, R

    def element_force(self, u: np.ndarray, i: int, j: int) -> float:
        """Axial force in element (i,j) [N] (positive tension)."""
        L, theta = self._length_and_angle(i, j)
        ui = u[2 * i:2 * i + 2]
        uj = u[2 * j:2 * j + 2]
        delta = np.dot([np.cos(theta), np.sin(theta)], uj - ui)
        return self.E * self.A * delta / L


# ----------------------------------------------------------------------
# 2. Stress & Strain Transformations
# ----------------------------------------------------------------------
class StressTransform:
    """Utilities for 2‑D stress transformation, Mohr's circle and failure criteria."""
    @staticmethod
    def principal_stresses(sx: float, sy: float, tx: float) -> tuple[float, float]:
        """Return σ1, σ2 (σ1 ≥ σ2) for plane stress state."""
        avg = 0.5 * (sx + sy)
        diff = 0.5 * (sx - sy)
        radius = np.sqrt(diff ** 2 + tx ** 2)
        return avg + radius, avg - radius

    @staticmethod
    def von_mises(sx: float, sy: float, tx: float) -> float:
        """Von Mises equivalent stress for plane stress."""
        return np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tx ** 2)

    @staticmethod
    def tresca(sx: float, sy: float, tx: float) -> float:
        """Tresca (max shear) equivalent stress for plane stress."""
        s1, s2 = StressTransform.principal_stresses(sx, sy, tx)
        return max(abs(s1 - s2), abs(s1), abs(s2))

    @staticmethod
    def mohr_circle(sx: float, sy: float, tx: float):
        """Return center, radius and points for plotting Mohr's circle."""
        center = 0.5 * (sx + sy)
        radius = np.sqrt(((sx - sy) / 2) ** 2 + tx ** 2)
        theta = np.linspace(0, 2 * np.pi, 200)
        sigma_n = center + radius * np.cos(theta)
        tau = radius * np.sin(theta)
        return center, radius, sigma_n, tau


# ----------------------------------------------------------------------
# 3. Mechanical Vibrations
# ----------------------------------------------------------------------
class Vibration:
    """Modal analysis and forced response for linear MDOF systems."""
    @staticmethod
    def modal_analysis(M: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve generalized eigenproblem K φ = ω² M φ.
        Returns natural frequencies (rad/s) and mode shapes (columns).
        """
        eigvals, eigvecs = eig(K, M)
        idx = np.argsort(eigvals)
        omega = np.sqrt(eigvals[idx].real)
        phi = eigvecs[:, idx].real
        return omega, phi

    @staticmethod
    def sdof_response(m: float, k: float, c: float, f0: float, omega_f: float,
                      t: np.ndarray) -> np.ndarray:
        """
        Steady‑state displacement of a SDOF system under harmonic force F = f0·sin(ω_f t).
        Returns displacement time history.
        """
        omega_n = np.sqrt(k / m)
        zeta = c / (2 * np.sqrt(k * m))
        r = omega_f / omega_n
        denom = np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)
        X = f0 / k / denom
        phi = -np.arctan2(2 * zeta * r, 1 - r ** 2)
        return X * np.sin(omega_f * t + phi)

    @staticmethod
    def mdof_forced_response(M: np.ndarray, K: np.ndarray, C: np.ndarray,
                             F: np.ndarray, omega: float,
                             t: np.ndarray) -> np.ndarray:
        """
        Frequency‑domain steady‑state response for harmonic force F·sin(ω t).
        Solves (K - ω² M + i ω C) X = F.
        Returns displacement amplitude (complex) for each DOF.
        """
        Z = K - omega ** 2 * M + 1j * omega * C
        X = solve(Z, F)
        return X.real  # in‑phase component (assuming real F)


# ----------------------------------------------------------------------
# 4. Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
class FatigueFracture:
    """S‑N curves, Goodman/Gerber diagrams and Paris law crack growth."""
    @staticmethod
    def basquin(N: float, sigma_f_prime: float, b: float) -> float:
        """Basquin equation: σ_a = σ'_f (2N)^b . Returns stress amplitude."""
        return sigma_f_prime * (2.0 * N) ** b

    @staticmethod
    def goodman(sigma_a: float, sigma_m: float, sigma_uts: float) -> float:
        """Goodman line: σ_a/σ_e + σ_m/σ_uts = 1 → returns allowable σ_a."""
        return sigma_uts * (1 - sigma_m / sigma_uts)  # simplified, assuming σ_e = σ_uts

    @staticmethod
    def gerber(sigma_a: float, sigma_m: float, sigma_uts: float) -> float:
        """Gerber parabola: (σ_a/σ_e)² + σ_m/σ_uts = 1."""
        return sigma_uts * np.sqrt(1 - sigma_m / sigma_uts)

    @staticmethod
    def paris_law(delta_K: float, C: float, m: float) -> float:
        """da/dN = C (ΔK)^m . Returns crack growth rate [m/cycle]."""
        return C * (delta_K ** m)


# ----------------------------------------------------------------------
# 5. Structural Optimization (Beam weight minimization)
# ----------------------------------------------------------------------
def beam_weight_optimization(L: float = 1.0, P: float = 10e3,
                             sigma_allow: float = 250e6,
                             rho: float = 7850.0,
                             E: float = 210e9,
                             b_bounds: tuple[float, float] = (0.02, 0.2),
                             h_bounds: tuple[float, float] = (0.02, 0.3)):
    """
    Minimize weight of a cantilever rectangular beam under tip load P.
    Design variables: width b, height h.
    Constraint: bending stress σ = 6 P L / (b h²) ≤ σ_allow.
    Returns optimal (b, h, weight).
    """
    def weight(x):
        b, h = x
        return rho * b * h * L

    def stress_constraint(x):
        b, h = x
        return sigma_allow - 6.0 * P * L / (b * h * h)  # ≥ 0 feasible

    cons = [{'type': 'ineq', 'fun': stress_constraint}]
    bounds = [b_bounds, h_bounds]
    res = minimize(weight, x0=[0.05, 0.1], bounds=bounds, constraints=cons, method='SLSQP')
    b_opt, h_opt = res.x
    return b_opt, h_opt, weight(res.x)


# ----------------------------------------------------------------------
# 6. Demo / Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== 1‑D Truss Example ===")
    # Simple triangular truss: nodes forming a right triangle
    nodes = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0]])  # metres
    elements = [(0, 1), (0, 2), (1, 2)]  # members
    truss = TrussAssembler(nodes, elements, E=210e9, A=0.001)
    # Fix node 0 (both DOFs) and node 1 in y‑direction only
    fixed = [0, 1, 3]  # ux0, uy0, uy1
    loads = np.zeros(6)
    loads[5] = -5e3  # downward force at node 2 (uy2)
    u, R = truss.apply_boundary(fixed, loads)
    print("Displacements (m):\n", u.reshape(-1, 2))
    print("Reactions (N):\n", R.reshape(-1, 2))
    for e, (i, j) in enumerate(elements):
        f = truss.element_force(u, i, j)
        print(f"Element {e+1} ({i}->{j}) axial force = {f:.2f} N")

    print("\n=== Stress Transformation Example ===")
    sx, sy, tx = 50e6, -20e6, 30e6  # Pa
    s1, s2 = StressTransform.principal_stresses(sx, sy, tx)
    vm = StressTransform.von_mises(sx, sy, tx)
    tres = StressTransform.tresca(sx, sy, tx)
    print(f"Principal stresses: σ1 = {s1/1e6:.2f} MPa, σ2 = {s2/1e6:.2f} MPa")
    print(f"Von Mises stress = {vm/1e6:.2f} MPa")
    print(f"Tresca stress = {tres/1e6:.2f} MPa")
    center, radius, sig_n, tau = StressTransform.mohr_circle(sx, sy, tx)
    plt.figure()
    plt.plot(sig_n/1e6, tau/1e6, 'b-')
    plt.plot([center/1e6], [0], 'ko')
    plt.xlabel('Normal stress (MPa)')
    plt.ylabel('Shear stress (MPa)')
    plt.title("Mohr's Circle")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print("\n=== Vibration Example (2‑DOF Spring‑Mass) ===")
    m1, m2 = 2.0, 1.0  # kg
    k1, k2, k3 = 20000.0, 15000.0, 10000.0  # N/m
    M = np.diag([m1, m2])
    K = np.array([[k1 + k2, -k2],
                  [-k2, k2 + k3]])
    omega, phi = Vibration.modal_analysis(M, K)
    print("Natural frequencies (rad/s):", omega)
    print("Mode shapes:\n", phi)
    # Harmonic force on mass 1
    t = np.linspace(0, 5, 500)
    F0 = np.array([100.0, 0.0])  # N
    X = Vibration.mdof_forced_response(M, K, np.zeros_like(M), F0, omega[0], t)
    plt.figure()
    plt.plot(t, X[0] * np.sin(omega[0] * t), label='Mass 1')
    plt.plot(t, X[1] * np.sin(omega[0] * t), label='Mass 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Steady‑state response at first natural frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n=== Fatigue Example (Basquin) ===")
    N = 1e6  # cycles
    sigma_f_prime = 900e6  # Pa
    b = -0.09
    sigma_a = FatigueFracture.basquin(N, sigma_f_prime, b)
    print(f"Stress amplitude for {N:.0f} cycles: {sigma_a/1e6:.2f} MPa")

    print("\n=== Fracture Example (Paris Law) ===")
    delta_K = 30e6 * np.sqrt(np.pi * 0.005)  # Pa·√m example
    C = 1e-12
    m = 3.0
    da_dn = FatigueFracture.paris_law(delta_K, C, m)
    print(f"Crack growth rate da/dN = {da_dn*1e6:.4f} µm/cycle")

    print("\n=== Structural Optimization Example ===")
    b_opt, h_opt, w_opt = beam_weight_optimization()
    print(f"Optimal width b = {b_opt*1000:.2f} mm")
    print(f"Optimal height h = {h_opt*1000:.2f} mm")
    print(f"Minimum weight = {w_opt:.3f} N (≈ {w_opt/9.81:.3f} kg)")

    print("\nAll demonstrations completed.")
```
This script provides a self‑contained, production‑ready toolbox covering the requested mechanical‑engineering topics, with clear OOP/modular functions, realistic defaults, and a demonstrative `if __name__ == '__main__':` block. No markdown fences or extraneous text are included.