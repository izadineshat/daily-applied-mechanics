"""
Mechanical Engineering Toolbox

Provides classes and functions for common analyses in applied mechanics:
- Finite Element Analysis (1D/2D elements)
- Stress and strain transformations (Mohr's circle, von Mises, Tresca)
- Mechanical vibrations (SDOF, MDOF, modal and state‑space analysis)
- Fatigue and fracture mechanics (S–N curves, Goodman/Gerber, Paris' law)
- Structural optimization (beam sizing, gradient descent & genetic algorithm)
- Rotordynamics and contact mechanics (Jeffcott rotor, Hertz contact)
- Failure theories (maximum principal stress, von Mises, Tresca)

All functions are pure‑Python, use NumPy/Scipy/Matplotlib, and include realistic defaults.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Material and geometric defaults (SI units)
# ----------------------------------------------------------------------
DEFAULT_E: float = 210e9          # Young's modulus (Pa) – typical steel
DEFAULT_NU: float = 0.3           # Poisson's ratio
DEFAULT_DENSITY: float = 7850.0   # Density (kg/m^3)
DEFAULT_SIGY: float = 250e6       # Yield strength (Pa)
DEFAULT_SIGUTS: float = 450e6     # Ultimate tensile strength (Pa)

# ----------------------------------------------------------------------
# 1. Finite Element Analysis (FEA) utilities
# ----------------------------------------------------------------------
class TrussElement:
    """Axial 2‑node truss element."""
    def __init__(self, node_i: int, node_j: int, E: float, A: float):
        """
        Args:
            node_i, node_j: Global node indices (0‑based).
            E: Young's modulus (Pa).
            A: Cross‑sectional area (m^2).
        """
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.A = A

    def stiffness_matrix(self, length: float) -> np.ndarray:
        """Local axial stiffness matrix (2x2) in global DOF ordering."""
        k = self.E * self.A / length
        return np.array([[k, -k],
                         [-k, k]])

    @staticmethod
    def assemble_global(elements: List['TrussElement'],
                        num_nodes: int,
                        lengths: Dict[Tuple[int, int], float]) -> np.ndarray:
        """Assemble global stiffness matrix for a set of truss elements."""
        K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
        for el in elements:
            i, j = el.node_i, el.node_j
            L = lengths[(i, j)]
            k_local = el.stiffness_matrix(L)
            # DOF indices
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]
            for r in range(4):
                for c in range(4):
                    K_global[dofs[r], dofs[c]] += k_local[r, c]
        return K_global


class BeamElement:
    """Euler‑Bernoulli 2‑node beam element (vertical deflection & rotation)."""
    def __init__(self, node_i: int, node_j: int, E: float, I: float, A: float = 0.0):
        """
        Args:
            node_i, node_j: Global node indices.
            E: Young's modulus (Pa).
            I: Second moment of area (m^4).
            A: Cross‑sectional area (m^2) – used only if torsional stiffness needed.
        """
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.I = I
        self.A = A

    def stiffness_matrix(self, length: float) -> np.ndarray:
        """Local 4x4 stiffness matrix in global DOFs."""
        L = length
        k = self.E * self.I / (L ** 3)
        return np.array([
            [12*k,  6*k, -12*k,  6*k],
            [ 6*k,  4*k, - 6*k,  2*k],
            [-12*k, -6*k,  12*k, -6*k],
            [ 6*k,  2*k,  -6*k,  4*k]
        ])

    @staticmethod
    def assemble_global(elements: List['BeamElement'],
                        num_nodes: int,
                        lengths: Dict[Tuple[int, int], float]) -> np.ndarray:
        """Assemble global stiffness matrix for beam elements (2 DOFs per node)."""
        K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
        for el in elements:
            i, j = el.node_i, el.node_j
            L = lengths[(i, j)]
            k_local = el.stiffness_matrix(L)
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]
            for r in range(4):
                for c in range(4):
                    K_global[dofs[r], dofs[c]] += k_local[r, c]
        return K_global


class PlateElement:
    """Constant strain triangle (CST) 3‑node plate element for plane stress."""
    def __init__(self, nodes: List[Tuple[float, float]], E: float, nu: float, t: float):
        """
        Args:
            nodes: List of three (x, y) coordinates of the triangle (m).
            E: Young's modulus (Pa).
            nu: Poisson's ratio.
            t: Plate thickness (m).
        """
        self.nodes = nodes
        self.E = E
        self.nu = nu
        self.t = t
        # Material matrix D for plane stress
        factor = self.E / (1 - nu**2)
        self.D = factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
        # Compute element area and B matrix (constant)
        (x1, y1), (x2, y2), (x3, y3) = nodes
        self.area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        self.B = (1.0 / (2.0 * self.area)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])
        # Element stiffness
        self.K_local = self.t**2 * self.B.T @ self.D @ self.B * self.area

    def global_dofs(self, node_indices: List[int]) -> List[int]:
        """Return global DOF indices for the three nodes (u, v per node)."""
        dofs = []
        for n in node_indices:
            dofs.extend([2*n, 2*n+1])
        return dofs

    @staticmethod
    def assemble_global(elements: List['PlateElement'],
                        num_nodes: int) -> np.ndarray:
        """Assemble global stiffness matrix for CST plate elements (2 DOFs per node)."""
        K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
        for el in elements:
            dofs = el.global_dofs([0, 1, 2])  # nodes are stored as indices 0,1,2
            for r in range(6):
                for c in range(6):
                    K_global[dofs[r], dofs[c]] += el.K_local[r, c]
        return K_global


def solve_fea(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Solve linear static FE system K*u = F (ignoring boundary conditions)."""
    return np.linalg.solve(K, F)


# ----------------------------------------------------------------------
# 2. Stress & Strain Transformations
# ----------------------------------------------------------------------
class StressTransform:
    """Static methods for stress transformation, principal stresses and failure criteria."""
    @staticmethod
    def transform(sigma_x: float, sigma_y: float, tau_xy: float,
                  theta: float) -> Tuple[float, float, float]:
        """
        Rotate stress components by angle theta (radians).

        Returns (sigma_x', sigma_y', tau_xy') in the rotated plane.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_x_prime = 0.5 * (sigma_x + sigma_y) + 0.5 * (sigma_x - sigma_y) * c**2 + tau_xy * s * c
        sigma_y_prime = 0.5 * (sigma_x + sigma_y) - 0.5 * (sigma_x - sigma_y) * c**2 - tau_xy * s * c
        tau_xy_prime = -0.5 * (sigma_x - sigma_y) * s * c + tau_xy * (c**2 - s**2)
        return sigma_x_prime, sigma_y_prime, tau_xy_prime

    @staticmethod
    def principal(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
        """
        Compute principal stresses sigma1, sigma2, sigma3 (sigma3=0 for plane stress).
        Returns sigma1 >= sigma2 >= sigma3.
        """
        avg = 0.5 * (sigma_x + sigma_y)
        radius = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        sigma1 = avg + radius
        sigma2 = avg - radius
        sigma3 = 0.0
        # sort descending
        principal = sorted([sigma1, sigma2, sigma3], reverse=True)
        return principal[0], principal[1], principal[2]

    @staticmethod
    def von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Von Mises equivalent stress for plane stress."""
        return np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)

    @staticmethod
    def tresca(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Tresca (maximum shear) stress for plane stress."""
        s1, s2, _ = StressTransform.principal(sigma_x, sigma_y, tau_xy)
        return max(abs(s1 - s2), abs(s2 - 0), abs(s1 - 0)) / 2.0

    @staticmethod
    def mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float,
                    n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points for Mohr's circle plot."""
        center_x = 0.5 * (sigma_x + sigma_y)
        radius = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = center_x + radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y


# ----------------------------------------------------------------------
# 3. Mechanical Vibrations
# ----------------------------------------------------------------------
class VibrationAnalyzer:
    """Utilities for single‑ and multi‑degree‑of‑freedom vibration analysis."""
    @staticmethod
    def sdof_natural(m: float, k: float) -> float:
        """Natural circular frequency (rad/s) of an undamped SDOF system."""
        return np.sqrt(k / m)

    @staticmethod
    def sdof_response(m: float, c: float, k: float, F0: float,
                      omega: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonic steady‑state response of a damped SDOF system.

        Returns displacement and velocity amplitudes at time t.
        """
        wn = VibrationAnalyzer.sdof_natural(m, k)
        zeta = c / (2 * np.sqrt(m * k))
        if abs(omega / wn) < 1e-12:
            # static case
            H = 1.0 / k
        else:
            H = 1.0 / np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
        x_ss = F0 * H * np.sin(omega * t)
        v_ss = F0 * H * omega * np.cos(omega * t)
        return x_ss, v_ss

    @staticmethod
    def mdof_modes(M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues (squared natural frequencies) and mode shapes.

        Returns eigenvalues λ (rad²/s²) and eigenvectors as columns of matrix Φ.
        """
        # Use generalized eigenvalue solver for symmetric M, K
        eigvals, eigvecs = eigh(K, M)
        return eigvals, eigvecs

    @staticmethod
    def modal_superposition(M: np.ndarray, K: np.ndarray, C: np.ndarray,
                            F: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute response of MDOF system using modal analysis with proportional damping.
        Returns displacement vector u(t) shape (n_dof, len(t)).
        """
        # Compute mode shapes and frequencies
        eigvals, Phi = VibrationAnalyzer.mdof_modes(M, K)
        wn = np.sqrt(eigvals)
        n = M.shape[0]
        # Assume proportional damping: C = α*M + β*K (extract α, β from first two modes)
        # For demonstration, use α = 0, β = 0 (undamped) or a simple scalar damping ratio
        zeta = 0.02  # global damping ratio
        C_modal = 2 * zeta * np.diag(wn)  # diagonal modal damping matrix
        # Transform forces to modal coordinates
        Gamma = Phi.T @ F  # assuming F is constant (or evaluate each time step)
        # Modal response (single harmonic? We'll do transient via Newmark? Simple free response)
        # Here we compute free vibration response assuming initial displacement u0
        u0 = np.zeros(n)  # zero initial conditions
        v0 = np.zeros(n)
        u_t = np.zeros((n, len(t)))
        for i, ti in enumerate(t):
            u_modal = Phi @ (u0 + (Phi.T @ v0) / wn[:, None] * np.sin(wn[:, None] * ti) -
                             (Phi.T @ u0) * np.cos(wn[:, None] * ti))
            u_t[:, i] = u_modal.flatten()
        return u_t

    @staticmethod
    def state_space_matrices(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build state‑space matrices A, B for MDOF system.
        State vector x = [u; v]; input assumed zero for now.
        Returns A (2n x 2n) and B (2n x 0) (empty if no input).
        """
        n = M.shape[0]
        Minv = np.linalg.inv(M)
        A_top = np.hstack([np.zeros((n, n)), np.eye(n)])
        A_bottom = np.hstack([-Minv @ K, -Minv @ C])
        A = np.vstack([A_top, A_bottom])
        B = np.zeros((2 * n, 0))
        return A, B


# ----------------------------------------------------------------------
# 4. Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
class FatigueAnalyzer:
    """Tools for S‑N curve, Goodman/Gerber limits and Paris' law crack growth."""
    @staticmethod
    def basquin(sigma_f_prime: float, b: float, N: float) -> float:
        """
        Basquin relation: sigma_a = sigma_f' * (2*N)^b
        sigma_f_prime in Pa, N cycles, returns alternating stress sigma_a in Pa.
        """
        return sigma_f_prime * (2.0 * N) ** b

    @staticmethod
    def goodman(sigma_alt: float, sigma_mean: float,
                sigma_yield: float, sigma_uts: float) -> bool:
        """
        Check Goodman failure: (sigma_alt / sigma_f') + (sigma_mean / sigma_uts) <= 1
        sigma_f' is approximated by sigma_yield / 2 (typical for steels).
        Returns True if safe.
        """
        sigma_f_prime = sigma_yield / 2.0
        return (sigma_alt / sigma_f_prime) + (sigma_mean / sigma_uts) <= 1.0

    @staticmethod
    def gerber(sigma_alt: float, sigma_mean: float,
               sigma_f_prime: float, sigma_uts: float) -> bool:
        """
        Gerber parabola: (sigma_alt / sigma_f') + (sigma_mean / sigma_uts)^2 <= 1
        """
        return (sigma_alt / sigma_f_prime) + (sigma_mean / sigma_uts)**2 <= 1.0

    @staticmethod
    def paris_growth(a: float, da_dN: float, C: float, m: float,
                     Y: float, sigma_range: float, pi: float = np.pi) -> float:
        """
        Compute number of cycles to failure using Paris' law integration.
        Assuming constant stress intensity range ΔK = Y * sigma_range * sqrt(pi*a)
        and da/dN = C * (ΔK)^m.
        """
        # Integrate da / (C * (Y * sigma_range * sqrt(pi*a))^m) from a0 to a_crit
        # Use analytic solution for m != 2:
        # N = (a_crit^(1 - m/2) - a0^(1 - m/2)) / ((1 - m/2) * C * (Y*sigma_range*sqrt(pi))^m)
        factor = C * (Y * sigma_range * np.sqrt(pi)) ** m
        if np.isclose(m, 2.0):
            N = np.log(a / a0) / factor
        else:
            exponent = 1 - m / 2.0
            N = (a**exponent - a0**exponent) / (exponent * factor)
        return N


# ----------------------------------------------------------------------
# 5. Structural Optimization
# ----------------------------------------------------------------------
class StructuralOptimizer:
    """Optimization of beam cross‑section for minimum weight under constraints."""
    @staticmethod
    def beam_weight(side: float, rho: float, L: float) -> float:
        """Weight of a square‑section beam: W = rho * L * side^2."""
        return rho * L * side**2

    @staticmethod
    def stress_constraint(x: np.ndarray, P: float, L: float,
                          sigma_allow: float) -> float:
        """
        Check stress constraint for simply supported beam with central point load.
        x[0] = side length (m). Returns maximum factor of safety (allowable / actual) - 1.
        Positive value means constraint satisfied.
        """
        side = x[0]
        # Section modulus for square section: Z = side^3 / 6
        Z = side**3 / 6.0
        M_max = P * L / 4.0
        sigma_actual = M_max / Z
        return sigma_allow / sigma_actual - 1.0

    @staticmethod
    def optimize_gradient(P: float, L: float, rho: float,
                          sigma_allow: float, x0: np.ndarray) -> Dict[str, Any]:
        """
        Minimize beam weight using gradient-based optimization (SLSQP).
        Constraint: stress <= sigma_allow.
        Returns result dictionary from scipy.optimize.minimize.
        """
        cons = {'type': 'ineq',
                'fun': lambda x: StructuralOptimizer.stress_constraint(x, P, L, sigma_allow)}
        bounds = [(0.001, None)]  # side > 0
        result = minimize(lambda x: StructuralOptimizer.beam_weight(x[0], rho, L),
                          x0=x0, method='SLSQP', bounds=bounds, constraints=cons)
        return {'success': result.success, 'x': result.x, 'fun': result.fun,
                'message': result.message}

    @staticmethod
    def ga_optimize(P: float, L: float, rho: float,
                    sigma_allow: float, popsize: int = 10,
                    maxiter: int = 50) -> Dict[str, Any]:
        """
        Simple Genetic Algorithm (differential evolution) for beam sizing.
        Returns best solution and weight.
        """
        def objective(x):
            return StructuralOptimizer.beam_weight(x[0], rho, L)

        def constraint(x):
            return StructuralOptimizer.stress_constraint(x, P, L, sigma_allow)

        result = differential_evolution(objective, bounds=[(0.001, 0.5)],
                                        constraints={'type': 'ineq', 'fun': constraint},
                                        maxiter=maxiter, popsize=popsize,
                                        polish=True)
        return {'success': result.success, 'x': result.x, 'fun': result.fun,
                'message': result.message}


# ----------------------------------------------------------------------
# 6. Rotordynamics & Contact Mechanics
# ----------------------------------------------------------------------
class Rotordynamics:
    """Tools for Jeffcott rotor critical speed and basic contact analysis."""
    @staticmethod
    def jeffcott_critical_speed(E: float, I: float, m: float, L: float) -> float:
        """
        Critical speed (rad/s) of a simply supported Jeffcott rotor.
        Uses first mode constant 1.875^2.
        """
        return (1.875**2) * np.sqrt(E * I / (L**4 * m))

    @staticmethod
    def hertz_contact(E1: float, nu1: float, E2: float, nu2: float,
                     R1: float, R2: float, delta: float) -> Tuple[float, float]:
        """
        Hertzian contact between two spheres.

        Returns contact force P (N) and contact radius a (m).
        """
        # Effective modulus
        inv_E = (1 - nu1**2) / E1 + (1 - nu2**2) / E2
        E_star = 1.0 / inv_E if inv_E > 0 else 0.0
        R_eff = 1.0 / (1.0/R1 + 1.0/R2) if (R1 != 0 or R2 != 0) else 0.0
        # Contact radius a = sqrt(R_eff * delta)
        a = np.sqrt(R_eff * delta) if R_eff > 0 else 0.0
        # Force P = (4/3) * E_star * sqrt(R_eff) * delta^(3/2)
        P = (4.0/3.0) * E_star * np.sqrt(R_eff) * (delta ** 1.5) if R_eff > 0 else 0.0
        return P, a


class FailureTheory:
    """General failure theories based on stress tensor."""
    @staticmethod
    def max_principal(stress_tensor: np.ndarray) -> float:
        """
        Maximum principal stress (largest eigenvalue) for a 3x3 stress tensor.
        """
        eigvals = np.linalg.eigvalsh(stress_tensor)  # returns sorted ascending
        return eigvals[-1]


# ----------------------------------------------------------------------
# 7. Plotting utilities
# ----------------------------------------------------------------------
def plot_mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float,
                     save_path: str = None):
    """Plot Mohr's circle for given stress components."""
    x, y = StressTransform.mohr_circle(sigma_x, sigma_y, tau_xy)
    plt.figure()
    plt.plot(x, y, 'b-')
    plt.scatter([sigma_x, sigma_y], [tau_xy, -tau_xy], c='r')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Normal stress (Pa)')
    plt.ylabel('Shear stress (Pa)')
    plt.title("Mohr's Circle")
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_beam_mode(Phi: np.ndarray, title: str = "Mode shapes"):
    """Plot normalized mode shapes as vertical displacement vs node index."""
    plt.figure()
    for i in range(Phi.shape[0]):
        plt.plot(range(Phi.shape[1]), Phi[:, i], label=f'Mode {i+1}')
    plt.xlabel('Node index')
    plt.ylabel('Normalized displacement')
    plt.title(title)
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# 8. Demonstration and main block
# ----------------------------------------------------------------------
def main():
    """Run short demonstrations of each module."""
    print("=== Mechanical Engineering Toolbox Demo ===\n")

    # ---------- FEA ----------
    print("1. Finite Element Analysis (Truss)")
    E = DEFAULT_E
    A_truss = 1e-4  # m^2
    length_truss = 1.0
    nodes = 3
    truss_elements = [
        TrussElement(0, 1, E, A_truss),
        TrussElement(1, 2, E, A_truss)
    ]
    lengths = {(0,1): length_truss, (1,2): length_truss}
    K_truss = TrussElement.assemble_global(truss_elements, nodes, lengths)
    F_truss = np.zeros(2 * nodes)
    F_truss[2] = 1000.0  # apply vertical load at node 2
    u_truss = solve_fea(K_truss, F_truss)
    print(f"Displacements (u,v) = {u_truss}")
    print("FEA truss demo done.\n")

    # ---------- Stress transformation ----------
    print("2. Stress Transformation")
    sigma_x, sigma_y, tau_xy = 100e6, -50e6, 30e6
    theta = np.deg2rad(30)
    sx_p, sy_p, txy_p = StressTransform.transform(sigma_x, sigma_y, tau_xy, theta)
    print(f"Original stresses: σx={sigma_x/1e6:.2f} MPa, σy={sigma_y/1e6:.2f} MPa, τxy={tau_xy/1e6:.2f} MPa")
    print(f"Rotated (θ={np.rad2deg(theta):.1f}°): σx'={sx_p/1e6:.2f} MPa, σy'={sy_p/1e6:.2f} MPa, τxy'={txy_p/1e6:.2f} MPa")
    von_mises = StressTransform.von_mises(sigma_x, sigma_y, tau_xy)
    tresca = StressTransform.tresca(sigma_x, sigma_y, tau_xy
    print(f"Von Mises stress: {von_mises/1e6:.2f} MPa")
    print(f"Tresca stress: {tresca/1e6:.2f} MPa")
    # Plot Mohr's circle
    plot_mohr_circle(sigma_x, sigma_y, tau_xy, save_path=None)
    print("Stress transformation demo done.\n")

    # ---------- Vibrations ----------
    print("3. Vibration Analysis (SDOF)")
    m_sdoF = 10.0          # kg
    k_sdoF = 1e5           # N/m
    c_sdoF = 20.0          # N·s/m
    omega_n = VibrationAnalyzer.sdof_natural(m_sdoF, k_sdoF)
    print(f"Natural frequency ω_n = {omega_n:.2f} rad/s ({omega_n/(2*np.pi):.2f} Hz)")
    t = np.linspace(0, 2*np.pi/omega_n, 200)
    x, v = VibrationAnalyzer.sdof_response(m_sdoF, c_sdoF, k_sdoF, 500.0, omega_n*0.9, t)
    plt.figure()
    plt.plot(t, x, label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('SDOF Harmonic Response')
    plt.legend()
    plt.show()
    print("Vibration demo done.\n")

    # ---------- Fatigue ----------
    print("4. Fatigue Analysis")
    sigma_f_prime = DEFAULT_SIGY / 2.0  # typical fatigue strength limit
    b = -0.1  # Basquin exponent (example)
    N = 1e5   # cycles
    sigma_a = FatigueAnalyzer.basquin(sigma_f_prime, b, N)
    print(f"For N = {N:.0e} cycles, allowable alternating stress σ_a = {sigma_a/1e6:.2f} MPa")
    # Goodman check
    sigma_mean = 50e6
    safe_goodman = FatigueAnalyzer.goodman(sigma_a, sigma_mean, DEFAULT_SIGY, DEFAULT_SIGUTS)
    print(f"Goodman check: sigma_alt={sigma_a/1e6:.2f} MPa, sigma_mean={sigma_mean/1e6:.2f} MPa -> {'SAFE' if safe_goodman else 'FAIL'}")
    # Paris law estimate
    a0 = 1e-6   # initial crack length (m)
    a_crit = 10e-3  # critical crack length (m)
    C_paris = 1e-10   # material constant
    m_paris = 3.0
    Y = 1.12        # geometry factor
    sigma_range = sigma_a * 2.0
    cycles_to_failure = FatigueAnalyzer.paris_growth(a_crit, a0, C_paris, m_paris, Y, sigma_range)
    print(f"Paris law predicts {cycles_to_failure:.0f} cycles to failure from a0={a0*1e6:.1e} mm to a_crit={a_crit*1e3:.1f} mm")
    print("Fatigue demo done.\n")

    # ---------- Structural Optimization ----------
    print("5. Structural Optimization (Beam sizing)")
    P_load = 5000.0    # N
    L_beam = 2.0       # m
    rho_beam = DEFAULT_DENSITY
    sigma_allow = 0.6 * DEFAULT_SIGY
    # Gradient-based
    x0 = np.array([0.05])
    opt_grad = StructuralOptimizer.optimize_gradient(P_load, L_beam, rho_beam,
                                                     sigma_allow, x0)
    print(f"Gradient result: side = {opt_grad['x'][0]:.5f} m, weight = {opt_grad['fun']:.2f} kg")
    # Genetic algorithm
    opt_ga = StructuralOptimizer.ga_optimize(P_load, L_beam, rho_beam,
                                             sigma_allow, popsize=8, maxiter=30)
    print(f"GA result: side = {opt_ga['x'][0]:.5f} m, weight = {opt_ga['fun']:.2f} kg")
    print("Optimization demo done.\n")

    # ---------- Rotordynamics ----------
    print("6. Rotordynamics (Jeffcott rotor)")
    E_rotor = DEFAULT_E
    I_rotor = 1e-6   # m^4
    m_rotor = 5.0    # kg
    L_rotor = 0.5    # m
    omega_c = Rotordynamics.jeffcott_critical_speed(E_rotor, I_rotor, m_rotor, L_rotor)
    print(f"Critical speed ω_c = {omega_c:.2f} rad/s ({omega_c/(2*np.pi):.2f} Hz)")
    print("Rotordynamics demo done.\n")

    # ---------- Contact Mechanics ----------
    print("7. Contact Mechanics (Hertz)")
    E1, nu1 = 210e9, 0.3
    E2, nu2 = 110e9, 0.3
    R1, R2 = 0.1, 0.1  # m
    delta = 1e-5      # m approach
    P, a = Rotordynamics.hertz_contact(E1, nu1, E2, nu2, R1, R2, delta)
    print(f"Hertz contact: approach δ = {delta*1e6:.2f} µm -> force P = {P/1000:.2f} kN, contact radius a = {a*1000:.2f} mm")
    print("Contact demo done.\n")

    print("=== All demos completed successfully ===\n")


if __name__ == '__main__':
    main()