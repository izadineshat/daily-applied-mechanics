#!/usr/bin/env python3
"""
Mechanical Engineering Toolbox

Provides modules for:
- Finite Element Analysis (truss, beam, plate stiffness matrices)
- Stress/strain transformations, Mohr's circle, yield criteria
- Mechanical vibrations (SDOF, MDOF, modal analysis)
- Fatigue (S-N, Goodman, Gerber, Paris' law)
- Structural optimization (beam cross-section weight minimization)
- Rotordynamics, contact mechanics, failure theories
"""

import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Tuple

# Engineering defaults
E_DEFAULT = 210e9          # Young's modulus for steel [Pa]
NU_DEFAULT = 0.3           # Poisson's ratio
RHO_DEFAULT = 7850         # Density [kg/m^3]

class Material:
    """Simple isotropic material model."""
    def __init__(self, E: float = E_DEFAULT, nu: float = NU_DEFAULT, rho: float = RHO_DEFAULT):
        self.E = E
        self.nu = nu
        self.rho = rho

    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def bulk_modulus(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))


# ----------------------------------------------------------------------
# Finite Element Analysis (1D Truss, 2D Beam, 2D Plate)
# ----------------------------------------------------------------------
class TrussElement:
    """1D truss element (axial deformation only)."""
    def __init__(self, node_i: int, node_j: int, length: float, material: Material, area: float):
        self.node_i = node_i
        self.node_j = node_j
        self.length = length
        self.material = material
        self.area = area
        self.k_local = self._local_stiffness()
        self._T = self._shape_matrix()

    def _local_stiffness(self) -> np.ndarray:
        EA = self.material.E * self.area
        return (EA / self.length) * np.array([[1.0, -1.0], [-1.0, 1.0]])

    def _shape_matrix(self) -> np.ndarray:
        # Identity for local axis aligned with element
        return np.eye(2)

    def global_stiffness(self, angle_rad: float = 0.0) -> np.ndarray:
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        # Transformation matrix from global (x_i, y_i, x_j, y_j) to local (u_i, u_j)
        T = np.array([[c, s, 0.0, 0.0],
                      [0.0, 0.0, c, s]])
        k_global = T.T @ self.k_local @ T
        return k_global


class BeamElement:
    """2D Euler-Bernoulli beam element (plane frame)."""
    def __init__(self, node_i: int, node_j: int, length: float, material: Material,
                 A: float, I: float):
        self.node_i = node_i
        self.node_j = node_j
        self.length = length
        self.material = material
        self.A = A
        self.I = I
        self.k_local = self._local_stiffness()

    def _local_stiffness(self) -> np.ndarray:
        L = self.length
        E = self.material.E
        A = self.A
        I = self.I
        # Local DOFs: [axial_i, rotation_i, axial_j, rotation_j]
        k = np.zeros((4, 4))
        k[0, 0] = E * A / L
        k[0, 2] = -E * A / L
        k[1, 1] = 12.0 * E * I / L**3
        k[1, 2] = 6.0 * E * I / L**2
        k[1, 3] = -12.0 * E * I / L**3
        k[2, 0] = -E * A / L
        k[2, 1] = 6.0 * E * I / L**2
        k[2, 2] = E * A / L
        k[2, 3] = -6.0 * E * I / L**2
        k[3, 1] = -12.0 * E * I / L**3
        k[3, 2] = -6.0 * E * I / L**2
        k[3, 3] = 12.0 * E * I / L**3
        return k

    def global_stiffness(self, angle_rad: float = 0.0) -> np.ndarray:
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        # Transformation matrix for beam (6 DOFs total)
        T = np.array([
            [c, 0.0, 0.0, -s, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [s, 0.0, 0.0, c, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        ])
        k_global = T.T @ self.k_local @ T
        return k_global


class PlateElement:
    """2D 4-node quadrilateral plane stress element."""
    def __init__(self, nodes: List[Tuple[float, float]], material: Material, thickness: float):
        self.nodes = np.asarray(nodes, dtype=float)
        self.material = material
        self.thickness = thickness
        self.ks = self._compute_stiffness_matrix()

    def _shape_functions(self, xi: float, eta: float) -> List[float]:
        return [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta)
        ]

    def _jacobian(self, xi: float, eta: float):
        dN_dxi = np.array([
            -0.25 * (1.0 - eta),
             0.25 * (1.0 - eta),
             0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta)
        ])
        dN_deta = np.array([
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi),
             0.25 * (1.0 - xi)
        ])
        J = np.zeros((2, 2))
        J[0, :] = np.dot(dN_dxi, self.nodes[:, 0])
        J[1, :] = np.dot(dN_deta, self.nodes[:, 1])
        return J, dN_dxi, dN_deta

    def _compute_stiffness_matrix(self) -> np.ndarray:
        # 2x2 Gauss quadrature
        pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]
        thickness = self.thickness
        E = self.material.E
        nu = self.material.nu
        G = self.material.shear_modulus

        # Plane stress constitutive matrix
        D = (E / (1.0 - nu**2)) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])

        k = np.zeros((8, 8))
        for xi in pts:
            for eta in pts:
                J, dN_dxi, dN_deta = self._jacobian(xi, eta)
                detJ = np.linalg.det(J)

                B = np.zeros((3, 8))
                for i in range(4):
                    dN_dx = (dN_dxi[i] * J[1, 1] - dN_deta[i] * J[0, 1]) / detJ
                    dN_dy = (-dN_dxi[i] * J[1, 0] + dN_deta[i] * J[0, 0]) / detJ
                    B[0, 2*i]   = dN_dx
                    B[1, 2*i+1] = dN_dy
                    B[2, 2*i]   = dN_dy
                    B[2, 2*i+1] = dN_dx

                B_T = B.T
                k += (B_T @ D @ B) * detJ * thickness

        return k


def assemble_global_stiffness(elements: List) -> np.ndarray:
    """Assemble global stiffness matrix from element contributions.
    Assumes 2 DOFs per node (e.g., truss)."""
    max_node = max(max(el.node_i, el.node_j) for el in elements)
    size = 2 * (max_node + 1)
    K_global = np.zeros((size, size))

    for el in elements:
        k_el = el.global_stiffness()
        # Map local DOFs to global indices: node*2 for x, node*2+1 for y
        idx = []
        for node in [el.node_i, el.node_j]:
            idx.extend([node*2, node*2+1])
        for i_loc, i_glob in enumerate(idx):
            for j_loc, j_glob in enumerate(idx):
                K_global[i_glob, j_glob] += k_el[i_loc, j_loc]

    return K_global


def apply_boundary_conditions(K_global: np.ndarray, bc: dict):
    """Partition global stiffness matrix for prescribed DOFs.
    Returns K_ff (free-free), free_dofs list, fixed_dofs list."""
    all_dofs = np.arange(K_global.shape[0])
    free_dofs = [i for i in all_dofs if i not in bc]
    fixed_dofs = list(bc.keys())
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
    return K_ff, free_dofs, fixed_dofs


# ----------------------------------------------------------------------
# Stress & Strain Transformations
# ----------------------------------------------------------------------
class StressTransformer:
    """Utilities for stress transformation and failure criteria."""
    @staticmethod
    def transform_stress(sigma_xx: float, sigma_yy: float, tau_xy: float,
                         theta: float) -> Tuple[float, float, float]:
        """Return stresses in a coordinate system rotated by angle theta (rad)."""
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_x = 0.5 * (sigma_xx + sigma_yy) + 0.5 * (sigma_xx - sigma_yy) * c**2 + tau_xy * s * c
        sigma_y = 0.5 * (sigma_xx + sigma_yy) - 0.5 * (sigma_xx - sigma_yy) * c**2 - tau_xy * s * c
        tau_xy_r = -0.5 * (sigma_xx - sigma_yy) * s * c + tau_xy * (c**2 - s**2)
        return sigma_x, sigma_y, tau_xy_r

    @staticmethod
    def mohrs_circle(sigma_xx: float, sigma_yy: float, tau_xy: float):
        """Return (center, radius) for Mohr's circle. Center is (mean stress, 0)."""
        center = ((sigma_xx + sigma_yy) / 2.0, 0.0)
        radius = np.sqrt(((sigma_xx - sigma_yy) / 2.0)**2 + tau_xy**2)
        return center, radius

    @staticmethod
    def von_mises(sigma_xx: float, sigma_yy: float, tau_xy: float) -> float:
        return np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3.0 * tau_xy**2)

    @staticmethod
    def tresca(sigma_xx: float, sigma_yy: float, tau_xy: float) -> float:
        """Tresca equivalent stress (max shear stress) for plane stress."""
        sigma_avg = (sigma_xx + sigma_yy) / 2.0
        radius = np.sqrt(((sigma_xx - sigma_yy) / 2.0)**2 + tau_xy**2)
        sigma1 = sigma_avg + radius
        sigma2 = sigma_avg - radius
        return np.max(np.abs([sigma1 - sigma2, sigma1, sigma2]))


# ----------------------------------------------------------------------
# Mechanical Vibrations
# ----------------------------------------------------------------------
def sdof_free_response(m: float, c: float, k: float, x0: float,
                        v0: float, t: np.ndarray) -> np.ndarray:
    """Free response of an SDOF system (mass m, damping c, stiffness k)."""
    omega_n = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    if zeta < 1.0:                     # underdamped
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)
        x = np.exp(-zeta * omega_n * t) * (
            x0 * np.cos(omega_d * t) + ((v0 + zeta * omega_n * x0) / omega_d) * np.sin(omega_d * t))
    elif zeta == 1.0:                  # critically damped
        x = np.exp(-omega_n * t) * (x0 + (v0 + omega_n * x0) * t)
    else:                              # overdamped
        s1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1.0))
        s2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1.0))
        A1 = (v0 - s2 * x0) / (s1 - s2)
        A2 = (v0 - s1 * x0) / (s2 - s1)
        x = A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)
    return x


def sdof_forced_response(m: float, c: float, k: float, F0: float,
                          omega: float, phi: float = 0.0,
                          t: np.ndarray = None, npoints: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Steady-state harmonic forced response of an SDOF system."""
    if t is None:
        omega_n = np.sqrt(k / m)
        period = 2.0 * np.pi / omega_n
        t = np.linspace(0.0, 5.0 * period, npoints)

    H = F0 / (k - m * omega**2 + 1j * c * omega) * np.exp(1j * phi)
    x = np.real(H * np.exp(1j * omega * t))
    return t, x


def modal_analysis(K_global: np.ndarray, M_global: np.ndarray):
    """Compute natural frequencies (Hz) and mode shapes from K and M."""
    eigvals, eigvecs = eig(K_global, M_global)
    idx = np.isreal(eigvals) & (eigvals > 0)
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])

    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    frequencies = np.sqrt(eigvals) / (2.0 * np.pi)   # Hz
    return frequencies, eigvecs


def mdof_state_space(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                     x0: np.ndarray, v0: np.ndarray, t: np.ndarray):
    """Solve MDOF system using state-space ODE integration."""
    n = M.shape[0]
    I = np.eye(n)
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((n, n)), I],
                  [-Minv @ K, -Minv @ C]])

    y0 = np.concatenate([x0, v0])
    def ode_func(t, y):
        return A @ y

    sol = solve_ivp(ode_func, [t[0], t[-1]], y0, t_eval=t, method='RK45')
    x = sol.y[:n, :]
    return sol.t, x


# ----------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
def sn_curve(S_max: float, S_min: float, N_ref: float, m: float = 3.0) -> float:
    """Basquin S-N curve (fully reversed loading). Returns cycles to failure."""
    sigma_a = (S_max - S_min) / 2.0
    if sigma_a <= 0:
        return np.inf
    S_f_prime = S_max * (2.0 * N_ref) ** (-m)
    N = (S_f_prime / sigma_a) ** (1.0 / m) / 2.0
    return N


def goodman_diagram(S_alt: float, S_mean: float, S_u: float, S_e: float) -> float:
    """Goodman safety factor: 1 / ((S_alt/S_e) + (S_mean/S_u))."""
    return 1.0 / (S_alt / S_e + S_mean / S_u)


def gerber_diagram(S_alt: float, S_mean: float, S_u: float, S_e: float) -> float:
    """Gerber safety factor: 1 / ((S_alt/S_e) + (S_mean/S_u)**2)."""
    return 1.0 / (S_alt / S_e + (S_mean / S_u)**2)


def paris_law(da: float, C: float, m: float, DeltaK: float) -> float:
    """Paris' law: da/dN = C * (DeltaK)^m.
    Returns number of cycles for a given crack increment da."""
    return da / (C * (DeltaK)**m)


# ----------------------------------------------------------------------
# Structural Optimization (Beam Cross‑section Weight Minimization)
# ----------------------------------------------------------------------
def beam_weight(A: float, I: float, length: float, material: Material) -> float:
    """Simple weight estimate: volume * density."""
    # Approximate cross‑section perimeter for a I‑shaped section.
    # For demonstration, use A as dominant term.
    return A * length * material.rho


def optimize_beam_cross_section(length: float = 1.0,
                                material: Material = Material(),
                                target_deflection: float = 0.001,
                                force: float = 1000.0):
    """Minimize weight w.r.t. area A and moment of inertia I (I = A**2)."""
    def objective(x: np.ndarray) -> float:
        A, I = x
        return beam_weight(A, I, length, material)

    def constraint_deflection(x: np.ndarray) -> float:
        A, I = x
        # Cantilever tip deflection under tip load: delta = F * L^3 / (3 * E * I)
        delta = force * length**3 / (3.0 * material.E * I)
        return target_deflection - delta   # must be >= 0

    cons = {'type': 'ineq', 'fun': constraint_deflection}
    bounds = [(1e-6, 1e-3), (1e-12, 1e-9)]
    x0 = [5e-5, 1e-10]

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return result


# ----------------------------------------------------------------------
# Rotordynamics, Contact Mechanics, Failure Theories
# ----------------------------------------------------------------------
class Rotordynamics:
    """Simplified rotor critical speed and gyration calculations."""
    @staticmethod
    def whirling_frequency(R: float, E: float, I: float, L: float, J: float = 0.0) -> float:
        """First critical speed of a rotating shaft (cantilever beam with tip mass J)."""
        k_eff = 3.0 * E * I / L**3
        if J > 0.0:
            omega_n = np.sqrt(k_eff / J)
        else:
            omega_n = np.sqrt(k_eff / 1e-12)   # avoid division by zero
        return omega_n / (2.0 * np.pi)        # Hz

    @staticmethod
    def gyration_radius(mass: float, inertia: float) -> float:
        """Radius of gyration."""
        return np.sqrt(inertia / mass)


class ContactMechanic:
    """Basic contact mechanics (Hertzian contact for spheres)."""
    @staticmethod
    def hertzian_contact(E1: float, nu1: float, E2: float, nu2: float,
                         R1: float, R2: float, load: float):
        """Return contact radius a and maximum pressure p0."""
        R = 1.0 / (1.0 / R1 + 1.0 / R2)
        E_star = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)
        a = (3.0 * load * R / (4.0 * E_star)) ** 0.25 * np.sqrt(R)
        p0 = 2.0 * load / (np.pi * a**2)
        return a, p0


def failure_theories(stress_tensor: np.ndarray):
    """Compute max normal stress, von Mises, and Tresca from a 3x3 stress tensor."""
    eigvals = np.linalg.eigvalsh(stress_tensor)   # assume symmetric
    sigma_max = np.max(eigvals)
    sigma_min = np.min(eigvals)

    s1, s2, s3 = eigvals
    vm = np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
    t = np.max(np.abs([s1 - s2, s2 - s3, s3 - s1]))
    return sigma_max, vm, t


# ----------------------------------------------------------------------
# Demonstration / Example Usage
# ----------------------------------------------------------------------
def demo_fea():
    """Simple truss FE analysis."""
    mat = Material()
    node_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    e0 = TrussElement(0, 1, length=1.0, material=mat, area=0.01)
    e1 = TrussElement(1, 2, length=1.0, material=mat, area=0.01)
    elements = [e0, e1]

    K = assemble_global_stiffness(elements)
    bc = {0: 0.0, 1: 0.0}      # fix node 0 DOFs
    n_dofs = K.shape[0]
    F = np.zeros(n_dofs)
    F[5] = -1000.0              # downward load at node 2 y-DOF

    free_dofs = [i for i in range(n_dofs) if i not in bc]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fc = K[np.ix_(free_dofs, list(bc.keys()))]
    u_c = np.array(list(bc.values()))

    u_f = np.linalg.solve(K_ff, -K_fc @ u_c + F[free_dofs])
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f

    print("Truss displacements (x0, y0, x1, y1, x2, y2):", u)
    return K


def demo_stress_transformation():
    """Stress transformation and Mohr's circle."""
    st = StressTransformer()
    sigma_xx, sigma_yy, tau_xy = 100.0, 40.0, 30.0
    angles = np.linspace(0.0, np.pi/2, 9)

    for th in angles:
        sx, sy, txy = st.transform_stress(sigma_xx, sigma_yy, tau_xy, th)
        print(f"θ={np.degrees(th):.1f}°: σx={sx:.2f}, σy={sy:.2f}, τxy={txy:.2f}")

    # Plot Mohr's circle
    center, radius = st.mohrs_circle(sigma_xx, sigma_yy, tau_xy)
    theta_mohr = np.linspace(0.0, 2.0 * np.pi, 200)
    x = center[0] + radius * np.cos(theta_mohr)
    y = radius * np.sin(theta_mohr)

    plt.figure()
    plt.plot(x, y, label='Mohr’s circle')
    plt.scatter([sigma_xx, sigma_yy], [tau_xy, -tau_xy], c='r')
    plt.xlabel('σ')
    plt.ylabel('τ')
    plt.title('Mohr’s Circle')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()


def demo_vibrations():
    """SDOF free response and 2-DOF modal analysis."""
    # SDOF free response
    m, c, k = 10.0, 20.0, 1e5
    x0, v0 = 0.01, 0.0
    t = np.linspace(0.0, 2.0, 1000)
    x = sdof_free_response(m, c, k, x0, v0, t)

    plt.figure()
    plt.plot(t, x)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('SDOF Free Response')
    plt.grid()
    plt.show()

    # 2-DOF modal analysis
    M = np.array([[1.0, 0.0], [0.0, 2.0]])
    K = np.array([[20.0, -5.0], [-5.0, 10.0]])

    freq, modes = modal_analysis(K, M)
    print("Natural frequencies (Hz):", freq)

    for i, mode in enumerate(modes.T):
        plt.figure()
        plt.plot([0.0, 1.0], [mode[0], mode[1]], 'o-')
        plt.title(f'Mode {i+1} shape')
        plt.grid()
    plt.show()


def demo_fatigue():
    """S-N curve, Goodman and Gerber diagrams."""
    S_max = 600.0    # MPa
    S_min = -600.0
    N_ref = 1e5
    m = 5.0

    N = sn_curve(S_max, S_min, N_ref, m)
    sigma_a = (S_max - S_min) / 2.0
    print(f"S-N: amplitude {sigma_a:.2f} MPa => {N:.2e} cycles to failure")

    S_u = 1000.0     # ultimate tensile strength
    S_e = S_u / 2.0  # approximate endurance limit
    S_alt = sigma_a
    S_mean = (S_max + S_min) / 2.0

    sf_goodman = goodman_diagram(S_alt, S_mean, S_u, S_e)
    sf_gerber = gerber_diagram(S_alt, S_mean, S_u, S_e)
    print(f"Goodman safety factor: {sf_goodman:.3f}")
    print(f"Gerber safety factor: {sf_gerber:.3f}")


def demo_optimization():
    """Beam cross‑section weight minimization."""
    result = optimize_beam_cross_section()
    print("Optimization result:")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Optimal A:", result.x[0], "I:", result.x[1])
    print("Weight:", result.fun)


def demo_rotordynamics():
    """Critical speed estimate and Hertzian contact."""
    rd = Rotordynamics()
    L = 0.5
    d = 0.05
    I = np.pi * d**4 / 64.0
    omega_crit = rd.whirling_frequency(R=0.025, E=210e9, I=I, L=L, J=0.1)
    print(f"Critical speed ≈ {omega_crit:.2f} Hz")

    a, p0 = ContactMechanic.hertzian_contact(210e9, 0.3, 210e9, 0.3, 0.1, 0.1, 1000.0)
    print(f"Hertzian contact radius: {a:.5e} m, max pressure: {p0:.2e} Pa")


def demo_failure_theories():
    """Evaluate maximum normal stress, von Mises and Tresca for a stress state."""
    sigma = np.array([[100.0, 20.0, 0.0],
                      [20.0, -50.0, 0.0],
                      [0.0, 0.0, 0.0]])
    max_normal, vm, t = failure_theories(sigma)
    print(f"Max normal stress: {max_normal:.2f} MPa")
    print(f"Von Mises stress: {vm:.2f} MPa")
    print(f"Tresca stress: {t:.2f} MPa")


if __name__ == '__main__':
    # Run a subset of demonstrations (uncomment to test all)
    demo_fea()
    demo_stress_transformation()
    demo_vibrations()
    demo_fatigue()
    demo_optimization()
    demo_rotordynamics()
    demo_failure_theories()