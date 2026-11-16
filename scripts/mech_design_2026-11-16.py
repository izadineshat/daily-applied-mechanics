#!/usr/bin/env python3
"""
Production-ready mechanical engineering toolkit.

Features:
- 1D truss element stiffness matrix
- 2D Euler-Bernoulli beam element stiffness matrix
- Plate bending stiffness (Kirchhoff-Love)
- Stress transformation and Mohr's circle visualization
- Von Mises and Tresca failure criteria
- SDOF and MDOF damped vibration analysis
- Fatigue life prediction (S-N curves, Goodman diagram)
- Structural optimization (weight minimization of rectangular beam)
- Rotor critical speed (whirling shaft) calculation
- Simple contact pressure calculation

Dependencies:
    numpy, scipy, matplotlib
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Constants (realistic engineering defaults)
# ----------------------------------------------------------------------
E_DEFAULT = 210e9          # Pa, steel
NU_DEFAULT = 0.3           # Poisson's ratio
RHO_DEFAULT = 7850.0       # kg/m^3, steel
G_DEFAULT = E_DEFAULT / (2 * (1 + NU_DEFAULT))  # Pa, shear modulus


# ----------------------------------------------------------------------
# Truss Element (1D bar)
# ----------------------------------------------------------------------
class TrussElement:
    """
    2-node axial truss element stiffness matrix (4x4).
    Nodes are defined by their 2D coordinates (x, y).
    """

    def __init__(self, node1: np.ndarray, node2: np.ndarray, area: float, E: float = E_DEFAULT):
        """
        Parameters
        ----------
        node1, node2 : np.ndarray
            2-element arrays with coordinates [x, y].
        area : float
            Cross-sectional area (m^2).
        E : float
            Young's modulus (Pa).
        """
        self.node1 = np.asarray(node1, dtype=float)
        self.node2 = np.asarray(node2, dtype=float)
        self.area = float(area)
        self.E = float(E)
        self.L = np.linalg.norm(self.node2 - self.node1)
        self.c = (self.node2 - self.node1) / self.L  # direction cosine vector

    def stiffness(self) -> np.ndarray:
        """Return the 4x4 element stiffness matrix."""
        k_ax = self.E * self.area / self.L
        c_out = np.outer(self.c, self.c)
        K = k_ax * np.block([
            [c_out, -c_out],
            [-c_out, c_out]
        ])
        return K.astype(float)


# ----------------------------------------------------------------------
# Beam Element (Euler-Bernoulli, 2 nodes, 2 DOFs each)
# ----------------------------------------------------------------------
class BeamElement:
    """
    2-node Euler-Bernoulli beam element stiffness matrix (4x4).
    DOFs: [v1, theta1, v2, theta2] (transverse displacement, rotation).
    """

    def __init__(self, node1: np.ndarray, node2: np.ndarray,
                 E: float = E_DEFAULT, I: float = 1e-6, L: float = 1.0):
        """
        Parameters
        ----------
        node1, node2 : np.ndarray
            2-element coordinate arrays.
        E : float
            Young's modulus.
        I : float
            Second moment of area.
        L : float
            Element length.
        """
        self.node1 = np.asarray(node1, dtype=float)
        self.node2 = np.asarray(node2, dtype=float)
        self.E = float(E)
        self.I = float(I)
        self.L = float(L)

    def stiffness(self) -> np.ndarray:
        """Return the 4x4 element stiffness matrix."""
        # Precomputed constants
        k1 = self.E * self.I / self.L**3
        k2 = 12 * self.E * self.I / self.L**3
        k3 = 4 * self.E * self.I / self.L**3
        k4 = 6 * self.E * self.I / self.L**3

        K = np.array([
            [k1, 0, 0, 0],
            [0, k2, 0, 0],
            [0, 0, k3, 0],
            [0, 0, 0, k4]
        ], dtype=float)

        # Sign change for the second node due to orientation
        sign = np.array([1, -1, 1, -1])
        K = np.diag(sign) @ K @ np.diag(sign)
        return K


# ----------------------------------------------------------------------
# Plate Bending (Kirchhoff-Love plate)
# ----------------------------------------------------------------------
def plate_bending_stiffness(E: float, h: float, nu: float) -> float:
    """
    Compute the plate bending stiffness D = Eh^3 / [12(1-nu^2)].

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    h : float
        Plate thickness (m).
    nu : float
        Poisson's ratio.

    Returns
    -------
    D : float
        Bending stiffness (N·m).
    """
    return E * h**3 / (12 * (1 - nu**2))


# ----------------------------------------------------------------------
# Stress Transformation
# ----------------------------------------------------------------------
class StressTransformer:
    """
    Transform 2D stress components from one coordinate system to another.
    """

    @staticmethod
    def transform(sigma_x: float, sigma_y: float, tau_xy: float,
                  theta_deg: float) -> tuple[float, float, float]:
        """
        Transform stresses by theta (degrees, positive CCW).

        Returns (sigma_x', sigma_y', tau_xy').
        """
        theta = np.deg2rad(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)

        sigma_x_prime = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
        sigma_y_prime = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
        tau_xy_prime = (sigma_y - sigma_x) * s * c + tau_xy * (c**2 - s**2)
        return sigma_x_prime, sigma_y_prime, tau_xy_prime

    @staticmethod
    def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float) -> tuple[float, float]:
        """
        Compute principal stresses sigma1, sigma2.
        """
        sigma_avg = (sigma_x + sigma_y) / 2
        radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        return sigma_avg + radius, sigma_avg - radius

    @staticmethod
    def mohr_circle_params(sigma_x: float, sigma_y: float, tau_xy: float):
        """
        Return center and radius of Mohr's circle.
        """
        sigma_avg, radius = StressTransformer.principal_stresses(sigma_x, sigma_y, tau_xy)
        return sigma_avg, radius


# ----------------------------------------------------------------------
# Failure Criteria
# ----------------------------------------------------------------------
class FailureCriteria:
    """
    Static failure criteria for 2D stress states.
    """

    @staticmethod
    def von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Von Mises equivalent stress."""
        return np.sqrt(sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2)

    @staticmethod
    def tresca(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Tresca equivalent stress (max principal stress difference)."""
        sigma1, sigma2 = StressTransformer.principal_stresses(sigma_x, sigma_y, tau_xy)
        return max(abs(sigma1 - sigma2), abs(sigma2 - (sigma_x + sigma_y) / 2), abs(sigma1 - (sigma_x + sigma_y) / 2))

    @staticmethod
    def maximum_normal(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Maximum normal stress (absolute)."""
        return max(abs(sigma_x), abs(sigma_y), abs(sigma_x + sigma_y) / 2, np.sqrt(sigma_x**2 + 3 * tau_xy**2))


# ----------------------------------------------------------------------
# Vibration - SDOF Damped System
# ----------------------------------------------------------------------
class SDOFDamped:
    """
    Analytical solution for a damped SDOF system.
    """

    def __init__(self, m: float, c: float, k: float,
                 x0: float = 0.0, v0: float = 0.0, dt: float = 0.001):
        """
        Parameters
        ----------
        m, c, k : float
            Mass, damping coefficient, stiffness.
        x0, v0 : float
            Initial displacement and velocity.
        dt : float
            Time step for numerical output.
        """
        self.m = m
        self.c = c
        self.k = k
        self.x0 = x0
        self.v0 = v0
        self.dt = dt

        # Natural frequency and damping ratio
        self.omega_n = np.sqrt(k / m)
        self.zeta = c / (2 * np.sqrt(k * m))

        # Precompute constants for underdamped case (zeta < 1)
        if self.zeta < 1:
            self.omega_d = self.omega_n * np.sqrt(1 - self.zeta**2)
            self.ratio = np.exp(-self.zeta * self.omega_n * np.arange(0, 10, self.dt))

    def displacement(self, t: float) -> float:
        """Displacement at time t (s)."""
        if self.zeta >= 1:
            # Critically or over-damped: use eigenvalue approach (simplified)
            return np.exp(-self.zeta * self.omega_n * t) * (
                self.x0 + (self.v0 + self.zeta * self.omega_n * self.x0) / (self.omega_n * np.sqrt(self.zeta**2 - 1)) *
                np.sinh(self.omega_n * np.sqrt(self.zeta**2 - 1) * t)
            )
        else:
            # Underdamped
            decay = np.exp(-self.zeta * self.omega_n * t)
            oscillation = (self.x0 * np.cos(self.omega_d * t) +
                           (self.v0 / self.omega_d + self.zeta * self.x0) * np.sin(self.omega_d * t))
            return decay * oscillation


# ----------------------------------------------------------------------
# Modal Analysis (MDOF)
# ----------------------------------------------------------------------
class ModalSystem:
    """
    Linear modal analysis for MDOF system: M * x_ddot + C * x_dot + K * x = 0.
    Uses undamped free vibration (eigenvalue) analysis.
    """

    def __init__(self, M: np.ndarray, K: np.ndarray):
        """
        Parameters
        ----------
        M, K : np.ndarray
            Mass and stiffness matrices (square, same shape).
        """
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.eigvals, self.eigvecs = eigh(self.K, self.M)  # eigenvalues sorted ascending

    def natural_frequencies(self) -> np.ndarray:
        """Return natural frequencies (rad/s)."""
        return np.sqrt(np.real(self.eigvals))

    def mode_shapes(self) -> np.ndarray:
        """Return mode shapes (columns)."""
        return self.eigvecs


# ----------------------------------------------------------------------
# Fatigue Life Prediction
# ----------------------------------------------------------------------
class FatigueLife:
    """
    Simple fatigue life estimator using an S-N curve and Goodman correction.
    """

    def __init__(self, sigma_a: float, sigma_f: float, b: float, c: float, sigma_u: float):
        """
        Parameters
        ----------
        sigma_a : float
            Stress amplitude (Pa).
        sigma_f : float
            Fatigue strength coefficient (Pa).
        b : float
            S-N curve slope (negative).
        c : float
            Goodman intercept (dimensionless).
        sigma_u : float
            Ultimate tensile strength (Pa).
        """
        self.sigma_a = sigma_a
        self.sigma_f = sigma_f
        self.b = b
        self.c = c
        self.sigma_u = sigma_u

    def cycles_to_failure(self) -> float:
        """Estimate cycles to failure using the S-N curve."""
        # S-N relation: sigma_a = sigma_f * N^b  => N = (sigma_a / sigma_f)^(1/b)
        if self.sigma_a <= 0:
            return np.inf
        N = (self.sigma_a / self.sigma_f) ** (1.0 / self.b)
        return N

    def allowable_cycles(self) -> float:
        """Apply Goodman line to compute allowable cycles."""
        sigma_m = self.sigma_a  # for fully reversed loading sigma_m = 0, but we keep general
        sigma_a_goodman = self.sigma_f * (1 - sigma_m / self.sigma_u)
        if sigma_a_goodman <= 0:
            return 0.0
        N_good = (sigma_a_goodman / self.sigma_f) ** (1.0 / self.b)
        return N_good


# ----------------------------------------------------------------------
# Structural Optimization (Weight Minimization)
# ----------------------------------------------------------------------
def beam_weight_objective(x: np.ndarray, length: float, load: float,
                          E: float, nu: float, rho: float) -> float:
    """
    Objective: minimize weight = rho * A * L.
    Variables x = [width, depth] (m).
    Constraints: bending stress <= allowable (using simply supported beam).
    """
    w, h = x[0], x[1]
    A = w * h
    weight = rho * A * length

    # Bending moment for central point load on simply supported beam
    M = load * length / 4.0
    # Section properties
    I = w * h**3 / 12.0
    c = h / 2.0
    sigma_max = M * c / I

    # Allowable stress (use material yield, approximated as 0.5*E)
    sigma_allow = 0.5 * E
    constraint = sigma_allow - sigma_max
    if constraint <= 0:
        # Return large penalty
        return weight + 1e6 * (constraint**2)

    return weight


def optimize_beam_weight(length: float, load: float,
                         E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                         rho: float = RHO_DEFAULT,
                         bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.01, 0.5), (0.01, 0.5))
                         ) -> dict:
    """
    Minimize weight of a rectangular beam subject to stress constraint.

    Returns optimization result dictionary.
    """
    def obj_func(x):
        return beam_weight_objective(x, length, load, E, nu, rho)

    cons = [{'type': 'ineq', 'fun': lambda x: beam_weight_objective(x, length, load, E, nu, rho) - 1e6}]
    # Actually we embed constraint inside objective to avoid complex constraints.

    res = minimize(obj_func, x0=[0.1, 0.1], method='L-BFGS-B', bounds=bounds)
    return {
        'success': res.success,
        'message': res.message,
        'x_opt': res.x,
        'weight': res.fun,
        'sigma_max': (load * length / 4.0) * (res.x[1] / 2) / (res.x[0] * res.x[1]**3 / 12.0)
    }


# ----------------------------------------------------------------------
# Rotor Critical Speed (Whirling Shaft)
# ----------------------------------------------------------------------
def rotor_critical_speed(E: float, I: float, m: float, L: float) -> float:
    """
    Estimate first critical speed using Rayleigh method (simply supported beam).

    Parameters
    ----------
    E : float
        Young's modulus.
    I : float
        Bending stiffness (m^4).
    m : float
        Mass per unit length (kg/m).
    L : float
        Shaft length (m).

    Returns
    -------
    omega_cr : float
        Critical angular speed (rad/s).
    """
    # Rayleigh formula: omega_cr = pi * sqrt(EI / (m L^4))
    return np.pi * np.sqrt(E * I / (m * L**4))


# ----------------------------------------------------------------------
# Contact Pressure (simple)
# ----------------------------------------------------------------------
def contact_pressure(load: float, area: float) -> float:
    """
    Compute average contact pressure.

    Parameters
    ----------
    load : float
        Normal force (N).
    area : float
        Contact area (m^2).

    Returns
    -------
    pressure : float
        Pressure (Pa).
    """
    if area <= 0:
        raise ValueError("Contact area must be positive.")
    return load / area


# ----------------------------------------------------------------------
# Demonstration / Main Block
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # ---- Truss Element ----
    n1 = np.array([0.0, 0.0])
    n2 = np.array([1.0, 0.0])
    truss = TrussElement(n1, n2, area=0.01, E=E_DEFAULT)
    K_truss = truss.stiffness()
    print("Truss element stiffness matrix (4x4):")
    print(K_truss)

    # ---- Beam Element ----
    bn1 = np.array([0.0, 0.0])
    bn2 = np.array([1.0, 0.0])
    beam = BeamElement(bn1, bn2, E=E_DEFAULT, I=1e-6, L=1.0)
    K_beam = beam.stiffness()
    print("\nBeam element stiffness matrix (4x4):")
    print(K_beam)

    # ---- Stress Transformation & Mohr's Circle ----
    sigma_x, sigma_y, tau_xy = 100e6, 20e6, 30e6  # Pa
    theta = 30.0
    sigma_x_t, sigma_y_t, tau_xy_t = StressTransformer.transform(sigma_x, sigma_y, tau_xy, theta)
    print(f"\nTransformed stresses (theta={theta} deg): sigma_x'={sigma_x_t/1e6:.2f} MPa, "
          f"sigma_y'={sigma_y_t/1e6:.2f} MPa, tau_xy'={tau_xy_t/1e6:.2f} MPa")
    sigma1, sigma2 = StressTransformer.principal_stresses(sigma_x, sigma_y, tau_xy)
    print(f"Principal stresses: sigma1={sigma1/1e6:.2f} MPa, sigma2={sigma2/1e6:.2f} MPa")
    center, radius = StressTransformer.mohr_circle_params(sigma_x, sigma_y, tau_xy)
    print(f"Mohr's circle center={center/1e6:.2f} MPa, radius={radius/1e6:.2f} MPa")

    # Plot Mohr's circle
    fig, ax = plt.subplots()
    theta_mohr = np.linspace(0, 2 * np.pi, 200)
    cx = center
    R = radius
    sigma_vals = cx + R * np.cos(theta_mohr)
    tau_vals = R * np.sin(theta_mohr)
    ax.plot(sigma_vals, tau_vals, 'b-')
    ax.plot([cx - R, cx + R], [0, 0], 'r--')
    ax.set_xlabel('Normal Stress (Pa)')
    ax.set_ylabel('Shear Stress (Pa)')
    ax.set_title('Mohr\'s Circle')
    plt.grid(True)
    plt.show()

    # ---- Failure Criteria ----
    vm = FailureCriteria.von_mises(sigma_x, sigma_y, tau_xy)
    tc = FailureCriteria.tresca(sigma_x, sigma_y, tau_xy)
    max_n = FailureCriteria.maximum_normal(sigma_x, sigma_y, tau_xy)
    print(f"\nFailure criteria: Von Mises={vm/1e6:.2f} MPa, Tresca={tc/1e6:.2f} MPa, Max Normal={max_n/1e6:.2f} MPa")

    # ---- SDOF Damped Vibration ----
    m_sdof = 100.0          # kg
    c_sdof = 10.0           # Ns/m
    k_sdof = 2000.0         # N/m
    sdof = SDOFDamped(m_sdof, c_sdof, k_sdof, x0=0.1, v0=0.0, dt=0.001)
    t_max = 5.0
    t = np.arange(0, t_max, sdof.dt)
    disp = [sdof.displacement(ti) for ti in t]
    plt.figure()
    plt.plot(t, disp, label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('SDOF Damped Vibration')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---- Modal Analysis (MDOF) ----
    # Simple 2-DOF mass and stiffness matrices
    M = np.diag([1.0, 2.0])          # kg
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])  # N/m
    modal = ModalSystem(M, K)
    wns = modal.natural_frequencies()
    print(f"\nModal natural frequencies (rad/s): {wns}")
    print("Mode shapes (columns):")
    print(modal.mode_shapes())

    # ---- Fatigue Life ----
    sigma_a = 50e6          # Pa
    sigma_f = 150e6         # Pa
    b = -0.1                # S-N slope (1/years typical)
    c = 0.5                 # Goodman intercept
    sigma_u = 400e6         # Pa
    fatigue = FatigueLife(sigma_a, sigma_f, b, c, sigma_u)
    N = fatigue.cycles_to_failure()
    N_allow = fatigue.allowable_cycles()
    print(f"\nFatigue life (cycles): N={N:.2e}, Allowable (Goodman)={N_allow:.2e}")

    # ---- Structural Optimization ----
    opt_result = optimize_beam_weight(length=1.0, load=1000.0,
                                      E=E_DEFAULT, nu=NU_DEFAULT, rho=RHO_DEFAULT)
    print("\nStructural optimization result:")
    print(f"Success: {opt_result['success']}")
    print(f"Optimal width (m): {opt_result['x_opt'][0]:.4f}")
    print(f"Optimal depth (m): {opt_result['x_opt'][1]:.4f}")
    print(f"Resulting weight (kg): {opt_result['weight']:.2f}")
    print(f"Max bending stress (Pa): {opt_result['sigma_max']:.2e}")

    # ---- Rotor Critical Speed ----
    I_shaft = 5e-6          # m^4
    m_per_len = 2.0         # kg/m
    L_shaft = 2.0           # m
    omega_cr = rotor_critical_speed(E_DEFAULT, I_shaft, m_per_len, L_shaft)
    print(f"\nRotor first critical speed: {omega_cr/2/np.pi:.2f} Hz")

    # ---- Contact Pressure ----
    load_contact = 5000.0   # N
    area_contact = 0.02     # m^2
    p_contact = contact_pressure(load_contact, area_contact)
    print(f"\nContact pressure: {p_contact/1e6:.2f} MPa")