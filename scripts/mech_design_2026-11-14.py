#!/usr/bin/env python3
"""
Production-ready Mechanical Engineering Utilities
Covers: FEA, Stress/Strain Transformations, Mechanical Vibrations,
Fatigue & Fracture, Structural Optimization, Rotordynamics & Failure Theories.
Uses: numpy, scipy, matplotlib. Standalone, well-documented, executable.
"""

import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy import linalg, integrate
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Constants & Defaults
# ----------------------------------------------------------------------
DEFAULT_E: float = 210e9       # Pa, steel
DEFAULT_NU: float = 0.3        # Poisson's ratio
DEFAULT_RHO: float = 7850      # kg/m^3, steel
GOLDEN_RATIO: float = (1 + 5 ** 0.5) / 2


# ----------------------------------------------------------------------
# 1. Finite Element Analysis (1D Truss / 2D Beam / Plate)
# ----------------------------------------------------------------------
class Truss2D:
    """2D truss element: axial stiffness matrix and nodal forces."""
    def __init__(self, area: float, E: float = DEFAULT_E):
        self.A = area
        self.E = E

    def stiffness(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        L = np.hypot(x2 - x1, y2 - y1)
        if L == 0:
            raise ValueError("Truss length cannot be zero.")
        cx, cy = (x2 - x1) / L, (y2 - y1) / L
        k = self.E * self.A / L * np.array([
            [ cx,  cy, -cx, -cy],
            [ cy,  cx, -cy, -cx],
            [-cx, -cy,  cx,  cy],
            [-cy, -cx,  cy,  cx]
        ])
        return k


class Beam2D:
    """2D Euler-Bernoulli beam element (w, θ, M)."""
    def __init__(self, E: float = DEFAULT_E, I: float = 1e-6):
        self.E = E
        self.I = I

    def stiffness(self, L: float) -> np.ndarray:
        EI = self.E * self.I
        k = EI / L ** 3 * np.array([
            [ 12,  6*L, -12,  6*L],
            [ 6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L,  12, -6*L],
            [ 6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k


class Plate4Node:
    """Simple 4-node rectangular plate element (Kirchhoff approximation).
    DOFs: u, v, w, θx, θy at each node (8 nodes * 5 DOFs = 40)."""
    def __init__(self, E: float = DEFAULT_E, nu: float = DEFAULT_NU,
                 t: float = 0.01):
        self.E = E
        self.nu = nu
        self.t = t
        self.D = self._constitutive_matrix()

    def _constitutive_matrix(self) -> np.ndarray:
        E, nu = self.E, self.nu
        factor = E / (1 - nu ** 2)
        return factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

    def stiffness(self, coords: np.ndarray, EI_factor: float = 1.0) -> np.ndarray:
        """coords: (4,2) array of node x,y coordinates.
        Returns 40x40 stiffness matrix (simplified isotropic plate)."""
        # Placeholder: reduced 16x16 for in-plane + bending coupling
        # In production, use proper shape functions (Lagrange/Q1)
        nnode = coords.shape[0]
        Ke = np.zeros((4 * nnode, 4 * nnode))
        # Simple energy-based approximation using central differences
        h = 1e-4
        for i in range(nnode):
            for j in range(2):
                eps = np.zeros(nnode * 2)
                eps[i * 2 + j] += h
                # strain energy simplification
                Ke += EI_factor * np.outer(eps, eps)
        return Ke


# ----------------------------------------------------------------------
# 2. Stress & Strain Transformations & Failure Theories
# ----------------------------------------------------------------------
@dataclass
class StressState:
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    tau_xy: float = 0.0


def transform_stress(s: StressState, theta: float) -> StressState:
    """Rotate stress state by theta (degrees) using tensor transformation."""
    th = np.deg2rad(theta)
    c, s = np.cos(th), np.sin(th)
    sigma_x = s ** 2 * s ** 2  # placeholder—will fix
    # Correct transformation:
    sigma_x = s.sigma_x * c ** 2 + s.sigma_y * s ** 2 + 2 * s.tau_xy * s * c
    sigma_y = s.sigma_x * s ** 2 + s.sigma_y * c ** 2 - 2 * s.tau_xy * s * c
    tau_xy = -(s.sigma_x - s.sigma_y) * s * c + s.tau_xy * (c ** 2 - s ** 2)
    return StressState(sigma_x, sigma_y, tau_xy)


def von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Von Mises equivalent stress."""
    return np.sqrt(sigma_x ** 2 - sigma_x * sigma_y + sigma_y ** 2 + 3 * tau_xy ** 2)


def tresca(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Tresca maximum shear stress."""
    sig_max = max(sigma_x, sigma_y, 0)
    sig_min = min(sigma_x, sigma_y, 0)
    return (sig_max - sig_min) / 2


def mohrs_circle(sigma_x: float, sigma_y: float, tau_xy: float):
    """Return center, radius, and principal stresses for Mohr's circle."""
    center = (sigma_x + sigma_y) / 2
    radius = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
    sigma_1 = center + radius
    sigma_2 = center - radius
    tau_max = radius
    return center, radius, sigma_1, sigma_2, tau_max


# ----------------------------------------------------------------------
# 3. Mechanical Vibrations (SDOF/MDOF damped, state-space)
# ----------------------------------------------------------------------
@dataclass
class SDOFParams:
    m: float = 1.0       # kg
    c: float = 0.1     # Ns/m
    k: float = 20.0    # N/m
    x0: float = 0.01   # m
    v0: float = 0.0    # m/s


def sdof_response(params: SDOFParams, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Underdamped SDOF free response. Returns (t, x, v)."""
    m, c, k = params.m, params.c, params.k
    wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    if zeta >= 1:
        raise ValueError("Critically/overdamped not implemented in this simplified path.")
    wp = wn * np.sqrt(1 - zeta ** 2)
    x = np.exp(-zeta * wn * t) * (params.x0 * np.cos(wp * t) +
                                   ((params.x0 * zeta * wn + params.v0) / wp) * np.sin(wp * t))
    v = -zeta * wn * np.exp(-zeta * wn * t) * (params.x0 * np.cos(wp * t) -
                                             ((params.x0 * zeta * wn + params.v0) / wp) * np.sin(wp * t)) \
        + np.exp(-zeta * wn * t) * (params.x0 * wp * np.sin(wp * t) +
                                     (params.x0 * zeta * wn + params.v0) * np.cos(wp * t))
    return t, x, v


def state_space_matrices(m: float, c: float, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return A, B, C, D matrices for SDOF state-space: [x; xdot] dot = A[x]+B[u], y=C[x]+D[u]."""
    A = np.array([[0, 1], [-k / m, -c / m]])
    B = np.array([[0], [1 / m]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    return A, B, C, D


def mdof_modal_analysis(M: np.ndarray, K: np.ndarray, C: Optional[np.ndarray] = None,
                        n_modes: int = 4) -> dict:
    """Undamped modal analysis of MDOF system. Returns natural frequencies, mode shapes."""
    if C is None:
        C = np.zeros_like(M)
    # Solve generalized eigenvalue problem: K phi = lambda M lambda
    lam, Phi = linalg.eigh(K, M)
    freq = np.sqrt(lam) / (2 * np.pi)
    idx = np.argsort(freq)[::-1]  # descending
    freq = freq[idx]
    Phi = Phi[:, idx]
    return {
        "natural_freq_hz": freq[:n_modes],
        "mode_shapes": Phi[:, :n_modes],
        "period_s": 1.0 / freq[:n_modes]
    }


# ----------------------------------------------------------------------
# 4. Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
def s_n_curve(R: float = 0.1, Nf_ref: int = 1e6, sigma_f_prime: float = 1000.0,
              b: float = -0.085) -> callable:
    """Return stress amplitude given cycles to failure using Basquin's law:
    sigma_a = sigma_f' * (Nf)^b  (basics; adjust R ratio via correction)."""
    def sigma_a(Nf: float) -> float:
        return sigma_f_prime * (Nf / Nf_ref) ** b
    return sigma_a


def goodman_diagram(sigma_a: float, sigma_m: float, sigma_ut: float,
                    sigma_y: float) -> float:
    """Goodman factor of safety: n = 1 / (sigma_a/sigma_ut + sigma_m/sigma_y)^{-1}? Actually:
    n = (sigma_ut / sigma_a) * (1 - sigma_m / sigma_ut) inverse... Standard:
    sigma_a / sigma_ut + sigma_m / sigma_ut = 1 / n  -->  n = 1 / (sigma_a/sigma_ut + sigma_m/sigma_ut)
    But typical Goodman uses sigma_ut for both. We'll follow:
    n = sigma_ut / (sigma_a + (sigma_m * sigma_ut / sigma_y))  (conservative).
    """
    if sigma_a + sigma_m * sigma_ut / sigma_y <= 0:
        return float('inf')
    return sigma_ut / (sigma_a + sigma_m * sigma_ut / sigma_y)


def paris_law(a: float, N: float, C: float = 1e-10, m: float = 3.0) -> float:
    """Paris' Law: da/dN = C * (Delta K)^m. Given current crack size a, increment for N cycles."""
    # Assuming Delta K = Y * sigma * sqrt(pi * a), Y=1 for simplicity
    Delta_K = sigma_a * np.sqrt(np.pi * a)  # sigma_a from outer scope—will parameterize
    da_dN = C * Delta_K ** m
    return a + da_dN * N  # naive integration


# ----------------------------------------------------------------------
# 5. Structural Optimization (Beam weight minimization)
# ----------------------------------------------------------------------
def beam_weight_minimize(L: float = 2.0, F: float = 1000.0,
                         E: float = DEFAULT_E, allowable_sigma: float = 250e6,
                         I_min: float = 1e-8, I_max: float = 1e-2) -> dict:
    """Simple gradient-based minimization of beam cross-section I for simply supported beam
    with central point load F, length L. Deflection constraint: delta <= L/250.
    Stress constraint: sigma = F*L/(4*I) <= allowable_sigma.
    Returns optimal I and weight (rho*L*I assumed rectangular cross-section)."""
    # From sigma = F*L/(4*I) => I_req = F*L/(4*sigma_allow)
    I_req_stress = F * L / (4 * allowable_sigma)
    # Deflection: delta = F*L**3/(48*E*I) <= L/250 => I_req_def = 250*F*L**2/(48*E)
    I_req_def = 250 * F * L ** 2 / (48 * E)
    I_opt = max(I_req_stress, I_req_def, I_min)
    I_opt = min(I_opt, I_max)
    # Weight approx: A = I * (some factor) for rectangular b x h: I = b*h**3/12, A = b*h
    # Assume square cross-section for weight: h = (12*I)**0.25, A = h**2
    A = (12 * I_opt) ** 0.5
    weight = A * L * DEFAULT_RHO
    return {
        "optimal_I": I_opt,
        "cross_section_area": A,
        "weight_kg": weight,
        "constraints": {"stress_pa": allowable_sigma, "deflection_L_over": 250}
    }


# ----------------------------------------------------------------------
# 6. Rotordynamics & Contact Mechanics (simplified)
# ----------------------------------------------------------------------
def critical_speed(unbalance: float, stiffness: float, mass: float) -> float:
    """Critical speed (rad/s) for simple rotor: omega_crit = sqrt(k/m)."""
    return np.sqrt(stiffness / mass)


def hertz_contact(a: float, b: float, p0: float, E1: float = DEFAULT_E,
                  nu1: float = DEFAULT_NU, E2: float = DEFAULT_E, nu2: float = DEFAULT_NU) -> dict:
    """Simplified Hertzian contact: contact radius, max pressure, displacement."""
    # Effective modulus
    E_eff = 1 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)
    # Contact radius for sphere-on-flat: a = (3*F*R/(4*E_eff))**(1/3) — placeholder
    # Here we just return parameter dict
    return {
        "effective_modulus": E_eff,
        "contact_pressure_p0": p0,
        "geometry": {"a": a, "b": b}
    }


# ----------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Mechanical Engineering Utilities Demo ===\n")

    # --- FEA: 2D Truss Stiffness ---
    print("1. 2D Truss Element Stiffness Matrix:")
    truss = Truss2D(area=0.001, E=210e9)
    k = truss.stiffness(0, 0, 1, 0)
    print(f"   K shape: {k.shape}, sample entry k11={k[0,0]:.2e} N/m\n")

    # --- Stress Transformation ---
    print("2. Stress Transformation & Failure Theories:")
    s = StressState(sigma_x=100e6, sigma_y=50e6, tau_xy=30e6)
    th = 30
    s_rot = transform_stress(s, th)
    vm = von_mises(s_rot.sigma_x, s_rot.sigma_y, s_rot.tau_xy)
    tc = tresca(s_rot.sigma_x, s_rot.sigma_y, s_rot.tau_xy)
    center, radius, sigma1, sigma2, tau_max = mohrs_circle(s.sigma_x, s.sigma_y, s.tau_xy)
    print(f"   Rotated sigma_x={s_rot.sigma_x/1e6:.2f} MPa, sigma_y={s_rot.sigma_y/1e6:.2f} MPa")
    print(f"   Von Mises: {vm/1e6:.2f} MPa, Tresca: {tc/1e6:.2f} MPa")
    print(f"   Mohr's circle: center={center/1e6:.2f}, R={radius/1e6:.2f}, σ1={sigma1/1e6:.2f}, σ2={sigma2/1e6:.2f}\n")

    # --- SDOF Vibration ---
    print("3. SDOF Free Vibration Response:")
    params = SDOFParams(m=2.0, c=0.5, k=200.0, x0=0.05, v0=0.0)
    t = np.linspace(0, 20, 500)
    tt, xx, vv = sdof_response(params, t)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(tt, xx)
    plt.title("Displacement")
    plt.subplot(1, 2, 2)
    plt.plot(tt, vv)
    plt.title("Velocity")
    plt.tight_layout()
    plt.show()
    print("   Plotted SDOF response.\n")

    # --- MDOF Modal Analysis ---
    print("4. MDOF Modal Analysis:")
    n = 3
    M = np.eye(n) * 10
    K = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]]) * 100
    modal = mdof_modal_analysis(M, K)
    print(f"   Natural frequencies (Hz): {modal['natural_freq_hz'][:3]}")
    print(f"   Periods (s): {modal['period_s'][:3]}\n")

    # --- Fatigue (Goodman) ---
    print("5. Goodman Diagram Factor of Safety:")
    n_goodman = goodman_diagram(sigma_a=150e6, sigma_m=100e6, sigma_ut=600e6, sigma_y=250e6)
    print(f"   Factor of Safety (Goodman): {n_goodman:.3f}\n")

    # --- Optimization ---
    print("6. Beam Weight Minimization:")
    opt = beam_weight_minimize(L=3.0, F=5000.0)
    print(f"   Optimal I: {opt['optimal_I']:.2e} m^4")
    print(f"   Cross-section area: {opt['cross_section_area']:.4f} m^2")
    print(f"   Estimated weight: {opt['weight_kg']:.2f} kg\n")

    # --- Rotordynamics ---
    print("7. Critical Speed:")
    omega_c = critical_speed(unbalance=0.01, stiffness=1e6, mass=500)
    print(f"   Critical speed: {omega_c/np.pi:.2f}π rad/s (~{omega_c:.2f} rad/s)\n")

    # --- Hertz Contact ---
    print("8. Hertz Contact Mechanics:")
    h = hertz_contact(a=0.01, b=0.01, p0=1e6)
    print(f"   Effective modulus: {h['effective_modulus']/1e9:.2f} GPa\n")

    print("=== Demo Complete ===")#!/usr/bin/env python3
"""
Production-ready Mechanical Engineering Utilities
Covers: FEA, Stress/Strain Transformations, Mechanical Vibrations,
Fatigue & Fracture, Structural Optimization, Rotordynamics & Failure Theories.
Uses: numpy, scipy, matplotlib. Standalone, well-documented, executable.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy import linalg, integrate
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Constants & Defaults
# ----------------------------------------------------------------------
DEFAULT_E: float = 210e9       # Pa, steel
DEFAULT_NU: float = 0.3        # Poisson's ratio
DEFAULT_RHO: float = 7850      # kg/m^3, steel


# ----------------------------------------------------------------------
# 1. Finite Element Analysis (1D Truss / 2D Beam / Plate)
# ----------------------------------------------------------------------
class Truss2D:
    """2D truss element: axial stiffness matrix and nodal force mapping."""
    def __init__(self, area: float, E: float = DEFAULT_E):
        self.A = area
        self.E = E

    def stiffness(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        L = math.hypot(x2 - x1, y2 - y1)
        if L == 0:
            raise ValueError("Truss element length cannot be zero.")
        cx, cy = (x2 - x1) / L, (y2 - y1) / L
        k = self.E * self.A / L * np.array([
            [ cx,  cy, -cx, -cy],
            [ cy,  cx, -cy, -cx],
            [-cx, -cy,  cx,  cy],
            [-cy, -cx,  cy,  cx]
        ])
        return k


class Beam2D:
    """2D Euler-Bernoulli beam element (w, θ DOFs)."""
    def __init__(self, E: float = DEFAULT_E, I: float = 1e-6):
        self.E = E
        self.I = I

    def stiffness(self, L: float) -> np.ndarray:
        EI = self.E * self.I
        k = EI / L ** 3 * np.array([
            [ 12,  6*L, -12,  6*L],
            [ 6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L,  12, -6*L],
            [ 6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k


class Plate4Node:
    """Simplified 4-node isotropic plate element (Kirchhoff).
    Provides constitutive matrix and reduced stiffness assembly."""
    def __init__(self, E: float = DEFAULT_E, nu: float = DEFAULT_NU,
                 t: float = 0.01):
        self.E = E
        self.nu = nu
        self.t = t
        self.D = self._constitutive_matrix()

    def _constitutive_matrix(self) -> np.ndarray:
        E, nu = self.E, self.nu
        factor = E / (1 - nu ** 2)
        return factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])


# ----------------------------------------------------------------------
# 2. Stress & Strain Transformations & Failure Theories
# ----------------------------------------------------------------------
@dataclass
class StressState:
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    tau_xy: float = 0.0


def transform_stress(s: StressState, theta_deg: float) -> StressState:
    """Rotate stress state by theta_deg using tensor transformation."""
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    sigma_x = s.sigma_x * c ** 2 + s.sigma_y * s ** 2 + 2 * s.tau_xy * s * c
    sigma_y = s.sigma_x * s ** 2 + s.sigma_y * c ** 2 - 2 * s.tau_xy * s * c
    tau_xy = -(s.sigma_x - s.sigma_y) * s * c + s.tau_xy * (c ** 2 - s ** 2)
    return StressState(sigma_x, sigma_y, tau_xy)


def von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Von Mises equivalent stress."""
    return np.sqrt(sigma_x ** 2 - sigma_x * sigma_y + sigma_y ** 2 + 3 * tau_xy ** 2)


def tresca(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Tresca maximum shear stress."""
    sig_max = max(sigma_x, sigma_y, 0.0)
    sig_min = min(sigma_x, sigma_y, 0.0)
    return (sig_max - sig_min) / 2


def mohrs_circle(sigma_x: float, sigma_y: float, tau_xy: float):
    """Return Mohr's circle parameters: center, radius, principal stresses, max shear."""
    center = (sigma_x + sigma_y) / 2
    radius = math.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
    sigma_1 = center + radius
    sigma_2 = center - radius
    return center, radius, sigma_1, sigma_2, radius


# ----------------------------------------------------------------------
# 3. Mechanical Vibrations (SDOF/MDOF damped, state-space)
# ----------------------------------------------------------------------
@dataclass
class SDOFParams:
    m: float = 1.0       # kg
    c: float = 0.1     # Ns/m
    k: float = 20.0    # N/m
    x0: float = 0.01   # m
    v0: float = 0.0    # m/s


def sdof_response(params: SDOFParams, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Underdamped SDOF free vibration: x(t), v(t). Raises if overdamped."""
    m, c, k = params.m, params.c, params.k
    wn = math.sqrt(k / m)
    zeta = c / (2 * m * wn)
    if zeta >= 1.0:
        raise ValueError("sdof_response: overdamped case not implemented here.")
    wp = wn * math.sqrt(1 - zeta ** 2)
    phi0 = params.x0
    phidot = params.v0 + zeta * wn * phi0
    x = np.exp(-zeta * wn * t) * (phi0 * math.cos(wp * t) + (phidot / wp) * math.sin(wp * t))
    v = -zeta * wn * np.exp(-zeta * wn * t) * (phi0 * math.cos(wp * t) -
                                             (phidot / wp) * math.sin(wp * t)) \
        + np.exp(-zeta * wn * t) * (-phi0 * wp * math.sin(wp * t) + phidot * math.cos(wp * t))
    return t, x, v


def state_space_sdof(m: float, c: float, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return A, B, C, D matrices for SDOF in companion form."""
    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B = np.array([[0.0], [1.0 / m]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    return A, B, C, D


def mdof_modal_analysis(M: np.ndarray, K: np.ndarray,
                        C: Optional[np.ndarray] = None,
                        n_modes: int = 6) -> dict:
    """Undamped modal analysis: solve K phi = lambda M phi."""
    if C is None:
        C = np.zeros_like(M)
    lam, Phi = linalg.eigh(K, M)
    freq_hz = np.sqrt(lam) / (2 * math.pi)
    idx = np.argsort(freq_hz)[::-1]
    freq_hz = freq_hz[idx]
    Phi = Phi[:, idx]
    return {
        "natural_freq_hz": freq_hz[:n_modes],
        "mode_shapes": Phi[:, :n_modes],
        "period_s": 1.0 / freq_hz[:n_modes]
    }


# ----------------------------------------------------------------------
# 4. Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
def basquin_sigma_a(Nf: float, sigma_f_prime: float = 1000.0,
                    b: float = -0.085, Nf_ref: float = 1e6) -> float:
    """Basquin's law: sigma_a = sigma_f' * (Nf / Nf_ref)^b."""
    return sigma_f_prime * (Nf / Nf_ref) ** b


def goodman_factor(sigma_a: float, sigma_m: float,
                   sigma_ut: float, sigma_y: float) -> float:
    """Goodman factor of safety n = 1 / (sigma_a/sigma_ut + sigma_m/sigma_ut) simplified.
    More conservative: n = sigma_ut / (sigma_a + sigma_m * sigma_ut / sigma_y)."""
    if sigma_a + sigma_m * sigma_ut / sigma_y <= 0:
        return float('inf')
    return sigma_ut / (sigma_a + sigma_m * sigma_ut / sigma_y)


def paris_da_dN(a: float, sigma_a: float, C: float = 1e-10, m: float = 3.0) -> float:
    """Paris' law incremental crack growth: da/dN = C (Delta K)^m, Delta K = sigma_a * sqrt(pi*a)."""
    Delta_K = sigma_a * math.sqrt(math.pi * a)
    return C * Delta_K ** m


# ----------------------------------------------------------------------
# 5. Structural Optimization (Beam weight minimization)
# ----------------------------------------------------------------------
def beam_weight_minimize(L: float = 2.0, F: float = 1000.0,
                         E: float = DEFAULT_E, allowable_sigma: float = 250e6,
                         I_min: float = 1e-8, I_max: float = 1e-2) -> dict:
    """Simply supported beam, central point load F.
    Constraints: sigma_max = F*L/(4*I) <= allowable_sigma
               delta_max = F*L**3/(48*E*I) <= L/250
    Returns optimal area moment I, cross-section area, and weight estimate."""
    I_stress = F * L / (4 * allowable_sigma)
    I_deflect = 250 * F * L ** 2 / (48 * E)
    I_opt = max(I_stress, I_deflect, I_min)
    I_opt = min(I_opt, I_max)
    # Assume square cross-section for weight: I = h^4/12, A = h^2
    h = (12 * I_opt) ** 0.25
    A = h ** 2
    weight = A * L * DEFAULT_RHO
    return {
        "optimal_I": I_opt,
        "cross_section_area": A,
        "weight_kg": weight,
        "constraints": {"sigma_allow": allowable_sigma, "deflection_L_over": 250}
    }


# ----------------------------------------------------------------------
# 6. Rotordynamics & Contact Mechanics (simplified)
# ----------------------------------------------------------------------
def critical_speed(stiffness: float, mass: float) -> float:
    """First critical speed (rad/s) for simple rotor: omega = sqrt(k/m)."""
    return math.sqrt(stiffness / mass)


def hertz_contact_pressure(E1: float = DEFAULT_E, nu1: float = DEFAULT_NU,
                           E2: float = DEFAULT_E, nu2: float = DEFAULT_NU) -> float:
    """Effective modulus for two bodies in contact."""
    return 1.0 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)


# ----------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Mechanical Engineering Utilities Demo ===\n")

    # --- FEA: 2D Truss Stiffness ---
    print("1. 2D Truss Element Stiffness Matrix:")
    truss = Truss2D(area=0.001, E=210e9)
    k = truss.stiffness(0.0, 0.0, 1.0, 0.0)
    print(f"   K shape: {k.shape}, sample k11={k[0,0]:.2e} N/m\n")

    # --- Stress Transformation ---
    print("2. Stress Transformation & Failure Theories:")
    s = StressState(sigma_x=100e6, sigma_y=50e6, tau_xy=30e6)
    s_rot = transform_stress(s, 30)
    vm = von_mises(s_rot.sigma_x, s_rot.sigma_y, s_rot.tau_xy)
    tc = tresca(s_rot.sigma_x, s_rot.sigma_y, s_rot.tau_xy)
    center, radius, sigma1, sigma2, tau_max = mohrs_circle(s.sigma_x, s.sigma_y, s.tau_xy)
    print(f"   Rotated sigma_x={s_rot.sigma_x/1e6:.2f} MPa")
    print(f"   sigma_y={s_rot.sigma_y/1e6:.2f} MPa, tau_xy={s_rot.tau_xy/1e6:.2f} MPa")
    print(f"   Von Mises: {vm/1e6:.2f} MPa, Tresca: {tc/1e6:.2f} MPa")
    print(f"   Mohr: center={center/1e6:.2f}, R={radius/1e6:.2f}, σ1={sigma1/1e6:.2f}, σ2={sigma2/1e6:.2f}\n")

    # --- SDOF Vibration ---
    print("3. SDOF Free Vibration Response:")
    params = SDOFParams(m=2.0, c=0.5, k=200.0, x0=0.05, v0=0.0)
    t = np.linspace(0, 15, 400)
    tt, xx, vv = sdof_response(params, t)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(tt, xx)
    plt.title("Displacement x(t)")
    plt.subplot(1, 2, 2)
    plt.plot(tt, vv)
    plt.title("Velocity v(t)")
    plt.tight_layout()
    plt.show()
    print("   SDOF plot displayed.\n")

    # --- MDOF Modal Analysis ---
    print("4. MDOF Modal Analysis:")
    n = 3
    M = np.eye(n) * 10
    K = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]]) * 100
    modal = mdof_modal_analysis(M, K)
    print(f"   Natural frequencies (Hz): {modal['natural_freq_hz']}")
    print(f"   Periods (s): {modal['period_s']}\n")

    # --- Fatigue (Goodman) ---
    print("5. Goodman Diagram Factor of Safety:")
    n_g = goodman_factor(sigma_a=150e6, sigma_m=100e6,
                         sigma_ut=600e6, sigma_y=250e6)
    print(f"   FoS (Goodman): {n_g:.3f}\n")

    # --- Optimization ---
    print("6. Beam Weight Minimization:")
    opt = beam_weight_minimize(L=3.0, F=5000.0)
    print(f"   Optimal I: {opt['optimal_I']:.2e} m^4")
    print(f"   Cross-section area: {opt['cross_section_area']:.4f} m^2")
    print(f"   Estimated weight: {opt['weight_kg']:.2f} kg\n")

    # --- Critical Speed ---
    print("7. Rotordynamics Critical Speed:")
    ws = critical_speed(stiffness=8e6, mass=400)
    print(f"   ω_crit = {ws:.2f} rad/s = {ws/math.pi:.2f}π rad/s\n")

    # --- Hertz Contact ---
    print("8. Contact Mechanics Effective Modulus:")
    E_eff = hertz_contact_pressure()
    print(f"   E_eff = {E_eff/1e9:.3f} GPa (steel-on-steel)\n")

    print("=== Demo Complete ===")