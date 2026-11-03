#!/usr/bin/env python3
"""
Solid Mechanics & Machine Design Toolkit
Covers: Finite Element Analysis (1D truss/beam), Stress/Strain Transformations
(Mohr's circle, Von Mises, Tresca), Mechanical Vibrations (SDOF/MDOF damped),
Fatigue & Fracture Mechanics (S-N curves, Goodman/Gerber, Paris' Law),
Structural Optimization (beam weight minimization), Rotordynamics/Contact/Failure Theories.
Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import minimize


# ---- 1. FINITE ELEMENT METHODS ----
def truss_element_stiffness(E: float, A: float, L: float) -> np.ndarray:
    """1D truss element stiffness matrix in global coordinates (2x2).

    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    A : float
        Cross-sectional area [m^2]
    L : float
        Element length [m]

    Returns
    -------
    k : np.ndarray
        2x2 stiffness matrix [N/m]
    """
    k = (E * A / L) * np.array([[1, -1], [-1, 1]], dtype=float)
    return k


def beam_element_stiffness(E: float, I: float, L: float) -> np.ndarray:
    """Euler-Bernoulli beam element stiffness matrix (4x4) in local coordinates.

    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    I : float
        Area moment of inertia [m^4]
    L : float
        Element length [m]

    Returns
    -------
    k : np.ndarray
        4x4 stiffness matrix [N*m]
    """
    k = (E * I / L ** 3) * np.array([
        [12, 6 * L, -12, 6 * L],
        [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]
    ], dtype=float)
    return k


# ---- 2. STRESS/STRAIN TRANSFORMATIONS ----
def transform_stress(sigma_x: float, sigma_y: float, tau_xy: float, theta: float):
    """Transform stress components by angle theta (radians) in 2D plane stress.

    Returns (sigma_xp, sigma_yp, tau_xyp)
    """
    c = np.cos(theta)
    s = np.sin(theta)
    sigma_xp = sigma_x * c ** 2 + sigma_y * s ** 2 + 2 * tau_xy * s * c
    sigma_yp = sigma_x * s ** 2 + sigma_y * c ** 2 - 2 * tau_xy * s * c
    tau_xyp = -(sigma_x - sigma_y) * s * c + tau_xy * (c ** 2 - s ** 2)
    return sigma_xp, sigma_yp, tau_xyp


def mohrs_circle(sigma_x: float, sigma_y: float, tau_xy: float):
    """Return Mohr's circle center, radius, and principal stresses.

    Returns (center, radius, sigma_1, sigma_3)
    """
    center = (sigma_x + sigma_y) / 2.0
    radius = np.sqrt(((sigma_x - sigma_y) / 2.0) ** 2 + tau_xy ** 2)
    sigma_1 = center + radius
    sigma_3 = center - radius
    return center, radius, sigma_1, sigma_3


def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float):
    """Compute principal stresses sigma_1, sigma_3 (sigma_1 >= sigma_3)."""
    center = (sigma_x + sigma_y) / 2.0
    radius = np.sqrt(((sigma_x - sigma_y) / 2.0) ** 2 + tau_xy ** 2)
    return center + radius, center - radius


def von_mises_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Von Mises equivalent stress for 2D plane stress state."""
    return np.sqrt(sigma_x ** 2 - sigma_x * sigma_y + sigma_y ** 2 + 3.0 * tau_xy ** 2)


def tresca_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Tresca (max shear) equivalent stress = sigma_1 - sigma_3."""
    s1, s3 = principal_stresses(sigma_x, sigma_y, tau_xy)
    return s1 - s3


# ---- 3. MECHANICAL VIBRATIONS ----
def sdof_natural_freq(m: float, k: float) -> tuple:
    """Undamped natural frequency for SDOF: omega_n [rad/s], f_n [Hz]."""
    omega_n = np.sqrt(k / m)
    f_n = omega_n / (2.0 * np.pi)
    return omega_n, f_n


def sdof_damped_response(m: float, c: float, k: float, F0: float = 0.0,
                        omega: float = 0.0, t_max: float = 10.0, dt: float = 0.01) -> tuple:
    """Harmonic force response of viscously damped SDOF system.

    Returns (t, x) arrays.
    """
    t = np.arange(0.0, t_max, dt)
    omega_n = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(m * k))
    if zeta >= 1.0:
        omega_d = 0.0
    else:
        omega_d = omega_n * np.sqrt(1.0 - zeta ** 2)
    denom = np.sqrt((omega_n ** 2 - omega ** 2) ** 2 + (2.0 * zeta * omega_n * omega) ** 2)
    A = F0 / denom if abs(denom) > 1e-12 else 0.0
    phase = np.arctan2(2.0 * zeta * omega_n * omega, omega_n ** 2 - omega ** 2)
    x = A * np.cos(omega * t - phase)
    return t, x


def mdof_modal_analysis(K: np.ndarray, M: np.ndarray) -> tuple:
    """Undamped modal analysis for MDOF system: K phi = lambda M phi.

    Returns (natural_freq_rad/s, mode_shapes_columns)
    """
    eigenvalues, eigenvectors = eigh(K, M)
    frequencies = np.sqrt(np.abs(eigenvalues))
    idx = np.argsort(frequencies)
    return frequencies[idx], eigenvectors[:, idx]


# ---- 4. FATIGUE & FRACTURE MECHANICS ----
def s_n_stress(stress_a: float, cycles_a: float, stress_b: float, cycles_b: float,
               target_cycles: float) -> float:
    """Log-log linear S-N curve interpolation.

    Given two points (sigma_a, N_a) and (sigma_b, N_b), returns stress at target_cycles.
    Assumes power-law relationship sigma = a * N^b in log-log space.
    """
    logN_a = np.log(cycles_a)
    logN_b = np.log(cycles_b)
    logS_a = np.log(stress_a)
    logS_b = np.log(stress_b)
    slope = (logS_b - logS_a) / (logN_b - logN_a)
    intercept = logS_a - slope * logN_a
    stress = np.exp(intercept + slope * np.log(target_cycles))
    return stress


def goodman_criterion(sigma_a: float, sigma_m: float, sigma_ut: float, sigma_y: float) -> float:
    """Goodman safety factor n.

    n = 1 / (sigma_a/sigma_ut + sigma_m/sigma_y)
    n >= 1 => safe per Goodman.
    """
    if sigma_ut == 0 or sigma_y == 0:
        return np.inf
    return 1.0 / (sigma_a / sigma_ut + sigma_m / sigma_y)


def gerber_criterion(sigma_a: float, sigma_m: float, sigma_ut: float, sigma_y: float) -> float:
    """Gerber safety factor n (parabolic).

    n = 1 / ( (sigma_a/sigma_ut) / (1 - (sigma_m/sigma_y)^2 ) )
    """
    denom = (sigma_a / sigma_ut) / (1.0 - (sigma_m / sigma_y) ** 2) if sigma_y != 0 else np.inf
    if denom <= 0:
        return np.inf
    return 1.0 / denom


def paris_law(da_dN: float, C: float, m: float) -> float:
    """Paris' Law for fatigue crack growth: da/dN = C * (Delta K)^m.

    Given da/dN, returns Delta K [Pa*m^0.5].
    """
    if C <= 0 or m <= 0:
        return 0.0
    return (da_dN / C) ** (1.0 / m)


# ---- 5. STRUCTURAL OPTIMIZATION ----
def beam_weight_minimization(L: float, load: float, allowable_stress: float,
                             bounds: tuple = (1e-6, 1e-2)) -> tuple:
    """Minimize cross-sectional area (hence weight) of a uniaxial beam subject to stress constraint.

    Problem: minimize A s.t. |load|/A <= allowable_stress.
    Returns (optimal_A, objective_value).
    """
    def obj(A):
        return A[0]

    def constraint(A):
        sigma = np.abs(load) / A[0]
        return allowable_stress - sigma

    x0 = np.array([1e-4])
    res = minimize(obj, x0, bounds=[bounds],
                   constraints={'type': 'ineq', 'fun': constraint})
    return res.x[0], res.fun


# ---- 6. ROTORDYNAMICS / CONTACT / FAILURE THEORIES ----
def critical_speed(rotor_mass: float, shaft_stiffness: float) -> float:
    """Simple critical speed estimate for a rotor: omega_crit = sqrt(k/m) [rad/s].

    Returns critical speed in RPM.
    """
    omega_crit = np.sqrt(shaft_stiffness / rotor_mass)
    return omega_crit * 60.0 / (2.0 * np.pi)


def hertzian_contact_pressure(F: float, R: float, E: float, nu: float = 0.3) -> float:
    """Maximum contact pressure for two spheres in contact (Hertz theory).

    p0 = (3*F/(2*pi*a^2)), a = (3*FR/(4*E*))^(1/3) where E* = E/(1-nu^2)
    Returns p0 [Pa].
    """
    E_star = E / (1.0 - nu ** 2)
    a = ((3.0 * F * R) / (4.0 * E_star)) ** (1.0 / 3.0)
    p0 = np.sqrt(3.0 * F / (2.0 * np.pi * a ** 2))
    return p0


def failure_theory(sigma_x: float, sigma_y: float, tau_xy: float, theory: str = 'max_normal') -> float:
    """Select 2D failure theory and return equivalent stress or safety factor indicator.

    theories: 'max_normal' (max principal stress magnitude),
              'tresca' (max shear stress = (sigma1-sigma3)/2),
              'von_mises' (von Mises equivalent stress).
    """
    s1, s3 = principal_stresses(sigma_x, sigma_y, tau_xy)
    if theory == 'max_normal':
        return max(abs(s1), abs(s3))
    elif theory == 'tresca':
        return (s1 - s3) / 2.0
    elif theory == 'von_mises':
        return von_mises_stress(sigma_x, sigma_y, tau_xy)
    else:
        raise ValueError(f"Unknown failure theory: {theory}")


# ---- MAIN EXECUTION BLOCK ----
if __name__ == '__main__':
    print("=== Solid Mechanics & Machine Design Toolkit Demo ===\n")

    # 1. FEA Truss Example
    print("--- 1. Truss Element Stiffness ---")
    E = 210e9       # Pa, steel
    A = 0.001       # m^2
    L = 1.0         # m
    k = truss_element_stiffness(E, A, L)
    print(f"Truss stiffness matrix:\n{k}\n")

    # 2. Stress Transformation & Mohr's Circle
    print("--- 2. Stress Transformation & Mohr's Circle ---")
    sigma_x, sigma_y, tau_xy = 100e6, 50e6, 30e6   # Pa
    theta = np.radians(30.0)
    sxp, syp, tuxy = transform_stress(sigma_x, sigma_y, tau_xy, theta)
    center, radius, sigma1, sigma3 = mohrs_circle(sigma_x, sigma_y, tau_xy)
    vm = von_mises_stress(sigma_x, sigma_y, tau_xy)
    tc = tresca_stress(sigma_x, sigma_y, tau_xy)
    print(f"Transformed stress sigma_x'={sxp/1e6:.2f} MPa, sigma_y'={syp/1e6:.2f} MPa, tau_xy'={tuxy/1e6:.2f} MPa")
    print(f"Mohr's Circle: Center={center/1e6:.2f} MPa, Radius={radius/1e6:.2f} MPa")
    print(f"Principal Stresses: sigma1={sigma1/1e6:.2f} MPa, sigma3={sigma3/1e6:.2f} MPa")
    print(f"Von Mises: {vm/1e6:.2f} MPa, Tresca: {tc/1e6:.2f} MPa\n")

    # 3. SDOF Vibrations
    print("--- 3. SDOF Damped Vibration ---")
    m, c, k = 10.0, 50.0, 2000.0   # kg, Ns/m, N/m
    omega_n, f_n = sdof_natural_freq(m, k)
    print(f"SDOF natural freq: {f_n:.2f} Hz (omega_n={omega_n:.2f} rad/s)")
    t, x = sdof_damped_response(m, c, k, F0=100.0, omega=20.0, t_max=5.0)
    plt.figure(figsize=(8, 3))
    plt.plot(t, x, label=f"F={F0}N, omega={omega_n:.1f} rad/s")
    plt.title("SDOF Harmonic Response (Damped)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/tmp/sdo_damped_response.png')
    plt.close()
    print("Plot saved to /tmp/sdo_damped_response.png\n")

    # 4. Fatigue & Fracture
    print("--- 4. Fatigue & Fracture Mechanics ---")
    # S-N curve example: (500 MPa at 1e3 cycles), (100 MPa at 1e6 cycles)
    stress_at_10k = s_n_stress(500e6, 1e3, 100e6, 1e6, 10e3)
    print(f"Estimated stress at 10^4 cycles: {stress_at_10k/1e6:.2f} MPa")
    # Goodman & Gerber
    sigma_a, sigma_m = 200e6, 100e6
    su, sy = 600e6, 500e6
    n_goodman = goodman_criterion(sigma_a, sigma_m, su, sy)
    n_gerber = gerber_criterion(sigma_a, sigma_m, su, sy)
    print(f"Goodman safety factor n = {n_goodman:.3f}")
    print(f"Gerber safety factor n  = {n_gerber:.3f}")
    # Paris' Law
    da_dN = 1e-10          # m/cycle
    C, m_par = 1e-10, 3.0
    Delta_K = paris_law(da_dN, C, m_par)
    print(f"Paris Law: da/dN={da_dN} => Delta K = {Delta_K:.4e} Pa*m^0.5\n")

    # 5. Structural Optimization
    print("--- 5. Beam Cross-Section Weight Minimization ---")
    L_beam, P, sigma_allow = 2.0, 5000.0, 250e6   # m, N, Pa
    A_opt, obj_val = beam_weight_minimization(L_beam, P, sigma_allow)
    print(f"Optimized cross-section area: {A_opt:.6f} m^2")
    print(f"Objective (area) value: {obj_val:.6f} m^2\n")

    # 6. Rotordynamics & Failure Theories
    print("--- 6. Rotordynamics & Failure Theories ---")
    omega_c = critical_speed(500.0, 2e7)   # kg, Nm/rad
    print(f"Critical speed: {omega_c:.1f} RPM")
    eq_vm = failure_theory(100e6, 50e6, 30e6, 'von_mises')
    eq_tres = failure_theory(100e6, 50e6, 30e6, 'tresca')
    print(f"Von Mises equivalent stress: {eq_vm/1e6:.2f} MPa")
    print(f"Tresca equivalent stress:      {eq_tres/1e6:.2f} MPa")
    # Hertz contact example
    F, R, E = 500.0, 0.02, 210e9   # N, m, Pa
    p0 = hertzian_contact_pressure(F, R, E)
    print(f"Hertzian max contact pressure: {p0/1e6:.2f} MPa\n")

    print("=== Demo Complete ===")