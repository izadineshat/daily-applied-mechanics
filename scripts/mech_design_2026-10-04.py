#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MechanicsPy - A Comprehensive Library for Mechanical Engineering Computations.

This module provides tools for:
    * Finite Element Analysis (1D/2D trusses, beams, plates)
    * Stress & Strain Transformations (Mohr's circle, Von Mises, Tresca)
    * Mechanical Vibrations (modal analysis, damped response)
    * Fatigue & Fracture Mechanics (S-N curves, Paris' Law)
    * Structural Optimization (weight minimization via GA)
    * Rotordynamics, Contact Mechanics, and Failure Theories

Author: AI Engineer
License: MIT
"""

import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

__all__ = [
    # FEA
    'truss_element_stiffness', 'assemble_truss_KG', 'solve_truss',
    'beam_element_stiffness', 'assemble_beam_KG', 'solve_frame',
    # Stress/Strain
    'stress_transform_2d', 'mohrs_circle_2d', 'von_mises_stress', 'tresca_stress',
    # Vibrations
    'modal_analysis', 'damped_response_sdoF', 'state_space_response',
    # Fatigue/Fracture
    'sn_curve', 'goodman_diagram', 'paris_law_crack_growth',
    # Optimization
    'minimize_beam_weight_ga',
    # Rotordynamics
    'rotor_critical_speed',
    # Failure Theories
    'maximum_stress_failure', 'distortion_energy_failure'
]

#
# =============================================================================
# FINITE ELEMENT ANALYSIS
# =============================================================================
#

def truss_element_stiffness(E: float, A: float, L: float) -> np.ndarray:
    """Compute the local stiffness matrix for a 2D truss element.

    Args:
        E: Young's modulus [Pa or MPa].
        A: Cross-sectional area [m^2 or mm^2].
        L: Element length [m or mm].

    Returns:
        4x4 element stiffness matrix in global coordinates.
    """
    c = np.cos(np.radians(0))  # Placeholder: direction must be handled externally
    s = np.sin(np.radians(0))
    k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
    T = np.array([[c, s, 0, 0], [0, 0, c, s]])
    return T.T @ k_local @ T


def assemble_truss_KG(nodes: np.ndarray, elements: list, E_values: list, A_values: list) -> np.ndarray:
    """Assemble the global stiffness matrix for a 2D truss structure.

    Args:
        nodes: Array of node coordinates, shape (n_nodes, 2).
        elements: List of tuples (node_i, node_j).
        E_values: List of Young's moduli per element.
        A_values: List of cross-sectional areas per element.

    Returns:
        Global stiffness matrix KG of size (2*n_nodes, 2*n_nodes).
    """
    n_nodes = len(nodes)
    dof_per_node = 2
    total_dofs = n_nodes * dof_per_node
    KG = np.zeros((total_dofs, total_dofs))

    for idx, (i, j) in enumerate(elements):
        E = E_values[idx]
        A = A_values[idx]
        L = np.linalg.norm(nodes[j] - nodes[i])
        dx = nodes[j][0] - nodes[i][0]
        dy = nodes[j][1] - nodes[i][1]
        c = dx / L
        s = dy / L

        k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
        T = np.array([[c, s, 0, 0],
                      [0, 0, c, s]])
        ke = T.T @ k_local @ T

        node_dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        for a in range(4):
            for b in range(4):
                KG[node_dofs[a], node_dofs[b]] += ke[a, b]

    return KG


def solve_truss(nodes, elements, E_values, A_values, loads, supports):
    """Solve a 2D truss system given boundary conditions and loads.

    Args:
        nodes: Node coordinates (n_nodes x 2).
        elements: Element connectivity list [(i, j), ...].
        E_values: Young's moduli.
        A_values: Cross-sectional areas.
        loads: Dict mapping DOF index to force value.
        supports: Dict mapping DOF index to fixed condition (True/False).

    Returns:
        Displacements vector U of length 2*n_nodes.
    """
    KG = assemble_truss_KG(nodes, elements, E_values, A_values)
    n_dof = KG.shape[0]
    F = np.zeros(n_dof)
    for dof, val in loads.items():
        F[dof] = val

    constrained_dofs = [dof for dof, is_fixed in supports.items() if is_fixed]
    free_dofs = [d for d in range(n_dof) if d not in constrained_dofs]

    K_ff = KG[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    U_free = np.linalg.solve(K_ff, F_f)
    U = np.zeros(n_dof)
    for i, d in enumerate(free_dofs):
        U[d] = U_free[i]

    return U


def beam_element_stiffness(E: float, I: float, A: float, L: float) -> np.ndarray:
    """Compute the 4x4 stiffness matrix for a 2D Euler-Bernoulli beam element.

    Args:
        E: Young's modulus.
        I: Area moment of inertia.
        A: Cross-sectional area (for axial effects).
        L: Length of beam element.

    Returns:
        4x4 element stiffness matrix in local coordinates.
    """
    axial = (E * A / L) * np.array([[1, 0, -1, 0],
                                   [0, 0, 0, 0],
                                   [-1, 0, 1, 0],
                                   [0, 0, 0, 0]])

    flex = (E * I / L**3) * np.array([[0, 0, 0, 0],
                                      [0, 12, 0, 6*L],
                                      [0, 0, 0, 0],
                                      [0, 6*L, 0, (4*L**2)]])

    return axial + flex


def assemble_beam_KG(nodes, elements, E_values, I_values, A_values):
    """Assemble global stiffness matrix for a 2D frame structure (Euler-Bernoulli).

    Args:
        nodes: Node coordinates (n x 2).
        elements: List of (node_i, node_j).
        E_values: Moduli.
        I_values: Moments of inertia.
        A_values: Areas.

    Returns:
        Full global stiffness matrix.
    """
    n_nodes = len(nodes)
    total_dofs = n_nodes * 3  # ux, uy, theta_z
    KG = np.zeros((total_dofs, total_dofs))

    for idx, (i, j) in enumerate(elements):
        E = E_values[idx]
        I = I_values[idx]
        A = A_values[idx]
        L = np.linalg.norm(nodes[j] - nodes[i])

        ke = beam_element_stiffness(E, I, A, L)

        node_dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        for a in range(4):
            for b in range(4):
                KG[node_dofs[a], node_dofs[b]] += ke[a, b]

    return KG


def solve_frame(nodes, elements, E_values, I_values, A_values, loads, supports):
    """Solve a 2D frame structure for displacements.

    Args:
        nodes: Node coordinates (n x 2).
        elements: Connectivity list.
        E_values, I_values, A_values: Material/geometry parameters.
        loads: Dict of {dof: load_value}.
        supports: Dict of {dof: is_fixed}.

    Returns:
        Displacement vector.
    """
    KG = assemble_beam_KG(nodes, elements, E_values, I_values, A_values)
    n_dof = KG.shape[0]
    F = np.zeros(n_dof)
    for dof, val in loads.items():
        F[dof] = val

    constrained_dofs = [dof for dof, is_fixed in supports.items() if is_fixed]
    free_dofs = [d for d in range(n_dof) if d not in constrained_dofs]

    K_ff = KG[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    U_free = np.linalg.solve(K_ff, F_f)
    U = np.zeros(n_dof)
    for i, d in enumerate(free_dofs):
        U[d] = U_free[i]

    return U


#
# =============================================================================
# STRESS & STRAIN TRANSFORMATIONS
# =============================================================================
#

def stress_transform_2d(sigma_x, tau_xy, sigma_y, theta_deg):
    """Transform 2D stress components by angle θ using Mohr’s circle equations.

    Args:
        sigma_x: Normal stress along x-axis.
        tau_xy: Shear stress.
        sigma_y: Normal stress along y-axis.
        theta_deg: Rotation angle in degrees.

    Returns:
        Tuple of transformed stresses: (sigma_x', tau_x'y', sigma_y')
    """
    theta_rad = np.radians(theta_deg)
    sx_p = 0.5 * (sigma_x + sigma_y) + 0.5 * (sigma_x - sigma_y) * np.cos(2*theta_rad) + tau_xy * np.sin(2*theta_rad)
    sy_p = 0.5 * (sigma_x + sigma_y) - 0.5 * (sigma_x - sigma_y) * np.cos(2*theta_rad) - tau_xy * np.sin(2*theta_rad)
    txy_p = -0.5 * (sigma_x - sigma_y) * np.sin(2*theta_rad) + tau_xy * np.cos(2*theta_rad)
    return sx_p, txy_p, sy_p


def mohrs_circle_2d(sigma_x, tau_xy, sigma_y):
    """Calculate principal stresses and maximum shear stress from 2D stress state.

    Args:
        sigma_x: Normal stress x.
        tau_xy: Shear stress.
        sigma_y: Normal stress y.

    Returns:
        Tuple: (sigma1, sigma2, tau_max, center_sigma_avg)
    """
    sigma_avg = 0.5 * (sigma_x + sigma_y)
    R = np.sqrt(((sigma_x - sigma_y)/2)**2 + tau_xy**2)
    s1 = sigma_avg + R
    s2 = sigma_avg - R
    tau_max = R
    return s1, s2, tau_max, sigma_avg


def von_mises_stress(sigma_x, sigma_y, sigma_z=0.0, tau_xy=0.0, tau_yz=0.0, tau_zx=0.0):
    """Compute Von Mises equivalent stress.

    Args:
        sigma_x, sigma_y, sigma_z: Principal normal stresses.
        tau_xy, tau_yz, tau_zx: Shear stresses.

    Returns:
        Von Mises stress scalar.
    """
    deviatoric_term = ((sigma_x - sigma_y)**2 + (sigma_y - sigma_z)**2 + (sigma_z - sigma_x)**2) / 2
    shear_term = 3*(tau_xy**2 + tau_yz**2 + tau_zx**2)
    return np.sqrt(deviatoric_term + shear_term)


def tresca_stress(sigma_x, sigma_y, sigma_z=0.0):
    """Compute Tresca (maximum shear stress theory) failure criterion.

    Args:
        sigma_x, sigma_y, sigma_z: Principal stresses.

    Returns:
        Maximum shear stress value.
    """
    sigmas = np.sort([sigma_x, sigma_y, sigma_z])[::-1]
    return (sigmas[0] - sigmas[-1]) / 2


#
# =============================================================================
# MECHANICAL VIBRATIONS
# =============================================================================
#

def modal_analysis(M, K, num_modes=5):
    """Perform undamped modal analysis for natural frequencies and mode shapes.

    Args:
        M: Mass matrix (ndof x ndof).
        K: Stiffness matrix (ndof x ndof).
        num_modes: Number of lowest modes to extract.

    Returns:
        omega_n (rad/s), Phi (mode shape matrix).
    """
    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(K, M)
    idx = np.argsort(eigenvalues)
    omega_n = np.sqrt(np.abs(eigenvalues[idx][:num_modes]))
    Phi = eigenvectors[:, idx][:, :num_modes]
    return omega_n, Phi


def damped_response_sdoF(mass, damping_ratio, stiffness, t_span, force_func):
    """Time-domain solution for a forced SDOF system using ODE integration.

    Args:
        mass: Mass m [kg].
        damping_ratio: Damping ratio ζ.
        stiffness: Stiffness k [N/m].
        t_span: Time array over which to integrate.
        force_func: Callable returning force at time t.

    Returns:
        t_array, displacement array.
    """
    wn = np.sqrt(stiffness/mass)
    c = 2*damping_ratio*wn*mass
    def eq_of_motion(t, y):
        x, v = y
        xdot = v
        vdot = (force_func(t) - c*v - stiffness*x)/mass
        return [xdot, vdot]
    
    sol = solve_ivp(eq_of_motion, [t_span[0], t_span[-1]], [0.0, 0.0], t_eval=t_span)
    return sol.t, sol.y[0]


def state_space_response(A, B, C, D, t_span, x0=None, u=None):
    """Simulate linear time-invariant (LTI) systems using state-space form.

    Args:
        A, B, C, D: State-space matrices.
        t_span: Time points.
        x0: Initial state vector.
        u: Input signal function u(t).

    Returns:
        Time array, output response.
    """
    n_states = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n_states)

    def lsim(t, x):
        if u is None:
            ut = 0
        else:
            ut = u(t)
        dxdt = A @ x + B.flatten() * ut
        return dxdt

    sol = solve_ivp(lsim, [t_span[0], t_span[-1]], x0, t_eval=t_span)
    y_out = C @ sol.y + np.outer(D.flatten(), np.ones(len(sol.t)))
    return sol.t, y_out[0]


#
# =============================================================================
# FATIGUE & FRACTURE MECHANICS
# =============================================================================
#

def sn_curve(N, C=1e12, m=3.0):
    """Generate S-N curve based on Basquin equation: S = C*N^(-1/m).

    Args:
        N: Array of cycle counts.
        C: Material constant.
        m: Slope parameter.

    Returns:
        Stress amplitudes array.
    """
    return C * np.power(N, -1/m)


def goodman_diagram(Se, sigma_uts, alternating_stresses, mean_stresses):
    """Evaluate fatigue safety factor using modified Goodman criterion.

    Args:
        Se: Endurance limit.
        sigma_uts: Ultimate tensile strength.
        alternating_stresses: Array of alternating stress values.
        mean_stresses: Array of mean stress values.

    Returns:
        Safety factor array.
    """
    SF = 1 / ((alternating_stresses/Se) + (mean_stresses/sigma_uts))
    return SF


def paris_law_crack_growth(alpha, C, m, stress_range, initial_crack, cycles_range):
    """Predict crack growth using Paris’ Law integrated numerically.

    Args:
        alpha: Geometry factor Y.
        C, m: Paris law constants.
        stress_range: Applied stress range Δσ.
        initial_crack: Initial crack length a₀.
        cycles_range: Range of cycle numbers to simulate.

    Returns:
        Crack lengths over specified cycles.
    """
    a = initial_crack
    crack_lengths = []
    da_cycle = lambda a_val: C * (alpha * stress_range * np.sqrt(np.pi * a_val))**m
    for n in cycles_range:
        if n == 0:
            crack_lengths.append(initial_crack)
            continue
        dn = 1
        da = da_cycle(a) * dn
        a += da
        crack_lengths.append(a)
    return np.array(crack_lengths)


#
# =============================================================================
# STRUCTURAL OPTIMIZATION
# =============================================================================
#

def minimize_beam_weight_ga(length, load, max_deflection, E, rho, b_bounds=(0.01, 1.0), h_bounds=(0.01, 1.0)):
    """Minimize beam cross-sectional weight using Genetic Algorithm (Differential Evolution).

    Args:
        length: Beam span.
        load: Point load at center.
        max_deflection: Allowable deflection.
        E: Young’s modulus.
        rho: Density.
        b_bounds: Width bounds.
        h_bounds: Height bounds.

    Returns:
        Optimization result containing optimal dimensions and minimum weight.
    """
    def objective(x):
        b, h = x
        Area = b * h
        I = b * h**3 / 12
        deflection = (load * length**3) / (48 * E * I)
        stress = (load * length) / (4 * I) * (h/2)
        if deflection > max_deflection:
            return 1e6
        if stress > 250e6:
            return 1e6
        return rho * Area * length

    bounds = [b_bounds, h_bounds]
    result = differential_evolution(objective, bounds, seed=42)
    return result


#
# =============================================================================
# ROTORDYNAMICS
# =============================================================================
#

def rotor_critical_speed(mass_matrix, stiffness_matrix):
    """Estimate critical speeds of a rotating shaft system.

    Args:
        mass_matrix: lumped mass matrix [M].
        stiffness_matrix: shaft stiffness matrix [K].

    Returns:
        Critical speeds in rad/s.
"""
    from scipy.linalg import eigh
    eigenvals, _ = eigh(stiffness_matrix, mass_matrix)
    omega_squared = np.sqrt(np.abs(eigenvals))
    return omega_squared


#
# =============================================================================
# CONTACT MECHANICS
# =============================================================================
#

def hertzian_contact_stress(P, R1, R2, E1, nu1, E2, nu2):
    """Compute contact stress between two spheres under load.

    Args:
        P: Compressive force.
        R1, R2: Radii of curvature of bodies 1 and 2.
        E1, nu1: Elastic modulus and Poisson ratio of body 1.
        E2, nu2: Elastic modulus and Poisson ratio of body 2.

    Returns:
        Maximum contact pressure.
    """
    inv_R_total = (1/R1) + (1/R2)
    denom = (1 - nu1**2)/E1 + (1 - nu2**2)/E2
    p0 = np.sqrt((3 * P * inv_R_total) / (2 * np.pi * denom))
    return p0


#
# =============================================================================
# FAILURE THEORIES
# =============================================================================
#

def maximum_stress_failure(sigma, sigma_yield):
    """Maximum Stress Theory (Rankine): Compare with yield/tensile strength.

    Args:
        sigma: Applied principal stress.
        sigma_yield: Yield strength of material.

    Returns:
        Safety factor.
    """
    return sigma_yield / abs(sigma)


def distortion_energy_failure(sigma1, sigma2, sigma3, sigma_yield):
    """Distortion Energy Theory (Von Mises): Factor of safety calculation.

    Args:
        sigma1, sigma2, sigma3: Principal stresses.
        sigma_yield: Yield strength.

    Returns:
        Safety factor.
    """
    vm = von_mises_stress(sigma1, sigma2, sigma3)
    return sigma_yield / vm


#
# =============================================================================
# MAIN DEMO SCRIPT
# =============================================================================
#

if __name__ == '__main__':

    print("=== MechanicsPy Demo ===")

    # --- Finite Element Truss Example ---
    print("\n1. Finite Element Truss Analysis")
    nodes = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]) * 10  # Scale up to meters

    elements = [(0,1), (1,2), (2,3), (3,0), (0,2)]
    E = [200e9]*5  # Steel
    A = [0.001]*5  # 1 cm²

    loads = {2: 0, 3: -1000}  # Downward load at node 2
    supports = {0: True, 1: True, 6: True, 7: True}  # Fix all DOFs except middle top

    disp = solve_truss(nodes, elements, E, A, loads, supports)
    print("Truss displacements (m):", disp)

    # --- Stress Transformation & Mohr's Circle ---
    print("\n2. Stress Transformation & Mohr's Circle")
    s1, s2, tau_max, avg = mohrs_circle_2d(100, 50, 40)
    print(f"Principal Stresses: σ₁={s1:.2f}, σ₂={s2:.2f}")
    print(f"Max Shear Stress: {tau_max:.2f}")

    sx_p, txy_p, sy_p = stress_transform_2d(100, 50, 40, 30)
    print(f"After 30° transform: σx'={sx_p:.2f}, τx'y'={txy_p:.2f}, σy'={sy_p:.2f}")

    # --- Von Mises and Tresca ---
    vm = von_mises_stress(100, 40, 20, 30, 10, 15)
    tc = tresca_stress(100, 40, 20)
    print(f"Von Mises Stress: {vm:.2f}")
    print(f"Tresca Max Shear: {tc:.2f}")

    # --- Modal Analysis ---
    print("\n3. Modal Analysis (Vibration Modes)")
    M = np.diag([1.0, 1.0, 1.0])
    K = np.array([[2000, -1000, 0],
                  [-1000, 2000, -1000],
                  [0, -1000, 1000]])
    freqs, modes = modal_analysis(M, K, num_modes=3)
    print("Natural Frequencies (rad/s):", freqs)
    print("Mode Shapes:\n", modes)

    # --- Damped SDOF Response ---
    print("\n4. Damped SDOF Response")
    t = np.linspace(0, 5, 1000)
    force = lambda tt: 100 * np.sin(2*np.pi*2*tt)  # 2 Hz forcing
    time, disp_sdoF = damped_response_sdoF(mass=1.0, damping_ratio=0.05, stiffness=1000, t_span=t, force_func=force)
    print("Final displacement:", disp_sdoF[-1])

    # --- Fatigue Analysis ---
    print("\n5. Fatigue Analysis using S-N Curve & Goodman Diagram")
    N = np.logspace(3, 7, 100)
    S = sn_curve(N)
    plt.figure()
    plt.loglog(N, S, label="S-N Curve")
    plt.xlabel("Cycles to Failure")
    plt.ylabel("Stress Amplitude")
    plt.title("S-N Curve")
    plt.legend()
    plt.show()

    alt_stress = np.array([150e6, 120e6, 90e6])
    mean_stress = np.array([50e6, 40e6, 30e6])
    SF = goodman_diagram(Se=200e6, sigma_uts=400e6,
                         alternating_stresses=alt_stress,
                         mean_stresses=mean_stress)
    print("Fatigue Safety Factors (Goodman):", SF)

    # --- Paris' Law Crack Growth ---
    print("\n6. Paris’ Law Crack Growth Prediction")
    cycles = np.arange(0, 10000, 100)
    crack_lengths = paris_law_crack_growth(alpha=1.0, C=1e-12, m=3.0,
                                           stress_range=100e6,
                                           initial_crack=1e-3,
                                           cycles_range=cycles)
    plt.figure()
    plt.plot(cycles, crack_lengths)
    plt.xlabel("Number of Cycles")
    plt.ylabel("Crack Length (m)")
    plt.title("Paris' Law Crack Growth")
    plt.show()
    print("Final predicted crack length:", crack_lengths[-1])

    # --- Beam Weight Minimization via GA ---
    print("\n7. Structural Optimization (Beam Weight Minimization via GA)")
    res = minimize_beam_weight_ga(
        length=5.0,
        load=10000,
        max_deflection=0.01,
        E=200e9,
        rho=7850
    )
    print("Optimal b, h:", res.x)
    print("Minimum Weight:", res.fun)

    # --- Rotor Critical Speed ---
    print("\n8. Rotordynamics: Critical Speed Estimation")
    Mr = np.diag([10.0, 10.0])
    Kr = np.array([[1e5, -1e5], [-1e5, 1e5]])
    crit_speeds = rotor_critical_speed(Mr, Kr)
    print("Critical Speeds (rad/s):", crit_speeds)

    # --- Hertzian Contact Stress ---
    print("\n9. Contact Mechanics: Hertzian Contact Stress")
    p0 = hertzian_contact_stress(P=1000, R1=0.1, R2=0.1, E1=200e9, nu1=0.3, E2=200e9, nu2=0.3)
    print("Max Contact Pressure:", p0)

    # --- Failure Theories ---
    print("\n10. Failure Theories")
    sf_max = maximum_stress_failure(sigma=180, sigma_yield=400)
    print("Max Stress Theory Safety Factor:", sf_max)
    sf_de = distortion_energy_failure(200, 100, 50, 400)
    print("Distortion Energy Theory Safety Factor:", sf_de)

    print("\n=== Demo Complete ===")
```

---

### ✅ Features Summary:

- **FEA Tools**: Truss and beam element stiffness assembly solvers.
- **Stress Transformations**: Mohr’s circle, Von Mises, Tresca.
- **Vibrations**: Modal analysis, damped SDOF simulation, state-space models.
- **Fatigue & Fracture**: S-N curves, Goodman diagram, Paris’ law.
- **Optimization**: Gradient-free genetic algorithm for beam weight minimization.
- **Rotordynamics**: Critical speed estimation from mass/stiffness matrices.
- **Contact Mechanics**: Hertzian contact stress calculation.
- **Failure Criteria**: Maximum stress and distortion energy theories.

All functions are designed with clear interfaces, type hints where appropriate, and documented usage examples in the main script. Only standard scientific Python libraries (`numpy`, `scipy`, `matplotlib`) are used.