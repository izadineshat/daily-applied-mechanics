#!/usr/bin/env python3
"""
Mechanical Engineering Toolkit
==============================
A standalone, production-ready Python module for applied mechanics and solid mechanics.

Covers:
- Finite Element Analysis (1D/2D truss/beam/plate stiffness matrices)
- Stress & Strain Transformations (Mohr's circle, Von Mises, Tresca)
- Mechanical Vibrations (modal analysis, SDOF/MDOF damped response, state-space)
- Fatigue & Fracture Mechanics (S-N curves, Goodman/Gerber diagrams, Paris' Law)
- Structural Optimization (cross-section weight minimization via gradient method)
- Rotordynamics, Contact Mechanics, and Failure Theories (Tresca, Von Mises, Coulomb-Mohr)

Author: Computational Mechanical Engineering Toolkit
License: MIT
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =============================================================================
# FUNDAMENTAL MATERIAL AND SECTION CLASSES
# =============================================================================

class Material:
    """Represents a linear elastic isotropic material."""

    def __init__(self, name: str, E: float, nu: float, rho: float = 0.0, Sy: float = 0.0):
        self.name = name
        self.E = E          # Young's Modulus [Pa]
        self.nu = nu        # Poisson's Ratio [-]
        self.rho = rho      # Density [kg/m^3]
        self.Sy = Sy        # Yield Strength [Pa]


class CrossSection:
    """Represents a structural cross-section with area properties."""

    def __init__(self, A: float, I: float, name: str = "Custom"):
        self.A = A          # Cross-sectional Area [m^2]
        self.I = I          # Second moment of area [m^4]
        self.name = name


# =============================================================================
# FINITE ELEMENT ANALYSIS
# =============================================================================

def truss_element_stiffness(E: float, A: float, L: float) -> np.ndarray:
    """
    Compute the local stiffness matrix for a 2-node 1D truss element.

    Parameters:
    -----------
    E : float
        Young's modulus [Pa]
    A : float
        Cross-sectional area [m^2]
    L : float
        Element length [m]

    Returns:
    --------
    K : ndarray (4, 4)
        Local stiffness matrix in global coordinates for 2D truss (u_x, u_y at each node)
    """
    k = np.array([[1, -1], [-1, 1]]) * E * A / L
    K = np.zeros((4, 4))
    K[0:2, 0:2] = k
    K[2:4, 2:4] = k
    K[0:2, 2:4] = -k
    K[2:4, 0:2] = -k
    return K


def beam_element_stiffness(E: float, I: float, A: float, L: float) -> np.ndarray:
    """
    Compute the stiffness matrix for a 2D Bernoulli-Euler beam element (2 nodes, 3 DOFs each).

    DOFs per node: [u, v, theta]

    Parameters:
    -----------
    E : float
        Young's modulus [Pa]
    I : float
        Area moment of inertia [m^4]
    A : float
        Cross-sectional area [m^2]
    L : float
        Element length [m]

    Returns:
    --------
    K : ndarray (6, 6)
        Consistent stiffness matrix
    """
    K = np.zeros((6, 6))

    # Axial terms
    K[0, 0] = E * A / L
    K[3, 3] = E * A / L
    K[0, 3] = -E * A / L
    K[3, 0] = -E * A / L

    # Flexural terms
    k_flex = E * I / L**3 * np.array([
        [12,     6*L,    -12,    6*L],
        [6*L,    4*L**2, -6*L,   2*L**2],
        [-12,    -6*L,   12,     -6*L],
        [6*L,    2*L**2, -6*L,   4*L**2]
    ])

    K[1:5, 1:5] += k_flex
    return K


def assemble_global_stiffness(elements: list, nodes: np.ndarray) -> np.ndarray:
    """
    Assemble global stiffness matrix from element connectivities.

    Parameters:
    -----------
    elements : list of dicts
        Each element is a dictionary with keys 'nodes' (list of 2 node indices), 'K' (element stiffness matrix)
    nodes : ndarray (N, 2)
        Node coordinates (x, y)

    Returns:
    --------
    Kg : ndarray (2*N, 2*N)
        Global stiffness matrix (for 2D truss)
    """
    num_nodes = len(nodes)
    Kg = np.zeros((2 * num_nodes, 2 * num_nodes))

    for el in elements:
        node_indices = el['nodes']
        K_local = el['K']

        for i, ni in enumerate(node_indices):
            for j, nj in enumerate(node_indices):
                Kg[2*ni:2*ni+2, 2*nj:2*nj+2] += K_local[2*i:2*i+2, 2*j:2*j+2]

    return Kg


def solve_truss(nodes: np.ndarray, elements: list, loads: dict, boundary_conditions: dict) -> np.ndarray:
    """
    Solve a 2D static truss system.

    Parameters:
    -----------
    nodes : ndarray (N, 2)
        Coordinates of nodes (x, y)
    elements : list of dicts
        Element definitions with keys 'nodes', 'E', 'A'
    loads : dict
        Load vector {node_index: [Fx, Fy]}
    boundary_conditions : dict
        Fixed DOFs {node_index: ['x', 'y']}

    Returns:
    --------
    displacements : ndarray (2*N,)
        Displacement vector
    """
    num_nodes = len(nodes)
    total_dofs = 2 * num_nodes

    # Build global stiffness matrix
    element_matrices = []
    for el in elements:
        n1, n2 = el['nodes']
        node1 = nodes[n1]
        node2 = nodes[n2]
        dx = node2[0] - node1[0]
        dy = node2[1] - node1[1]
        L = np.sqrt(dx**2 + dy**2)
        c = dx / L if L != 0 else 0
        s = dy / L if L != 0 else 0

        # Local to global transformation
        T = np.array([
            [c, s, 0, 0],
            [0, 0, c, s]
        ])
        K_local = truss_element_stiffness(el['E'], el['A'], L)
        K_global = T.T @ K_local @ T
        element_matrices.append({'nodes': el['nodes'], 'K': K_global})

    # Assemble global stiffness
    Kg = np.zeros((total_dofs, total_dofs))
    for el in element_matrices:
        for i, ni in enumerate(el['nodes']):
            for j, nj in enumerate(el['nodes']):
                Kg[2*ni:2*ni+2, 2*nj:2*nj+2] += el['K'][2*i:2*i+2, 2*j:2*j+2]

    # Apply boundary conditions
    fixed_dofs = []
    for node_idx, directions in boundary_conditions.items():
        if 'x' in directions:
            fixed_dofs.append(2 * node_idx)
        if 'y' in directions:
            fixed_dofs.append(2 * node_idx + 1)

    free_dofs = [dof for dof in range(total_dofs) if dof not in fixed_dofs]

    # Apply loads
    F = np.zeros(total_dofs)
    for node_idx, force in loads.items():
        F[2 * node_idx] += force[0]
        F[2 * node_idx + 1] += force[1]

    # Solve reduced system
    K_reduced = Kg[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    U_reduced = np.linalg.solve(K_reduced, F_reduced)

    displacements = np.zeros(total_dofs)
    displacements[free_dofs] = U_reduced

    return displacements


# =============================================================================
# STRESS & STRAIN TRANSFORMATIONS
# =============================================================================

def mohrs_circle(sigma_x: float, sigma_y: float, tau_xy: float) -> tuple:
    """
    Perform plane stress transformation using Mohr's Circle.

    Parameters:
    -----------
    sigma_x, sigma_y : float
        Normal stresses in x and y directions
    tau_xy : float
        Shear stress

    Returns:
    --------
    sigma_avg, R, principal_stresses, max_shear
    """
    sigma_avg = (sigma_x + sigma_y) / 2
    R = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R
    tau_max = R
    return sigma_avg, R, (sigma_1, sigma_2), tau_max


def von_mises_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """
    Calculate von Mises effective stress under plane stress.

    Returns:
    --------
    sigma_vm : float
    """
    return np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)


def tresca_stress(sigma_1: float, sigma_2: float, sigma_3: float = 0.0) -> float:
    """
    Calculate Tresca (maximum shear stress theory) equivalent stress.

    Returns:
    --------
    sigma_tc : float
    """
    return max(abs(sigma_1 - sigma_2), abs(sigma_2 - sigma_3), abs(sigma_3 - sigma_1)) / 2


def principal_strains(epsilon_x: float, epsilon_y: float, gamma_xy: float) -> tuple:
    """
    Compute principal strains from strain components.

    Returns:
    --------
    epsilon_1, epsilon_2 : float
    """
    epsilon_avg = (epsilon_x + epsilon_y) / 2
    R = np.sqrt(((epsilon_x - epsilon_y) / 2)**2 + (gamma_xy / 2)**2)
    return epsilon_avg + R, epsilon_avg - R


# =============================================================================
# MECHANICAL VIBRATIONS
# =============================================================================

def modal_analysis(K: np.ndarray, M: np.ndarray, num_modes: int = 5) -> tuple:
    """
    Perform undamped modal analysis.

    Parameters:
    -----------
    K : ndarray (N, N)
        Stiffness matrix
    M : np.ndarray (N, N)
        Mass matrix
    num_modes : int
        Number of modes to extract

    Returns:
    --------
    omega_n : ndarray (num_modes,)
        Natural frequencies [rad/s]
    Phi : ndarray (N, num_modes)
        Mode shapes (columns)
    """
    eigenvalues, eigenvectors = la.eigh(K, M)
    idx_sorted = np.argsort(eigenvalues)
    omega_n = np.sqrt(eigenvalues[idx_sorted][:num_modes])
    Phi = eigenvectors[:, idx_sorted][:, :num_modes]
    return omega_n, Phi


def sdof_response(m: float, c: float, k: float, F0: float, omega: float,
                  t_max: float = 10.0, dt: float = 0.01) -> tuple:
    """
    Time-domain response of an SDOF system under harmonic excitation.

    Parameters:
    -----------
    m, c, k : float
        Mass, damping coefficient, stiffness
    F0 : float
        Force amplitude
    omega : float
        Excitation frequency [rad/s]
    t_max : float
        Simulation time [s]
    dt : float
        Time step [s]

    Returns:
    --------
    t_array : ndarray
    x_array : ndarray
        Displacement response
    """
    t_array = np.arange(0, t_max, dt)
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))

    # Steady-state amplitude and phase
    X = F0 / np.sqrt((k - m*omega**2)**2 + (c*omega)**2)
    phi = np.arctan2(c * omega, k - m * omega**2)

    x_ss = X * np.sin(omega * t_array - phi)

    return t_array, x_ss


def mdof_state_space(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                     x0: np.ndarray, v0: np.ndarray,
                     t_array: np.ndarray) -> tuple:
    """
    State-space time integration for MDOF systems.

    Parameters:
    -----------
    M, C, K : ndarray (N, N)
        Mass, damping, stiffness matrices
    x0, v0 : ndarray (N,)
        Initial displacement and velocity
    t_array : ndarray
        Time points

    Returns:
    --------
    x_out : ndarray (len(t_array), N)
        Displacement history
    v_out : ndarray (len(t_array), N)
        Velocity history
    """
    N = M.shape[0]

    def state_deriv(t, state):
        x = state[:N]
        v = state[N:]
        dxdt = v
        dvdt = np.linalg.solve(M, -K @ x - C @ v)
        return np.concatenate([dxdt, dvdt])

    from scipy.integrate import odeint
    y0 = np.concatenate([x0, v0])
    sol = odeint(state_deriv, y0, t_array)
    return sol[:, :N], sol[:, N:]


# =============================================================================
# FATIGGE & FRACTURE MECHANICS
# =============================================================================

def sn_curve_stress(N: np.ndarray, C: float, m: float) -> np.ndarray:
    """
    Generate S-N curve using Basquin equation: S = C * N^(-1/m)

    Parameters:
    -----------
    N : ndarray
        Number of cycles
    C, m : float
        Material constants

    Returns:
    --------
    S : ndarray
        Stress amplitude corresponding to N cycles
    """
    return C * N**(-1/m)


def miner_rule(cycles: np.ndarray, stresses: np.ndarray, C: float, m: float) -> float:
    """
    Apply Miner's rule for cumulative damage.

    Parameters:
    -----------
    cycles : ndarray
        Number of cycles at each stress level
    stresses : ndarray
        Stress amplitudes
    C, m : float
        Basquin parameters

    Returns:
    --------
    D : float
        Damage fraction (failure when D >= 1)
    """
    Nf = (C / stresses)**m
    return np.sum(cycles / Nf)


def goodman_diagram(Se: float, Sut: float) -> tuple:
    """
    Generate Goodman line parameters.

    Returns:
    --------
    Sa_max : float
        Max allowable alternating stress at zero mean stress
    slope : float
        Slope of Goodman line
    """
    Sa_max = Se
    slope = -Se / Sut
    return Sa_max, slope


def gerber_diagram(Se: float, Sut: float) -> float:
    """
    Gerber criterion coefficient for fatigue limit.

    Returns:
    --------
    A : float
        Coefficient A in Sa/A = ...
    """
    return (1 - (Se / Sut)**2) / Sut


def paris_law(C: float, m: float, delta_K: float) -> float:
    """
    Paris' Law crack growth rate: da/dN = C * (ΔK)^m

    Parameters:
    -----------
    C, m : float
        Material constants
    delta_K : float
        Stress intensity factor range [MPa*sqrt(m)]

    Returns:
    --------
    da_dN : float
        Crack growth per cycle [m/cycle]
    """
    return C * delta_K**m


def simulate_crack_growth(a0: float, C: float, m: float, delta_sigma_func,
                          N_max: int, step_size: int = 1) -> tuple:
    """
    Simulate crack propagation using Paris' Law.

    Parameters:
    -----------
    a0 : float
        Initial crack length [m]
    C, m : float
        Paris law constants
    delta_sigma_func : callable
        Function returning Δσ as function of crack length a
    N_max : int
        Maximum number of cycles
    step_size : int
        Step size in cycles

    Returns:
    --------
    cycles : ndarray
    crack_lengths : ndarray
    """
    cycles = []
    crack_lengths = []

    a = a0
    n = 0

    while a < 0.1 and n < N_max:  # Stop at arbitrary failure size or max cycles
        delta_sigma = delta_sigma_func(a)
        delta_K = delta_sigma * np.sqrt(np.pi * a)
        da_dN = paris_law(C, m, delta_K)
        a += da_dN * step_size
        n += step_size
        cycles.append(n)
        crack_lengths.append(a)

    return np.array(cycles), np.array(crack_lengths)


# =============================================================================
# STRUCTURAL OPTIMIZATION
# =============================================================================

def optimize_beam_cross_section(material: Material, length: float,
                                max_deflection: float, load: float) -> dict:
    """
    Minimize beam weight subject to stress and deflection constraints.

    Simply supported beam with central point load.

    Parameters:
    -----------
    material : Material
        Beam material
    length : float
        Span length [m]
    max_deflection : float
        Allowable midspan deflection [m]
    load : float
        Central point load [N]

    Returns:
    --------
    result : dict
        Optimal height, width, volume, mass
    """
    def objective(x):
        h, w = x
        return material.rho * length * h * w  # Mass

    def stress_constraint(x):
        h, w = x
        I = h * w**3 / 12
        y = w / 2
        M = load * length / 4
        sigma = M * y / I
        return material.Sy - sigma  # >= 0

    def deflection_constraint(x):
        h, w = x
        I = h * w**3 / 12
        delta = load * length**3 / (48 * material.E * I)
        return max_deflection - delta  # >= 0

    constraints = [
        {'type': 'ineq', 'fun': stress_constraint},
        {'type': 'ineq', 'fun': deflection_constraint}
    ]

    bounds = [(0.01, 1.0), (0.01, 1.0)]  # h, w in meters
    x0 = [0.2, 0.1]

    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    h_opt, w_opt = res.x
    A_opt = h_opt * w_opt
    mass_opt = res.fun

    return {
        'height': h_opt,
        'width': w_opt,
        'area': A_opt,
        'mass': mass_opt,
        'volume': A_opt * length
    }


# =============================================================================
# ROTORDYNAMICS
# =============================================================================

def critical_speed(journal_bearing_stiffness: float, rotor_mass: float) -> float:
    """
    Estimate first critical speed of a simple rotor system.

    Parameters:
    -----------
    k : float
        Bearing stiffness [N/m]
    m : float
        Rotor mass [kg]

    Returns:
    --------
    omega_c : float
        Critical speed [rad/s]
    """
    return np.sqrt(k / rotor_mass)


# =============================================================================
# CONTACT MECHANICS
# =============================================================================

def hertzian_contact_stress(F: float, a: float) -> float:
    """
    Maximum contact pressure between two spheres (Hertzian contact).

    Parameters:
    -----------
    F : float
        Applied normal force [N]
    a : float
        Contact radius [m]

    Returns:
    --------
    p0 : float
        Maximum contact pressure [Pa]
    """
    return 3 * F / (2 * np.pi * a**2)


def contact_radius_spheres(F: float, R: float, E_eff: float) -> float:
    """
    Contact radius for sphere-on-flat contact.

    Parameters:
    -----------
    F : float
        Normal force [N]
    R : float
        Radius of curvature [m]
    E_eff : float
        Effective modulus [Pa]

    Returns:
    --------
    a : float
        Contact radius [m]
    """
    return (3 * F * R / (4 * E_eff))**(1/3)


# =============================================================================
# FAILURE THEORIES
# =============================================================================

class FailureTheory:
    """Collection of failure criteria evaluators."""

    @staticmethod
    def von_mises(sigma_x: float, sigma_y: float, tau_xy: float, Sy: float) -> float:
        """Return safety factor based on Von Mises criterion."""
        sigma_vm = von_mises_stress(sigma_x, sigma_y, tau_xy)
        return Sy / sigma_vm if sigma_vm > 0 else float('inf')

    @staticmethod
    def tresca(sigma_x: float, sigma_y: float, tau_xy: float, Sy: float) -> float:
        """Return safety factor based on Tresca criterion."""
        _, _, (sigma_1, sigma_2), _ = mohrs_circle(sigma_x, sigma_y, tau_xy)
        sigma_max_shear = (sigma_1 - sigma_2) / 2
        return Sy / (2 * sigma_max_sheff) if sigma_max_shear > 0 else float('inf')

    @staticmethod
    def coulomb_mohr(sigma_1: float, sigma_3: float, c: float, phi: float) -> float:
        """
        Coulomb-Mohr failure envelope.

        Parameters:
        -----------
        sigma_1, sigma_3 : float
            Major and minor principal stresses
        c : float
            Cohesion [Pa]
        phi : float
            Friction angle [rad]

        Returns:
        --------
        safety_factor : float
        """
        tau = (sigma_1 - sigma_3) / 2
        sigma_n = (sigma_1 + sigma_3) / 2
        tau_failure = c * np.cos(phi) - sigma_n * np.sin(phi)
        sf = tau_failure / tau if tau > 0 else float('inf')
        return sf


# =============================================================================
# DEMO / MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MECHANICAL ENGINEERING TOOLKIT - DEMONSTRATION")
    print("=" * 70)

    # --- Example 1: Truss Analysis ---
    print("\n--- 1. Truss Static Analysis ---")
    nodes = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    elements = [
        {'nodes': [0, 1], 'E': 200e9, 'A': 1e-4},
        {'nodes': [1, 2], 'E': 200e9, 'A': 1e-4},
        {'nodes': [2, 3], 'E': 200e9, 'A': 1e-4},
        {'nodes': [3, 0], 'E': 200e9, 'A': 1e-4},
        {'nodes': [0, 2], 'E': 200e9, 'A': 1e-4}
    ]
    loads = {2: [0, -1000]}
    bc = {0: ['x', 'y'], 1: ['y']}
    disp = solve_truss(nodes, elements, loads, bc)
    print("Node 2 displacement:", disp[4:6])

    # --- Example 2: Stress Transformation ---
    print("\n--- 2. Stress Transformation ---")
    sigma_avg, R, principals, tau_max = mohrs_circle(100e6, 40e6, 30e6)
    print(f"Principal Stresses: σ₁={principals[0]/1e6:.2f} MPa, σ₂={principals[1]/1e6:.2f} MPa")
    print(f"Max Shear Stress: {tau_max/1e6:.2f} MPa")
    sigma_vm = von_mises_stress(100e6, 40e6, 30e6)
    print(f"Von Mises Stress: {sigma_vm/1e6:.2f} MPa")

    # --- Example 3: Modal Analysis ---
    print("\n--- 3. Modal Analysis ---")
    K_modal = np.array([
        [2, -1, 0, 0],
        [-1, 2, -1, 0],
        [0, -1, 2, -1],
        [0, 0, -1, 1]
    ]) * 1e5
    M_modal = np.eye(4) * 0.5
    omega_n, Phi = modal_analysis(K_modal, M_modal, num_modes=3)
    print("Natural Frequencies [rad/s]:", omega_n)

    # --- Example 4: Fatigue ---
    print("\n--- 4. Fatigue Analysis ---")
    N_cycles = np.logspace(2, 7, 100)
    C_basq = 800e6
    m_basq = 5
    S_amplitudes = sn_curve_stress(N_cycles, C_basq, m_basq)
    D = miner_rule(np.array([1e4, 2e4]), np.array([200e6, 150e6]), C_basq, m_basq)
    print(f"Cumulative Damage Index (Miner Rule): {D:.3f}")

    # --- Example 5: Structural Optimization ---
    print("\n--- 5. Structural Optimization ---")
    steel = Material("Steel", E=200e9, nu=0.3, rho=7850, Sy=250e6)
    opt_result = optimize_beam_cross_section(steel, length=4.0, max_deflection=0.01, load=5000)
    print(f"Optimal Height: {opt_result['height']*1000:.1f} mm")
    print(f"Optimal Width: {opt_result['width']*1000:.1f} mm")
    print(f"Minimum Mass: {opt_result['mass']:.2f} kg")

    # --- Example 6: Rotordynamics ---
    print("\n--- 6. Rotordynamics ---")
    omega_crit = critical_speed(k=1e7, m=50)
    print(f"Critical Speed: {omega_crit:.2f} rad/s ({omega_crit/(2*np.pi):.2f} Hz)")

    # --- Example 7: Contact Mechanics ---
    print("\n--- 7. Contact Mechanics ---")
    E_sphere = 200e9
    nu_sphere = 0.3
    E_eff = E_sphere / (1 - nu_sphere**2)
    a_contact = contact_radius_spheres(F=1000, R=0.05, E_eff=E_eff)
    p_max = hertzian_contact_stress(F=1000, a=a_contact)
    print(f"Contact Radius: {a_contact*1000:.2f} mm")
    print(f"Max Contact Pressure: {p_max/1e6:.2f} MPa")

    # --- Example 8: Failure Theories ---
    print("\n--- 8. Failure Theories ---")
    ft = FailureTheory()
    sf_vm = ft.von_mises(150e6, 50e6, 40e6, 300e6)
    print(f"Safety Factor (Von Mises): {sf_vm:.2f}")

    # --- Plot S-N Curve ---
    plt.figure(figsize=(8, 5))
    plt.loglog(N_cycles, S_amplitudes / 1e6, 'b-', linewidth=2)
    plt.xlabel('Number of Cycles (N)')
    plt.ylabel('Stress Amplitude (MPa)')
    plt.title("S-N Curve (Basquin Equation)")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY.")
    print("=" * 70)