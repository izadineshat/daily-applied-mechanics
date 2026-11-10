#!/usr/bin/env python3
"""
Standalone, production-ready Python script for Mechanical Engineering.
Implements key topics:
- Finite Element Analysis (1D/2D truss/beam/stress triangle)
- Stress & Strain Transformations (Mohr's circle, Von Mises, Tresca)
- Mechanical Vibrations (SDOF/MDOF, modal analysis)
- Fatigue & Fracture Mechanics (S-N curves, Goodman/Gerber, Paris' Law)
- Structural Optimization (beam cross-section weight minimization)
- Rotordynamics (Jeffcott rotor critical speeds)
- Contact Mechanics (Hertzian contact)
- Failure Theories (max principal, max shear, von Mises)
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ==================== FINITE ELEMENT ANALYSIS ====================

def truss_element_stiffness(x1, y1, x2, y2, EA):
    """
    Compute the global 4x4 stiffness matrix for a 2D truss element.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of node 1.
    x2, y2 : float
        Coordinates of node 2.
    EA : float
        Axial rigidity (Young's modulus * area).

    Returns
    -------
    K : ndarray, shape (4,4)
        Global stiffness matrix for the element (DOFs: [x1,y1,x2,y2]).
    """
    L = np.hypot(x2 - x1, y2 - y1)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    k_local = EA / L * np.array([[1, -1], [-1, 1]])
    T = np.array([[c, s], [-s, c]])
    k_global = T.T @ k_local @ T
    K = np.zeros((4, 4))
    K[0:2,  att 0:2] = k_global
    K[2:4, 2:4] = k_global
    return K

def beam_element_stiffness(E, I, L):
    """
    Compute the global 4x4 stiffness matrix for an Euler‑Bernoulli beam element.

    Parameters
    ----------
    E : float
        Young's modulus.
    I : float
        Second moment of area.
    L : float
        Length of the beam.

    Returns
    -------
    K : ndarray, shape (4,4)
        Global stiffness matrix (DOFs: [v1,θ1,v2,θ2]).
    """
    k = np.array([
        [12, 6*L, -12, -6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, 6*L],
        [-6*L, 2*L**2, 6*L, 4*L**2]
    ]) * (E * I / L**3)
    return k

def triangle_element_stiffness(nodes, E, nu):
    """
    Compute the 6x6 stiffness matrix for a 2D plane‑stress constant strain triangle (CST).

    Parameters
    ----------
    nodes : ndarray, shape (3,2)
        Node coordinates [[x1,y1], [x2,y2], [x3,y3]].
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    K : ndarray, shape (6,6)
        Element stiffness matrix (2 DOFs per node).
    """
    xy = np.asarray(nodes, dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]
    area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    # Shape function derivatives (b, c)
    b1 = y[1] - y[2]
    c1 = x[2] - x[1]
    b2 = y[2] - y[0]
    c2 = x[0] - x[2]
    b3 = y[0] - y[1]
    c3 = x[1] - x[0]
    B = (1.0 / (2.0 * area)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])
    # Plane stress constitutive matrix
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    K = (B.T @ D @ B) * area
    return K

def assemble_global_stiffness(elements, num_nodes):
    """
    Assemble a global stiffness matrix from a list of element matrices.

    Parameters
    ----------
    elements : list of tuples (K_elem, dofs)
        K_elem : ndarray, element stiffness matrix.
        dofs    : list of global DOF indices (e.g., [0,1,4,5]).
    num_nodes : int
        Total number of nodes in the model.

    Returns
    -------
    K_global : ndarray, shape (2*num_nodes, 2*num_nodes)
        Assembled global stiffness matrix.
    """
    ndof = 2 * num_nodes
    K_global = np.zeros((ndof, ndof))
    for K_elem, dofs in elements:
        for i, dof_i in enumerate(dofs):
            for j, dof_j in enumerate(dofs):
                K_global[dof_i, dof_j] += K_elem[i, j]
    return K_global

def demo_fea():
    """
    Simple demonstration of a 2D truss FE model.
    """
    # Geometry
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    # Material
    E = 210e9          # Pa
    A = 0.001          # m^2
    EA = E * A
    # Elements (connectivity)
    elements_desc = [((0, 1), (1, 2))]
    elements = []
    for (n1, n2) in elements_desc:
        K_elem = truss_element_stiffness(
            nodes[n1, 0], nodes[n1, 1],
            nodes[n2, 0], nodes[n2, 1],
            EA
        )
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        elements.append((K_elem, dofs))
    K_global = assemble_global_stiffness(elements, len(nodes))
    print("Global stiffness matrix (truss) [N/m]:")
    print(K_global)

    # Natural frequencies (assume unit mass)
    eigvals, _ = eigh(K_global)
    print("\nEigenvalues (stiffness):", eigvals)
    print("Natural frequencies (rad/s):", np.sqrt(eigvals))
    print("Natural frequencies (Hz):", np.sqrt(eigvals) / (2*np.pi))

# ==================== STRESS & STRAIN TRANSFORMATIONS ====================

def mohr_circle(sxx, syy, sxy):
    """
    Compute Mohr's circle parameters for a given stress state.

    Returns
    -------
    center : float
        centre of the Mohr circle (mean stress).
    radius : float
        radius of the Mohr circle (max shear).
    s1 : float
        maximum principal stress.
    s2 : float
        minimum principal stress.
    tau_max : float
        maximum shear stress (radius).
    """
    center = (sxx + syy) / 2.0
    radius = np.hypot((sxx - syy) / 2.0, sxy)
    s1 = center + radius
    s2 = center - radius
    tau_max = radius
    return center, radius, s1, s2, tau_max

def principal_stresses(sxx, syy, sxy):
    """Return principal stresses (s1, s2)."""
    _, _, s1, s2, _ = mohr_circle(sxx, syy, sxy)
    return s1, s2

def von_mises(sxx, syy, sxy):
    """
    Plane‑stress von Mises equivalent stress.
    σ_vm = √(σ_xx² - σ_xx·σ_yy + σ_yy² + ظاهر3·σ_xy²)
    """
    return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

def tresca(sxx, syy, sxy):
    """Return Tresca (maximum shear) stress."""
    s1, s2 = principal_stresses(sxx, syy, sxy)
    return (s1 - s2) / 2.0

def transform_stress(sxx, syy, sxy, theta):
    """
    Transform stress components to a plane rotated by theta (rad).

    Returns
    -------
    sxx_prime, syy_prime, sxy_prime
    """
    c = np.cos(theta)
    s = np.sin(theta)
    sxx_prime = (sxx + syy)/2 + (sxx - syy)/2 * c + sxy * s
    syy_prime = (sxx + syy)/2 - (sxx - syy)/2 * c - sxy * s
    sxy_prime = -(sxx - syy)/2 * s + sxy * c
    return sxx_prime, syy_prime, sxy_prime

def demo_stress():
    sxx, syy, sxy = 100e6, -50e6, 30e6
    center, radius, s1, s2, tau_max = mohr_circle(sxx, syy, sxy)
    print("\nStress state: sxx={} MPa, syy={} MPa, sxy={} MPa".format(sxx/1e6, syy/1e6, sxy/1e6))
    print("Mohr's circle centre={} MPa, radius={} MPa".format(center/1e6, radius/1e6))
    print("Principal stresses: s1={} MPa, s2={} MPa".format(s1/1e6, s2/1e6))
    print("Maximum shear (Tresca): {} MPa".format(tresca(sxx, syy, sxy)/1e6))
    print("Von Mises equivalent stress: {} MPa".format(von_mises(sxx, syy, sxy)/1e6))

    # Plot Mohr's circle
    theta = np.linspace(0, 2*np.pi, 200)
    plt.figure()
    xc = center / 1e6
    rc = radius / 1e6
    plt.plot(xc + rc*np.cos(theta), rc*np.sin(theta), label="Mohr's circle")
    plt.scatter([s1/1e6, s2/1e6], [0, 0], color='red', label='Principal stresses')
    plt.axhline(gest, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Normal stress (MPa)')
    plt.ylabel('Shear stress (MPa)')
    plt.title("Mohr's Circle")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==================== MECHANICAL VIBRATIONS ====================

class SDOF:
    """Single-degree-of-freedom spring-mass-damper system."""
    def __init__(self, m, k, c=0.0):
        self.m = m
        self.k = k
 chez
        self.c = c
        self.omega_n = np.sqrt(k / m)
        self.zeta = c / (2 * np.sqrt(k * m)) if c > 0 else 0.0

    def free_response(self, t, x0=0.0, v0=0.0):
        """
        Undriven response of the SDOF system.
        """
        if self.zeta == 0:
            # undamped
            omega = self.omega_n
            x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        else:
            # underdamped
            omega_d = self.omega_n * np.sqrt(1 - self.zeta**2)
            x = np.exp(-self.zeta * self.omega_n * t) * (
                x0 * np.cos(omega_d * t) +
                (v0 + self.zeta * self.omega_n * x0) / omega_d * np.sin(omega_d * t)
            )
        return x

    def forced_response(self, t, F0, omega_f, x0=0.0, v0=0.0):
        """
        Harmonic forced response at frequency omega_f.
        Force F(t) = F0 * sin(omega_f * t).
        """
        # Transient part (free vibration with initial conditions)
        x_trans = self.free_response(t, x0, v0)

        # Steady‑state part
        if self.zeta == 0:
            H = 1 / (self.k - self.m * omega_f**2)
        else:
            H = 1 / (self.k - self.m * omega_f**2 + 1j * self.c * omega_f)

        A = F0 * np.abs(H)
        phi = np.angle(H)
        x_ss = A * np.sin(omega_f * t + phi)

        return x_trans + x_ss

    def resonance_frequency(self):
        """
        Return the frequency (Hz) at which the steady‑state amplitude peaks
        (valid for underdamped systems).
        """
        if self.zeta < np.sqrt(2)/2:
            f_res = self.omega_n / (2*np.pi) * np.sqrt(1 -  Issue 2*self.zeta**2)
        else:
            f_res = self.omega_n / (2*np.pi)
        return f_res

    def __str__(self):
        return (f"SDOF(m={self.m}, k={self.k}, c={self.c}, ζ={self.zeta})")

class MDOF:
    """Multi-degree-of-freedom system with proportional (Rayleigh) damping."""
    def __init__(self, M, K, C=None, damping_type='Rayleigh'):
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.n = self.M.shape[0]

        if C is None:
            # Default Rayleigh coefficients
            alpha = 0.001
            beta = 1e-6
            self.C = alpha * self.M + beta * self.K
        else:
            self.C = np.asarray(C, dtype=float)

        # Undamped eigenvalues/eigenvectors
        self.eigvals, self.eigvecs = eigh(self.K, self.M)
        self.omega_n = np.sqrt(self.eigvals)
        # Modal damping ratios
        alpha = 0.001
        beta = 1e-6
        self.zeta = alpha/(2*self.omega_n) + beta*self.omega_n/2.0

    def modal_analysis(self):
        """Return natural frequencies (Hz) and mode shapes (columns)."""
        freqs = self.omega_n / (2*np.pi)
        return freqs, self.eigvecs

    def forced_response(self, t, F_vec, omega_f):
        """
        Steady‑state harmonic response to a harmonic load F_vec * sin(omega_f * t).
        Returns displacement vector over time (same for all time, steady state).
        """
        Z = self.K - (omega_f**2) * self.M + 1j * omega_f * self.C
        X = np.linalg.solve(Z, np.asarray(F_vec, dtype=complex))
        return np.real(X)

def demo_vibrations():
    # Example: 2‑DOF spring–mass system
    m1, m2 = 2.简介
    m2 = 1.5
    M = np.diag([m1, m2])
    k1, k2, k3 = 5000.0, 3000.0, 2000.0
    K = np.array([[k1 + k3, -k3],
                  [-k3, k2 + k3]])
    C = None   # Rayleigh damping inside MDOF

    sys = MDOF(M, K, C)
    freqs, modes = sys.modal_analysis()
    print("2‑DOF System Modal Analysis:")
    print("Natural frequencies (Hz):", freqs)
    print("Mode shapes (columns):")
    print(modes)

    # Plot mode shapes
    plt.figure()
    for i in range(sys.n):
        plt.plot([0, modes[0,i]], [0, modes[1,i]], '-o')
    plt.title("Mode shapes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    # Demonstrate SDOF forced response
    sdof = SDOF(m=1.0, k=1000.0, c=10.0)
    t = np.linspace(0, 5, 1000)
    F0 = 50.0
    f_force = 5.0
    omega_force = 2 * np.pi * f_force
    x = sdof.forced_response(t, F0, omega_force, x0=0.0, v0=0.0)

    plt.figure()
    plt.plot(t, x)
    plt.title("SDOF Forced Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)
    plt.show()

# ==================== FATIGUE & FRACTURE MECHANICS ====================

def sn_curve(N, sigma_f_prime, b):
    """
    Basquin S‑N curve: σ = σ_f' * N**b
    """
    return sigma_f_prime * (N ** b)

def goodman_corrected_stress(sigma_a, sigma_m, sigma_u, N, sigma_f_prime, b):
    """
    Goodman line correction for mean stress.
    Allowable alternating stress for given N and mean stress sigma_m.
    """
    return sigma_f_prime * (N ** b) * (1 - sigma_m / sigma_u)

def pariss_law(da, dK, C, m):
    """Paris' law: da/dN = C * (ΔK)**m."""
    return C * (dK ** m)

def fatigue_life(a0, a_crit, sigma_a, Y, C, m, tol=1e-8):
    """
    Integrate Paris' law to estimate cycles to grow a crack from a0 to a_crit.

    ΔK = Y * sigma_a * sqrt(pi * a)   (tension‑tension loading assumed).
    """
    def integrand(a):
        dK = Y * sigma_a * np.sqrt(np.pi * a)
        return 1.0 / (C * (dK ** m))
    N, _ = quad(integrand, a0, a_crit, epsabs=tol, epsrel=tol)
    return N

def demo_fatigue():
    # Material defaults (steel)
    sigma_f_prime = 800e6      # Pa
    b = -0.09
    sigma_u = 500e6            # Pa

    # Goodman diagram example
    sigma_m_vals = np.linspace(-sigma_u, sigma_u, 200)
    N_plot = 1e5
    sigma_a_allow = sigma_f_prime * (N_plot ** b) * (1 - sigma_m_vals / sigma_u)

    plt.figure()
    plt.plot(sigma_m_vals/1e6, sigma_a_allow/1e6, 'b-', label='Goodman line (N={})'.format(N_plot))
    plt.fill_between(sigma_m_vals/1e6, 0, sigma_a_allow/1e6, color='blue', alpha=0.2)
    plt.xlabel('Mean stress σ_m (MPa)')
    plt.ylabel('Allowable alternating stress σ_a (MPa)')
    plt.title('Goodman Diagram')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Paris law example
    C = 3.0e-11
    m = 3.0
    Y = 1.12
    sigma_a = 300e6
    a0 = 1e-3
    a_crit = 5e-3
    N = fatigue_life(a0, a_crit, sigma_a, Y, C, m)
    print("\nFatigue life (Paris’ law): {} cycles".format(N))

# ==================== STRUCTURAL OPTIMIZATION ====================

def beam_weight(b, h, L, rho):
    """Weight of a cantilever beam: W = ρ·L·b·h."""
    return rho * L * b * h

def beam_stress(b, h, L, P):
    """Maximum normal stress (MPa) in a cantilever with tip load P."""
    I = b * h**3 / 12.0
    sigma = 6 * P * L / (b * h**2)   # Pa
    return sigma

def beam_deflection(b, h, L, P, E):
    """Maximum tip deflection (m) of a cantilever."""
    I = b * h**3 / 12.0
    delta = P * L**3 / (3 * E * I)
    return delta

def optimize_beam_cross_section(L=2.0, P=10000.0, rho=7850.0,
                               E=210e9, nu=0.3,
                               sigma_allow=250e6, delta_allow=2e-3):
    """
    Minimize beam weight subject to stress and deflection constraints.
    Design variables: width b, height h.
    """
    # Objective
    def objective(vars):
        b, h = vars
        return beam_weight(b, h, L, rho)

    # Constraints (must be non‑negative)
    cons = [
        {'type': 'ineq',
         'fun': lambda vars: sigma_allow - beam_stress(vars[0], vars[1], L, P)},
        {'type': 'ineq',
         'fun': lambda vars: delta_allow - beam_deflection(vars[0], vars[1], L, P, E)}
    ]

    # Bounds (keep dimensions positive)
    bounds = [(0.001, 0.5), (0.001, 0.5)]

    # Initial guess
    x0 = [0.05, 0.10]

    # Gradient‑based optimizer (SLSQP)
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 1000})
    print("\n--- Structural Optimization (SLSQP) ---")
    print("Optimal b = {:.4f} m, h = {:.4f} m".format(res.x[0], res.x[1]))
    print("Minimum weight = {:.2f} kg".format(res.fun))
    print("Success:", res.success)

    # Differential Evolution (global search)
    res_de = differential_evolution(objective, bounds=bounds,
                                    constraints=cons,
                                    maxiter=1000,
                                    polish=True)
    print("\n--- Structural Optimization (Differential Evolution) ---")
    print("Optimal b = {:.4f} m, h = {:.4f} m".format(res_de.x[0], res_de.x[1]))
    print("Minimum weight = {:.2f} kg".format(res_de.fun))

# ==================== ROTORDYNAMICS ====================

class JeffcottRotor:
    """
    Simple Jeffcott rotor model (rigid disk on flexible shaft).
    Natural frequency (critical speed) ω_c = sqrt(k/m).
    Damping can be added.
    """
    def __init__(self, m, k, c=0.0, Omega=0.0, J=0.0):
        self.m = m
        self.k = k
        self.c = c
        self.Omega = Omega
        self.J = J
        self.omega_n = np.sqrt(k / m)
        self.zeta = c / (2 * np.sqrt(k * m)) if c > 0 else 0.0

    def critical_speed(self):
        """Return critical speed in rad/s and Hz."""
        return self.omega_n, self.omega_n / (2*np.pi)

    def whirling_response(self, t, x0=0.0, v0=0.0):
        """Free vibration of the rotor (whirling) – same as SDOF."""
        if self.zeta == 0:
            x = x0 * np.cos(self.omega_n * t) + (v0 / self.omega_n) * np.sin(self.omega_n * t)
        else:
            omega_d = self.omega_n * np.sqrt(1 - self.zeta**2)
            x = np.exp(-self.zeta * self.omega_n * t) * (
                x0 * np.cos(omega_d * t) +
                (v0 + self.zeta * self.omega_n * x0) / omega_d * np.sin(omega_d * t)
            )
        return x

def demo_rotordynamics():
    # Typical values for a small industrial rotor
    m = 5.0          # kg
    k = 1.0e6        # N/m
    c = 10.0         # N·s/m
    rotor = JeffcottRotor(m, k, c)
    wc, fc = rotor.critical_speed()
    print("\n--- Rotordynamics ---")
    print("Mass m = {} kg, Stiffness k = {} N/m".format(m, k))
    print("Critical speed ω_c = {:.2f} rad/s (≈ {:.2f} Hz)".format(wc, fc))
    print("Damping ratio ζ = {:.4f}".format(rotor.zeta))

    # Plot whirling response
    t = np.linspace(0, 0.1, 1000)
    x = rotor.whirling_response(t, x0=0.001, v0=0.0)
    plt.figure()
    plt.plot(t, x*1e3)   # convert to mm
    plt.title("Rotor Whirling (Free Vibration)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacements (mm)")
    plt.grid(True)
    plt.show()

# ==================== CONTACT MECHANICS ====================

def hertzian_contact(F, R1, R2, E1, nu1, E2, nu2):
    """
    Hertzian contact between two spheres.

    Returns
    -------
    a : float
        Contact radius.
    p0 : float
        Maximum contact pressure.
    delta : float
        Approach (indentation) at the center.
    """
    # Effective modulus
    inv_E = (1 - nu1**2) / E1 + (1 - nu2**2) / E2
    E_star = 1.0 / inv_E
    # Effective radius
    R_star = (R1 * R2) / (R1 + R2)
    a = np.sqrt((3 * F * R_star) / (4 * E_star))
    p0 = 3 * F / (2 * np.pi * a**2)
    delta = a**2 / R_star
    return a, p0, delta

def demo_contact():
    # Steel sphere (R1=50mm) on Aluminum sphere (R2=40mm)
    F = 1000.0                 # N
    R1 = 0.05                  # m
    R2 = 0.04
    E1, nu1 = 210e9, 0.3       # steel
    E2, nu2 = 70e9, 0.33       # aluminum

    a, p0, delta = hertzian_contact(F, R1, R2, E1, nu1, E2, nu2)
    print("\n--- Contact Mechanics (Hertzian) ---")
    print("Load F = {} N".format(F))
    print("Contact radius a = {:.4e} m".format(a))
    print("Maximum pressure p0 = {:.2e} Pa".format(p0))
    print("Indentation δ = {:.4e} m".format(delta))

# ==================== FAILURE THEORIES ====================

def max_principal_stress(s1, s2):
    """Maximum principal stress theory (Rankine)."""
    return max(s1, s2)

def max_shear_stress(s1, s2):
    """Maximum shear stress theory (Tresca)."""
    return (s1 - s2) / 2.0

def von_mises_stress(sxx, syy, sxy):
    """Distortion‑energy theory (von Mises)."""
    return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

def distortion_energy_stress(s1, s2):
    """
    Equivalent stress from distortion energy (same as von Mises for plane stress).
    """
    return np.sqrt(((s1 - s2)/2)**2 * 3)

def demo_failure_theories():
    sxx, syy, sxy = 150e6, -50e6, 20e6
    s1, s2 = principal_stresses(sxx, syy, sxy)
    sigma_y = 250e6   # material yield strength

    print("\n--- Failure Theories (Stress State) ---")
    print("σ_xx={} MPa, σ_yy={} MPa, τ_xy={} MPa".format(sxx/1e6, syy/1e6, sxy/1e6))
    print("Principal stresses: s1={} MPa, s2={} MPa".format(s1/1e6, s2/1e6))

    f_max_principal = max_principal_stress(s1, s2) / sigma_y
    f_max_shear = max_shear_stress(s1, s2) / sigma_y
    f_vonmises = von_mises_stress(sxx, syy, sxy) / sigma_y

    print("Safety factor (Max Principal): {:.2f}".format(f_max_principal))
    print("Safety factor (Max Shear): {:.2f}".format(f_max_shear))
    print("Safety factor (Von Mises): {:.2f}".format(f_vonmises))

    if f_max_principal < 1.0 or f_max_shear < 1.0 or f_vonmises <  niezwy:
        print("=> Failure predicted by one or more theories.")
    else:
        print("=> All theories indicate safe design (SF > 1).")

# ==================== MAIN DEMONSTRATION ====================

def main():
    demo_fea()
    demo_stress()
    demo_vibrations()
    demo_fatigue()
    optimize_beam_cross_section()
    demo_rotordynamics()
    demo_contact()
    demo_failure_theories()

if __name__ == '__main__':
    main()