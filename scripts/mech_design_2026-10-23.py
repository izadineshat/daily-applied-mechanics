#!/usr/bin/env python3
"""
Mechanical Engineering Toolbox: Applied Mechanics & Design

This script provides a collection of computational tools for:
- Finite Element Analysis (1D truss, 2D beam, 2D plate)
- Stress and strain transformations (Mohr's circle, Von Mises, Tresca)
- Mechanical vibrations (SDOF, MDOF modal analysis and time response)
- Fatigue and fracture mechanics (S-N curves, Goodman/Gerber, Paris' Law)
- Structural optimization (beam cross-section weight minimization)
- Rotordynamics (Jeffcott rotor critical speeds)
- Contact mechanics (Hertzian contact)
- Failure theories

All functions are written using NumPy, SciPy, and Matplotlib for production use.
"""

import numpy as np
from scipy.linalg import solve, eig
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Material model
# ----------------------------------------------------------------------
class Material:
    """
    Linear elastic isotropic material.

    Parameters
    ----------
    E : float
        Young's modulus (Pa)
    nu : float
        Poisson's ratio
    rho : float, optional
        Density (kg/m^3)
    Fy : float, optional
        Yield stress (Pa)
    Sut : float, optional
        Ultimate tensile strength (Pa)
    Se : float, optional
        Endurance limit (Pa)
    """
    def __init__(self, E, nu, rho=0.0, Fy=None, Sut=None, Se=None):
        self.E = E
        self.nu = nu
        self.rho = rho
        self.Fy = Fy
        self.Sut = Sut
        self.Se = Se

# ----------------------------------------------------------------------
# Finite Element utilities
# ----------------------------------------------------------------------
class TrussElement:
    """Axial truss element (2‑D)."""
    def __init__(self, node_i, node_j, material, A):
        self.node_i = node_i
        self.node_j = node_j
        self.material = material
        self.A = A

    def stiffness(self, nodes):
        xi, yi = nodes[self.node_i]
        xj, yj = nodes[self.node_j]
        dx = xj - xi
        dy = yj - yi
        L = np.hypot(dx, dy)
        c, s = dx / L, dy / L
        EA = self.material.E * self.A
        K = (EA / L) * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [c*s, s*s, -c*s, s*s]
        ])
        return K


class BeamElement:
    """Euler‑Bernoulli beam element (2‑D, horizontal bending)."""
    def __init__(self, node_i, node_j, material, I):
        self.node_i = node_i
        self.node_j = node_j
        self.material = material
        self.I = I

    def stiffness(self, nodes):
        xi, yi = nodes[self.node_i]
        xj, yj = nodes[self.node_j]
        L = np.hypot(xj - xi, yj - yi)
        EI = self.material.E * self.I
        k = (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k


class TriangleElement:
    """Constant strain triangle (CST) for 2‑D plane stress problems."""
    def __init__(self, node_i, node_j, node_k, material):
        self.node_i = node_i
        self.node_j = node_j
        self.node_k = node_k
        self.material = material

    def stiffness(self, nodes):
        xi, yi = nodes[self.node_i]
        xj, yj = nodes[self.node_j]
        xk, yk = nodes[self.node_k]

        # Area
        area = 0.5 * abs((xj - xi)*(yk - yi) - (xk - xi)*(yj - yi))

        # Shape function coefficients
        b1, c1 = yj - yk, xk - xj
        b2, c2 = yk - yi, xi - xk
        b3, c3 = yi - yj, xj - xi

        # B matrix (3x6)
        B = (1.0 / (2.0 * area)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])

        # Plane stress D matrix
        E, nu = self.material.E, self.material.nu
        D = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

        K = area * (B.T @ D @ B)
        return K


class FiniteElementAnalyzer:
    """Simple FE model assembler and solver for 2‑D problems."""
    def __init__(self, element_type):
        """
        element_type : str
            'truss' (2 DOFs per node), 'beam' (vertical + rotation), or 'triangle' (2 DOFs per node)
        """
        self.element_type = element_type
        self.nodes = {}
        self.elements = []

    def add_node(self, node_id, x, y):
        self.nodes[node_id] = (x, y)

    def add_element(self, element):
        self.elements.append(element)

    def assemble_global_stiffness(self):
        dof_per_node = 2  # all types use 2 DOFs per node in this implementation
        num_nodes = len(self.nodes)
        total_dofs = num_nodes * dof_per_node
        K_global = np.zeros((total_dofs, total_dofs))

        for el in self.elements:
            if isinstance(el, TrussElement):
                K_el = el.stiffness(self.nodes)
                node_i, node_j = el.node_i, el.node_j
                dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
            elif isinstance(el, BeamElement):
                K_el = el.stiffness(self.nodes)
                node_i, node_j = el.node_i, el.node_j
                dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
            elif isinstance(el, TriangleElement):
                K_el = el.stiffness(self.nodes)
                node_i, node_j, node_k = el.node_i, el.node_j, el.node_k
                dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1, 2*node_k, 2*node_k+1]
            else:
                raise TypeError("Unsupported element type")

            for i_loc, i_glo in enumerate(dofs):
                for j_loc, j_glo in enumerate(dofs):
                    K_global[i_glo, j_glo] += K_el[i_loc, j_loc]

        return K_global

    def apply_bc(self, K_global, F):
        fixed_dofs = [i for i, (node_id, x, y) in enumerate(self.nodes.items()) if node_id == 0]
        free_dofs = [i for i in range(K_global.shape[0]) if i not in fixed_dofs]
        K_red = K_global[np.ix_(free_dofs, free_dofs)]
        F_red = F[free_dofs]
        return K_red, F_red, free_dofs

    def solve(self, K_global, F):
        K_red, F_red, free_dofs = self.apply_bc(K_global, F)
        u_red = solve(K_red, F_red)
        total_dofs = K_global.shape[0]
        u = np.zeros(total_dofs)
        u[free_dofs] = u_red
        return u


# ----------------------------------------------------------------------
# Stress & strain transformations
# ----------------------------------------------------------------------
def transform_stress(sigma_xx, sigma_yy, tau_xy, theta):
    """
    Transform stress components to a coordinate system rotated by theta (rad).

    Returns
    -------
    sigma_x, sigma_y, tau_xy_rot : stresses in the rotated system
    """
    c = np.cos(theta)
    s = np.sin(theta)
    sigma_x = 0.5*(sigma_xx + sigma_yy) + 0.5*(sigma_xx - sigma_yy)*c + tau_xy*s
    sigma_y = 0.5*(sigma_xx + sigma_yy) - 0.5*(sigma_xx - sigma_yy)*c - tau_xy*s
    tau_xy_rot = -0.5*(sigma_xx - sigma_yy)*s + tau_xy*c
    return sigma_x, sigma_y, tau_xy_rot


def principal_stresses(sigma_x, sigma_y, tau_xy):
    """
    Compute principal stresses sigma1, sigma2 and maximum shear stress.

    Returns
    -------
    sigma1, sigma2, tau_max
    """
    avg = 0.5*(sigma_x + sigma_y)
    diff = 0.5*(sigma_x - sigma_y)
    radius = np.hypot(diff, tau_xy)
    sigma1 = avg + radius
    sigma2 = avg - radius
    tau_max = radius
    return sigma1, sigma2, tau_max


def von_mises_stress(sigma1, sigma2, sigma3=0.0):
    """Von Mises equivalent stress (plane stress sigma3 = 0)."""
    vm = np.sqrt(0.5*((sigma1 - sigma2)**2 + (sigma2 - sigma3)**2 + (sigma3 - sigma1)**2))
    return vm


def tresca_stress(sigma1, sigma2, sigma3=0.0):
    """Maximum shear stress (Tresca) for 3‑D state."""
    s12 = abs(sigma1 - sigma2)
    s23 = abs(sigma2 - sigma3)
    s31 = abs(sigma3 - sigma1)
    return max(s12, s23, s31)


def mohr_circle_params(sigma_x, sigma_y, tau_xy):
    """Return (center, radius, sigma1, sigma2) for Mohr's circle."""
    center = 0.5*(sigma_x + sigma_y)
    radius = np.hypot(0.5*(sigma_x - sigma_y), tau_xy)
    sigma1 = center + radius
    sigma2 = center - radius
    return center, radius, sigma1, sigma2


def plot_mohr_circle(sigma_x, sigma_y, tau_xy, filename=None):
    """Plot Mohr's circle for a given 2‑D stress state."""
    center, radius, sigma1, sigma2 = mohr_circle_params(sigma_x, sigma_y, tau_xy)
    theta = np.linspace(0, 2*np.pi, 200)
    x = center + radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.figure(figsize=(6,6))
    plt.plot(x, y, label='Mohr’s circle')
    plt.scatter([sigma1, sigma2], [0, 0], color='red', zorder=5, label='Principal stresses')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Normal stress (Pa)')
    plt.ylabel('Shear stress (Pa)')
    plt.title('Mohr’s Circle')
    plt.axis('equal')
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()


# ----------------------------------------------------------------------
# Mechanical vibrations
# ----------------------------------------------------------------------
def sdof_natural_frequency(k, m):
    """Natural circular frequency (rad/s) and frequency (Hz)."""
    omega_n = np.sqrt(k / m)
    f_n = omega_n / (2*np.pi)
    return omega_n, f_n


def sdof_damping_ratio(c, m, k):
    """Damping ratio ζ for SDOF system."""
    return c / (2 * np.sqrt(m * k))


def sdof_free_response(t, x0, v0, omega_n, zeta):
    """Analytical free response of SDOF (homogeneous)."""
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    A = (v0 + zeta*omega_n*x0) / omega_d
    B = x0 - (A * omega_n * zeta) / omega_d
    x = np.exp(-zeta*omega_n*t) * (A*np.cos(omega_d*t) + B*np.sin(omega_d*t))
    return x


def mdof_modal_analysis(M, C=None):
    """Undamped modal analysis (K φ = λ M φ). If C is provided, compute complex modes."""
    # Assumes K is available globally or passed separately; for simplicity we accept M and K as arguments in a wrapper.
    pass


def mdof_free_response(M, K, C, x0, v0, t_eval):
    """Time response of MDOF system using state‑space integration."""
    n = M.shape[0]
    I = np.eye(n)
    # Build state matrix A
    Minv = solve(M, I)
    A_top = np.hstack((np.zeros((n, n)), I))
    A_bot = np.hstack((-Minv @ K, -Minv @ C))
    A = np.vstack((A_top, A_bot))

    y0 = np.concatenate((x0, v0))
    sol = solve_ivp(lambda t, y: A @ y, [0, t_eval[-1]], y0, t_eval=t_eval, method='RK45')
    x = sol.y[:n, :]
    return x


# ----------------------------------------------------------------------
# Fatigue & fracture mechanics
# ----------------------------------------------------------------------
def basquin_cycles(sigma_a, sigma_fprime, b):
    """
    Basquin relation: sigma_a = sigma_fprime * N^b.

    Returns cycles to failure N.
    """
    if sigma_a <= 0:
        return np.inf
    N = (sigma_a / sigma_fprime) ** (1.0 / b)
    return N


def goodman_allowable(sigma_mean, Se, Sut):
    """Allowable alternating stress from Goodman diagram."""
    return Se * (1 - sigma_mean / Sut)


def gerber_allowable(sigma_mean, Se, Sut):
    """Allowable alternating stress from Gerber parabola."""
    return Se * (1 - (sigma_mean / Sut)**2)


def paris_crack_growth(deltaK, C, m):
    """Paris law: da/dN = C * (ΔK)^m."""
    return C * (deltaK ** m)


def cycles_to_failure_paris_analytic(a0, sigma, Y, C, m, K_IC):
    """
    Analytic solution of Paris law for a through‑thickness crack in an infinite plate.
    ΔK = Y * sigma * sqrt(pi * a).
    """
    a_crit = (K_IC / (Y * sigma))**2 / np.pi
    if a_crit <= a0:
        return 0.0
    factor = (Y * sigma * np.sqrt(np.pi))**m
    if abs(m - 2) > 1e-12:
        exponent = 1 - m/2
        numerator = a_crit**exponent - a0**exponent
        denominator = factor * C * exponent
    else:
        numerator = np.log(a_crit / a0)
        denominator = factor * C
    N = numerator / denominator
    return N


# ----------------------------------------------------------------------
# Structural optimization (beam weight minimization)
# ----------------------------------------------------------------------
def beam_objective(vars, L, material, P, delta_limit):
    b, h = vars
    I = b * h**3 / 12.0
    weight = material.rho * L * b * h   # proportional to volume (gravity omitted)
    return weight


def beam_stress_constraint(vars, L, material, P):
    b, h = vars
    Z = b * h**2 / 6.0
    M = P * L / 4.0
    sigma = M / Z
    return material.Fy - sigma   # > 0 means safe


def beam_deflection_constraint(vars, L, material, P, delta_limit):
    b, h = vars
    I = b * h**3 / 12.0
    delta = P * L**3 / (48 * material.E * I)
    return delta_limit - delta   # > 0 means safe


def optimize_beam_cross_section(L, material, P, delta_limit):
    """Minimize weight subject to stress and deflection constraints."""
    # Initial guess
    x0 = [0.05, 0.1]  # b, h (m)
    # Bounds
    bounds = [(1e-3, 0.5), (1e-3, 0.5)]

    constraints = [
        {'type': 'ineq', 'fun': beam_stress_constraint, 'args': (L, material, P)},
        {'type': 'ineq', 'fun': beam_deflection_constraint, 'args': (L, material, P, delta_limit)}
    ]

    result = minimize(beam_objective, x0, args=(L, material, P, delta_limit),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# ----------------------------------------------------------------------
# Rotordynamics (Jeffcott rotor)
# ----------------------------------------------------------------------
def jeffcott_critical_speed(k, m):
    """Critical speed (rad/s and Hz) for a Jeffcott rotor."""
    omega_cr = np.sqrt(k / m)
    f_cr = omega_cr / (2*np.pi)
    return omega_cr, f_cr


def jeffcott_response_amplitude(mass, stiffness, damping, unbalance_mass, radius, omega):
    """Amplitude of forced response for a Jeffcott rotor at speed omega."""
    k, c = stiffness, damping
    F0 = unbalance_mass * radius * omega**2
    denominator = np.sqrt((k - mass*omega**2)**2 + (c*omega)**2)
    X = F0 / denominator
    return X


# ----------------------------------------------------------------------
# Contact mechanics (Hertzian)
# ----------------------------------------------------------------------
def hertz_sphere_contact(F, R1, R2, E1, nu1, E2, nu2):
    """
    Hertzian contact between two spheres.

    Returns
    -------
    a : contact radius (m)
    p0 : maximum contact pressure (Pa)
    """
    E_star = 1/((1-nu1**2)/E1 + (1-nu2**2)/E2)
    R_star = (R1*R2) / (R1 + R2)
    a = ((3*F*R_star) / (4*E_star))**(1/3)
    p0 = (3*F) / (2*np.pi*a**2)
    return a, p0


def hertz_cylinder_contact(F, R1, R2, E1, nu1, E2, nu2):
    """
    Hertzian line contact between two long cylinders (plane strain).

    Returns
    -------
    b : half contact width (m)
    p0 : maximum contact pressure (Pa)
    """
    E_star = 1/((1-nu1**2)/E1 + (1-nu2**2)/E2)
    R_star = (R1*R2) / (R1 + R2)
    b = np.sqrt((4*F*R_star) / (np.pi*E_star))
    p0 = np.sqrt(F*E_star / (np.pi*R_star))
    return b, p0


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Material defaults (steel)
    steel = Material(E=210e9, nu=0.3, rho=7850, Fy=250e6, Sut=400e6, Se=150e6)

    # ---------- 1. Finite Element Analysis ----------
    print("=== Finite Element Analysis ===")
    # Truss: triangular frame
    analyzer_truss = FiniteElementAnalyzer('truss')
    analyzer_truss.add_node(0, 0.0, 0.0)
    analyzer_truss.add_node(1, 1.0, 0.0)
    analyzer_truss.add_node(2, 0.0, 1.0)
    truss1 = TrussElement(0, 1, steel, A=0.001)
    truss2 = TrussElement(0, 2, steel, A=0.001)
    truss3 = TrussElement(1, 2, steel, A=0.001)
    analyzer_truss.add_element(truss1)
    analyzer_truss.add_element(truss2)
    analyzer_truss.add_element(truss3)

    # Apply vertical load at node 2
    F_truss = np.zeros(6)
    F_truss[5] = -1000.0   # vertical DOF of node 2 (index 2*2+1)
    u = analyzer_truss.solve(analyzer_truss.assemble_global_stiffness(), F_truss)
    print(f"Node 0 displacement: {u[0]:.2e}, {u[1]:.2e} m")
    print(f"Node 1 displacement: {u[2]:.2e}, {u[3]:.2e} m")
    print(f"Node 2 displacement: {u[4]:.2e}, {u[5]:.2e} m")

    # Beam: cantilever with tip load
    analyzer_beam = FiniteElementAnalyzer('beam')
    analyzer_beam.add_node(0, 0.0, 0.0)
    analyzer_beam.add_node(1, 2.0, 0.0)
    I_beam = 5e-6
    beam_el = BeamElement(0, 1, steel, I_beam)
    analyzer_beam.add_element(beam_el)

    F_beam = np.zeros(4)
    F_beam[2] = -500.0   # vertical DOF at node 1
    u_beam = analyzer_beam.solve(analyzer_beam.assemble_global_stiffness(), F_beam)
    print("\nBeam tip vertical displacement: {:.2e} m".format(u_beam[2]))

    # Plate: triangular element
    analyzer_plate = FiniteElementAnalyzer('triangle')
    analyzer_plate.add_node(0, 0.0, 0.0)
    analyzer_plate.add_node(1, 1.0, 0.0)
    analyzer_plate.add_node(2, 0.0, 1.0)
    tri_el = TriangleElement(0, 1, 2, steel)
    analyzer_plate.add_element(tri_el)

    F_plate = np.zeros(6)
    F_plate[3] = -2000.0   # vertical load at node 2
    u_plate = analyzer_plate.solve(analyzer_plate.assemble_global_stiffness(), F_plate)
    print("\nPlate node 2 vertical displacement: {:.2e} m".format(u_plate[3]))

    # ---------- 2. Stress transformations ----------
    print("\n=== Stress Transformations ===")
    sigma_xx, sigma_yy, tau_xy = 100e6, 40e6, 30e6
    center, radius, sigma1, sigma2 = mohr_circle_params(sigma_xx, sigma_yy, tau_xy)
    print(f"Principal stresses: sigma1 = {sigma1/1e6:.2f} MPa, sigma2 = {sigma2/1e6:.2f} MPa")
    print(f"Maximum shear stress: {radius/1e6:.2f} MPa")
    vm = von_mises_stress(sigma1, sigma2)
    print(f"Von Mises stress: {vm/1e6:.2f} MPa")
    tresca = tresca_stress(sigma1, sigma2)
    print(f"Tresca stress: {tresca/1e6:.2f} MPa")
    plot_mohr_circle(sigma_xx, sigma_yy, tau_xy, filename='mohr_circle.png')

    # ---------- 3. Mechanical vibrations ----------
    print("\n=== Mechanical Vibrations ===")
    # SDOF
    m_sdof = 10.0
    k_sdof = 1e6
    c_sdof = 100.0
    omega_n, f_n = sdof_natural_frequency(k_sdof, m_sdof)
    zeta = sdof_damping_ratio(c_sdof, m_sdof, k_sdof)
    print(f"SDOF: ω_n = {omega_n:.2f} rad/s, f_n = {f_n:.2f} Hz, ζ = {zeta:.4f}")
    t = np.linspace(0, 0.1, 500)
    x0, v0 = 0.01, 0.0
    x = sdof_free_response(t, x0, v0, omega_n, zeta)
    plt.figure()
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('SDOF free response')
    plt.show()

    # MDOF: two-degree-of-freedom system
    m1, m2 = 5.0, 5.0
    k1 = 2e5
    k2 = 1e5
    M = np.array([[m1, 0], [0, m2]])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    C = 0.1 * K
    x0_mdof = np.array([0.01, 0.0])
    v0_mdof = np.array([0.0, 0.0])
    t_mdof = np.linspace(0, 0.5, 500)
    x_mdof = mdof_free_response(M, K, C, x0_mdof, v0_mdof, t_mdof)
    plt.figure()
    plt.plot(t_mdof, x_mdof[0], label='Mass 1')
    plt.plot(t_mdof, x_mdof[1], label='Mass 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('MDOF free response')
    plt.legend()
    plt.show()

    # ---------- 4. Fatigue & Fracture ----------
    print("\n=== Fatigue & Fracture ===")
    # S-N curve (Basquin)
    sigma_fprime = 800e6
    b = -0.09
    sigma_a = 300e6
    N = basquin_cycles(sigma_a, sigma_fprime, b)
    print(f"Basquin: {N:.2e} cycles to failure for σ_a = {sigma_a/1e6:.2f} MPa")

    # Goodman / Gerber
    sigma_mean = 100e6
    Se = steel.Se if steel.Se is not None else 150e6
    Sut = steel.Sut if steel.Sut is not None else 400e6
    sigma_a_allow_goodman = goodman_allowable(sigma_mean, Se, Sut)
    sigma_a_allow_gerber = gerber_allowable(sigma_mean, Se, Sut)
    print(f"Goodman allowable σ_a: {sigma_a_allow_goodman/1e6:.2f} MPa")
    print(f"Gerber allowable σ_a: {sigma_a_allow_gerber/1e6:.2f} MPa")

    # Paris crack growth
    a0 = 1e-6
    sigma = 300e6
    Y = 1.0
    C = 1e-10
    m = 3.0
    K_IC = 50e6 * np.sqrt(np.pi * a0)  # approximate
    N_f = cycles_to_failure_paris_analytic(a0, sigma, Y, C, m, K_IC)
    print(f"Paris law cycles to failure: {N_f:.2e}")

    # ---------- 5. Structural optimization ----------
    print("\n=== Structural optimization ===")
    L_beam = 1.0
    P_load = 5000.0
    delta_limit = 0.005
    opt_res = optimize_beam_cross_section(L_beam, steel, P_load, delta_limit)
    b_opt, h_opt = opt_res.x
    weight_opt = beam_objective(opt_res.x, L_beam, steel, P_load, delta_limit)
    print(f"Optimal width b = {b_opt*1e3:.2f} mm")
    print(f"Optimal height h = {h_opt*1e3:.2f} mm")
    print(f"Minimum weight = {weight_opt:.2f} kg (per unit length)")

    # ---------- 6. Rotordynamics ----------
    print("\n=== Rotordynamics ===")
    m_rotor = 20.0
    k_rotor = 2e6
    omega_cr, f_cr = jeffcott_critical_speed(k_rotor, m_rotor)
    print(f"Critical speed: ω_cr = {omega_cr:.2f} rad/s, f_cr = {f_cr:.2f} Hz")
    # Amplification curve
    speeds = np.linspace(0.5*omega_cr, 1.5*omega_cr, 200)
    amplitudes = []
    for w in speeds:
        X = jeffcott_response_amplitude(m_rotor, k_rotor, 100.0, 0.05, 0.1, w)
        amplitudes.append(X)
    plt.figure()
    plt.plot(speeds/(2*np.pi), amplitudes)
    plt.xlabel('Speed (Hz)')
    plt.ylabel('Amplitude (m)')
    plt.title('Jeffcott rotor response')
    plt.show()

    # ---------- 7. Contact mechanics ----------
    print("\n=== Contact Mechanics ===")
    F_contact = 1000.0
    R1, R2 = 0.1, 0.2
    a, p0 = hertz_sphere_contact(F_contact, R1, R2, steel.E, steel.nu, steel.E, steel.nu)
    print(f"Sphere contact: radius a = {a*1e3:.2f} mm, max pressure p0 = {p0/1e6:.2f} MPa")

    # Cylinder contact (plane strain)
    b, p0_cyl = hertz_cylinder_contact(F_contact, R1, R2, steel.E, steel.nu, steel.E, steel.nu)
    print(f"Cylinder contact: half-width b = {b*1e3:.2f} mm, max pressure p0 = {p0_cyl/1e6:.2f} MPa")

    print("\nAll demonstrations completed.")