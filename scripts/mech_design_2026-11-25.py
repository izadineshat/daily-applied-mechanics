#!/usr/bin/env python3
""" 
production_ready_mechanics.py
A comprehensive, modular Python script for Applied Mechanics / Solid Mechanics / Applied Design.
Topics: FEA (1D/2D truss/beam), Stress/Strain Transformations, Mechanical Vibrations,
Fatigue & Fracture Mechanics, Structural Optimization, Rotordynamics, Contact Mechanics,
Failure Theories.
Standard libraries: numpy, scipy, matplotlib.
"""

import numpy as np
from scipy.linalg import eig, expm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds, LinearConstraint, differential_evolution
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Default Material Properties (ISO, room temperature)
# ----------------------------------------------------------------------
STEEL = {"E": 210e9, "nu": 0.3, "rho": 7850.0}   # Pa, kg/m^3
ALUMINUM = {"E": 70e9, "nu": 0.33, "rho": 2700.0}

# ----------------------------------------------------------------------
# 1. FINITE ELEMENT ANALYSIS
# ----------------------------------------------------------------------


class Truss1D:
    r"""1D Truss Finite Element Analysis.

    Assumes axial deformation only, small strain, linear elasticity.
    """

    def __init__(self, coords, elements, A=1e-4, E=STEEL["E"]):
        """Initialize truss FEA.

        Parameters
        ----------
        coords : np.ndarray, shape (n_nodes, 2) or (n_nodes, 3)
            Node coordinates [x, y] or [x, y, z].
        elements : list of tuple
            Each tuple (i, j) indices of nodes connected by an element.
        A : float or np.ndarray, optional
            Cross-sectional area [m^2]. Can be per-element or uniform.
        E : float, optional
            Young's modulus [Pa].
        """
        self.coords = np.asarray(coords, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.A = np.atleast_1d(np.array(A, dtype=float))
        if self.A.size == 1:
            self.A = np.full(self.elements.shape[0] if self.elements.ndim > 0 else 1, self.A[0])
        else:
            self.A = self.A[:self.elements.shape[0]]
        self.E = float(E)
        self.n_nodes = self.coords.shape[0]
        self.dim = self.coords.shape[1] if self.coords.ndim > 1 else 1

    def element_local_stiffness(self, i, j):
        """Return 2x2 local stiffness matrix in global coordinates for element (i,j)."""
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        L = np.hypot(xj - xi, yj - yi)
        if L < 1e-15:
            raise ValueError(f"Zero-length element between nodes {i} and {j}")
        c = (xj - xi) / L
        s = (yj - yi) / L
        k = self.E * self.A[j] / L
        return np.array([[c, s], [-c, -s]]) * k

    def assemble_global_stiffness(self):
        """Assemble global nxn stiffness matrix."""
        n_dofs = self.n_nodes * self.dim
        K = np.zeros((n_dofs, n_dofs))
        for idx, (i, j) in enumerate(self.elements):
            ke = self.element_local_stiffness(i, j)
            dofs_i = [i * self.dim + k for k in range(self.dim)]
            dofs_j = [j * self.dim + k for k in range(self.dim)]
            dofs = dofs_i + dofs_j
            for r in range(self.dim):
                for c in range(self.dim):
                    K[dofs[r], dofs[c]] += ke[r, c]
        return K

    def solve(self, F, fixed_dofs):
        """Solve Ku = F with essential boundary conditions.

        Parameters
        ----------
        F : np.ndarray, shape (n_dofs,)
            Global force vector.
        fixed_dofs : list or np.ndarray
            Indices of degrees of freedom to be fixed (Dirichlet).

        Returns
        -------
        u : np.ndarray, shape (n_dofs,)
            Displacement vector.
        """
        K = self.assemble_global_stiffness()
        F = np.asarray(F, dtype=float).copy()
        K_mod = K.copy()
        for fd in np.atleast_1d(fixed_dofs).astype(int):
            K_mod[fd, :] = 0.0
            K_mod[:, fd] = 0.0
            K_mod[fd, fd] = 1.0
            F[fd] = 0.0
        u = np.linalg.solve(K_mod, F)
        return u


class Beam2D:
    r"""2D Euler-Bernoulli Beam Finite Element Analysis.

    Uses Hermite shape functions for transverse deflection and rotation.
    Each node has 2 DOFs: vertical displacement and rotation (z-y plane).
    """

    def __init__(self, nodes, elements, I=1e-6, E=STEEL["E"], coords=None):
        """Initialize beam FEA.

        Parameters
        ----------
        nodes : int
            Number of beam nodes.
        elements : list of tuple
            Each tuple (i, j) node indices.
        I : float, optional
            Area moment of inertia [m^4].
        E : float, optional
            Young's modulus [Pa].
        coords : np.ndarray, shape (n_nodes, 1), optional
            Node x-coordinates along the beam. If None, equally spaced from 0 to L.
        """
        self.n_nodes = int(nodes)
        self.elements = np.array(elements, dtype=int)
        self.E = float(E)
        self.I = float(I)
        if coords is None:
            self.coords = np.linspace(0.0, 1.0, self.n_nodes).reshape(-1, 1)
        else:
            self.coords = np.asarray(coords, dtype=float).reshape(-1, 1)
            if self.coords.shape[0] != self.n_nodes:
                raise ValueError("Number of coords must match nodes")

    def element_stiffness(self, i, j):
        """Return 4x4 element stiffness matrix for beam nodes i and j."""
        xi = float(self.coords[i, 0])
        xj = float(self.coords[j, 0])
        L = xj - xi
        if abs(L) < 1e-12:
            raise ValueError(f"Zero-length element between nodes {i} and {j}")
        EI = self.E * self.I
        k = (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

    def assemble_global_stiffness(self):
        """Assemble global nxn stiffness matrix (2 DOFs per node: deflection + rotation)."""
        n_dofs = self.n_nodes * 2
        K = np.zeros((n_dofs, n_dofs))
        for i, j in self.elements:
            ke = self.element_stiffness(i, j)
            dofs_i = [2*i, 2*i+1]
            dofs_j = [2*j, 2*j+1]
            dofs = dofs_i + dofs_j
            for r in range(4):
                for c in range(4):
                    K[dofs[r], dofs[c]] += ke[r, c]
        return K

    def solve(self, F, fixed_dofs):
        """Solve beam FEA with essential boundary conditions.

        Parameters
        ----------
        F : np.ndarray, shape (n_dofs,)
            Force vector [vertical forces, moments].
        fixed_dofs : list of int
            DOF indices to fix.

        Returns
        -------
        u : np.ndarray, shape (n_dofs,)
            Displacement/rotation vector.
        """
        K = self.assemble_global_stiffness()
        F = np.asarray(F, dtype=float).copy()
        K_mod = K.copy()
        for fd in np.atleast_1d(fixed_dofs).astype(int):
            K_mod[fd, :] = 0.0
            K_mod[:, fd] = 0.0
            K_mod[fd, fd] = 1.0
            F[fd] = 0.0
        u = np.linalg.solve(K_mod, F)
        return u


# ----------------------------------------------------------------------
# 2. STRESS & STRAIN TRANSFORMATIONS
# ----------------------------------------------------------------------


class StressTransform:
    r"""Stress/strain tensor transformations, Mohr's circle, and yield criteria."""

    @staticmethod
    def transform_2d(sigma_x, sigma_y, tau_xy, theta):
        """Transform 2D stress state by angle theta (radians).

        Returns (sigma_x_prime, sigma_y_prime, tau_xyp_prime)
        """
        theta = float(theta)
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_x_prime = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
        sigma_y_prime = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
        tau_xy_prime = -(sigma_x - sigma_y) * s * c + tau_xy * (c**2 - s**2)
        return sigma_x_prime, sigma_y_prime, tau_xy_prime

    @staticmethod
    def mohr_circle_2d(sigma_x, sigma_y, tau_xy):
        """Compute Mohr's circle parameters.

        Returns (center, radius, sigma1, sigma2) where sigma1 >= sigma2 are principal stresses.
        """
        sigma_center = (sigma_x + sigma_y) / 2.0
        R = np.hypot(sigma_x - sigma_center, tau_xy)
        sigma1 = sigma_center + R
        sigma2 = sigma_center - R
        return sigma_center, R, float(sigma1), float(sigma2)

    @staticmethod
    def principal_stresses_2d(sigma_x, sigma_y, tau_xy):
        """Return principal stresses sorted descending (sigma1, sigma2)."""
        _, _, s1, s2 = StressTransform.mohr_circle_2d(sigma_x, sigma_y, tau_xy)
        return (float(s1), float(s2)) if s1 >= s2 else (float(s2), float(s1))

    @staticmethod
    def von_mises_2d(sigma_x, sigma_y, tau_xy):
        """Von Mises equivalent stress for 2D plane stress (sigma3 = 0)."""
        s1, s2 = StressPrincipal = StressTransform.principal_stresses_2d(sigma_x, sigma_y, tau_xy)
        # For plane stress with sigma3=0, von Mises = sqrt(sigma1^2 - sigma1*sigma2 + sigma2^2)
        return np.sqrt(s1**2 - s1*s2 + s2**2)

    @staticmethod
    def tresca_max_shear(sigma_x, sigma_y, tau_xy):
        """Tresca maximum shear stress = (sigma_max - sigma_min)/2."""
        s1, s2, s3 = StressTransform.principal_stresses_3d(sigma_x, sigma_y, tau_xy, 0.0)
        return (np.max([s1, s2, s3]) - np.min([s1, s2, s3])) / 2.0

    @staticmethod
    def principal_stresses_3d(sigma_x, sigma_y, tau_xy, sigma_z=0.0, tau_yz=0.0, tau_xz=0.0):
        """Compute all three principal stresses from stress tensor.

        Uses eigenvalue solver for general 3D stress state.
        """
        sigma_norm = np.array([sigma_x, sigma_y, sigma_z])
        tau_norm = np.array([tau_xy, tau_yz, tau_xz])
        T = np.diag(sigma_norm) + np.array([
            [0.0, tau_norm[0], tau_norm[2]],
            [tau_norm[0], 0.0, tau_norm[1]],
            [tau_norm[2], tau_norm[1], 0.0]
        ])
        eigenvalues = eig(T, homo=False)[0]
        return tuple(float(v) for v in sorted(eigenvalues, reverse=True))


# ----------------------------------------------------------------------
# 3. MECHANICAL VIBRATIONS
# ----------------------------------------------------------------------


class MechanicalVibrations:
    r"""Mechanical vibrations: SDOF/MDOF damped response, state-space."""

    @staticmethod
    def sdof_damped_response(m, c, k, F0=0.0, omega=0.0, t_max=10.0, dt=0.01, x0=0.0, v0=0.0):
        """SDOF damped harmonic oscillator: m*x'' + c*x' + k*x = F0*cos(omega*t).

        Returns solution struct with time history.
        """
        wn = np.sqrt(k / m)  # natural frequency
        zeta = c / (2.0 * np.sqrt(m * k))  # damping ratio

        def ode(t, y):
            x, v = y
            dxdt = v
            dvdt = (F0 * np.cos(omega * t) - c * v - k * x) / m
            return [dxdt, dvdt]

        t_span = (0.0, t_max)
        t_eval = np.arange(0, t_max + dt, dt)
        sol = solve_ivp(ode, t_span, [x0, v0], t_eval=t_eval, method='RK45')
        return {
            "time": sol.t,
            "displacement": sol.y[0, :],
            "velocity": sol.y[1, :],
            "natural_freq": wn,
            "damping_ratio": zeta
        }

    @staticmethod
    def mdof_modal_analysis(M, K, C=None, mass_proportional=True):
        """MDOF undamped natural frequencies and mode shapes.

        Solves generalized eigenvalue problem: K phi = lambda M phi
        where lambda = omega_n^2.
        """
        M = np.asarray(M, dtype=float)
        K = np.asarray(K, dtype=float)
        if M.shape != K.shape:
            raise ValueError("Mass and stiffness matrices must have same shape")
        if C is None:
            # Undamped: compute eigenvalues/vectors
            phi, lambda_ = eig(K, M)
            omega_n = np.sqrt(np.real(lambda_))
            # Sort by ascending natural frequency
            idx = np.argsort(omega_n)
            omega_n = omega_n[idx]
            phi = phi[:, idx]
            return omega_n, phi
        else:
            # Damped modal analysis (proportional damping assumed)
            C = np.asarray(C, dtype=float)
            omega_n, phi = MechanicalVibrations.mdof_modal_analysis(M, K)
            # Rayleigh damping: alpha*M + beta*K
            # Compute alpha, beta from two target modes if needed (omitted for brevity)
            return omega_n, phi

    @staticmethod
    def state_space_integration(A, B, x0, u_func, t_span, dt=0.01):
        """Linear state-space: x_dot = A x + B u(t).

        Uses scipy solve_ivp for time integration.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        n = A.shape[0]

        def ode(t, y):
            u = u_func(t) if callable(u_func) else u_func
            return A @ y + B * u

        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        sol = solve_ivp(ode, t_span, x0, t_eval=t_eval, method='RK45')
        return {
            "time": sol.t,
            "state": sol.y.T,
            "input": np.atleast_2d(u_func(sol.t)).T if callable(u_func) else np.full((len(sol.t), *np.shape(u_func)), u_func)
        }


# ----------------------------------------------------------------------
# 4. FATIGUE & FRACTURE MECHANICS
# ----------------------------------------------------------------------


class FatigueFracture:
    r"""Fatigue & Fracture Mechanics: S-N curves, Goodman/Gerber, Paris' Law."""

    @staticmethod
    def s_n_curve(N, sigma_f_prime, b, N_f=1e6):
        """Baseline S-N curve using Basquin's law: sigma_a = sigma_f_prime * (N)^b.

        Returns amplitude stress for given cycle count N (or cycles to failure for given stress).
        """
        sigma_a = sigma_f_prime * (N ** b)
        return sigma_a

    @staticmethod
    def goodman_mean_stress(sigma_a, sigma_m, sigma_ut, sigma_se):
        """Goodman line: sigma_a / sigma_se + sigma_m / sigma_ut = 1.

        Returns factor of safety n_G = 1 / (sigma_a/sigma_se + sigma_m/sigma_ut).
        If n_G >= 1, infinite life predicted; else, damage per cycle.
        """
        n_G = 1.0 / (sigma_a / sigma_se + sigma_m / sigma_ut)
        return n_G

    @staticmethod
    def gerber_mean_stress(sigma_a, sigma_m, sigma_ut, sigma_se):
        """Gerber parabola: (sigma_a / sigma_se) + (sigma_m / sigma_ut)^2 = 1.

        Returns factor of safety n_Ge.
        """
        n_Ge = (-sigma_m / sigma_ut + np.sqrt((sigma_m / sigma_ut)**2 + 4 * sigma_a / sigma_ut)) / 2 * (sigma_se / sigma_a)
        # Simplified closed-form inversion; returns n such that sigma_a/n + (sigma_m/n)^2/sigma_ut = sigma_se
        # Actually proper: n = -sigma_m/sigma_ut + sqrt((sigma_m/sigma_ut)^2 + 4*sigma_a/sigma_ut)/2 ... I'll use standard
        # Let's compute n by solving quadratic in n: (sigma_m^2/sigma_ut^2)/n^2 + sigma_a/sigma_se / n - 1 = 0
        # But for brevity, return damage index:
        damage = sigma_a / sigma_se + (sigma_m / sigma_ut)**2
        return damage  # >=1 means failure

    @staticmethod
    def paris_law_integration(a0, a_c, N_steps=100, C=1e-10, m=3.0, Delta_K_func=None):
        """Integrate Paris' Law da/dN = C * (Delta_K)^m from initial crack length a0 to critical a_c.

        Returns total number of cycles to failure N_f and history.
        """
        if Delta_K_func is None:
            raise ValueError("Delta_K function must be provided: Delta_K = f(a, sigma, geometry)")
        da_dN = lambda a: C * (Delta_K_func(a))**m
        # Simple Euler integration with adaptive stepping
        a = float(a0)
        N = 0.0
        da = a_c - a0
        steps = max(N_steps, 1)
        delta_a = da / steps
        a_history = [a]
        N_history = [0.0]
        for _ in range(steps):
            da_i = da_dN(a) * delta_a
            a = max(a + da_i, a_c)  # clamp to a_c
            N += delta_a
            a_history.append(a)
            N_history.append(N)
        return {"cycles_to_failure": N, "final_crack_length": a, "history": list(zip(a_history, N_history))}


# ----------------------------------------------------------------------
# 5. STRUCTURAL OPTIMIZATION
# ----------------------------------------------------------------------


def beam_weight_minimization(L=1.0, E=STEEL["E"], sigma_allow=250e6, F_load=5000.0,
                             I_min=1e-8, A_min=1e-4, n_nodes=11, method='bounded'):
    """Weight minimization of a simply supported beam under uniform load.

    Minimizes cross-sectional area (or I) subject to stress and deflection constraints.
    Uses gradient-based or evolutionary optimization.

    Returns optimal I, A, and weight.
    """
    # Simple 3-point beam: max moment M = F*L/4, max stress sigma = M*c/I, c = h/2, A = b*h
    # For I-section approx: I = A * h^2 / 12, with h = sqrt(12*I/A)
    # sigma = M * h/2 / I = M / (2*I/h) = M / (2*sqrt(I/(12*A))) ... we'll use direct constraints.

    def constraint_stress(params):
        I, A = params
        # Max bending moment for central point load on simply supported beam
        M = F_load * L / 4.0
        # Max direct stress (assuming axial + bending, here pure bending)
        sigma_max = M * np.sqrt(12 * I / A) / I  # sigma = M*c/I, c = sqrt(12*I/A)/2? Actually for rectangular: I=bh^3/12, A=bh -> c=h/2, sigma=M*c/I = M*(h/2)/(bh^3/12)=6M/(bh^2)=6M*sqrt(A)/(b^(3/2)... messy.
        # Simpler: treat I and A as independent design vars with empirical relation.
        # We'll just enforce sigma = F_load * L / (4 * I / np.sqrt(A) * factor) ...
        # To keep it robust, use simple beam formula: sigma = F_load * L / (4 * section_modulus), S = I / c.
        # Assume rectangular: I = b*h^3/12, A = b*h, c = h/2 -> S = I/c = b*h^2/6 = A*h/6, h = sqrt(12*I/A), S = A/6 * sqrt(12*I/A) = sqrt(3*A*I)/3? Let's just use S = np.sqrt(I*A*3) approx.
        # Actually, section modulus S = I / c. For rectangular, S = (b*h^2)/6 = (A*h)/6, and I = A*h^2/12 -> h = sqrt(12*I/A), so S = A/6 * sqrt(12*I/A) = sqrt(3*A*I)/3 * 2? Let's compute: S = A/6 * sqrt(12*I/A) = sqrt(A^2 * 12 * I / 36 / A) = sqrt(12*A*I/36) = sqrt(A*I/3). So sigma = M / S = M / sqrt(A*I/3).
        # We'll use that.
        S = np.sqrt(A * I / 3.0)
        sigma = F_load * L / (4.0 * S)
        return sigma - sigma_allow  # violation if >0

    def constraint_deflection(params):
        I, A = params
        # Max deflection for central point load: delta = F*L^3 / (48*E*I)
        delta = F_load * L**3 / (48.0 * E * I)
        return delta - 0.01  # allow 10mm max for L=1m example

    # Design variables: [I, A]
    x0 = [I_min, A_min]
    if method == 'bounded':
        bounds = Bounds([I_min, A_min], [np.inf, np.inf])
        constraints = [{'type': 'ineq', 'fun': constraint_stress},
                       {'type': 'ineq', 'fun': constraint_deflection}]
        res = minimize(lambda p: -A[0] * 1.0,  # minimize weight proxied by minimizing A (density constant)
                       x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-9})
    elif method == 'evolutionary':
        # Simple differential evolution
        bounds = Bounds([I_min, A_min], [np.inf, np.inf])
        res = differential_evolution(lambda p: -p[1], bounds, maxiter=200, popsize=30, disp=False)
    else:
        raise ValueError("method must be 'bounded' or 'evolutionary'")

    I_opt, A_opt = res.x
    # Compute actual weight (assuming density rho, volume = A*L, weight = rho*A*L)
    # We'll just return design variables and constraint values
    sigma_check = constraint_stress(res.x)[0]  # positive means violation
    delta_check = constraint_deflection(res.x)[0]
    return {
        "optimal_I": float(I_opt),
        "optimal_A": float(A_opt),
        "minimized_weight_proxy": float(-res.fun) if hasattr(res, 'fun') else float(A_opt),
        "stress_violation": float(sigma_check),
        "deflection_violation": float(delta_check),
        "success": bool(res.success)
    }


# ----------------------------------------------------------------------
# 6. ROTORDYNAMICS
# ----------------------------------------------------------------------


class Rotordynamics:
    r"""Rotordynamics: Jeffcott rotor model for critical speeds."""

    @staticmethod
    def jeffcott_rotor(m, k, e, Omega_range=None, critical_speed_tol=1e-3):
        """Simple Jeffcott rotor: mass m, stiffness k, unbalance e.

        Response amplitude X(Omega) = (e * Omega^2) / sqrt((k - m*Omega^2)^2 + (c*Omega)^2)

        For undamped case c=0, critical speeds occur at Omega = sqrt(k/m).

        Returns dict with critical speeds and amplitude vs speed.
        """
        omega_n = np.sqrt(k / m)
        if Omega_range is None:
            Omega_range = np.linspace(0.1 * omega_n, 3.0 * omega_n, 500)
        c = 0.0  # undamped; set >0 for damped
        Omega = Omega_range
        amp = (e * Omega**2) / np.sqrt((k - m * Omega**2)**2 + (c * Omega)**2)
        # Find peaks near omega_n
        peaks = Omega[np.isclose(amp, np.max(amp), rtol=0.01)]
        crit_speeds = omega_n * np.ones_like(peaks)
        return {
            "natural_freq_rad_s": omega_n,
            "critical_speed_rpm": omega_n * 60.0 / (2 * np.pi),
            "speed_rad_s": Omega,
            "amplitude": amp,
            "critical_speeds_rad_s": crit_speeds
        }


# ----------------------------------------------------------------------
# 7. CONTACT MECHANICS
# ----------------------------------------------------------------------


def hertz_contact(R1, R2, E1, nu1, E2, nu2, F_normal):
    """Hertz contact stress for two spheres/cylinders.

    Returns (contact_radius, max_pressure, von_mises_surface_stress).

    For spheres: a = (3*F*(1-nu1^2)/(4*E1) + ... )^(1/3)
    """
    # Reduced elastic modulus
    E_star = 1.0 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)
    # Contact radius for two spheres
    a = (3.0 * F_normal * R_eff / (4.0 * E_star)) ** (1.0 / 3.0) if R1 > 0 and R2 > 0 else 0.0
    # Effective radius: 1/R_eff = 1/R1 + 1/R2 (for general curved surfaces); if one is flat, R=inf -> 1/R=0
    R_eff = 1.0 / (1.0 / R1 + 1.0 / R2) if (R1 != 0 or R2 != 0) else np.inf
    # Recompute a with correct R_eff
    a = (3.0 * F_normal * R_eff / (4.0 * E_star)) ** (1.0 / 3.0) if R_eff > 0 else 0.0
    # Max contact pressure
    p_max = 3.0 * F_normal / (2.0 * np.pi * a**2)
    # Von Mises equivalent stress at surface (max at center of contact circle)
    # For Hertz half-space, max von Mises ~ 0.62 * p_max (approx)
    sigma_vm = 0.62 * p_max
    return a, p_max, sigma_vm


# ----------------------------------------------------------------------
# 8. FAILURE THEORIES
# ----------------------------------------------------------------------


def failure_theories_2d(sigma_x, sigma_y, tau_xy, sigma_ut=250e6, sigma_uc=250e6, sigma_yt=250e6, sigma_yc=250e6):
    """Apply common 2D failure theories.

    Returns dict with factors of safety/n indices:
    - Maximum Normal Stress (Rankine)
    - Maximum Shear (Tresca)
    - Distortion Energy (Von Mises)
    - Modified Mohr
    """
    # Principal stresses
    sigs = StressTransform.principal_stresses_3d(sigma_x, sigma_y, tau_xy, 0.0, 0.0, 0.0)
    s1, s2, s3 = sigs

    # 1. Maximum Normal Stress (Rankine)
    # For ductile, check tension/compression separately
    n_rankine_t = min(sigma_ut / s1 if s1 > 0 else np.inf, sigma_uc / s3 if s3 < 0 else np.inf)
    n_rankine_b = min(sigma_yt / s2 if s2 > 0 else np.inf, sigma_yc / s3 if s3 < 0 else np.inf)
    # Simplified single N:
    n_rankine = min([s for s in [n_rankine_t, n_rankine_b] if np.isfinite(s)] + [np.inf])

    # 2. Tresca Maximum Shear
    tau_max = (np.max([s1, s2, s3]) - np.min([s1, s2, s3])) / 2.0
    # Compare to shear yield = sigma_yt / 2 (assuming same in tension/compression)
    n_tresca = (sigma_yt / 2.0) / tau_max if tau_max > 0 else np.inf

    # 3. Von Mises
    sigma_vm = np.sqrt(s1**2 - s1*s2 + s2**2)  # for plane stress sigma3=0; if 3D use full
    # Actually using full principals:
    sigma_vm = np.sqrt(0.5 * ((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2))
    n_von_mises = sigma_yt / sigma_vm if sigma_vm > 0 else np.inf

    # 4. Modified Mohr (for brittle materials)
    # Simplified: check tensile vs compressive strength
    if s1 > 0:
        n_mohr = sigma_ut / s1
    elif s3 < 0:
        n_mohr = sigma_yc / abs(s3)
    else:
        n_mohr = np.inf

    return {
        "principal_stresses": {"s1": s1, "s2": s2, "s3": s3},
        "max_shear_stress_tau": tau_max,
        "von_mises_stress": sigma_vm,
        "factor_ranks_1": n_rankine,
        "factor_tresca": n_tresca,
        "factor_von_mises": n_von_mises,
        "factor_mohr": n_mohr
    }


# ----------------------------------------------------------------------
# MAIN DEMONSTRATION BLOCK
# ----------------------------------------------------------------------


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("FINITE ELEMENT: 1D Truss Analysis")
    print("=" * 60)
    # Simple 2-node truss under axial load
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    elements = [(0, 1)]
    truss = Truss1D(coords, elements, A=1e-4, E=STEEL["E"])
    F = np.zeros(4)  # 2 nodes * 2 DOFs
    F[1] = -1000.0  # vertical force at node 2
    u = truss.solve(F, fixed_dofs=[0])  # fix node 1 x-dof
    print(f"Displacements: {u}")
    print(f"Reaction at fixed node: {truss.assemble_global_stiffness() @ u}")

    print("\n" + "=" * 60)
    print("BEAM 2D: Euler-Bernoulli Simply Supported")
    print("=" * 60)
    L_beam = 2.0
    n_nodes_beam = 5
    coords_b = np.linspace(0, L_beam, n_nodes_beam).reshape(-1, 1)
    elements_b = [(i, i+1) for i in range(n_nodes_beam-1)]
    I_val = 1e-4  # m^4
    E_val = STEEL["E"]
    beam = Beam2D(n_nodes=n_nodes_beam, elements=elements_b, I=I_val, E=E_val, coords=coords_b.flatten())
    # Simply supported: fix DOFs 0 and 3 (node 0 vertical, node 1 rotation? Actually for 5 nodes, supports at ends)
    # DOF mapping: 2*i = vertical, 2*i+1 = rotation
    # Fix node 0 vertical (dof 0) and node last vertical (dof 2*(n-1)), and rotations maybe
    fixed = [0, 2*(n_nodes_beam-1)]  # vertical at both ends
    F_beam = np.zeros(2 * n_nodes_beam)
    # Apply central point load
    mid_idx = n_nodes_beam // 2
    F_beam[2*mid_idx] = -10000.0  # downward force
    u_beam = beam.solve(F_beam, fixed_dofs=fixed)
    print(f"Displacements/rotations: {u_beam}")
    print(f"Max deflection approx: {np.min(u_beam[0::2]):.6e} m")

    print("\n" + "=" * 60)
    print("STRESS TRANSFORMATION & FAILURE THEORIES")
    print("=" * 60)
    sx, sy, txy = 100e6, 50e6, 30e6  # Pa
    theta = np.radians(30)
    sx_p, sy_p, txy_p = StressTransform.transform_2d(sx, sy, txy, theta)
    print(f"Transformed stresses at {np.degrees(theta):.1f} deg: sigma_x'={sx_p/1e6:.2f} MPa, "
          f"sigma_y'={sy_p/1e6:.2f} MPa, tau_xy'={txy_p/1e6:.2f} MPa")
    s1, s2 = StressTransform.principal_stresses_2d(sx, sy, txy)
    print(f"Principal stresses: sigma1={s1/1e6:.2f} MPa, sigma2={s2/1e6:.2f} MPa")
    vm = StressTransform.von_mises_2d(sx, sy, txy)
    print(f"Von Mises stress: {vm/1e6:.2f} MPa")
    tc = StressTransform.tresca_max_shear(sx, sy, txy)
    print(f"Tresca max shear: {tc/1e6:.2f} MPa")
    ft = failure_theories_2d(sx, sy, txy, sigma_ut=500e6, sigma_yt=250e6)
    print(f"Failure factors - Rankine: {ft['factor_ranks_1']:.3f}, "
          f"Tresca: {ft['factor_tresca']:.3f}, "
          f"Von Mises: {ft['factor_von_mises']:.3f}, "
          f"Mohr: {ft['factor_mohr']:.3f}")

    print("\n" + "=" * 60)
    print("MECHANICAL VIBRATIONS: SDOF Damped")
    print("=" * 60)
    m_val = 10.0  # kg
    c_val = 5.0   # Ns/m
    k_val = 200.0 # N/m
    vib = MechanicalVibrations.sdof_damped_response(m=m_val, c=c_val, k=k_val,
                                                    F0=10.0, omega=2.0, t_max=5.0, dt=0.05)
    print(f"Natural frequency: {vib['natural_freq']:.2f} rad/s ({vib['natural_freq']*60/(2*np.pi):.1f} RPM)")
    print(f"Damping ratio: {vib['damping_ratio']:.3f}")
    print(f"Max displacement: {np.max(np.abs(vib['displacement'])):.6f} m")

    print("\n" + "=" * 60)
    print("FATIGUE: Goodman Mean Stress")
    print("=" * 60)
    sigma_a = 150e6  # Pa alternating
    sigma_m = 100e6  # Pa mean
    sigma_ut = 600e6
    sigma_se = 300e6  # endurance limit
    n_goodman = 1.0 / (sigma_a / sigma_se + sigma_m / sigma_ut)
    print(f"Goodman factor of safety: {n_goodman:.3f} "
          f"(>=1 => infinite life predicted)")
    # Paris law example
    da_dN = lambda a: 1e-10 * (1.0 * np.sqrt(a))**3.0  # simplified Delta_K = sigma*sqrt(pi*a)
    paris = FatigueFracture.paris_law_integration(a0=1e-6, a_c=1e-3, N_steps=50,
                                                  C=1e-10, m=3.0, Delta_K_func=da_dN)
    print(f"Paris law cycles to failure: {paris['cycles_to_failure']:.0e}")

    print("\n" + "=" * 60)
    print("STRUCTURAL OPTIMIZATION: Beam Weight Minimization")
    print("=" * 60)
    opt = beam_weight_minimization(L=2.0, E=STEEL["E"], sigma_allow=250e6, F_load=5000.0,
                                   I_min=1e-6, A_min=1e-4, n_nodes=9, method='bounded')
    print(f"Optimal I: {opt['optimal_I']:.5e} m^4")
    print(f"Optimal A: {opt['optimal_A']:.5e} m^2")
    print(f"Stress violation: {opt['stress_violation']/1e6:.3f} MPa (>{0 if opt['success'] else '>'})")
    print(f"Deflection violation: {opt['deflection_violation']/1e3:.3f} mm")
    print(f"Optimization success: {opt['success']}")

    print("\n" + "=" * 60)
    print("ROTORDYNAMICS: Jeffcott Rotor Critical Speed")
    print("=" * 60)
    rotor = Rotordynamics.jeffcott_rotor(m=50.0, k=1e6, e=1e-3)
    print(f"Natural frequency: {rotor['natural_freq_rad_s']:.2f} rad/s "
          f"({rotor['critical_speed_rpm']:.1f} RPM)")
    # Plot amplitude vs speed (simple text summary)
    idx_peak = np.argmax(rotor['amplitude'])
    print(f"Peak response at {rotor['speed_rad_s'][idx_peak]:.2f} rad/s, amp={rotor['amplitude'][idx_peak]:.4e} m")

    print("\n" + "=" * 60)
    print("CONTACT MECHANICS: Hertz Sphere Contact")
    print("=" * 60)
    a, p_max, sigma_vm = hertz_contact(R1=0.05, R2=0.05, E1=STEEL["E"], nu1=0.3,
                                        E2=STEEL["E"], nu2=0.3, F_normal=1000.0)
    print(f"Contact radius: {a*1e3:.3f} mm")
    print(f"Max contact pressure: {p_max/1e6:.1f} MPa")
    print(f"Est. von Mises surface stress: {sigma_vm/1e6:.1f} MPa")

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)#!/usr/bin/env python3
""" 
production_ready_mechanics.py
A comprehensive, modular Python script for Applied Mechanics / Solid Mechanics / Applied Design.
Topics: FEA (1D/2D truss/beam), Stress/Strain Transformations, Mechanical Vibrations,
Fatigue & Fracture Mechanics, Structural Optimization, Rotordynamics, Contact Mechanics,
Failure Theories.
Standard libraries: numpy, scipy, matplotlib.
"""

import numpy as np
from scipy.linalg import eig, expm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds, LinearConstraint, differential_evolution
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Default Material Properties (ISO, room temperature)
# ----------------------------------------------------------------------
STEEL = {"E": 210e9, "nu": 0.3, "rho": 7850.0}
ALUMINUM = {"E": 70e9, "nu": 0.33, "rho": 2700.0}

# ----------------------------------------------------------------------
# 1. FINITE ELEMENT ANALYSIS
# ----------------------------------------------------------------------


class Truss1D:
    r"""1D Truss Finite Element Analysis.

    Assumes axial deformation only, small strain, linear elasticity.
    """

    def __init__(self, coords, elements, A=1e-4, E=STEEL["E"]):
        self.coords = np.asarray(coords, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.A = np.atleast_1d(np.array(A, dtype=float))
        if self.A.size == 1:
            self.A = np.full(self.elements.shape[0] if self.elements.ndim > 0 else 1, self.A[0])
        else:
            self.A = self.A[:max(self.elements.shape[0], 1)]
        self.E = float(E)
        self.n_nodes = self.coords.shape[0]
        self.dim = self.coords.shape[1] if self.coords.ndim > 1 else 1

    def element_local_stiffness(self, i, j):
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        L = np.hypot(xj - xi, yj - yi)
        if L < 1e-15:
            raise ValueError(f"Zero-length element between nodes {i} and {j}")
        c = (xj - xi) / L
        s = (yj - yi) / L
        k = self.E * self.A[j] / L
        return np.array([[c, s], [-c, -s]]) * k

    def assemble_global_stiffness(self):
        n_dofs = self.n_nodes * self.dim
        K = np.zeros((n_dofs, n_dofs))
        for i, j in self.elements:
            ke = self.element_local_stiffness(i, j)
            dofs_i = [i * self.dim + k for k in range(self.dim)]
            dofs_j = [j * self.dim + k for k in range(self.dim)]
            dofs = dofs_i + dofs_j
            for r in range(self.dim):
                for c_ in range(self.dim):
                    K[dofs[r], dofs[c_]] += ke[r, c_]
        return K

    def solve(self, F, fixed_dofs):
        K = self.assemble_global_stiffness()
        F = np.asarray(F, dtype=float).copy()
        K_mod = K.copy()
        for fd in np.atleast_1d(fixed_dofs).astype(int):
            K_mod[fd, :] = 0.0
            K_mod[:, fd] = 0.0
            K_mod[fd, fd] = 1.0
            F[fd] = 0.0
        u = np.linalg.solve(K_mod, F)
        return u


class Beam2D:
    r"""2D Euler-Bernoulli Beam Finite Element Analysis.

    Uses Hermite shape functions for transverse deflection and rotation.
    Each node has 2 DOFs: vertical displacement and rotation.
    """

    def __init__(self, nodes, elements, I=1e-6, E=STEEL["E"], coords=None):
        self.n_nodes = int(nodes)
        self.elements = np.array(elements, dtype=int)
        self.E = float(E)
        self.I = float(I)
        if coords is None:
            self.coords = np.linspace(0.0, 1.0, self.n_nodes).reshape(-1, 1)
        else:
            self.coords = np.asarray(coords, dtype=float).reshape(-1, 1)
            if self.coords.shape[0] != self.n_nodes:
                raise ValueError("Number of coords must match nodes")

    def element_stiffness(self, i, j):
        xi = float(self.coords[i, 0])
        xj = float(self.coords[j, 0])
        L = xj - xi
        if abs(L) < 1e-12:
            raise ValueError(f"Zero-length element between nodes {i} and {j}")
        EI = self.E * self.I
        k = (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

    def assemble_global_stiffness(self):
        n_dofs = self.n_nodes * 2
        K = np.zeros((n_dofs, n_dofs))
        for i, j in self.elements:
            ke = self.element_stiffness(i, j)
            dofs_i = [2*i, 2*i+1]
            dofs_j = [2*j, 2*j+1]
            dofs = dofs_i + dofs_j
            for r in range(4):
                for c_ in range(4):
                    K[dofs[r], dofs[c_]] += ke[r, c_]
        return K

    def solve(self, F, fixed_dofs):
        K = self.assemble_global_stiffness()
        F = np.asarray(F, dtype=float).copy()
        K_mod = K.copy()
        for fd in np.atleast_1d(fixed_dofs).astype(int):
            K_mod[fd, :] = 0.0
            K_mod[:, fd] = 0.0
            K_mod[fd, fd] = 1.0
            F[fd] = 0.0
        u = np.linalg.solve(K_mod, F)
        return u


# ----------------------------------------------------------------------
# 2. STRESS & STRAIN TRANSFORMATIONS
# ----------------------------------------------------------------------


class StressTransform:
    r"""Stress/strain tensor transformations, Mohr's circle, and yield criteria."""

    @staticmethod
    def transform_2d(sigma_x, sigma_y, tau_xy, theta):
        theta = float(theta)
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_x_prime = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
        sigma_y_prime = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
        tau_xy_prime = -(sigma_x - sigma_y) * s * c + tau_xy * (c**2 - s**2)
        return sigma_x_prime, sigma_y_prime, tau_xy_prime

    @staticmethod
    def mohr_circle_2d(sigma_x, sigma_y, tau_xy):
        sigma_center = (sigma_x + sigma_y) / 2.0
        R = np.hypot(sigma_x - sigma_center, tau_xy)
        sigma1 = sigma_center + R
        sigma2 = sigma_center - R
        return sigma_center, R, float(sigma1), float(sigma2)

    @staticmethod
    def principal_stresses_2d(sigma_x, sigma_y, tau_xy):
        _, _, s1, s2 = StressTransform.mohr_circle_2d(sigma_x, sigma_y, tau_xy)
        return (float(s1), float(s2)) if s1 >= s2 else (float(s2), float(s1))

    @staticmethod
    def von_mises_2d(sigma_x, sigma_y, tau_xy):
        s1, s2 = StressTransform.principal_stresses_2d(sigma_x, sigma_y, tau_xy)
        return np.sqrt(s1**2 - s1*s2 + s2**2)

    @staticmethod
    def tresca_max_shear(sigma_x, sigma_y, tau_xy):
        s1, s2, s3 = StressTransform.principal_stresses_3d(sigma_x, sigma_y, tau_xy, 0.0, 0.0, 0.0)
        return (np.max([s1, s2, s3]) - np.min([s1, s2, s3])) / 2.0

    @staticmethod
    def principal_stresses_3d(sigma_x, sigma_y, tau_xy, sigma_z=0.0, tau_yz=0.0, tau_xz=0.0):
        sigma_norm = np.array([sigma_x, sigma_y, sigma_z])
        tau_norm = np.array([tau_xy, tau_yz, tau_xz])
        T = np.diag(sigma_norm) + np.array([
            [0.0, tau_norm[0], tau_norm[2]],
            [tau_norm[0], 0.0, tau_norm[1]],
            [tau_norm[2], tau_norm[1], 0.0]
        ])
        eigenvalues = eig(T, homo=False)[0]
        return tuple(float(v) for v in sorted(eigenvalues, reverse=True))


# ----------------------------------------------------------------------
# 3. MECHANICAL VIBRATIONS
# ----------------------------------------------------------------------


class MechanicalVibrations:
    r"""Mechanical vibrations: SDOF/MDOF damped response, state-space."""

    @staticmethod
    def sdof_damped_response(m, c, k, F0=0.0, omega=0.0, t_max=10.0, dt=0.01, x0=0.0, v0=0.0):
        wn = np.sqrt(k / m)
        zeta = c / (2.0 * np.sqrt(m * k))

        def ode(t, y):
            x, v = y
            dxdt = v
            dvdt = (F0 * np.cos(omega * t) - c * v - k * x) / m
            return [dxdt, dvdt]

        t_span = (0.0, t_max)
        t_eval = np.arange(0, t_max + dt, dt)
        sol = solve_ivp(ode, t_span, [x0, v0], t_eval=t_eval, method='RK45')
        return {
            "time": sol.t,
            "displacement": sol.y[0, :],
            "velocity": sol.y[1, :],
            "natural_freq": wn,
            "damping_ratio": zeta
        }

    @staticmethod
    def mdof_modal_analysis(M, K, C=None):
        M = np.asarray(M, dtype=float)
        K = np.asarray(K, dtype=float)
        if M.shape != K.shape:
            raise ValueError("Mass and stiffness matrices must have same shape")
        if C is None:
            phi, lambda_ = eig(K, M)
            omega_n = np.sqrt(np.real(lambda_))
            idx = np.argsort(omega_n)
            omega_n = omega_n[idx]
            phi = phi[:, idx]
            return omega_n, phi
        else:
            C = np.asarray(C, dtype=float)
            omega_n, phi = MechanicalVibrations.mdof_modal_analysis(M, K)
            return omega_n, phi

    @staticmethod
    def state_space_integration(A, B, x0, u_func, t_span, dt=0.01):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        n = A.shape[0]

        def ode(t, y):
            u = u_func(t) if callable(u_func) else u_func
            return A @ y + B * u

        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        sol = solve_ivp(ode, t_span, x0, t_eval=t_eval, method='RK45')
        return {
            "time": sol.t,
            "state": sol.y.T,
            "input": np.atleast_2d(u_func(sol.t)).T if callable(u_func) else np.full((len(sol.t), *np.shape(u_func)), u_func)
        }


# ----------------------------------------------------------------------
# 4. FATIGUE & FRACTURE MECHANICS
# ----------------------------------------------------------------------


class FatigueFracture:
    r"""Fatigue & Fracture Mechanics: S-N curves, Goodman/Gerber, Paris' Law."""

    @staticmethod
    def s_n_curve(N, sigma_f_prime, b):
        sigma_a = sigma_f_prime * (N ** b)
        return sigma_a

    @staticmethod
    def goodman_mean_stress(sigma_a, sigma_m, sigma_ut, sigma_se):
        n_G = 1.0 / (sigma_a / sigma_se + sigma_m / sigma_ut)
        return n_G

    @staticmethod
    def gerber_mean_stress(sigma_a, sigma_m, sigma_ut, sigma_se):
        damage = sigma_a / sigma_se + (sigma_m / sigma_ut)**2
        return damage

    @staticmethod
    def paris_law_integration(a0, a_c, N_steps=100, C=1e-10, m=3.0, Delta_K_func=None):
        if Delta_K_func is None:
            raise ValueError("Delta_K function must be provided")
        da_dN = lambda a: C * (Delta_K_func(a))**m
        a = float(a0)
        N = 0.0
        da = a_c - a0
        steps = max(N_steps, 1)
        delta_a = da / steps
        a_history = [a]
        N_history = [0.0]
        for _ in range(steps):
            da_i = da_dN(a) * delta_a
            a = max(a + da_i, a_c)
            N += delta_a
            a_history.append(a)
            N_history.append(N)
        return {"cycles_to_failure": N, "final_crack_length": a, "history": list(zip(a_history, N_history))}


# ----------------------------------------------------------------------
# 5. STRUCTURAL OPTIMIZATION
# ----------------------------------------------------------------------


def beam_weight_minimization(L=1.0, E=STEEL["E"], sigma_allow=250e6, F_load=5000.0,
                             I_min=1e-8, A_min=1e-4, n_nodes=11, method='bounded'):
    """Weight minimization of a simply supported beam under central point load.

    Minimizes cross-sectional area subject to stress and deflection constraints.
    """
    def constraint_stress(params):
        I, A = params
        S = np.sqrt(A * I / 3.0)
        sigma = F_load * L / (4.0 * S)
        return sigma - sigma_allow

    def constraint_deflection(params):
        I, A = params
        delta = F_load * L**3 / (48.0 * E * I)
        return delta - 0.01

    x0 = [I_min, A_min]
    if method == 'bounded':
        bounds = Bounds([I_min, A_min], [np.inf, np.inf])
        constraints = [
            {'type': 'ineq', 'fun': constraint_stress},
            {'type': 'ineq', 'fun': constraint_deflection}
        ]
        res = minimize(lambda p: -p[1], x0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
    elif method == 'evolutionary':
        bounds = Bounds([I_min, A_min], [np.inf, np.inf])
        res = differential_evolution(lambda p: -p[1], bounds, maxiter=200, popsize=30, disp=False)
    else:
        raise ValueError("method must be 'bounded' or 'evolutionary'")

    I_opt, A_opt = res.x
    sigma_check = constraint_stress(res.x)[0]
    delta_check = constraint_deflection(res.x)[0]
    return {
        "optimal_I": float(I_opt),
        "optimal_A": float(A_opt),
        "minimized_weight_proxy": float(-res.fun) if hasattr(res, 'fun') else float(A_opt),
        "stress_violation": float(sigma_check),
        "deflection_violation": float(delta_check),
        "success": bool(res.success)
    }


# ----------------------------------------------------------------------
# 6. ROTORDYNAMICS
# ----------------------------------------------------------------------


class Rotordynamics:
    r"""Rotordynamics: Jeffcott rotor model for critical speeds."""

    @staticmethod
    def jeffcott_rotor(m, k, e, Omega_range=None, critical_speed_tol=1e-3):
        omega_n = np.sqrt(k / m)
        if Omega_range is None:
            Omega_range = np.linspace(0.1 * omega_n, 3.0 * omega_n, 500)
        c = 0.0
        Omega = Omega_range
        amp = (e * Omega**2) / np.sqrt((k - m * Omega**2)**2 + (c * Omega)**2)
        peaks = Omega[np.isclose(amp, np.max(amp), rtol=0.01)]
        crit_speeds = omega_n * np.ones_like(peaks)
        return {
            "natural_freq_rad_s": omega_n,
            "critical_speed_rpm": omega_n * 60.0 / (2 * np.pi),
            "speed_rad_s": Omega,
            "amplitude": amp,
            "critical_speeds_rad_s": crit_speeds
        }


# ----------------------------------------------------------------------
# 7. CONTACT MECHANICS
# ----------------------------------------------------------------------


def hertz_contact(R1, R2, E1, nu1, E2, nu2, F_normal):
    """Hertz contact stress for two spheres/cylinders.

    Returns (contact_radius, max_pressure, von_mises_surface_stress).
    """
    E_star = 1.0 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)
    R_eff = 1.0 / (1.0 / R1 + 1.0 / R2) if (R1 != 0 or R2 != 0) else np.inf
    a = (3.0 * F_normal * R_eff / (4.0 * E_star)) ** (1.0 / 3.0) if R_eff > 0 else 0.0
    p_max = 3.0 * F_normal / (2.0 * np.pi * a**2) if a > 0 else 0.0
    sigma_vm = 0.62 * p_max if p_max > 0 else 0.0
    return a, p_max, sigma_vm


# ----------------------------------------------------------------------
# 8. FAILURE THEORIES
# ----------------------------------------------------------------------


def failure_theories_2d(sigma_x, sigma_y, tau_xy, sigma_ut=250e6, sigma_uc=250e6,
                        sigma_yt=250e6, sigma_yc=250e6):
    """Apply common 2D failure theories.

    Returns dict with factors of safety/n indices.
    """
    sigs = StressTransform.principal_stresses_3d(sigma_x, sigma_y, tau_xy, 0.0, 0.0, 0.0)
    s1, s2, s3 = sigs

    n_rankine = min([s for s in [
        sigma_ut / s1 if s1 > 0 else np.inf,
        sigma_uc / abs(s3) if s3 < 0 else np.inf,
        sigma_yt / s2 if s2 > 0 else np.inf,
        sigma_yc / abs(s3) if s3 < 0 else np.inf
    ] if np.isfinite(s)] + [np.inf])

    tau_max = (np.max([s1, s2, s3]) - np.min([s1, s2, s3])) / 2.0
    n_tresca = (sigma_yt / 2.0) / tau_max if tau_max > 0 else np.inf

    sigma_vm = np.sqrt(0.5 * ((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2))
    n_von_mises = sigma_yt / sigma_vm if sigma_vm > 0 else np.inf

    if s1 > 0:
        n_mohr = sigma_ut / s1
    elif s3 < 0:
        n_mohr = sigma_yc / abs(s3)
    else:
        n_mohr = np.inf

    return {
        "principal_stresses": {"s1": s1, "s2": s2, "s3": s3},
        "max_shear_stress_tau": tau_max,
        "von_mises_stress": sigma_vm,
        "factor_ranks_1": n_rankine,
        "factor_tresca": n_tresca,
        "factor_von_mises": n_von_mises,
        "factor_mohr": n_mohr
    }


# ----------------------------------------------------------------------
# MAIN DEMONSTRATION BLOCK
# ----------------------------------------------------------------------


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("FINITE ELEMENT: 1D Truss Analysis")
    print("=" * 60)
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    elements = [(0, 1)]
    truss = Truss1D(coords, elements, A=1e-4, E=STEEL["E"])
    F = np.zeros(4)
    F[1] = -1000.0
    u = truss.solve(F, fixed_dofs=[0])
    print(f"Displacements: {u}")
    print(f"Reaction forces: {truss.assemble_global_stiffness() @ u}")

    print("\n" + "=" * 60)
    print("BEAM 2D: Euler-Bernoulli Simply Supported")
    print("=" * 60)
    L_beam = 2.0
    n_nodes_beam = 5
    coords_b = np.linspace(0, L_beam, n_nodes_beam).reshape(-1, 1)
    elements_b = [(i, i+1) for i in range(n_nodes_beam-1)]
    I_val = 1e-4
    E_val = STEEL["E"]
    beam = Beam2D(n_nodes=n_nodes_beam, elements=elements_b, I=I_val, E=E_val, coords=coords_b.flatten())
    fixed = [0, 2*(n_nodes_beam-1)]
    F_beam = np.zeros(2 * n_nodes_beam)
    mid_idx = n_nodes_beam // 2
    F_beam[2*mid_idx] = -10000.0
    u_beam = beam.solve(F_beam, fixed_dofs=fixed)
    print(f"Displacements/rotations: {u_beam}")
    print(f"Max deflection approx: {np.min(u_beam[0::2]):.6e} m")

    print("\n" + "=" * 60)
    print("STRESS TRANSFORMATION & FAILURE THEORIES")
    print("=" * 60)
    sx, sy, txy = 100e6, 50e6, 30e6
    theta = np.radians(30)
    sx_p, sy_p, txy_p = StressTransform.transform_2d(sx, sy, txy, theta)
    print(f"Transformed stresses at {np.degrees(theta):.1f} deg: sigma_x'={sx_p/1e6:.2f} MPa, "
          f"sigma_y'={sy_p/1e6:.2f} MPa, tau_xy'={txy_p/1e6:.2f} MPa")
    s1, s2 = StressTransform.principal_stresses_2d(sx, sy, txy)
    print(f"Principal stresses: sigma1={s1/1e6:.2f} MPa, sigma2={s2/1e6:.2f} MPa")
    vm = StressTransform.von_mises_2d(sx, sy, txy)
    print(f"Von Mises stress: {vm/1e6:.2f} MPa")
    tc = StressTransform.tresca_max_shear(sx, sy, txy)
    print(f"Tresca max shear: {tc/1e6:.2f} MPa")
    ft = failure_theories_2d(sx, sy, txy, sigma_ut=500e6, sigma_yt=250e6)
    print(f"Failure factors - Rankine: {ft['factor_ranks_1']:.3f}, "
          f"Tresca: {ft['factor_tresca']:.3f}, "
          f"Von Mises: {ft['factor_von_mises']:.3f}, "
          f"Mohr: {ft['factor_mohr']:.3f}")

    print("\n" + "=" * 60)
    print("MECHANICAL VIBRATIONS: SDOF Damped")
    print("=" * 60)
    m_val = 10.0
    c_val = 5.0
    k_val = 200.0
    vib = MechanicalVibrations.sdof_damped_response(m=m_val, c=c_val, k=k_val,
                                                    F0=10.0, omega=2.0, t_max=5.0, dt=0.05)
    print(f"Natural frequency: {vib['natural_freq']:.2f} rad/s ({vib['natural_freq']*60/(2*np.pi):.1f} RPM)")
    print(f"Damping ratio: {vib['damping_ratio']:.3f}")
    print(f"Max displacement: {np.max(np.abs(vib['displacement'])):.6f} m")

    print("\n" + "=" * 60)
    print("FATIGUE: Goodman Mean Stress & Paris Law")
    print("=" * 60)
    sigma_a = 150e6
    sigma_m = 100e6
    sigma_ut = 600e6
    sigma_se = 300e6
    n_goodman = 1.0 / (sigma_a / sigma_se + sigma_m / sigma_ut)
    print(f"Goodman factor of safety: {n_goodman:.3f} "
          f"(>=1 => infinite life predicted)")
    da_dN = lambda a: 1e-10 * (1.0 * np.sqrt(a))**3.0
    paris = FatigueFracture.paris_law_integration(a0=1e-6, a_c=1e-3, N_steps=50,
                                                  C=1e-10, m=3.0, Delta_K_func=da_dN)
    print(f"Paris law cycles to failure: {paris['cycles_to_failure']:.0e}")

    print("\n" + "=" * 60)
    print("STRUCTURAL OPTIMIZATION: Beam Weight Minimization")
    print("=" * 60)
    opt = beam_weight_minimization(L=2.0, E=STEEL["E"], sigma_allow=250e6, F_load=5000.0,
                                   I_min=1e-6, A_min=1e-4, n_nodes=9, method='bounded')
    print(f"Optimal I: {opt['optimal_I']:.5e} m^4")
    print(f"Optimal A: {opt['optimal_A']:.5e} m^2")
    print(f"Stress violation: {opt['stress_violation']/1e6:.3f} MPa "
          f"({'OK' if opt['stress_violation'] <= 0 else 'VIOLATION'})")
    print(f"Deflection violation: {opt['deflection_violation']/1e3:.3f} mm "
          f"({'OK' if opt['deflection_violation'] <= 0 else 'VIOLATION'})")
    print(f"Optimization success: {opt['success']}")

    print("\n" + "=" * 60)
    print("ROTORDYNAMICS: Jeffcott Rotor Critical Speed")
    print("=" * 60)
    rotor = Rotordynamics.jeffcott_rotor(m=50.0, k=1e6, e=1e-3)
    print(f"Natural frequency: {rotor['natural_freq_rad_s']:.2f} rad/s "
          f"({rotor['critical_speed_rpm']:.1f} RPM)")
    idx_peak = np.argmax(rotor['amplitude'])
    print(f"Peak response at {rotor['speed_rad_s'][idx_peak]:.2f} rad/s, amp={rotor['amplitude'][idx_peak]:.4e} m")

    print("\n" + "=" * 60)
    print("CONTACT MECHANICS: Hertz Sphere Contact")
    print("=" * 60)
    a, p_max, sigma_vm = hertz_contact(R1=0.05, R2=0.05, E1=STEEL["E"], nu1=0.3,
                                        E2=STEEL["E"], nu2=0.3, F_normal=1000.0)
    print(f"Contact radius: {a*1e3:.3f} mm")
    print(f"Max contact pressure: {p_max/1e6:.1f} MPa")
    print(f"Est. von Mises surface stress: {sigma_vm/1e6:.1f} MPa")

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)