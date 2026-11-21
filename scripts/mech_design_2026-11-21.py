"""
mechanical_engineering_toolkit.py
=================================

A standalone, modular, production-ready toolkit for applied mechanics and
solid mechanics computations.  Every routine is written in pure Python
(NumPy / SciPy) with realistic SI defaults and extensive docstrings.

Modules
-------
1.  Finite Element Analysis
     - 2-D truss elements
     - 2-D Euler-Bernoulli frame (beam) elements
2.  Stress & Strain Transformations
     - plane-stress rotation, principal stresses, Mohr's circle
     - von Mises and Tresca equivalent stresses
3.  Mechanical Vibrations
     - SDOF free & harmonic response
     - MDOF modal analysis (mass-normalised modes)
     - state-space simulation
4.  Fatigue & Fracture Mechanics
     - Basquin S-N curve
     - Goodman / Gerber mean-stress corrections
     - Paris' law crack-growth integration
5.  Structural Optimization
     - SLSQP weight minimisation of a rectangular cantilever
     - genetic algorithm for the same problem
6.  Rotordynamics / Contact Mechanics / Failure Theories
     - Jeffcott rotor unbalance response
     - Hertzian spherical contact
     - Rankine / Tresca / von Mises safety factors

Dependencies: numpy, scipy, matplotlib
Run:  python mechanical_engineering_toolkit.py
"""

import numpy as np
from numpy import sqrt, pi
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.integrate import quad, solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. FINITE ELEMENT ANALYSIS
# =============================================================================

def truss2d_element_stiffness(E, A, x1, y1, x2, y2):
    """
    Return the 4x4 local/global stiffness matrix of a 2-D axial truss member.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    A : float
        Cross-sectional area [m^2].
    x1, y1 : float
        Coordinates of node i [m].
    x2, y2 : float
        Coordinates of node j [m].

    Returns
    -------
    ke : ndarray, shape (4, 4)
        Element stiffness matrix in global coordinates.
    """
    L = float(np.hypot(x2 - x1, y2 - y1))
    if L <= 0.0:
        raise ValueError("Zero-length truss element.")
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    k = E * A / L
    return k * np.array([
        [ c * c,  c * s, -c * c, -c * s],
        [ c * s,  s * s, -c * s, -s * s],
        [-c * c, -c * s,  c * c,  c * s],
        [-c * s, -s * s,  c * s,  s * s],
    ], dtype=float)


def assemble_truss(nodes, members, E, A):
    """
    Assemble the global stiffness matrix of a 2-D truss.

    Parameters
    ----------
    nodes : array_like, shape (n_nodes, 2)
        Nodal coordinates [m].
    members : list of (int, int)
        Connectivity list; each tuple (i, j) defines a member.
    E : float
        Young's modulus [Pa] (uniform for all members).
    A : float
        Cross-sectional area [m^2] (uniform for all members).

    Returns
    -------
    K : ndarray, shape (2*n_nodes, 2*n_nodes)
        Global stiffness matrix.
    """
    nodes = np.asarray(nodes, dtype=float)
    n_nodes = nodes.shape[0]
    K = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=float)
    for (i, j) in members:
        ke = truss2d_element_stiffness(
            E, A, nodes[i, 0], nodes[i, 1], nodes[j, 0], nodes[j, 1]
        )
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]
    return K


def solve_truss(nodes, members, E, A, fixed_dofs, loads):
    """
    Solve a linear 2-D truss static problem.

    Parameters
    ----------
    nodes, members, E, A :
        See :func:`assemble_truss`.
    fixed_dofs : sequence of int
        Constrained global DOF indices.
    loads : dict {int: float}
        Nodal load vector components [N]; keys are global DOF indices.

    Returns
    -------
    u : ndarray, shape (2*n_nodes,)
        Nodal displacement vector [m].
    reactions : ndarray, shape (2*n_nodes,)
        Reaction / residual force vector [N].
    K : ndarray
        Global stiffness matrix.
    """
    nodes = np.asarray(nodes, dtype=float)
    n_dof = 2 * nodes.shape[0]
    K = assemble_truss(nodes, members, E, A)

    F = np.zeros(n_dof, dtype=float)
    for dof, val in loads.items():
        F[int(dof)] += float(val)

    fixed = np.asarray(sorted(set(int(d) for d in fixed_dofs)), dtype=int)
    free = np.setdiff1d(np.arange(n_dof), fixed)

    u = np.zeros(n_dof, dtype=float)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    reactions = K @ u - F
    return u, reactions, K


def truss_member_forces(nodes, members, E, A, u):
    """
    Compute the axial force in every 2-D truss member.

    Returns
    -------
    forces : ndarray
        Axial member forces [N]; positive = tension.
    """
    nodes = np.asarray(nodes, dtype=float)
    u = np.asarray(u, dtype=float)
    forces = np.zeros(len(members), dtype=float)
    for k, (i, j) in enumerate(members):
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        L = float(np.hypot(x2 - x1, y2 - y1))
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        elongation = (-c * u[2 * i] - s * u[2 * i + 1]
                      + c * u[2 * j] + s * u[2 * j + 1])
        forces[k] = E * A / L * elongation
    return forces


def beam2d_element_stiffness(E, A, I, x1, y1, x2, y2):
    """
    Return the 6x6 stiffness matrix of a 2-D Euler-Bernoulli frame element.

    Local DOF ordering: [u_i, v_i, theta_i, u_j, v_j, theta_j].

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    A : float
        Cross-sectional area [m^2].
    I : float
        Second moment of area about the bending axis [m^4].
    x1, y1, x2, y2 : float
        End-point coordinates [m].

    Returns
    -------
    ke : ndarray, shape (6, 6)
        Element stiffness matrix in global coordinates.
    """
    L = float(np.hypot(x2 - x1, y2 - y1))
    if L <= 0.0:
        raise ValueError("Zero-length beam element.")
    c = (x2 - x1) / L
    s = (y2 - y1) / L

    EA_L = E * A / L
    EI = E * I
    k1 = 12.0 * EI / L ** 3
    k2 = 6.0 * EI / L ** 2
    k3 = 4.0 * EI / L
    k4 = 2.0 * EI / L

    k_local = np.array([
        [ EA_L,   0.0,   0.0, -EA_L,   0.0,   0.0],
        [  0.0,    k1,    k2,   0.0,   -k1,    k2],
        [  0.0,    k2,    k3,   0.0,   -k2,    k4],
        [-EA_L,   0.0,   0.0,  EA_L,   0.0,   0.0],
        [  0.0,   -k1,   -k2,   0.0,     k1,   -k2],
        [  0.0,    k2,    k4,   0.0,   -k2,    k3],
    ], dtype=float)

    T = np.array([
        [ c,  s, 0.0,  0.0, 0.0, 0.0],
        [-s,  c, 0.0,  0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c,   s, 0.0],
        [0.0, 0.0, 0.0, -s,   c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    return T.T @ k_local @ T


def assemble_frame(nodes, elements, E, A, I):
    """
    Assemble the global stiffness matrix of a 2-D frame.

    Each node has three DOFs: [u, v, theta].
    """
    nodes = np.asarray(nodes, dtype=float)
    n_dof = 3 * nodes.shape[0]
    K = np.zeros((n_dof, n_dof), dtype=float)
    for (i, j) in elements:
        ke = beam2d_element_stiffness(
            E, A, I, nodes[i, 0], nodes[i, 1], nodes[j, 0], nodes[j, 1]
        )
        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += ke[a, b]
    return K


def solve_frame(nodes, elements, E, A, I, fixed_dofs, loads):
    """
    Solve a linear 2-D frame static problem.

    Returns
    -------
    u, reactions, K
        Displacements [m, rad], reactions [N, N*m] and global stiffness.
    """
    nodes = np.asarray(nodes, dtype=float)
    n_dof = 3 * nodes.shape[0]
    K = assemble_frame(nodes, elements, E, A, I)

    F = np.zeros(n_dof, dtype=float)
    for dof, val in loads.items():
        F[int(dof)] += float(val)

    fixed = np.asarray(sorted(set(int(d) for d in fixed_dofs)), dtype=int)
    free = np.setdiff1d(np.arange(n_dof), fixed)

    u = np.zeros(n_dof, dtype=float)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    reactions = K @ u - F
    return u, reactions, K


# =============================================================================
# 2. STRESS & STRAIN TRANSFORMATIONS
# =============================================================================

def stress_transform_2d(sx, sy, txy, theta_deg):
    """
    Rotate a plane-stress state through angle ``theta_deg`` (CCW positive).

    Returns
    -------
    (sx_p, sy_p, txy_p) : tuple of floats
    """
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    sx_p = sx * c * c + sy * s * s + 2.0 * txy * s * c
    sy_p = sx * s * s + sy * c * c - 2.0 * txy * s * c
    txy_p = -(sx - sy) * s * c + txy * (c * c - s * s)
    return float(sx_p), float(sy_p), float(txy_p)


def principal_stresses_2d(sx, sy, txy):
    """
    Return principal stresses and the principal angle (in degrees).

    Returns
    -------
    s1, s2, theta_p_deg
    """
    avg = 0.5 * (sx + sy)
    R = float(np.hypot(0.5 * (sx - sy), txy))
    s1 = avg + R
    s2 = avg - R
    theta_p = 0.5 * np.degrees(np.arctan2(2.0 * txy, sx - sy))
    return float(s1), float(s2), float(theta_p)


def von_mises_2d(sx, sy, txy):
    """Plane-stress von Mises equivalent stress [same units as inputs]."""
    return float(sqrt(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy))


def von_mises_3d(sigma):
    """
    Von Mises equivalent stress from a full 3-D stress tensor.

    Parameters
    ----------
    sigma : array_like, shape (3, 3)
        Cauchy stress tensor [Pa].

    Returns
    -------
    float
        Equivalent stress [Pa].
    """
    sigma = np.asarray(sigma, dtype=float)
    mean = np.trace(sigma) / 3.0
    dev = sigma - mean * np.eye(3)
    return float(sqrt(1.5 * np.sum(dev * dev)))


def tresca_3d(sigma):
    """Tresca (maximum shear) equivalent stress from a 3-D tensor."""
    sigma = np.asarray(sigma, dtype=float)
    p = np.linalg.eigvalsh(sigma)
    return float(p[-1] - p[0])


def plot_mohr_circle(sx, sy, txy, filename="mohr_circle.png"):
    """Save a Mohr's circle plot for a given plane-stress state."""
    center = 0.5 * (sx + sy)
    R = float(np.hypot(0.5 * (sx - sy), txy))
    s1, s2, _ = principal_stresses_2d(sx, sy, txy)

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    sc = center + R * np.cos(theta)
    tc = R * np.sin(theta)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sc / 1e6, tc / 1e6, "b-", lw=2, label="Mohr's circle")
    ax.plot([sx / 1e6, sy / 1e6], [txy / 1e6, -txy / 1e6], "ro",
            label="Applied state")
    ax.plot([s1 / 1e6, s2 / 1e6], [0, 0], "ks", label="Principal stresses")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\sigma$ [MPa]")
    ax.set_ylabel(r"$\tau$ [MPa]")
    ax.set_title("Mohr's Circle (plane stress)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


# =============================================================================
# 3. MECHANICAL VIBRATIONS
# =============================================================================

class SDOF:
    """
    Single-degree-of-freedom damped oscillator:  m x'' + c x' + k x = f(t).
    """

    def __init__(self, m, k, c=None, zeta=None):
        self.m = float(m)
        self.k = float(k)
        if c is None:
            if zeta is None:
                zeta = 0.0
            c = 2.0 * zeta * sqrt(self.k * self.m)
        self.c = float(c)
        self.wn = sqrt(self.k / self.m)
        self.zeta = self.c / (2.0 * sqrt(self.k * self.m))
        if self.zeta < 1.0:
            self.wd = self.wn * sqrt(1.0 - self.zeta ** 2)
        else:
            self.wd = 0.0

    def free_response(self, t, x0, v0):
        """
        Analytical free response for initial displacement ``x0`` and velocity
        ``v0`` (arrays ``t`` and scalars are accepted).
        """
        t = np.asarray(t, dtype=float)
        z, wn = self.zeta, self.wn
        if z < 1.0:
            wd = self.wd
            env = np.exp(-z * wn * t)
            A = x0
            B = (v0 + z * wn * x0) / wd
            return env * (A * np.cos(wd * t) + B * np.sin(wd * t))
        if abs(z - 1.0) < 1e-10:
            return np.exp(-wn * t) * (x0 + (v0 + wn * x0) * t)
        # Overdamped
        root = wn * sqrt(z * z - 1.0)
        s1 = -z * wn + root
        s2 = -z * wn - root
        C1 = (v0 - s2 * x0) / (s1 - s2)
        C2 = (s1 * x0 - v0) / (s1 - s2)
        return C1 * np.exp(s1 * t) + C2 * np.exp(s2 * t)

    def harmonic_amplitude(self, omega, F0=1.0):
        """
        Steady-state amplitude for a base force F0*cos(omega t).

        Returns
        -------
        (X, phase_rad)
        """
        r = omega / self.wn
        denom = sqrt((1.0 - r * r) ** 2 + (2.0 * self.zeta * r) ** 2)
        X = F0 / self.k / denom
        phase = np.arctan2(2.0 * self.zeta * r, 1.0 - r * r)
        return float(X), float(phase)


def modal_analysis(M, K):
    """
    Solve the generalised eigenproblem K phi = omega^2 M phi.

    Parameters
    ----------
    M : ndarray
        Symmetric positive-definite mass matrix.
    K : ndarray
        Symmetric positive semi-definite stiffness matrix.

    Returns
    -------
    omega : ndarray
        Natural angular frequencies [rad/s].
    phi : ndarray
        Mass-normalised mode-shape matrix (columns are modes).
    """
    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    eigvals, phi = eigh(K, M)
    eigvals = np.clip(eigvals, 0.0, None)
    omega = np.sqrt(eigvals)
    return omega, phi


def state_space_response(A, B, C, x0, t_eval, u_func=None):
    """
    Simulate a linear time-invariant state-space system

        x' = A x + B u(t)
        y  = C x

    Parameters
    ----------
    A, B, C : ndarray
    x0 : ndarray
        Initial state.
    t_eval : ndarray
        Output time vector.
    u_func : callable or None
        Input function ``u(t)`` returning a vector.

    Returns
    -------
    Y : ndarray, shape (len(t_eval), n_outputs)
    """
    x0 = np.asarray(x0, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    def rhs(t, x):
        if u_func is None:
            return A @ x
        return A @ x + B @ np.atleast_1d(u_func(t))

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), x0, t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    X = sol.y.T
    return X @ np.asarray(C, dtype=float).T if np.ndim(C) == 2 else X * C


# =============================================================================
# 4. FATIGUE & FRACTURE MECHANICS
# =============================================================================

def basquin_strength(N, sigma_f_prime, b):
    """
    Basquin S-N relation:  sigma_a = sigma_f' (2N)^b.

    Parameters
    ----------
    N : float or array_like
        Number of cycles to failure.
    sigma_f_prime : float
        Fatigue strength coefficient [Pa].
    b : float
        Fatigue strength exponent (typically -0.05 to -0.12).
    """
    return sigma_f_prime * (2.0 * np.asarray(N, dtype=float)) ** b


def basquin_life(sigma_a, sigma_f_prime, b):
    """Inverse Basquin relation: cycles to failure for stress amplitude."""
    sigma_a = np.asarray(sigma_a, dtype=float)
    return 0.5 * (sigma_a / sigma_f_prime) ** (1.0 / b)


def goodman_factor(sigma_a, sigma_m, Se, Su):
    """
    Goodman mean-stress safety factor.

        n = 1 / (sigma_a/Se + sigma_m/Su)

    Returns ``inf`` when the state is on the origin.
    """
    denom = sigma_a / Se + sigma_m / Su
    if abs(denom) < 1e-16:
        return np.inf
    return 1.0 / denom


def gerber_factor(sigma_a, sigma_m, Se, Su):
    """
    Gerber mean-stress safety factor:

        n*sigma_a/Se + (n*sigma_m/Su)^2 = 1

    Returns
    -------
    n : float
    """
    if abs(sigma_m) < 1e-16:
        return Se / sigma_a if sigma_a > 0 else np.inf
    a = (sigma_m / Su) ** 2
    bcoef = sigma_a / Se
    c = -1.0
    disc = bcoef * bcoef - 4.0 * a * c
    if disc < 0.0:
        return 0.0
    n = (-bcoef + sqrt(disc)) / (2.0 * a)
    return float(n)


def paris_life(a0, ac, delta_sigma, C_paris, m_paris, Y=1.12):
    """
    Integrate Paris' law  da/dN = C (dK)^m  from a0 to ac.

    dK = Y * delta_sigma * sqrt(pi * a)   [MPa*sqrt(m)]

    Parameters
    ----------
    a0, ac : float
        Initial and critical crack length [m].
    delta_sigma : float
        Stress range [MPa].
    C_paris : float
        Paris coefficient [(m/cycle)/(MPa*sqrt(m))^m].
    m_paris : float
        Paris exponent (typically 2.5-4 for steels).
    Y : float
        Geometry factor (default 1.12 for an edge crack).

    Returns
    -------
    N : float
        Cycles to grow the crack from a0 to ac.
    """
    if ac <= a0:
        raise ValueError("ac must be greater than a0.")
    pref = C_paris * (Y * delta_sigma * sqrt(pi)) ** m_paris

    def integrand(a):
        return 1.0 / (pref * a ** (0.5 * m_paris))

    N, _ = quad(integrand, a0, ac)
    return float(N)


def plot_sn_and_goodman(sigma_f_prime, b, Se, Su, filename="fatigue.png"):
    """Save an S-N curve plus a Goodman diagram."""
    N = np.logspace(3, 7, 200)
    sa = basquin_strength(N, sigma_f_prime, b) / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].loglog(N, sa, "b-", lw=2)
    axes[0].set_xlabel("Cycles to failure, N")
    axes[0].set_ylabel(r"Stress amplitude $\sigma_a$ [MPa]")
    axes[0].set_title("Basquin S-N Curve")
    axes[0].grid(True, which="both", alpha=0.3)

    sm = np.linspace(0.0, Su / 1e6, 200)
    axes[1].plot(sm, Se / 1e6 * (1.0 - sm / (Su / 1e6)), "g-", lw=2,
                 label="Goodman")
    axes[1].plot(sm, (Se / 1e6) * (1.0 - (sm / (Su / 1e6)) ** 2), "r--",
                 lw=2, label="Gerber")
    axes[1].set_xlabel(r"Mean stress $\sigma_m$ [MPa]")
    axes[1].set_ylabel(r"Alternating stress $\sigma_a$ [MPa]")
    axes[1].set_title("Mean-Stress Failure Diagrams")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


# =============================================================================
# 5. STRUCTURAL OPTIMIZATION
# =============================================================================

def beam_stress(P, L, b, h):
    """Maximum bending stress of a tip-loaded rectangular cantilever."""
    return 6.0 * P * L / (b * h ** 2)


def beam_deflection(P, L, E, b, h):
    """Tip deflection of a tip-loaded rectangular cantilever."""
    I = b * h ** 3 / 12.0
    return P * L ** 3 / (3.0 * E * I)


def optimize_beam_section_slsqp(P, L, E, rho, sigma_allow, delta_allow,
                                b_bounds, h_bounds, x0=None):
    """
    Minimise the mass of a rectangular cantilever cross-section subject to
    bending-stress and tip-deflection constraints using SLSQP.

    Returns
    -------
    result : scipy OptimizeResult with fields ``x`` (b, h) and ``fun``.
    """
    if x0 is None:
        x0 = (0.03, 0.10)

    def mass(x):
        return rho * x[0] * x[1] * L

    def stress_con(x):
        return sigma_allow - beam_stress(P, L, x[0], x[1])

    def defl_con(x):
        return delta_allow - beam_deflection(P, L, E, x[0], x[1])

    constraints = [
        {"type": "ineq", "fun": stress_con},
        {"type": "ineq", "fun": defl_con},
    ]
    bounds = [tuple(b_bounds), tuple(h_bounds)]
    return minimize(mass, np.asarray(x0, dtype=float), method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 200, "ftol": 1e-12})


def genetic_algorithm(objective, bounds, n_pop=80, n_gen=200,
                      cx_prob=0.9, mut_prob=0.2, seed=0):
    """
    A small, dependency-free real-coded genetic algorithm (minimisation).

    Parameters
    ----------
    objective : callable
        Function of a parameter vector returning a scalar fitness.
    bounds : array_like, shape (n_var, 2)
        Lower and upper bounds for each variable.
    n_pop, n_gen : int
        Population size and number of generations.
    cx_prob, mut_prob : float
        Crossover and per-individual mutation probabilities.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    best : ndarray
        Best parameter vector found.
    best_fit : float
        Corresponding objective value.
    history : list of float
        Best fitness per generation.
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    n_var = bounds.shape[0]

    pop = lo + rng.random((n_pop, n_var)) * span
    fit = np.array([objective(ind) for ind in pop])
    history = []

    for _ in range(n_gen):
        order = np.argsort(fit)
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]

        while len(new_pop) < n_pop:
            i1, i2 = rng.integers(0, n_pop, 2)
            p1 = pop[i1] if fit[i1] < fit[i2] else pop[i2]
            i1, i2 = rng.integers(0, n_pop, 2)
            p2 = pop[i1] if fit[i1] < fit[i2] else pop[i2]

            if rng.random() < cx_prob:
                alpha = rng.random(n_var)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = (1.0 - alpha) * p1 + alpha * p2
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if rng.random() < mut_prob:
                    mask = rng.random(n_var) < 0.5
                    child[mask] += (rng.normal(0.0, 1.0, mask.sum())
                                    * span[mask] * 0.1)
                np.clip(child, lo, hi, out=child)

            new_pop.append(c1)
            if len(new_pop) < n_pop:
                new_pop.append(c2)

        pop = np.asarray(new_pop)
        fit = np.array([objective(ind) for ind in pop])
        history.append(float(fit.min()))

    idx = int(np.argmin(fit))
    return pop[idx], float(fit[idx]), history


def beam_penalty_objective(x, P, L, E, rho, sigma_allow, delta_allow,
                           penalty=1e3):
    """
    Penalised mass objective used by the genetic algorithm.

    Constraint violations are squared and normalised by their allowable
    values before being multiplied by ``penalty`` [kg].
    """
    b, h = x
    if b <= 0.0 or h <= 0.0:
        return 1e30
    mass = rho * b * h * L
    s_ratio = beam_stress(P, L, b, h) / sigma_allow
    d_ratio = beam_deflection(P, L, E, b, h) / delta_allow
    viol = max(0.0, s_ratio - 1.0) ** 2 + max(0.0, d_ratio - 1.0) ** 2
    return mass + penalty * viol


# =============================================================================
# 6. ROTORDYNAMICS, CONTACT MECHANICS & FAILURE THEORIES
# =============================================================================

def jeffcott_rotor_amplitude(omega, m, k, c, e):
    """
    Steady-state synchronous whirl amplitude of a Jeffcott (Laval) rotor.

    X(omega) = m e omega^2 / sqrt((k - m omega^2)^2 + (c omega)^2)

    Parameters
    ----------
    omega : float or array_like
        Rotational speed [rad/s].
    m : float
        Disc mass [kg].
    k : float
        Shaft stiffness [N/m].
    c : float
        Viscous damping coefficient [N*s/m].
    e : float
        Mass eccentricity [m].

    Returns
    -------
    X : ndarray
        Whirl amplitude [m].
    """
    omega = np.asarray(omega, dtype=float)
    num = m * e * omega ** 2
    den = np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)
    return num / den


def hertz_sphere_contact(F, R1, R2, E1, E2, nu1, nu2):
    """
    Hertzian contact of two spheres (R2 -> inf for a sphere on a flat).

    Returns
    -------
    dict with keys:
        ``a``          contact radius [m]
        ``p_max``      maximum Hertz pressure [Pa]
        ``delta``      elastic approach [m]
        ``E_star``     effective modulus [Pa]
        ``R_eff``      effective radius [m]
    """
    if not np.isfinite(R2):
        R_eff = R1
    else:
        R_eff = 1.0 / (1.0 / R1 + 1.0 / R2)
    E_star = 1.0 / ((1.0 - nu1 ** 2) / E1 + (1.0 - nu2 ** 2) / E2)
    a = (3.0 * F * R_eff / (4.0 * E_star)) ** (1.0 / 3.0)
    p_max = 3.0 * F / (2.0 * pi * a ** 2)
    delta = a ** 2 / R_eff
    return {"a": float(a), "p_max": float(p_max),
            "delta": float(delta), "E_star": float(E_star),
            "R_eff": float(R_eff)}


def failure_theories(sigma, Sy):
    """
    Evaluate classical failure theories for a 3-D stress tensor.

    Parameters
    ----------
    sigma : array_like, shape (3, 3)
        Cauchy stress tensor [Pa].
    Sy : float
        Yield strength [Pa].

    Returns
    -------
    dict
        ``principal`` principal stresses [Pa],
        ``von_mises`` equivalent stress [Pa],
        ``tresca`` equivalent stress [Pa],
        ``rankine_max`` maximum absolute principal stress [Pa],
        ``safety`` dictionary of safety factors.
    """
    sigma = np.asarray(sigma, dtype=float)
    p = np.linalg.eigvalsh(sigma)
    vm = von_mises_3d(sigma)
    tr = float(p[-1] - p[0])
    rankine = float(np.max(np.abs(p)))
    return {
        "principal": p,
        "von_mises": vm,
        "tresca": tr,
        "rankine_max": rankine,
        "safety": {
            "von_mises": Sy / vm if vm > 0 else np.inf,
            "tresca": Sy / tr if tr > 0 else np.inf,
            "rankine": Sy / rankine if rankine > 0 else np.inf,
        },
    }


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_fea_truss():
    print("\n" + "-" * 70)
    print("DEMO 1 : 2-D TRUSS FINITE ELEMENT ANALYSIS")
    print("-" * 70)

    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
    members = [(0, 2), (1, 2)]
    E = 200e9          # Steel
    A = 1.0e-4         # 100 mm^2
    fixed = [0, 1, 2, 3]
    loads = {5: -10.0e3}   # 10 kN downward at node 2

    u, reactions, K = solve_truss(nodes, members, E, A, fixed, loads)
    forces = truss_member_forces(nodes, members, E, A, u)

    print(f"  Node 2 displacement : "
          f"ux = {u[4]*1e3:+.3f} mm, uy = {u[5]*1e3:+.3f} mm")
    for k, (i, j) in enumerate(members):
        print(f"  Member {i}-{j}: axial force = {forces[k]/1e3:+.3f} kN "
              f"({'tension' if forces[k] >= 0 else 'compression'})")
    print(f"  Vertical reaction sum : {reactions[1]+reactions[3]:+.3f} N")


def demo_fea_beam():
    print("\n" + "-" * 70)
    print("DEMO 2 : 2-D FRAME / BEAM FINITE ELEMENT ANALYSIS")
    print("-" * 70)

    L = 1.0
    E = 210e9
    b, h = 0.05, 0.10
    A = b * h
    I = b * h ** 3 / 12.0
    P = 1000.0
    n_el = 5

    nodes = np.column_stack((np.linspace(0.0, L, n_el + 1),
                             np.zeros(n_el + 1)))
    elements = [(k, k + 1) for k in range(n_el)]
    fixed = [0, 1, 2]
    tip_dof = 3 * n_el + 1
    loads = {tip_dof: -P}

    u, reactions, K = solve_frame(nodes, elements, E, A, I, fixed, loads)
    delta_num = u[tip_dof]
    delta_exact = -P * L ** 3 / (3.0 * E * I)

    print(f"  Tip deflection (FEA)     : {delta_num*1e3:+.4f} mm")
    print(f"  Tip deflection (analytic): {delta_exact*1e3:+.4f} mm")
    print(f"  Relative error           : "
          f"{abs((delta_num-delta_exact)/delta_exact)*100:.3e} %")
    print(f"  Vertical reaction at wall: {reactions[1]:+.3f} N")


def demo_stress_transformations():
    print("\n" + "-" * 70)
    print("DEMO 3 : STRESS TRANSFORMATIONS & MOHR'S CIRCLE")
    print("-" * 70)

    sx, sy, txy = 100e6, 40e6, 60e6
    s1, s2, theta_p = principal_stresses_2d(sx, sy, txy)

    print(f"  Applied state        : sx = {sx/1e6:.1f} MPa, "
          f"sy = {sy/1e6:.1f} MPa, txy = {txy/1e6:.1f} MPa")
    print(f"  Principal stresses   : s1 = {s1/1e6:.2f} MPa, "
          f"s2 = {s2/1e6:.2f} MPa")
    print(f"  Principal angle      : {theta_p:.2f} deg")
    print(f"  von Mises (2-D)      : {von_mises_2d(sx, sy, txy)/1e6:.2f} MPa")
    print(f"  Tresca (2-D)         : {(s1 - s2)/1e6:.2f} MPa")

    for ang in (30.0, 45.0, 60.0):
        sp = stress_transform_2d(sx, sy, txy, ang)
        print(f"  Rotated by {ang:5.1f} deg : "
              f"sx' = {sp[0]/1e6:7.2f}, sy' = {sp[1]/1e6:7.2f}, "
              f"txy' = {sp[2]/1e6:7.2f} MPa")

    fname = plot_mohr_circle(sx, sy, txy)
    print(f"  Saved Mohr's-circle plot -> {fname}")


def demo_vibrations():
    print("\n" + "-" * 70)
    print("DEMO 4 : MECHANICAL VIBRATIONS")
    print("-" * 70)

    m, k = 10.0, 1.0e5
    zeta = 0.05
    osc = SDOF(m, k, zeta=zeta)
    print(f"  SDOF: wn = {osc.wn:.3f} rad/s "
          f"({osc.wn/(2*pi):.3f} Hz), zeta = {osc.zeta:.3f}, "
          f"wd = {osc.wd:.3f} rad/s")

    t = np.linspace(0.0, 0.5, 1000)
    x = osc.free_response(t, x0=0.01, v0=0.0)
    X_h, ph = osc.harmonic_amplitude(osc.wn, F0=100.0)
    print(f"  Harmonic amplitude at resonance (100 N): "
          f"{X_h*1e3:.3f} mm, phase = {np.degrees(ph):.1f} deg")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, x * 1e3, "b-", lw=1.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [mm]")
    ax.set_title("SDOF Free Response (ζ = 0.05)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("sdof_response.png", dpi=120)
    plt.close(fig)

    # --- MDOF modal analysis -------------------------------------------------
    m_val = 2.0
    k_val = 1.0e6
    M = np.diag([m_val, m_val, m_val])
    K = np.array([
        [ 2.0 * k_val, -k_val,          0.0],
        [ -k_val,       2.0 * k_val,   -k_val],
        [ 0.0,         -k_val,          k_val],
    ])
    omega, phi = modal_analysis(M, K)
    print("  3-DOF modal analysis:")
    for i, w in enumerate(omega):
        print(f"    mode {i+1}: wn = {w:10.3f} rad/s  "
              f"({w/(2*pi):8.3f} Hz)  shape = "
              f"{np.round(phi[:, i], 4)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(phi.shape[1]):
        ax.plot(np.arange(1, 4), phi[:, i] / np.max(np.abs(phi[:, i])),
                "o-", label=f"Mode {i+1}")
    ax.set_xlabel("Mass index")
    ax.set_ylabel("Normalised mode-shape component")
    ax.set_title("3-DOF Mode Shapes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("mdof_modes.png", dpi=120)
    plt.close(fig)


def demo_fatigue():
    print("\n" + "-" * 70)
    print("DEMO 5 : FATIGUE & FRACTURE MECHANICS")
    print("-" * 70)

    sigma_f, b = 900e6, -0.09
    Se, Su = 250e6, 600e6

    sa = 200e6
    N = basquin_life(sa, sigma_f, b)
    print(f"  Basquin life at sigma_a = {sa/1e6:.1f} MPa: "
          f"{N:.3e} cycles")

    sm = 100e6
    n_g = goodman_factor(sa, sm, Se, Su)
    n_ger = gerber_factor(sa, sm, Se, Su)
    print(f"  Goodman safety factor (sm = {sm/1e6:.1f} MPa): {n_g:.3f}")
    print(f"  Gerber  safety factor (sm = {sm/1e6:.1f} MPa): {n_ger:.3f}")

    a0, ac = 1.0e-3, 5.0e-2
    dsigma = 100.0
    C_paris, m_paris = 1.0e-11, 3.0
    N_paris = paris_life(a0, ac, dsigma, C_paris, m_paris, Y=1.12)
    print(f"  Paris law: crack {a0*1e3:.1f} mm -> {ac*1e3:.1f} mm at "
          f"Δσ = {dsigma:.0f} MPa")
    print(f"            predicted life = {N_paris:.3e} cycles")

    fname = plot_sn_and_goodman(sigma_f, b, Se, Su)
    print(f"  Saved S-N / Goodman plot -> {fname}")


def demo_optimization():
    print("\n" + "-" * 70)
    print("DEMO 6 : STRUCTURAL OPTIMIZATION (RECTANGULAR CANTILEVER)")
    print("-" * 70)

    P, L = 1000.0, 1.0
    E, rho = 210e9, 7850.0
    sigma_allow = 250e6
    delta_allow = 5.0e-3

    res = optimize_beam_section_slsqp(
        P, L, E, rho, sigma_allow, delta_allow,
        b_bounds=(0.005, 0.20), h_bounds=(0.010, 0.40),
        x0=(0.03, 0.10),
    )
    b_opt, h_opt = res.x
    print(f"  SLSQP optimum : b = {b_opt*1e3:7.2f} mm, "
          f"h = {h_opt*1e3:7.2f} mm, mass = {res.fun:.3f} kg")
    print(f"    stress = {beam_stress(P, L, b_opt, h_opt)/1e6:.2f} MPa "
          f"(allow {sigma_allow/1e6:.0f} MPa)")
    print(f"    deflect = {beam_deflection(P, L, E, b_opt, h_opt)*1e3:.3f} mm "
          f"(allow {delta_allow*1e3:.1f} mm)")

    obj = lambda x: beam_penalty_objective(
        x, P, L, E, rho, sigma_allow, delta_allow, penalty=1e3)
    best, best_fit, hist = genetic_algorithm(
        obj, bounds=[(0.005, 0.20), (0.010, 0.40)],
        n_pop=60, n_gen=120, seed=42,
    )
    print(f"  GA optimum    : b = {best[0]*1e3:7.2f} mm, "
          f"h = {best[1]*1e3:7.2f} mm, objective = {best_fit:.3f} kg")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist, "b-", lw=1.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective [kg]")
    ax.set_title("GA Convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ga_convergence.png", dpi=120)
    plt.close(fig)


def demo_rotordynamics():
    print("\n" + "-" * 70)
    print("DEMO 7 : ROTORDYNAMICS (JEFFCOTT ROTOR)")
    print("-" * 70)

    m, k, c = 10.0, 1.0e6, 200.0
    e = 2.0e-5
    wn = sqrt(k / m)
    zeta = c / (2.0 * sqrt(k * m))
    print(f"  Undamped critical speed : {wn:.2f} rad/s "
          f"= {wn*60/(2*pi):.1f} rpm")
    print(f"  Damping ratio           : {zeta:.4f}")

    omega = np.linspace(1.0, 3.0 * wn, 400)
    X = jeffcott_rotor_amplitude(omega, m, k, c, e)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(omega / wn, X * 1e3, "b-", lw=2)
    ax.axvline(1.0, color="r", ls="--", lw=1,
               label="Critical speed ω/ωn = 1")
    ax.set_xlabel(r"Speed ratio $\omega/\omega_n$")
    ax.set_ylabel("Whirl amplitude [mm]")
    ax.set_title("Jeffcott Rotor Synchronous Response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("jeffcott_response.png", dpi=120)
    plt.close(fig)
    print("  Saved rotor response plot -> jeffcott_response.png")


def demo_contact_and_failure():
    print("\n" + "-" * 70)
    print("DEMO 8 : HERTZ CONTACT & FAILURE THEORIES")
    print("-" * 70)

    F = 500.0
    R1, R2 = 0.01, np.inf
    E1 = E2 = 210e9
    nu1 = nu2 = 0.3
    hz = hertz_sphere_contact(F, R1, R2, E1, E2, nu1, nu2)
    print(f"  Hertz contact (F = {F:.0f} N, R = {R1*1e3:.0f} mm):")
    print(f"    contact radius  a     = {hz['a']*1e6:.2f} µm")
    print(f"    maximum pressure p_max= {hz['p_max']/1e6:.1f} MPa")
    print(f"    elastic approach δ    = {hz['delta']*1e9:.1f} nm")

    sigma = np.array([
        [120.0,  40.0,  0.0],
        [ 40.0,  30.0,  0.0],
        [  0.0,   0.0, -20.0],
    ]) * 1e6
    Sy = 350e6
    out = failure_theories(sigma, Sy)
    print("  Failure theories for a 3-D stress state:")
    print(f"    principal stresses = {np.round(out['principal']/1e6, 2)} MPa")
    print(f"    von Mises = {out['von_mises']/1e6:7.2f} MPa  -> "
          f"n = {out['safety']['von_mises']:.3f}")
    print(f"    Tresca    = {out['tresca']/1e6:7.2f} MPa  -> "
          f"n = {out['safety']['tresca']:.3f}")
    print(f"    Rankine   = {out['rankine_max']/1e6:7.2f} MPa  -> "
          f"n = {out['safety']['rankine']:.3f}")


def main():
    print("=" * 70)
    print(" MECHANICAL ENGINEERING TOOLKIT - DEMONSTRATION SUITE")
    print("=" * 70)

    demo_fea_truss()
    demo_fea_beam()
    demo_stress_transformations()
    demo_vibrations()
    demo_fatigue()
    demo_optimization()
    demo_rotordynamics()
    demo_contact_and_failure()

    print("\n" + "=" * 70)
    print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()