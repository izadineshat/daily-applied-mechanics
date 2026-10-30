"""
Mechanical Engineering Toolbox
==============================
Standalone production-ready script covering:
- Finite Element stiffness matrices (1D/2D truss, 2D beam, CST plate)
- Stress/strain transformations, Mohr's circle, Von Mises, Tresca, failure theories
- Mechanical vibrations (SDOF/MDOF modal analysis, damped response, state-space, Jeffcott rotor)
- Fatigue & fracture mechanics (S-N/Basquin, Goodman/Gerber, Paris' law)
- Structural optimization (beam cross-section weight minimization, gradient & GA)
- Hertzian contact mechanics

Uses only numpy, scipy, matplotlib.
"""

import numpy as np
from scipy import linalg, optimize
from scipy.integrate import quad
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Finite Element Analysis - Stiffness Matrices
# -----------------------------------------------------------------------------
def truss_stiffness_1d(E: float, A: float, L: float) -> np.ndarray:
    """
    1D truss (axial) element stiffness matrix in local coordinates.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    A : float
        Cross-sectional area [m^2].
    L : float
        Element length [m].

    Returns
    -------
    k : (2, 2) ndarray
        Local stiffness matrix.
    """
    k = (E * A / L) * np.array([[1.0, -1.0],
                                [-1.0, 1.0]])
    return k


def truss_stiffness_2d(x1: float, y1: float, x2: float, y2: float,
                       E: float, A: float):
    """
    2D truss element stiffness matrix in global coordinates.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Node coordinates [m].
    E : float
        Young's modulus [Pa].
    A : float
        Cross-sectional area [m^2].

    Returns
    -------
    k_g : (4, 4) ndarray
        Global stiffness matrix (DOFs: ux1, uy1, ux2, uy2).
    L : float
        Element length [m].
    c, s : float
        Direction cosines.
    """
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L <= 0.0:
        raise ValueError("Element length must be positive.")
    c = dx / L
    s = dy / L
    # Local axial DOFs projected onto global: d_local = T @ d_global
    T = np.array([[c, s, 0.0, 0.0],
                  [0.0, 0.0, c, s]])
    k_local = truss_stiffness_1d(E, A, L)
    k_g = T.T @ k_local @ T
    return k_g, L, c, s


def beam_stiffness_2d_local(E: float, A: float, I: float, L: float) -> np.ndarray:
    """
    2D Euler-Bernoulli beam element stiffness matrix in local coordinates.

    DOF order: [u1, v1, theta1, u2, v2, theta2].

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    A : float
        Cross-sectional area [m^2].
    I : float
        Second moment of area [m^4].
    L : float
        Element length [m].

    Returns
    -------
    k : (6, 6) ndarray
        Local stiffness matrix.
    """
    k = np.zeros((6, 6))
    # Axial terms
    k[0, 0] = E * A / L
    k[0, 3] = -E * A / L
    k[3, 0] = -E * A / L
    k[3, 3] = E * A / L
    # Bending terms
    EI_L3 = E * I / L**3
    EI_L2 = E * I / L**2
    EI_L = E * I / L
    k[1, 1] = 12.0 * EI_L3
    k[1, 2] = 6.0 * EI_L2
    k[1, 4] = -12.0 * EI_L3
    k[1, 5] = 6.0 * EI_L2

    k[2, 1] = 6.0 * EI_L2
    k[2, 2] = 4.0 * EI_L
    k[2, 4] = -6.0 * EI_L2
    k[2, 5] = 2.0 * EI_L

    k[4, 1] = -12.0 * EI_L3
    k[4, 2] = -6.0 * EI_L2
    k[4, 4] = 12.0 * EI_L3
    k[4, 5] = -6.0 * EI_L2

    k[5, 1] = 6.0 * EI_L2
    k[5, 2] = 2.0 * EI_L
    k[5, 4] = -6.0 * EI_L2
    k[5, 5] = 4.0 * EI_L
    return k


def beam_transformation_2d(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """
    2D beam coordinate transformation matrix (local to global).

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Node coordinates [m].

    Returns
    -------
    T : (6, 6) ndarray
        Transformation matrix such that d_global = T @ d_local.
    """
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L <= 0.0:
        raise ValueError("Element length must be positive.")
    c = dx / L
    s = dy / L
    R = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    T = np.zeros((6, 6))
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    return T


def beam_stiffness_2d_global(x1: float, y1: float, x2: float, y2: float,
                             E: float, A: float, I: float):
    """
    2D beam element stiffness matrix in global coordinates.

    Returns
    -------
    k_g : (6, 6) ndarray
        Global stiffness matrix (DOFs: ux1, uy1, theta1, ux2, uy2, theta2).
    L : float
        Element length [m].
    """
    L = np.hypot(x2 - x1, y2 - y1)
    k_local = beam_stiffness_2d_local(E, A, I, L)
    T = beam_transformation_2d(x1, y1, x2, y2)
    k_g = T.T @ k_local @ T
    return k_g, L


def cst_stiffness_plane_stress(xy: np.ndarray, E: float, nu: float,
                               t: float = 1.0) -> np.ndarray:
    """
    Constant Strain Triangle (CST) element stiffness matrix, plane stress.

    Parameters
    ----------
    xy : (3, 2) ndarray
        Nodal coordinates [[x1, y1], [x2, y2], [x3, y3]] [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    t : float, optional
        Thickness [m]. Default 1.0.

    Returns
    -------
    k : (6, 6) ndarray
        Element stiffness matrix (DOFs: ux1, uy1, ux2, uy2, ux3, uy3).
    """
    xy = np.asarray(xy, dtype=float)
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    # Signed area
    A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if abs(A) < 1e-14:
        raise ValueError("Degenerate triangle (zero area).")

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i] = b[i]
        B[1, 2 * i + 1] = c[i]
        B[2, 2 * i] = c[i]
        B[2, 2 * i + 1] = b[i]
    B /= (2.0 * A)

    D = E / (1.0 - nu**2) * np.array([[1.0, nu, 0.0],
                                      [nu, 1.0, 0.0],
                                      [0.0, 0.0, (1.0 - nu) / 2.0]])

    k = t * A * (B.T @ D @ B)
    return k


# -----------------------------------------------------------------------------
# Stress & Strain Transformations
# -----------------------------------------------------------------------------
def stress_transformation_2d(sx: float, sy: float, txy: float,
                             theta_deg: float):
    """
    Transform 2D plane stress state by angle theta (counterclockwise).

    Parameters
    ----------
    sx, sy : float
        Normal stresses [Pa].
    txy : float
        Shear stress [Pa].
    theta_deg : float
        Rotation angle [degrees].

    Returns
    -------
    sxp, syp, txyp : float
        Transformed stress components [Pa].
    """
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    sxp = sx * c**2 + sy * s**2 + 2.0 * txy * s * c
    syp = sx * s**2 + sy * c**2 - 2.0 * txy * s * c
    txyp = (sy - sx) * s * c + txy * (c**2 - s**2)
    return sxp, syp, txyp


def principal_stresses_2d(sx: float, sy: float, txy: float):
    """
    Principal stresses and maximum in-plane shear for 2D stress state.

    Parameters
    ----------
    sx, sy : float
        Normal stresses [Pa].
    txy : float
        Shear stress [Pa].

    Returns
    -------
    s1, s2 : float
        Principal stresses [Pa].
    tau_max : float
        Maximum in-plane shear stress [Pa].
    theta_p : float
        Principal plane angle [radians].
    """
    avg = (sx + sy) / 2.0
    R = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
    s1 = avg + R
    s2 = avg - R
    tau_max = R
    theta_p = 0.5 * np.arctan2(2.0 * txy, sx - sy)
    return s1, s2, tau_max, theta_p


def von_mises_stress(sx: float, sy: float, txy: float,
                     sz: float = 0.0, txz: float = 0.0,
                     tyz: float = 0.0) -> float:
    """
    Von Mises equivalent stress for a general 3D stress state.

    Parameters
    ----------
    sx, sy, sz : float
        Normal stresses [Pa].
    txy, txz, tyz : float
        Shear stresses [Pa].

    Returns
    -------
    vm : float
        Von Mises stress [Pa].
    """
    S = np.array([[sx, txy, txz],
                  [txy, sy, tyz],
                  [txz, tyz, sz]])
    s_m = np.trace(S) / 3.0
    dev = S - s_m * np.eye(3)
    vm = np.sqrt(1.5 * np.sum(dev * dev))
    return vm


def von_mises_stress_2d(sx: float, sy: float, txy: float) -> float:
    """Plane-stress Von Mises stress (sz = 0)."""
    return np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)


def tresca_stress_2d(sx: float, sy: float, txy: float) -> float:
    """
    Tresca equivalent stress (max shear * 2) for plane stress (sigma3 = 0).
    """
    s1, s2, _, _ = principal_stresses_2d(sx, sy, txy)
    principals = np.array([s1, s2, 0.0])
    return principals.max() - principals.min()


def mohr_circle_2d(sx: float, sy: float, txy: float, n_points: int = 200):
    """
    Compute Mohr's circle data for 2D stress state.

    Parameters
    ----------
    sx, sy : float
        Normal stresses [Pa].
    txy : float
        Shear stress [Pa].
    n_points : int
        Number of points on the circle.

    Returns
    -------
    center : float
        Circle center [Pa].
    radius : float
        Circle radius [Pa].
    sigma_circle : ndarray
        Normal stress coordinates [Pa].
    tau_circle : ndarray
        Shear stress coordinates [Pa].
    s1, s2 : float
        Principal stresses [Pa].
    """
    center = (sx + sy) / 2.0
    radius = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
    theta = np.linspace(0.0, np.pi, n_points)  # physical angle, 2*theta on circle
    sigma_circle = center + radius * np.cos(2.0 * theta)
    tau_circle = radius * np.sin(2.0 * theta)
    s1, s2, _, _ = principal_stresses_2d(sx, sy, txy)
    return center, radius, sigma_circle, tau_circle, s1, s2


def failure_safety_factor(sx: float, sy: float, txy: float,
                          Sy: float, theory: str = "von_mises") -> float:
    """
    Factor of safety based on static failure theories.

    Parameters
    ----------
    sx, sy : float
        Normal stresses [Pa].
    txy : float
        Shear stress [Pa].
    Sy : float
        Yield strength [Pa].
    theory : str
        'von_mises', 'tresca', or 'max_principal'.

    Returns
    -------
    n : float
        Factor of safety (infty if equivalent stress is zero).
    """
    if theory == "von_mises":
        seq = von_mises_stress_2d(sx, sy, txy)
    elif theory == "tresca":
        seq = tresca_stress_2d(sx, sy, txy)
    elif theory == "max_principal":
        s1, s2, _, _ = principal_stresses_2d(sx, sy, txy)
        seq = max(abs(s1), abs(s2))
    else:
        raise ValueError(f"Unknown failure theory: {theory}")
    if seq <= 0.0:
        return np.inf
    return Sy / seq


# -----------------------------------------------------------------------------
# Mechanical Vibrations
# -----------------------------------------------------------------------------
def sdof_natural_frequency(m: float, k: float) -> float:
    """Undamped natural frequency [rad/s] of a SDOF system."""
    return np.sqrt(k / m)


def sdof_damping_ratio(c: float, m: float, k: float) -> float:
    """Damping ratio [-] of a SDOF system."""
    return c / (2.0 * np.sqrt(m * k))


def sdof_frequency_response(m: float, c: float, k: float,
                            omega: np.ndarray):
    """
    Frequency response function (receptance) X/F for SDOF system.

    Parameters
    ----------
    m, c, k : float
        Mass [kg], damping [Ns/m], stiffness [N/m].
    omega : ndarray
        Excitation frequencies [rad/s].

    Returns
    -------
    H : ndarray
        Complex displacement per unit force [m/N].
    """
    omega = np.asarray(omega, dtype=float)
    H = 1.0 / (k - m * omega**2 + 1j * c * omega)
    return H


def mdof_modal_analysis(M: np.ndarray, K: np.ndarray,
                        num_modes: int = None):
    """
    Solve generalized eigenproblem K phi = omega^2 M phi.

    Parameters
    ----------
    M : (n, n) ndarray
        Mass matrix.
    K : (n, n) ndarray
        Stiffness matrix.
    num_modes : int, optional
        Number of lowest modes to return.

    Returns
    -------
    omega_n : (num_modes,) ndarray
        Natural frequencies [rad/s].
    Phi : (n, num_modes) ndarray
        Mass-normalized mode shapes.
    """
    if num_modes is None:
        num_modes = M.shape[0]
    w2, Phi = linalg.eigh(K, M)
    idx = np.argsort(w2)[:num_modes]
    w2 = w2[idx]
    Phi = Phi[:, idx]
    omega_n = np.sqrt(np.maximum(w2, 0.0))
    # Mass-normalize modes: phi^T M phi = 1
    for i in range(Phi.shape[1]):
        norm = np.sqrt(Phi[:, i] @ M @ Phi[:, i])
        Phi[:, i] /= norm
    return omega_n, Phi


def modal_damping_ratios(Phi: np.ndarray, M: np.ndarray,
                         C: np.ndarray, omega_n: np.ndarray):
    """
    Modal damping ratios for proportional damping (mass-normalized modes).

    zeta_i = (phi_i^T C phi_i) / (2 * omega_i)
    """
    n_modes = Phi.shape[1]
    zeta = np.zeros(n_modes)
    for i in range(n_modes):
        phi = Phi[:, i]
        c_modal = phi @ C @ phi
        zeta[i] = c_modal / (2.0 * omega_n[i])
    return zeta


def mdof_harmonic_response(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                           F: np.ndarray, omega_range: np.ndarray):
    """
    Steady-state harmonic response X(omega) = [K - w^2 M + i w C]^-1 F.

    Parameters
    ----------
    M, C, K : (n, n) ndarray
        System matrices.
    F : (n,) ndarray
        Force amplitude vector.
    omega_range : ndarray
        Frequency vector [rad/s].

    Returns
    -------
    X : (n, len(omega_range)) complex ndarray
        Complex displacement response.
    """
    n_dof = M.shape[0]
    X = np.zeros((n_dof, len(omega_range)), dtype=complex)
    for i, w in enumerate(omega_range):
        Z = K - w**2 * M + 1j * w * C
        X[:, i] = np.linalg.solve(Z, F)
    return X


def state_space_matrices(M: np.ndarray, C: np.ndarray,
                         K: np.ndarray):
    """
    First-order state-space form: z_dot = A z + B u, y = C_out z + D u.

    State z = [x; x_dot].
    """
    n = M.shape[0]
    M_inv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -M_inv @ K
    A[n:, n:] = -M_inv @ C
    B = np.zeros((2 * n, n))
    B[n:, :] = M_inv
    C_out = np.hstack([np.eye(n), np.zeros((n, n))])
    D = np.zeros((n, n))
    return A, B, C_out, D


def jeffcott_critical_speed(m: float, k: float) -> float:
    """Undamped critical speed [rad/s] of a Jeffcott rotor."""
    return np.sqrt(k / m)


def jeffcott_unbalance_response(m: float, c: float, k: float,
                                m_unbalance: float, e: float,
                                omega_range: np.ndarray):
    """
    Steady-state unbalance response of a Jeffcott rotor.

    F_unbalance = m_unbalance * e * omega^2
    X = F / |k - m w^2 + i c w|
    """
    omega_range = np.asarray(omega_range, dtype=float)
    F = m_unbalance * e * omega_range**2
    denom = np.abs(k - m * omega_range**2 + 1j * c * omega_range)
    X = np.where(denom > 0.0, F / denom, np.inf)
    return X


# -----------------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# -----------------------------------------------------------------------------
def basquin_sigma(N_f: float, sigma_f_prime: float, b: float) -> float:
    """
    Basquin strain-life / stress-life relation:
    sigma_a = sigma_f_prime * (2 N_f)^b
    """
    return sigma_f_prime * (2.0 * N_f)**b


def basquin_cycles(sigma_a: float, sigma_f_prime: float, b: float) -> float:
    """
    Invert Basquin to estimate cycles to failure.
    N_f = 0.5 * (sigma_a / sigma_f_prime)^(1/b)
    """
    return 0.5 * (sigma_a / sigma_f_prime)**(1.0 / b)


def goodman_diagram(sigma_a: float, sigma_m: float,
                    Sut: float, Se: float):
    """
    Goodman mean-stress correction.

    Returns
    -------
    n : float
        Factor of safety.
    sigma_a_allow : float
        Allowable alternating stress.
    """
    damage = sigma_a / Se + sigma_m / Sut
    n = 1.0 / damage if damage > 0.0 else np.inf
    sigma_a_allow = Se * (1.0 - sigma_m / Sut)
    return n, sigma_a_allow


def gerber_diagram(sigma_a: float, sigma_m: float,
                   Sut: float, Se: float):
    """
    Gerber mean-stress correction (parabolic).
    """
    damage = sigma_a / Se + (sigma_m / Sut)**2
    n = 1.0 / damage if damage > 0.0 else np.inf
    sigma_a_allow = Se * (1.0 - (sigma_m / Sut)**2)
    return n, sigma_a_allow


def stress_intensity_center_crack(sigma: float, a: float,
                                  Y: float = 1.0) -> float:
    """
    Mode I stress intensity factor for a center crack in an infinite plate.
    K = Y * sigma * sqrt(pi * a)
    """
    return Y * sigma * np.sqrt(np.pi * a)


def paris_law(delta_K: float, C: float, m: float) -> float:
    """
    Paris' law crack growth rate: da/dN = C (delta_K)^m.
    """
    return C * delta_K**m


def crack_growth_cycles(a_i: float, a_f: float, C: float, m: float,
                        delta_K_func, *args, **kwargs) -> float:
    """
    Integrate Paris' law to compute cycles to grow crack from a_i to a_f.

    N = integral_{a_i}^{a_f} da / [C (delta_K(a))^m]
    """
    def integrand(a):
        dK = delta_K_func(a, *args, **kwargs)
        if dK <= 0.0:
            return np.inf
        return 1.0 / (C * dK**m)
    N, _ = quad(integrand, a_i, a_f, epsabs=1e-12, epsrel=1e-10, limit=200)
    return N


# -----------------------------------------------------------------------------
# Structural Optimization
# -----------------------------------------------------------------------------
def beam_rect_weight(b: float, h: float, L: float, rho: float) -> float:
    """Weight of a rectangular beam: W = rho * b * h * L."""
    return rho * b * h * L


def beam_rect_constraints(b: float, h: float, L: float,
                          M_max: float, sigma_allow: float,
                          delta_allow: float, E: float):
    """
    Constraint violations for a simply supported rectangular beam under
    central point load derived from M_max.

    Returns
    -------
    g_stress : float
        sigma_max - sigma_allow (<= 0 feasible).
    g_deflection : float
        delta_max - delta_allow (<= 0 feasible).
    """
    I = b * h**3 / 12.0
    c = h / 2.0
    sigma_max = M_max * c / I
    # Equivalent central point load: M_max = F L / 4  =>  F = 4 M_max / L
    F = 4.0 * M_max / L
    delta_max = F * L**3 / (48.0 * E * I)
    return sigma_max - sigma_allow, delta_max - delta_allow


def objective_beam_penalty(x, L, rho, M_max, sigma_allow,
                           delta_allow, E):
    """Penalty-based objective for beam weight minimization."""
    b, h = x
    if b <= 0.0 or h <= 0.0:
        return 1e12
    weight = beam_rect_weight(b, h, L, rho)
    g1, g2 = beam_rect_constraints(b, h, L, M_max, sigma_allow,
                                   delta_allow, E)
    penalty = 1e8 * (max(0.0, g1)**2 + max(0.0, g2)**2)
    return weight + penalty


def gradient_descent_beam(x0, L, rho, M_max, sigma_allow,
                          delta_allow, E,
                          bounds=((0.01, 0.3), (0.01, 0.5)),
                          lr=0.01, max_iter=1000, tol=1e-8):
    """
    Simple numerical gradient descent with bound projection.
    """
    x = np.array(x0, dtype=float)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    args = (L, rho, M_max, sigma_allow, delta_allow, E)
    for _ in range(max_iter):
        grad = np.zeros_like(x)
        eps = 1e-7
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = eps
            grad[j] = (objective_beam_penalty(x + dx, *args) -
                       objective_beam_penalty(x - dx, *args)) / (2.0 * eps)
        x_new = x - lr * grad
        x_new = np.clip(x_new, lo, hi)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, objective_beam_penalty(x, *args)


def scipy_gradient_beam(x0, L, rho, M_max, sigma_allow,
                        delta_allow, E,
                        bounds=((0.01, 0.3), (0.01, 0.5))):
    """Gradient-based optimization via scipy.optimize.minimize (L-BFGS-B)."""
    args = (L, rho, M_max, sigma_allow, delta_allow, E)
    res = optimize.minimize(objective_beam_penalty, x0, args=args,
                            method="L-BFGS-B", bounds=bounds)
    return res.x, res.fun


def genetic_algorithm_beam(L, rho, M_max, sigma_allow,
                           delta_allow, E,
                           bounds=((0.01, 0.3), (0.01, 0.5)),
                           pop_size=60, generations=80,
                           mutation_rate=0.1, crossover_rate=0.8,
                           elitism=2, seed=42):
    """
    Real-coded genetic algorithm for beam cross-section weight minimization.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    args = (L, rho, M_max, sigma_allow, delta_allow, E)
    fitness = np.array([objective_beam_penalty(ind, *args) for ind in pop])

    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    for _ in range(generations):
        new_pop = []
        # Elitism
        elite_idx = np.argsort(fitness)[:elitism]
        new_pop.extend(pop[elite_idx].tolist())

        while len(new_pop) < pop_size:
            # Tournament selection (size 2)
            def tournament():
                i1, i2 = rng.integers(0, pop_size, size=2)
                return pop[i1] if fitness[i1] < fitness[i2] else pop[i2]

            p1 = tournament()
            p2 = tournament()

            if rng.random() < crossover_rate:
                alpha = rng.random()
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = alpha * p2 + (1.0 - alpha) * p1
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            for child in (c1, c2):
                for j in range(dim):
                    if rng.random() < mutation_rate:
                        child[j] += rng.normal(0.0, (hi[j] - lo[j]) * 0.05)
                        child[j] = np.clip(child[j], lo[j], hi[j])
                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break

        pop = np.array(new_pop[:pop_size])
        fitness = np.array([objective_beam_penalty(ind, *args) for ind in pop])
        gen_best = int(np.argmin(fitness))
        if fitness[gen_best] < best_fit:
            best_fit = fitness[gen_best]
            best = pop[gen_best].copy()

    return best, best_fit


# -----------------------------------------------------------------------------
# Contact Mechanics (Hertzian)
# -----------------------------------------------------------------------------
def hertzian_sphere_flat(F: float, R: float,
                         E1: float, nu1: float,
                         E2: float, nu2: float):
    """
    Hertzian contact between a sphere and a flat surface.

    Returns
    -------
    a : float
        Contact radius [m].
    p0 : float
        Maximum contact pressure [Pa].
    tau_max : float
        Approx. maximum shear stress [Pa] (~0.31 p0).
    z_tau_max : float
        Depth of max shear [m] (~0.48 a).
    E_star : float
        Effective elastic modulus [Pa].
    """
    E_star = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)
    a = (3.0 * F * R / (4.0 * E_star))**(1.0 / 3.0)
    p0 = 3.0 * F / (2.0 * np.pi * a**2)
    tau_max = 0.31 * p0
    z_tau_max = 0.48 * a
    return a, p0, tau_max, z_tau_max, E_star


def hertzian_cylinder_cylinder(F_per_length: float, R1: float, R2: float,
                               E1: float, nu1: float,
                               E2: float, nu2: float):
    """
    Hertzian line contact between two cylinders (per unit length).

    Returns
    -------
    b : float
        Half-width of contact patch [m].
    p0 : float
        Maximum pressure [Pa].
    E_star : float
        Effective modulus [Pa].
    """
    # Equivalent radius for two convex cylinders in contact
    R_eq = R1 * R2 / (R1 + R2)
    E_star = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)
    b = np.sqrt(4.0 * F_per_length * R_eq / (np.pi * E_star))
    p0 = 2.0 * F_per_length / (np.pi * b)
    return b, p0, E_star


# -----------------------------------------------------------------------------
# Demonstration / Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    plt.rcParams["figure.autolayout"] = True

    print("=" * 70)
    print("MECHANICAL ENGINEERING TOOLBOX - DEMONSTRATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Finite Element Stiffness Matrices
    # -------------------------------------------------------------------------
    print("\n[1] Finite Element Stiffness Matrices")
    E_steel = 200e9
    A_bar = 1e-3
    L_bar = 1.0
    k1d = truss_stiffness_1d(E_steel, A_bar, L_bar)
    print("1D Truss stiffness (E=200 GPa, A=1e-3 m^2, L=1 m):")
    print(k1d)

    k2d, L, c, s = truss_stiffness_2d(0.0, 0.0, 1.0, 0.0, E_steel, A_bar)
    print(f"\n2D Truss global stiffness (horizontal bar, L={L} m):")
    print(k2d)

    E_b = 200e9
    A_b = 2e-3
    I_b = 1.6e-6
    L_b = 2.0
    k_beam_local = beam_stiffness_2d_local(E_b, A_b, I_b, L_b)
    k_beam_global, _ = beam_stiffness_2d_global(0.0, 0.0, L_b, 0.0,
                                                E_b, A_b, I_b)
    print("\n2D Beam local stiffness (first 3x3 block shown):")
    print(k_beam_local[:3, :3])
    print("2D Beam global stiffness equals local for horizontal beam:",
          np.allclose(k_beam_local, k_beam_global))

    # CST plate
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    k_cst = cst_stiffness_plane_stress(xy, E=70e9, nu=0.33, t=0.01)
    print("\nCST plane stress stiffness (70 GPa, nu=0.33, t=10 mm):")
    print(k_cst)

    # -------------------------------------------------------------------------
    # 2. Stress & Strain Transformations
    # -------------------------------------------------------------------------
    print("\n[2] Stress & Strain Transformations")
    sx, sy, txy = 100e6, 40e6, 50e6  # Pa
    theta = 30.0
    sxp, syp, txyp = stress_transformation_2d(sx, sy, txy, theta)
    print(f"Stress state: sx={sx/1e6:.1f} MPa, sy={sy/1e6:.1f} MPa, txy={txy/1e6:.1f} MPa")
    print(f"Transformed at theta={theta} deg: sx'={sxp/1e6:.2f} MPa, "
          f"sy'={syp/1e6:.2f} MPa, txy'={txyp/1e6:.2f} MPa")

    s1, s2, tau_max, theta_p = principal_stresses_2d(sx, sy, txy)
    print(f"Principal stresses: s1={s1/1e6:.2f} MPa, s2={s2/1e6:.2f} MPa")
    print(f"Max in-plane shear: {tau_max/1e6:.2f} MPa at theta_p={np.degrees(theta_p):.2f} deg")

    vm = von_mises_stress_2d(sx, sy, txy)
    tresca = tresca_stress_2d(sx, sy, txy)
    print(f"Von Mises stress: {vm/1e6:.2f} MPa")
    print(f"Tresca equivalent stress: {tresca/1e6:.2f} MPa")

    Sy = 250e6
    n_vm = failure_safety_factor(sx, sy, txy, Sy, theory="von_mises")
    n_tresca = failure_safety_factor(sx, sy, txy, Sy, theory="tresca")
    print(f"Factor of safety (Von Mises, Sy=250 MPa): {n_vm:.3f}")
    print(f"Factor of safety (Tresca, Sy=250 MPa): {n_tresca:.3f}")

    # Mohr's circle plot
    center, radius, sig_circ, tau_circ, s1p, s2p = mohr_circle_2d(sx, sy, txy)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sig_circ / 1e6, tau_circ / 1e6, label="Mohr's circle")
    ax.plot([sx / 1e6, sy / 1e6], [txy / 1e6, -txy / 1e6], "o", label="Original points")
    ax.plot([s1p / 1e6, s2p / 1e6], [0.0, 0.0], "ro", label="Principal stresses")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(center / 1e6, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Normal stress [MPa]")
    ax.set_ylabel("Shear stress [MPa]")
    ax.set_title("Mohr's Circle - 2D Stress State")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # -------------------------------------------------------------------------
    # 3. Mechanical Vibrations
    # -------------------------------------------------------------------------
    print("\n[3] Mechanical Vibrations")
    m_sdof = 10.0
    k_sdof = 10000.0
    c_sdof = 20.0
    wn = sdof_natural_frequency(m_sdof, k_sdof)
    zeta = sdof_damping_ratio(c_sdof, m_sdof, k_sdof)
    print(f"SDOF: m={m_sdof} kg, k={k_sdof} N/m, c={c_sdof} Ns/m")
    print(f"Natural frequency: {wn:.2f} rad/s ({wn/(2*np.pi):.2f} Hz)")
    print(f"Damping ratio: {zeta:.4f}")

    w_range = np.linspace(0, 150, 400)
    H = sdof_frequency_response(m_sdof, c_sdof, k_sdof, w_range)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(w_range, np.abs(H), lw=1.5)
    ax.axvline(wn, color="r", ls="--", label="Natural frequency")
    ax.set_xlabel("Excitation frequency [rad/s]")
    ax.set_ylabel("|H(w)| [m/N]")
    ax.set_title("SDOF Frequency Response")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MDOF modal analysis (2-DOF shear building)
    M_mdof = np.diag([2.0, 2.0])
    K_mdof = np.array([[4000.0, -1000.0],
                       [-1000.0, 4000.0]])
    C_mdof = 0.02 * K_mdof  # proportional damping
    omega_n, Phi = mdof_modal_analysis(M_mdof, K_mdof)
    print("\nMDOF 2-DOF system:")
    print("Natural frequencies [rad/s]:", omega_n)
    print("Natural frequencies [Hz]:", omega_n / (2 * np.pi))
    print("Mode shapes (mass-normalized):")
    print(Phi)
    zeta_modal = modal_damping_ratios(Phi, M_mdof, C_mdof, omega_n)
    print("Modal damping ratios:", zeta_modal)

    fig, ax = plt.subplots(figsize=(5, 4))
    dofs = [1, 2]
    for i in range(Phi.shape[1]):
        ax.plot(dofs, Phi[:, i], marker="o", label=f"Mode {i+1} (wn={omega_n[i]:.1f})")
    ax.set_xlabel("DOF")
    ax.set_ylabel("Mode shape component")
    ax.set_title("MDOF Mode Shapes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Harmonic response MDOF
    F_vec = np.array([1000.0, 0.0])
    w_resp = np.linspace(0, 60, 300)
    X_resp = mdof_harmonic_response(M_mdof, C_mdof, K_mdof, F_vec, w_resp)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(w_resp, np.abs(X_resp[0, :]), label="DOF 1")
    ax.plot(w_resp, np.abs(X_resp[1, :]), label="DOF 2")
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("|X| [m]")
    ax.set_title("MDOF Harmonic Response")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # State-space
    A_ss, B_ss, C_ss, D_ss = state_space_matrices(M_mdof, C_mdof, K_mdof)
    print("\nState-space A matrix (4x4):")
    print(A_ss)
    print("State-space B matrix (4x2):")
    print(B_ss)

    # Jeffcott rotor
    m_rotor = 50.0
    k_rotor = 1.0e6
    c_rotor = 500.0
    m_e = 0.01
    e_rotor = 0.05
    w_crit = jeffcott_critical_speed(m_rotor, k_rotor)
    w_rotor = np.linspace(0, 1.5 * w_crit, 300)
    X_rotor = jeffcott_unbalance_response(m_rotor, c_rotor, k_rotor,
                                          m_e, e_rotor, w_rotor)
    print(f"\nJeffcott rotor critical speed: {w_crit:.1f} rad/s ({w_crit/(2*np.pi):.1f} Hz)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(w_rotor, X_rotor * 1e3, lw=1.5)
    ax.axvline(w_crit, color="r", ls="--", label="Critical speed")
    ax.set_xlabel("Rotor speed [rad/s]")
    ax.set_ylabel("Unbalance response [mm]")
    ax.set_title("Jeffcott Rotor Unbalance Response")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # 4. Fatigue & Fracture Mechanics
    # -------------------------------------------------------------------------
    print("\n[4] Fatigue & Fracture Mechanics")
    Sut = 600e6
    Se = 280e6
    sigma_f_prime = 900e6
    b_exp = -0.09

    N_range = np.logspace(3, 7, 100)
    sigma_range = basquin_sigma(N_range, sigma_f_prime, b_exp)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(N_range, sigma_range / 1e6, lw=1.5, label="Basquin S-N")
    ax.set_xlabel("Cycles to failure N_f")
    ax.set_ylabel("Alternating stress [MPa]")
    ax.set_title("S-N Curve (Basquin)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    sigma_a_test = 100e6
    sigma_m_test = 150e6
    n_g, sa_allow_g = goodman_diagram(sigma_a_test, sigma_m_test, Sut, Se)
    n_gr, sa_allow_gr = gerber_diagram(sigma_a_test, sigma_m_test, Sut, Se)
    print(f"\nMean stress fatigue: sigma_a={sigma_a_test/1e6:.0f} MPa, "
          f"sigma_m={sigma_m_test/1e6:.0f} MPa")
    print(f"Goodman: n={n_g:.3f}, sigma_a_allow={sa_allow_g/1e6:.1f} MPa")
    print(f"Gerber: n={n_gr:.3f}, sigma_a_allow={sa_allow_gr/1e6:.1f} MPa")

    # Goodman / Gerber diagram plot
    sigma_m_plot = np.linspace(0, Sut, 200)
    sa_goodman = Se * (1 - sigma_m_plot / Sut)
    sa_gerber = Se * (1 - (sigma_m_plot / Sut)**2)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sigma_m_plot / 1e6, sa_goodman / 1e6, lw=1.5, label="Goodman")
    ax.plot(sigma_m_plot / 1e6, sa_gerber / 1e6, lw=1.5, ls="--", label="Gerber")
    ax.plot([sigma_m_test / 1e6], [sigma_a_test / 1e6], "ro", label="Operating point")
    ax.set_xlabel("Mean stress [MPa]")
    ax.set_ylabel("Alternating stress [MPa]")
    ax.set_title("Goodman and Gerber Fatigue Diagrams")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fracture mechanics
    sigma_applied = 200e6
    a_crack = 0.005  # m
    K = stress_intensity_center_crack(sigma_applied, a_crack, Y=1.12)
    print(f"\nFracture: sigma={sigma_applied/1e6:.0f} MPa, a={a_crack*1000:.1f} mm")
    print(f"Stress intensity factor K = {K/1e6:.2f} MPa√m")

    # Paris law integration
    C_paris = 1.0e-12  # (m/cycle)/(MPa√m)^m
    m_paris = 3.0
    delta_sigma = 100e6
    Y_paris = 1.12

    def delta_K_func(a, delta_sigma, Y):
        return Y * delta_sigma * np.sqrt(np.pi * a)

    N_paris = crack_growth_cycles(0.002, 0.010, C_paris, m_paris,
                                  delta_K_func, delta_sigma, Y_paris)
    print(f"Paris law: cycles to grow crack 2 mm -> 10 mm: {N_paris:.1f} cycles")

    # -------------------------------------------------------------------------
    # 5. Structural Optimization
    # -------------------------------------------------------------------------
    print("\n[5] Structural Optimization - Rectangular Beam")
    L_opt = 1.0
    rho_opt = 7850.0
    M_max_opt = 5000.0
    sigma_allow_opt = 250e6
    delta_allow_opt = 0.005
    E_opt = 200e9

    x0 = [0.05, 0.1]
    # Custom gradient descent
    x_gd, f_gd = gradient_descent_beam(x0, L_opt, rho_opt, M_max_opt,
                                       sigma_allow_opt, delta_allow_opt,
                                       E_opt)
    # SciPy gradient
    x_scipy, f_scipy = scipy_gradient_beam(x0, L_opt, rho_opt, M_max_opt,
                                           sigma_allow_opt, delta_allow_opt,
                                           E_opt)
    # GA
    x_ga, f_ga = genetic_algorithm_beam(L_opt, rho_opt, M_max_opt,
                                        sigma_allow_opt, delta_allow_opt,
                                        E_opt)

    print(f"Gradient descent:   b={x_gd[0]:.4f} m, h={x_gd[1]:.4f} m, W={f_gd:.4f} N")
    print(f"SciPy L-BFGS-B:     b={x_scipy[0]:.4f} m, h={x_scipy[1]:.4f} m, W={f_scipy:.4f} N")
    print(f"Genetic algorithm:  b={x_ga[0]:.4f} m, h={x_ga[1]:.4f} m, W={f_ga:.4f} N")

    # Check constraints for GA result
    g1, g2 = beam_rect_constraints(x_ga[0], x_ga[1], L_opt, M_max_opt,
                                   sigma_allow_opt, delta_allow_opt, E_opt)
    print(f"GA constraint violations: stress={g1/1e6:.2f} MPa, "
          f"deflection={g2*1000:.3f} mm")

    # -------------------------------------------------------------------------
    # 6. Contact Mechanics
    # -------------------------------------------------------------------------
    print("\n[6] Contact Mechanics - Hertzian Contact")
    F_cont = 1000.0
    R_cont = 0.05
    E1 = E2 = 200e9
    nu1 = nu2 = 0.3
    a, p0, tau_max_c, z_tau, E_star = hertzian_sphere_flat(F_cont, R_cont,
                                                            E1, nu1, E2, nu2)
    print(f"Sphere-on-flat: F={F_cont} N, R={R_cont} m")
    print(f"Contact radius a = {a*1000:.3f} mm")
    print(f"Max pressure p0 = {p0/1e6:.2f} MPa")
    print(f"Max shear stress ~ {tau_max_c/1e6:.2f} MPa at depth {z_tau*1000:.2f} mm")

    # Cylinder line contact
    F_l = 5000.0  # N/m
    R1 = 0.05
    R2 = 0.05
    b_cont, p0_cont, E_star_c = hertzian_cylinder_cylinder(F_l, R1, R2,
                                                           E1, nu1, E2, nu2)
    print(f"\nCylinder-on-cylinder (per unit length): F={F_l} N/m, R={R1} m")
    print(f"Half-width b = {b_cont*1000:.3f} mm")
    print(f"Max pressure p0 = {p0_cont/1e6:.2f} MPa")

    # -------------------------------------------------------------------------
    # Show all figures
    # -------------------------------------------------------------------------
    print("\nAll demonstrations complete. Showing figures...")
    plt.show()
```