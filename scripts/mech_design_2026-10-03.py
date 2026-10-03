"""
Standalone mechanical engineering utilities for applied mechanics, solid mechanics,
and applied design.

Modules implemented:
- 1D/2D truss finite element analysis
- 2D Euler-Bernoulli beam finite element analysis
- 2D plane-stress plate quadrilateral elements
- Thin-plate bending stiffness element
- Stress and strain transformations, Mohr's circle, von Mises, Tresca
- SDOF/MDOF vibration analysis, modal analysis, state-space simulation
- Fatigue S-N curves, Goodman/Gerber corrections, Miner's rule, Paris crack growth
- Simple rotordynamics (Jeffcott rotor)
- Hertzian contact mechanics
- Beam cross-section weight optimization using gradient and genetic algorithms

This script uses only NumPy, SciPy, and Matplotlib.
"""

import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad
from math import factorial, pi, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1.0e-12


def safe_solve(A, b):
    """Solve A x = b with a least-squares fallback for singular systems."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.size == 0:
        return b.copy()
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def solve_constrained(K, F, fixed_dofs=(), fixed_values=()):
    """
    Solve K U = F with specified constrained DOFs.

    Parameters
    ----------
    K : (n, n) array
        Global stiffness matrix.
    F : (n,) array
        Global force vector.
    fixed_dofs : sequence of int
        DOFs that are prescribed.
    fixed_values : sequence of float
        Prescribed DOF values, aligned with fixed_dofs.

    Returns
    -------
    U : (n,) array
        Full displacement vector.
    """
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    n = K.shape[0]
    U = np.zeros(n, dtype=float)

    fixed = np.asarray(list(fixed_dofs), dtype=int)
    if fixed.size > 0:
        fixed = np.unique(fixed)
        vals = np.asarray(list(fixed_values), dtype=float).reshape(-1)
        if vals.size == 1:
            vals = np.full(fixed.size, vals[0], dtype=float)
        if vals.size != fixed.size:
            raise ValueError("fixed_values must be aligned with fixed_dofs.")

        free = np.setdiff1d(np.arange(n), fixed, assume_unique=False)
        if free.size > 0:
            Kff = K[np.ix_(free, free)]
            reg = 1.0e-12 * max(1.0, np.trace(Kff) / max(1, Kff.shape[0]))
            Kff = Kff + reg * np.eye(free.size, dtype=float)
            Ff = F[free].copy()
            if fixed.size > 0:
                Kcf = K[np.ix_(free, fixed)]
                Ff = Ff - Kcf @ vals
            U[free] = safe_solve(Kff, Ff)

        U[fixed] = vals
    return U


def generalized_frequencies(K, M):
    """
    Solve the generalized eigenproblem K phi = lambda M phi.

    Returns
    -------
    freqs : array
        Natural frequencies in Hz.
    modes : array
        Right eigenvectors associated with the retained modes.
    """
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)

    if K.size == 0 or M.size == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)

    reg = 1.0e-12 * max(1.0, np.trace(M) / max(1, M.shape[0]))
    Mreg = M + reg * np.eye(M.shape[0], dtype=float)

    try:
        vals, vecs = eig(K, Mreg)
    except Exception:
        A = np.linalg.inv(Mreg) @ K
        vals, vecs = np.linalg.eig(A)

    vals = np.asarray(vals, dtype=complex)
    vecs = np.asarray(vecs, dtype=complex)

    real_ok = np.abs(vals.imag) < 1.0e-5 * np.maximum(1.0, np.abs(vals.real))
    positive = vals.real > 1.0e-10
    mask = real_ok & positive & np.isfinite(vals.real)

    vals = vals.real[mask]
    vecs = vecs[:, mask]

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    freqs = np.sqrt(vals) / (2.0 * pi)
    norms = np.linalg.norm(vecs, axis=0)
    norms = np.where(norms < EPS, 1.0, norms)
    modes = (vecs / norms).real

    return freqs, modes


def gauss_points_2d(n=2):
    """Return 2D Gauss-Legendre quadrature points and weights on [-1, 1]^2."""
    pts, ws = np.polynomial.legendre.leggauss(n)
    out = []
    for xi, wxi in zip(pts, ws):
        for eta, weta in zip(pts, ws):
            out.append((float(xi), float(eta), float(wxi * weta)))
    return np.asarray(out, dtype=float)


class MohrCircle:
    """Mohr's circle for a 2D plane-stress state."""

    def __init__(self, sigma_x, sigma_y, tau_xy):
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.tau_xy = float(tau_xy)

        self.center = 0.5 * (self.sigma_x + self.sigma_y)
        self.radius = sqrt(((self.sigma_x - self.sigma_y) / 2.0) ** 2 + self.tau_xy ** 2)
        self.theta_p = 0.5 * np.arctan2(2.0 * self.tau_xy, self.sigma_x - self.sigma_y)

    def principal_stresses(self):
        return self.center + self.radius, self.center - self.radius

    def max_shear_stress(self):
        return self.radius

    def transform(self, theta):
        """
        Transform stress components to axes rotated by theta.

        Sign convention:
        sigma_x' = ... + tau_xy sin(2theta)
        tau_x'y' = -((sigma_x - sigma_y)/2) sin(2theta) + tau_xy cos(2theta)
        """
        theta = float(theta)
        s2 = 2.0 * theta
        sigma_x_p = (
            0.5 * (self.sigma_x + self.sigma_y)
            + 0.5 * (self.sigma_x - self.sigma_y) * np.cos(s2)
            + self.tau_xy * np.sin(s2)
        )
        sigma_y_p = (
            0.5 * (self.sigma_x + self.sigma_y)
            - 0.5 * (self.sigma_x - self.sigma_y) * np.cos(s2)
            - self.tau_xy * np.sin(s2)
        )
        tau_x_y_p = (
            -0.5 * (self.sigma_x - self.sigma_y) * np.sin(s2)
            + self.tau_xy * np.cos(s2)
        )
        return sigma_x_p, sigma_y_p, tau_x_y_p


class StressState:
    """General stress state with failure-theory and transformation utilities."""

    def __init__(self, sigma):
        sigma = np.asarray(sigma, dtype=float)

        if sigma.shape == (3,):
            sx, sy, tau = sigma
            self.matrix = np.array(
                [
                    [sx, tau, 0.0],
                    [tau, sy, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            self.is_plane = True
        elif sigma.shape == (2, 2):
            sx, sy, tau = sigma[0, 0], sigma[1, 1], sigma[0, 1]
            self.matrix = np.array(
                [
                    [sx, tau, 0.0],
                    [tau, sy, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            self.is_plane = True
        elif sigma.shape == (3, 3):
            self.matrix = sigma.copy()
            self.is_plane = False
        else:
            raise ValueError("StressState expects (3,), (2,2), or (3,3) input.")

        self.sx = self.matrix[0, 0]
        self.sy = self.matrix[1, 1]
        self.tau_xy = self.matrix[0, 1]

    def principal_stresses(self):
        """Return principal stresses sorted from maximum to minimum."""
        vals = np.linalg.eigvalsh(self.matrix)
        return vals[::-1].copy()

    def von_mises(self):
        s = self.principal_stresses()
        return sqrt(
            0.5 * (
                (s[0] - s[1]) ** 2
                + (s[1] - s[2]) ** 2
                + (s[2] - s[0]) ** 2
            )
        )

    def tresca(self):
        s = self.principal_stresses()
        return s[0] - s[-1]

    def max_shear(self):
        s = self.principal_stresses()
        return 0.5 * (s[0] - s[-1])

    def transform(self, theta):
        """Return transformed plane stress components at angle theta."""
        if not self.is_plane:
            raise ValueError("Transformation method is defined for plane stress states.")
        mc = MohrCircle(self.sx, self.sy, self.tau_xy)
        return mc.transform(theta)


class FailureTheories:
    """Static failure theory checks."""

    @staticmethod
    def von_mises(stress):
        return StressState(stress).von_mises()

    @staticmethod
    def tresca(stress):
        return StressState(stress).tresca()

    @staticmethod
    def max_shear(stress):
        return StressState(stress).max_shear()

    @staticmethod
    def maximum_principal_stress(stress):
        return StressState(stress).principal_stresses()[0]


class SNCurve:
    """
    Simplified S-N curve using a straight log-log line between a fatigue strength
    at N0 and the endurance limit Se at Nf.

    For N >= Nf, the stress amplitude is constant at Se.
    """

    def __init__(self, Se=350.0e6, Sut=600.0e6, N0=1.0e3, Nf=1.0e6):
        self.Se = float(Se)
        self.Sut = float(Sut)
        self.N0 = float(N0)
        self.Nf = float(Nf)

        S0 = 0.9 * self.Sut
        if S0 <= self.Se:
            S0 = 1.2 * self.Se

        self.S0 = S0
        if self.Nf <= self.N0:
            raise ValueError("Nf must be greater than N0.")

        self.b = np.log(self.Se / self.S0) / np.log(self.Nf / self.N0)

    def stress_at_cycles(self, N):
        """Return completely reversed stress amplitude for N cycles to failure."""
        N = float(N)
        if N <= self.N0:
            return max(self.S0, self.Se)
        if N >= self.Nf:
            return self.Se
        return self.S0 * (N / self.N0) ** self.b

    def cycles_to_failure(self, S):
        """Return cycles to failure for amplitude S."""
        S = float(S)
        if S <= self.Se:
            return np.inf
        if S >= self.S0:
            return self.N0
        return self.Nf * (S / self.Se) ** (1.0 / self.b)


class GoodmanCorrection:
    """Mean-stress correction for finite-life or infinite-life S-N use."""

    def __init__(self, method="goodman"):
        self.method = method.lower()
        if self.method not in {"goodman", "gerber"}:
            raise ValueError("method must be 'goodman' or 'gerber'.")

    def allowable_amplitude(self, Sm, Se, Sut):
        """
        Allowable completely reversed amplitude for a tensile mean stress Sm.
        Compressives mean stresses are typically not a concern for this simple model.
        """
        Sm = float(max(0.0, Sm))
        Se = float(Se)
        Sut = float(Sut)

        if Sut <= 0.0:
            raise ValueError("Sut must be positive.")
        if Sm >= Sut:
            return 0.0

        if self.method == "goodman":
            return Se * (1.0 - Sm / Sut)
        return Se * sqrt(max(0.0, 1.0 - (Sm / Sut) ** 2))


class MinerRule:
    """Linear cumulative damage rule."""

    def __init__(self):
        self.damage = 0.0
        self.history = []

    def add_block(self, cycles, cycles_to_failure):
        cycles = float(cycles)
        ctf = float(cycles_to_failure)
        if ctf <= 0.0:
            return self.damage
        if not np.isfinite(ctf):
            return self.damage

        incremental = cycles / ctf
        self.damage += incremental
        self.history.append((cycles, ctf, incremental))
        return self.damage

    def is_failed(self, limit=1.0):
        return self.damage >= limit


class ParisLaw:
    """
    Paris-Erdogan crack growth law:
        da/dN = C (Delta K - Delta K_th)^m
    """

    def __init__(self, C=1.0e-9, m=3.0, Delta_K_threshold=1.0e-4):
        self.C = float(C)
        self.m = float(m)
        self.Delta_K_threshold = float(Delta_K_threshold)

    def stress_intensity_factor(self, delta_sigma, a, geometry=1.12):
        """Mode I stress intensity factor for a crack of length a under delta_sigma."""
        a = float(a)
        delta_sigma = float(delta_sigma)
        geometry = float(geometry)
        return geometry * delta_sigma * sqrt(pi * a)

    def da_dn(self, a, delta_sigma, geometry=1.12):
        delta_K = self.stress_intensity_factor(delta_sigma, a, geometry)
        driving = delta_K - self.Delta_K_threshold
        if driving <= 0.0:
            return 0.0
        return self.C * driving ** self.m

    def cycles_to_failure(self, a0, acritical, delta_sigma, geometry=1.12):
        """Integrate crack growth from a0 to acritical."""
        a0 = float(a0)
        acritical = float(acritical)

        if acritical <= a0:
            return 0.0
        if self.da_dn(a0, delta_sigma, geometry) <= EPS:
            return np.inf

        def integrand(a):
            return 1.0 / self.da_dn(a, delta_sigma, geometry)

        value, _ = quad(integrand, a0, acritical, limit=200)
        return max(0.0, value)


class FractureMechanics:
    """Basic linear elastic fracture mechanics utilities."""

    def __init__(self, K_IC=None):
        self.K_IC = K_IC

    def stress_intensity_factor(self, sigma, a, geometry=1.12):
        return geometry * float(sigma) * sqrt(pi * float(a))

    def critical_stress(self, a, K_IC=None, geometry=1.12):
        K_IC = self.K_IC if K_IC is None else K_IC
        if K_IC is None:
            raise ValueError("K_IC is required for critical_stress.")
        return K_IC / (float(geometry) * sqrt(pi * float(a)))

    def j_integral(self, K, E, nu=0.3, plane_strain=True):
        K = float(K)
        E = float(E)
        if plane_strain:
            return K ** 2 * (1.0 - nu ** 2) / E
        return K ** 2 / E


class SDOFDampedSystem:
    """Single-degree-of-freedom damped system."""

    def __init__(self, m, k, c=0.0):
        if m <= 0.0 or k <= 0.0:
            raise ValueError("m and k must be positive.")
        self.m = float(m)
        self.k = float(k)
        self.c = float(c)

        self.omega_n = sqrt(self.k / self.m)
        self.zeta = self.c / (2.0 * sqrt(self.m * self.k))
        self.omega_d = self.omega_n * sqrt(max(0.0, 1.0 - self.zeta ** 2))
        self.f_n = self.omega_n / (2.0 * pi)

    def frequency_response(self, omega, F0=1.0):
        """
        Complex frequency response H(omega) for a harmonic force F0 cos(omega t).
        Returns real, imaginary, and magnitude.
        """
        omega = float(omega)
        F0 = complex(F0)
        H = F0 / (self.k - self.m * omega ** 2 + 1j * self.c * omega)
        return H.real, H.imag, abs(H)

    def time_response(self, F0=0.0, omega=0.0, x0=0.0, v0=0.0, t_end=5.0, n=500):
        """Time response to F(t) = F0 cos(omega t)."""
        t_eval = np.linspace(0.0, float(t_end), int(n))

        def rhs(t, y):
            x, v = y
            F = float(F0) * np.cos(omega * t)
            ax = (-self.k * x - self.c * v + F) / self.m
            return [v, ax]

        sol = solve_ivp(
            rhs,
            (0.0, float(t_end)),
            [float(x0), float(v0)],
            t_eval=t_eval,
            method="RK45",
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        return sol.t, sol.y


class MDOFSystem:
    """Mass-spring-damper MDOF system in Cartesian coordinates."""

    def __init__(self, K, M, C=None):
        self.K = np.asarray(K, dtype=float)
        self.M = np.asarray(M, dtype=float)
        n = self.M.shape[0]

        if C is None:
            self.C = np.zeros((n, n), dtype=float)
        else:
            self.C = np.asarray(C, dtype=float)

        if not (self.K.shape == (n, n) and self.C.shape == (n, n)):
            raise ValueError("K, M, and C must have compatible square dimensions.")

    def natural_frequencies(self):
        """Natural frequencies and mode shapes."""
        return generalized_frequencies(self.K, self.M)

    def state_matrix(self):
        """First-order state matrix A for state [x; v]."""
        n = self.M.shape[0]
        I = np.eye(n, dtype=float)

        reg = 1.0e-12 * max(1.0, np.trace(self.M) / n)
        Mreg = self.M + reg * I

        M_inv_K = safe_solve(Mreg, self.K)
        M_inv_C = safe_solve(Mreg, self.C)

        A = np.block(
            [
                [np.zeros((n, n), dtype=float), I],
                [-M_inv_K, -M_inv_C],
            ]
        )
        return A

    def time_response(self, f0=None, omega=0.0, x0=None, v0=None, t_end=5.0, n=500):
        """
        Time response to F(t) = f0 cos(omega t).
        f0 may be a vector.
        """
        n = self.M.shape[0]

        if x0 is None:
            x0 = np.zeros(n, dtype=float)
        if v0 is None:
            v0 = np.zeros(n, dtype=float)
        if f0 is None:
            f0 = np.zeros(n, dtype=float)
        else:
            f0 = np.asarray(f0, dtype=float).reshape(-1)
            if f0.size == 1:
                f0 = np.full(n, float(f0[0]), dtype=float)
            if f0.size != n:
                raise ValueError("f0 must be scalar or length equal to number of DOFs.")

        A = self.state_matrix()
        M_inv = safe_solve(self.M, np.eye(n, dtype=float))
        B = np.block(
            [
                np.zeros((n, n), dtype=float),
                M_inv,
            ]
        )

        def rhs(t, y):
            F = f0 * np.cos(omega * t)
            return A @ y + B @ F

        sol = solve_ivp(
            rhs,
            (0.0, float(t_end)),
            np.concatenate([np.asarray(x0, dtype=float), np.asarray(v0, dtype=float)]),
            t_eval=np.linspace(0.0, float(t_end), int(n)),
            method="RK45",
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        return sol.t, sol.y


class StateSpaceSystem:
    """Continuous-time state-space system x' = A x + B u, y = C x + D u."""

    def __init__(self, A, B=None, C=None, D=None):
        self.A = np.asarray(A, dtype=float)
        n = self.A.shape[0]

        if B is None:
            self.B = np.zeros((n, 1), dtype=float)
        else:
            self.B = np.asarray(B, dtype=float)

        if C is None:
            self.C = np.eye(n, dtype=float)
        else:
            self.C = np.asarray(C, dtype=float)

        if D is None:
            m_in = self.B.shape[1]
            m_out = self.C.shape[0]
            self.D = np.zeros((m_out, m_in), dtype=float)
        else:
            self.D = np.asarray(D, dtype=float)

    def simulate(self, t_eval, u=None, x0=None):
        """
        Simulate the system.
        u may be None, a scalar/array of inputs for each t_eval entry, or a callable u(t).
        """
        n = self.A.shape[0]
        if x0 is None:
            x0 = np.zeros(n, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != n:
            raise ValueError("x0 length must match state dimension.")

        t_eval = np.asarray(t_eval, dtype=float)

        if u is None:
            m_in = self.B.shape[1]

            def u_func(t):
                return np.zeros(m_in, dtype=float)

        elif callable(u):
            u_func = lambda t: np.asarray(u(t), dtype=float).reshape(-1)

        else:
            u_arr = np.asarray(u, dtype=float)
            if u_arr.ndim == 1 and u_arr.size == t_eval.size:
                u_func = lambda t: np.interp(t, t_eval, u_arr)
            else:
                raise ValueError("Array input u must be a 1D array aligned with t_eval.")

        def rhs(t, y):
            return self.A @ y + self.B @ u_func(t)

        sol = solve_ivp(
            rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1.0e-8,
            atol=1.0e-10,
        )

        y_out = np.zeros((t_eval.size, self.C.shape[0]), dtype=float)
        for i, t in enumerate(t_eval):
            u_t = u_func(t)
            y_out[i] = self.C @ sol.y[:, i] + self.D @ u_t

        return sol.t, sol.y, y_out

    def frequency_response(self, omega):
        """Scalar/matrix frequency response H(jw) = C (jw I - A)^-1 B + D."""
        n = self.A.shape[0]
        I = np.eye(n, dtype=float)
        M = 1j * float(omega) * I - self.A
        return self.C @ np.linalg.inv(M) @ self.B + self.D


class JeffcottRotor:
    """
    Simple Jeffcott rotor model with lumped disk mass, bearing stiffness,
    viscous damping, and gyroscopic coupling.

    Equations:
        m x'' + c x' + G Omega y' + k x = 0
        m y'' + c y' - G Omega x' + k y = 0
    """

    def __init__(self, m, k, c=0.0, G=0.0):
        if m <= 0.0 or k <= 0.0:
            raise ValueError("m and k must be positive.")
        self.m = float(m)
        self.k = float(k)
        self.c = float(c)
        self.G = float(G)

    def state_matrix(self, omega):
        m = self.m
        k = self.k
        c = self.c
        G = self.G
        Om = float(omega)

        A = np.zeros((4, 4), dtype=float)
        A[0, 1] = 1.0
        A[2, 3] = 1.0

        A[1, 0] = -k / m
        A[1, 1] = -c / m
        A[1, 2] = -G * Om / m

        A[3, 0] = G * Om / m
        A[3, 2] = -k / m
        A[3, 3] = -c / m

        return A

    def eigenvalues(self, omega):
        return np.linalg.eigvals(self.state_matrix(omega))

    def campbell(self, speeds):
        """Return speeds and complex eigenvalues for Campbell diagram data."""
        speeds = np.asarray(speeds, dtype=float)
        values = np.array([self.eigenvalues(sp) for sp in speeds], dtype=complex)
        return speeds, values

    def critical_speed(self):
        """
        Approximate critical speed.
        For zero gyroscopic coupling, the undamped natural frequency is the critical speed.
        """
        if abs(self.G) <= EPS:
            return sqrt(self.k / self.m) / (2.0 * pi)

        base = sqrt(self.k / self.m)
        speeds = np.linspace(0.0, 4.0 * base, 200)
        imag_mags = np.array(
            [np.min(np.abs(self.eigenvalues(sp).imag)) for sp in speeds]
        )
        idx = int(np.argmin(imag_mags))
        return float(speeds[idx]) / (2.0 * pi)


class HertzianContact:
    """Hertzian contact of a sphere on a plane."""

    def __init__(self, E=200.0e9, nu=0.3, R=0.01):
        self.E = float(E)
        self.nu = float(nu)
        self.R = float(R)
        self.E_star = self.E / (1.0 - self.nu ** 2)

    def contact_radius(self, F):
        F = float(F)
        if F <= 0.0:
            return 0.0
        return (3.0 * F * self.R / (4.0 * self.E_star)) ** (1.0 / 3.0)

    def contact_area(self, F):
        a = self.contact_radius(F)
        return pi * a ** 2

    def max_pressure(self, F):
        F = float(F)
        if F <= 0.0:
            return 0.0
        a = self.contact_radius(F)
        return 2.0 * F / (pi * a ** 2)


class TrussElement2D:
    """2D pin-jointed truss element."""

    def __init__(self, node_i, node_j, E, A):
        self.node_i = int(node_i)
        self.node_j = int(node_j)
        self.E = float(E)
        self.A = float(A)

    def length(self, coords):
        return float(np.linalg.norm(coords[self.node_j] - coords[self.node_i]))

    def stiffness(self, coords):
        i, j = self.node_i, self.node_j
        p_i = coords[i]
        p_j = coords[j]
        L = float(np.linalg.norm(p_j - p_i))
        if L < EPS:
            raise ValueError("Zero-length truss element.")

        dx = p_j[0] - p_i[0]
        dy = p_j[1] - p_i[1]
        c = dx / L
        s = dy / L

        k = (self.E * self.A / L) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ],
            dtype=float,
        )
        return k

    def strain_stress(self, U, coords):
        i, j = self.node_i, self.node_j
        L = float(np.linalg.norm(coords[j] - coords[i]))
        c = (coords[j][0] - coords[i][0]) / L
        s = (coords[j][1] - coords[i][1]) / L

        ui = U[2 * i]
        vi = U[2 * i + 1]
        uj = U[2 * j]
        vj = U[2 * j + 1]

        elongation = (uj - ui) * c + (vj - vi) * s
        strain = elongation / L
        stress = self.E * strain
        force = self.A * stress
        return strain, stress, force


class Truss2D:
    """2D truss finite element model."""

    def __init__(self, E=200.0e9, A=1.0e-4, rho=7850.0):
        self.nodes = []
        self.elements = []
        self.E = float(E)
        self.A = float(A)
        self.rho = float(rho)
        self.fixed = {}
        self.forces = {}

    def add_node(self, x, y):
        self.nodes.append([float(x), float(y)])
        return len(self.nodes) - 1

    def add_element(self, i, j, E=None, A=None):
        self.elements.append(
            TrussElement2D(i, j, self.E if E is None else float(E), self.A if A is None else float(A))
        )

    def apply_displacement(self, dof, value):
        self.fixed[int(dof)] = float(value)

    def apply_force(self, dof, value):
        dof = int(dof)
        self.forces[dof] = self.forces.get(dof, 0.0) + float(value)

    def _assemble(self):
        coords = np.asarray(self.nodes, dtype=float)
        n = 2 * len(coords)
        K = np.zeros((n, n), dtype=float)
        M = np.zeros((n, n), dtype=float)

        for el in self.elements:
            i, j = el.node_i, el.node_j
            L = el.length(coords)
            Kloc = el.stiffness(coords)
            K[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += Kloc[:2, :2]
            K[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] += Kloc[:2, 2:]
            K[2 * j: 2 * j + 2, 2 * i: 2 * i + 2] += Kloc[2:, :2]
            K[2 * j: 2 * j + 2, 2 * j: 2 * j + 2] += Kloc[2:, 2:]

            m = self.rho * el.A * L
            M[2 * i, 2 * i] += 0.5 * m
            M[2 * i + 1, 2 * i + 1] += 0.5 * m
            M[2 * j, 2 * j] += 0.5 * m
            M[2 * j + 1, 2 * j + 1] += 0.5 * m

        F = np.zeros(n, dtype=float)
        for dof, val in self.forces.items():
            F[dof] += val

        return K, M, F

    def solve(self):
        K, M, F = self._assemble()
        fixed_dofs = sorted(self.fixed.keys())
        fixed_vals = [self.fixed[dof] for dof in fixed_dofs]
        U = solve_constrained(K, F, fixed_dofs, fixed_vals)

        coords = np.asarray(self.nodes, dtype=float)
        element_results = []
        for el in self.elements:
            strain, stress, force = el.strain_stress(U, coords)
            element_results.append(
                {
                    "element": el,
                    "strain": strain,
                    "stress": stress,
                    "axial_force": force,
                }
            )
        return U, element_results

    def reactions(self, U):
        K, _, F = self._assemble()
        R = K @ U - F
        free = np.setdiff1d(np.arange(K.shape[0]), np.asarray(sorted(self.fixed.keys()), dtype=int))
        R[free] = 0.0
        return R

    def natural_frequencies(self):
        K, M, _ = self._assemble()
        fixed = np.asarray(sorted(self.fixed.keys()), dtype=int)
        free = np.setdiff1d(np.arange(K.shape[0]), fixed)
        if free.size == 0:
            return np.zeros(0, dtype=float)
        Kff = K[np.ix_(free, free)]
        Mff = M[np.ix_(free, free)]
        freqs, _ = generalized_frequencies(Kff, Mff)
        return freqs


class BeamElement2D:
    """Euler-Bernoulli 2D beam element with two translational/rotational DOFs per node."""

    def __init__(self, node_i, node_j, E, I, A=None, rho=7850.0):
        self.node_i = int(node_i)
        self.node_j = int(node_j)
        self.E = float(E)
        self.I = float(I)
        self.A = None if A is None else float(A)
        self.rho = float(rho)

    def length(self, x):
        return float(x[self.node_j] - x[self.node_i])

    def local_stiffness(self, L):
        L = float(L)
        if L <= 0.0:
            raise ValueError("Beam element length must be positive.")
        EI = self.E * self.I
        k = EI / L ** 3 * np.array(
            [
                [12.0, -6.0 * L, -12.0, 6.0 * L],
                [-6.0 * L, 4.0 * L ** 2, 6.0 * L, 2.0 * L ** 2],
                [-12.0, 6.0 * L, 12.0, -6.0 * L],
                [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2],
            ],
            dtype=float,
        )
        return k

    def local_mass_matrix(self, L):
        L = float(L)
        if self.A is None:
            return np.zeros((4, 4), dtype=float)
        m = self.rho * self.A * L
        return m / 420.0 * np.array(
            [
                [156.0, 22.0 * L, 54.0, -13.0 * L],
                [22.0 * L, 4.0 * L ** 2, 13.0 * L, -8.0 * L ** 2],
                [54.0, 13.0 * L, 156.0, -22.0 * L],
                [-13.0 * L, -8.0 * L ** 2, -22.0 * L, 4.0 * L ** 2],
            ],
            dtype=float,
        )

    def local_udl_force(self, w, L):
        """Equivalent nodal forces for a distributed load w (positive upward)."""
        return np.array(
            [
                0.5 * w * L,
                w * L ** 2 / 12.0,
                0.5 * w * L,
                -w * L ** 2 / 12.0,
            ],
            dtype=float,
        )


class Beam2D:
    """2D Euler-Bernoulli beam finite element model along x."""

    def __init__(self, E=200.0e9, I=1.0e-7, A=1.0e-4, rho=7850.0):
        self.nodes = []
        self.elements = []
        self.E = float(E)
        self.I = float(I)
        self.A = float(A)
        self.rho = float(rho)
        self.fixed = {}
        self.forces = {}
        self.udl = {}

    def add_node(self, x):
        self.nodes.append(float(x))
        return len(self.nodes) - 1

    def add_element(self, i, j, E=None, I=None, A=None):
        self.elements.append(
            BeamElement2D(
                i,
                j,
                self.E if E is None else float(E),
                self.I if I is None else float(I),
                None if A is None else float(A),
                self.rho,
            )
        )

    def apply_displacement(self, dof, value):
        self.fixed[int(dof)] = float(value)

    def apply_point_load(self, node, F):
        dof = 2 * int(node)
        self.forces[dof] = self.forces.get(dof, 0.0) + float(F)

    def apply_moment(self, node, M):
        dof = 2 * int(node) + 1
        self.forces[dof] = self.forces.get(dof, 0.0) + float(M)

    def add_udl(self, element_index, w):
        element_index = int(element_index)
        self.udl[element_index] = self.udl.get(element_index, 0.0) + float(w)

    def _assemble(self):
        x = np.asarray(self.nodes, dtype=float)
        n = 2 * len(x)
        K = np.zeros((n, n), dtype=float)
        M = np.zeros((n, n), dtype=float)

        for idx, el in enumerate(self.elements):
            i, j = el.node_i, el.node_j
            L = el.length(x)
            Kloc = el.local_stiffness(L)
            Mloc = el.local_mass_matrix(L)

            K[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += Kloc[:2, :2]
            K[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] += Kloc[:2, 2:]
            K[2 * j: 2 * j + 2, 2 * i: 2 * i + 2] += Kloc[2:, :2]
            K[2 * j: 2 * j + 2, 2 * j: 2 * j + 2] += Kloc[2:, 2:]

            M[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += Mloc[:2, :2]
            M[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] += Mloc[:2, 2:]
            M[2 * j: 2 * j + 2, 2 * i: 2 * i + 2] += Mloc[2:, :2]
            M[2 * j: 2 * j + 2, 2 * j: 2 * j + 2] += Mloc[2:, 2:]

        F = np.zeros(n, dtype=float)
        for dof, val in self.forces.items():
            F[dof] += val

        for idx, el in enumerate(self.elements):
            if idx in self.udl:
                w = self.udl[idx]
                L = el.length(x)
                f = el.local_udl_force(w, L)
                F[2 * el.node_i: 2 * el.node_i + 2] += f[:2]
                F[2 * el.node_j: 2 * el.node_j + 2] += f[2:]

        return K, M, F, x

    def solve(self):
        K, M, F, x = self._assemble()
        fixed_dofs = sorted(self.fixed.keys())
        fixed_vals = [self.fixed[dof] for dof in fixed_dofs]
        U = solve_constrained(K, F, fixed_dofs, fixed_vals)
        return U, x

    def element_moments(self, U, x):
        moments = []
        for el in self.elements:
            i, j = el.node_i, el.node_j
            L = el.length(x)
            Kloc = el.local_stiffness(L)
            u_local = np.array(
                [U[2 * i], U[2 * i + 1], U[2 * j], U[2 * j + 1]],
                dtype=float,
            )
            m_local = Kloc @ u_local
            moments.append([float(m_local[1]), float(m_local[3])])
        return np.asarray(moments, dtype=float)

    def max_bending_stress(self, U, x, c=None):
        """Maximum absolute bending stress using sigma = M c / I."""
        if c is None:
            c = 0.0
        moments = self.element_moments(U, x)
        if moments.size == 0:
            return 0.0
        return float(np.max(np.abs(moments)) * c / self.I)

    def natural_frequencies(self):
        K, M, _, _ = self._assemble()
        fixed = np.asarray(sorted(self.fixed.keys()), dtype=int)
        free = np.setdiff1d(np.arange(K.shape[0]), fixed)
        if free.size == 0:
            return np.zeros(0, dtype=float)
        Kff = K[np.ix_(free, free)]
        Mff = M[np.ix_(free, free)]
        freqs, _ = generalized_frequencies(Kff, Mff)
        return freqs


class PlaneStressQuadElement:
    """4-node bilinear quadrilateral plane-stress element."""

    @staticmethod
    def shape_functions(xi, eta):
        N = 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_dxi = 0.25 * np.array(
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            dtype=float,
        )
        dN_deta = 0.25 * np.array(
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
            dtype=float,
        )
        return N, dN_dxi, dN_deta

    def __init__(self, nodes, E, nu, t):
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.shape != (4, 2):
            raise ValueError("PlaneStressQuadElement expects 4 (x, y) nodes.")
        self.E = float(E)
        self.nu = float(nu)
        self.t = float(t)

        if self.nu >= 1.0 or self.nu <= -1.0:
            raise ValueError("Poisson ratio must be in (-1, 1).")

        self.D = (self.E / (1.0 - self.nu ** 2)) * np.array(
            [
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self.nu)],
            ],
            dtype=float,
        )

    def _b_matrix(self, xi, eta):
        _, dN_dxi, dN_deta = self.shape_functions(xi, eta)
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]

        J = np.array(
            [
                [float(np.dot(dN_dxi, x)), float(np.dot(dN_deta, x))],
                [float(np.dot(dN_dxi, y)), float(np.dot(dN_deta, y))],
            ],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if abs(detJ) < EPS:
            return np.zeros((3, 8), dtype=float)

        JinvT = np.linalg.inv(J).T
        dN_dx = JinvT[0, 0] * dN_dxi + JinvT[0, 1] * dN_deta
        dN_dy = JinvT[1, 0] * dN_dxi + JinvT[1, 1] * dN_deta

        B = np.zeros((3, 8), dtype=float)
        for i in range(4):
            B[0, 2 * i] = dN_dx[i]
            B[1, 2 * i + 1] = dN_dy[i]
            B[2, 2 * i] = dN_dy[i]
            B[2, 2 * i + 1] = dN_dx[i]
        return B

    def stiffness(self, n_gauss=2):
        K = np.zeros((8, 8), dtype=float)
        for xi, eta, w in gauss_points_2d(n_gauss):
            B = self._b_matrix(xi, eta)
            J = np.array(
                [
                    [
                        float(
                            0.25
                            * (
                                -1.0 * (1.0 - eta) * (self.nodes[0, 0] - self.nodes[1, 0])
                                + 1.0 * (1.0 - eta) * (self.nodes[1, 0] - self.nodes[0, 0])
                                + 1.0 * (1.0 + eta) * (self.nodes[2, 0] - self.nodes[3, 0])
                                - 1.0 * (1.0 + eta) * (self.nodes[3, 0] - self.nodes[2, 0])
                            )
                        ),
                        0.0,
                    ],
                    [0.0, 0.0],
                ]
            )
            # Simpler and reliable Jacobian determinant:
            _, dN_dxi, dN_deta = self.shape_functions(xi, eta)
            x = self.nodes[:, 0]
            y = self.nodes[:, 1]
            J = np.array(
                [
                    [float(np.dot(dN_dxi, x)), float(np.dot(dN_deta, x))],
                    [float(np.dot(dN_dxi, y)), float(np.dot(dN_deta, y))],
                ],
                dtype=float,
            )
            detJ = float(np.linalg.det(J))
            if abs(detJ) < EPS:
                continue
            K += w * detJ * (B.T @ self.D @ B)
        return K

    def strain_stress(self, U_local):
        B = self._b_matrix(0.0, 0.0)
        strain = B @ U_local
        stress = self.D @ strain
        return strain, stress


class PlatePlaneStress2D:
    """Thin in-plane plate model using 4-node plane-stress quadrilaterals."""

    def __init__(self, E=70.0e9, nu=0.33, t=0.005):
        self.nodes = []
        self.elements = []
        self.E = float(E)
        self.nu = float(nu)
        self.t = float(t)
        self.fixed = {}
        self.forces = {}

    def add_node(self, x, y):
        self.nodes.append([float(x), float(y)])
        return len(self.nodes) - 1

    def add_element(self, n1, n2, n3, n4):
        self.elements.append((int(n1), int(n2), int(n3), int(n4)))

    def apply_displacement(self, dof, value):
        self.fixed[int(dof)] = float(value)

    def apply_force(self, dof, value):
        dof = int(dof)
        self.forces[dof] = self.forces.get(dof, 0.0) + float(value)

    def _assemble(self):
        n = 2 * len(self.nodes)
        K = np.zeros((n, n), dtype=float)

        for n1, n2, n3, n4 in self.elements:
            nodes = [
                self.nodes[n1],
                self.nodes[n2],
                self.nodes[n3],
                self.nodes[n4],
            ]
            el = PlaneStressQuadElement(nodes, self.E, self.nu, self.t)
            Kloc = el.stiffness(n_gauss=2)
            dofs = [2 * k + d for k in (n1, n2, n3, n4) for d in (0, 1)]
            for a in range(8):
                for b in range(8):
                    K[dofs[a], dofs[b]] += Kloc[a, b]

        F = np.zeros(n, dtype=float)
        for dof, val in self.forces.items():
            F[dof] += val
        return K, F

    def solve(self):
        K, F = self._assemble()
        fixed_dofs = sorted(self.fixed.keys())
        fixed_vals = [self.fixed[dof] for dof in fixed_dofs]
        U = solve_constrained(K, F, fixed_dofs, fixed_vals)
        return U

    def element_stresses(self, U):
        out = []
        for n1, n2, n3, n4 in self.elements:
            nodes = [self.nodes[k] for k in (n1, n2, n3, n4)]
            el = PlaneStressQuadElement(nodes, self.E, self.nu, self.t)
            dofs = [2 * k + d for k in (n1, n2, n3, n4) for d in (0, 1)]
            U_local = U[dofs]
            _, stress = el.strain_stress(U_local)
            out.append(stress)
        return np.asarray(out, dtype=float)

    def max_von_mises(self, U):
        stresses = self.element_stresses(U)
        if stresses.size == 0:
            return 0.0
        values = [FailureTheories.von_mises(s[:3]) for s in stresses]
        return float(np.max(values))


class ThinPlateBendingElement:
    """
    4-node thin-plate bending element with 3 DOFs per node: [w, theta_x, theta_y].

    This implementation uses a 12-term polynomial interpolation sufficient for the
    12 nodal DOFs and evaluates stiffness by numerical quadrature. It is a practical
    demonstration element for small-deflection Kirchhoff plate bending.
    """

    _EXPS = np.array(
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (2, 1),
            (0, 2),
            (1, 2),
            (3, 0),
            (3, 1),
            (0, 3),
            (1, 3),
        ],
        dtype=int,
    )

    def __init__(self, nodes, E, nu, t):
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.shape != (4, 2):
            raise ValueError("ThinPlateBendingElement expects 4 (x, y) nodes.")

        self.E = float(E)
        self.nu = float(nu)
        self.t = float(t)

        self.xc = float(np.mean(self.nodes[:, 0]))
        self.yc = float(np.mean(self.nodes[:, 1]))
        self.hx = 0.5 * (float(np.max(self.nodes[:, 0])) - float(np.min(self.nodes[:, 0])))
        self.hy = 0.5 * (float(np.max(self.nodes[:, 1])) - float(np.min(self.nodes[:, 1])))

        if self.hx < EPS or self.hy < EPS:
            raise ValueError("Plate element must have positive x and y extents.")

        D1 = self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))
        D2 = self.nu * D1
        D3 = self.E * self.t ** 3 / (24.0 * (1.0 + self.nu))
        self.D = np.array(
            [
                [D1, D2, 0.0],
                [D2, D1, 0.0],
                [0.0, 0.0, D3],
            ],
            dtype=float,
        )

        self._build_shape_functions()
        self.K = self._stiffness()

    def _basis_value_vector(self, r, s):
        out = np.zeros(12, dtype=float)
        for idx, (a, b) in enumerate(self._EXPS):
            val_r = r ** a if a > 0 else 1.0
            val_s = s ** b if b > 0 else 1.0
            out[idx] = val_r * val_s
        return out

    def _basis_derivative_vector(self, r, s, dr, ds):
        out = np.zeros(12, dtype=float)
        for idx, (a, b) in enumerate(self._EXPS):
            if dr > a or ds > b:
                continue
            coeff = factorial(a) // factorial(a - dr)
            coeff *= factorial(b) // factorial(b - ds)
            rr = r ** (a - dr) if a - dr > 0 else 1.0
            ss = s ** (b - ds) if b - ds > 0 else 1.0
            out[idx] = coeff * rr * ss
        return out

    def _build_shape_functions(self):
        A = np.zeros((12, 12), dtype=float)

        for i, node in enumerate(self.nodes):
            r = float((node[0] - self.xc) / self.hx)
            s = float((node[1] - self.yc) / self.hy)

            A[3 * i, :] = self._basis_value_vector(r, s)
            A[3 * i + 1, :] = self._basis_derivative_vector(r, s, 1, 0) / self.hx
            A[3 * i + 2, :] = self._basis_derivative_vector(r, s, 0, 1) / self.hy

        # Each row of self._coeff holds coefficients for one nodal shape function.
        self._coeff = np.linalg.pinv(A).T

    def _shape_values(self, r, s):
        basis = self._basis_value_vector(r, s)
        return basis @ self._coeff

    def _shape_second_derivatives(self, r, s):
        rr_basis = self._basis_derivative_vector(r, s, 2, 0)
        ss_basis = self._basis_derivative_vector(r, s, 0, 2)
        rs_basis = self._basis_derivative_vector(r, s, 1, 1)

        rr = rr_basis @ self._coeff
        ss = ss_basis @ self._coeff
        rs = rs_basis @ self._coeff
        return rr, ss, rs

    def _b_matrix(self, r, s):
        rr, ss, rs = self._shape_second_derivatives(r, s)
        B = np.zeros((3, 12), dtype=float)
        B[0, :] = rr / (self.hx ** 2)
        B[1, :] = ss / (self.hy ** 2)
        B[2, :] = 2.0 * rs / (self.hx * self.hy)
        return B

    def _stiffness(self):
        K = np.zeros((12, 12), dtype=float)
        for r, s, w in gauss_points_2d(4):
            B = self._b_matrix(r, s)
            K += w * self.hx * self.hy * (B.T @ self.D @ B)
        return 0.5 * (K + K.T)

    def moments(self, U_local):
        B = self._b_matrix(0.0, 0.0)
        curvature = B @ U_local
        return self.D @ curvature


class ThinPlateBending2D:
    """Thin-plate bending finite element model using ThinPlateBendingElement."""

    def __init__(self, E=200.0e9, nu=0.3, t=0.01):
        self.nodes = []
        self.elements = []
        self.E = float(E)
        self.nu = float(nu)
        self.t = float(t)
        self.fixed = {}
        self.forces = {}

    def add_node(self, x, y):
        self.nodes.append([float(x), float(y)])
        return len(self.nodes) - 1

    def add_element(self, n1, n2, n3, n4):
        self.elements.append([int(n1), int(n2), int(n3), int(n4)])

    def apply_displacement(self, dof, value):
        self.fixed[int(dof)] = float(value)

    def apply_force(self, dof, value):
        dof = int(dof)
        self.forces[dof] = self.forces.get(dof, 0.0) + float(value)

    def _assemble(self):
        n = 3 * len(self.nodes)
        K = np.zeros((n, n), dtype=float)

        for el_nodes in self.elements:
            nodes = [self.nodes[k] for k in el_nodes]
            el = ThinPlateBendingElement(nodes, self.E, self.nu, self.t)
            Kloc = el.K
            dofs = []
            for node in el_nodes:
                dofs.extend([3 * node + 0, 3 * node + 1, 3 * node + 2])
            for a in range(12):
                for b in range(12):
                    K[dofs[a], dofs[b]] += Kloc[a, b]

        F = np.zeros(n, dtype=float)
        for dof, val in self.forces.items():
            F[dof] += val
        return K, F

    def solve(self):
        K, F = self._assemble()
        fixed_dofs = sorted(self.fixed.keys())
        fixed_vals = [self.fixed[dof] for dof in fixed_dofs]
        U = solve_constrained(K, F, fixed_dofs, fixed_vals)
        return U

    def element_moments(self, U):
        moments = []
        for el_nodes in self.elements:
            nodes = [self.nodes[k] for k in el_nodes]
            el = ThinPlateBendingElement(nodes, self.E, self.nu, self.t)
            dofs = []
            for node in el_nodes:
                dofs.extend([3 * node + 0, 3 * node + 1, 3 * node + 2])
            moments.append(el.moments(U[dofs]))
        return np.asarray(moments, dtype=float)

    def max_moment(self, U):
        moments = self.element_moments(U)
        if moments.size == 0:
            return 0.0
        return float(np.max(np.abs(moments)))


class BeamWeightOptimizer:
    """
    Rectangular cantilever beam cross-section optimization.

    Objective: minimize mass = rho * L * b * h.
    Constraints:
        sigma_max = 6 M / (b h^2) <= sigma_allow
        delta_max = P L^3 / (3 E I) <= delta_allow
    """

    def __init__(
        self,
        E=200.0e9,
        rho=7850.0,
        L=1.0,
        M=500.0,
        P=1000.0,
        sigma_allow=200.0e6,
        delta_allow=5.0e-3,
        bounds=((0.02, 0.20), (0.05, 0.40)),
    ):
        self.E = float(E)
        self.rho = float(rho)
        self.L = float(L)
        self.M = float(M)
        self.P = float(P)
        self.sigma_allow = float(sigma_allow)
        self.delta_allow = float(delta_allow)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)

    def metrics(self, x):
        b, h = np.asarray(x, dtype=float).reshape(-1)
        b = float(b)
        h = float(h)
        if b <= 0.0 or h <= 0.0:
            raise ValueError("Beam dimensions must be positive.")

        I = b * h ** 3 / 12.0
        A = b * h
        mass = self.rho * self.L * A
        sigma = 6.0 * self.M / (b * h ** 2)
        delta = self.P * self.L ** 3 / (3.0 * self.E * I)
        return mass, I, sigma, delta

    def objective(self, x):
        return self.metrics(x)[0]

    def _constraints(self, x):
        _, _, sigma, delta = self.metrics(x)
        return [
            {"type": "ineq", "fun": lambda x, s=sigma: self.sigma_allow - s},
            {"type": "ineq", "fun": lambda x, d=delta: self.delta_allow - d},
        ]

    def gradient_optimize(self, x0=None):
        if x0 is None:
            b0 = 0.5 * (self.bounds[0][0] + self.bounds[0][1])
            h0 = 0.5 * (self.bounds[1][0] + self.bounds[1][1])
            x0 = [b0, h0]

        res = minimize(
            self.objective,
            x0,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._constraints,
            options={"ftol": 1.0e-12, "maxiter": 200},
        )
        x = res.x
        mass, I, sigma, delta = self.metrics(x)
        return {
            "success": bool(res.success),
            "x": x,
            "b": float(x[0]),
            "h": float(x[1]),
            "mass": float(mass),
            "I": float(I),
            "sigma": float(sigma),
            "delta": float(delta),
            "objective": float(res.fun),
        }

    def penalty_objective(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        b, h = x
        if b <= 0.0 or h <= 0.0:
            return 1.0e12
        mass, _, sigma, delta = self.metrics(x)
        penalty = 0.0
        if sigma > self.sigma_allow:
            penalty += 1.0e8 * ((sigma / self.sigma_allow - 1.0) ** 2)
        if delta > self.delta_allow:
            penalty += 1.0e8 * ((delta / self.delta_allow - 1.0) ** 2)
        return mass + penalty

    def genetic_optimize(self, population=40, generations=80, seed=1):
        ga = GeneticAlgorithm(
            bounds=self.bounds,
            population=population,
            generations=generations,
            seed=seed,
        )
        best_x, best_val = ga.run(self.penalty_objective)
        mass, I, sigma, delta = self.metrics(best_x)
        return {
            "x": best_x,
            "b": float(best_x[0]),
            "h": float(best_x[1]),
            "mass": float(mass),
            "I": float(I),
            "sigma": float(sigma),
            "delta": float(delta),
            "penalty_objective": float(best_val),
        }


class GeneticAlgorithm:
    """Simple real-coded genetic algorithm with tournament selection."""

    def __init__(
        self,
        bounds,
        population=30,
        generations=50,
        elite=2,
        mutation_rate=0.25,
        mutation_scale=0.05,
        seed=None,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.population = int(population)
        self.generations = int(generations)
        self.elite = int(elite)
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.rng = np.random.default_rng(seed)

        if self.population < 4:
            raise ValueError("Population size must be at least 4.")
        if self.elite >= self.population:
            self.elite = 1

    def _clip(self, x):
        return np.clip(np.asarray(x, dtype=float), self.low, self.high)

    def _random_point(self):
        span = self.high - self.low
        return self._clip(self.low + self.rng.random(self.low.size) * span)

    def _crossover(self, a, b):
        alpha = float(self.rng.random())
        return self._clip(alpha * a + (1.0 - alpha) * b)

    def _mutate(self, x):
        span = self.high - self.low
        std = self.mutation_scale * span
        std = np.where(std > EPS, std, 1.0e-6)
        return self._clip(x + self.rng.normal(0.0, 1.0, x.size) * std)

    def run(self, objective):
        pop = [self._random_point() for _ in range(self.population)]

        for _ in range(self.generations):
            scores = np.array([objective(x) for x in pop], dtype=float)
            order = np.argsort(scores)
            elite = [pop[i].copy() for i in order[: self.elite]]

            new_pop = elite
            while len(new_pop) < self.population:
                i = int(self.rng.integers(0, self.population))
                j = int(self.rng.integers(0, self.population))

                p1 = pop[i] if scores[i] <= scores[j] else pop[j]
                p2 = pop[i] if self.rng.random() < 0.5 else pop[j]

                child = self._crossover(p1, p2)
                if self.rng.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_pop.append(child)

            pop = new_pop

        scores = np.array([objective(x) for x in pop], dtype=float)
        best_idx = int(np.argmin(scores))
        return pop[best_idx], float(scores[best_idx])


def demonstrate_mohr_and_failures():
    """Stress transformations and failure theories."""
    print("\n" + "=" * 80)
    print("Stress Transformations and Failure Theories")
    print("=" * 80)

    stress = StressState([120.0e6, 40.0e6, -35.0e6])
    mc = MohrCircle(stress.sx, stress.sy, stress.tau_xy)

    sigma1, sigma2 = mc.principal_stresses()
    print(f"Plane stress state: sigma_x = {stress.sx / 1e6:.1f} MPa, "
          f"sigma_y = {stress.sy / 1e6:.1f} MPa, tau_xy = {stress.tau_xy / 1e6:.1f} MPa")
    print(f"Principal stresses: sigma_1 = {sigma1 / 1e6:.2f} MPa, "
          f"sigma_2 = {sigma2 / 1e6:.2f} MPa")
    print(f"Maximum shear stress: {mc.max_shear_stress() / 1e6:.2f} MPa")
    print(f"Principal plane angle theta_p = {np.degrees(mc.theta_p):.2f} deg")

    sxp, syp, taup = mc.transform(np.radians(30.0))
    print(f"Transformed at 30 deg: sigma_x' = {sxp / 1e6:.2f} MPa, "
          f"sigma_y' = {syp / 1e6:.2f} MPa, tau_x'y' = {taup / 1e6:.2f} MPa")

    print(f"Von Mises stress: {stress.von_mises() / 1e6:.2f} MPa")
    print(f"Tresca stress:    {stress.tresca() / 1e6:.2f} MPa")
    print(f"Max shear stress:  {stress.max_shear() / 1e6:.2f} MPa")


def demonstrate_vibrations():
    """SDOF, MDOF, and state-space demonstrations."""
    print("\n" + "=" * 80)
    print("Mechanical Vibrations")
    print("=" * 80)

    sdo = SDOFDampedSystem(m=10.0, k=4000.0, c=20.0)
    print(
        f"SDOF: f_n = {sdo.f_n:.4f} Hz, zeta = {sdo.zeta:.4f}, "
        f"f_d = {sdo.omega_d / (2.0 * pi):.4f} Hz"
    )

    omega = 1.5 * sdo.omega_n
    _, _, magnitude = sdo.frequency_response(omega, F0=100.0)
    print(f"SDOF frequency response magnitude at 1.5*f_n: {magnitude:.4f} m/N")

    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    K = np.array([[2.0, -1.0], [-1.0, 1.0]], dtype=float)
    C = 0.05 * (np.sqrt(np.diag(K)) @ np.diag(np.sqrt(np.diag(M))))
    mdo = MDOFSystem(K, M, C)
    freqs, _ = mdo.natural_frequencies()
    print("MDOF natural frequencies [Hz]:", np.round(freqs, 4))

    t, y = mdo.time_response(f0=[100.0, 0.0], omega=0.8 * freqs[0] * 2.0 * pi,
                             x0=[0.0, 0.0], v0=[0.0, 0.0], t_end=10.0, n=200)
    print(
        f"MDOF forced response final displacement: "
        f"[{y[-1, 0]:.4f}, {y[-1, 1]:.4f}] m at t = {t[-1]:.1f} s"
    )

    A = np.array([[0.0, 1.0], [-4000.0 / 10.0, -20.0 / 10.0]], dtype=float)
    B = np.array([[0.0], [1.0 / 10.0]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)
    D = np.zeros((1, 1), dtype=float)
    ss = StateSpaceSystem(A, B, C, D)
    t_eval = np.linspace(0.0, 5.0, 100)
    _, state, out = ss.simulate(t_eval, u=None, x0=[1.0, 0.0])
    print(f"State-space free response initial x = 1.0, final x = {out[-1, 0]:.6f}")


def demonstrate_fatigue_fracture():
    """Fatigue, mean-stress correction, Miner rule, and Paris crack growth."""
    print("\n" + "=" * 80)
    print("Fatigue and Fracture Mechanics")
    print("=" * 80)

    sn = SNCurve(Se=350.0e6, Sut=600.0e6, N0=1.0e3, Nf=1.0e6)
    S = 400.0e6
    Nf = sn.cycles_to_failure(S)
    print(f"S-N curve: for S = {S / 1e6:.1f} MPa, estimated cycles to failure = {Nf:.3e}")

    goodman = GoodmanCorrection("goodman")
    gerber = GoodmanCorrection("gerber")
    Sm = 150.0e6
    Se = 350.0e6
    Sut = 600.0e6
    print(
        f"Mean stress Sm = {Sm / 1e6:.1f} MPa: Goodman amplitude = "
        f"{goodman.allowable_amplitude(Sm, Se, Sut) / 1e6:.1f} MPa, "
        f"Gerber amplitude = {gerber.allowable_amplitude(Sm, Se, Sut) / 1e6:.1f} MPa"
    )

    miner = MinerRule()
    blocks = [(1000, 5.0e5), (3000, 2.0e5), (500, 1.0e5)]
    for cycles, ctf in blocks:
        miner.add_block(cycles, ctf)
        print(f"Miner block: {cycles:>6d} cycles, cycles_to_failure = {ctf:.3e}, "
              f"accumulated damage = {miner.damage:.4f}")
    print(f"Miner failure threshold reached: {miner.is_failed()}")

    paris = ParisLaw(C=1.0e-9, m=3.0, Delta_K_threshold=1.0e-4)
    delta_sigma = 100.0e6
    cycles = paris.cycles_to_failure(a0=1.0e-4, acritical=5.0e-3, delta_sigma=delta_sigma)
    print(f"Paris crack growth cycles from a = 0.1 mm to 5.0 mm: {cycles:.3e}")

    fm = FractureMechanics(K_IC=50.0e3)
    K = fm.stress_intensity_factor(sigma=100.0e6, a=1.0e-4)
    print(f"Mode I K for sigma = 100 MPa, a = 0.1 mm: {K / 1e3:.2f} MPa*sqrt(m)")
    print(f"J integral, plane strain: {fm.j_integral(K, E=200.0e9, nu=0.3) / 1e3:.2f} kJ/m^2")


def demonstrate_fea():
    """FEA demonstrations for truss, beam, plane-stress plate, and thin-plate bending."""
    print("\n" + "=" * 80)
    print("Finite Element Analysis")
    print("=" * 80)

    # 2D truss
    print("\n--- 2D Truss ---")
    truss = Truss2D(E=200.0e9, A=1.0e-4, rho=7850.0)
    n0 = truss.add_node(0.0, 0.0)
    n1 = truss.add_node(2.0, 0.0)
    n2 = truss.add_node(1.0, 1.0)
    truss.add_element(n0, n2)
    truss.add_element(n1, n2)

    truss.apply_displacement(2 * n0 + 0, 0.0)
    truss.apply_displacement(2 * n0 + 1, 0.0)
    truss.apply_displacement(2 * n1 + 0, 0.0)
    truss.apply_displacement(2 * n1 + 1, 0.0)
    truss.apply_force(2 * n2 + 1, -10000.0)

    U, elem = truss.solve()
    print(f"Truss node 2 displacement: u = {U[2 * n2]:.6e} m, v = {U[2 * n2 + 1]:.6e} m")
    print(f"Truss max element axial stress: {np.max(np.abs([e['stress'] for e in elem])) / 1e6:.2f} MPa")
    truss_freqs = truss.natural_frequencies()
    print("Truss first natural frequency [Hz]:", f"{truss_freqs[0]:.4f}" if truss_freqs.size else "n/a")

    # 2D beam
    print("\n--- 2D Euler-Bernoulli Beam ---")
    beam = Beam2D(E=200.0e9, I=1.0e-7, A=1.0e-4, rho=7850.0)
    L = 2.0
    n_el = 4
    for i in range(n_el + 1):
        beam.add_node(i * L / n_el)
    for i in range(n_el):
        beam.add_element(i, i + 1)

    beam.apply_displacement(0, 0.0)
    beam.apply_displacement(1, 0.0)
    beam.apply_point_load(n_el, -5000.0)

    U, x = beam.solve()
    print(f"Beam tip deflection: {U[2 * n_el]:.6e} m")
    print(f"Beam tip slope: {U[2 * n_el + 1]:.6e} rad")
    c = 0.025
    print(f"Beam max bending stress: {beam.max_bending_stress(U, x, c=c) / 1e6:.2f} MPa")
    beam_freqs = beam.natural_frequencies()
    print("Beam first natural frequency [Hz]:", f"{beam_freqs[0]:.4f}" if beam_freqs.size else "n/a")

    # Plane-stress thin plate under axial load
    print("\n--- 2D Plane-Stress Plate ---")
    plate = PlatePlaneStress2D(E=70.0e9, nu=0.33, t=0.005)
    nx, ny = 4, 3
    w, h = 0.2, 0.1
    node_ids = {}
    for j in range(ny):
        for i in range(nx):
            node_ids[(i, j)] = plate.add_node(i * w / (nx - 1), j * h / (ny - 1))

    for j in range(ny - 1):
        for i in range(nx - 1):
            plate.add_element(
                node_ids[(i, j)],
                node_ids[(i + 1, j)],
                node_ids[(i + 1, j + 1)],
                node_ids[(i, j + 1)],
            )

    for j in range(ny):
        n_left = node_ids[(0, j)]
        plate.apply_displacement(2 * n_left + 0, 0.0)
        plate.apply_displacement(2 * n_left + 1, 0.0)

    F_total = 10000.0
    for j in range(ny):
        n_right = node_ids[(nx - 1, j)]
        plate.apply_force(2 * n_right + 0, F_total / ny)

    U = plate.solve()
    print(f"Plane-stress plate max von Mises stress: {plate.max_von_mises(U) / 1e6:.2f} MPa")

    # Thin-plate bending: simply supported square plate with central load
    print("\n--- Thin Plate Bending ---")
    thin = ThinPlateBending2D(E=200.0e9, nu=0.3, t=0.01)
    span = 1.0
    n = 3
    node_ids = {}
    for j in range(n):
        for i in range(n):
            node_ids[(i, j)] = thin.add_node(i * span / (n - 1), j * span / (n - 1))

    for j in range(n - 1):
        for i in range(n - 1):
            thin.add_element(
                node_ids[(i, j)],
                node_ids[(i + 1, j)],
                node_ids[(i + 1, j + 1)],
                node_ids[(i, j + 1)],
            )

    for (i, j), nid in node_ids.items():
        if i == 0 or j == 0 or i == n - 1 or j == n - 1:
            thin.apply_displacement(3 * nid + 0, 0.0)

    center = node_ids[(1, 1)]
    thin.apply_force(3 * center + 0, -1000.0)

    U = thin.solve()
    print(f"Thin plate central deflection under 1000 N: {abs(U[3 * center + 0]):.6e} m")
    print(f"Thin plate maximum bending moment: {thin.max_moment(U):.4f} N*m")


def demonstrate_optimization():
    """Beam cross-section weight minimization using gradient and GA methods."""
    print("\n" + "=" * 80)
    print("Structural Optimization")
    print("=" * 80)

    opt = BeamWeightOptimizer(
        E=200.0e9,
        rho=7850.0,
        L=1.0,
        M=500.0,
        P=1000.0,
        sigma_allow=200.0e6,
        delta_allow=5.0e-3,
        bounds=((0.02, 0.20), (0.05, 0.40)),
    )

    grad = opt.gradient_optimize()
    print("Gradient/SLSQP optimization:")
    print(
        f"  b = {grad['b'] * 1e3:.3f} mm, h = {grad['h'] * 1e3:.3f} mm, "
        f"mass = {grad['mass']:.3f} kg"
    )
    print(f"  sigma = {grad['sigma'] / 1e6:.2f} MPa, delta = {grad['delta']:.6e} m")

    ga = opt.genetic_optimize(population=40, generations=80, seed=42)
    print("Genetic algorithm optimization:")
    print(
        f"  b = {ga['b'] * 1e3:.3f} mm, h = {ga['h'] * 1e3:.3f} mm, "
        f"mass = {ga['mass']:.3f} kg"
    )
    print(f"  sigma = {ga['sigma'] / 1e6:.2f} MPa, delta = {ga['delta']:.6e} m")


def demonstrate_rotor_contact():
    """Rotordynamics and Hertzian contact mechanics."""
    print("\n" + "=" * 80)
    print("Rotordynamics and Contact Mechanics")
    print("=" * 80)

    rotor = JeffcottRotor(m=2.0, k=1.0e6, c=5.0, G=10.0)
    crit = rotor.critical_speed()
    print(f"Jeffcott rotor critical speed: {crit:.2f} Hz")

    speeds = np.linspace(0.0, 500.0, 100)
    _, eigvals = rotor.campbell(speeds)
    min_imag = float(np.min(np.abs(eigvals.imag)))
    print(f"Minimum Campbell imaginary part over 0-500 rad/s: {min_imag:.2f} rad/s")

    contact = HertzianContact(E=200.0e9, nu=0.3, R=0.01)
    F = 100.0
    a = contact.contact_radius(F)
    p0 = contact.max_pressure(F)
    area = contact.contact_area(F)
    print(
        f"Hertzian sphere-plane contact under F = {F:.1f} N: "
        f"a = {a * 1e3:.4f} mm, p_max = {p0 / 1e6:.2f} MPa, area = {area * 1e6:.4f} mm^2"
    )


def generate_report_plots(save=False, filename="mechanical_demo_plots.png"):
    """Generate representative engineering plots. If save=False, plots are closed."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), squeeze=False)

    # S-N curve
    sn = SNCurve(Se=350.0e6, Sut=600.0e6, N0=1.0e3, Nf=1.0e6)
    N = np.logspace(3, 6, 100)
    S = np.array([sn.stress_at_cycles(n) for n in N])
    axes[0, 0].semilogx(N, S / 1e6, color="tab:blue", linewidth=2)
    axes[0, 0].set_title("S-N Curve")
    axes[0, 0].set_xlabel("Cycles to Failure")
    axes[0, 0].set_ylabel("Stress Amplitude [MPa]")
    axes[0, 0].grid(True, which="both", linestyle="--", alpha=0.5)

    # SDOF frequency response
    sdo = SDOFDampedSystem(m=10.0, k=4000.0, c=20.0)
    omega = np.linspace(0.1 * sdo.omega_n, 3.0 * sdo.omega_n, 150)
    magnitudes = [abs(sdo.frequency_response(w, F0=1.0)[2]) for w in omega]
    axes[0, 1].semilogx(omega / (2.0 * pi), magnitudes, color="tab:red", linewidth=2)
    axes[0, 1].set_title("SDOF Frequency Response Magnitude")
    axes[0, 1].set_xlabel("Frequency [Hz]")
    axes[0, 1].set_ylabel("Magnitude [m/N]")
    axes[0, 1].grid(True, which="both", linestyle="--", alpha=0.5)

    # Goodman/Gerber curves
    goodman = GoodmanCorrection("goodman")
    gerber = GoodmanCorrection("gerber")
    Sm = np.linspace(0.0, 600.0e6, 150)
    Se = 350.0e6
    Sut = 600.0e6
    g_amp = np.array([goodman.allowable_amplitude(x, Se, Sut) for x in Sm])
    r_amp = np.array([gerber.allowable_amplitude(x, Se, Sut) for x in Sm])
    axes[1, 0].plot(Sm / 1e6, g_amp / 1e6, label="Goodman", color="tab:blue")
    axes[1, 0].plot(Sm / 1e6, r_amp / 1e6, label="Gerber", color="tab:green")
    axes[1, 0].set_title("Mean-Stress Corrections")
    axes[1, 0].set_xlabel("Mean Stress [MPa]")
    axes[1, 0].set_ylabel("Allowable Amplitude [MPa]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Beam optimization sample mass surface
    opt = BeamWeightOptimizer()
    b = np.linspace(opt.bounds[0][0], opt.bounds[0][1], 40)
    h = np.linspace(opt.bounds[1][0], opt.bounds[1][1], 40)
    B, H = np.meshgrid(b, h)
    mass = np.zeros_like(B)
    for ii in range(B.shape[0]):
        for jj in range(B.shape[1]):
            try:
                mass[ii, jj] = opt.objective(np.array([B[ii, jj], H[ii, jj]]))
            except Exception:
                mass[ii, jj] = np.nan
    axes[1, 1].contourf(B, H, mass, levels=20, cmap="viridis")
    axes[1, 1].set_title("Beam Mass [kg] vs Cross-Section")
    axes[1, 1].set_xlabel("Width b [m]")
    axes[1, 1].set_ylabel("Depth h [m]")
    axes[1, 1].cbar = fig.colorbar(axes[1, 1].contourf(B, H, mass, levels=20, cmap="viridis"), ax=axes[1, 1])
    axes[1, 1].cbar.set_label("Mass [kg]")

    fig.tight_layout()
    if save:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the complete demonstration suite."""
    print(
        "Mechanical Engineering Applied Mechanics / Solid Mechanics / Applied Design"
    )
    print("Standalone demonstration script using NumPy, SciPy, and Matplotlib.")

    demonstrate_mohr_and_failures()
    demonstrate_vibrations()
    demonstrate_fatigue_fracture()
    demonstrate_fea()
    demonstrate_optimization()
    demonstrate_rotor_contact()
    generate_report_plots(save=False)

    print("\n" + "=" * 80)
    print("All demonstrations completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()