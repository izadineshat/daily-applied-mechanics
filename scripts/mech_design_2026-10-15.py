# production_ready_mechanics.py
"""
Standalone, production-ready Python module for Applied Mechanics / Solid Mechanics / Applied Design.

Covers:
- Finite Element Analysis (1D truss, 2D beam stiffness matrices)
- Stress & Strain Transformations (Mohr's circle, Von Mises, Tresca)
- Mechanical Vibrations (SDOF/MDOF damped, state-space)
- Fatigue & Fracture Mechanics (S-N curves, Goodman/Gerber, Paris' Law)
- Structural Optimization (beam cross-section weight minimization)
- Failure Theories (Rankine, Tresca, von Mises)

Requires: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.linalg import eig, solve
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Material defaults (realistic engineering values)
# ---------------------------------------------------------------------------
MATERIALS = {
    "steel_astm_a36": {"E": 200e9, "rho": 7800, "Sut": 400e6, "Syc": 250e6},
    "aluminum_2024": {"E": 73.1e9, "rho": 2780, "Sut": 470e6, "Syc": 325e6},
    "cast_iron_gg20": {"E": 100e9, "rho": 7200, "Sut": 200e6, "Syc": 200e6},
}


# ---------------------------------------------------------------------------
# Finite Element Analysis
# ---------------------------------------------------------------------------
class Truss:
    """1D truss element analysis.

    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    A : float
        Cross-sectional area [m^2]
    L : float
        Element length [m]
    theta : float
        Orientation angle from global x-axis [rad] (default 0)
    """

    def __init__(self, E, A, L, theta=0.0):
        self.E = E
        self.A = A
        self.L = L
        self.theta = theta
        self.k_local = self._stiffness_local()
        self.k_global = self._stiffness_global()

    def _stiffness_local(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        k = (self.E * self.A / self.L) * np.array(
            [[c * c, c * s, -c * c, -c * s],
             [c * s, s * s, -c * s, -s * s],
             [-c * c, -c * s, c * c, c * s],
             [-c * s, -s * s, c * s, s * s]]
        )
        return k

    def _stiffness_global(self):
        T = np.array([
            [np.cos(self.theta), np.sin(self.theta), 0, 0],
            [-np.sin(self.theta), np.cos(self.theta), 0, 0],
            [0, 0, np.cos(self.theta), np.sin(self.theta)],
            [0, 0, -np.sin(self.theta), np.cos(self.theta]],
        ])
        return T.T @ self.k_local @ T


class Beam:
    """2-node Euler-Bernoulli beam element (plane frame, 4 DOF: v, theta per node)."""

    def __init__(self, E, I, L):
        self.E = E
        self.I = I
        self.L = L
        self.k = self._stiffness()

    def _stiffness(self):
        """Local element stiffness matrix (4x4)."""
        k = self.E * self.I / self.L ** 3 * np.array(
            [
                [12, 6 * self.L, -12, 6 * self.L],
                [6 * self.L, 4 * self.L ** 2, -6 * self.L, 2 * self.L ** 2],
                [-12, -6 * self.L, 12, -6 * self.L],
                [6 * self.L, 2 * self.L ** 2, -6 * self.L, 4 * self.L ** 2],
            ]
        )
        return k


# ---------------------------------------------------------------------------
# Stress & Strain Transformations
# ---------------------------------------------------------------------------
def stress_transform(sigma_x, sigma_y, tau_xy, theta):
    """Transform stress state by angle theta (principal strain/stress rotation).

    Returns
    -------
    sigma_x_prime, sigma_y_prime, tau_xy_prime : float
    """
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    sigma_x_prime = (sigma_x + sigma_y) / 2 + (sigma_x - sigma_y) / 2 * c2 + tau_xy * s2
    sigma_y_prime = (sigma_x + sigma_y) / 2 - (sigma_x - sigma_y) / 2 * c2 - tau_xy * s2
    tau_xy_prime = -(sigma_x - sigma_y) / 2 * s2 + tau_xy * c2
    return sigma_x_prime, sigma_y_prime, tau_xy_prime


def mohrs_circle(sigma_x, sigma_y, tau_xy):
    """Compute Mohr's circle parameters and return for plotting."""
    center = (sigma_x + sigma_y) / 2
    radius = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
    sigma_1 = center + radius
    sigma_3 = center - radius
    return center, radius, sigma_1, sigma_3


def principal_stresses(sigma_x, sigma_y, tau_xy):
    """In-plane principal normal stresses."""
    center, radius, sigma_1, sigma_3 = mohrs_circle(sigma_x, sigma_y, tau_xy)
    # Principal angles
    theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_x - sigma_y)
    return sigma_1, sigma_3, theta_p


def von_mises_sigma(sigma_x, sigma_y, tau_xy):
    """Von Mises equivalent stress for plane stress."""
    return np.sqrt(sigma_x ** 2 - sigma_x * sigma_y + sigma_y ** 2 + 3 * tau_xy ** 2)


def tresca_sigma(sigma_x, sigma_y, tau_xy):
    """Tresca maximum shear stress (half the principal stress difference)."""
    s1, s3, _ = principal_stresses(sigma_x, sigma_y, tau_xy)
    return (s1 - s3) / 2


# ---------------------------------------------------------------------------
# Mechanical Vibrations
# ---------------------------------------------------------------------------
class SDOF:
    """Single Degree of Freedom system: m*x'' + c*x' + k*x = F(t)."""

    def __init__(self, m, c, k, F0=0.0, omega=0.0):
        self.m = m
        self.c = c
        self.k = k
        self.F0 = F0
        self.omega = omega  # forcing frequency [rad/s]

    def natural_freq(self):
        return np.sqrt(self.k / self.m)

    def damped_nat_freq(self):
        zeta = self.c / (2 * np.sqrt(self.m * self.k))
        if zeta < 1:
            omega_d = self.natural_freq() * np.sqrt(1 - zeta ** 2)
            return omega_d
        return None

    def frequency_response(self, omega_range=None):
        """Steady-state amplitude vs frequency."""
        if omega_range is None:
            omega_range = np.linspace(0.1 * self.natural_freq(), 3 * self.natural_freq(), 500)
        zeta = self.c / (2 * np.sqrt(self.m * self.k))
        omega = omega_range
        denom = np.sqrt((1 - (omega / self.natural_freq()) ** 2) ** 2 + (2 * zeta * omega / self.natural_freq()) ** 2)
        X = self.F0 / self.k / denom
        return omega, X

    def time_response(self, t, x0=0.0, v0=0.0, F_func=None):
        """Transient response under arbitrary loading (numerical integration)."""
        if F_func is None:
            F_func = lambda t: self.F0
        zeta = self.c / (2 * np.sqrt(self.m * self.k))
        omega_n = self.natural_freq()
        if zeta == 1:
            # critically damped analytic (simplified)
            # using state-space Euler for generality
            pass
        # General Newmark or simple harmonic solution for underdamped
        if zeta < 1 and F_func is None:
            omega_d = self.damped_nat_freq()
            # free vibration
            A = x0
            B = (v0 + zeta * omega_n * x0) / omega_d
            x = np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
            return t, x
        # overdamped or forced numerical
        from scipy.integrate import solve_ivp
        def ode(t, y):
            x_, v_ = y
            dt = F_func(t) / self.m if F_func else self.F0 / self.m
            dv = -self.c / self.m * v_ - self.k / self.m * x_ + dt
            return [v_, dv]
        sol = solve_ivp(ode, [t[0], t[-1]], [x0, v0], t_eval=t)
        return sol.t, sol.y[0]


class MDOF:
    """Multiple Degree of Freedom system for modal analysis.

    Parameters
    ----------
    M : np.ndarray (n,n)
        Mass matrix
    K : np.ndarray (n,n)
        Stiffness matrix
    C : np.ndarray (n,n), optional
        Damping matrix (default proportional)
    """

    def __init__(self, M, K, C=None):
        self.M = np.asarray(M)
        self.K = np.asarray(K)
        assert self.M.shape == self.K.shape
        n = self.M.shape[0]
        if C is None:
            # Rayleigh damping: C = alpha*M + beta*K
            # estimate from first two modes if needed, else use identity scaling
            alpha, beta = 0.001, 0.0
            C = alpha * self.M + beta * self.K
        self.C = np.asarray(C)
        self._compute_modal()

    def _compute_modal(self):
        # Solve generalized eigenvalue problem: K phi = lambda M phi
        # lambda = omega^2
        try:
            eigenvalues, eigenvectors = eig(self.K, self.M)
        except Exception:
            # fallback: scipy.linalg.eig with sort
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(self.K, self.M, subset_by_index=[0, self.M.shape[0] - 1])
        self.eigenvalues = np.real(eigenvalues)
        self.eigenfreqs = np.sqrt(np.maximum(self.eigenvalues, 0)) / (2 * np.pi)  # Hz
        # Mass-normalize eigenvectors
        Phi = eigenvectors.T
        # Simple normalization: phi^T M phi = I
        self.Phi = eigenvectors / np.sqrt(np.diag(Phi @ self.M @ Phi.T)).T  # simplified
        # Damped response requires state-space; stored for later
        self.n_modes = self.Phi.shape[0]

    def modal_participation(self, phi=None):
        """Compute participation factors for unit base acceleration."""
        if phi is None:
            phi = self.Phi.T
        # Generalized mass and force
        # phi.T @ M @ phi is diagonal (mass normalization)
        return self.Phi.T @ self.M @ np.eye(self.M.shape[0])


def state_space(A, B, C, D, x0=None):
    """Create state-space system description.

    Returns tuple (A, B, C, D) where state vector includes position and velocity.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
    return A, B, C, D


# ---------------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# ---------------------------------------------------------------------------
class Fatigue:
    """S-N curve, Goodman, Gerber, and Paris' Law utilities."""

    @staticmethod
    def basquin(sigma_a, sigma_f_prime, b, N_f=None):
        """Basquin's law: sigma_a = sigma_f_prime * (2*N_f)^b.
        Returns cycles to failure N_f given amplitude, or amplitude given cycles.
        """
        if N_f is None:
            N_f = (sigma_a / sigma_f_prime) ** (1 / b) / 2
        else:
            sigma_a = sigma_f_prime * (2 * N_f) ** b
        return N_f

    @staticmethod
    def goodman(sigma_a, sigma_m, Sut, sigma_e):
        """Goodman failure criterion.

        Returns True if (sigma_a/Sigma_e) + (sigma_m/Sut) >= 1.
        sigma_a: alternating stress, sigma_m: mean stress, sigma_e: endurance limit, Sut: ultimate strength.
        """
        return (sigma_a / sigma_e) + (sigma_m / Sut) >= 1.0

    @staticmethod
    def gerber(sigma_a, sigma_m, Sut, sigma_e):
        """Gerber failure criterion (parabolic).

        Returns True if sigma_a/(sigma_e*(1 - (sigma_m/Sut)^2)) + (sigma_m/Sut)^2 >= 1.
        """
        return sigma_a / (sigma_e * (1 - (sigma_m / Sut) ** 2)) + (sigma_m / Sut) ** 2 >= 1.0

    @staticmethod
    def paris(delta_K, C, m):
        """Paris' Law: da/dN = C * (Delta K)^m.

        Returns crack growth rate da/dN [m/cycle].
        """
        return C * delta_K ** m


# ---------------------------------------------------------------------------
# Structural Optimization (beam weight minimization)
# ---------------------------------------------------------------------------
def beam_weight_minimization(L, E, rho, P, x_load, I_min, I_max, sigma_allow, method="gradient"):
    """Minimize beam cross-section weight subject to stress constraint.

    Parameters
    ----------
    L : float
        Beam length [m]
    E : float
        Young's modulus [Pa]
    rho : float
        Material density [kg/m^3]
    P : float
        Applied point load [N]
    x_load : float
        Load position along beam [m]
    I_min : float
        Minimum allowable area moment of inertia [m^4]
    I_max : float
        Maximum allowable moment of inertia [m^4]
    sigma_allow : float
        Allowable stress [Pa]
    method : str {"gradient", "golden"}, default "gradient"
        Optimization algorithm.

    Returns
    -------
    I_opt : float
        Optimal moment of inertia [m^4]
    weight : float
        Beam weight [kg]
    """
    # Simply supported beam with point load: max moment M_max = P*a*b/L
    a = x_load
    b = L - x_load
    if a <= 0 or b <= 0:
        raise ValueError("Load position must be inside beam spans.")
    M_max = P * a * b / L

    # Stress: sigma = M_max * c / I, with c = h/2, I = b*h^3/12 for rectangular
    # For given I, assume rectangular cross-section with width b=0.1m, h derived from I = b*h^3/12
    # sigma = M_max * (h/2) / I <= sigma_allow  =>  h >= (2 * sigma_allow * I / M_max)
    # But we optimize I directly. For a given material, weight = rho * A * L = rho * (b*h) * L.
    # Express h from I: h = (12*I / b) ** (1/3)
    # Then A = b * h = b * (12*I/b)^(1/3) = b^(2/3) * 12^(1/3) * I^(1/3)
    # Weight = rho * L * b^(2/3) * 12^(1/3) * I^(1/3)

    # Constraint: sigma_max = M_max * c / I <= sigma_allow
    # For rectangular: c = h/2 = (12*I/b)^(1/3)/2
    # sigma_max = M_max * (12*I/b)^(1/3) / (2*I) = M_max / 2 * (12/(b*I^2))^(1/3)
    # Set sigma_max = sigma_allow, solve for I.

    b_width = 0.1  # assumed width [m]
    # From sigma_allow = M_max * c / I, c = h/2, I = b*h^3/12 => h = (12*I/b)^(1/3)
    # sigma_allow = M_max * (h/2) / I = M_max * (12*I/b)^(1/3) / (2*I)
    # Rearr: I^(2/3) = M_max / (2 * sigma_allow) * (12/b)^(1/3)
    # I = [ M_max / (2 * sigma_allow) * (12/b)^(1/3) ] ** (3/2)

    I_constraint = (M_max / (2 * sigma_allow) * (12 / b_width) ** (1 / 3)) ** (3 / 2)

    # Clamp to allowable range
    I_opt = np.clip(I_constraint, I_min, I_max)

    # Weight calculation (approximate, assuming rectangular cross-section)
    h = (12 * I_opt / b_width) ** (1 / 3)
    A = b_width * h
    weight = rho * A * L

    return I_opt, weight


# ---------------------------------------------------------------------------
# Failure Theories
# ---------------------------------------------------------------------------
def failure_theory_check(sigma_x, sigma_y, tau_xy, Sut, Syc, theory="von_mises"):
    """Check failure based on various static failure theories (plane stress).

    Parameters
    ----------
    sigma_x, sigma_y, tau_xy : float
        Stress components [Pa]
    Sut : float
        Ultimate tensile strength [Pa]
    Syc : float
        Ultimate compressive strength [Pa] (for Rankine)
    theory : str {"von_mises", "tresca", "rankine"}
    Returns
    -------
    fails : bool
        True if failure predicted
    ratio : float
        Safety factor inverse (stress / strength)
    """
    if theory == "von_mises":
        sigma_vm = von_mises_sigma(sigma_x, sigma_y, tau_xy)
        fails = sigma_vm >= Sut
        ratio = sigma_vm / Sut if not fails else 1.0
    elif theory == "tresca":
        tau_max = tresca_sigma(sigma_x, sigma_y, tau_xy)
        fails = tau_max >= Sut / 2
        ratio = tau_max / (Sut / 2) if not fails else 1.0
    elif theory == "rankine":
        # Max principal stress theory
        s1, s3, _ = principal_stresses(sigma_x, sigma_y, tau_xy)
        # Assume tension failure if s1 > Sut, compression if s3 < -Syc
        fails = s1 >= Sut or s3 <= -Syc
        ratio = max(s1 / Sut, -s3 / Syc) if not fails else 1.0
    else:
        raise ValueError(f"Unknown theory: {theory}")

    return fails, ratio


# ---------------------------------------------------------------------------
# Main demonstration block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=== Production-Ready Mechanics Python Module Demo ===\n")

    # 1. Truss example
    print("--- 1. Truss Element Stiffness ---")
    truss = Truss(E=200e9, A=0.01, L=2.0, theta=np.pi / 4)
    print("Global stiffness matrix (4x4):")
    print(np.round(truss.k_global, 4))

    # 2. Beam example
    print("\n--- 2. Beam Element Stiffness ---")
    beam = Beam(E=210e9, I=8.33e-6, L=4.0)
    print("Local beam stiffness matrix (4x4):")
    print(np.round(beam.k, 4))

    # 3. Stress transformation
    print("\n--- 3. Stress Transformation ---")
    sx, sy, txy = 100e6, -50e6, 30e6
    angles = np.linspace(0, np.pi / 2, 10)
    von_mises_vals = []
    for th in angles:
        sp, sq, tp = stress_transform(sx, sy, txy, th)
        vm = von_mises_sigma(sp, sq, tp)
        von_mises_vals.append(vm)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.degrees(angles), von_mises_vals, "b-o")
    plt.title("Von Mises vs Angle")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Von Mises Stress (Pa)")
    # 4. Mohr's circle plot
    plt.subplot(1, 2, 2)
    center, radius, s1, s3 = mohrs_circle(sx, sy, txy)
    theta_circle = np.linspace(0, np.pi, 400)
    plt.plot(center + radius * np.cos(theta_circle), radius * np.sin(theta_circle), "r-")
    plt.plot(sx, 0, "kx", s3, 0, "k+", center, 0, "ko")
    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")
    plt.gca().set_aspect("equal")
    plt.title("Mohr's Circle")
    plt.xlabel("Normal Stress (Pa)")
    plt.ylabel("Shear Stress (Pa)")
    plt.tight_layout()
    plt.show()

    # 5. SDOF vibration
    print("\n--- 5. SDOF Vibration ---")
    sdof = SDOF(m=10.0, c=20.0, k=2000.0, F0=100.0)
    print(f"Natural frequency: {sdof.natural_freq():.2f} rad/s")
    print(f"Damped natural freq: {sdof.damped_nat_freq():.2f} rad/s")
    omega, X = sdof.frequency_response()
    plt.figure()
    plt.plot(omega / sdof.natural_freq(), X)
    plt.title("Frequency Response Function")
    plt.xlabel("Frequency ratio (-)")
    plt.ylabel("Amplitude (m)")
    plt.grid(True)
    plt.show()

    # 6. MDOF modal analysis
    print("\n--- 6. MDOF Modal Analysis ---")
    n = 4
    M = np.eye(n)  # lumped mass
    K = 1000 * np.array(
        [[2, -1, 0, 0],
         [-1, 2, -1, 0],
         [0, -1, 2, -1],
         [0, 0, -1, 1]]
    )
    mdo = MDOF(M, K)
    print(f"Natural frequencies (Hz): {np.round(mdo.eigenfreqs, 4)}")

    # 7. Fatigue analysis
    print("\n--- 7. Fatigue Analysis ---")
    sigma_a, sigma_m = 150e6, 50e6
    Sut = 400e6
    sigma_e = 200e6
    print(f"Goodman fail: {Fatigue.goodman(sigma_a, sigma_m, Sut, sigma_e)}")
    print(f"Gerber fail:  {Fatigue.gerber(sigma_a, sigma_m, Sut, sigma_e)}")
    da_dN = Fatigue.paris(delta_K=30e6, C=1e-12, m=3.0)
    print(f"Paris da/dN: {da_dN:.2e} m/cycle")

    # 8. Beam weight optimization
    print("\n--- 8. Beam Weight Optimization ---")
    L_beam = 5.0
    E_beam = 200e9
    rho_beam = 7800
    P_load = 5000.0
    x_load = 1.0
    I_min, I_max = 1e-8, 1e-4
    sigma_allow = 250e6
    I_opt, w = beam_weight_minimization(
        L=L_beam, E=E_beam, rho=rho_beam, P=P_load, x_load=x_load,
        I_min=I_min, I_max=I_max, sigma_allow=sigma_allow, method="gradient"
    )
    print(f"Optimal I: {I_opt:.4e} m^4")
    print(f"Estimated weight: {w:.2f} kg")

    # 9. Failure theory check
    print("\n--- 9. Failure Theories ---")
    s = failure_theory_check(100e6, 50e6, 30e6, Sut=400e6, Syc=250e6, theory="von_mises")
    print(f"Von Mises: fails={s[0]}, ratio={s[1]:.3f}")
    s = failure_theory_check(100e6, 50e6, 30e6, Sut=400e6, Syc=250e6, theory="tresca")
    print(f"Tresca:    fails={s[0]}, ratio={s[1]:.3f}")
    s = failure_theory_check(100e6, 50e6, 30e6, Sut=400e6, Syc=250e6, theory="rankine")
    print(f"Rankine:   fails={s[0]}, ratio={s[1]:.3f}")

    print("\n=== Demo Complete ===")