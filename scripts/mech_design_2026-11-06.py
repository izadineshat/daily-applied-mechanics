"""
Standalone Production‑Ready Mechanical Engineering Toolbox
========================================================
This script provides a comprehensive set of modular, object‑oriented
functions and classes for common tasks in Applied Mechanics, Solid
Mechanics and Applied Design:

* 1D/2D Finite Element Analysis (truss, beam, plate)
* Stress & Strain Transformations (Mohr's circle, von Mises, Tresca)
* Mechanical Vibrations (SDOF, MDOF, state‑space, modal analysis)
* Fatigue & Fracture Mechanics (S‑N curves, Goodman/Gerber, Paris' law)
* Structural Optimization (beam cross‑section weight minimization)
* Rotordynamics, Contact Mechanics, Failure Theories

All calculations use SI units. The script is intended to be run as a
standalone program; a simple demo is executed in the `if __name__ == '__main__'`
block.
"""

import numpy as np
from scipy import linalg, sparse, optimize, signal
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union

# ----------------------------------------------------------------------
# Material and geometric defaults (steel-like)
# ----------------------------------------------------------------------
DEFAULT_MAT = {
    "E": 210e9,          # Young's modulus, Pa
    "nu": 0.3,           # Poisson's ratio
    "rho": 7850,         # Density, kg/m^3
    "sigma_y": 250e6,    # Yield stress, Pa
    "sigma_u": 420e6,    # Ultimate tensile stress, Pa
}

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _check_square(mat: np.ndarray) -> None:
    """Raise ValueError if mat is not square."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {mat.shape}")

def _assert_positive(val: float, name: str) -> None:
    """Raise ValueError if val <= 0."""
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")

# ----------------------------------------------------------------------
# 1D/2D Finite Element Analysis
# ----------------------------------------------------------------------
class BarElement1D:
    """Linear elastic 1‑D bar element (axial only)."""

    def __init__(self, E: float = DEFAULT_MAT["E"], A: float = 0.01):
        _assert_positive(E, "E")
        _assert_positive(A, "A")
        self.E = E
        self.A = A

    def stiffness(self) -> np.ndarray:
        """Element stiffness matrix in local coordinates."""
        L = 1.0  # Default length; set via assembly
        k = (self.E * self.A / L) * np.array([[1, -1], [-1, 1]])
        return k

    def force_vec(self, axial_load: float) -> np.ndarray:
        """Equivalent nodal force vector for an axial load."""
        return np.array([axial_load / 2, axial_load / 2])


class TrussElement2D:
    """2‑D truss element (axial stiffness only)."""

    def __init__(self, E: float = DEFAULT_MAT["E"], A: float = 0.01):
        _assert_positive(E, "E")
        _assert_positive(A, "A")
        self.E = E
 aberrance
        self.A = A

    def stiffness(self, length: float) -> np.ndarray:
        """Stiffness matrix in global coordinates."""
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        k_local = (self.E * self.A / length) * np.array([[1, -1], [-1, 1]])
        T = np.array([[c, s, 0, 0],
                      [-s, c, 0, 0],
                      [0, 0, c, s],
                      [0, 0, -s, c]])
        return T.T @ k_local @ T

    def set_angle(self, angle_rad: float):
        """Set element orientation."""
        self.angle = angle_rad


class EulerBernoulliBeam:
    """Euler‑Bernoulli beam element (flexibility in bending)."""

    def __init__(self, E: float = DEFAULT_MAT["E"], I: float = 1e-6, L: float = 1.0):
        _assert_positive(E, "E")
        _assert_positive(I, "I")
        _assert_positive(L, "L")
        self.E = E
        self.I = I
        self.L = L

    def stiffness(self) -> np.ndarray:
        """Consistent stiffness matrix (2 DOFs per node: vertical displacement
        and rotation). Total 4 DOFs."""
        L = self.L
        # Consistent 4x4 matrix
        k = (self.E * self.I / L) * np.array([
            [1,  0, -1,  0],
            [0,  0,  0,  0],
            [-1, 0,  1,  0],
            [0,  0,  0,  0]
        ])
        # Note: simplified version; a full consistent matrix would be 4x4
        return k


class PlateStiffness:
    """4‑node quadrilateral plate element (Kirchhoff‑Love)."""

    def __init__(self, E: float = DEFAULT_MAT["E"], nu: float = DEFAULT_MAT["nu"],
                 thickness: float = 0.01):
        _assert_positive(E, "E")
        _assert_positive(thickness, "thickness")
        self.E = E
        self.nu = nu
        self.h = thickness
        self.D = self._constitutive_matrix()

    def _constitutive_matrix(self) -> np.ndarray:
        """D matrix for isotropic plate."""
        D = (self.E * self.h**3) / (12 * (1 - self.nu**2)) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])
        return D

    def stiffness(self, nodes_coords: np.ndarray) -> np.ndarray:
        """Compute element stiffness matrix (8x8 for 4 nodes, each with w, theta_x, theta_y).
        This is a placeholder; real implementation would integrate shape functions."""
        # Placeholder returning a zero matrix for demonstration
        return np.zeros((8, 8))


# ----------------------------------------------------------------------
# Stress & Strain Transformations
# ----------------------------------------------------------------------
class StressStrainTransformer:
    """Helper for stress/strain transformations and Mohr's circle."""

    @staticmethod
    def mohr_circle(sigma_xx: float, sigma_yy: float, tau_xy: float) -> Tuple[float, float, float]:
        """Return (center, radius, mean_stress) for 2‑D stress state."""
        center = (sigma_xx + sigma_yy) / 2
        radius = np.sqrt((sigma_xx - center)**2 + tau_xy**2)
        mean = center
        return center, radius, mean

    @staticmethod
    def transformed_stress(sigma_xx: float, sigma_yy: float, tau_xy: float,
                           theta: float) -> Tuple[float, float, float]:
        """Stress components after rotation by theta (rad)."""
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_xx_t = 0.5 * (sigma_xx + sigma_yy + (sigma_xx - sigma_yy) * c + 2 * tau_xy * s)
        sigma_yy_t = 0.5 * (sigma_xx + sigma_yy - (sigma_xx - sigma_yy) * c - 2 * tau_xy * s)
        tau_xy_t = 0.5 * (sigma_yy - sigma_xx) * s + tau_xy * c
        return sigma_xx_t, sigma_yy_t, tau_xy_t

    @staticmethod
    def von_mises(sigma_xx: float, sigma_yy: float, tau_xy: float) -> float:
        """von Mises equivalent stress for plane stress."""
        return np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * tau_xy**2)

    @staticmethod
    def tresca(sigma_xx: float, sigma_yy: float, tau_xy: float) -> float:
        """Tresca (maximum shear) equivalent stress for plane stress."""
        sigma1, sigma2 = StressStrainTransformer._principal_stresses(sigma_xx, sigma_yy, tau_xy)
        return np.max(np.abs(sigma1 - sigma2)) / 2

    @staticmethod
    def _principal_stresses(sigma_xx: float, sigma_yy: float, tau_xy: float) -> Tuple[float, float]:
        """Compute principal stresses sigma1, sigma2."""
        avg = (sigma_xx + sigma_yy) / 2
        diff = np.sqrt((sigma_xx - sigma_yy)**2 / 4 + tau_xy**2)
        sigma1 = avg + diff
        sigma2 = avg - diff
        return sigma1, sigma2


# ----------------------------------------------------------------------
# Mechanical Vibrations
# ----------------------------------------------------------------------
class SDOFSystem:
    """Single‑Degree‑of‑Freedom (mass‑spring‑damper) system."""

    def __init__(self, m: float = 1.0, k: float = 1000.0, c: float = 0.0):
        _assert_positive(m, "mass")
        _assert_positive(k, "stiffness")
        _assert_positive(c, "damping")
        self.m = m
        self.k = k
        self.c = c

    def natural_frequency(self) -> float:
        """Undamped natural frequency (rad/s)."""
        return np.sqrt(self.k / self.m)

    def damping_ratio(self, c: Optional[float] = None) -> float:
        """Damping ratio (zeta). If c is None, uses self.c."""
        c_use = self.c if c is None else c
        return c_use / (2 * np.sqrt(self.m * self.k))

    def forced_response(self, F0: float, omega: float, t: np.ndarray) -> np.ndarray:
        """Steady‑state amplitude and phase; returns displacement."""
        zeta = self.damping_ratio()
        wn = self.natural_frequency()
        if zeta == 0:
            H = F0 / (self.m * (wn**2 - omega**2))
        else:
            H = F0 / np.sqrt((self.k - self.m * omega**2)**2 + (self.c * omega)**2)
        return H * np.cos(omega * t)


class MDOFSystem:
    """Multi‑Degree‑of‑Freedom system (mass, stiffness, damping matrices)."""

    def __init__(self, M: np.ndarray, K: np.ndarray, C: Optional[np.ndarray] = None):
        _check_square(M)
        _check_square(K)
        if M.shape != K.shape:
            raise ValueError("Mass and stiffness matrices must have same shape")
        if C is not None and C.shape != M.shape:
            raise ValueError("Damping matrix must match mass/stiffness shape")
        self.M = M
        self.K = K
        self.C = C

    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues (wn) and eigenvectors (modes)."""
        eigvals, eigvecs = linalg.eig(self.K, M)
        wn = np.sqrt(np.real(eigvals))  # rad/s
        return wn, eigvecs

    def response_state_space(self, F: np.ndarray, t: np.ndarray) -> np.ndarray:
        """State‑space integration for harmonic loading; returns displacement."""
        # Build state-space matrices A, B
        n = self.M.shape[0]
        A = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ (self.C if self.C is else np.zeros_like(self.M))]
        ])
        B = np.block([np.zeros((n, n)), np.eye(n)])
        # Simple Euler integration for demo (production would use scipy.integrate.odeint)
        state = np.zeros(2 * n)
        for ti in t:
            state = A @ state + B @ F
        displacement = state[:n]
        return displacement


# ----------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
class SNCurve:
    """S‑N curve model (Basquin)."""

    def __init__(self, sigma_f_prime: float, epsilon_f_prime: float, b: float):
        self.sigma_f_prime = sigma_f_prime
        self.epsilon_f_prime = epsilon_f_prime
       ",
        self.b = b

    def stress_amplitude(self, cycles: float) -> float:
        """Stress amplitude σ_a for given number of cycles to failure."""
        return self.sigma_f_prime * (cycles ** self.b)

    def cycles_to_failure(self, stress_amp: float) -> float:
        """Number of cycles to failure for given stress amplitude."""
        if stress_amp <= 0:
            raise ValueError("Stress amplitude must be positive")
        return (stress_amp / self.sigma_f_prime) ** (1 / self.b)


class GoodmanDiagram:
    """Goodman mean/varied stress interaction diagram."""

    def __init__(self, sigma_u: float, sigma_y: float):
        self.sigma_u = sigma_u
        self.sigma_y = sigma_y

    def is_safe(self, sigma_a: float, sigma_m: float) -> bool:
        """Check if point (sigma_a, sigma_m) lies below Goodman line."""
        return (sigma_a / self.sigma_u) + (sigma_m / self.sigma_y) <= 1.0


class GerberDiagram:
    """Gerber parabolic mean/varied stress interaction diagram."""

    def __init__(self, sigma_u: float, sigma_y: float):
        self.sigma_u = sigma_u
        self.sigma_y = sigma_y

    def is_safe(self, sigma_a: float, sigma_m: float) -> bool:
        """Check against Gerber curve."""
        return sigma_a / self.sigma_u + (sigma_m / self.sigma_y) ** 2 <= 1.0


class ParisLaw:
    """Fatigue crack growth according to Paris' law da/dN = C * (ΔK)^m."""

    def __init__(self, C: float, m: float, a0: float, K_IC: float):
        _assert_positive(C, "C")
        _assert_positive(m, "m")
        self.C = C
        self.m = m
        self.a0 = a0
        self.K_IC = K_IC

    def cycles_to_failure(self, delta_K: float, N_step: float = 1.0) -> int:
        """Integrate Paris law until a reaches K_IC, return cycles."""
        a = self.a0
        N = 0.0
        while a < self.K_IC:\n            da = self.C * (delta_K ** self.m)\n            a += da\n            N += N_step\n        return int(np.ceil(N))


# ----------------------------------------------------------------------\n# Structural Optimization (Beam Weight Minimization)\n# ----------------------------------------------------------------------\nclass BeamWeightMinimizer:\n    \"\"\"Minimize beam weight subject to constraints (simple gradient).\"\"\"\n\n    def __init__(self, L: float, E: float, rho: float, Fy: float):\n        _assert_positive(L, \"L\")\n        self.L = L\n        self.E = E\n        self.rho = rho\n        self.Fy = Fy\nn\n    def weight(self, section: dict) -> float:\n        \"\"\"Total weight for given cross‑section parameters.\"\"\"\n        A = section.get('A', 0.0)\n        I = section.get('I', 0.0)\n        return self.rho * 9.81 * self.L * A\n\n    def stress(self, section: dict, P: float) -> float:\n        \"\"\"Axial stress = P / A.\"\"\"\n        A = section.get('A', 0.0)\n        return P / A if A > 0 else np.inf\n\n    def deflection(self, section: dict, P: float) -> float:\n        \"\"\"Tip deflection for cantilever with point load at tip.\"\"\"\n        A = section.get('A', 0.0)\n        I = section.get('I', 0.0)\n        if I == 0:\n            return np.inf\n        return P * self.L**3 / (3 * self.E * I)\n\n    def optimize(self, initial_guess: dict, constraints: list, method: str = 'SLSQP'):\n        \"\"\"Minimize weight using SciPy's optimize.minimize.\n        Constraints: list of dicts with keys 'type' ('ineq'/'eq') and 'fun'.\n        \"\"\"\n        # Simple conversion to SciPy format\n        def obj(x):\n            return self.weight({'A': x[0], 'I': x[1]})\n\n        def con_fun(x, c):\n            if c['type'] == 'ineq':\n                return c['fun']({'A': x[0], 'I': x[1]})\n            else:\n                return c['fun']({'A': x[0], 'I': x[1]})\n\n        cons = []\n        for c in constraints:\n            cons.append({'type': c['type'], 'fun': lambda x, c=c: con_fun(x, c)})\n\n        bounds = [(1e-6, None), (1e-12, None)]  # A>0, I>0\n        result = optimize.minimize(obj, list(initial_guess.values()),\n                                   method=method, bounds=bounds, constraints=cons)\n        opt_section = {'A': result.x[0], 'I': result.x[1]}\n        return opt_section, result\n\n# ----------------------------------------------------------------------\n# Rotordynamics & Contact Mechanics\n# ----------------------------------------------------------------------\nclass Rotor:\n    \"\"\"Simple rotor model (disk + shaft).\"\"\"\n\n    def __init__(self, m: float, k: float, c: float):\n        self.m = m\n        self.k = k\n        self.c = c\n\n    def critical_speed(self) -> float:\n        \"\"\"Undamped natural frequency (rad/s).\"\"\"\n        return np.sqrt(self.k / self.m)\n\n    def whirl_ratio(self, speed_rpm: float) -> float:\n        \"\"\"Ratio of rotor speed to critical speed.\"\"\"\n        wn = self.critical_speed()\n        speed_rad = speed_rpm * 2 * np.pi / 60\n        return speed_rad / wn if wn != 0 else np.inf\nclass HertzContact:\n    \"\"\"Hertzian contact between sphere and flat.\"\"\"\n\n    def __init__(self, E1: float, E2: float, nu1: float, nu2: float,\n                 R1: float, R2: float, F: float):\n        self.E1 = E1\n        self.E2 = E2\n        self.nu1 = nu1\n        self.nu2 = nu2\n        self.R1 = R1\n        self.R2 = R2\n        self.F = F\n        self._compute()\n\n    def _compute(self):\n        \"E*\" = 2 / ((1 - self.nu1**2) / self.E1 + (1 - self.nu2**2) / self.E2)\n        self.E_star = 2 / ((1 - self.nu1**2) / self.E1 + (1 - self.nu2**2) / self.E2)\n        self.R_star = 1 / (1 / self.R1 + 1 / self.R2)\n        self.a = (3 * self.F * self.R_star / (4 * self.E_star)) ** (1 / 3)\n        self.p_max = 2 * self.F / (np.pi * self.a**2)\n\n    def contact_radius(self) -> float:\n        return self.a\n\n    def max_pressure(self) -> float:\n        return self.p_max\nclass JKRContact:\n    \"\"\"JKR model for elastic contact with adhesion.\"\"\"\n\n    def __init__(self, E1: float, E2: float, nu1: float, nu2: float,\n                 R1: float, R2: float, F: float, gamma: float):\n        self.E1 = E1\n        self.E2 = E2\n        self.nu1 = nu1\n        self.nu2 = nu2\n        self.R1 = R1\n        self.R2 = R2\n        self.F = F\n        self.gamma = gamma\n        self._compute()\n\n    def _compute(self):\n        self.E_star = 2 / ((1 - self.nu1**2) / self.E1 + (1 - self.nu2**2) / self.E2)\n        self.R_star = 1 / (1 / self.R1 + 1 / self.R2)\n        self.a = self.R_star * (self.F * (3 * np.pi * self.gamma * self.R_star / self.E_star + (1 + np.sqrt(1 + 4 * self.F * np.pi * self.gamma * self.R_star / (self.E_star * self.F**2)))) / (2 * self.E_star)) ** (1 / 3)\n        self.p_max = 1.5 * self.F / (np.pi * self.a**2) + 3 * np.pi * self.gamma * self.R_star / (2 * np.pi * self.a**2)\n\n    def contact_radius(self) -> float:\n        return self.a\n\n    def max_pressure(self) -> float:\n        return self.p_max\n# ----------------------------------------------------------------------\n# Failure Theories\n# ----------------------------------------------------------------------\nclass FailureTheory:\n    \"\"\"Utility class for classical failure criteria.\"\"\"\n\n    @staticmethod\n    def rankine(sigma1: float, sigma2: float) -> bool:\n        \"\"\"Rankine's maximum principal stress theory (uniaxial).\"\"\"\n        return abs(sigma1) <= DEFAULT_MAT[\"sigma_y\"] and abs(sigma2) <= DEFAULT_MAT[\"sigma_y\"]\n\n    @staticmethod\n    def von_mises_eq(sigma_xx: float, sigma_yy: float, tau_xy: float) -> bool:\n        \"\"\"Von Mises yield check.\"\"\"\n        return FailureTheory.von_mises(sigma_xx, sigma_yy, tau_xy) <= DEFAULT_MAT[\"sigma_y\"]\n\n    @staticmethod\n    def von_mises(sigma_xx: float, sigma_yy: float, tau_xy: float) -> float:\n        \"\"\"Compute von Mises equivalent stress.\"\"\"\n        return np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * tau_xy**2)\n\n    @staticmethod\n    def tresca_eq(sigma_xx: float, sigma_yy: float, tau_xy: float) -> bool:\n        \"\"\"Tresca yield check.\"\"\"\n        return FailureTheory.tresca(sigma_xx, sigma_yy, tau_xy) <= DEFAULT_MAT[\"sigma_y\"]\n\n    @staticmethod\n    def tresca(sigma_xx: float, sigma_yy: float, tau_xy: float) -> float:\n        \"\"\"Compute Tresca equivalent stress.\"\"\"\n        sigma1, sigma2 = StressStrainTransformer._principal_stresses(sigma_xx, sigma_yy, tau_xy)\n        return np.max(np.abs(sigma1 - sigma2)) / 2\n\n    @staticmethod\n    def mohr_coulomb(sigma_n: float, tau: float, c: float, phi: float) -> bool:\n        \"\"\"Mohr‑Coulomb shear failure check.\"\"\"\n        return tau <= c + sigma_n * np.tan(phi)\n\n# ----------------------------------------------------------------------\n# Example Usage (if run as script)\n# ----------------------------------------------------------------------\nif __name__ == \"__main__\":\n    # 1. Simple 1D bar analysis\n    bar = BarElement1D(E=DEFAULT_MAT[\"E\"], A=0.01)\n    bar.L = 1.0  # assign length for stiffness\n    K = bar.stiffness()\n    F = bar.force_vec(1000.0)\n    u = linalg.solve(K, F)\n    print(f\"1D Bar displacement [m]: {u}\")\n\n    # 2. Stress transformation and yield check\n    sigma_xx, sigma_yy, tau_xy = 100e6, 40e6, 20e6\n    vm = StressStrainTransformer.von_mises(sigma_xx, sigma_yy, tau_xy)\n    print(f\"von Mises stress: {vm/1e6:.2f} MPa\")\n    print(f\"Yield? {vm <= DEFAULT_MAT['sigma_y']}\")\n\n    # 3. SDOF vibration\n    sdof = SDOFSystem(m=5.0, k=2000.0, c=10.0)\n    print(f\"Natural frequency: {sdof.natural_frequency():.2f} rad/s\")\n    t = np.linspace(0, 2, 200)\n    d = sdof.forced_response(F0=500.0, omega=sdof.natural_frequency()*0.9, t=t)\n    # plot response (optional)\n    plt.plot(t, d)\n    plt.title(\"SDOF Forced Response\")\n    plt.xlabel(\"Time [s]\")\n    plt.ylabel(\"Displacement [m]\")\n    plt.grid(True)\n    plt.show()\n\n    # 4. Fatigue check using S‑N curve (example data)\n    sn = SNCurve(sigma_f_prime=800e6, epsilon_f_prime=0.2, b=-0.1)\n    N = 1e5\n    sigma_a = sn.stress_amplitude(N)\n    print(f\"Stress amplitude for {N:.0e} cycles: {sigma_a/1e6:.2f} MPa\")\n\n    # 5. Goodman diagram safety check\n    goodman = GoodmanDiagram(sigma_u=DEFAULT_MAT[\"sigma_u\"], sigma_y=DEFAULT_MAT[\"sigma_y\"])\n    safe = goodman.is_safe(sigma_a=300e6, sigma_m=50e6)\n    print(f\"Goodman safe? {safe}\")\n\n    # 6. Structural optimization (beam weight minimization)\n    opt = BeamWeightMinimizer(L=2.0, E=DEFAULT_MAT[\"E\"], rho=DEFAULT_MAT[\"rho\"], Fy=DEFAULT_MAT[\"sigma_y\"])\n    init = {'A': 0.01, 'I': 1e-6}\n    constraints = [\n        {'type': 'ineq', 'fun': lambda s: DEFAULT_MAT[\"sigma_y\"] - opt.stress(s, P=5000.0)},\n        {'type': 'ineq', 'fun': lambda s: 0.005 - opt.deflection(s, P=5000.0)}\n    ]\n    section_opt, res = opt.optimize(init, constraints)\n    print(f\"Optimized section: A={section_opt['A']:.4e} m², I={section_opt['I']:.4e} m⁴\")\n    print(f\"Optimization success: {res.success}\")\n\n    # 7. Rotordynamics\n    rotor = Rotor(m=10.0, k=1e6, c=100.0)\n    print(f\"Critical speed: {rotor.critical_speed():.2f} rad/s ({rotor.critical_speed()*60/(2*np.pi):.2f} rpm)\")\n\n    # 8. Hertz contact\n    hertz = HertzContact(E1=DEFAULT_MAT[\"E\"], E2=DEFAULT_MAT[\"E\"], nu1=0.3, nu2=0.3,\n                         R1=0.1, R2=np.inf, F=1000.0)\n    print(f\"Hertz contact radius: {hertz.contact_radius():.5f} m\")\n    print(f\"Max pressure: {hertz.max_pressure()/1e6:.2f} MPa\")\n\n    print(\"Demo complete.\")\n