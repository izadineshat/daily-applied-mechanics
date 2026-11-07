from __future__ import annotations

import numpy as np
from scipy import linalg, integrate, optimize, signal
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

# =============================================================================
# MECHANICAL ENGINEERING TOOLKIT
# Finite Element Analysis | Stress/Strain | Vibrations | Fatigue/Fracture
# Optimization | Rotordynamics | Contact | Failure Theories
# =============================================================================

# -----------------------------------------------------------------------------
# 1. MATERIAL MODEL
# -----------------------------------------------------------------------------
@dataclass
class Material:
    """Isotropic linear elastic material with realistic defaults (structural steel)."""
    name: str = "Structural Steel"
    E: float = 200e9          # Young's modulus [Pa]
    nu: float = 0.30          # Poisson's ratio [-]
    rho: float = 7850.0       # Density [kg/m^3]
    Sy: float = 250e6         # Yield strength [Pa]
    Sut: float = 400e6        # Ultimate tensile strength [Pa]

    def G(self) -> float:
        """Shear modulus [Pa]."""
        return self.E / (2.0 * (1.0 + self.nu))

    def D_plane_stress(self) -> np.ndarray:
        """Plane stress constitutive matrix [Pa] (Voigt: exx, eyy, gxy)."""
        factor = self.E / (1.0 - self.nu**2)
        return factor * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0],
        ])

    def D_plane_strain(self) -> np.ndarray:
        """Plane strain constitutive matrix [Pa]."""
        factor = self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return factor * np.array([
            [1.0 - self.nu, self.nu, 0.0],
            [self.nu, 1.0 - self.nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * self.nu) / 2.0],
        ])


# -----------------------------------------------------------------------------
# 2. FINITE ELEMENT ANALYSIS
# -----------------------------------------------------------------------------
class TrussElement:
    """2D linear truss element (2 nodes, 2 DOF/node: ux, uy)."""

    def __init__(self, node_i: int, node_j: int, A: float, material: Material):
        self.node_i = node_i
        self.node_j = node_j
        self.A = A
        self.mat = material

    def dofs(self, dof_per_node: int = 2) -> List[int]:
        return [self.node_i * dof_per_node,
                self.node_i * dof_per_node + 1,
                self.node_j * dof_per_node,
                self.node_j * dof_per_node + 1]

    def geometry(self, nodes: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float, float]:
        x1, y1 = nodes[self.node_i]
        x2, y2 = nodes[self.node_j]
        L = np.hypot(x2 - x1, y2 - y1)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        return L, c, s, x2 - x1

    def stiffness(self, nodes: Dict[int, Tuple[float, float]]) -> np.ndarray:
        L, c, s, _ = self.geometry(nodes)
        k_local = (self.A * self.mat.E / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        T = np.array([
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, c, s],
            [0.0, 0.0, -s, c],
        ])
        return T.T @ k_local @ T

    def axial_force(self, u_global: np.ndarray, nodes: Dict[int, Tuple[float, float]]) -> float:
        dofs = self.dofs()
        d_global = u_global[dofs]
        L, c, s, _ = self.geometry(nodes)
        T = np.array([
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, c, s],
            [0.0, 0.0, -s, c],
        ])
        d_local = T @ d_global
        strain = (d_local[1] - d_local[0]) / L
        return self.mat.E * strain * self.A


class Beam2DElement:
    """2D Euler-Bernoulli beam element (2 nodes, 3 DOF/node: ux, uy, theta).
    Includes axial + bending stiffness."""

    def __init__(self, node_i: int, node_j: int, A: float, I: float, material: Material):
        self.node_i = node_i
        self.node_j = node_j
        self.A = A
        self.I = I
        self.mat = material

    def dofs(self, dof_per_node: int = 3) -> List[int]:
        return [self.node_i * dof_per_node,     # ux1
                self.node_i * dof_per_node + 1, # uy1
                self.node_i * dof_per_node + 2, # th1
                self.node_j * dof_per_node,     # ux2
                self.node_j * dof_per_node + 1, # uy2
                self.node_j * dof_per_node + 2] # th2

    def geometry(self, nodes: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
        x1, y1 = nodes[self.node_i]
        x2, y2 = nodes[self.node_j]
        L = np.hypot(x2 - x1, y2 - y1)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        return L, c, s

    def stiffness(self, nodes: Dict[int, Tuple[float, float]]) -> np.ndarray:
        L, c, s = self.geometry(nodes)
        k = np.zeros((6, 6))

        # Axial block (DOFs 0 and 3)
        k_ax = (self.A * self.mat.E / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        k[0, 0] = k_ax[0, 0]; k[0, 3] = k_ax[0, 1]
        k[3, 0] = k_ax[1, 0]; k[3, 3] = k_ax[1, 1]

        # Bending block (DOFs 1,2,4,5)
        EI = self.mat.E * self.I
        k_b = EI / L**3 * np.array([
            [12.0,   6.0 * L,  -12.0,   6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0,  -6.0 * L,   12.0,  -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ])
        bend_idx = [1, 2, 4, 5]
        for ii, a in enumerate(bend_idx):
            for jj, b in enumerate(bend_idx):
                k[a, b] = k_b[ii, jj]

        # Transformation to global
        T = np.zeros((6, 6))
        R = np.array([[c, s], [-s, c]])
        T[0:2, 0:2] = R
        T[3:5, 3:5] = R
        T[2, 2] = 1.0
        T[5, 5] = 1.0
        return T.T @ k @ T

    def end_forces(self, u_global: np.ndarray, nodes: Dict[int, Tuple[float, float]]) -> np.ndarray:
        dofs = self.dofs()
        d_global = u_global[dofs]
        L, c, s = self.geometry(nodes)
        k = self.stiffness(nodes)
        T = np.zeros((6, 6))
        R = np.array([[c, s], [-s, c]])
        T[0:2, 0:2] = R
        T[3:5, 3:5] = R
        T[2, 2] = 1.0
        T[5, 5] = 1.0
        d_local = T @ d_global
        return k @ d_local


class CSTElement:
    """Constant Strain Triangle (CST) for plane stress/strain (3 nodes, 2 DOF/node)."""

    def __init__(self, node_ids: Tuple[int, int, int], thickness: float,
                 material: Material, plane_stress: bool = True):
        self.node_ids = node_ids
        self.t = thickness
        self.mat = material
        self.plane_stress = plane_stress

    def dofs(self, dof_per_node: int = 2) -> List[int]:
        ids = self.node_ids
        return [ids[0]*dof_per_node, ids[0]*dof_per_node+1,
                ids[1]*dof_per_node, ids[1]*dof_per_node+1,
                ids[2]*dof_per_node, ids[2]*dof_per_node+1]

    def stiffness(self, nodes: Dict[int, Tuple[float, float]]) -> np.ndarray:
        (x1, y1), (x2, y2), (x3, y3) = (nodes[i] for i in self.node_ids)
        detJ = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
        A = 0.5 * abs(detJ)
        if A == 0.0:
            raise ValueError("CST element has zero area.")

        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.zeros((3, 6))
        B[0, 0::2] = b
        B[1, 1::2] = c
        B[2, 0::2] = c
        B[2, 1::2] = b
        B /= (2.0 * A)

        D = self.mat.D_plane_stress() if self.plane_stress else self.mat.D_plane_strain()
        return self.t * A * (B.T @ D @ B)

    def stress(self, u_global: np.ndarray, nodes: Dict[int, Tuple[float, float]]) -> np.ndarray:
        dofs = self.dofs()
        d = u_global[dofs]
        (x1, y1), (x2, y2), (x3, y3) = (nodes[i] for i in self.node_ids)
        detJ = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
        A = 0.5 * abs(detJ)
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.zeros((3, 6))
        B[0, 0::2] = b
        B[1, 1::2] = c
        B[2, 0::2] = c
        B[2, 1::2] = b
        B /= (2.0 * A)
        D = self.mat.D_plane_stress() if self.plane_stress else self.mat.D_plane_strain()
        return D @ B @ d


class FEASolver:
    """General sparse FE assembler/solver for truss, beam, or CST elements."""

    def __init__(self, num_nodes: int, dof_per_node: int = 2):
        self.num_nodes = num_nodes
        self.dof_per_node = dof_per_node
        self.ndof = num_nodes * dof_per_node
        self.elements: List[object] = []
        self.F = np.zeros(self.ndof)
        self.constraints: Dict[int, float] = {}
        self.u: Optional[np.ndarray] = None

    def add_element(self, element: object) -> None:
        self.elements.append(element)

    def add_force(self, dof: int, value: float) -> None:
        self.F[dof] += value

    def add_constraint(self, dof: int, value: float = 0.0) -> None:
        self.constraints[dof] = value

    def _assemble(self) -> csr_matrix:
        rows, cols, data = [], [], []
        for el in self.elements:
            dofs = el.dofs(self.dof_per_node)
            ke = el.stiffness({})  # placeholder; replaced below
        # Rebuild with geometry access
        rows, cols, data = [], [], []
        for el in self.elements:
            dofs = el.dofs(self.dof_per_node)
            ke = el.stiffness(self._nodes)
            for i in range(len(dofs)):
                for j in range(len(dofs)):
                    rows.append(dofs[i]); cols.append(dofs[j]); data.append(ke[i, j])
        return coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()

    def solve(self, nodes: Dict[int, Tuple[float, float]]) -> np.ndarray:
        self._nodes = nodes
        K = self._assemble()
        F = self.F.copy()
        for dof, val in self.constraints.items():
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = val
        K = K.tocsr()
        self.u = spsolve(K, F)
        return self.u

    def reactions(self) -> Dict[int, float]:
        if self.u is None:
            raise RuntimeError("Solve first.")
        K = self._assemble()
        res = K @ self.u - self.F
        return {dof: float(res[dof]) for dof in self.constraints}


# -----------------------------------------------------------------------------
# 3. STRESS & STRAIN TRANSFORMATIONS
# -----------------------------------------------------------------------------
@dataclass
class StressState2D:
    """Plane stress state: sx, sy normal stresses, tau_xy shear stress [Pa]."""
    sx: float
    sy: float
    tau: float

    def principal_stresses(self) -> Tuple[float, float]:
        avg = 0.5 * (self.sx + self.sy)
        R = np.sqrt(((self.sx - self.sy) / 2.0)**2 + self.tau**2)
        return avg + R, avg - R  # s1, s2

    def principal_angle(self) -> float:
        """Angle to major principal plane [rad]."""
        return 0.5 * np.arctan2(2.0 * self.tau, self.sx - self.sy)

    def max_shear_inplane(self) -> float:
        return abs(self.principal_stresses()[0] - self.principal_stresses()[1]) / 2.0

    def max_shear_abs(self) -> float:
        """Absolute max shear including sigma3 = 0 (plane stress)."""
        s1, s2 = self.principal_stresses()
        principals = [s1, s2, 0.0]
        return (max(principals) - min(principals)) / 2.0

    def von_mises(self) -> float:
        return np.sqrt(self.sx**2 - self.sx * self.sy + self.sy**2 + 3.0 * self.tau**2)

    def tresca(self) -> float:
        s1, s2 = self.principal_stresses()
        principals = [s1, s2, 0.0]
        return max(principals) - min(principals)

    def mohr_circle(self) -> Dict[str, float]:
        s1, s2 = self.principal_stresses()
        center = 0.5 * (self.sx + self.sy)
        radius = self.max_shear_inplane()
        return {"center": center, "radius": radius, "s1": s1, "s2": s2,
                "theta_p": self.principal_angle()}

    def rotate(self, theta: float) -> StressState2D:
        """Transform stresses by angle theta [rad] (CCW)."""
        c = np.cos(2.0 * theta)
        sn = np.sin(2.0 * theta)
        sx_p = 0.5 * (self.sx + self.sy) + 0.5 * (self.sx - self.sy) * c + self.tau * sn
        sy_p = 0.5 * (self.sx + self.sy) - 0.5 * (self.sx - self.sy) * c - self.tau * sn
        tau_p = -0.5 * (self.sx - self.sy) * sn + self.tau * c
        return StressState2D(sx_p, sy_p, tau_p)

    def failure_ratios(self, Sy: float, Sut: Optional[float] = None) -> Dict[str, float]:
        s1, s2 = self.principal_stresses()
        s3 = 0.0
        principals = np.array([s1, s2, s3])
        s_max = principals.max(); s_min = principals.min()
        sigma_vm = self.von_mises()
        tau_max = (s_max - s_min) / 2.0
        out = {
            "max_principal_ratio": max(abs(s1), abs(s2), abs(s3)) / Sy,
            "tresca_ratio": tau_max / (0.5 * Sy),
            "von_mises_ratio": sigma_vm / Sy,
        }
        if Sut is not None:
            # Maximum normal stress theory with ultimate for brittle
            out["brittle_ratio"] = max(abs(s_max), abs(s_min)) / Sut
        return out


def plot_mohr_circle(state: StressState2D, ax: Optional[plt.Axes] = None) -> plt.Axes:
    mc = state.mohr_circle()
    if ax is None:
        _, ax = plt.subplots()
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mc["center"] + mc["radius"] * np.cos(theta),
            mc["radius"] * np.sin(theta), label="Mohr's circle")
    ax.plot([state.sx, state.sy], [state.tau, -state.tau], "ro-", label="Stress points")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(mc["center"], color="k", ls="--", lw=0.5)
    ax.set_xlabel("Normal stress $\\sigma$ [Pa]")
    ax.set_ylabel("Shear stress $\\tau$ [Pa]")
    ax.set_title("Mohr's Circle (2D)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


# -----------------------------------------------------------------------------
# 4. MECHANICAL VIBRATIONS
# -----------------------------------------------------------------------------
class SDOFSystem:
    """Single-degree-of-freedom damped oscillator."""

    def __init__(self, m: float = 10.0, k: float = 1e5, c: float = 100.0):
        self.m = m
        self.k = k
        self.c = c

    @property
    def wn(self) -> float:
        return np.sqrt(self.k / self.m)

    @property
    def cc(self) -> float:
        return 2.0 * np.sqrt(self.k * self.m)

    @property
    def zeta(self) -> float:
        return self.c / self.cc

    @property
    def wd(self) -> float:
        return self.wn * np.sqrt(max(0.0, 1.0 - self.zeta**2))

    def free_response(self, t: np.ndarray, x0: float, v0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        zeta = self.zeta
        wn = self.wn
        if zeta < 1.0:
            wd = self.wd
            x = np.exp(-zeta * wn * t) * (x0 * np.cos(wd * t) +
                                          (v0 + zeta * wn * x0) / wd * np.sin(wd * t))
            v = np.exp(-zeta * wn * t) * ((v0 + zeta * wn * x0) * np.cos(wd * t) -
                                          (x0 * wd + zeta * wn * (v0 + zeta * wn * x0) / wd) * np.sin(wd * t))
        elif zeta == 1.0:
            x = (x0 + (v0 + wn * x0) * t) * np.exp(-wn * t)
            v = (v0 - wn * (x0 + (v0 + wn * x0) * t)) * np.exp(-wn * t)
        else:
            # Overdamped (closed form via lambda1, lambda2)
            lam1 = -wn * (zeta + np.sqrt(zeta**2 - 1.0))
            lam2 = -wn * (zeta - np.sqrt(zeta**2 - 1.0))
            C2 = (v0 - lam2 * x0) / (lam1 - lam2)
            C1 = x0 - C2
            x = C1 * np.exp(lam1 * t) + C2 * np.exp(lam2 * t)
            v = C1 * lam1 * np.exp(lam1 * t) + C2 * lam2 * np.exp(lam2 * t)
        a = np.gradient(v, t)
        return x, v, a

    def frequency_response(self, omega: Union[float, np.ndarray], F0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        omega = np.atleast_1d(omega)
        X = F0 / np.sqrt((self.k - self.m * omega**2)**2 + (self.c * omega)**2)
        phase = -np.arctan2(self.c * omega, self.k - self.m * omega**2)
        return X, phase


class MDOFSystem:
    """Multi-degree-of-freedom vibrating system with modal analysis."""

    def __init__(self, M: np.ndarray, C: np.ndarray, K: np.ndarray):
        self.M = np.asarray(M, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.n = self.M.shape[0]

    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Undamped natural frequencies [rad/s] and mass-normalized mode shapes."""
        w2, phi = linalg.eigh(self.K, self.M)
        w2 = np.maximum(w2, 0.0)
        wn = np.sqrt(w2)
        for i in range(phi.shape[1]):
            norm = phi[:, i] @ self.M @ phi[:, i]
            phi[:, i] /= np.sqrt(norm)
        return wn, phi

    def state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Continuous-time state space: zdot = A z + B u, y = C z + D u."""
        n = self.n
        Minv = linalg.inv(self.M)
        Z = np.zeros((n, n))
        I = np.eye(n)
        A = np.block([[Z, I], [-Minv @ self.K, -Minv @ self.C]])
        B = np.block([[Z], [Minv]])
        C = np.block([[I, Z]])
        D = np.zeros((n, n))
        return A, B, C, D

    def time_response(self, t: np.ndarray, x0: np.ndarray, v0: np.ndarray,
                      u_func: Optional[Callable[[float], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """State-space integration via solve_ivp."""
        from scipy.integrate import solve_ivp
        A, B, C, D = self.state_space()
        z0 = np.concatenate([x0, v0])

        def rhs(tt, zz):
            force = u_func(tt) if u_func is not None else np.zeros(self.n)
            return A @ zz + B @ force

        sol = solve_ivp(rhs, (t[0], t[-1]), z0, t_eval=t, method="RK45", rtol=1e-8, atol=1e-10)
        return sol.t, sol.y


# -----------------------------------------------------------------------------
# 5. FATIGUE & FRACTURE MECHANICS
# -----------------------------------------------------------------------------
class SNCurve:
    """Basquin-type S-N curve: sigma_a = sigma_f' * (2N)^b."""

    def __init__(self, sigma_f_prime: float = 1000e6, b: float = -0.10,
                 endurance_limit: Optional[float] = None):
        self.sigma_f_prime = sigma_f_prime
        self.b = b
        self.endurance_limit = endurance_limit

    def stress_for_life(self, N: float) -> float:
        if N <= 0:
            raise ValueError("Life must be positive.")
        sigma = self.sigma_f_prime * (2.0 * N) ** self.b
        if self.endurance_limit is not None:
            sigma = max(sigma, self.endurance_limit)
        return sigma

    def life_for_stress(self, sigma_a: float) -> float:
        if sigma_a <= 0:
            return np.inf
        if self.endurance_limit is not None and sigma_a <= self.endurance_limit:
            return np.inf
        return 0.5 * (sigma_a / self.sigma_f_prime) ** (1.0 / self.b)

    def plot(self, N_min: float = 1e1, N_max: float = 1e7, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
        N = np.logspace(np.log10(N_min), np.log10(N_max), 200)
        sigma = np.array([self.stress_for_life(n) for n in N])
        ax.loglog(N, sigma / 1e6, label=f"{self.sigma_f_prime/1e6:.0f} MPa (2N)^({self.b:.2f})")
        ax.set_xlabel("Cycles to failure N [-]")
        ax.set_ylabel("Stress amplitude [MPa]")
        ax.set_title("S-N Curve (Basquin)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        return ax


def goodman_safety_factor(sa: float, sm: float, se: float, sut: float) -> float:
    """Goodman line: sa/se + sm/sut = 1/n."""
    denom = sa / se + sm / sut
    if denom <= 0:
        return np.inf
    return 1.0 / denom


def gerber_safety_factor(sa: float, sm: float, se: float, sut: float) -> float:
    """Gerber parabola: sa = se * (1 - (sm/sut)^2), scaled by n."""
    if sm == 0.0:
        return se / sa if sa > 0 else np.inf
    a = se * (sm / sut) ** 2
    b = sa
    c = -se
    disc = b**2 - 4 * a * c
    if disc < 0:
        return np.inf
    return (-b + np.sqrt(disc)) / (2 * a)


def plot_goodman_gerber(se: float = 200e6, sut: float = 400e6,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    sm = np.linspace(0, sut, 200)
    sa_goodman = se * (1.0 - sm / sut)
    sa_gerber = se * (1.0 - (sm / sut)**2)
    ax.plot(sm / 1e6, sa_goodman / 1e6, label="Goodman", lw=2)
    ax.plot(sm / 1e6, sa_gerber / 1e6, label="Gerber", lw=2, ls="--")
    ax.axvline(sut / 1e6, color="k", ls=":", label="Sut")
    ax.axhline(se / 1e6, color="k", ls=":", label="Se")
    ax.set_xlabel("Mean stress [MPa]")
    ax.set_ylabel("Alternating stress [MPa]")
    ax.set_title("Goodman / Gerber Fatigue Diagrams")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def paris_law_cycles(a0: float, ac: float, delta_sigma: float, C: float,
                     m: float, Y: float = 1.0) -> float:
    """Integrate Paris' law da/dN = C (Delta K)^m for center crack."""
    def integrand(a):
        dK = delta_sigma * np.sqrt(np.pi * a) * Y
        return 1.0 / (C * dK**m)
    N, _ = integrate.quad(integrand, a0, ac, epsabs=1e-14, epsrel=1e-12)
    return float(N)


# -----------------------------------------------------------------------------
# 6. STRUCTURAL OPTIMIZATION
# -----------------------------------------------------------------------------
def beam_rect_weight(x: np.ndarray, L: float, rho: float) -> float:
    b, h = x
    return rho * b * h * L


def beam_rect_stress(x: np.ndarray, L: float, P: float) -> float:
    b, h = x
    M = P * L / 4.0
    I = b * h**3 / 12.0
    return M * (h / 2.0) / I


def beam_rect_deflection(x: np.ndarray, L: float, P: float, E: float) -> float:
    b, h = x
    I = b * h**3 / 12.0
    return P * L**3 / (48.0 * E * I)


def optimize_beam_slsqp(L: float = 2.0, P: float = 10e3, rho: float = 7850.0,
                        E: float = 200e9, Sy: float = 250e6,
                        delta_lim: float = 5e-3,
                        b0: float = 0.05, h0: float = 0.10) -> Dict:
    """Gradient-based (SLSQP) weight minimization of rectangular simply supported beam."""
    def obj(x):
        return beam_rect_weight(x, L, rho)

    cons = [
        {"type": "ineq", "fun": lambda x: Sy - beam_rect_stress(x, L, P)},
        {"type": "ineq", "fun": lambda x: delta_lim - beam_rect_deflection(x, L, P, E)},
    ]
    bounds = [(1e-3, 0.5), (1e-3, 0.5)]
    res = minimize(obj, [b0, h0], method="SLSQP", bounds=bounds,
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    return {"x": res.x, "weight": res.fun, "success": res.success,
            "message": res.message, "stress": beam_rect_stress(res.x, L, P),
            "deflection": beam_rect_deflection(res.x, L, P, E)}


def optimize_beam_ga(L: float = 2.0, P: float = 10e3, rho: float = 7850.0,
                     E: float = 200e9, Sy: float = 250e6,
                     delta_lim: float = 5e-3) -> Dict:
    """Genetic-algorithm-like minimization via differential evolution with penalty."""

    def penalized(x):
        b, h = x
        if b <= 0 or h <= 0:
            return 1e12
        w = beam_rect_weight(x, L, rho)
        stress = beam_rect_stress(x, L, P)
        defl = beam_rect_deflection(x, L, P, E)
        pen = 0.0
        if stress > Sy:
            pen += 1e8 * (stress / Sy - 1.0)**2
        if defl > delta_lim:
            pen += 1e8 * (defl / delta_lim - 1.0)**2
        return w + pen

    bounds = [(1e-3, 0.5), (1e-3, 0.5)]
    res = differential_evolution(penalized, bounds, tol=1e-10, polish=True,
                                 maxiter=300, seed=42)
    x = res.x
    return {"x": x, "weight": beam_rect_weight(x, L, rho), "success": res.success,
            "message": res.message, "stress": beam_rect_stress(x, L, P),
            "deflection": beam_rect_deflection(x, L, P, E)}


# -----------------------------------------------------------------------------
# 7. ROTORDYNAMICS / CONTACT / FAILURE THEORIES
# -----------------------------------------------------------------------------
class JeffcottRotor:
    """Simple Jeffcott rotor with unbalance harmonic excitation."""

    def __init__(self, m: float = 50.0, k: float = 2e7, c: float = 500.0,
                 e: float = 0.0005):
        self.m = m
        self.k = k
        self.c = c
        self.e = e  # eccentricity [m]

    def critical_speed(self) -> float:
        return np.sqrt(self.k / self.m)

    def damping_ratio(self) -> float:
        return self.c / (2.0 * np.sqrt(self.k * self.m))

    def unbalance_response(self, omega: Union[float, np.ndarray]) -> np.ndarray:
        omega = np.atleast_1d(omega)
        denom = np.sqrt((self.k - self.m * omega**2)**2 + (self.c * omega)**2)
        return self.m * self.e * omega**2 / denom


def hertzian_contact_spheres(F: float, R1: float, R2: float,
                             E1: float, nu1: float, E2: float, nu2: float) -> Dict[str, float]:
    """Hertzian point contact between two spheres."""
    R = R1 * R2 / (R1 + R2)
    inv_E_star = (1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2
    E_star = 1.0 / inv_E_star
    a = (3.0 * F * R / (4.0 * E_star)) ** (1.0 / 3.0)
    p0 = 3.0 * F / (2.0 * np.pi * a**2)
    p_avg = F / (np.pi * a**2)
    z_tau_max = 0.48 * a
    tau_max = 0.31 * p0
    return {
        "contact_radius": a,
        "max_pressure": p0,
        "avg_pressure": p_avg,
        "E_star": E_star,
        "effective_radius": R,
        "depth_max_shear": z_tau_max,
        "max_shear_stress": tau_max,
    }


def failure_theories(s1: float, s2: float, s3: float,
                     Sy: float, Sut: Optional[float] = None) -> Dict[str, float]:
    """Evaluate Rankine, Tresca, Von Mises failure ratios."""
    s = np.sort([s1, s2, s3])
    s_min, s_mid, s_max = s
    tau_max = (s_max - s_min) / 2.0
    sigma_vm = np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
    out = {
        "max_principal_ratio": max(abs(s1), abs(s2), abs(s3)) / Sy,
        "tresca_ratio": tau_max / (0.5 * Sy),
        "von_mises_ratio": sigma_vm / Sy,
        "fail_max_principal": bool(max(abs(s1), abs(s2), abs(s3)) >= Sy),
        "fail_tresca": bool(tau_max >= 0.5 * Sy),
        "fail_von_mises": bool(sigma_vm >= Sy),
    }
    if Sut is not None:
        out["brittle_max_normal_ratio"] = max(abs(s_max), abs(s_min)) / Sut
        out["fail_brittle"] = bool(max(abs(s_max), abs(s_min)) >= Sut)
    return out


# -----------------------------------------------------------------------------
# 8. DEMONSTRATION & VISUALIZATION
# -----------------------------------------------------------------------------
def demo_fea_truss() -> None:
    print("\n" + "=" * 70)
    print("FEA Demo: 2D Truss (3-bar)")
    print("=" * 70)
    mat = Material(E=200e9, nu=0.3, rho=7850.0, Sy=250e6)
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    solver = FEASolver(num_nodes=4, dof_per_node=2)
    # Elements: bottom, left, diagonal
    elems = [
        TrussElement(0, 1, 5e-4, mat),
        TrussElement(0, 2, 5e-4, mat),
        TrussElement(1, 2, 5e-4 / np.sqrt(2.0), mat),
        TrussElement(1, 3, 5e-4, mat),
        TrussElement(2, 3, 5e-4, mat),
    ]
    for e in elems:
        solver.add_element(e)
    # Constraints: node 0 fixed (ux, uy), node 2 fixed ux (roller)
    solver.add_constraint(0, 0.0)   # node 0 ux
    solver.add_constraint(1, 0.0)   # node 0 uy
    solver.add_constraint(4, 0.0)   # node 2 ux
    # Load: node 3 downward 10 kN
    solver.add_force(7, -10e3)
    u = solver.solve(nodes)
    print("Nodal displacements [m]:")
    for i in range(4):
        print(f"  Node {i}: ux={u[2*i]: .6e}, uy={u[2*i+1]: .6e}")
    print("Reactions [N]:", solver.reactions())
    # Plot deformed vs undeformed
    fig, ax = plt.subplots(figsize=(6, 5))
    scale = 50.0
    for e in elems:
        i, j = e.node_i, e.node_j
        x = [nodes[i][0], nodes[j][0]]
        y = [nodes[i][1], nodes[j][1]]
        ax.plot(x, y, "k-", lw=2, alpha=0.4)
        xd = [nodes[i][0] + scale * u[2*i], nodes[j][0] + scale * u[2*j]]
        yd = [nodes[i][1] + scale * u[2*i+1], nodes[j][1] + scale * u[2*j+1]]
        ax.plot(xd, yd, "r-", lw=1.5)
    ax.scatter([n[0] for n in nodes.values()], [n[1] for n in nodes.values()],
               c="k", zorder=5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Truss: Undeformed (black) / Deformed (red, scaled)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")


def demo_fea_beam() -> None:
    print("\n" + "=" * 70)
    print("FEA Demo: Cantilever Beam (2D, 1 element)")
    print("=" * 70)
    mat = Material(E=210e9, Sy=300e6)
    nodes = {0: (0.0, 0.0), 1: (1.5, 0.0)}
    I = 1e-6  # m^4
    A = 5e-3  # m^2
    solver = FEASolver(num_nodes=2, dof_per_node=3)
    el = Beam2DElement(0, 1, A, I, mat)
    solver.add_element(el)
    # Clamp node 0: ux, uy, theta
    solver.add_constraint(0, 0.0)
    solver.add_constraint(1, 0.0)
    solver.add_constraint(2, 0.0)
    # Downward tip load at node 1 (uy dof = 4)
    solver.add_force(4, -5e3)
    u = solver.solve(nodes)
    print(f"Tip displacement uy = {u[4]: .6e} m")
    print(f"Tip rotation      = {u[5]: .6e} rad")
    f = el.end_forces(u, nodes)
    print(f"Fixed end forces/moments: N={f[0]: .1f} N, Vy={f[1]: .1f} N, Mz={f[2]: .1f} Nm")
    # Simple plot
    fig, ax = plt.subplots(figsize=(5, 4))
    scale = 500.0
    x = [nodes[0][0], nodes[1][0]]
    y = [nodes[0][1], nodes[1][1]]
    ax.plot(x, y, "k-", lw=3)
    xd = [nodes[0][0], nodes[1][0] + scale * u[4]]
    yd = [nodes[0][1], nodes[1][1] + scale * u[5] * 0.0]  # ignore rotation for drawing
    ax.plot(xd, yd, "r-", lw=2)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Cantilever tip load: undeformed vs deformed (scaled)")
    ax.grid(True, alpha=0.3)


def demo_stress_transformations() -> None:
    print("\n" + "=" * 70)
    print("Stress Transformation Demo")
    print("=" * 70)
    state = StressState2D(sx=60e6, sy=20e6, tau=30e6)
    s1, s2 = state.principal_stresses()
    print(f"Given: sx={state.sx/1e6:.1f} MPa, sy={state.sy/1e6:.1f} MPa, tau={state.tau/1e6:.1f} MPa")
    print(f"Principal stresses: s1={s1/1e6:.2f} MPa, s2={s2/1e6:.2f} MPa")
    print(f"Max in-plane shear: {state.max_shear_inplane()/1e6:.2f} MPa")
    print(f"Absolute max shear: {state.max_shear_abs()/1e6:.2f} MPa")
    print(f"Von Mises: {state.von_mises()/1e6:.2f} MPa")
    print(f"Tresca equiv: {state.tresca()/1e6:.2f} MPa")
    ratios = state.failure_ratios(Sy=250e6, Sut=400e6)
    print(f"Failure ratios: VM={ratios['von_mises_ratio']:.3f}, Tresca={ratios['tresca_ratio']:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_mohr_circle(state, ax=axes[0])
    # Rotated stress state
    theta = np.deg2rad(30)
    state_r = state.rotate(theta)
    axes[1].bar(["sx", "sy", "tau"], [state.sx/1e6, state.sy/1e6, state.tau/1e6],
                color="steelblue", alpha=0.7, label="Original")
    axes[1].bar(["sx'", "sy'", "tau'"], [state_r.sx/1e6, state_r.sy/1e6, state_r.tau/1e6],
                color="indianred", alpha=0.7, label="Rotated 30 deg")
    axes[1].set_ylabel("Stress [MPa]")
    axes[1].set_title("Stress Transformation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()


def demo_vibrations() -> None:
    print("\n" + "=" * 70)
    print("Vibration Demo: SDOF & MDOF")
    print("=" * 70)
    sdof = SDOFSystem(m=10.0, k=1e5, c=200.0)
    print(f"SDOF: wn={sdof.wn:.1f} rad/s, zeta={sdof.zeta:.4f}, wd={sdof.wd:.1f} rad/s")
    t = np.linspace(0, 5, 1000)
    x, v, a = sdof.free_response(t, x0=0.05, v0=0.0)
    omegas = np.linspace(0, 2 * sdof.wn, 400)
    X, phase = sdof.frequency_response(omegas, F0=100.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(t, x, "b-")
    axes[0, 0].set_title("SDOF Free Response (x0=0.05 m)")
    axes[0, 0].set_xlabel("t [s]"); axes[0, 0].set_ylabel("x [m]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(omegas, X / 1e-3, "r-")
    axes[0, 1].axvline(sdof.wn, color="k", ls="--", label="wn")
    axes[0, 1].set_title("SDOF Frequency Response (F0=100 N)")
    axes[0, 1].set_xlabel("omega [rad/s]"); axes[0, 1].set_ylabel("|X| [mm]")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    # MDOF: 2-DOF spring mass
    m1, m2 = 10.0, 15.0
    k1, k2 = 2e5, 1.5e5
    M = np.diag([m1, m2])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    C = np.zeros((2, 2))
    mdof = MDOFSystem(M, C, K)
    wn, phi = mdof.modal_analysis()
    print(f"MDOF natural frequencies: wn={[f'{w:.1f}' for w in wn]} rad/s")
    print("Mode shapes (mass normalized):\n", np.round(phi, 4))

    axes[1, 0].bar([1, 2], wn, color="seagreen")
    axes[1, 0].set_xticks([1, 2]); axes[1, 0].set_xlabel("Mode")
    axes[1, 0].set_ylabel("Natural frequency [rad/s]")
    axes[1, 0].set_title("MDOF Modal Frequencies")
    axes[1, 0].grid(True, alpha=0.3)

    # Time response of MDOF initial displacement
    t2 = np.linspace(0, 5, 1000)
    t_out, z = mdof.time_response(t2, x0=np.array([0.02, 0.0]), v0=np.zeros(2))
    axes[1, 1].plot(t_out, z[0, :], label="m1")
    axes[1, 1].plot(t_out, z[1, :], label="m2")
    axes[1, 1].set_title("MDOF Free Response")
    axes[1, 1].set_xlabel("t [s]"); axes[1, 1].set_ylabel("Displacement [m]")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()
    plt.tight_layout()


def demo_fatigue() -> None:
    print("\n" + "=" * 70)
    print("Fatigue & Fracture Demo")
    print("=" * 70)
    sn = SNCurve(sigma_f_prime=900e6, b=-0.09, endurance_limit=250e6)
    Ns = [1e3, 1e5, 1e6, 1e7]
    for N in Ns:
        print(f"N={N:.0e} cycles -> sigma_a={sn.stress_for_life(N)/1e6:.1f} MPa")
    print(f"Life at 300 MPa: {sn.life_for_stress(300e6):.2e} cycles")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sn.plot(ax=axes[0])
    plot_goodman_gerber(se=250e6, sut=450e6, ax=axes[1])
    plt.tight_layout()

    # Paris law
    N_paris = paris_law_cycles(a0=0.005, ac=0.02, delta_sigma=80e6,
                               C=1e-11, m=3.0, Y=1.12)
    print(f"Paris law cycles to grow 5mm -> 20mm: {N_paris:.1f} cycles")


def demo_optimization() -> None:
    print("\n" + "=" * 70)
    print("Structural Optimization Demo: Rectangular Beam")
    print("=" * 70)
    res_grad = optimize_beam_slsqp()
    res_ga = optimize_beam_ga()
    print(f"SLSQP: b={res_grad['x'][0]*1000:.1f} mm, h={res_grad['x'][1]*1000:.1f} mm, "
          f"W={res_grad['weight']:.3f} kg, stress={res_grad['stress']/1e6:.1f} MPa, "
          f"defl={res_grad['deflection']*1000:.2f} mm")
    print(f"GA   : b={res_ga['x'][0]*1000:.1f} mm, h={res_ga['x'][1]*1000:.1f} mm, "
          f"W={res_ga['weight']:.3f} kg, stress={res_ga['stress']/1e6:.1f} MPa, "
          f"defl={res_ga['deflection']*1000:.2f} mm")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["SLSQP", "GA"], [res_grad['weight'], res_ga['weight']],
           color=["steelblue", "indianred"])
    ax.set_ylabel("Weight [kg]")
    ax.set_title("Beam Weight Minimization")
    ax.grid(True, alpha=0.3)


def demo_rotor_contact_failure() -> None:
    print("\n" + "=" * 70)
    print("Rotordynamics / Contact / Failure Demo")
    print("=" * 70)
    rotor = JeffcottRotor(m=100.0, k=4e8, c=2e4, e=0.0002)
    print(f"Critical speed: {rotor.critical_speed():.1f} rad/s "
          f"({rotor.critical_speed()*60/(2*np.pi):.0f} rpm), zeta={rotor.damping_ratio():.4f}")
    omega = np.linspace(0, 1.5 * rotor.critical_speed(), 400)
    X = rotor.unbalance_response(omega)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(omega, X * 1000, "m-", lw=2)
    ax.axvline(rotor.critical_speed(), color="k", ls="--", label="Critical speed")
    ax.set_xlabel("Rotor speed [rad/s]")
    ax.set_ylabel("Unbalance response [mm]")
    ax.set_title("Jeffcott Rotor Unbalance Response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    contact = hertzian_contact_spheres(F=5e3, R1=0.05, R2=0.05,
                                       E1=200e9, nu1=0.3, E2=200e9, nu2=0.3)
    print("Hertzian contact (spheres):")
    for k, v in contact.items():
        print(f"  {k}: {v:.4e}" if isinstance(v, float) else f"  {k}: {v}")

    ft = failure_theories(s1=180e6, s2=40e6, s3=-20e6, Sy=250e6, Sut=400e6)
    print("Failure theory check (s1=180, s2=40, s3=-20 MPa):")
    for k, v in ft.items():
        print(f"  {k}: {v}")
    plt.tight_layout()


def main() -> None:
    plt.close("all")
    demo_fea_truss()
    demo_fea_beam()
    demo_stress_transformations()
    demo_vibrations()
    demo_fatigue()
    demo_optimization()
    demo_rotor_contact_failure()
    print("\nAll demonstrations complete. Close plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()