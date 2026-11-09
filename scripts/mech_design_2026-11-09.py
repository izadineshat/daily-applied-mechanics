import math
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union

import numpy as np
from scipy import linalg as la
from scipy import optimize as opt
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# =============================================================================
# STRESS & STRAIN TRANSFORMATIONS
# =============================================================================

@dataclass
class StressState2D:
    sx: float = 0.0
    sy: float = 0.0
    txy: float = 0.0


@dataclass
class StressState3D:
    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0
    txy: float = 0.0
    tyz: float = 0.0
    tzx: float = 0.0


def principal_stresses_2d(sx: float, sy: float, txy: float) -> Tuple[float, float, float]:
    """Return principal stresses (s1, s2) and principal plane angle (rad) for 2D state."""
    avg = 0.5 * (sx + sy)
    rad = math.sqrt(0.25 * (sx - sy) ** 2 + txy ** 2)
    s1 = avg + rad
    s2 = avg - rad
    theta_p = 0.5 * math.atan2(2.0 * txy, sx - sy)
    return s1, s2, theta_p


def mohrs_circle_2d(sx: float, sy: float, txy: float) -> Dict[str, float]:
    """Mohr's circle parameters for a 2D stress state."""
    center = 0.5 * (sx + sy)
    radius = math.sqrt(0.25 * (sx - sy) ** 2 + txy ** 2)
    s1, s2, theta_p = principal_stresses_2d(sx, sy, txy)
    return {
        "center": center,
        "radius": radius,
        "sigma_1": s1,
        "sigma_2": s2,
        "tau_max": radius,
        "theta_p": theta_p,
    }


def von_mises_stress_2d(sx: float, sy: float, txy: float) -> float:
    """Von Mises equivalent stress for plane stress (sz = 0)."""
    return math.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * txy ** 2)


def von_mises_stress_3d(sx: float, sy: float, sz: float,
                        txy: float, tyz: float, tzx: float) -> float:
    """Von Mises equivalent stress for a general 3D state."""
    term = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
    shear = 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
    return math.sqrt(term + shear)


def principal_stresses_3d(sx: float, sy: float, sz: float,
                          txy: float, tyz: float, tzx: float) -> Tuple[float, float, float]:
    """Return ordered principal stresses s1 >= s2 >= s3 from 3D stress tensor."""
    T = np.array([[sx, txy, tzx],
                  [txy, sy, tyz],
                  [tzx, tyz, sz]], dtype=float)
    vals = la.eigvalsh(T)  # ascending
    return float(vals[2]), float(vals[1]), float(vals[0])


def tresca_equivalent_stress(sx: float, sy: float, sz: float = 0.0,
                             txy: float = 0.0, tyz: float = 0.0, tzx: float = 0.0) -> float:
    """Tresca equivalent stress = sigma_max - sigma_min."""
    s1, s2, s3 = principal_stresses_3d(sx, sy, sz, txy, tyz, tzx)
    return s1 - s3


def factor_of_safety_von_mises(sigma_vm: float, yield_strength: float) -> float:
    if sigma_vm <= 0.0:
        return math.inf
    return yield_strength / sigma_vm


def factor_of_safety_tresca(sigma_tresca: float, yield_strength: float) -> float:
    if sigma_tresca <= 0.0:
        return math.inf
    return yield_strength / sigma_tresca


# =============================================================================
# FINITE ELEMENT ANALYSIS (TRUSS, 2D BEAM, CST PLATE)
# =============================================================================

class TrussModel:
    """2D linear elastic truss FEM solver (2 DOF per node)."""

    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.elements: List[Tuple[int, int, float, float]] = []  # ni, nj, E, A
        self.loads: Dict[int, Tuple[float, float]] = {}
        self.fixed: Dict[int, Tuple[bool, bool]] = {}

    def add_node(self, nid: int, x: float, y: float) -> None:
        self.nodes[nid] = (x, y)

    def add_element(self, ni: int, nj: int, E: float, A: float) -> None:
        self.elements.append((ni, nj, E, A))

    def add_load(self, nid: int, fx: float, fy: float) -> None:
        self.loads[nid] = (fx, fy)

    def fix_node(self, nid: int, ux: bool = True, uy: bool = True) -> None:
        self.fixed[nid] = (ux, uy)

    def _assemble(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        ids = sorted(self.nodes.keys())
        index = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        K = np.zeros((2 * n, 2 * n))
        F = np.zeros(2 * n)

        for nid, (fx, fy) in self.loads.items():
            i = index[nid]
            F[2 * i] += fx
            F[2 * i + 1] += fy

        for ni, nj, E, A in self.elements:
            xi, yi = self.nodes[ni]
            xj, yj = self.nodes[nj]
            dx = xj - xi
            dy = yj - yi
            L = math.hypot(dx, dy)
            if L <= 0.0:
                raise ValueError(f"Zero-length truss element {ni}-{nj}")
            c = dx / L
            s = dy / L
            T = np.array([[c, s, 0.0, 0.0],
                          [0.0, 0.0, c, s]])
            k_loc = (E * A / L) * np.array([[1.0, -1.0],
                                            [-1.0, 1.0]])
            k_glob = T.T @ k_loc @ T
            dofs = [2 * index[ni], 2 * index[ni] + 1,
                    2 * index[nj], 2 * index[nj] + 1]
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += k_glob[a, b]
        return K, F, index

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        K, F, index = self._assemble()
        free: List[int] = []
        fixed: List[int] = []
        for nid, i in index.items():
            ux_fix, uy_fix = self.fixed.get(nid, (False, False))
            if not ux_fix:
                free.append(2 * i)
            else:
                fixed.append(2 * i)
            if not uy_fix:
                free.append(2 * i + 1)
            else:
                fixed.append(2 * i + 1)

        free = np.array(free, dtype=int)
        fixed = np.array(fixed, dtype=int)

        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        uf = la.solve(Kff, Ff, assume_a="sym")

        u = np.zeros_like(F)
        u[free] = uf
        R = K @ u - F
        return u, R, index


class Beam2DElement:
    """Euler-Bernoulli 2D beam element (axial + bending), 3 DOF per node."""

    def __init__(self, ni: int, nj: int, E: float, A: float, I: float,
                 coords: Dict[int, Tuple[float, float]]) -> None:
        self.ni = ni
        self.nj = nj
        self.E = E
        self.A = A
        self.I = I
        xi, yi = coords[ni]
        xj, yj = coords[nj]
        dx = xj - xi
        dy = yj - yi
        self.L = math.hypot(dx, dy)
        self.c = dx / self.L
        self.s = dy / self.L

    def local_stiffness(self) -> np.ndarray:
        E, A, I, L = self.E, self.A, self.I, self.L
        k = np.zeros((6, 6))
        k[0, 0] = E * A / L
        k[0, 3] = -E * A / L
        k[3, 0] = -E * A / L
        k[3, 3] = E * A / L
        EI_L3 = E * I / L ** 3
        bend = np.array([
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2],
        ]) * EI_L3
        idx = [1, 2, 4, 5]
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] = bend[a, b]
        return k

    def transformation_matrix(self) -> np.ndarray:
        c, s = self.c, self.s
        R = np.array([[c, s, 0.0],
                      [-s, c, 0.0],
                      [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        return T

    def global_stiffness(self) -> np.ndarray:
        return self.transformation_matrix().T @ self.local_stiffness() @ self.transformation_matrix()


class Beam2DModel:
    """2D linear elastic beam FEM solver (3 DOF per node: ux, uy, rotz)."""

    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.elements: List[Tuple[int, int, float, float, float]] = []  # ni, nj, E, A, I
        self.loads: Dict[int, Tuple[float, float, float]] = {}
        self.fixed: Dict[int, Tuple[bool, bool, bool]] = {}

    def add_node(self, nid: int, x: float, y: float) -> None:
        self.nodes[nid] = (x, y)

    def add_element(self, ni: int, nj: int, E: float, A: float, I: float) -> None:
        self.elements.append((ni, nj, E, A, I))

    def add_load(self, nid: int, fx: float, fy: float, mz: float = 0.0) -> None:
        self.loads[nid] = (fx, fy, mz)

    def fix_node(self, nid: int, ux: bool = True, uy: bool = True, rot: bool = True) -> None:
        self.fixed[nid] = (ux, uy, rot)

    def _assemble(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        ids = sorted(self.nodes.keys())
        index = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        ndof = 3 * n
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        for nid, (fx, fy, mz) in self.loads.items():
            i = index[nid]
            F[3 * i] += fx
            F[3 * i + 1] += fy
            F[3 * i + 2] += mz

        for ni, nj, E, A, I in self.elements:
            elem = Beam2DElement(ni, nj, E, A, I, self.nodes)
            k = elem.global_stiffness()
            dofs = [3 * index[ni], 3 * index[ni] + 1, 3 * index[ni] + 2,
                    3 * index[nj], 3 * index[nj] + 1, 3 * index[nj] + 2]
            for a in range(6):
                for b in range(6):
                    K[dofs[a], dofs[b]] += k[a, b]
        return K, F, index

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        K, F, index = self._assemble()
        free: List[int] = []
        fixed: List[int] = []
        for nid, i in index.items():
            bcs = self.fixed.get(nid, (False, False, False))
            for comp, fix in enumerate(bcs):
                dof = 3 * i + comp
                if fix:
                    fixed.append(dof)
                else:
                    free.append(dof)
        free = np.array(free, dtype=int)
        fixed = np.array(fixed, dtype=int)
        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        uf = la.solve(Kff, Ff, assume_a="sym")
        u = np.zeros_like(F)
        u[free] = uf
        R = K @ u - F
        return u, R, index


class CSTElement:
    """Constant Strain Triangle (CST) element for plane stress/strain."""

    def __init__(self, node_ids: Tuple[int, int, int], E: float, nu: float,
                 thickness: float, coords: Dict[int, Tuple[float, float]]) -> None:
        self.node_ids = node_ids
        self.E = E
        self.nu = nu
        self.t = thickness
        self.coords = coords
        (x1, y1), (x2, y2), (x3, y3) = [coords[n] for n in node_ids]
        detJ = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if abs(detJ) < 1e-14:
            raise ValueError("Degenerate triangle with zero area")
        self.A = 0.5 * abs(detJ)
        self.b = [y2 - y3, y3 - y1, y1 - y2]
        self.c = [x3 - x2, x1 - x3, x2 - x1]

    def stiffness(self) -> np.ndarray:
        E, nu, t, A = self.E, self.nu, self.t, self.A
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = self.b[i]
            B[1, 2 * i + 1] = self.c[i]
            B[2, 2 * i] = self.c[i]
            B[2, 2 * i + 1] = self.b[i]
        B /= (2.0 * A)
        D = E / (1.0 - nu ** 2) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
        return t * A * (B.T @ D @ B)


class PlateModel:
    """Simple CST plate assembly (plane stress, 2 DOF per node)."""

    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.elements: List[Tuple[int, int, int, float, float, float]] = []  # n1,n2,n3,E,nu,t
        self.loads: Dict[int, Tuple[float, float]] = {}
        self.fixed: Dict[int, Tuple[bool, bool]] = {}

    def add_node(self, nid: int, x: float, y: float) -> None:
        self.nodes[nid] = (x, y)

    def add_element(self, n1: int, n2: int, n3: int, E: float, nu: float, t: float) -> None:
        self.elements.append((n1, n2, n3, E, nu, t))

    def add_load(self, nid: int, fx: float, fy: float) -> None:
        self.loads[nid] = (fx, fy)

    def fix_node(self, nid: int, ux: bool = True, uy: bool = True) -> None:
        self.fixed[nid] = (ux, uy)

    def _assemble(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        ids = sorted(self.nodes.keys())
        index = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        K = np.zeros((2 * n, 2 * n))
        F = np.zeros(2 * n)
        for nid, (fx, fy) in self.loads.items():
            i = index[nid]
            F[2 * i] += fx
            F[2 * i + 1] += fy
        for n1, n2, n3, E, nu, t in self.elements:
            elem = CSTElement((n1, n2, n3), E, nu, t, self.nodes)
            k = elem.stiffness()
            dofs = [2 * index[n1], 2 * index[n1] + 1,
                    2 * index[n2], 2 * index[n2] + 1,
                    2 * index[n3], 2 * index[n3] + 1]
            for a in range(6):
                for b in range(6):
                    K[dofs[a], dofs[b]] += k[a, b]
        return K, F, index

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        K, F, index = self._assemble()
        free: List[int] = []
        fixed: List[int] = []
        for nid, i in index.items():
            bcs = self.fixed.get(nid, (False, False))
            for comp, fix in enumerate(bcs):
                dof = 2 * i + comp
                if fix:
                    fixed.append(dof)
                else:
                    free.append(dof)
        free = np.array(free, dtype=int)
        fixed = np.array(fixed, dtype=int)
        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        uf = la.solve(Kff, Ff, assume_a="sym")
        u = np.zeros_like(F)
        u[free] = uf
        R = K @ u - F
        return u, R, index


# =============================================================================
# MECHANICAL VIBRATIONS
# =============================================================================

class SDOFSystem:
    """Single-degree-of-freedom mass-spring-damper system."""

    def __init__(self, m: float, k: float, c: float = 0.0) -> None:
        self.m = m
        self.k = k
        self.c = c

    @property
    def omega_n(self) -> float:
        return math.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * math.sqrt(self.k * self.m))

    @property
    def omega_d(self) -> float:
        if self.zeta < 1.0:
            return self.omega_n * math.sqrt(1.0 - self.zeta ** 2)
        return 0.0

    def free_response(self, t: np.ndarray, x0: float, v0: float) -> np.ndarray:
        wn = self.omega_n
        z = self.zeta
        if z < 1.0:
            wd = self.omega_d
            alpha = -z * wn
            x = np.exp(alpha * t) * (x0 * np.cos(wd * t) +
                                     (v0 - alpha * x0) / wd * np.sin(wd * t))
        elif z == 1.0:
            x = (x0 + (v0 + wn * x0) * t) * np.exp(-wn * t)
        else:
            lam1 = -z * wn + wn * math.sqrt(z ** 2 - 1.0)
            lam2 = -z * wn - wn * math.sqrt(z ** 2 - 1.0)
            C2 = (v0 - lam1 * x0) / (lam2 - lam1)
            C1 = x0 - C2
            x = C1 * np.exp(lam1 * t) + C2 * np.exp(lam2 * t)
        return x

    def frequency_response(self, omega: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        denom = np.sqrt((self.k - self.m * omega ** 2) ** 2 + (self.c * omega) ** 2)
        X = 1.0 / denom
        phi = np.arctan2(self.c * omega, self.k - self.m * omega ** 2)
        return X, phi


class MDOFSystem:
    """Multi-degree-of-freedom linear vibrating system."""

    def __init__(self, M: np.ndarray, K: np.ndarray, C: Optional[np.ndarray] = None) -> None:
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.C = np.asarray(C, dtype=float) if C is not None else np.zeros_like(self.M)

    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve generalized eigenproblem K phi = omega^2 M phi."""
        if not np.allclose(self.M, self.M.T):
            raise ValueError("Mass matrix must be symmetric")
        w2, phi = la.eigh(self.K, self.M)
        w2 = np.maximum(w2, 0.0)
        omega = np.sqrt(w2)
        # Mass-normalize modes
        for i in range(phi.shape[1]):
            m_norm = phi[:, i] @ self.M @ phi[:, i]
            phi[:, i] /= math.sqrt(m_norm)
        return omega, phi

    def state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.M.shape[0]
        Z = np.zeros((n, n))
        I = np.eye(n)
        Minv = la.inv(self.M)
        A = np.block([[Z, I],
                      [-Minv @ self.K, -Minv @ self.C]])
        B = np.block([[Z],
                      [Minv]])
        Cmat = np.block([[I, Z]])
        D = np.zeros((n, n))
        return A, B, Cmat, D

    def free_response_state_space(self, t: np.ndarray,
                                  x0: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A, B, Cmat, D = self.state_space()
        z0 = np.concatenate([x0, v0])

        def rhs(tt, zz):
            return A @ zz

        sol = solve_ivp(rhs, (t[0], t[-1]), z0, t_eval=t, method="RK45", rtol=1e-8, atol=1e-10)
        z = sol.y
        return z[: self.M.shape[0], :], z[self.M.shape[0]:, :]


# =============================================================================
# FATIGUE & FRACTURE MECHANICS
# =============================================================================

class SNCurve:
    """Basquin strain-life / stress-life curve: sigma_a = sigma_f_prime * (2N)^b."""

    def __init__(self, sigma_f_prime: float, b: float) -> None:
        self.sigma_f_prime = sigma_f_prime
        self.b = b

    def stress_for_life(self, N: float) -> float:
        return self.sigma_f_prime * (2.0 * N) ** self.b

    def cycles_to_failure(self, sigma_a: float) -> float:
        if sigma_a <= 0.0:
            return math.inf
        return 0.5 * (sigma_a / self.sigma_f_prime) ** (1.0 / self.b)


def goodman_factor_of_safety(sigma_a: float, sigma_m: float, Sut: float, Se: float) -> float:
    denom = sigma_a / Se + sigma_m / Sut
    return 1.0 / denom if denom > 0 else math.inf


def gerber_factor_of_safety(sigma_a: float, sigma_m: float, Sut: float, Se: float) -> float:
    a = (sigma_m / Sut) ** 2
    b = sigma_a / Se
    if a == 0.0:
        return 1.0 / b if b > 0 else math.inf
    disc = b * b + 4.0 * a
    return (-b + math.sqrt(disc)) / (2.0 * a)


def paris_law(dK: float, C: float, m: float) -> float:
    """da/dN = C * dK^m."""
    return C * dK ** m


def cycles_to_crack_growth(a0: float, ac: float, delta_sigma: float,
                           Y: float, C: float, m: float) -> float:
    """Integrate Paris' law from a0 to ac under constant amplitude loading."""
    def integrand(a: float) -> float:
        dK = Y * delta_sigma * math.sqrt(math.pi * a)
        return 1.0 / paris_law(dK, C, m)
    N, _ = quad(integrand, a0, ac, epsabs=1e-12, epsrel=1e-10, limit=200)
    return N


def stress_intensity_factor(delta_sigma: float, a: float, Y: float = 1.0) -> float:
    return Y * delta_sigma * math.sqrt(math.pi * a)


def fracture_check(K: float, K_Ic: float) -> Dict[str, float]:
    return {"K": K, "K_Ic": K_Ic, "margin": K_Ic - K, "safe": K <= K_Ic}


# =============================================================================
# STRUCTURAL OPTIMIZATION (BEAM SIZING)
# =============================================================================

class RectangularBeamDesign:
    """Simply supported rectangular beam with central point load.

    Design variables: width b, height h.
    Constraints: bending stress, deflection.
    Objective: minimize weight.
    """

    def __init__(self, L: float, P: float, E: float, rho: float,
                 sigma_yield: float, delta_allow: float, fs: float = 1.5) -> None:
        self.L = L
        self.P = P
        self.E = E
        self.rho = rho
        self.sigma_yield = sigma_yield
        self.delta_allow = delta_allow
        self.fs = fs

    def weight(self, x: np.ndarray) -> float:
        b, h = x
        return self.rho * b * h * self.L

    def I(self, x: np.ndarray) -> float:
        b, h = x
        return b * h ** 3 / 12.0

    def c(self, x: np.ndarray) -> float:
        return x[1] / 2.0

    def max_moment(self) -> float:
        return self.P * self.L / 4.0

    def stress(self, x: np.ndarray) -> float:
        return self.max_moment() * self.c(x) / self.I(x)

    def deflection(self, x: np.ndarray) -> float:
        return self.P * self.L ** 3 / (48.0 * self.E * self.I(x))

    def constraint_stress(self, x: np.ndarray) -> float:
        return self.sigma_yield / self.fs - self.stress(x)

    def constraint_deflection(self, x: np.ndarray) -> float:
        return self.delta_allow - self.deflection(x)

    def constraint_positive(self, x: np.ndarray) -> float:
        return min(x[0], x[1]) - 1e-4


def optimize_beam_gradient(design: RectangularBeamDesign,
                           x0: np.ndarray,
                           bounds: List[Tuple[float, float]]) -> opt.OptimizeResult:
    cons = [
        {"type": "ineq", "fun": design.constraint_stress},
        {"type": "ineq", "fun": design.constraint_deflection},
        {"type": "ineq", "fun": design.constraint_positive},
    ]
    res = opt.minimize(design.weight, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-10, "maxiter": 500})
    return res


def tournament_selection(pop: np.ndarray, fitness: np.ndarray, k: int = 3) -> np.ndarray:
    idx = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop[best]


def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    alpha = np.random.rand()
    return alpha * p1 + (1.0 - alpha) * p2


def mutate(child: np.ndarray, bounds: List[Tuple[float, float]], rate: float) -> np.ndarray:
    for i in range(len(child)):
        if np.random.rand() < rate:
            lo, hi = bounds[i]
            child[i] = np.random.uniform(lo, hi)
    return child


def optimize_beam_ga(objective: Callable[[np.ndarray], float],
                     bounds: List[Tuple[float, float]],
                     n_vars: int,
                     pop_size: int = 60,
                     generations: int = 150,
                     mutation_rate: float = 0.15,
                     elitism: int = 2,
                     seed: Optional[int] = 42) -> Tuple[np.ndarray, float]:
    if seed is not None:
        np.random.seed(seed)
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, n_vars))
    best_hist: List[float] = []
    for _ in range(generations):
        fitness = np.array([objective(ind) for ind in pop])
        order = np.argsort(fitness)
        new_pop = [pop[i] for i in order[:elitism]]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            child = crossover(p1, p2)
            child = mutate(child, bounds, mutation_rate)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = np.array([objective(ind) for ind in pop])
        best_hist.append(float(np.min(fitness)))
    best_idx = int(np.argmin(fitness))
    return pop[best_idx], float(fitness[best_idx]), best_hist


def penalty_objective(x: np.ndarray, design: RectangularBeamDesign) -> float:
    w = design.weight(x)
    pen = 0.0
    for g in [design.constraint_stress(x), design.constraint_deflection(x),
              design.constraint_positive(x)]:
        if g < 0:
            pen += 1e6 * (-g) ** 2
    return w + pen


# =============================================================================
# ROTORDYNAMICS (JEFFCOTT ROTOR)
# =============================================================================

class JeffcottRotor:
    """Jeffcott rotor: lumped mass on a massless shaft with damping and eccentricity."""

    def __init__(self, m: float, k: float, c: float, e: float) -> None:
        self.m = m
        self.k = k
        self.c = c
        self.e = e

    @property
    def omega_n(self) -> float:
        return math.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * math.sqrt(self.k * self.m))

    def unbalanced_force(self, omega: float) -> float:
        return self.m * self.e * omega ** 2

    def amplitude(self, omega: float) -> float:
        den = math.sqrt((self.k - self.m * omega ** 2) ** 2 + (self.c * omega) ** 2)
        return self.m * self.e * omega ** 2 / den

    def phase(self, omega: float) -> float:
        return math.atan2(self.c * omega, self.k - self.m * omega ** 2)

    def frequency_response(self, omega_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([self.amplitude(w) for w in omega_array])
        phi = np.array([self.phase(w) for w in omega_array])
        return X, phi


# =============================================================================
# CONTACT MECHANICS (HERTZIAN)
# =============================================================================

class HertzContact:
    @staticmethod
    def sphere_on_flat(force: float, radius: float, E: float, nu: float,
                       radius2: float = math.inf) -> Dict[str, float]:
        if radius2 == math.inf:
            R = radius
        else:
            R = 1.0 / (1.0 / radius + 1.0 / radius2)
        inv_E_star = 2.0 * (1.0 - nu ** 2) / E
        E_star = 1.0 / inv_E_star
        a = (3.0 * force * R / (4.0 * E_star)) ** (1.0 / 3.0)
        p0 = 3.0 * force / (2.0 * math.pi * a ** 2)
        p_avg = force / (math.pi * a ** 2)
        tau_max = 0.31 * p0
        z_tau_max = 0.48 * a
        return {
            "effective_radius": R,
            "effective_modulus": E_star,
            "contact_radius": a,
            "max_pressure": p0,
            "avg_pressure": p_avg,
            "max_shear_stress": tau_max,
            "depth_max_shear": z_tau_max,
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_fem() -> None:
    print("=" * 70)
    print("FINITE ELEMENT ANALYSIS DEMONSTRATIONS")
    print("=" * 70)

    # ---- Truss: simple 2-bar structure ----
    truss = TrussModel()
    truss.add_node(1, 0.0, 0.0)
    truss.add_node(2, 1.0, 0.0)
    truss.add_node(3, 1.0, 1.0)
    E_st = 210e9
    A_truss = 5e-4
    truss.add_element(1, 2, E_st, A_truss)
    truss.add_element(2, 3, E_st, A_truss)
    truss.add_element(1, 3, E_st, A_truss)
    truss.fix_node(1, True, True)
    truss.fix_node(2, False, True)
    truss.add_load(3, 0.0, -10000.0)
    u, R, idx = truss.solve()
    print("\n[Truss] Nodal displacements (ux, uy):")
    for nid, i in idx.items():
        print(f"  Node {nid}: {u[2*i]:.6e}, {u[2*i+1]:.6e} m")
    print(f"  Max displacement norm: {np.linalg.norm(u):.6e} m")

    # ---- Beam: cantilever with end point load ----
    beam = Beam2DModel()
    beam.add_node(1, 0.0, 0.0)
    beam.add_node(2, 2.0, 0.0)
    E_b = 210e9
    A_b = 2e-3
    I_b = 1.6e-6
    beam.add_element(1, 2, E_b, A_b, I_b)
    beam.fix_node(1, True, True, True)
    beam.add_load(2, 0.0, -5000.0, 0.0)
    u_b, R_b, idx_b = beam.solve()
    print("\n[Beam] Tip displacement (ux, uy, rotz):")
    i = idx_b[2]
    print(f"  ux={u_b[3*i]:.6e} m, uy={u_b[3*i+1]:.6e} m, rot={u_b[3*i+2]:.6e} rad")

    # ---- Plate: single CST element under shear ----
    plate = PlateModel()
    plate.add_node(1, 0.0, 0.0)
    plate.add_node(2, 0.5, 0.0)
    plate.add_node(3, 0.0, 0.5)
    plate.add_element(1, 2, 3, E_st, 0.3, 0.01)
    plate.fix_node(1, True, True)
    plate.add_load(2, 1000.0, 0.0)
    plate.add_load(3, 0.0, 1000.0)
    u_p, R_p, idx_p = plate.solve()
    print("\n[Plate CST] Displacements:")
    for nid, i in idx_p.items():
        print(f"  Node {nid}: ux={u_p[2*i]:.6e}, uy={u_p[2*i+1]:.6e} m")


def demo_stress() -> None:
    print("\n" + "=" * 70)
    print("STRESS & STRAIN TRANSFORMATIONS")
    print("=" * 70)
    sx, sy, txy = 80.0, 20.0, 45.0  # MPa
    circle = mohrs_circle_2d(sx, sy, txy)
    print(f"\n2D Stress state: sx={sx}, sy={sy}, txy={txy} MPa")
    print(f"  Principal stresses: s1={circle['sigma_1']:.2f}, s2={circle['sigma_2']:.2f} MPa")
    print(f"  Max shear: {circle['tau_max']:.2f} MPa, theta_p={math.degrees(circle['theta_p']):.2f} deg")
    vm = von_mises_stress_2d(sx, sy, txy)
    tres = tresca_equivalent_stress(sx, sy, 0.0, txy)
    print(f"  Von Mises: {vm:.2f} MPa")
    print(f"  Tresca eq.: {tres:.2f} MPa")
    Sy = 250.0
    print(f"  FoS (von Mises): {factor_of_safety_von_mises(vm, Sy):.3f}")
    print(f"  FoS (Tresca): {factor_of_safety_tresca(tres, Sy):.3f}")

    # 3D example
    sx, sy, sz = 100.0, 40.0, -20.0
    txy, tyz, tzx = 30.0, 10.0, 20.0
    vm3 = von_mises_stress_3d(sx, sy, sz, txy, tyz, tzx)
    s1, s2, s3 = principal_stresses_3d(sx, sy, sz, txy, tyz, tzx)
    print(f"\n3D Stress state principal stresses: {s1:.2f}, {s2:.2f}, {s3:.2f} MPa")
    print(f"  Von Mises (3D): {vm3:.2f} MPa")
    print(f"  FoS (von Mises): {factor_of_safety_von_mises(vm3, Sy):.3f}")


def demo_vibrations() -> None:
    print("\n" + "=" * 70)
    print("MECHANICAL VIBRATIONS")
    print("=" * 70)

    # SDOF free response
    sdof = SDOFSystem(m=10.0, k=1e5, c=200.0)
    print(f"\n[SDOF] wn={sdof.omega_n:.2f} rad/s, zeta={sdof.zeta:.4f}, wd={sdof.omega_d:.2f} rad/s")
    t = np.linspace(0, 2.0, 500)
    x = sdof.free_response(t, x0=0.01, v0=0.0)
    print(f"  Initial x=0.01 m, v=0 -> x(2s)={x[-1]:.6e} m")

    # Frequency response sweep
    omega = np.linspace(0, 500, 400)
    X, phi = sdof.frequency_response(omega)
    res_idx = np.argmax(X)
    print(f"  Resonance approx at omega={omega[res_idx]:.2f} rad/s, Xmax={X[res_idx]:.6e} m/N")

    # MDOF modal analysis: 2-DOF shear building
    m1, m2 = 2.0, 2.0
    k1, k2 = 1e6, 8e5
    M = np.array([[m1, 0.0], [0.0, m2]])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    mdof = MDOFSystem(M, K)
    w, phi = mdof.modal_analysis()
    print("\n[MDOF] Natural frequencies:")
    for i in range(len(w)):
        print(f"  Mode {i+1}: w={w[i]:.2f} rad/s, f={w[i]/(2*math.pi):.2f} Hz")
        print(f"           phi={phi[:, i]}")


def demo_fatigue() -> None:
    print("\n" + "=" * 70)
    print("FATIGUE & FRACTURE MECHANICS")
    print("=" * 70)

    sn = SNCurve(sigma_f_prime=1000.0, b=-0.09)
    for N in [1e3, 1e4, 1e5, 1e6]:
        print(f"[S-N] N={N:.0f} cycles -> sigma_a={sn.stress_for_life(N):.1f} MPa")

    sigma_a, sigma_m = 150.0, 50.0
    Sut, Se = 600.0, 200.0
    print(f"\nGoodman FoS: {goodman_factor_of_safety(sigma_a, sigma_m, Sut, Se):.3f}")
    print(f"Gerber FoS:  {gerber_factor_of_safety(sigma_a, sigma_m, Sut, Se):.3f}")

    # Paris law
    C = 1e-12
    m = 3.0
    a0, ac = 0.002, 0.010  # m
    delta_sigma = 100.0  # MPa
    Y = 1.12
    N = cycles_to_crack_growth(a0, ac, delta_sigma, Y, C, m)
    print(f"\n[Paris] Crack growth {a0*1000:.2f} mm -> {ac*1000:.2f} mm: N={N:.1f} cycles")
    K = stress_intensity_factor(delta_sigma, ac, Y)
    print(f"  K at ac: {K:.2f} MPa√m")
    print(f"  Fracture check (K_Ic=60 MPa√m): {fracture_check(K, 60.0)}")


def demo_optimization() -> None:
    print("\n" + "=" * 70)
    print("STRUCTURAL OPTIMIZATION (RECTANGULAR BEAM)")
    print("=" * 70)

    design = RectangularBeamDesign(
        L=3.0, P=20000.0, E=210e9, rho=7850.0,
        sigma_yield=250e6, delta_allow=0.010, fs=1.5
    )
    x0 = np.array([0.05, 0.1])
    bounds = [(0.02, 0.2), (0.02, 0.4)]

    res_grad = optimize_beam_gradient(design, x0, bounds)
    print("\n[Gradient SLSQP]")
    print(f"  b={res_grad.x[0]:.4f} m, h={res_grad.x[1]:.4f} m")
    print(f"  Weight={res_grad.fun:.3f} N")
    print(f"  Stress={design.stress(res_grad.x)/1e6:.1f} MPa")
    print(f"  Deflection={design.deflection(res_grad.x)*1000:.2f} mm")

    best_ga, f_ga, hist = optimize_beam_ga(
        lambda x: penalty_objective(x, design), bounds, n_vars=2,
        pop_size=50, generations=100, mutation_rate=0.1, elitism=2, seed=7
    )
    print("\n[Genetic Algorithm]")
    print(f"  b={best_ga[0]:.4f} m, h={best_ga[1]:.4f} m")
    print(f"  Weight={design.weight(best_ga):.3f} N")
    print(f"  Stress={design.stress(best_ga)/1e6:.1f} MPa")
    print(f"  Deflection={design.deflection(best_ga)*1000:.2f} mm")


def demo_rotor() -> None:
    print("\n" + "=" * 70)
    print("ROTORDYNAMICS (JEFFCOTT ROTOR)")
    print("=" * 70)
    rotor = JeffcottRotor(m=50.0, k=2e6, c=1500.0, e=0.001)
    print(f"\nCritical speed (undamped): {rotor.omega_n:.1f} rad/s = {rotor.omega_n*60/(2*math.pi):.1f} rpm")
    print(f"Damping ratio: {rotor.zeta:.4f}")
    omega = np.linspace(0, 1.5 * rotor.omega_n, 200)
    X, phi = rotor.frequency_response(omega)
    idx_peak = np.argmax(X)
    print(f"Peak response: {X[idx_peak]:.6e} m at omega={omega[idx_peak]:.1f} rad/s")


def demo_contact() -> None:
    print("\n" + "=" * 70)
    print("CONTACT MECHANICS (HERTZIAN SPHERE ON FLAT)")
    print("=" * 70)
    res = HertzContact.sphere_on_flat(
        force=5000.0, radius=0.05, E=210e9, nu=0.3
    )
    print(f"\nForce=5000 N, R=50 mm, E=210 GPa, nu=0.3")
    for k, v in res.items():
        unit = "m" if k in ("contact_radius", "depth_max_shear") else ("Pa" if "pressure" in k or "stress" in k or "modulus" in k else "")
        print(f"  {k}: {v:.4e} {unit}")


def plot_all() -> None:
    plt.close("all")

    # SDOF response
    sdof = SDOFSystem(m=10.0, k=1e5, c=150.0)
    t = np.linspace(0, 3, 600)
    x = sdof.free_response(t, 0.02, 0.0)

    # Frequency response
    omega = np.linspace(0, 400, 500)
    X, phi = sdof.frequency_response(omega)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(t, x, "b-", lw=1.5)
    axs[0, 0].set_title("SDOF Free Response")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Displacement (m)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(omega, X, "r-", lw=1.5)
    axs[0, 1].set_title("SDOF Frequency Response")
    axs[0, 1].set_xlabel("Omega (rad/s)")
    axs[0, 1].set_ylabel("X (m/N)")
    axs[0, 1].grid(True, alpha=0.3)

    # Goodman diagram
    Sut, Se = 600.0, 200.0
    sigma_m = np.linspace(0, Sut, 200)
    sigma_a_goodman = Se * (1 - sigma_m / Sut)
    sigma_a_gerber = Se * (1 - (sigma_m / Sut) ** 2)
    axs[1, 0].plot(sigma_m, sigma_a_goodman, "k-", label="Goodman")
    axs[1, 0].plot(sigma_m, sigma_a_gerber, "k--", label="Gerber")
    axs[1, 0].fill_between(sigma_m, 0, sigma_a_goodman, alpha=0.2)
    axs[1, 0].plot([0, Sut], [Se, 0], "ro")
    axs[1, 0].set_title("Fatigue Failure Diagrams")
    axs[1, 0].set_xlabel("Mean stress (MPa)")
    axs[1, 0].set_ylabel("Alternating stress (MPa)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Rotor response
    rotor = JeffcottRotor(m=50.0, k=2e6, c=1500.0, e=0.001)
    w = np.linspace(0, 1.5 * rotor.omega_n, 400)
    Xr, _ = rotor.frequency_response(w)
    axs[1, 1].plot(w, Xr, "m-", lw=1.5)
    axs[1, 1].axvline(rotor.omega_n, color="k", linestyle="--", label="wn")
    axs[1, 1].set_title("Jeffcott Rotor Response")
    axs[1, 1].set_xlabel("Omega (rad/s)")
    axs[1, 1].set_ylabel("Amplitude (m)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_mechanical_toolkit() -> None:
    demo_fem()
    demo_stress()
    demo_vibrations()
    demo_fatigue()
    demo_optimization()
    demo_rotor()
    demo_contact()
    plot_all()


if __name__ == "__main__":
    demo_mechanical_toolkit()