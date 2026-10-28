from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict, Optional
from scipy import linalg, optimize, integrate
import matplotlib.pyplot as plt


# ============================================================================
# MATERIAL AND SECTION PROPERTIES
# ============================================================================
@dataclass
class Material:
    """Linear elastic isotropic material properties.

    Defaults correspond to structural steel.
    """
    name: str = "Structural Steel"
    E: float = 200e9          # Young's modulus [Pa]
    nu: float = 0.30          # Poisson's ratio [-]
    rho: float = 7850.0       # Density [kg/m^3]
    yield_strength: float = 250e6   # Yield strength [Pa]
    ultimate_strength: float = 400e6  # Ultimate tensile strength [Pa]
    endurance_limit: float = 200e6    # Fatigue endurance limit [Pa]


@dataclass
class Section:
    """Cross-section properties for 1D elements."""
    name: str = "Rectangular"
    A: float = 0.01           # Cross-sectional area [m^2]
    I: float = 8.333e-6       # Second moment of area about bending axis [m^4]
    J: float = 1.667e-5       # Polar/torsional constant [m^4]


# ============================================================================
# FINITE ELEMENT ANALYSIS
# ============================================================================
class Truss2D:
    """2D pin-jointed truss finite element model.

    Each node has two translational DOFs: [ux, uy].
    Elements are defined by (node_i, node_j, mat_idx, sec_idx).
    """

    def __init__(self,
                 nodes: List[Tuple[float, float]],
                 elements: List[Tuple[int, int, int, int]],
                 materials: List[Material],
                 sections: List[Section]) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = elements
        self.materials = materials
        self.sections = sections
        self.n_nodes = len(nodes)
        self.ndof = 2 * self.n_nodes
        self.K: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None

    def _element_stiffness(self, ei: int) -> Tuple[np.ndarray, List[int], float]:
        n_i, n_j, mi, si = self.elements[ei]
        E = self.materials[mi].E
        A = self.sections[si].A
        x_i, y_i = self.nodes[n_i]
        x_j, y_j = self.nodes[n_j]
        dx = x_j - x_i
        dy = y_j - y_i
        L = float(np.hypot(dx, dy))
        c = dx / L
        s = dy / L
        k = (E * A / L) * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ])
        dofs = [2 * n_i, 2 * n_i + 1, 2 * n_j, 2 * n_j + 1]
        return k, dofs, L

    def assemble(self) -> np.ndarray:
        self.K = np.zeros((self.ndof, self.ndof))
        for ei in range(len(self.elements)):
            k, dofs, _ = self._element_stiffness(ei)
            for a, da in enumerate(dofs):
                for b, db in enumerate(dofs):
                    self.K[da, db] += k[a, b]
        return self.K

    def solve(self,
              fixed_dofs: List[int],
              force_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve K u = F with prescribed fixed DOFs.

        Returns nodal displacements and reactions at fixed DOFs.
        """
        if self.K is None:
            self.assemble()
        F = np.asarray(force_vector, dtype=float).reshape(-1)
        fixed = np.asarray(fixed_dofs, dtype=int)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        Kff = self.K[np.ix_(free, free)]
        Ff = F[free]
        u = np.zeros(self.ndof)
        u[free] = linalg.solve(Kff, Ff, assume_a="sym")
        self.u = u
        self.F = F
        reactions = self.K @ u - F
        return u, reactions[fixed]

    def element_forces_stresses(self) -> List[Dict]:
        """Return axial force and stress for every element."""
        if self.u is None:
            raise RuntimeError("Model must be solved before recovering forces.")
        results = []
        for ei in range(len(self.elements)):
            n_i, n_j, mi, si = self.elements[ei]
            E = self.materials[mi].E
            A = self.sections[si].A
            _, dofs, L = self._element_stiffness(ei)
            x_i, y_i = self.nodes[n_i]
            x_j, y_j = self.nodes[n_j]
            dx = x_j - x_i
            dy = y_j - y_i
            c = dx / L
            s = dy / L
            u_e = self.u[list(dofs)]
            axial_disp = np.array([-c, -s, c, s]) @ u_e
            strain = axial_disp / L
            stress = E * strain
            force = stress * A
            results.append({
                "element": ei,
                "nodes": (n_i, n_j),
                "L": L,
                "strain": strain,
                "stress": stress,
                "axial_force": force,
            })
        return results


class Beam2D:
    """2D Euler-Bernoulli beam finite element model.

    Each node has three DOFs: [ux, uy, rot_z].
    Positive local x axis runs from node i to node j.
    """

    def __init__(self,
                 nodes: List[Tuple[float, float]],
                 elements: List[Tuple[int, int, int, int]],
                 materials: List[Material],
                 sections: List[Section]) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = elements
        self.materials = materials
        self.sections = sections
        self.n_nodes = len(nodes)
        self.ndof = 3 * self.n_nodes
        self.K: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None

    @staticmethod
    def _local_stiffness(E: float, A: float, I: float, L: float) -> np.ndarray:
        k = np.zeros((6, 6))
        k_ax = E * A / L
        k[0, 0] = k_ax
        k[0, 3] = -k_ax
        k[3, 0] = -k_ax
        k[3, 3] = k_ax
        kb = E * I / L ** 3
        bend = kb * np.array([
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
        ])
        bend_idx = [1, 2, 4, 5]
        for a in range(4):
            for b in range(4):
                k[bend_idx[a], bend_idx[b]] = bend[a, b]
        return k

    @staticmethod
    def _transformation(x_i: float, y_i: float,
                        x_j: float, y_j: float) -> Tuple[np.ndarray, float]:
        dx = x_j - x_i
        dy = y_j - y_i
        L = float(np.hypot(dx, dy))
        c = dx / L
        s = dy / L
        R = np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        Gamma = np.zeros((6, 6))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        return Gamma, L

    def assemble(self) -> np.ndarray:
        self.K = np.zeros((self.ndof, self.ndof))
        for (n_i, n_j, mi, si) in self.elements:
            E = self.materials[mi].E
            A = self.sections[si].A
            I = self.sections[si].I
            x_i, y_i = self.nodes[n_i]
            x_j, y_j = self.nodes[n_j]
            Gamma, L = self._transformation(x_i, y_i, x_j, y_j)
            k_loc = self._local_stiffness(E, A, I, L)
            k_glob = Gamma.T @ k_loc @ Gamma
            dofs = [3 * n_i, 3 * n_i + 1, 3 * n_i + 2,
                    3 * n_j, 3 * n_j + 1, 3 * n_j + 2]
            for a, da in enumerate(dofs):
                for b, db in enumerate(dofs):
                    self.K[da, db] += k_glob[a, b]
        return self.K

    def solve(self,
              fixed_dofs: List[int],
              force_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.K is None:
            self.assemble()
        F = np.asarray(force_vector, dtype=float).reshape(-1)
        fixed = np.asarray(fixed_dofs, dtype=int)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        u = np.zeros(self.ndof)
        u[free] = linalg.solve(self.K[np.ix_(free, free)], F[free], assume_a="sym")
        self.u = u
        self.F = F
        reactions = self.K @ u - F
        return u, reactions[fixed]

    def element_end_forces(self) -> List[Dict]:
        """Return local axial force, shear, and end moments for each element."""
        if self.u is None:
            raise RuntimeError("Model must be solved before recovering forces.")
        out = []
        for (n_i, n_j, mi, si) in self.elements:
            E = self.materials[mi].E
            A = self.sections[si].A
            I = self.sections[si].I
            x_i, y_i = self.nodes[n_i]
            x_j, y_j = self.nodes[n_j]
            Gamma, L = self._transformation(x_i, y_i, x_j, y_j)
            k_loc = self._local_stiffness(E, A, I, L)
            dofs = [3 * n_i, 3 * n_i + 1, 3 * n_i + 2,
                    3 * n_j, 3 * n_j + 1, 3 * n_j + 2]
            u_g = self.u[list(dofs)]
            u_loc = Gamma @ u_g
            f_loc = k_loc @ u_loc
            out.append({
                "nodes": (n_i, n_j),
                "L": L,
                "N": f_loc[0],
                "V_i": f_loc[1],
                "M_i": f_loc[2],
                "V_j": f_loc[4],
                "M_j": f_loc[5],
            })
        return out


class Quad4PlaneStress:
    """4-node bilinear quadrilateral plane-stress element (isoparametric).

    Uses a 2x2 Gauss integration rule. Each node has [ux, uy].
    """

    def __init__(self,
                 nodes: List[Tuple[float, float]],
                 elements: List[Tuple[int, int, int, int]],
                 material: Material,
                 thickness: float = 0.01) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = elements
        self.material = material
        self.thickness = thickness
        self.ndof = 2 * len(nodes)
        self.K: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.element_data: List[Dict] = []
        self.gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        self.gw = [1.0, 1.0]

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ])
        dN_dxi = np.array([
            -(1 - eta) / 4,
            (1 - eta) / 4,
            (1 + eta) / 4,
            -(1 + eta) / 4,
        ])
        dN_deta = np.array([
            -(1 - xi) / 4,
            -(1 + xi) / 4,
            (1 + xi) / 4,
            (1 - xi) / 4,
        ])
        return N, dN_dxi, dN_deta

    def _D_matrix(self) -> np.ndarray:
        E = self.material.E
        nu = self.material.nu
        return E / (1.0 - nu ** 2) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

    def assemble(self) -> np.ndarray:
        D = self._D_matrix()
        self.K = np.zeros((self.ndof, self.ndof))
        self.element_data = []
        for (n1, n2, n3, n4) in self.elements:
            coords = self.nodes[[n1, n2, n3, n4]]
            ke = np.zeros((8, 8))
            for xi in self.gp:
                for eta in self.gp:
                    _, dN_dxi, dN_deta = self._shape_functions(xi, eta)
                    J = np.array([
                        [dN_dxi @ coords[:, 0], dN_deta @ coords[:, 0]],
                        [dN_dxi @ coords[:, 1], dN_deta @ coords[:, 1]],
                    ])
                    detJ = float(np.linalg.det(J))
                    invJ = np.linalg.inv(J)
                    dN_dxy = invJ @ np.vstack([dN_dxi, dN_deta])
                    B = np.zeros((3, 8))
                    for i in range(4):
                        B[0, 2 * i] = dN_dxy[0, i]
                        B[1, 2 * i + 1] = dN_dxy[1, i]
                        B[2, 2 * i] = dN_dxy[1, i]
                        B[2, 2 * i + 1] = dN_dxy[0, i]
                    ke += self.thickness * (B.T @ D @ B) * detJ
            dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                    2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
            for a, da in enumerate(dofs):
                for b, db in enumerate(dofs):
                    self.K[da, db] += ke[a, b]
            self.element_data.append({"nodes": (n1, n2, n3, n4),
                                      "coords": coords,
                                      "dofs": dofs})
        return self.K

    def solve(self,
              fixed_dofs: List[int],
              force_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.K is None:
            self.assemble()
        F = np.asarray(force_vector, dtype=float).reshape(-1)
        fixed = np.asarray(fixed_dofs, dtype=int)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        u = np.zeros(self.ndof)
        u[free] = linalg.solve(self.K[np.ix_(free, free)], F[free], assume_a="sym")
        self.u = u
        self.F = F
        reactions = self.K @ u - F
        return u, reactions[fixed]

    def element_stresses(self) -> List[Dict]:
        """Recover plane-stress strains/stresses at element centers."""
        if self.u is None:
            raise RuntimeError("Model must be solved before recovering stresses.")
        D = self._D_matrix()
        out = []
        for ed in self.element_data:
            coords = ed["coords"]
            dofs = ed["dofs"]
            u_e = self.u[list(dofs)]
            _, dN_dxi, dN_deta = self._shape_functions(0.0, 0.0)
            J = np.array([
                [dN_dxi @ coords[:, 0], dN_deta @ coords[:, 0]],
                [dN_dxi @ coords[:, 1], dN_deta @ coords[:, 1]],
            ])
            invJ = np.linalg.inv(J)
            dN_dxy = invJ @ np.vstack([dN_dxi, dN_deta])
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i] = dN_dxy[0, i]
                B[1, 2 * i + 1] = dN_dxy[1, i]
                B[2, 2 * i] = dN_dxy[1, i]
                B[2, 2 * i + 1] = dN_dxy[0, i]
            strain = B @ u_e
            stress = D @ strain
            sx, sy, t = stress
            vm = float(np.sqrt(sx * sx - sx * sy + sy * sy + 3.0 * t * t))
            out.append({
                "nodes": ed["nodes"],
                "strain": strain,
                "stress": stress,
                "von_mises": vm,
            })
        return out


# ============================================================================
# STRESS AND STRAIN TRANSFORMATIONS
# ============================================================================
class StressState:
    """3D stress state with transformations, failure criteria, and Mohr's circle."""

    def __init__(self,
                 sx: float = 0.0,
                 sy: float = 0.0,
                 sz: float = 0.0,
                 tau_xy: float = 0.0,
                 tau_yz: float = 0.0,
                 tau_zx: float = 0.0) -> None:
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.tau_xy = tau_xy
        self.tau_yz = tau_yz
        self.tau_zx = tau_zx
        self.tensor = np.array([
            [sx, tau_xy, tau_zx],
            [tau_xy, sy, tau_yz],
            [tau_zx, tau_yz, sz],
        ])

    def principal_stresses(self) -> np.ndarray:
        """Return principal stresses sorted sigma1 >= sigma2 >= sigma3."""
        vals = np.linalg.eigvalsh(self.tensor)
        return vals[::-1]

    def von_mises(self) -> float:
        s1, s2, s3 = self.principal_stresses()
        return float(np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)))

    def von_mises_plane_stress(self) -> float:
        return float(np.sqrt(self.sx ** 2 - self.sx * self.sy +
                             self.sy ** 2 + 3.0 * self.tau_xy ** 2))

    def tresca(self) -> float:
        """Tresca equivalent stress = sigma1 - sigma3."""
        s1, _, s3 = self.principal_stresses()
        return float(s1 - s3)

    def max_shear(self) -> float:
        s1, _, s3 = self.principal_stresses()
        return float(0.5 * (s1 - s3))

    def mohr_circle_2d(self) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """2D Mohr's circle in the x-y plane.

        Returns center, radius, theta, sigma(theta), tau(theta).
        """
        center = 0.5 * (self.sx + self.sy)
        radius = np.sqrt(((self.sx - self.sy) / 2.0) ** 2 + self.tau_xy ** 2)
        theta = np.linspace(0.0, np.pi, 181)
        sigma = center + radius * np.cos(2.0 * theta)
        tau = radius * np.sin(2.0 * theta)
        return float(center), float(radius), theta, sigma, tau

    def transform(self, theta_deg: float) -> Tuple[float, float, float]:
        """Transform in-plane stresses by angle theta (degrees, CCW)."""
        th = np.radians(theta_deg)
        c = np.cos(th)
        s = np.sin(th)
        sxp = self.sx * c ** 2 + self.sy * s ** 2 + 2.0 * self.tau_xy * s * c
        syp = self.sx * s ** 2 + self.sy * c ** 2 - 2.0 * self.tau_xy * s * c
        txp = (self.sy - self.sx) * s * c + self.tau_xy * (c ** 2 - s ** 2)
        return float(sxp), float(syp), float(txp)

    def failure_ratios(self,
                       Syt: Optional[float] = None,
                       Sut: Optional[float] = None) -> Dict[str, float]:
        """Failure ratios relative to yield/ultimate strengths."""
        Syt = Syt if Syt is not None else 250e6
        vm = self.von_mises()
        tres = self.tresca()
        s1, _, s3 = self.principal_stresses()
        rankine = float(max(abs(s1), abs(s3)))
        return {
            "von_mises_stress": vm,
            "tresca_stress": tres,
            "rankine_stress": rankine,
            "vm_ratio": vm / Syt,
            "tresca_ratio": tres / Syt,
            "rankine_ratio": rankine / Syt,
        }


# ============================================================================
# MECHANICAL VIBRATIONS
# ============================================================================
class SDOFSystem:
    """Single-degree-of-freedom damped oscillator: m xdd + c xd + k x = F(t)."""

    def __init__(self, m: float = 10.0, k: float = 1e5, c: float = 100.0) -> None:
        self.m = m
        self.k = k
        self.c = c
        self.wn = float(np.sqrt(k / m))
        self.cc = 2.0 * np.sqrt(k * m)
        self.zeta = c / self.cc
        self.wd = self.wn * np.sqrt(max(0.0, 1.0 - self.zeta ** 2))

    def free_response(self, x0: float, v0: float, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if self.zeta < 1.0:
            x = (np.exp(-self.zeta * self.wn * t) *
                 (x0 * np.cos(self.wd * t) +
                  ((v0 + self.zeta * self.wn * x0) / self.wd) * np.sin(self.wd * t)))
        else:
            x = (x0 + (v0 + self.wn * x0) * t) * np.exp(-self.wn * t)
        return x

    def harmonic_amplitude(self, omega: float) -> Tuple[float, float]:
        """Steady-state amplitude per unit force and phase lag."""
        r = omega / self.wn
        X = (1.0 / self.k) / np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * self.zeta * r) ** 2)
        phase = float(np.arctan2(2.0 * self.zeta * r, 1.0 - r ** 2))
        return float(X), phase

    def frequency_response(self, omega: float) -> float:
        return self.harmonic_amplitude(omega)[0]


class MDOFSystem:
    """Multi-degree-of-freedom linear vibratory system with viscous damping."""

    def __init__(self,
                 M: np.ndarray,
                 K: np.ndarray,
                 C: Optional[np.ndarray] = None) -> None:
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.C = np.asarray(C, dtype=float) if C is not None else np.zeros_like(M)
        self.n = self.M.shape[0]

    def undamped_modal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve generalized eigenproblem K phi = omega^2 M phi (mass normalized)."""
        lam, phi = linalg.eigh(self.K, self.M)
        wn = np.sqrt(np.maximum(lam, 0.0))
        for i in range(self.n):
            phi[:, i] /= np.sqrt(phi[:, i] @ self.M @ phi[:, i])
        return wn, phi

    def state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (A, B, C, D) for xdot = A x + B u, y = C x + D u."""
        Minv = linalg.inv(self.M)
        Z = np.zeros((self.n, self.n))
        I = np.eye(self.n)
        A = np.block([[Z, I],
                      [-Minv @ self.K, -Minv @ self.C]])
        B = np.block([[Z],
                      [Minv]])
        Cmat = np.block([[I, Z],
                         [Z, I]])
        D = np.zeros((2 * self.n, self.n))
        return A, B, Cmat, D

    def newmark_beta(self,
                     F_func: Callable[[float], np.ndarray],
                     dt: float,
                     T: float,
                     gamma: float = 0.5,
                     beta: float = 0.25,
                     u0: Optional[np.ndarray] = None,
                     v0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Newmark-beta time integration for prescribed force vector F_func(t)."""
        if u0 is None:
            u0 = np.zeros(self.n)
        if v0 is None:
            v0 = np.zeros(self.n)
        u0 = np.asarray(u0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        N = int(np.round(T / dt)) + 1
        t = np.linspace(0.0, T, N)
        U = np.zeros((self.n, N))
        V = np.zeros((self.n, N))
        A = np.zeros((self.n, N))
        U[:, 0] = u0
        V[:, 0] = v0
        A[:, 0] = linalg.solve(self.M, F_func(0.0) - self.C @ v0 - self.K @ u0)

        a0 = 1.0 / (beta * dt ** 2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        a6 = dt * (1.0 - gamma)
        a7 = dt * gamma

        Keff = self.K + a1 * self.C + a0 * self.M
        Keff_inv = linalg.inv(Keff)

        for i in range(N - 1):
            t_next = t[i + 1]
            P_eff = (F_func(t_next) +
                     self.M @ (a0 * U[:, i] + a2 * V[:, i] + a3 * A[:, i]) +
                     self.C @ (a1 * U[:, i] + a4 * V[:, i] + a5 * A[:, i]))
            U[:, i + 1] = Keff_inv @ P_eff
            A[:, i + 1] = a0 * (U[:, i + 1] - U[:, i]) - a2 * V[:, i] - a3 * A[:, i]
            V[:, i + 1] = V[:, i] + a6 * A[:, i] + a7 * A[:, i + 1]
        return t, U, V, A


# ============================================================================
# FATIGUE AND FRACTURE MECHANICS
# ============================================================================
class SNCurve:
    """Basquin-type high-cycle fatigue S-N relationship.

    sigma_a = sigma_f_prime * (2 N)^b
    """

    def __init__(self,
                 sigma_f_prime: float = 900e6,
                 b: float = -0.085,
                 epsilon_f_prime: float = 0.5,
                 c: float = -0.6,
                 E: float = 200e9) -> None:
        self.sigma_f_prime = sigma_f_prime
        self.b = b
        self.epsilon_f_prime = epsilon_f_prime
        self.c = c
        self.E = E

    def stress_vs_cycles(self, N: np.ndarray) -> np.ndarray:
        N = np.asarray(N, dtype=float)
        return self.sigma_f_prime * (2.0 * N) ** self.b

    def cycles_for_stress(self, sigma: float) -> float:
        return float(0.5 * (sigma / self.sigma_f_prime) ** (1.0 / self.b))

    def strain_vs_cycles(self, N: np.ndarray) -> np.ndarray:
        N = np.asarray(N, dtype=float)
        return ((self.sigma_f_prime / self.E) * (2.0 * N) ** self.b +
                self.epsilon_f_prime * (2.0 * N) ** self.c)


class FatigueCriteria:
    """Mean stress correction and safety factor evaluation."""

    @staticmethod
    def goodman(sigma_a: float, sigma_m: float, Se: float, Sut: float) -> float:
        """Allowable alternating stress from Goodman line."""
        return float(Se * (1.0 - sigma_m / Sut))

    @staticmethod
    def gerber(sigma_a: float, sigma_m: float, Se: float, Sut: float) -> float:
        """Allowable alternating stress from Gerber parabola."""
        return float(Se * (1.0 - (sigma_m / Sut) ** 2))

    @staticmethod
    def safety_factor_goodman(sigma_a: float,
                              sigma_m: float,
                              Se: float,
                              Sut: float,
                              Sy: Optional[float] = None) -> float:
        n_fat = 1.0 / (sigma_a / Se + sigma_m / Sut)
        if Sy is None:
            return float(n_fat)
        n_yield = 1.0 / ((sigma_a + sigma_m) / Sy)
        return float(min(n_fat, n_yield))


class FractureMechanics:
    """Linear elastic fracture mechanics helpers."""

    @staticmethod
    def stress_intensity_factor(Y: float, sigma: float, a: float) -> float:
        """Mode I stress intensity factor K = Y sigma sqrt(pi a)."""
        return float(Y * sigma * np.sqrt(np.pi * a))

    @staticmethod
    def paris_law(C: float, m: float, delta_K: float) -> float:
        return float(C * delta_K ** m)

    @staticmethod
    def cycles_to_failure(C: float,
                          m: float,
                          Y: float,
                          sigma_max: float,
                          sigma_min: float,
                          a_i: float,
                          a_f: float,
                          Kc: Optional[float] = None) -> float:
        """Integrate Paris' law da/dN = C (Delta K)^m."""
        delta_sigma = sigma_max - sigma_min

        def integrand(a: float) -> float:
            dK = Y * delta_sigma * np.sqrt(np.pi * a)
            if Kc is not None and dK >= Kc:
                return np.inf
            return 1.0 / (C * dK ** m)

        N, _ = integrate.quad(integrand, a_i, a_f,
                              epsabs=1e-12, epsrel=1e-10, limit=200)
        return float(N)


# ============================================================================
# STRUCTURAL OPTIMIZATION
# ============================================================================
class BeamCrossSectionOptimizer:
    """Weight minimization of a rectangular beam cross section.

    Simply supported beam with a central point load. Constraints on bending
    stress and midspan deflection.
    """

    def __init__(self,
                 L: float = 3.0,
                 E: float = 200e9,
                 rho: float = 7850.0,
                 P: float = 20000.0,
                 allowable_stress: float = 250e6,
                 allowable_deflection: float = 0.006,
                 b_min: float = 0.02,
                 b_max: float = 0.5,
                 h_min: float = 0.02,
                 h_max: float = 0.5) -> None:
        self.L = L
        self.E = E
        self.rho = rho
        self.P = P
        self.allowable_stress = allowable_stress
        self.allowable_deflection = allowable_deflection
        self.b_min = b_min
        self.b_max = b_max
        self.h_min = h_min
        self.h_max = h_max

    def _weight(self, x: np.ndarray) -> float:
        b, h = x
        return float(self.rho * b * h * self.L)

    def _I(self, x: np.ndarray) -> float:
        b, h = x
        return b * h ** 3 / 12.0

    def _constraints(self, x: np.ndarray) -> np.ndarray:
        """Return constraint margins (positive = feasible)."""
        b, h = x
        I = self._I(x)
        M = self.P * self.L / 4.0
        sigma = M * (h / 2.0) / I
        delta = self.P * self.L ** 3 / (48.0 * self.E * I)
        return np.array([self.allowable_stress - sigma,
                         self.allowable_deflection - delta])

    def run_slsqp(self,
                  x0: Optional[np.ndarray] = None) -> optimize.OptimizeResult:
        if x0 is None:
            x0 = np.array([0.1, 0.2])
        cons = [
            {"type": "ineq", "fun": lambda x: self._constraints(x)[0]},
            {"type": "ineq", "fun": lambda x: self._constraints(x)[1]},
        ]
        bounds = [(self.b_min, self.b_max), (self.h_min, self.h_max)]
        return optimize.minimize(self._weight,
                                 x0,
                                 method="SLSQP",
                                 bounds=bounds,
                                 constraints=cons,
                                 options={"ftol": 1e-12, "maxiter": 500})

    def run_ga(self,
               popsize: int = 60,
               generations: int = 120,
               seed: int = 42) -> Tuple[np.ndarray, float, np.ndarray]:
        """Simple real-coded genetic algorithm with penalty constraints."""
        rng = np.random.default_rng(seed)
        lower = np.array([self.b_min, self.h_min])
        upper = np.array([self.b_max, self.h_max])
        pop = rng.uniform(lower, upper, size=(popsize, 2))

        def fitness(x: np.ndarray) -> float:
            pen = float(np.sum(np.maximum(0.0, -self._constraints(x)) ** 2)) * 1e9
            return self._weight(x) + pen

        best = None
        best_fit = np.inf
        history = []
        for _ in range(generations):
            fits = np.array([fitness(ind) for ind in pop])
            idx = np.argsort(fits)
            if fits[idx[0]] < best_fit:
                best_fit = fits[idx[0]]
                best = pop[idx[0]].copy()
            history.append(best_fit)
            new_pop = [pop[idx[0]], pop[idx[1]]]
            while len(new_pop) < popsize:
                a, b_ = rng.choice(popsize, size=2, replace=False)
                p1 = pop[a] if fits[a] < fits[b_] else pop[b_]
                a, b_ = rng.choice(popsize, size=2, replace=False)
                p2 = pop[a] if fits[a] < fits[b_] else pop[b_]
                alpha = rng.random()
                child = alpha * p1 + (1.0 - alpha) * p2
                if rng.random() < 0.2:
                    child += rng.normal(0.0, 0.02, size=2)
                child = np.clip(child, lower, upper)
                new_pop.append(child)
            pop = np.array(new_pop)
        return best, float(best_fit), np.array(history)


# ============================================================================
# ROTORDYNAMICS
# ============================================================================
class JeffcottRotor:
    """Simple Jeffcott rotor with unbalance excitation."""

    def __init__(self,
                 m: float = 50.0,
                 k: float = 2e7,
                 c: float = 2000.0,
                 eccentricity: float = 0.0002) -> None:
        self.m = m
        self.k = k
        self.c = c
        self.e = eccentricity
        self.wn = float(np.sqrt(k / m))

    def critical_speed(self) -> float:
        return self.wn

    def unbalanced_response(self, omega: float) -> Tuple[float, float]:
        """Steady-state whirl amplitude and phase for unbalance m e omega^2."""
        denom = np.sqrt((self.k - self.m * omega ** 2) ** 2 +
                        (self.c * omega) ** 2)
        X = self.m * self.e * omega ** 2 / denom
        phi = float(np.arctan2(self.c * omega, self.k - self.m * omega ** 2))
        return float(X), phi

    def frequency_sweep(self,
                        w_min: float,
                        w_max: float,
                        n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        w = np.linspace(w_min, w_max, n)
        X = np.array([self.unbalanced_response(wi)[0] for wi in w])
        return w, X


# ============================================================================
# CONTACT MECHANICS
# ============================================================================
class HertzContact:
    """Hertzian contact solutions for smooth elastic bodies."""

    @staticmethod
    def effective_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
        return float(1.0 / ((1.0 - nu1 ** 2) / E1 + (1.0 - nu2 ** 2) / E2))

    @staticmethod
    def sphere_sphere(R1: float,
                      R2: float,
                      F: float,
                      E1: float,
                      nu1: float,
                      E2: float,
                      nu2: float) -> Tuple[float, float, float]:
        R = R1 * R2 / (R1 + R2)
        E = HertzContact.effective_modulus(E1, nu1, E2, nu2)
        a = (3.0 * F * R / (4.0 * E)) ** (1.0 / 3.0)
        p_max = 3.0 * F / (2.0 * np.pi * a ** 2)
        p_mean = F / (np.pi * a ** 2)
        return float(a), float(p_max), float(p_mean)

    @staticmethod
    def cylinder_cylinder(R1: float,
                          R2: float,
                          F: float,
                          L: float,
                          E1: float,
                          nu1: float,
                          E2: float,
                          nu2: float) -> Tuple[float, float]:
        R = R1 * R2 / (R1 + R2)
        E = HertzContact.effective_modulus(E1, nu1, E2, nu2)
        b = np.sqrt(4.0 * F * R / (np.pi * L * E))
        p_max = 2.0 * F / (np.pi * L * b)
        return float(b), float(p_max)


# ============================================================================
# DEMONSTRATIONS
# ============================================================================
def _demo_truss() -> None:
    print("\n" + "=" * 70)
    print("FEA Demo: 2D Truss")
    print("=" * 70)
    mat = Material()
    sec = Section(A=2e-4, I=1e-8, J=2e-8)
    nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    elements = [(0, 1, 0, 0), (1, 2, 0, 0), (0, 2, 0, 0)]
    truss = Truss2D(nodes, elements, [mat], [sec])
    truss.assemble()
    F = np.zeros(6)
    F[3] = -10000.0  # downward load at node 1
    fixed = [0, 1, 4]  # node 0 fixed; node 2 x restrained
    u, R = truss.solve(fixed, F)
    print("Nodal displacements [ux, uy] per node:\n", u.reshape(-1, 2))
    print("Reactions at fixed DOFs:", R)
    for res in truss.element_forces_stresses():
        print(f"Element {res['element']} nodes {res['nodes']}: "
              f"N={res['axial_force']:.2f} N, sigma={res['stress']:.3e} Pa")


def _demo_beam() -> None:
    print("\n" + "=" * 70)
    print("FEA Demo: 2D Euler-Bernoulli Beam (Cantilever)")
    print("=" * 70)
    mat = Material()
    sec = Section(name="Rect", A=0.01, I=8.333e-6, J=1.667e-5)
    nodes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    elements = [(0, 1, 0, 0), (1, 2, 0, 0)]
    beam = Beam2D(nodes, elements, [mat], [sec])
    beam.assemble()
    F = np.zeros(9)
    F[7] = -5000.0  # vertical load at tip node 2
    fixed = [0, 1, 2]  # clamp node 0
    u, R = beam.solve(fixed, F)
    print(f"Tip vertical displacement: {u[7]:.6e} m")
    print("Reactions at fixed DOFs:", R)
    for ef in beam.element_end_forces():
        print(ef)


def _demo_plate() -> None:
    print("\n" + "=" * 70)
    print("FEA Demo: 4-Node Quad Plane Stress")
    print("=" * 70)
    mat = Material()
    nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    elements = [(0, 1, 2, 3)]
    plate = Quad4PlaneStress(nodes, elements, mat, thickness=0.01)
    plate.assemble()
    F = np.zeros(8)
    F[2] = 1000.0   # tensile traction on node 1
    F[4] = 1000.0   # tensile traction on node 2
    fixed = [0, 1, 6, 7]  # left edge fixed
    u, R = plate.solve(fixed, F)
    print("Nodal displacements [ux, uy] per node:\n", u.reshape(-1, 2))
    for s in plate.element_stresses():
        print(f"Element nodes {s['nodes']}: "
              f"stress=[{s['stress'][0]:.3e}, {s['stress'][1]:.3e}, {s['stress'][2]:.3e}] Pa, "
              f"VM={s['von_mises']:.3e} Pa")


def _demo_stress() -> None:
    print("\n" + "=" * 70)
    print("Stress Transformation Demo")
    print("=" * 70)
    state = StressState(sx=60e6, sy=20e6, tau_xy=30e6, sz=0.0)
    s1, s2, s3 = state.principal_stresses()
    print(f"Principal stresses: {s1:.3e}, {s2:.3e}, {s3:.3e} Pa")
    print(f"Von Mises stress:   {state.von_mises():.3e} Pa")
    print(f"Tresca stress:      {state.tresca():.3e} Pa")
    print(f"Max shear stress:   {state.max_shear():.3e} Pa")
    ratios = state.failure_ratios(Syt=250e6)
    print("Failure ratios:", ratios)

    center, radius, theta, sigma, tau = state.mohr_circle_2d()
    plt.figure(figsize=(6, 5))
    plt.plot(sigma / 1e6, tau / 1e6, label="Mohr circle")
    plt.plot([state.sx / 1e6, state.sy / 1e6],
             [state.tau_xy / 1e6, -state.tau_xy / 1e6], "o-", label="Stress points")
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(center / 1e6, color="k", ls="--", label="Center")
    plt.xlabel("Normal stress (MPa)")
    plt.ylabel("Shear stress (MPa)")
    plt.title("Mohr's Circle (2D)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    angles = np.linspace(0.0, 180.0, 181)
    sxp = np.array([state.transform(a)[0] for a in angles])
    plt.figure(figsize=(6, 4))
    plt.plot(angles, sxp / 1e6)
    plt.xlabel("Rotation angle theta (deg)")
    plt.ylabel("Transformed sigma_x' (MPa)")
    plt.title("Plane Stress Transformation")
    plt.grid(True)


def _demo_vibration() -> None:
    print("\n" + "=" * 70)
    print("Mechanical Vibrations Demo")
    print("=" * 70)
    sdof = SDOFSystem(m=10.0, k=1e5, c=50.0)
    print(f"SDOF: wn={sdof.wn:.2f} rad/s, zeta={sdof.zeta:.4f}, wd={sdof.wd:.2f} rad/s")
    t = np.linspace(0.0, 5.0, 1000)
    x = sdof.free_response(x0=0.01, v0=0.0, t=t)
    plt.figure(figsize=(6, 4))
    plt.plot(t, x)
    plt.title("SDOF Free Vibration Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)

    w = np.linspace(0.0, 200.0, 500)
    X = np.array([sdof.frequency_response(wi) for wi in w])
    plt.figure(figsize=(6, 4))
    plt.plot(w, X)
    plt.title("SDOF Frequency Response (displacement per unit force)")
    plt.xlabel("Excitation frequency (rad/s)")
    plt.ylabel("X/F0 (m/N)")
    plt.grid(True)

    # 2-DOF damped system
    m1 = m2 = 10.0
    k1 = k2 = 1e5
    M = np.diag([m1, m2])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    C = np.array([[100.0, -20.0], [-20.0, 40.0]])
    mdof = MDOFSystem(M, K, C)
    wn, phi = mdof.undamped_modal()
    print("Undamped natural frequencies (rad/s):", np.round(wn, 4))
    print("Mass-normalized mode shapes:\n", np.round(phi, 4))
    A, B, Cmat, D = mdof.state_space()
    print("State-space eigenvalues:\n", np.round(linalg.eigvals(A), 4))

    def F_step(tt: float) -> np.ndarray:
        return np.array([5000.0 if tt >= 0.0 else 0.0, 0.0])

    dt = 0.005
    T = 2.0
    tt, U, V, Acc = mdof.newmark_beta(F_step, dt, T)
    plt.figure(figsize=(6, 4))
    plt.plot(tt, U.T)
    plt.title("2-DOF Newmark-Beta Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.legend(["Mass 1", "Mass 2"])
    plt.grid(True)


def _demo_fatigue() -> None:
    print("\n" + "=" * 70)
    print("Fatigue and Fracture Mechanics Demo")
    print("=" * 70)
    sn = SNCurve(sigma_f_prime=900e6, b=-0.085, E=200e9)
    N = np.logspace(3, 7, 60)
    sigma = sn.stress_vs_cycles(N)
    plt.figure(figsize=(6, 4))
    plt.semilogx(N, sigma / 1e6)
    plt.title("Basquin S-N Curve")
    plt.xlabel("Cycles to failure")
    plt.ylabel("Stress amplitude (MPa)")
    plt.grid(True, which="both")

    Se = 200e6
    Sut = 600e6
    Sy = 400e6
    sigma_m = np.linspace(0.0, Sut, 200)
    sa_goodman = np.maximum(0.0,
                            np.array([FatigueCriteria.goodman(0, sm, Se, Sut)
                                      for sm in sigma_m]))
    sa_gerber = np.maximum(0.0,
                           np.array([FatigueCriteria.gerber(0, sm, Se, Sut)
                                     for sm in sigma_m]))
    plt.figure(figsize=(6, 4))
    plt.plot(sigma_m / 1e6, sa_goodman / 1e6, label="Goodman")
    plt.plot(sigma_m / 1e6, sa_gerber / 1e6, label="Gerber")
    plt.plot([0.0, Sy / 1e6], [Sy / 1e6, 0.0], "k--", label="Yield boundary")
    plt.xlabel("Mean stress (MPa)")
    plt.ylabel("Allowable alternating stress (MPa)")
    plt.title("Mean Stress Failure Diagrams")
    plt.legend()
    plt.grid(True)

    n = FatigueCriteria.safety_factor_goodman(
        sigma_a=80e6, sigma_m=100e6, Se=Se, Sut=Sut, Sy=Sy)
    print(f"Goodman safety factor (sa=80 MPa, sm=100 MPa): {n:.3f}")

    # Paris law crack growth
    C = 1e-11
    m = 3.0
    Y = 1.12
    sigma_max = 200e6
    sigma_min = 50e6
    a_i = 0.001
    a_f = 0.01
    N_paris = FractureMechanics.cycles_to_failure(
        C, m, Y, sigma_max, sigma_min, a_i, a_f)
    print(f"Paris law cycles for crack {a_i*1000:.1f} mm -> {a_f*1000:.1f} mm: "
          f"{N_paris:.2e}")


def _demo_optimization() -> None:
    print("\n" + "=" * 70)
    print("Structural Optimization Demo: Rectangular Beam Cross Section")
    print("=" * 70)
    opt = BeamCrossSectionOptimizer(L=3.0, P=20000.0,
                                    allowable_deflection=0.006)
    res_slsqp = opt.run_slsqp(x0=np.array([0.1, 0.2]))
    print("SLSQP result:")
    print("  b, h =", np.round(res_slsqp.x, 5), "m")
    print("  Weight =", float(res_slsqp.fun), "kg")
    print("  Success =", res_slsqp.success)
    b, h = res_slsqp.x
    print(f"  Section: A={b*h:.4e} m^2, I={opt._I(res_slsqp.x):.4e} m^4")

    best_ga, fit_ga, hist = opt.run_ga(popsize=40, generations=80)
    print("GA result:")
    print("  b, h =", np.round(best_ga, 5), "m")
    print("  Best fitness (weight + penalty) =", fit_ga)

    plt.figure(figsize=(6, 4))
    plt.semilogy(hist)
    plt.title("GA Convergence History")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.grid(True)


def _demo_rotor() -> None:
    print("\n" + "=" * 70)
    print("Rotordynamics Demo: Jeffcott Rotor")
    print("=" * 70)
    rotor = JeffcottRotor(m=50.0, k=2e7, c=2000.0, eccentricity=0.0002)
    wc = rotor.critical_speed()
    print(f"Critical speed: {wc:.1f} rad/s = {wc * 60.0 / (2.0 * np.pi):.0f} rpm")
    w = np.linspace(0.0, 3.0 * wc, 400)
    X = np.array([rotor.unbalanced_response(wi)[0] for wi in w])
    plt.figure(figsize=(6, 4))
    plt.plot(w / wc, X * 1e3)
    plt.title("Jeffcott Rotor Unbalance Response")
    plt.xlabel("Speed ratio Omega / Omega_cr")
    plt.ylabel("Whirl amplitude (mm)")
    plt.grid(True)


def _demo_contact() -> None:
    print("\n" + "=" * 70)
    print("Contact Mechanics Demo: Hertz Contact")
    print("=" * 70)
    E1 = E2 = 200e9
    nu1 = nu2 = 0.3
    a, p_max, p_mean = HertzContact.sphere_sphere(
        0.05, 0.05, 5000.0, E1, nu1, E2, nu2)
    print(f"Sphere-sphere contact: a={a*1000:.3f} mm, "
          f"p_max={p_max/1e6:.2f} MPa, p_mean={p_mean/1e6:.2f} MPa")
    b, p_max2 = HertzContact.cylinder_cylinder(
        0.05, 0.05, 5000.0, 0.1, E1, nu1, E2, nu2)
    print(f"Cylinder-cylinder contact: b={b*1000:.3f} mm, "
          f"p_max={p_max2/1e6:.2f} MPa")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    _demo_truss()
    _demo_beam()
    _demo_plate()
    _demo_stress()
    _demo_vibration()
    _demo_fatigue()
    _demo_optimization()
    _demo_rotor()
    _demo_contact()
    plt.show()