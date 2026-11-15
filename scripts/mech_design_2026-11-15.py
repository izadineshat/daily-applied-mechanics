#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mechanical_engineering_toolkit.py
=================================

A self-contained, production-oriented toolkit for common applied-mechanics
and machine-design calculations.  Every module below is written with plain
NumPy/SciPy data structures so that the code can be dropped into a larger
solver, a design script or an optimisation loop without modification.

Contents
--------
1.  Stress & strain transformations  (`StressState`, `mohr_circle_2d`)
2.  2-D truss finite elements       (`Truss2D`)
3.  Euler-Bernoulli beam elements   (`BeamFEM`, `beam_element_stiffness`, ...)
4.  Mechanical vibrations           (`SDOFSystem`, `MDOFSystem`)
5.  Fatigue & fracture              (`SNCurve`, `MeanStressCriteria`,
                                     `miner_damage`, `ParisLaw`)
6.  Structural optimisation         (`BeamSectionOptimizer`,
                                     `genetic_algorithm`)
7.  Rotordynamics                   (`JeffcottRotor`)
8.  Contact mechanics               (`HertzContact`)
9.  Failure theories                (`failure_theories`)

Run ``python mechanical_engineering_toolkit.py --help`` for the CLI, or simply
``python mechanical_engineering_toolkit.py`` to execute every demonstration.

Author: Particle-1 (Particle AI)
License: MIT
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, linalg, optimize, signal

import matplotlib

matplotlib.use("Agg")  # head-less safe; switch to an interactive backend if needed
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_1d(value, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or validate a 1-D sequence of length ``n``."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or length-{n}, got {arr.shape}")
    return arr.astype(float).copy()


def _print_header(title: str) -> None:
    """Pretty console section separator used by the demo block."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# 1.  STRESS / STRAIN TRANSFORMATIONS
# ===========================================================================
@dataclass
class StressState:
    """A full 3-D Cauchy stress state (Pa by convention).

    Parameters
    ----------
    sx, sy, sz : normal stresses on the x, y, z faces.
    txy, tyz, txz : engineering shear stresses.

    Notes
    -----
    The class deliberately keeps the six independent components; derived
    quantities (principal stresses, invariants, equivalent stresses) are
    computed on demand so a single instance can be re-used after rotation.
    """

    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0
    txy: float = 0.0
    tyz: float = 0.0
    txz: float = 0.0

    # -- basic tensors -----------------------------------------------------
    def tensor(self) -> np.ndarray:
        """Return the symmetric 3x3 stress tensor."""
        return np.array(
            [
                [self.sx, self.txy, self.txz],
                [self.txy, self.sy, self.tyz],
                [self.txz, self.tyz, self.sz],
            ],
            dtype=float,
        )

    @classmethod
    def from_tensor(cls, s: np.ndarray) -> "StressState":
        """Build a `StressState` from a (nearly) symmetric 3x3 array."""
        s = np.asarray(s, dtype=float)
        if s.shape != (3, 3):
            raise ValueError("stress tensor must be 3x3")
        s = 0.5 * (s + s.T)  # symmetrise
        return cls(s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2])

    # -- principal quantities ---------------------------------------------
    def principal_stresses(self) -> np.ndarray:
        """Principal stresses sorted in descending order (sigma_1 >= sigma_2 >= sigma_3)."""
        return np.sort(np.linalg.eigvalsh(self.tensor()))[::-1]

    def principal_directions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (principal stresses, principal directions) with descending order."""
        w, v = np.linalg.eigh(self.tensor())
        order = np.argsort(w)[::-1]
        return w[order], v[:, order]

    def max_shear(self) -> float:
        """Maximum shear stress tau_max = (sigma_1 - sigma_3) / 2."""
        p = self.principal_stresses()
        return 0.5 * (p[0] - p[2])

    def hydrostatic(self) -> float:
        """Mean (hydrostatic) stress."""
        return (self.sx + self.sy + self.sz) / 3.0

    def invariants(self) -> Tuple[float, float, float]:
        """Return (I1, I2, I3) stress invariants."""
        s = self.tensor()
        i1 = float(np.trace(s))
        i2 = float(
            s[0, 0] * s[1, 1]
            + s[1, 1] * s[2, 2]
            + s[2, 2] * s[0, 0]
            - s[0, 1] ** 2
            - s[1, 2] ** 2
            - s[0, 2] ** 2
        )
        i3 = float(np.linalg.det(s))
        return i1, i2, i3

    # -- equivalent stresses ----------------------------------------------
    def von_mises(self) -> float:
        """Von Mises equivalent stress (J2 theory)."""
        sx, sy, sz = self.sx, self.sy, self.sz
        txy, tyz, txz = self.txy, self.tyz, self.txz
        return math.sqrt(
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + txz**2)
        )

    def tresca(self) -> float:
        """Tresca (maximum-shear) equivalent stress = sigma_1 - sigma_3."""
        p = self.principal_stresses()
        return float(p[0] - p[2])

    # -- transformations ---------------------------------------------------
    def rotate_2d(self, theta_deg: float) -> "StressState":
        """Rotate the state about the z-axis by `theta_deg` (counter-clockwise)."""
        th = math.radians(theta_deg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotated = R @ self.tensor() @ R.T
        return StressState.from_tensor(rotated)

    def principal_angle_2d(self) -> float:
        """Principal-plane angle for plane stress, in degrees, in (-90, 90]."""
        theta = 0.5 * math.atan2(2.0 * self.txy, self.sx - self.sy)
        return math.degrees(theta)


def mohr_circle_2d(
    sigma_x: float, sigma_y: float, tau_xy: float
) -> Dict[str, float]:
    """Return the geometric description of the 2-D Mohr circle.

    Returns
    -------
    dict with keys
        ``center``  : (sigma_x + sigma_y)/2
        ``radius``  : sqrt(((sigma_x - sigma_y)/2)^2 + tau_xy^2)
        ``sigma_1`` : center + radius
        ``sigma_2`` : center - radius
        ``tau_max`` : radius
        ``theta_p`` : principal angle in degrees
    """
    center = 0.5 * (sigma_x + sigma_y)
    radius = math.hypot(0.5 * (sigma_x - sigma_y), tau_xy)
    return {
        "center": center,
        "radius": radius,
        "sigma_1": center + radius,
        "sigma_2": center - radius,
        "tau_max": radius,
        "theta_p": math.degrees(0.5 * math.atan2(2.0 * tau_xy, sigma_x - sigma_y)),
    }


# ===========================================================================
# 2.  2-D TRUSS FINITE ELEMENT MODEL
# ===========================================================================
class Truss2D:
    """Linear-elastic 2-D pin-jointed truss solver.

    Parameters
    ----------
    nodes : sequence of (x, y) coordinates [m].
    members : sequence of node-index pairs ``(i, j)``.
    E : Young's modulus (scalar or per-member) [Pa].
    A : cross-sectional area (scalar or per-member) [m^2].
    supports : ``{node: (fix_x, fix_y)}`` with booleans.
    loads : ``{node: (Fx, Fy)}`` nodal forces [N].

    Attributes after :meth:`solve`
    -------------------------------
    u : global displacement vector (2*N,)
    reactions : global reaction vector (2*N,)
    """

    def __init__(
        self,
        nodes: Sequence[Sequence[float]],
        members: Sequence[Sequence[int]],
        E: float,
        A: float,
        supports: Dict[int, Tuple[bool, bool]],
        loads: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be an (n, 2) array")
        self.members = [tuple(int(v) for v in m) for m in members]
        n = len(self.members)
        self.E = _as_1d(E, n, "E")
        self.A = _as_1d(A, n, "A")
        self.supports = {int(k): (bool(v[0]), bool(v[1])) for k, v in supports.items()}
        self.loads = {int(k): (float(v[0]), float(v[1])) for k, v in (loads or {}).items()}

        self.K: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.reactions: Optional[np.ndarray] = None
        self.lengths = np.zeros(n)
        self.cosines = np.zeros(n)
        self.sines = np.zeros(n)

    # ------------------------------------------------------------------
    def assemble(self) -> np.ndarray:
        """Assemble and return the global stiffness matrix."""
        ndof = 2 * self.nodes.shape[0]
        K = np.zeros((ndof, ndof))
        for e, (i, j) in enumerate(self.members):
            xi, yi = self.nodes[i]
            xj, yj = self.nodes[j]
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy)
            if L <= 0.0:
                raise ValueError(f"zero-length member {e}")
            c, s = dx / L, dy / L
            self.lengths[e] = L
            self.cosines[e] = c
            self.sines[e] = s
            ea = self.E[e] * self.A[e] / L
            k = ea * np.array(
                [
                    [c * c, c * s, -c * c, -c * s],
                    [c * s, s * s, -c * s, -s * s],
                    [-c * c, -c * s, c * c, c * s],
                    [-c * s, -s * s, c * s, s * s],
                ]
            )
            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K[np.ix_(dofs, dofs)] += k
        self.K = K
        return K

    # ------------------------------------------------------------------
    def solve(self) -> np.ndarray:
        """Apply boundary conditions, solve for displacements and reactions."""
        K = self.assemble()
        ndof = K.shape[0]
        F = np.zeros(ndof)
        for node, (fx, fy) in self.loads.items():
            F[2 * node] += fx
            F[2 * node + 1] += fy

        fixed = np.zeros(ndof, dtype=bool)
        for node, (fx_fixed, fy_fixed) in self.supports.items():
            if fx_fixed:
                fixed[2 * node] = True
            if fy_fixed:
                fixed[2 * node + 1] = True

        free = ~fixed
        u = np.zeros(ndof)
        if np.any(free):
            u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

        self.u = u
        self.reactions = K @ u - F
        self._fixed = fixed
        self._forces = F
        return u

    # ------------------------------------------------------------------
    def member_results(self) -> List[Dict[str, float]]:
        """Return a list of member dicts with length, stress, force, elongation."""
        if self.u is None:
            raise RuntimeError("call solve() first")
        out: List[Dict[str, float]] = []
        for e, (i, j) in enumerate(self.members):
            c, s, L = self.cosines[e], self.sines[e], self.lengths[e]
            ue = np.array(
                [self.u[2 * i], self.u[2 * i + 1], self.u[2 * j], self.u[2 * j + 1]]
            )
            strain = (-c * ue[0] - s * ue[1] + c * ue[2] + s * ue[3]) / L
            stress = self.E[e] * strain
            out.append(
                {
                    "length": float(L),
                    "strain": float(strain),
                    "stress": float(stress),
                    "force": float(stress * self.A[e]),
                    "elongation": float(strain * L),
                }
            )
        return out

    def max_abs_stress(self) -> float:
        """Largest absolute axial stress in the truss [Pa]."""
        return max(abs(r["stress"]) for r in self.member_results())


# ===========================================================================
# 3.  EULER-BERNOULLI BEAM FINITE ELEMENTS
# ===========================================================================
def beam_element_stiffness(E: float, I: float, L: float) -> np.ndarray:
    """4x4 Euler-Bernoulli beam element stiffness matrix.

    DOF order per element: ``[v_i, theta_i, v_j, theta_j]``.
    Sign convention: v positive in +y, theta positive counter-clockwise.
    """
    return (E * I / L**3) * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
        ]
    )


def beam_consistent_load(q: float, L: float) -> np.ndarray:
    """Consistent nodal load vector for a uniform distributed load ``q`` (+y)."""
    return np.array(
        [q * L / 2.0, q * L * L / 12.0, q * L / 2.0, -q * L * L / 12.0]
    )


class BeamFEM:
    """Plane-frame beam solver restricted to transverse (Euler-Bernoulli) behaviour.

    Parameters
    ----------
    nodes : x-coordinates of the beam nodes (only one spatial coordinate needed).
    elements : node-index pairs.
    E, I : Young's modulus and second moment of area (scalar or per element).
    supports : ``{node: (fix_v, fix_theta)}``.
    nodal_loads : ``{node: (Fy, M)}``.
    distributed_loads : ``{element_index: q}`` with q in +y direction.
    """

    def __init__(
        self,
        nodes: Sequence[float],
        elements: Sequence[Sequence[int]],
        E: float,
        I: float,
        supports: Dict[int, Tuple[bool, bool]],
        nodal_loads: Optional[Dict[int, Tuple[float, float]]] = None,
        distributed_loads: Optional[Dict[int, float]] = None,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float).ravel()
        self.elements = [tuple(int(v) for v in e) for e in elements]
        n = len(self.elements)
        self.E = _as_1d(E, n, "E")
        self.I = _as_1d(I, n, "I")
        self.supports = {int(k): (bool(v[0]), bool(v[1])) for k, v in supports.items()}
        self.nodal_loads = {
            int(k): (float(v[0]), float(v[1])) for k, v in (nodal_loads or {}).items()
        }
        self.distributed_loads = {
            int(k): float(v) for k, v in (distributed_loads or {}).items()
        }

        self.K: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.reactions: Optional[np.ndarray] = None
        self.ke: List[np.ndarray] = []
        self.Le: List[float] = []

    # ------------------------------------------------------------------
    def solve(self) -> np.ndarray:
        """Assemble, apply supports and return the nodal displacement vector."""
        n_nodes = self.nodes.size
        ndof = 2 * n_nodes
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)
        self.ke, self.Le = [], []

        for e, (i, j) in enumerate(self.elements):
            L = abs(self.nodes[j] - self.nodes[i])
            if L <= 0.0:
                raise ValueError(f"zero-length element {e}")
            ke = beam_element_stiffness(self.E[e], self.I[e], L)
            self.ke.append(ke)
            self.Le.append(L)
            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K[np.ix_(dofs, dofs)] += ke
            q = self.distributed_loads.get(e, 0.0)
            if q != 0.0:
                F[dofs] += beam_consistent_load(q, L)

        for node, (fy, m) in self.nodal_loads.items():
            F[2 * node] += fy
            F[2 * node + 1] += m

        fixed = np.zeros(ndof, dtype=bool)
        for node, (vfix, tfix) in self.supports.items():
            if vfix:
                fixed[2 * node] = True
            if tfix:
                fixed[2 * node + 1] = True

        u = np.zeros(ndof)
        free = ~fixed
        if np.any(free):
            u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

        self.K, self.u = K, u
        self.reactions = K @ u - F
        self._fixed = fixed
        return u

    # ------------------------------------------------------------------
    def element_end_forces(self, e: int) -> np.ndarray:
        """Element end forces ``[V_i, M_i, V_j, M_j]`` (element on nodes)."""
        if self.u is None:
            raise RuntimeError("call solve() first")
        i, j = self.elements[e]
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        q = self.distributed_loads.get(e, 0.0)
        fe = beam_consistent_load(q, self.Le[e])
        return self.ke[e] @ self.u[dofs] - fe

    def element_moment(self, e: int, x: float) -> float:
        """Internal bending moment at distance ``x`` from the element's first node.

        Sagging-positive convention.
        """
        f = self.element_end_forces(e)
        q = self.distributed_loads.get(e, 0.0)
        return -f[1] + f[0] * x + q * x * x / 2.0

    def element_shear(self, e: int, x: float) -> float:
        """Internal shear force at distance ``x`` from the element's first node."""
        f = self.element_end_forces(e)
        q = self.distributed_loads.get(e, 0.0)
        return f[0] + q * x

    def bending_moment_diagram(self, n_per_element: int = 21):
        """Return ``(x, M)`` arrays sampling the whole bending moment diagram."""
        if self.u is None:
            raise RuntimeError("call solve() first")
        xs, ms = [], []
        for e, (i, j) in enumerate(self.elements):
            x0 = self.nodes[i]
            L = self.Le[e]
            local = np.linspace(0.0, L, n_per_element)
            xs.append(x0 + local)
            ms.append(np.array([self.element_moment(e, xi) for xi in local]))
        return np.concatenate(xs), np.concatenate(ms)

    def max_bending_stress(self, c: Optional[float] = None, I: Optional[float] = None) -> float:
        """Largest bending stress, using per-element ``I`` and distance ``c`` to fibre."""
        if self.u is None:
            raise RuntimeError("call solve() first")
        x, m = self.bending_moment_diagram()
        if I is None:
            I = float(np.max(self.I))
        if c is None:
            c = 0.5 * 0.1  # placeholder; caller should pass the true distance
        return float(np.max(np.abs(m)) * c / I)


# ===========================================================================
# 4.  MECHANICAL VIBRATIONS
# ===========================================================================
class SDOFSystem:
    """Damped single-degree-of-freedom system ``m x'' + c x' + k x = F(t)``."""

    def __init__(self, m: float, k: float, zeta: float = 0.0, c: Optional[float] = None) -> None:
        if m <= 0.0 or k <= 0.0:
            raise ValueError("m and k must be positive")
        self.m = float(m)
        self.k = float(k)
        self.wn = math.sqrt(self.k / self.m)
        if c is None:
            self.zeta = float(zeta)
            self.c = 2.0 * self.zeta * math.sqrt(self.k * self.m)
        else:
            self.c = float(c)
            self.zeta = self.c / (2.0 * math.sqrt(self.k * self.m))
        self.wd = self.wn * math.sqrt(max(0.0, 1.0 - self.zeta**2))

    # ------------------------------------------------------------------
    def free_response(
        self, t: Sequence[float], x0: float = 0.0, v0: float = 0.0
    ) -> np.ndarray:
        """Closed-form free-vibration response for any damping ratio."""
        t = np.asarray(t, dtype=float)
        wn, z = self.wn, self.zeta
        if z < 1.0 - 1e-12:
            wd = wn * math.sqrt(1.0 - z * z)
            a = x0
            b = (v0 + z * wn * x0) / wd
            return np.exp(-z * wn * t) * (a * np.cos(wd * t) + b * np.sin(wd * t))
        if abs(z - 1.0) <= 1e-12:
            return (x0 + (v0 + wn * x0) * t) * np.exp(-wn * t)
        r1 = -z * wn + wn * math.sqrt(z * z - 1.0)
        r2 = -z * wn - wn * math.sqrt(z * z - 1.0)
        c1 = (v0 - r2 * x0) / (r1 - r2)
        c2 = (r1 * x0 - v0) / (r1 - r2)
        return c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)

    # ------------------------------------------------------------------
    def step_response(self, t: Sequence[float], f0: float = 1.0) -> np.ndarray:
        """Response to a step force of amplitude ``f0`` (zero initial conditions)."""
        t = np.asarray(t, dtype=float)
        if self.zeta >= 1.0:
            # fall back to numerical integration for overdamped / critical
            _, y = self.forced_response(t, np.full_like(t, f0))
            return y
        z = self.zeta
        wd = self.wd
        static = f0 / self.k
        return static * (
            1.0
            - np.exp(-z * self.wn * t)
            * (np.cos(wd * t) + (z / math.sqrt(1.0 - z * z)) * np.sin(wd * t))
        )

    # ------------------------------------------------------------------
    def forced_response(
        self,
        t: Sequence[float],
        force: Sequence[float],
        x0: float = 0.0,
        v0: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Time integration of an arbitrary force history via state space.

        Returns ``(t, x)``.
        """
        t = np.asarray(t, dtype=float)
        force = np.asarray(force, dtype=float)
        A = np.array([[0.0, 1.0], [-self.k / self.m, -self.c / self.m]])
        B = np.array([[0.0], [1.0 / self.m]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        sys = signal.StateSpace(A, B, C, D)
        x0_vec = np.array([x0, v0], dtype=float)
        _, y, _ = signal.lsim(sys, U=force, T=t, X0=x0_vec)
        return t, np.asarray(y, dtype=float)

    # ------------------------------------------------------------------
    def magnification(self, r: float) -> float:
        """Dynamic magnification factor for harmonic base/force excitation."""
        r = np.asarray(r, dtype=float)
        return 1.0 / np.sqrt((1.0 - r * r) ** 2 + (2.0 * self.zeta * r) ** 2)

    def phase_lag(self, r: float) -> float:
        """Phase lag (radians) between force and displacement."""
        r = np.asarray(r, dtype=float)
        return np.arctan2(2.0 * self.zeta * r, 1.0 - r * r)


class MDOFSystem:
    """Linear n-DOF system ``M x'' + C x' + K x = f(t)``."""

    def __init__(self, M: np.ndarray, K: np.ndarray, C: Optional[np.ndarray] = None) -> None:
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        if self.M.shape != self.K.shape or self.M.ndim != 2:
            raise ValueError("M and K must be square matrices of equal size")
        self.n = self.M.shape[0]
        self.C = None if C is None else np.asarray(C, dtype=float)
        self.wn: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the generalised eigenproblem, returning (wn [rad/s], mass-normalised modes)."""
        w2, phi = linalg.eigh(self.K, self.M)
        w2 = np.clip(w2, 0.0, None)
        wn = np.sqrt(w2)
        for i in range(self.n):
            v = phi[:, i]
            norm = math.sqrt(max(float(v @ self.M @ v), 1e-300))
            phi[:, i] = v / norm
        self.wn, self.phi = wn, phi
        return wn, phi

    # ------------------------------------------------------------------
    def rayleigh_damping(self, alpha: float, beta: float) -> np.ndarray:
        """Set a Rayleigh damping matrix ``C = alpha*M + beta*K``."""
        self.C = alpha * self.M + beta * self.K
        return self.C

    def modal_damping_ratios(self, alpha: float, beta: float) -> np.ndarray:
        """Damping ratios implied by Rayleigh coefficients on each mode."""
        if self.wn is None:
            self.modal_analysis()
        return 0.5 * (alpha / self.wn + beta * self.wn)

    # ------------------------------------------------------------------
    def state_space(self):
        """Return ``(A, B, C, D)`` in first-order (displacement output) form."""
        n = self.n
        C = self.C if self.C is not None else np.zeros_like(self.M)
        Minv = np.linalg.inv(self.M)
        A = np.block(
            [
                [np.zeros((n, n)), np.eye(n)],
                [-Minv @ self.K, -Minv @ C],
            ]
        )
        B = np.vstack([np.zeros((n, n)), Minv])
        Cout = np.hstack([np.eye(n), np.zeros((n, n))])
        D = np.zeros((n, n))
        return A, B, Cout, D

    # ------------------------------------------------------------------
    def frequency_response(self, omega: float, force: Optional[np.ndarray] = None) -> np.ndarray:
        """Complex displacement amplitudes for harmonic excitation at ``omega``."""
        C = self.C if self.C is not None else np.zeros_like(self.M)
        if force is None:
            force = np.ones(self.n)
        dyn = -omega**2 * self.M + 1j * omega * C + self.K
        return np.linalg.solve(dyn, np.asarray(force, dtype=complex))

    def forced_response(self, t: Sequence[float], force: np.ndarray) -> np.ndarray:
        """Time-domain response to a ``(n, len(t))`` or ``(len(t), n)`` force array."""
        t = np.asarray(t, dtype=float)
        force = np.asarray(force, dtype=float)
        if force.shape[0] == self.n and force.shape[1] == t.size:
            U = force.T
        elif force.shape[0] == t.size and force.shape[1] == self.n:
            U = force
        else:
            raise ValueError("force must be (n, len(t)) or (len(t), n)")
        A, B, C, D = self.state_space()
        sys = signal.StateSpace(A, B, C, D)
        _, y, _ = signal.lsim(sys, U=U, T=t)
        return np.asarray(y, dtype=float)


# ===========================================================================
# 5.  FATIGUE & FRACTURE MECHANICS
# ===========================================================================
class SNCurve:
    """Basquin-type stress-life curve ``sigma_a = sigma_f' (2N)^b``."""

    def __init__(
        self, sigma_f_prime: float = 900e6, b: float = -0.09, sigma_e: float = 250e6
    ) -> None:
        if b >= 0.0:
            raise ValueError("Basquin exponent b must be negative")
        self.sigma_f_prime = float(sigma_f_prime)
        self.b = float(b)
        self.sigma_e = float(sigma_e)  # endurance limit (fully reversed)

    def stress_amplitude(self, N: float) -> np.ndarray:
        """Stress amplitude corresponding to life ``N`` cycles."""
        N = np.asarray(N, dtype=float)
        return self.sigma_f_prime * (2.0 * N) ** self.b

    def life(self, sigma_a: float) -> np.ndarray:
        """Cycles to failure for a fully reversed stress amplitude."""
        sigma_a = np.asarray(sigma_a, dtype=float)
        return 0.5 * (sigma_a / self.sigma_f_prime) ** (1.0 / self.b)

    def is_infinite_life(self, sigma_a: float) -> bool:
        """True when the amplitude is below the endurance limit."""
        return bool(np.all(np.asarray(sigma_a) <= self.sigma_e))


class MeanStressCriteria:
    """Mean-stress correction models (Goodman, Gerber, Soderberg)."""

    def __init__(self, Se: float, Sut: float, Sy: Optional[float] = None) -> None:
        if Se <= 0.0 or Sut <= 0.0:
            raise ValueError("Se and Sut must be positive")
        self.Se = float(Se)
        self.Sut = float(Sut)
        self.Sy = None if Sy is None else float(Sy)

    # -- equivalent fully reversed amplitude ------------------------------
    def goodman_equivalent_amplitude(self, sa: float, sm: float) -> np.ndarray:
        sa = np.asarray(sa, dtype=float)
        sm = np.asarray(sm, dtype=float)
        return sa / (1.0 - sm / self.Sut)

    def gerber_equivalent_amplitude(self, sa: float, sm: float) -> np.ndarray:
        sa = np.asarray(sa, dtype=float)
        sm = np.asarray(sm, dtype=float)
        # solve  sa_eq/Se ... equivalent amplitude from Gerber parabola
        return sa / (1.0 - (sm / self.Sut) ** 2)

    # -- safety factors ----------------------------------------------------
    def goodman_safety_factor(self, sa: float, sm: float) -> np.ndarray:
        sa = np.asarray(sa, dtype=float)
        sm = np.asarray(sm, dtype=float)
        return 1.0 / (sa / self.Se + sm / self.Sut)

    def soderberg_safety_factor(self, sa: float, sm: float) -> np.ndarray:
        if self.Sy is None:
            raise ValueError("Sy required for the Soderberg criterion")
        sa = np.asarray(sa, dtype=float)
        sm = np.asarray(sm, dtype=float)
        return 1.0 / (sa / self.Se + sm / self.Sy)

    def gerber_safety_factor(self, sa: float, sm: float) -> np.ndarray:
        sa = np.asarray(sa, dtype=float)
        sm = np.asarray(sm, dtype=float)
        a = (sm / self.Sut) ** 2
        bq = sa / self.Se
        # avoid division by zero for pure alternating loading
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.where(a > 1e-14, (-bq + np.sqrt(bq**2 + 4.0 * a)) / (2.0 * a), self.Se / sa)
        return n


def miner_damage(cycle_counts: Sequence[float], lives: Sequence[float]) -> float:
    """Palmgren-Miner linear damage sum ``D = sum(n_i / N_i)``."""
    n = np.asarray(cycle_counts, dtype=float)
    N = np.asarray(lives, dtype=float)
    if n.shape != N.shape:
        raise ValueError("cycle_counts and lives must have the same shape")
    return float(np.sum(n / N))


class ParisLaw:
    """Paris-Erdogan fatigue crack growth: ``da/dN = C (dK)^m``."""

    def __init__(self, C: float = 1e-12, m: float = 3.0, Y: float = 1.0) -> None:
        """
        Parameters
        ----------
        C : Paris coefficient [(m/cycle) / (MPa*sqrt(m))^m].
        m : Paris exponent (dimensionless, typically 2 < m < 4 for metals).
        Y : geometry factor (1.0 for a centre crack in an infinite plate).
        """
        if C <= 0.0 or m <= 0.0:
            raise ValueError("C and m must be positive")
        self.C = float(C)
        self.m = float(m)
        self.Y = float(Y)

    def delta_k(self, sigma_range: float, a: float) -> np.ndarray:
        """Stress-intensity range [MPa*sqrt(m)] for crack length ``a`` [m]."""
        a = np.asarray(a, dtype=float)
        return self.Y * sigma_range * np.sqrt(np.pi * a)

    def growth_rate(self, sigma_range: float, a: float) -> np.ndarray:
        """Crack growth rate da/dN [m/cycle]."""
        return self.C * self.delta_k(sigma_range, a) ** self.m

    def cycles_to_grow(self, a0: float, af: float, sigma_range: float) -> float:
        """Number of cycles to grow a crack from ``a0`` to ``af`` (closed form)."""
        if a0 <= 0.0 or af <= a0:
            raise ValueError("require 0 < a0 < af")
        k = self.C * (self.Y * sigma_range * math.sqrt(math.pi)) ** self.m
        m = self.m
        if abs(m - 2.0) < 1e-12:
            return math.log(af / a0) / k
        return (af ** (1.0 - m / 2.0) - a0 ** (1.0 - m / 2.0)) / ((1.0 - m / 2.0) * k)

    def cycles_to_grow_numeric(self, a0: float, af: float, sigma_range: float) -> float:
        """Numerically integrated counterpart of :meth:`cycles_to_grow`."""
        integrand = lambda a: 1.0 / float(self.growth_rate(sigma_range, a))  # noqa: E731
        val, _ = integrate.quad(integrand, a0, af, limit=200)
        return float(val)

    def critical_crack_length(self, sigma_range: float, K_IC: float) -> float:
        """Crack length at which ``dK`` reaches the fracture toughness ``K_IC``."""
        return (K_IC / (self.Y * sigma_range)) ** 2 / math.pi


# ===========================================================================
# 6.  STRUCTURAL OPTIMISATION
# ===========================================================================
class BeamSectionOptimizer:
    """Minimum-weight rectangular cantilever cross-section.

    The design vector is ``x = [b, h]`` (width, height) and the objective is the
    cross-sectional area ``A = b*h`` (proportional to weight for a fixed length).

    Constraints
    -----------
    * bending stress      sigma = 6 P L / (b h^2) <= sigma_allow
    * tip deflection      delta = 4 P L^3 / (E b h^3) <= defl_allow
    * aspect ratio        h / b <= aspect_max
    """

    def __init__(
        self,
        length: float = 1.0,
        load: float = 5.0e3,
        E: float = 210e9,
        sigma_allow: float = 165e6,
        defl_allow: float = 1.0 / 300.0,
        aspect_max: float = 10.0,
        b_bounds: Tuple[float, float] = (0.01, 0.5),
        h_bounds: Tuple[float, float] = (0.01, 0.5),
        rho: float = 7850.0,
    ) -> None:
        self.L = float(length)
        self.P = float(load)
        self.E = float(E)
        self.sigma_allow = float(sigma_allow)
        self.defl_allow = float(defl_allow)
        self.aspect_max = float(aspect_max)
        self.bounds = np.array([b_bounds, h_bounds], dtype=float)
        self.rho = float(rho)

    # ------------------------------------------------------------------
    def responses(self, x: Sequence[float]) -> Tuple[float, float]:
        """Return ``(bending stress, tip deflection)`` for design ``x=[b, h]``."""
        b, h = float(x[0]), float(x[1])
        sigma = 6.0 * self.P * self.L / (b * h * h)
        delta = 4.0 * self.P * self.L**3 / (self.E * b * h**3)
        return sigma, delta

    def objective(self, x: Sequence[float]) -> float:
        """Cross-sectional area (m^2)."""
        return float(x[0] * x[1])

    def mass(self, x: Sequence[float]) -> float:
        """Beam mass in kg."""
        return self.objective(x) * self.L * self.rho

    def is_feasible(self, x: Sequence[float], tol: float = 1e-9) -> bool:
        b, h = float(x[0]), float(x[1])
        sigma, delta = self.responses(x)
        return (
            sigma <= self.sigma_allow * (1.0 + tol)
            and delta <= self.defl_allow * (1.0 + tol)
            and h / b <= self.aspect_max * (1.0 + tol)
        )

    # ------------------------------------------------------------------
    def solve_slsqp(self, x0: Optional[Sequence[float]] = None):
        """Gradient-based SLSQP solution returning the scipy ``OptimizeResult``."""
        if x0 is None:
            x0 = [0.05, 0.10]
        constraints = [
            {"type": "ineq", "fun": lambda x: 1.0 - self.responses(x)[0] / self.sigma_allow},
            {"type": "ineq", "fun": lambda x: 1.0 - self.responses(x)[1] / self.defl_allow},
            {"type": "ineq", "fun": lambda x: self.aspect_max - x[1] / x[0]},
        ]
        res = optimize.minimize(
            self.objective,
            np.asarray(x0, dtype=float),
            method="SLSQP",
            bounds=[tuple(self.bounds[0]), tuple(self.bounds[1])],
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 400},
        )
        return res

    def solve_ga(self, seed: int = 0):
        """Genetic-algorithm solution with a quadratic death penalty."""

        def penalised(x: np.ndarray) -> float:
            sigma, delta = self.responses(x)
            pen = 0.0
            pen += max(0.0, sigma / self.sigma_allow - 1.0) ** 2
            pen += max(0.0, delta / self.defl_allow - 1.0) ** 2
            pen += max(0.0, x[1] / (self.aspect_max * x[0]) - 1.0) ** 2
            return self.objective(x) + 1e3 * pen

        return genetic_algorithm(
            penalised, self.bounds, pop_size=80, generations=120, seed=seed
        )


def genetic_algorithm(
    objective: Callable[[np.ndarray], float],
    bounds: Iterable[Sequence[float]],
    pop_size: int = 60,
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.15,
    n_elite: int = 2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Real-coded genetic algorithm minimising ``objective`` over a box.

    Uses tournament selection, whole-arithmetic (blend) crossover, Gaussian
    mutation with 10 % of the span and elitist replacement.

    Returns
    -------
    (best_design_vector, best_objective_value)
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    n_var = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    pop = lo + rng.random((pop_size, n_var)) * span
    fitness = np.array([objective(ind) for ind in pop])
    best_idx = int(np.argmin(fitness))
    best_x, best_f = pop[best_idx].copy(), float(fitness[best_idx])

    for _ in range(generations):
        # tournament selection (size 2)
        idx1 = rng.integers(0, pop_size, pop_size)
        idx2 = rng.integers(0, pop_size, pop_size)
        parents = np.where(
            (fitness[idx1] < fitness[idx2])[:, None], pop[idx1], pop[idx2]
        )

        offspring = np.empty_like(pop)
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[min(i + 1, pop_size - 1)]
            if rng.random() < crossover_rate:
                alpha = rng.random(n_var)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = alpha * p2 + (1.0 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        mask = rng.random((pop_size, n_var)) < mutation_rate
        noise = rng.normal(0.0, 1.0, (pop_size, n_var)) * 0.10 * span
        offspring = np.where(mask, offspring + noise, offspring)
        offspring = np.clip(offspring, lo, hi)

        elite = np.argsort(fitness)[:n_elite]
        offspring[:n_elite] = pop[elite]
        pop = offspring
        fitness = np.array([objective(ind) for ind in pop])

        idx = int(np.argmin(fitness))
        if fitness[idx] < best_f:
            best_f, best_x = float(fitness[idx]), pop[idx].copy()

    return best_x, best_f


# ===========================================================================
# 7.  ROTORDYNAMICS  (JEFFCOTT ROTOR)
# ===========================================================================
class JeffcottRotor:
    """Single-disc symmetric rotor on a flexible shaft (Jeffcott/Laval rotor)."""

    def __init__(
        self,
        mass: float = 10.0,
        stiffness: float = 1.0e6,
        damping_ratio: float = 0.02,
        eccentricity: float = 1.0e-5,
    ) -> None:
        if mass <= 0.0 or stiffness <= 0.0:
            raise ValueError("mass and stiffness must be positive")
        self.m = float(mass)
        self.k = float(stiffness)
        self.zeta = float(damping_ratio)
        self.e = float(eccentricity)
        self.wn = math.sqrt(self.k / self.m)
        self.c = 2.0 * self.zeta * math.sqrt(self.k * self.m)

    # ------------------------------------------------------------------
    def critical_speed_rad_s(self) -> float:
        """Undamped critical (synchronous) speed [rad/s]."""
        return self.wn

    def critical_speed_rpm(self) -> float:
        """Undamped critical speed [rpm]."""
        return self.wn * 60.0 / (2.0 * math.pi)

    def unbalance_amplitude(self, omega: float) -> np.ndarray:
        """Steady-state whirl amplitude for a given spin speed ``omega`` [rad/s]."""
        omega = np.asarray(omega, dtype=float)
        num = self.m * self.e * omega**2
        den = np.sqrt((self.k - self.m * omega**2) ** 2 + (self.c * omega) ** 2)
        return num / den

    def amplification_factor(self, r: float) -> np.ndarray:
        """Dimensionless amplification ``X/(m e / ...)`` vs frequency ratio r."""
        r = np.asarray(r, dtype=float)
        return r**2 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * self.zeta * r) ** 2)

    def peak_amplitude(self) -> float:
        """Approximate maximum whirl amplitude (at resonance) [m]."""
        return self.e / (2.0 * self.zeta)


# ===========================================================================
# 8.  CONTACT MECHANICS  (HERTZ)
# ===========================================================================
class HertzContact:
    """Closed-form Hertzian elastic contact solutions."""

    @staticmethod
    def equivalent_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
        """Effective elastic modulus ``E*`` for two bodies."""
        return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)

    @staticmethod
    def sphere_on_flat(
        F: float, R: float, E1: float, nu1: float, E2: float, nu2: float
    ) -> Dict[str, float]:
        """Point contact between an elastic sphere and an elastic half-space.

        Returns
        -------
        dict with contact radius ``a`` [m], approach ``delta`` [m],
        peak pressure ``p0`` [Pa] and mean pressure ``p_mean`` [Pa].
        """
        if F <= 0.0 or R <= 0.0:
            raise ValueError("F and R must be positive")
        Estar = HertzContact.equivalent_modulus(E1, nu1, E2, nu2)
        a = (3.0 * F * R / (4.0 * Estar)) ** (1.0 / 3.0)
        p0 = 3.0 * F / (2.0 * math.pi * a * a)
        delta = a * a / R
        return {
            "a": float(a),
            "delta": float(delta),
            "p0": float(p0),
            "p_mean": float(F / (math.pi * a * a)),
            "E_star": float(Estar),
        }

    @staticmethod
    def cylinder_on_flat(
        F_per_length: float, R: float, E1: float, nu1: float, E2: float, nu2: float
    ) -> Dict[str, float]:
        """Line contact between an elastic cylinder and an elastic half-space."""
        if F_per_length <= 0.0 or R <= 0.0:
            raise ValueError("F_per_length and R must be positive")
        Estar = HertzContact.equivalent_modulus(E1, nu1, E2, nu2)
        b = math.sqrt(4.0 * F_per_length * R / (math.pi * Estar))
        p0 = 2.0 * F_per_length / (math.pi * b)
        return {
            "b": float(b),
            "p0": float(p0),
            "p_mean": float(F_per_length / (2.0 * b)),
            "E_star": float(Estar),
        }


# ===========================================================================
# 9.  FAILURE THEORIES
# ===========================================================================
def failure_theories(
    sigma_1: float, sigma_2: float, sigma_3: float, Sy: float, Sut: Optional[float] = None
) -> Dict[str, float]:
    """Compare classical ductile/brittle failure criteria.

    Parameters
    ----------
    sigma_1, sigma_2, sigma_3 : principal stresses [Pa].
    Sy : tensile yield strength [Pa].
    Sut : ultimate tensile strength [Pa] (used only for the Coulomb-Mohr entry).

    Returns
    -------
    dict of equivalent stresses and safety factors.
    """
    if Sy <= 0.0:
        raise ValueError("Sy must be positive")
    s = StressState()
    s.sx, s.sy, s.sz = sigma_1, sigma_2, sigma_3

    vm = s.von_mises()
    tresca = s.tresca()
    max_principal = max(abs(sigma_1), abs(sigma_2), abs(sigma_3))

    out: Dict[str, float] = {
        "von_mises_stress": vm,
        "von_mises_sf": Sy / vm if vm > 0.0 else float("inf"),
        "tresca_stress": tresca,
        "tresca_sf": Sy / tresca if tresca > 0.0 else float("inf"),
        "max_principal_stress": max_principal,
        "max_principal_sf": Sy / max_principal if max_principal > 0.0 else float("inf"),
    }
    if Sut is not None:
        out["coulomb_mohr_sf"] = (
            Sut / sigma_1 if sigma_1 > 0.0 else float("inf")
        )
    return out


# ===========================================================================
# DEMONSTRATIONS
# ===========================================================================
def demo_truss() -> None:
    _print_header("2-D TRUSS FINITE ELEMENT DEMONSTRATION")
    nodes = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.5)]
    members = [(0, 1), (0, 2), (1, 2)]
    E = 200e9
    A = 5.0e-4
    supports = {0: (True, True), 1: (False, True)}
    loads = {2: (0.0, -20_000.0)}

    truss = Truss2D(nodes, members, E, A, supports, loads)
    u = truss.solve()
    print("Node displacements [mm]:")
    for i in range(len(nodes)):
        print(f"  node {i}: ux = {u[2*i]*1e3:+.4f}, uy = {u[2*i+1]*1e3:+.4f}")

    print("\nMember results:")
    for e, r in enumerate(truss.member_results()):
        print(
            f"  member {members[e]}: L = {r['length']:.4f} m, "
            f"stress = {r['stress']/1e6:+9.3f} MPa, "
            f"force = {r['force']/1e3:+9.3f} kN"
        )

    total_reaction = truss.reactions[1] + truss.reactions[3] + truss.reactions[5]
    print(f"\nSum of vertical reactions = {total_reaction:+.1f} N "
          f"(applied = {loads[2][1]:+.1f} N)")


def demo_beam(plot: bool = False) -> None:
    _print_header("EULER-BERNOULLI BEAM FINITE ELEMENT DEMONSTRATION")
    L = 1.0
    E = 210e9
    b, h = 0.05, 0.10
    I = b * h**3 / 12.0
    q = -1000.0  # downward distributed load [N/m]

    n_el = 8
    nodes = np.linspace(0.0, L, n_el + 1)
    elements = [(i, i + 1) for i in range(n_el)]
    beam = BeamFEM(
        nodes,
        elements,
        E,
        I,
        supports={0: (True, True)},
        distributed_loads={i: q for i in range(n_el)},
    )
    u = beam.solve()

    tip = u[2 * n_el]
    analytic = q * L**4 / (8.0 * E * I)
    print(f"Cantilever tip deflection  FE = {tip*1e3:+.5f} mm, "
          f"analytic = {analytic*1e3:+.5f} mm")

    x, m = beam.bending_moment_diagram()
    print(f"Max |bending moment| = {np.max(np.abs(m)):.2f} N*m")
    print(f"Max bending stress    = {np.max(np.abs(m)) * (h/2) / I / 1e6:.3f} MPa")

    print("Reaction at the clamped end: "
          f"V = {beam.reactions[0]:+.2f} N, M = {beam.reactions[1]:+.2f} N*m")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        ax[0].plot(np.linspace(0.0, L, n_el + 1), u[0::2] * 1e3, "o-")
        ax[0].set_ylabel("deflection [mm]")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(x, m, "-")
        ax[1].set_xlabel("x [m]")
        ax[1].set_ylabel("bending moment [N m]")
        ax[1].grid(True, alpha=0.3)
        fig.suptitle("Cantilever under uniform distributed load")
        fig.tight_layout()
        fig.savefig("beam_demo.png", dpi=150)
        print("Figure written to beam_demo.png")


def demo_stress() -> None:
    _print_header("STRESS TRANSFORMATION / FAILURE CRITERIA DEMONSTRATION")
    s = StressState(sx=120e6, sy=-40e6, sz=0.0, txy=60e6)
    p = s.principal_stresses()
    print(f"Stress state: sx = {s.sx/1e6:.1f}, sy = {s.sy/1e6:.1f}, "
          f"txy = {s.txy/1e6:.1f} MPa")
    print(f"Principal stresses: s1 = {p[0]/1e6:.2f}, s2 = {p[1]/1e6:.2f}, "
          f"s3 = {p[2]/1e6:.2f} MPa")
    print(f"Principal angle     = {s.principal_angle_2d():+.2f} deg")
    print(f"Max shear stress    = {s.max_shear()/1e6:.2f} MPa")
    print(f"Von Mises stress    = {s.von_mises()/1e6:.2f} MPa")
    print(f"Tresca stress       = {s.tresca()/1e6:.2f} MPa")

    mc = mohr_circle_2d(s.sx, s.sy, s.txy)
    print(f"Mohr circle: center = {mc['center']/1e6:.2f} MPa, "
          f"radius = {mc['radius']/1e6:.2f} MPa")

    rotated = s.rotate_2d(30.0)
    print(f"After 30 deg rotation: sx' = {rotated.sx/1e6:.2f}, "
          f"sy' = {rotated.sy/1e6:.2f}, txy' = {rotated.txy/1e6:.2f} MPa")

    ft = failure_theories(p[0], p[1], p[2], Sy=350e6, Sut=600e6)
    print("\nFailure-theory comparison (Sy = 350 MPa):")
    for key, val in ft.items():
        print(f"  {key:24s} = {val:.4f}")


def demo_vibration(plot: bool = False) -> None:
    _print_header("MECHANICAL VIBRATION DEMONSTRATION")
    sdof = SDOFSystem(m=10.0, k=1.0e6, zeta=0.05)
    print(f"SDOF: fn = {sdof.wn/(2*math.pi):.3f} Hz, "
          f"zeta = {sdof.zeta:.3f}, wd = {sdof.wd:.3f} rad/s")

    t = np.linspace(0.0, 0.15, 2000)
    x_free = sdof.free_response(t, x0=1e-3)
    _, x_step = sdof.forced_response(t, np.full_like(t, 200.0))
    print(f"Free response peak amplitude    = {np.max(np.abs(x_free))*1e3:.4f} mm")
    print(f"Forced step steady-state value  = {x_step[-1]*1e3:.4f} mm "
          f"(F/k = {200.0/sdof.k*1e3:.4f} mm)")

    r = np.linspace(0.0, 3.0, 200)
    mag = sdof.magnification(r)
    print(f"Peak magnification (r = {r[np.argmax(mag)]:.3f}) = {np.max(mag):.3f}")

    # 3-DOF chain system
    m = 2.0
    k = 1.0e5
    M = np.eye(3) * m
    K = np.array(
        [
            [2 * k, -k, 0.0],
            [-k, 2 * k, -k],
            [0.0, -k, k],
        ]
    )
    mdof = MDOFSystem(M, K)
    wn, phi = mdof.modal_analysis()
    print("\n3-DOF chain modal analysis:")
    for i, w in enumerate(wn):
        print(f"  mode {i+1}: {w/(2*math.pi):8.3f} Hz   shape = "
              f"{np.round(phi[:, i], 4)}")

    mdof.rayleigh_damping(alpha=0.5, beta=1e-4)
    zetas = mdof.modal_damping_ratios(0.5, 1e-4)
    print(f"Rayleigh modal damping ratios = {np.round(zetas, 4)}")

    H = mdof.frequency_response(omega=wn[0] * 1.1, force=np.array([1.0, 0.0, 0.0]))
    print(f"FRF magnitude at 1.1*wn1 = {np.abs(H)}")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(t * 1e3, x_step * 1e3)
        ax[0].set_xlabel("time [ms]")
        ax[0].set_ylabel("displacement [mm]")
        ax[0].set_title("SDOF step response")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(r, mag)
        ax[1].set_xlabel("frequency ratio r")
        ax[1].set_ylabel("magnification")
        ax[1].set_title("SDOF magnification curve")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("vibration_demo.png", dpi=150)
        print("Figure written to vibration_demo.png")


def demo_fatigue() -> None:
    _print_header("FATIGUE & FRACTURE MECHANICS DEMONSTRATION")
    sn = SNCurve(sigma_f_prime=900e6, b=-0.09, sigma_e=250e6)
    sa = 200e6
    print(f"Stress amplitude {sa/1e6:.1f} MPa -> life = {sn.life(sa):,.0f} cycles")
    print(f"Endurance limit   = {sn.sigma_e/1e6:.1f} MPa, "
          f"infinite life: {sn.is_infinite_life(sa)}")

    crit = MeanStressCriteria(Se=250e6, Sut=600e6, Sy=350e6)
    sm = 150e6
    print(f"\nMean stress {sm/1e6:.1f} MPa with alternating {sa/1e6:.1f} MPa:")
    print(f"  Goodman safety factor  = {float(crit.goodman_safety_factor(sa, sm)):.3f}")
    print(f"  Gerber safety factor   = {float(crit.gerber_safety_factor(sa, sm)):.3f}")
    print(f"  Soderberg safety factor= {float(crit.soderberg_safety_factor(sa, sm)):.3f}")

    damage = miner_damage([1e4, 5e4, 2e5], [1e5, 4e5, 1e7])
    print(f"\nMiner damage for a 3-block spectrum = {damage:.5f}")

    paris = ParisLaw(C=1e-12, m=3.0, Y=1.0)
    a0, af = 1.0e-3, 20.0e-3
    dsigma = 100e6
    n_closed = paris.cycles_to_grow(a0, af, dsigma / 1e6)
    n_num = paris.cycles_to_grow_numeric(a0, af, dsigma / 1e6)
    print(f"\nParis law: C = {paris.C:g}, m = {paris.m:g}, Y = {paris.Y:g}")
    print(f"  crack growth {a0*1e3:.1f} -> {af*1e3:.1f} mm at dsigma = "
          f"{dsigma/1e6:.0f} MPa")
    print(f"  closed-form cycles = {n_closed:,.0f}")
    print(f"  numeric integration = {n_num:,.0f}")
    print(f"  da/dN at a0         = {float(paris.growth_rate(100.0, a0))*1e9:.4f} nm/cycle")


def demo_optimization() -> None:
    _print_header("STRUCTURAL OPTIMISATION DEMONSTRATION")
    opt = BeamSectionOptimizer(
        length=1.0,
        load=5.0e3,
        E=210e9,
        sigma_allow=165e6,
        defl_allow=1.0 / 300.0,
        aspect_max=10.0,
    )
    res = opt.solve_slsqp()
    b, h = res.x
    sigma, delta = opt.responses(res.x)
    print("SLSQP result:")
    print(f"  b = {b*1e3:.3f} mm, h = {h*1e3:.3f} mm, A = {res.fun*1e6:.2f} mm^2")
    print(f"  mass        = {opt.mass(res.x):.4f} kg")
    print(f"  stress      = {sigma/1e6:.3f} MPa  (allow {opt.sigma_allow/1e6:.1f})")
    print(f"  deflection  = {delta*1e3:.3f} mm   (allow {opt.defl_allow*1e3:.3f})")
    print(f"  feasible    = {opt.is_feasible(res.x)}")

    x_ga, f_ga = opt.solve_ga(seed=42)
    print("\nGenetic-algorithm result:")
    print(f"  b = {x_ga[0]*1e3:.3f} mm, h = {x_ga[1]*1e3:.3f} mm, "
          f"A = {f_ga*1e6:.2f} mm^2")
    print(f"  feasible = {opt.is_feasible(x_ga)}")


def demo_rotor(plot: bool = False) -> None:
    _print_header("ROTORDYNAMICS DEMONSTRATION")
    rotor = JeffcottRotor(mass=10.0, stiffness=1.0e6, damping_ratio=0.02,
                          eccentricity=1.0e-5)
    print(f"Jeffcott rotor: wn = {rotor.wn:.2f} rad/s "
          f"({rotor.critical_speed_rpm():.1f} rpm)")
    print(f"Peak whirl amplitude at resonance ~ {rotor.peak_amplitude()*1e3:.3f} mm")

    speeds_rpm = np.array([600.0, 1500.0, 2400.0, 3016.0, 3600.0, 4800.0])
    speeds = speeds_rpm * 2.0 * math.pi / 60.0
    amps = rotor.unbalance_amplitude(speeds)
    for rpm, a in zip(speeds_rpm, amps):
        print(f"  {rpm:7.1f} rpm -> whirl amplitude {a*1e3:.4f} mm")

    if plot:
        w = np.linspace(1.0, 3.0 * rotor.wn, 400)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(w * 60.0 / (2 * math.pi), rotor.unbalance_amplitude(w) * 1e3)
        ax.axvline(rotor.critical_speed_rpm(), ls="--", c="k", lw=1)
        ax.set_xlabel("spin speed [rpm]")
        ax.set_ylabel("whirl amplitude [mm]")
        ax.set_title("Jeffcott rotor unbalance response")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("rotor_demo.png", dpi=150)
        print("Figure written to rotor_demo.png")


def demo_contact() -> None:
    _print_header("HERTZIAN CONTACT MECHANICS DEMONSTRATION")
    E_steel, nu_steel = 210e9, 0.30
    res = HertzContact.sphere_on_flat(
        F=1000.0, R=0.010, E1=E_steel, nu1=nu_steel, E2=E_steel, nu2=nu_steel
    )
    print("Steel sphere (R = 10 mm) on a steel flat, F = 1 kN:")
    print(f"  contact radius a   = {res['a']*1e3:.4f} mm")
    print(f"  approach delta     = {res['delta']*1e6:.4f} um")
    print(f"  peak pressure p0   = {res['p0']/1e6:.3f} MPa")
    print(f"  mean pressure      = {res['p_mean']/1e6:.3f} MPa")

    line = HertzContact.cylinder_on_flat(
        F_per_length=5.0e4, R=0.020, E1=E_steel, nu1=nu_steel,
        E2=E_steel, nu2=nu_steel
    )
    print("\nSteel cylinder (R = 20 mm) on a steel flat, F' = 50 kN/m:")
    print(f"  contact half-width b = {line['b']*1e3:.4f} mm")
    print(f"  peak pressure p0     = {line['p0']/1e6:.3f} MPa")


def demo_failure() -> None:
    _print_header("FAILURE THEORY DEMONSTRATION")
    res = failure_theories(sigma_1=180e6, sigma_2=60e6, sigma_3=-20e6,
                           Sy=350e6, Sut=600e6)
    for key, val in res.items():
        print(f"  {key:24s} = {val:.4f}")


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mechanical engineering analysis toolkit demonstrations."
    )
    parser.add_argument(
        "--demo",
        default="all",
        choices=[
            "all", "truss", "beam", "stress", "vibration",
            "fatigue", "optimization", "rotor", "contact", "failure",
        ],
        help="which demonstration to run (default: all)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="write summary figures to the working directory",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    selection = args.demo

    demos = {
        "truss": demo_truss,
        "beam": lambda: demo_beam(plot=args.plot),
        "stress": demo_stress,
        "vibration": lambda: demo_vibration(plot=args.plot),
        "fatigue": demo_fatigue,
        "optimization": demo_optimization,
        "rotor": lambda: demo_rotor(plot=args.plot),
        "contact": demo_contact,
        "failure": demo_failure,
    }

    if selection == "all":
        for name, fn in demos.items():
            fn()
    else:
        demos[selection]()

    if args.plot:
        plt.close("all")
    print("\nAll requested analyses completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())