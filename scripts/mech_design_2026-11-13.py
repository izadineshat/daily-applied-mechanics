#!/usr/bin/env python3
"""
mechanical_engineering_toolkit.py

A self-contained Python toolkit for common mechanical engineering calculations:
- Finite element analysis: 1D bar, 2D truss, Euler-Bernoulli beam, rectangular plate bending element
- Stress and strain transformations, Mohr's circle, principal stresses, von Mises, Tresca
- Mechanical vibrations: SDOF and MDOF damped response, modal analysis, state-space formulation
- Fatigue and fracture mechanics: S-N/Basquin, Goodman/Gerber/Soderberg, Paris law
- Structural optimization: beam cross-section weight minimization using SLSQP and differential evolution
- Rotordynamics: Jeffcott rotor response
- Contact mechanics: Hertz sphere-plane and cylinder-plane contact
- Failure theories: maximum normal stress, Tresca, von Mises, Mohr-Coulomb

Dependencies: numpy, scipy, matplotlib.
Run this file directly to execute demonstration routines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import differential_evolution, minimize

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# General material and section definitions
# ---------------------------------------------------------------------------
@dataclass
class Material:
    """Engineering material properties."""
    name: str
    E: float = 210.0e9            # Young's modulus [Pa]
    nu: float = 0.30              # Poisson's ratio [-]
    rho: float = 7850.0           # Density [kg/m^3]
    yield_strength: float = 355.0e6   # Yield strength [Pa]
    ultimate_strength: float = 500.0e6  # Ultimate tensile strength [Pa]
    fatigue_strength_coeff: float = 900.0e6  # Basquin sigma_f' [Pa]
    fatigue_strength_exponent: float = -0.10  # Basquin exponent b [-]
    fracture_toughness: float = 50.0e6  # Plane-strain fracture toughness [Pa sqrt(m)]

    @classmethod
    def steel(cls) -> "Material":
        return cls(
            name="Structural steel",
            E=210.0e9,
            nu=0.30,
            rho=7850.0,
            yield_strength=355.0e6,
            ultimate_strength=500.0e6,
            fatigue_strength_coeff=900.0e6,
            fatigue_strength_exponent=-0.10,
            fracture_toughness=50.0e6,
        )

    @classmethod
    def aluminum(cls) -> "Material":
        return cls(
            name="Aluminum alloy 6061-T6",
            E=69.0e9,
            nu=0.33,
            rho=2700.0,
            yield_strength=276.0e6,
            ultimate_strength=310.0e6,
            fatigue_strength_coeff=450.0e6,
            fatigue_strength_exponent=-0.10,
            fracture_toughness=29.0e6,
        )


@dataclass
class ISection:
    """Rectangular cross-section properties."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def inertia_strong_axis(self) -> float:
        return self.width * self.height ** 3 / 12.0

    @property
    def inertia_weak_axis(self) -> float:
        return self.height * self.width ** 3 / 12.0

    @property
    def section_modulus_strong_axis(self) -> float:
        return self.inertia_strong_axis / (self.height / 2.0)


# ---------------------------------------------------------------------------
# Finite element analysis
# ---------------------------------------------------------------------------
def bar_element_stiffness(E: float, A: float, L: float) -> np.ndarray:
    """Return the 2x2 local axial stiffness matrix of a 1D bar element."""
    if L <= 0.0:
        raise ValueError("Element length L must be positive.")
    k = E * A / L
    return k * np.array([[1.0, -1.0],
                         [-1.0, 1.0]], dtype=float)


def truss_element_stiffness(E: float, A: float, L: float, angle: float) -> np.ndarray:
    """
    Return the 4x4 global stiffness matrix of a 2D truss element.

    Local DOF ordering is [u_i, v_i, u_j, v_j].
    """
    if L <= 0.0:
        raise ValueError("Element length L must be positive.")
    c = math.cos(angle)
    s = math.sin(angle)
    k = E * A / L
    return k * np.array([
        [ c * c,  c * s, -c * c, -c * s],
        [ c * s,  s * s, -c * s, -s * s],
        [-c * c, -c * s,  c * c,  c * s],
        [-c * s, -s * s,  c * s,  s * s],
    ], dtype=float)


def assemble_truss_stiffness(
    coords: Sequence[Sequence[float]],
    elements: Sequence[Tuple[int, int, float, float]],
) -> np.ndarray:
    """
    Assemble a 2D truss global stiffness matrix.

    Parameters
    ----------
    coords : sequence of (x, y) node coordinates.
    elements : sequence of (node_i, node_j, E, A).

    Returns
    -------
    K : ndarray of shape (2*n_nodes, 2*n_nodes)
    """
    n = len(coords)
    K = np.zeros((2 * n, 2 * n), dtype=float)
    for i, j, E, A in elements:
        dx = float(coords[j][0] - coords[i][0])
        dy = float(coords[j][1] - coords[i][1])
        L = math.hypot(dx, dy)
        if L <= 0.0:
            raise ValueError(f"Zero-length truss element between nodes {i} and {j}.")
        angle = math.atan2(dy, dx)
        ke = truss_element_stiffness(E, A, L, angle)
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(dofs, dofs)] += ke
    return K


@dataclass
class TrussSolution:
    """Results from a 2D truss analysis."""
    displacements: np.ndarray
    reactions: np.ndarray
    axial_stresses: np.ndarray
    axial_forces: np.ndarray
    stiffness_matrix: np.ndarray


def solve_truss(
    coords: Sequence[Sequence[float]],
    elements: Sequence[Tuple[int, int, float, float]],
    fixed_dofs: Iterable[int],
    loads: Dict[int, float],
) -> TrussSolution:
    """
    Solve a linear-elastic 2D truss problem.

    Parameters
    ----------
    coords : node coordinates [(x, y), ...].
    elements : list of (node_i, node_j, E, A).
    fixed_dofs : global DOF indices constrained to zero.
    loads : mapping from global DOF index to nodal force.

    Global DOF ordering is [u0, v0, u1, v1, ...].
    """
    n = len(coords)
    ndof = 2 * n
    K = assemble_truss_stiffness(coords, elements)

    F = np.zeros(ndof, dtype=float)
    for dof, value in loads.items():
        F[int(dof)] += float(value)

    fixed = sorted({int(d) for d in fixed_dofs})
    fixed_set = set(fixed)
    free = np.array([d for d in range(ndof) if d not in fixed_set], dtype=int)

    u = np.zeros(ndof, dtype=float)
    if free.size > 0:
        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        u[free] = np.linalg.solve(Kff, Ff)

    R = K @ u - F

    axial_stresses = np.zeros(len(elements), dtype=float)
    axial_forces = np.zeros(len(elements), dtype=float)

    for idx, (i, j, E, A) in enumerate(elements):
        dx = float(coords[j][0] - coords[i][0])
        dy = float(coords[j][1] - coords[i][1])
        L = math.hypot(dx, dy)
        c = dx / L
        s = dy / L
        local_elongation = (
            -c * u[2 * i] - s * u[2 * i + 1]
            + c * u[2 * j] + s * u[2 * j + 1]
        )
        stress = E * local_elongation / L
        axial_stresses[idx] = stress
        axial_forces[idx] = A * stress

    return TrussSolution(u, R, axial_stresses, axial_forces, K)


def beam_element_stiffness(E: float, I: float, L: float) -> np.ndarray:
    """
    Return the 4x4 Euler-Bernoulli beam element stiffness matrix.

    DOF ordering: [v_i, theta_i, v_j, theta_j].
    """
    if L <= 0.0:
        raise ValueError("Element length L must be positive.")
    EI = E * I
    return (EI / L ** 3) * np.array([
        [ 12.0,   6.0 * L, -12.0,   6.0 * L],
        [  6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
        [-12.0,  -6.0 * L,  12.0,  -6.0 * L],
        [  6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
    ], dtype=float)


def _plate_basis(x: float, y: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the 12-term rectangular plate polynomial basis and its derivatives.

    The basis is:
    1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3, x^3 y, x y^3

    Returns
    -------
    w, w_x, w_y, w_xx, w_yy, w_xy : each length-12 arrays
    """
    w = np.array([
        1.0, x, y,
        x * x, x * y, y * y,
        x ** 3, x * x * y, x * y * y, y ** 3,
        x ** 3 * y, x * y ** 3,
    ], dtype=float)

    wx = np.array([
        0.0, 1.0, 0.0,
        2.0 * x, y, 0.0,
        3.0 * x * x, 2.0 * x * y, y * y, 0.0,
        3.0 * x * x * y, y ** 3,
    ], dtype=float)

    wy = np.array([
        0.0, 0.0, 1.0,
        0.0, x, 2.0 * y,
        0.0, x * x, 2.0 * x * y, 3.0 * y * y,
        x ** 3, 3.0 * x * y * y,
    ], dtype=float)

    wxx = np.array([
        0.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        6.0 * x, 2.0 * y, 0.0, 0.0,
        6.0 * x * y, 0.0,
    ], dtype=float)

    wyy = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 2.0,
        0.0, 0.0, 2.0 * x, 6.0 * y,
        0.0, 6.0 * x * y,
    ], dtype=float)

    wxy = np.array([
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 2.0 * x, 2.0 * y, 0.0,
        3.0 * x * x, 3.0 * y * y,
    ], dtype=float)

    return w, wx, wy, wxx, wyy, wxy


def rectangular_plate_element_stiffness(
    a: float,
    b: float,
    thickness: float,
    E: float,
    nu: float,
    n_gauss: int = 2,
) -> np.ndarray:
    """
    Return the 12x12 Kirchhoff rectangular plate bending element stiffness matrix.

    DOF ordering per node: [w, theta_x, theta_y], where
    theta_x = dw/dy and theta_y = -dw/dx for the usual plate convention.

    Local coordinates: x in [0, a], y in [0, b].
    """
    if a <= 0.0 or b <= 0.0 or thickness <= 0.0:
        raise ValueError("Plate dimensions and thickness must be positive.")
    if nu >= 0.5:
        raise ValueError("Poisson's ratio must be less than 0.5 for plane stress.")

    nodes = [(0.0, 0.0), (a, 0.0), (a, b), (0.0, b)]

    # Build the nodal condition matrix M such that M @ coefficients = nodal DOFs.
    M = np.zeros((12, 12), dtype=float)
    for inode, (x, y) in enumerate(nodes):
        w, wx, wy, _, _, _ = _plate_basis(x, y)
        M[3 * inode + 0, :] = w
        M[3 * inode + 1, :] = wx
        M[3 * inode + 2, :] = wy

    try:
        P = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(M)

    # Bending constitutive matrix D for isotropic Kirchhoff plate.
    D = (E * thickness ** 3 / (12.0 * (1.0 - nu * nu))) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ], dtype=float)

    xi, wi = leggauss(int(n_gauss))
    K = np.zeros((12, 12), dtype=float)

    for ix in range(n_gauss):
        for iy in range(n_gauss):
            x = 0.5 * (xi[ix] + 1.0) * a
            y = 0.5 * (xi[iy] + 1.0) * b
            w, wx, wy, wxx, wyy, wxy = _plate_basis(x, y)

            N = P.T @ w
            Nxx = P.T @ wxx
            Nyy = P.T @ wyy
            Nxy = P.T @ wxy

            # Curvature vector [w_xx, w_yy, 2 w_xy]
            B = np.vstack((Nxx, Nyy, 2.0 * Nxy))
            detJ_weight = 0.25 * a * b
            K += detJ_weight * wi[ix] * wi[iy] * (B.T @ D @ B)

    return K


# ---------------------------------------------------------------------------
# Stress and strain transformations
# ---------------------------------------------------------------------------
def stress_transformation(
    sigma_x: float,
    sigma_y: float,
    tau_xy: float,
    theta_rad: float,
) -> Tuple[float, float, float]:
    """
    Transform plane stress components to an inclined plane.

    Returns (sigma_x', sigma_y', tau_x'y').
    """
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    sx_p = sigma_x * c * c + sigma_y * s * s + 2.0 * tau_xy * c * s
    sy_p = sigma_x * s * s + sigma_y * c * c - 2.0 * tau_xy * c * s
    txy_p = (sigma_y - sigma_x) * c * s + tau_xy * (c * c - s * s)
    return sx_p, sy_p, txy_p


def principal_stresses_2d(
    sigma_x: float,
    sigma_y: float,
    tau_xy: float,
) -> Tuple[float, float, float]:
    """
    Return 2D principal stresses and the principal plane angle.

    Returns (sigma_1, sigma_2, theta_p), with theta_p in radians.
    """
    avg = 0.5 * (sigma_x + sigma_y)
    radius = math.hypot(0.5 * (sigma_x - sigma_y), tau_xy)
    s1 = avg + radius
    s2 = avg - radius
    theta_p = 0.5 * math.atan2(2.0 * tau_xy, sigma_x - sigma_y)
    return s1, s2, theta_p


def mohr_circle(
    sigma_x: float,
    sigma_y: float,
    tau_xy: float,
) -> Tuple[float, float, float, float]:
    """
    Return Mohr's circle data.

    Returns (center, radius, sigma_1, sigma_2).
    """
    center = 0.5 * (sigma_x + sigma_y)
    radius = math.hypot(0.5 * (sigma_x - sigma_y), tau_xy)
    s1 = center + radius
    s2 = center - radius
    return center, radius, s1, s2


def von_mises_2d(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Return the 2D plane-stress von Mises equivalent stress."""
    return math.sqrt(
        sigma_x ** 2
        - sigma_x * sigma_y
        + sigma_y ** 2
        + 3.0 * tau_xy ** 2
    )


def tresca_2d(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Return the 2D plane-stress Tresca equivalent stress."""
    s1, s2, _ = principal_stresses_2d(sigma_x, sigma_y, tau_xy)
    return max(s1, s2, 0.0) - min(s1, s2, 0.0)


@dataclass
class StressState:
    """Three-dimensional stress state."""
    sigma: np.ndarray

    def __post_init__(self) -> None:
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(3, 3)
        # Enforce symmetry to remove small numerical asymmetry.
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    @classmethod
    def from_components(
        cls,
        sigma_x: float = 0.0,
        sigma_y: float = 0.0,
        sigma_z: float = 0.0,
        tau_xy: float = 0.0,
        tau_yz: float = 0.0,
        tau_xz: float = 0.0,
    ) -> "StressState":
        """Create a StressState from engineering stress components."""
        return cls(np.array([
            [sigma_x, tau_xy, tau_xz],
            [tau_xy, sigma_y, tau_yz],
            [tau_xz, tau_yz, sigma_z],
        ], dtype=float))

    def principal_stresses(self) -> np.ndarray:
        """Return principal stresses sorted in ascending order."""
        return np.sort(np.linalg.eigvalsh(self.sigma))

    def von_mises(self) -> float:
        """Return the von Mises equivalent stress."""
        p = self.principal_stresses()
        return math.sqrt(
            0.5 * (
                (p[0] - p[1]) ** 2
                + (p[1] - p[2]) ** 2
                + (p[2] - p[0]) ** 2
            )
        )

    def tresca(self) -> float:
        """Return the Tresca maximum-shear equivalent stress."""
        p = self.principal_stresses()
        return float(p[-1] - p[0])

    def max_normal_stress(self) -> float:
        """Return the maximum absolute principal stress."""
        p = self.principal_stresses()
        return float(np.max(np.abs(p)))

    def mohr_coulomb_index(self, Sut: float, Suc: float) -> float:
        """
        Return a simple Mohr-Coulomb failure index.

        For tensile/compressive material strengths Sut and Suc (both positive),
        the index is approximately:
            sigma_1 / Sut - sigma_3 / Suc
        with special handling for uniaxial tension or compression.
        """
        p = self.principal_stresses()
        s1 = float(p[-1])
        s3 = float(p[0])
        if s1 > 0.0 and s3 < 0.0:
            return s1 / Sut - s3 / Suc
        if s1 > 0.0:
            return s1 / Sut
        return -s3 / Suc


# ---------------------------------------------------------------------------
# Mechanical vibrations
# ---------------------------------------------------------------------------
@dataclass
class ModalResult:
    """Modal analysis results."""
    omega_n: np.ndarray
    natural_frequencies_hz: np.ndarray
    mode_shapes: np.ndarray
    damping_ratios: Optional[np.ndarray]
    modal_masses: np.ndarray
    modal_stiffnesses: np.ndarray


def modal_analysis(
    K: np.ndarray,
    M: np.ndarray,
    C: Optional[np.ndarray] = None,
    neigs: Optional[int] = None,
) -> ModalResult:
    """
    Perform undamped or proportionally damped modal analysis.

    Solves the generalized eigenvalue problem:
        K phi = omega^2 M phi

    If a damping matrix C is supplied, modal damping ratios are computed as:
        zeta_i = (phi_i^T C phi_i) / (2 sqrt((phi_i^T M phi_i)(phi_i^T K phi_i)))
    """
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    if K.shape != M.shape:
        raise ValueError("K and M must have the same shape.")

    if neigs is None:
        eigvals, eigvecs = eigh(K, M)
    else:
        eigvals, eigvecs = eigh(K, M, subset_by_index=(0, int(neigs) - 1))

    eigvals = np.maximum(eigvals, 0.0)
    omega = np.sqrt(eigvals)

    order = np.argsort(omega)
    omega = omega[order]
    eigvecs = eigvecs[:, order]

    # Normalize each mode so that its largest absolute component is unity.
    for j in range(eigvecs.shape[1]):
        max_abs = float(np.max(np.abs(eigvecs[:, j])))
        if max_abs > 1e-15:
            eigvecs[:, j] /= max_abs

    modal_masses = np.diag(eigvecs.T @ M @ eigvecs)
    modal_stiffnesses = np.diag(eigvecs.T @ K @ eigvecs)

    damping_ratios = None
    if C is not None:
        C = np.asarray(C, dtype=float)
        modal_damping = np.diag(eigvecs.T @ C @ eigvecs)
        damping_ratios = np.zeros_like(modal_damping)
        for j in range(len(modal_damping)):
            denom = 2.0 * math.sqrt(modal_masses[j] * modal_stiffnesses[j])
            if denom > 0.0:
                damping_ratios[j] = modal_damping[j] / denom

    return ModalResult(
        omega_n=omega,
        natural_frequencies_hz=omega / (2.0 * math.pi),
        mode_shapes=eigvecs,
        damping_ratios=damping_ratios,
        modal_masses=modal_masses,
        modal_stiffnesses=modal_stiffnesses,
    )


def mdof_state_space(
    K: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the first-order state-space form of a linear MDOF system.

    State vector: x = [q; qdot]
    dx/dt = A x + B F(t)
    """
    n = M.shape[0]
    MinvK = np.linalg.solve(M, K)
    MinvC = np.linalg.solve(M, C)
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-MinvK, -MinvC],
    ])
    B = np.vstack([
        np.zeros((n, n)),
        np.linalg.inv(M),
    ])
    return A, B


def mdof_response(
    K: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    force_func: Optional[Callable[[float], np.ndarray]],
    t: Sequence[float],
    q0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a damped MDOF system in state-space form using scipy.integrate.solve_ivp.

    Parameters
    ----------
    force_func : callable t -> force vector of length n, or None for free response.
    t : time vector.
    q0, v0 : initial displacements and velocities.

    Returns
    -------
    q, qdot : arrays of shape (n, len(t))
    """
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    n = M.shape[0]
    A, B = mdof_state_space(K, M, C)

    t = np.asarray(t, dtype=float)
    q0 = np.zeros(n, dtype=float) if q0 is None else np.asarray(q0, dtype=float)
    v0 = np.zeros(n, dtype=float) if v0 is None else np.asarray(v0, dtype=float)
    y0 = np.concatenate((q0, v0))

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        F = np.zeros(n, dtype=float) if force_func is None else np.asarray(
            force_func(time), dtype=float
        ).reshape(-1)
        return A @ state + B @ F

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        y0,
        t_eval=t,
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(f"MDOF ODE integration failed: {sol.message}")

    return sol.y[:n, :], sol.y[n:, :]


@dataclass
class SDofProperties:
    """Natural frequency and damping properties of a SDOF system."""
    m: float
    k: float
    c: float

    @property
    def wn(self) -> float:
        return math.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * math.sqrt(self.k * self.m))

    @property
    def wd(self) -> float:
        return self.wn * math.sqrt(max(0.0, 1.0 - self.zeta ** 2))


def sdof_response(
    m: float,
    k: float,
    c: float,
    x0: float = 0.0,
    v0: float = 0.0,
    t: Optional[Sequence[float]] = None,
    force_func: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """
    Compute SDOF displacement response.

    If force_func is None, the response is computed analytically for free vibration.
    Otherwise, a damped forced response is integrated numerically.
    """
    if m <= 0.0 or k <= 0.0:
        raise ValueError("Mass and stiffness must be positive.")

    if t is None:
        wn = math.sqrt(k / m)
        zeta = c / (2.0 * math.sqrt(k * m))
        if zeta > 0.0:
            decay_time = 5.0 / (zeta * wn)
        else:
            decay_time = 5.0 * (2.0 * math.pi / wn)
        t = np.linspace(0.0, max(10.0, decay_time), 1200)

    t = np.asarray(t, dtype=float)
    wn = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(k * m))

    if force_func is None:
        if zeta < 1.0 - 1e-12:
            wd = wn * math.sqrt(1.0 - zeta * zeta)
            env = np.exp(-zeta * wn * t)
            x = env * (
                x0 * np.cos(wd * t)
                + (v0 + zeta * wn * x0) / wd * np.sin(wd * t)
            )
        elif zeta < 1.0 + 1e-12:
            x = (x0 + (v0 + wn * x0) * t) * np.exp(-wn * t)
        else:
            s = math.sqrt(zeta * zeta - 1.0)
            r1 = -zeta * wn + wn * s
            r2 = -zeta * wn - wn * s
            A = (v0 - r2 * x0) / (r1 - r2)
            B = x0 - A
            x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
    else:
        A_mat = np.array([[0.0, 1.0], [-k / m, -c / m]], dtype=float)
        B_vec = np.array([0.0, 1.0 / m], dtype=float)

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            f = float(force_func(time))
            return A_mat @ state + B_vec * f

        sol = solve_ivp(
            rhs,
            (float(t[0]), float(t[-1])),
            [float(x0), float(v0)],
            t_eval=t,
            rtol=1e-9,
            atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(f"SDOF ODE integration failed: {sol.message}")
        x = sol.y[0]

    return x


# ---------------------------------------------------------------------------
# Fatigue and fracture mechanics
# ---------------------------------------------------------------------------
def basquin_life(
    stress_amplitude: float,
    fatigue_strength_coeff: float,
    fatigue_strength_exponent: float,
) -> float:
    """
    Basquin S-N life estimate.

    Uses:
        S_a = sigma_f' * N^b
    so:
        N = (S_a / sigma_f')^(1/b)
    """
    if stress_amplitude <= 0.0:
        return float("inf")
    if fatigue_strength_coeff <= 0.0 or fatigue_strength_exponent == 0.0:
        return float("inf")
    N = (stress_amplitude / fatigue_strength_coeff) ** (1.0 / fatigue_strength_exponent)
    return float(max(N, 1.0))


def s_n_curve(
    cycles: Sequence[float],
    fatigue_strength_coeff: float,
    fatigue_strength_exponent: float,
) -> np.ndarray:
    """Return Basquin stress amplitudes for a range of cycles."""
    N = np.asarray(cycles, dtype=float)
    return fatigue_strength_coeff * np.power(N, fatigue_strength_exponent)


def goodman_life(
    stress_amplitude: float,
    mean_stress: float,
    ultimate_strength: float,
    fatigue_strength_coeff: float,
    fatigue_strength_exponent: float,
) -> float:
    """
    Finite-life estimate with Goodman mean-stress correction.

    The corrected alternating stress is:
        S_a,corr = S_a / (1 - S_m / S_u)
    """
    if ultimate_strength <= 0.0:
        raise ValueError("Ultimate strength must be positive.")
    denom = 1.0 - mean_stress / ultimate_strength
    if denom <= 0.0:
        return 0.0
    return basquin_life(
        stress_amplitude / denom,
        fatigue_strength_coeff,
        fatigue_strength_exponent,
    )


def gerber_life(
    stress_amplitude: float,
    mean_stress: float,
    ultimate_strength: float,
    fatigue_strength_coeff: float,
    fatigue_strength_exponent: float,
) -> float:
    """
    Finite-life estimate with Gerber mean-stress correction.

    The corrected alternating stress is:
        S_a,corr = S_a / (1 - (S_m / S_u)^2)
    """
    if ultimate_strength <= 0.0:
        raise ValueError("Ultimate strength must be positive.")
    denom = 1.0 - (mean_stress / ultimate_strength) ** 2
    if denom <= 0.0:
        return 0.0
    return basquin_life(
        stress_amplitude / denom,
        fatigue_strength_coeff,
        fatigue_strength_exponent,
    )


def soderberg_life(
    stress_amplitude: float,
    mean_stress: float,
    yield_strength: float,
    fatigue_strength_coeff: float,
    fatigue_strength_exponent: float,
) -> float:
    """
    Finite-life estimate with Soderberg mean-stress correction.

    The corrected alternating stress is:
        S_a,corr = S_a / (1 - S_m / S_y)
    """
    if yield_strength <= 0.0:
        raise ValueError("Yield strength must be positive.")
    denom = 1.0 - mean_stress / yield_strength
    if denom <= 0.0:
        return 0.0
    return basquin_life(
        stress_amplitude / denom,
        fatigue_strength_coeff,
        fatigue_strength_exponent,
    )


def critical_crack_size(
    fracture_toughness: float,
    sigma_max: float,
    geometry_factor: float,
) -> float:
    """Return critical crack length from K_IC = Y sigma_max sqrt(pi a)."""
    if sigma_max <= 0.0:
        return float("inf")
    return (fracture_toughness / (geometry_factor * sigma_max)) ** 2 / math.pi


def stress_intensity_mode_I(
    sigma: float,
    crack_length: float,
    geometry_factor: float,
) -> float:
    """Return mode-I stress intensity factor K_I = Y sigma sqrt(pi a)."""
    return geometry_factor * sigma * math.sqrt(math.pi * max(crack_length, 0.0))


def paris_law_life(
    a0: float,
    sigma_max: float,
    stress_ratio: float,
    geometry_factor: float,
    C: float,
    m: float,
    K_IC: Optional[float] = None,
    a_final: Optional[float] = None,
) -> float:
    """
    Estimate fatigue crack growth life using Paris' law.

        da/dN = C (Delta K)^m
        Delta K = Y Delta sigma sqrt(pi a)

    If a_final is not supplied and K_IC is supplied, the final crack length is
    obtained from K_max = K_IC.
    """
    if a0 <= 0.0:
        raise ValueError("Initial crack length a0 must be positive.")

    if a_final is None:
        if K_IC is None:
            raise ValueError("Provide either a_final or K_IC.")
        a_final = critical_crack_size(K_IC, sigma_max, geometry_factor)

    if a_final <= a0:
        return 0.0

    delta_sigma = sigma_max * (1.0 - stress_ratio)
    if delta_sigma <= 0.0:
        return float("inf")

    A = geometry_factor * delta_sigma * math.sqrt(math.pi)
    if C <= 0.0 or A <= 0.0:
        return float("inf")

    if abs(m - 2.0) < 1e-12:
        N = math.log(a_final / a0) / (C * A * A)
    else:
        exponent = 1.0 - 0.5 * m
        N = (a_final ** exponent - a0 ** exponent) / (
            C * (A ** m) * exponent
        )

    return float(max(N, 0.0))


# ---------------------------------------------------------------------------
# Structural optimization
# ---------------------------------------------------------------------------
@dataclass
class BeamOptimizationResult:
    """Result of a beam cross-section weight optimization."""
    b: float
    h: float
    weight: float
    max_stress: float
    max_deflection: float
    iterations: int
    success: bool
    method: str


def beam_stress_rect(P: float, L: float, b: float, h: float) -> float:
    """Maximum bending stress in a simply supported rectangular beam with central load P."""
    if b <= 0.0 or h <= 0.0:
        raise ValueError("Width and height must be positive.")
    return 3.0 * P * L / (2.0 * b * h * h)


def beam_deflection_rect(P: float, L: float, E: float, b: float, h: float) -> float:
    """Maximum deflection of a simply supported rectangular beam with central load P."""
    if b <= 0.0 or h <= 0.0:
        raise ValueError("Width and height must be positive.")
    return P * L ** 3 / (4.0 * E * b * h ** 3)


def optimize_beam_section_gradient(
    P: float,
    L: float,
    E: float,
    rho: float,
    sigma_allow: float,
    delta_allow: float,
    b_bounds: Tuple[float, float],
    h_bounds: Tuple[float, float],
    x0: Optional[Tuple[float, float]] = None,
) -> BeamOptimizationResult:
    """
    Minimize beam weight using a gradient-based SLSQP method.

    Design variables: [width b, height h].
    Constraints: bending stress <= sigma_allow, midspan deflection <= delta_allow.
    """
    if x0 is None:
        x0 = (
            0.5 * (b_bounds[0] + b_bounds[1]),
            0.5 * (h_bounds[0] + h_bounds[1]),
        )

    def objective(x: np.ndarray) -> float:
        b, h = float(x[0]), float(x[1])
        return rho * b * h * L

    def constraint_stress(x: np.ndarray) -> float:
        b, h = float(x[0]), float(x[1])
        return sigma_allow - beam_stress_rect(P, L, b, h)

    def constraint_deflection(x: np.ndarray) -> float:
        b, h = float(x[0]), float(x[1])
        return delta_allow - beam_deflection_rect(P, L, E, b, h)

    constraints = [
        {"type": "ineq", "fun": constraint_stress},
        {"type": "ineq", "fun": constraint_deflection},
    ]

    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="SLSQP",
        bounds=[b_bounds, h_bounds],
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    b, h = float(result.x[0]), float(result.x[1])
    return BeamOptimizationResult(
        b=b,
        h=h,
        weight=objective(result.x),
        max_stress=beam_stress_rect(P, L, b, h),
        max_deflection=beam_deflection_rect(P, L, E, b, h),
        iterations=int(result.nit),
        success=bool(result.success),
        method="SLSQP",
    )


def optimize_beam_section_ga(
    P: float,
    L: float,
    E: float,
    rho: float,
    sigma_allow: float,
    delta_allow: float,
    b_bounds: Tuple[float, float],
    h_bounds: Tuple[float, float],
    seed: int = 12345,
    popsize: int = 15,
    maxiter: int = 200,
) -> BeamOptimizationResult:
    """
    Minimize beam weight using differential evolution, a genetic/evolutionary algorithm.

    Constraints are enforced with a large quadratic/linear penalty.
    """
    bounds = [b_bounds, h_bounds]

    def penalized_objective(x: np.ndarray) -> float:
        b, h = float(x[0]), float(x[1])
        weight = rho * b * h * L
        sigma = beam_stress_rect(P, L, b, h)
        delta = beam_deflection_rect(P, L, E, b, h)

        penalty = 0.0
        if sigma > sigma_allow:
            penalty += 1.0e12 * (sigma - sigma_allow) / max(sigma_allow, 1e-12)
        if delta > delta_allow:
            penalty += 1.0e12 * (delta - delta_allow) / max(delta_allow, 1e-12)
        return weight + penalty

    result = differential_evolution(
        penalized_objective,
        bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-8,
        polish=True,
        updating="immediate",
    )

    b, h = float(result.x[0]), float(result.x[1])
    return BeamOptimizationResult(
        b=b,
        h=h,
        weight=rho * b * h * L,
        max_stress=beam_stress_rect(P, L, b, h),
        max_deflection=beam_deflection_rect(P, L, E, b, h),
        iterations=int(result.nit),
        success=bool(result.success),
        method="differential_evolution",
    )


# ---------------------------------------------------------------------------
# Rotordynamics
# ---------------------------------------------------------------------------
@dataclass
class JeffcottResponse:
    """Jeffcott rotor unbalance response at a specified operating speed."""
    omega_n: float
    zeta: float
    frequency_ratio: float
    amplitude: float
    phase_rad: float


def jeffcott_unbalance_response(
    unbalance: float,
    rotor_mass: float,
    stiffness: float,
    damping: float,
    omega: float,
) -> JeffcottResponse:
    """
    Return the steady-state unbalance response of a Jeffcott rotor.

    unbalance : U = m_e * e [kg m]
    rotor_mass : total rotor mass m [kg]
    stiffness : support/rotor lateral stiffness k [N/m]
    damping : viscous damping coefficient c [N s/m]
    omega : operating angular speed [rad/s]
    """
    if rotor_mass <= 0.0 or stiffness <= 0.0:
        raise ValueError("Rotor mass and stiffness must be positive.")

    omega_n = math.sqrt(stiffness / rotor_mass)
    zeta = damping / (2.0 * math.sqrt(stiffness * rotor_mass))
    r = omega / omega_n

    denom = math.sqrt(
        (1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2
    )
    amplitude = 0.0 if denom <= 1e-15 else (unbalance / rotor_mass) * (r * r) / denom
    phase = math.atan2(2.0 * zeta * r, 1.0 - r * r)

    return JeffcottResponse(
        omega_n=omega_n,
        zeta=zeta,
        frequency_ratio=r,
        amplitude=amplitude,
        phase_rad=phase,
    )


# ---------------------------------------------------------------------------
# Contact mechanics
# ---------------------------------------------------------------------------
@dataclass
class HertzSphereContact:
    """Hertzian sphere-plane contact results."""
    force: float
    effective_modulus: float
    contact_radius: float
    approach: float
    max_pressure: float


def hertz_sphere_plane(
    force: float,
    sphere_radius: float,
    E1: float,
    nu1: float,
    E2: float,
    nu2: float,
) -> HertzSphereContact:
    """
    Return Hertzian sphere-plane contact results for a normal force.

    The effective modulus is:
        1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2
    """
    if force <= 0.0 or sphere_radius <= 0.0:
        raise ValueError("Force and sphere radius must be positive.")

    E_star = 1.0 / (
        (1.0 - nu1 * nu1) / E1 + (1.0 - nu2 * nu2) / E2
    )
    a = (3.0 * force * sphere_radius / (4.0 * E_star)) ** (1.0 / 3.0)
    approach = a * a / sphere_radius
    p_max = 3.0 * force / (2.0 * math.pi * a * a)

    return HertzSphereContact(
        force=force,
        effective_modulus=E_star,
        contact_radius=a,
        approach=approach,
        max_pressure=p_max,
    )


@dataclass
class HertzCylinderContact:
    """Hertzian cylinder-plane line contact results."""
    force: float
    length: float
    effective_modulus: float
    half_width: float
    max_pressure: float


def hertz_cylinder_plane(
    force: float,
    length: float,
    cylinder_radius: float,
    E1: float,
    nu1: float,
    E2: float,
    nu2: float,
) -> HertzCylinderContact:
    """
    Return Hertzian cylinder-plane line contact results for a normal force.

    The effective modulus is:
        1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2
    """
    if force <= 0.0 or length <= 0.0 or cylinder_radius <= 0.0:
        raise ValueError("Force, contact length, and cylinder radius must be positive.")

    E_star = 1.0 / (
        (1.0 - nu1 * nu1) / E1 + (1.0 - nu2 * nu2) / E2
    )
    half_width = math.sqrt(4.0 * force * cylinder_radius / (math.pi * length * E_star))
    p_max = 2.0 * force / (math.pi * half_width * length)

    return HertzCylinderContact(
        force=force,
        length=length,
        effective_modulus=E_star,
        half_width=half_width,
        max_pressure=p_max,
    )


# ---------------------------------------------------------------------------
# Failure theories
# ---------------------------------------------------------------------------
class FailureTheory:
    """Common static failure criteria and safety factors."""

    @staticmethod
    def max_normal_stress(stress: StressState) -> float:
        return stress.max_normal_stress()

    @staticmethod
    def tresca(stress: StressState) -> float:
        return stress.tresca()

    @staticmethod
    def von_mises(stress: StressState) -> float:
        return stress.von_mises()

    @staticmethod
    def safety_factor(
        stress: StressState,
        strength: float,
        theory: str = "von_mises",
    ) -> float:
        """Return safety factor = strength / equivalent stress."""
        theory_lower = theory.lower()
        if theory_lower in {"von_mises", "vonmises", "distortion_energy", "de"}:
            equivalent = stress.von_mises()
        elif theory_lower in {"tresca", "max_shear", "max_shear_stress"}:
            equivalent = stress.tresca()
        elif theory_lower in {"max_normal", "max_principal", "max_normal_stress"}:
            equivalent = stress.max_normal_stress()
        else:
            raise ValueError(f"Unknown failure theory: {theory}")

        if equivalent <= 0.0:
            return float("inf")
        return float(strength / equivalent)

    @staticmethod
    def mohr_coulomb_safety_factor(
        stress: StressState,
        ultimate_tensile_strength: float,
        ultimate_compressive_strength: float,
    ) -> float:
        """Return a simple Mohr-Coulomb safety factor."""
        index = stress.mohr_coulomb_index(
            ultimate_tensile_strength,
            ultimate_compressive_strength,
        )
        if index <= 0.0:
            return float("inf")
        return float(1.0 / index)


# ---------------------------------------------------------------------------
# Demonstration routines
# ---------------------------------------------------------------------------
def _demo_fea() -> None:
    print("\n=== Finite Element Analysis Demo ===")
    coords = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.8),
    ]
    E = 210.0e9
    A = 1.0e-4
    elements = [
        (0, 1, E, A),
        (1, 2, E, A),
        (2, 0, E, A),
    ]
    fixed_dofs = [0, 1]       # Node 0 fully fixed
    loads = {5: -1000.0}      # Vertical downward force at node 2
    sol = solve_truss(coords, elements, fixed_dofs, loads)

    print("Nodal displacements [m]:")
    print(sol.displacements.reshape(-1, 2))
    print("Axial stresses [Pa]:")
    print(sol.axial_stresses)
    print("Reaction vector [N]:")
    print(sol.reactions.reshape(-1, 2))

    Kp = rectangular_plate_element_stiffness(
        a=0.5,
        b=0.3,
        thickness=0.005,
        E=210.0e9,
        nu=0.30,
        n_gauss=2,
    )
    print("Rectangular plate element stiffness shape:", Kp.shape)
    print("Trace of plate stiffness matrix:", np.trace(Kp))

    Kb = beam_element_stiffness(E=210.0e9, I=1.0e-6, L=1.0)
    print("Beam element stiffness shape:", Kb.shape)


def _demo_stress() -> None:
    print("\n=== Stress Transformation and Failure Theory Demo ===")
    sigma_x = 80.0e6
    sigma_y = 20.0e6
    tau_xy = 30.0e6

    s1, s2, theta_p = principal_stresses_2d(sigma_x, sigma_y, tau_xy)
    center, radius, smax, smin = mohr_circle(sigma_x, sigma_y, tau_xy)
    vm = von_mises_2d(sigma_x, sigma_y, tau_xy)
    tr = tresca_2d(sigma_x, sigma_y, tau_xy)

    print(f"Principal stresses: sigma1 = {s1/1e6:.3f} MPa, sigma2 = {s2/1e6:.3f} MPa")
    print(f"Principal angle: {math.degrees(theta_p):.3f} deg")
    print(f"Mohr circle: center = {center/1e6:.3f} MPa, radius = {radius/1e6:.3f} MPa")
    print(f"Von Mises stress: {vm/1e6:.3f} MPa")
    print(f"Tresca stress: {tr/1e6:.3f} MPa")

    stress = StressState.from_components(sigma_x, sigma_y, 0.0, tau_xy, 0.0, 0.0)
    print("Failure safety factor (von Mises, Sy = 355 MPa):",
          FailureTheory.safety_factor(stress, 355.0e6, "von_mises"))
    print("Failure safety factor (Tresca, Sy = 355 MPa):",
          FailureTheory.safety_factor(stress, 355.0e6, "tresca"))
    print("Mohr-Coulomb safety factor (Sut=500 MPa, Suc=500 MPa):",
          FailureTheory.mohr_coulomb_safety_factor(stress, 500.0e6, 500.0e6))

    # Mohr's circle plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Circle((center, 0.0), radius, fill=False, lw=2))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.plot([sigma_x, sigma_y], [0.0, 0.0], "ko", ms=5)
    ax.set_xlabel("Normal stress [Pa]")
    ax.set_ylabel("Shear stress [Pa]")
    ax.set_title("Mohr's Circle")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("mohr_circle.png", dpi=140)
    plt.close(fig)


def _demo_vibration() -> None:
    print("\n=== Mechanical Vibrations Demo ===")
    K = np.array([
        [2.0e7, -1.0e7],
        [-1.0e7, 2.0e7],
    ], dtype=float)
    M = np.eye(2) * 1000.0
    C = 0.02 * K

    modal = modal_analysis(K, M, C)
    print("Natural frequencies [Hz]:", modal.natural_frequencies_hz)
    print("Damping ratios [-]:", modal.damping_ratios)
    print("Mode shapes:\n", modal.mode_shapes)

    # Plot mode shapes
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(modal.mode_shapes.shape[1]):
        ax.plot([0.0, 1.0], modal.mode_shapes[:, j], marker="o", label=f"Mode {j+1}")
    ax.set_xlabel("Normalized coordinate")
    ax.set_ylabel("Normalized mode amplitude")
    ax.set_title("MDOF Mode Shapes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("modal_modes.png", dpi=140)
    plt.close(fig)

    # SDOF free response
    x = sdof_response(m=1.0, k=10000.0, c=10.0, x0=1.0, v0=0.0)
    print("SDOF displacement response length:", x.size)
    print("SDOF response first 5 samples:", x[:5])


def _demo_fatigue() -> None:
    print("\n=== Fatigue and Fracture Mechanics Demo ===")
    sigma_f = 900.0e6
    b = -0.10

    N = basquin_life(300.0e6, sigma_f, b)
    print(f"Basquin life at 300 MPa: {N:.3e} cycles")

    N_goodman = goodman_life(300.0e6, 50.0e6, 500.0e6, sigma_f, b)
    N_gerber = gerber_life(300.0e6, 50.0e6, 500.0e6, sigma_f, b)
    N_soderberg = soderberg_life(300.0e6, 50.0e6, 355.0e6, sigma_f, b)
    print(f"Goodman-corrected life: {N_goodman:.3e} cycles")
    print(f"Gerber-corrected life: {N_gerber:.3e} cycles")
    print(f"Soderberg-corrected life: {N_soderberg:.3e} cycles")

    N_paris = paris_law_life(
        a0=0.001,
        sigma_max=300.0e6,
        stress_ratio=0.1,
        geometry_factor=1.0,
        C=1.0e-29,
        m=3.0,
        K_IC=50.0e6,
    )
    print(f"Paris law crack-growth life: {N_paris:.3e} cycles")

    N_plot = np.logspace(3, 8, 100)
    Sa_plot = s_n_curve(N_plot, sigma_f, b)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(N_plot, Sa_plot / 1e6)
    ax.set_xlabel("Cycles N")
    ax.set_ylabel("Stress amplitude [MPa]")
    ax.set_title("Basquin S-N Curve")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("sn_curve.png", dpi=140)
    plt.close(fig)


def _demo_optimization() -> None:
    print("\n=== Structural Optimization Demo ===")
    P = 10.0e3
    L = 2.0
    E = 210.0e9
    rho = 7850.0
    sigma_allow = 200.0e6
    delta_allow = 0.005

    grad = optimize_beam_section_gradient(
        P=P,
        L=L,
        E=E,
        rho=rho,
        sigma_allow=sigma_allow,
        delta_allow=delta_allow,
        b_bounds=(0.02, 0.30),
        h_bounds=(0.02, 0.40),
    )
    print("SLSQP optimization:")
    print(f"  b = {grad.b:.6f} m, h = {grad.h:.6f} m")
    print(f"  weight = {grad.weight:.3f} kg")
    print(f"  max stress = {grad.max_stress/1e6:.3f} MPa")
    print(f"  max deflection = {grad.max_deflection*1e3:.4f} mm")
    print(f"  success = {grad.success}, iterations = {grad.iterations}")

    ga = optimize_beam_section_ga(
        P=P,
        L=L,
        E=E,
        rho=rho,
        sigma_allow=sigma_allow,
        delta_allow=delta_allow,
        b_bounds=(0.02, 0.30),
        h_bounds=(0.02, 0.40),
        seed=123,
        popsize=10,
        maxiter=80,
    )
    print("Differential evolution optimization:")
    print(f"  b = {ga.b:.6f} m, h = {ga.h:.6f} m")
    print(f"  weight = {ga.weight:.3f} kg")
    print(f"  max stress = {ga.max_stress/1e6:.3f} MPa")
    print(f"  max deflection = {ga.max_deflection*1e3:.4f} mm")
    print(f"  success = {ga.success}, iterations = {ga.iterations}")


def _demo_rotordynamics_contact_failure() -> None:
    print("\n=== Rotordynamics, Contact Mechanics, and Failure Theory Demo ===")
    rotor = jeffcott_unbalance_response(
        unbalance=0.01,
        rotor_mass=20.0,
        stiffness=2.0e6,
        damping=200.0,
        omega=300.0,
    )
    print(f"Jeffcott critical speed: {rotor.omega_n:.3f} rad/s")
    print(f"Frequency ratio: {rotor.frequency_ratio:.4f}")
    print(f"Unbalance response amplitude: {rotor.amplitude*1e6:.4f} um")

    sphere = hertz_sphere_plane(
        force=1000.0,
        sphere_radius=0.02,
        E1=210.0e9,
        nu1=0.30,
        E2=210.0e9,
        nu2=0.30,
    )
    print(f"Hertz sphere-plane contact radius: {sphere.contact_radius*1e3:.4f} mm")
    print(f"Hertz sphere-plane max pressure: {sphere.max_pressure/1e6:.3f} MPa")

    cyl = hertz_cylinder_plane(
        force=2000.0,
        length=0.05,
        cylinder_radius=0.03,
        E1=210.0e9,
        nu1=0.30,
        E2=210.0e9,
        nu2=0.30,
    )
    print(f"Hertz cylinder-plane half-width: {cyl.half_width*1e3:.4f} mm")
    print(f"Hertz cylinder-plane max pressure: {cyl.max_pressure/1e6:.3f} MPa")

    stress = StressState.from_components(120.0e6, 40.0e6, 0.0, 20.0e6, 0.0, 0.0)
    print("Failure safety factor (von Mises):",
          FailureTheory.safety_factor(stress, 250.0e6, "von_mises"))
    print("Failure safety factor (Tresca):",
          FailureTheory.safety_factor(stress, 250.0e6, "tresca"))
    print("Failure safety factor (max normal):",
          FailureTheory.safety_factor(stress, 250.0e6, "max_normal"))


def run_all_demos() -> None:
    """Execute all demonstration routines."""
    np.set_printoptions(precision=6, suppress=True)
    _demo_fea()
    _demo_stress()
    _demo_vibration()
    _demo_fatigue()
    _demo_optimization()
    _demo_rotordynamics_contact_failure()
    print("\nAll demonstrations completed.")
    print("Figures saved: mohr_circle.png, modal_modes.png, sn_curve.png")


if __name__ == "__main__":
    run_all_demos()