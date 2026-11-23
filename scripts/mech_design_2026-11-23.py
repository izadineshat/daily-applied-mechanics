#!/usr/bin/env python3
"""Standalone mechanical engineering analysis toolkit."""

from __future__ import annotations

import numpy as np
from scipy import linalg, optimize, signal
import matplotlib.pyplot as plt


# ===================== Stress & Strain Transformations =====================
def normal_stress_at_angle(sigma_x: float, sigma_y: float, tau_xy: float, theta_deg: float) -> float:
    """Normal stress on a plane inclined at theta."""
    theta = np.deg2rad(theta_deg)
    return 0.5 * (sigma_x + sigma_y) + 0.5 * (sigma_x - sigma_y) * np.cos(2.0 * theta) + tau_xy * np.sin(2.0 * theta)


def shear_stress_at_angle(sigma_x: float, sigma_y: float, tau_xy: float, theta_deg: float) -> float:
    """Engineering shear stress on a plane inclined at theta."""
    theta = np.deg2rad(theta_deg)
    return -0.5 * (sigma_x - sigma_y) * np.sin(2.0 * theta) + tau_xy * np.cos(2.0 * theta)


def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float):
    """Principal stresses for plane stress/strain."""
    disc = np.sqrt((sigma_x - sigma_y) ** 2 + 4.0 * tau_xy ** 2)
    s_avg = 0.5 * (sigma_x + sigma_y)
    s1 = s_avg + 0.5 * disc
    s2 = s_avg - 0.5 * disc
    return s1, s2


def mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float):
    """Return Mohr's circle data."""
    s_avg = 0.5 * (sigma_x + sigma_y)
    radius = np.sqrt(((sigma_x - sigma_y) ** 2) / 4.0 + tau_xy ** 2)
    s1, s2 = principal_stresses(sigma_x, sigma_y, tau_xy)
    return {
        "center": (s_avg, 0.0),
        "radius": radius,
        "sigma_1": s1,
        "sigma_2": s2,
        "tau_max": radius,
    }


def von_mises_stress(sigma_1: float, sigma_2: float, sigma_3: float) -> float:
    """Von Mises stress."""
    return np.sqrt(
        sigma_1 ** 2 + sigma_2 ** 2 + sigma_3 ** 2
        - sigma_1 * sigma_2 - sigma_2 * sigma_3 - sigma_3 * sigma_1
    )


def max_shear_stress(sigma_1: float, sigma_2: float, sigma_3: float) -> float:
    """Tresca maximum shear stress."""
    return 0.5 * max(
        abs(sigma_1 - sigma_2),
        abs(sigma_2 - sigma_3),
        abs(sigma_1 - sigma_3),
    )


# ===================== Truss / Beam FEA =====================
class TrussAssembler:
    def __init__(self, nodes: np.ndarray, elements: list, E: float = 200e9, A: float = 1e-4):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = elements  # list of (i, j, A, E)
        self.E = E
        self.A = A

    def assemble(self) -> np.ndarray:
        n_nodes = len(self.nodes)
        n_dof = 2 * n_nodes
        K = np.zeros((n_dof, n_dof))
        for elem in self.elements:
            i, j, A, E = elem
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
            L = np.hypot(x2 - x1, y2 - y1)
            k = E * A / L
            c = (x2 - x1) / L
            s = (y2 - y1) / L
            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K_local = k * np.array([
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ])
            for r in range(4):
                for col in range(4):
                    K[dofs[r], dofs[col]] += K_local[r, col]
        return K


def simply_supported_beam_deflection(F: float, L: float, E: float, I: float) -> float:
    """Max deflection of simply supported beam with central point load."""
    return F * L ** 3 / (48.0 * E * I)


# ===================== Vibrations =====================
class SDOF:
    def __init__(self, mass: float, stiffness: float, damping: float = 0.0):
        self.m = mass
        self.k = stiffness
        self.c = damping

    def omega_n(self) -> float:
        return np.sqrt(self.k / self.m)

    def zeta(self) -> float:
        return self.c / (2.0 * np.sqrt(self.k * self.m))

    def response(self, t: np.ndarray, x0: float, v0: float, force: float = 0.0):
        m = self.m
        k = self.k
        c = self.c
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        B = np.array([[0.0], [1.0 / m]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        sys = signal.StateSpace(A, B, C, D)
        t = np.asarray(t, dtype=float)
        u = np.full((1, len(t)), force, dtype=float)
        X0 = np.array([[x0], [v0]])
        Y0 = X0.copy()
        t_out, y, x = signal.lsim(sys, u, t, X0=X0, Y0=Y0)
        return y[0].copy(), y[1].copy()


class MDOF:
    def __init__(self, mass: np.ndarray, stiffness: np.ndarray, damping: np.ndarray = None):
        self.M = np.asarray(mass, dtype=float)
        self.K = np.asarray(stiffness, dtype=float)
        if damping is None:
            self.C = np.zeros_like(self.K)
        else:
            self.C = np.asarray(damping, dtype=float)

    def modal_analysis(self):
        vals, vecs = linalg.eigh(self.K, self.M)
        wn = np.sqrt(vals)
        phi = vecs / np.sqrt(np.diag(self.M))[:, None]
        zeta = np.zeros(len(wn))
        for i in range(len(wn)):
            zeta[i] = 0.5 * (self.C @ phi[:, i]) @ phi[:, i] / wn[i]
        return wn, phi, zeta


# ===================== Fatigue & Fracture Mechanics =====================
def basquin_life(sigma_a: float, sigma_f: float, b: float) -> float:
    """Basquin fatigue life: sigma_a = sigma_f * (2N)^(-b)."""
    return 0.5 * (sigma_f / sigma_a) ** (1.0 / b)


def paris_law_life(inicial: float, final: float, delta_sigma: float, C: float, m: float, Y: float = 1.0) -> float:
    """Paris law crack growth integration."""
    from scipy.integrate import quad

    def integrand(a: float):
        delta_K = Y * delta_sigma * np.sqrt(np.pi * a)
        return 1.0 / (C * (delta_K ** m))

    N, _ = quad(integrand, inicial, final, limit=100)
    return N


def goodman_allowable_amplitude(sigma_mean: float, sigma_ut: float, sigma_max: float) -> float:
    """Goodman line allowable stress amplitude."""
    return (sigma_max - sigma_mean) * sigma_ut / (sigma_ut - sigma_mean)


# ===================== Structural Optimization =====================
def optimize_beam_weight_round_bar(F: float, L: float, E: float, sigma_allow: float, delta_allow: float, rho: float):
    """Minimize round bar weight subject to stress and deflection constraints."""

    def weight(d: np.ndarray) -> float:
        return rho * np.pi / 4.0 * L * d[0] ** 2

    def stress_ok(d: np.ndarray) -> float:
        return sigma_allow - 16.0 * F * L / (np.pi * d[0] ** 3)

    def deflection_ok(d: np.ndarray) -> float:
        return delta_allow - 16.0 * F * L ** 3 / (3.0 * E * np.pi * d[0] ** 4)

    cons = [
        {"type": "ineq", "fun": stress_ok},
        {"type": "ineq", "fun": deflection_ok},
    ]
    res = optimize.minimize(
        weight,
        x0=np.array([0.01]),
        bounds=[(1e-4, 0.1)],
        constraints=cons,
        method="SLSQP",
    )
    d = res.x[0]
    w = rho * np.pi / 4.0 * L * d ** 2
    return d, w, res.success


# ===================== Rotordynamics / Contact / Failure =====================
def hertz_contact_force(delta: float, E_star: float, R_star: float) -> float:
    """Hertzian contact force between two curved surfaces."""
    if delta <= 0.0:
        return 0.0
    return (4.0 / 3.0) * E_star * np.sqrt(R_star) * delta ** 1.5


def rotordynamic_bearing_forces(bearing_stiffness: float, bearing_damping: float, omega: float, force: float) -> tuple:
    """Simple SDOF rotordynamic bearing force / response parameters."""
    k = bearing_stiffness
    c = bearing_damping
    m = 1.0
    wn = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    return wn, zeta, force / (k * (1.0 - (force / (k * wn ** 2)) if force < k * wn ** 2 else 0.0))


# ===================== Main =====================
def main() -> None:
    # Stress transformation example
    sigma_x, sigma_y, tau_xy = 100e6, -50e6, 25e6
    s1, s2 = principal_stresses(sigma_x, sigma_y, tau_xy)
    s3 = 0.0
    vm = von_mises_stress(s1, s2, s3)
    ms = max_shear_stress(s1, s2, s3)
    print("Principal stresses:", s1, s2, s3)
    print("Von Mises:", vm)
    print("Max shear:", ms)

    # Truss assembly
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    elements = [(0, 1, 1e-4, 200e9), (1, 2, 1e-4, 200e9), (0, 2, 1e-4, 200e9)]
    ta = TrussAssembler(nodes, elements)
    K = ta.assemble()
    print("Truss K shape:", K.shape)

    # Beam deflection
    F = 1000.0
    L = 2.0
    E = 200e9
    I = 1e-6
    delta = simply_supported_beam_deflection(F, L, E, I)
    print("Beam deflection:", delta)

    # Vibration
    sf = SDOF(mass=1.0, stiffness=100.0, damping=0.05)
    wn = sf.omega_n()
    zeta = sf.zeta()
    t = np.linspace(0, 10, 100)
    x, v = sf.response(t, 0.0, 0.0, force=0.0)
    print("SDOF omega_n:", wn, "zeta:", zeta)

    # Fatigue
    sigma_a = 50e6
    sigma_f = 400e6
    b = 0.08
    N = basquin_life(sigma_a, sigma_f, b)
    print("Fatigue life:", N)

    # Optimization
    d, w, ok = optimize_beam_weight_round_bar(
        F=1000.0,
        L=2.0,
        E=200e9,
        sigma_allow=200e6,
        delta_allow=0.001,
        rho=7850.0,
    )
    print("Optimized diameter:", d, "weight:", w, ok)

    # Contact
    F_h = hertz_contact_force(delta=0.001, E_star=100e9, R_star=0.01)
    print("Hertz contact force:", F_h)

    plt.show()


if __name__ == "__main__":
    main()