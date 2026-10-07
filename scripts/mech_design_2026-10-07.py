#!/usr/bin/env python3
"""
mech_toolkit.py
A collection of reusable functions and classes for common mechanical engineering
calculations: finite element stiffness (truss, beam, plate), stress/strain
transformations, vibration analysis, fatigue/fracture, structural optimization,
rotordynamics, contact mechanics and failure theories.

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable


# ======================
#  Finite Element Tools
# ======================

class TrussElement:
    """2‑node linear truss element in 2‑D space.

    Attributes
    ----------
    E : float
        Young's modulus.
    A : float
        Cross‑sectional area.
    nodes : Tuple[int, int]
        Global node numbers (i, j).
    coords : Tuple[np.ndarray, np.ndarray]
        Nodal coordinates (xi, yi), (xj, yj).
    """
    def __init__(self, E: float, A: float, nodes: Tuple[int, int],
                 coords: Tuple[np.ndarray, np.ndarray]):
        self.E = E
        self.A = A
        self.nodes = nodes
        self.coords = coords
        self.length = np.linalg.norm(coords[1] - coords[0])
        self._direction = (coords[1] - coords[0]) / self.length
        self._c, self._s = self._direction

    def local_stiffness(self) -> np.ndarray:
        """Element stiffness matrix in local coordinates (axial only)."""
        k = self.E * self.A / self.length
        return np.array([[ k, -k],
                         [-k,  k]])

    def transformation_matrix(self) -> np.ndarray:
        """Transformation matrix from local to global DOFs."""
        T = np.array([[ self._c,  self._s, 0.0, 0.0],
                      [ 0.0,   0.0,   self._c,  self._s]])
        return T

    def global_stiffness(self) -> np.ndarray:
        """Element stiffness matrix in global coordinates."""
        k_local = self.local_stiffness()
        T = self.transformation_matrix()
        return T.T @ k_local @ T


class BeamElement:
    """2‑node Euler‑Bernoulli beam element (axial + bending in local x‑y plane).

    Degrees of freedom per node: [u, v, theta] (axial, transverse, rotation).
    """
    def __init__(self, E: float, A: float, I: float, L: float,
                 nodes: Tuple[int, int],
                 coords: Tuple[np.ndarray, np.ndarray]):
        self.E = E
        self.A = A
        self.I = I
        self.L = L
        self.nodes = nodes
        self.coords = coords
        # direction cosines (same as TrussElement)
        self._c = (coords[1][0] - coords[0][0]) / L
        self._s = (coords[1][1] - coords[0][1]) / L

    def local_stiffness(self) -> np.ndarray:
        """12×12 local stiffness matrix (ordered: u1, v1, θ1, u2, v2, θ2)."""
        k_axial = self.E * self.A / self.L
        k_bend = self.E * self.I / self.L**3
        k = np.zeros((6, 6))
        # axial
        k[0, 0] = k[3, 3] = k_axial
        k[0, 3] = k[3, 0] = -k_axial
        # bending
        k[1, 1] = k[4, 4] = 12 * k_bend
        k[1, 4] = k[4, 1] = -12 * k_bend
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = 6 * k_bend / self.L
        k[4, 2] = k[2, 4] = k[4, 5] = k[5, 4] = -6 * k_bend / self.L
        k[2, 2] = k[5, 5] = 4 * k_bend * self.L
        k[2, 5] = k[5, 2] = 2 * k_bend * self.L
        return k

    def transformation_matrix(self) -> np.ndarray:
        """6×6 transformation matrix (local → global)."""
        T = np.zeros((6, 6))
        # axial part
        T[0, 0] = T[3, 3] = self._c
        T[0, 3] = T[3, 0] = self._s
        T[0, 3] = -self._s  # will be corrected below
        # Actually we need full 2‑D rotation for u,v and same for rotations
        R = np.array([[ self._c,  self._s],
                      [-self._s,  self._c]])
        T[0:2, 0:2] = R
        T[3:5, 3:5] = R
        # rotations transform as scalar (out‑of‑plane) → same
        T[2, 2] = T[5, 5] = 1.0
        return T

    def global_stiffness(self) -> np.ndarray:
        """Global stiffness matrix."""
        T = self.transformation_matrix()
        return T.T @ self.local_stiffness() @ T


def assemble_global_stiffness(elements: List[object],
                              num_nodes: int,
                              dof_per_node: int) -> np.ndarray:
    """Assemble global stiffness matrix from a list of element objects.

    Each element must provide a method `global_stiffness()` returning a
    (dof_per_node*len(nodes), ) square matrix and an attribute `nodes`
    (tuple of global node indices).
    """
    size = num_nodes * dof_per_node
    K = np.zeros((size, size))
    for ele in elements:
        k_glob = ele.global_stiffness()
        dofs = []
        for n in ele.nodes:
            dofs.extend([n*dof_per_node + i for i in range(dof_per_node)])
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += k_glob[i, j]
    return K


def apply_boundary_conditions(K: np.ndarray,
                              F: np.ndarray,
                              fixed_dofs: List[int],
                              prescribed_vals: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Dirichlet boundary conditions using the penalty method (large number)."""
    if prescribed_vals is None:
        prescribed_vals = [0.0] * len(fixed_dofs)
    K_mod = K.copy()
    F_mod = F.copy()
    penalty = 1e12 * np.max(np.abs(K)) if np.max(np.abs(K)) != 0 else 1e12
    for dof, val in zip(fixed_dofs, prescribed_vals):
        K_mod[dof, dof] += penalty
        F_mod[dof] += penalty * val
    return K_mod, F_mod


def linear_static_solve(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Solve K*u = F for displacement vector u."""
    return solve(K, F)


# ======================
#  Stress / Strain Tools
# ======================

def mohr_circle(sigx: float, sigy: float, tauxy: float) -> Tuple[float, float, float, float]:
    """Return principal stresses, max shear stress, and principal angle (degrees)."""
    avg = (sigx + sigy) / 2.0
    R = np.sqrt(((sigx - sigy) / 2.0)**2 + tauxy**2)
    sig1 = avg + R
    sig2 = avg - R
    tau_max = R
    theta_p = 0.5 * np.degrees(np.arctan2(2*tauxy, sigx - sigy))
    return sig1, sig2, tau_max, theta_p


def von_mises(stress: np.ndarray) -> float:
    """Von Mises equivalent stress from a 3x3 stress tensor."""
    s = stress - np.trace(stress)/3.0 * np.eye(3)
    return np.sqrt(1.5 * np.sum(s*s))


def tresca(stress: np.ndarray) -> float:
    """Tresca equivalent stress (max shear * 2)."""
    eigvals = np.linalg.eigvalsh(stress)
    return np.max(eigvals) - np.min(eigvals)


# ======================
#  Vibration Tools
# ======================

def modal_analysis(M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem K*phi = omega^2 * M*phi.
    Returns natural frequencies (rad/s) and mode shapes (columns)."""
    eigvals, eigvecs = eig(K, M)
    idx = np.argsort(eigvals)
    omega = np.sqrt(eigvals[idx])
    phi = eigvecs[:, idx]
    return omega, phi


def sdofforced_response(m: float, k: float, c: float,
                        F0: float, omega: float,
                        t: np.ndarray) -> np.ndarray:
    """Steady‑state harmonic response amplitude of a SDOF system."""
    r = omega / np.sqrt(k/m)
    zeta = c / (2*np.sqrt(m*k))
    X = F0 / k * 1.0 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    phase = np.arctan2(2*zeta*r, 1 - r**2)
    return X * np.sin(omega*t - phase)


def mdof_modal_superposition(M: np.ndarray, K: np.ndarray, C: np.ndarray,
                             F: np.ndarray, t: np.ndarray) -> np.ndarray:
    """MDOF forced response using modal superposition (assuming proportional damping)."""
    omega, phi = modal_analysis(M, K)
    # modal matrices
    Mn = phi.T @ M @ phi
    Kn = phi.T @ K @ phi
    Cn = phi.T @ C @ phi
    # modal forces
    Fn = phi.T @ F
    # solve each modal SDOF using Newmark beta (average acceleration)
    dt = t[1] - t[0]
    beta = 0.25
    gamma = 0.5
    u_modal = np.zeros((len(omega), len(t)))
    v_modal = np.zeros_like(u_modal)
    a_modal = np.zeros_like(u_modal)
    for i in range(len(omega)):
        mi = Mn[i, i]
        ki = Kn[i, i]
        ci = Cn[i, i]
        fi = Fn[i, :]
        # effective stiffness
        keff = ki + gamma/(beta*dt)*ci + 1/(beta*dt**2)*mi
        # initial conditions
        u_modal[i,0] = 0.0
        v_modal[i,0] = 0.0
        a_modal[i,0] = (fi[0] - ci*v_modal[i,0] - ki*u_modal[i,0])/mi
        for n in range(1, len(t)):
            dp = fi[n] + mi*(1/(beta*dt**2)*u_modal[i,n-1] + 1/(beta*dt)*v_modal[i,n-1] + (1/(2*beta)-1)*a_modal[i,n-1]) \
                 + ci*(gamma/(beta*dt)*u_modal[i,n-1] + (gamma/beta -1)*v_modal[i,n-1] + dt*(gamma/(2*beta)-1)*a_modal[i,n-1])
            u_modal[i,n] = dp / keff
            v_modal[i,n] = gamma/(beta*dt)*(u_modal[i,n]-u_modal[i,n-1]) + (1-gamma/beta)*v_modal[i,n-1] + dt*(1-gamma/(2*beta))*a_modal[i,n-1]
            a_modal[i,n] = 1/(beta*dt**2)*(u_modal[i,n]-u_modal[i,n-1]) - 1/(beta*dt)*v_modal[i,n-1] - (1/(2*beta)-1)*a_modal[i,n-1]
    u_phys = phi @ u_modal
    return u_phys  # shape (dof, time)


def state_space(A: np.ndarray, B: np.ndarray,
                C: np.ndarray = None, D: np.ndarray = None) -> dict:
    """Return a dictionary representing a linear state‑space model."""
    if C is None:
        C = np.eye(A.shape[0])
    if D is None:
        D = np.zeros((C.shape[0], B.shape[1]))
    return {"A": A, "B": B, "C": C, "D": D}


# ======================
#  Fatigue & Fracture
# ======================

def basquin(N: float, sigma_f_prime: float, b: float) -> float:
    """Basquin equation: sigma_a = sigma_f' * (2N)^b."""
    return sigma_f_prime * (2.0 * N) ** b


def goodman(sigma_a: float, sigma_m: float,
            sigma_e: float, sigma_uts: float) -> bool:
    """Goodman criterion: sigma_a/sigma_e + sigma_m/sigma_uts <= 1."""
    return sigma_a / sigma_e + sigma_m / sigma_uts <= 1.0


def gerber(sigma_a: float, sigma_m: float,
           sigma_e: float, sigma_uts: float) -> bool:
    """Gerber criterion: (sigma_a/sigma_e)^2 + sigma_m/sigma_uts <= 1."""
    return (sigma_a / sigma_e)**2 + sigma_m / sigma_uts <= 1.0


def paris_law(delta_K: float, C: float, m: float) -> float:
    """Paris law: da/dN = C * (ΔK)^m."""
    return C * (delta_K) ** m


# ======================
#  Structural Optimization
# ======================

def optimize_beam_rectangular(L: float, w_load: float,
                              sigma_allow: float, E: float,
                              rho: float) -> dict:
    """Minimize weight of a simply‑supported rectangular beam under a uniform load.

    Design variables: height h (b fixed to b0). Weight = rho * b * h * L.
    Bending stress sigma = M*c/I <= sigma_allow, with M = w*L^2/8, c = h/2,
    I = b*h^3/12.
    """
    b0 = 0.05  # fixed width [m]

    def weight(h):
        return rho * b0 * h[0] * L

    def stress_constraint(h):
        h = h[0]
        M = w_load * L**2 / 8.0
        I = b0 * h**3 / 12.0
        sigma = M * (h/2) / I
        return sigma_allow - sigma  # >=0

    cons = ({'type': 'ineq', 'fun': stress_constraint})
    bounds = [(0.01, 0.5)]  # h in [1cm, 50cm]
    res = minimize(weight, x0=[0.1], bounds=bounds, constraints=cons)
    h_opt = res.x[0]
    return {
        "height_opt_m": h_opt,
        "weight_N": weight([h_opt]) * 9.81,  # convert mass to weight
        "bending_stress_MPa": (w_load * L**2 / 8.0) * (h_opt/2) / (b0 * h_opt**3 / 12.0) / 1e6,
        "success": res.success,
        "message": res.message
    }


# ======================
#  Rotordynamics
# ======================

def jeffcott_critical_speed(m: float, k: float) -> float:
    """Critical speed (rad/s) of a simple Jeffcott rotor."""
    return np.sqrt(k / m)


def jeffcott_unbalance_response(m: float, k: float, c: float,
                                e: float, Omega: np.ndarray) -> np.ndarray:
    """Steady‑state displacement amplitude due to mass unbalance e* m."""
    omega_n = np.sqrt(k/m)
    zeta = c/(2*np.sqrt(m*k))
    r = Omega / omega_n
    X = m * e * r**2 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    return X


# ======================
#  Contact Mechanics
# ======================

def hertzian_sphere_contact(F: float, R1: float, R2: float,
                            E1: float, nu1: float,
                            E2: float, nu2: float) -> Tuple[float, float]:
    """Hertzian contact for two spheres.
    Returns contact radius a [m] and maximum pressure p0 [Pa].
    """
    R = 1.0 / (1.0/R1 + 1.0/R2)
    E_star = 1.0 / ((1 - nu1**2)/E1 + (1 - nu2**2)/E2)
    a = ( (3.0 * F * R) / (4.0 * E_star) ) ** (1.0/3.0)
    p0 = (3.0 * F) / (2.0 * np.pi * a**2)
    return a, p0


# ======================
#  Failure Theories
# ======================

def max_normal_stress(sigma1: float, sigma2: float,
                      sigma_allow: float) -> bool:
    """Maximum normal stress theory."""
    return max(abs(sigma1), abs(sigma2)) <= sigma_allow


def max_shear_stress(tau_xy: float, sigma_x: float, sigma_y: float,
                     tau_allow: float) -> bool:
    """Maximum shear stress (Tresca) theory."""
    tau_max = np.sqrt(((sigma_x - sigma_y)/2.0)**2 + tau_xy**2)
    return tau_max <= tau_allow


def mohr_coulomb(sigma_n: float, tau: float,
                 c: float, phi: float) -> bool:
    """Mohr‑Coulomb failure criterion."""
    return tau <= c + sigma_n * np.tan(np.radians(phi))


# ======================
#  Demo / Test Block
# ======================

if __name__ == "__main__":
    np.random.seed(42)

    # ---- 1D Truss Example ----
    E = 210e9          # Pa
    A = 0.01           # m^2
    coords = (np.array([0.0, 0.0]), np.array([2.0, 0.0]))
    truss = TrussElement(E, A, (0, 1), coords)
    K_ele = truss.global_stiffness()
    print("Truss element stiffness (N/m):\n", K_ele)

    # ---- Beam Example ----
    Lb = 2.0
    Ib = 8.33e-6       # m^4 for 0.05x0.2 rectangle
    beam = BeamElement(E, A, Ib, Lb, (0, 1),
                       (np.array([0.0, 0.0]), np.array([Lb, 0.0]))
                      )
    K_beam = beam.global_stiffness()
    print("\nBeam element stiffness (N/m, N/rad):\n", K_beam)

    # ---- Assemble a simple 2‑node frame (truss + beam) ----
    num_nodes = 2
    dof_per_node = 3   # [u, v, theta]
    K_global = assemble_global_stiffness([truss, beam], num_nodes, dof_per_node)
    # Apply a vertical downward load at node 2 (DOF 1)
    F = np.zeros(num_nodes*dof_per_node)
    F[1] = -10e3       # -10 kN
    # Fix node 0 (all DOFs)
    fixed = [0, 1, 2]
    Kbc, Fbc = apply_boundary_conditions(K_global, F, fixed)
    U = linear_static_solve(Kbc, Fbc)
    print("\nNodal displacements [u, v, theta] (m, m, rad):")
    print(U.reshape((num_nodes, dof_per_node)))

    # ---- Stress Transformation Example ----
    sigx, sigy, tauxy = 50e6, -20e6, 30e6
    s1, s2, tmax, theta = mohr_circle(sigx, sigy, tauxy)
    print(f"\nMohr's circle: σ1={s1/1e6:.2f} MPa, σ2={s2/1e6:.2f} MPa, "
          f"τmax={tmax/1e6:.2f} MPa, θp={theta:.2f}°")
    stress_tensor = np.array([[sigx, tauxy, 0],
                              [tauxy, sigy, 0],
                              [0, 0, 0]])
    print(f"Von Mises stress: {von_mises(stress_tensor)/1e6:.2f} MPa")
    print(f"Tresca stress: {tresca(stress_tensor)/1e6:.2f} MPa")

    # ---- Vibration Example ----
    m1, m2 = 10.0, 10.0          # kg
    k1, k2 = 2e5, 2e5            # N/m
    M = np.diag([m1, m2])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    omega, phi = modal_analysis(M, K)
    print(f"\nNatural frequencies (rad/s): {omega}")
    print("Mode shapes (columns):\n", phi)
    # SDOF forced response
    t = np.linspace(0, 5, 500)
    X = sdofforced_response(m=m1, k=k1, c=20.0, F0=500.0,
                            omega=omega[0], t=t)
    plt.figure()
    plt.plot(t, X)
    plt.title("SDOF Steady‑State Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.tight_layout()
    plt.show()

    # ---- Fatigue Example ----
    N = 1e6
    sigma_f_prime = 900e6
    b = -0.12
    sigma_a = basquin(N, sigma_f_prime, b)
    print(f"\nBasquin: sigma_a for N={N:.0f} cycles = {sigma_a/1e6:.2f} MPa")
    print("Goodman check:", goodman(sigma_a, 100e6, 200e6, 600e6))
    print("Gerber check:", gerber(sigma_a, 100e6, 200e6, 600e6))
    # Paris law
    da_dN = paris_law(delta_K=30e6, C=1e-12, m=3.0)
    print(f"Paris law da/dN = {da_dN:.2e} m/cycle")

    # ---- Structural Optimization Example ----
    opt_res = optimize_beam_rectangular(L=2.0, w_load=5e3,
                                        sigma_allow=250e6,
                                        E=210e9, rho=7850)
    print("\nOptimization result:")
    for k, v in opt_res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ---- Rotordynamics Example ----
    m_rot = 5.0
    k_rot = 1e6
    c_rot = 500.0
    e_unb = 0.001  # m
    Omega = np.linspace(0, 500, 300)
    X_unb = jeffcott_unbalance_response(m_rot, k_rot, c_rot, e_unb, Omega)
    plt.figure()
    plt.plot(Omega, X_unb*1e3)
    plt.title("Jeffcott Rotor Unbalance Response")
    plt.xlabel("Speed [rad/s]")
    plt.ylabel("Displacement amplitude [mm]")
    plt.tight_layout()
    plt.show()

    # ---- Contact Mechanics Example ----
    F_cont = 1000.0          # N
    R1 = R2 = 0.05           # m
    E1 = E2 = 210e9
    nu1 = nu2 = 0.3
    a, p0 = hertzian_sphere_contact(F_cont, R1, R2, E1, nu1, E2, nu2)
    print(f"\nHertzian contact: radius a = {a*1e3:.3f} mm, max pressure p0 = {p0/1e6:.2f} MPa")

    # ---- Failure Theory Example ----
    print("\nFailure checks:")
    print("Max normal stress:", max_normal_stress(s1, s2, 300e6))
    print("Max shear stress:", max_shear_stress(tauxy, sigx, sigy, 150e6))
    print("Mohr‑Coulomb:", mohr_coulomb(sig_n=50e6, tau=30e6, c=5e6, phi=30))


# End of script