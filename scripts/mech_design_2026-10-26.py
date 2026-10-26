import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
from abc import ABC, abstractmethod

# ==============================================================================
# 1. FINITE ELEMENT ANALYSIS
# ==============================================================================

@dataclass
class Material:
    E: float = 200e9      # Young's modulus (Pa)
    nu: float = 0.3       # Poisson's ratio
    rho: float = 7850     # Density (kg/m^3)
    sigma_y: float = 250e6 # Yield strength (Pa)

@dataclass
class Section:
    A: float = 1e-4       # Cross-sectional area (m^2)
    I: float = 8.33e-10   # Second moment of area (m^4)
    J: float = 1.67e-10   # Torsional constant (m^4)

class FEModel:
    def __init__(self, ndof_per_node: int):
        self.ndof_per_node = ndof_per_node
        self.nodes = {}           # node_id -> (x, y, z)
        self.elements = []        # list of element objects
        self.boundary_conditions = {}  # dof_index -> prescribed_value
        self.forces = {}          # dof_index -> force_value

    def add_node(self, nid: int, coords: Tuple[float, ...]):
        self.nodes[nid] = np.array(coords, dtype=float)

    def add_element(self, element):
        self.elements.append(element)

    def apply_bc(self, dof: int, value: float = 0.0):
        self.boundary_conditions[dof] = value

    def apply_force(self, dof: int, value: float):
        self.forces[dof] = value

    def assemble_global_stiffness(self) -> np.ndarray:
        n_nodes = len(self.nodes)
        K = np.zeros((n_nodes * self.ndof_per_node, n_nodes * self.ndof_per_node))
        for elem in self.elements:
            k_local = elem.stiffness_matrix()
            dofs = elem.dof_indices()
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += k_local[i, j]
        return K

    def assemble_global_mass(self) -> np.ndarray:
        n_nodes = len(self.nodes)
        M = np.zeros((n_nodes * self.ndof_per_node, n_nodes * self.ndof_per_node))
        for elem in self.elements:
            m_local = elem.mass_matrix()
            dofs = elem.dof_indices()
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    M[gi, gj] += m_local[i, j]
        return M

    def solve_static(self) -> np.ndarray:
        K = self.assemble_global_stiffness()
        F = np.zeros(K.shape[0])
        for dof, val in self.forces.items():
            F[dof] = val
        # Apply boundary conditions (penalty method)
        penalty = 1e12
        for dof, val in self.boundary_conditions.items():
            K[dof, dof] += penalty
            F[dof] += penalty * val
        U = la.solve(K, F, assume_a='sym')
        return U

    def modal_analysis(self, n_modes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        K = self.assemble_global_stiffness()
        M = self.assemble_global_mass()
        # Apply boundary conditions to K and M (remove constrained DOFs)
        free_dofs = [i for i in range(K.shape[0]) if i not in self.boundary_conditions]
        K_free = K[np.ix_(free_dofs, free_dofs)]
        M_free = M[np.ix_(free_dofs, free_dofs)]
        eigvals, eigvecs = la.eigh(K_free, M_free, subset_by_index=(0, n_modes-1))
        freqs = np.sqrt(eigvals) / (2*np.pi)
        # Expand eigenvectors to full size
        phi_full = np.zeros((K.shape[0], n_modes))
        phi_full[free_dofs, :] = eigvecs
        return freqs, phi_full

class TrussElement1D:
    def __init__(self, eid: int, n1: int, n2: int, material: Material, section: Section, model: FEModel):
        self.eid = eid
        self.n1 = n1
        self.n2 = n2
        self.material = material
        self.section = section
        self.model = model

    def length(self) -> float:
        x1 = self.model.nodes[self.n1]
        x2 = self.model.nodes[self.n2]
        return np.linalg.norm(x2 - x1)

    def direction_cosines(self) -> np.ndarray:
        x1 = self.model.nodes[self.n1]
        x2 = self.model.nodes[self.n2]
        vec = x2 - x1
        L = np.linalg.norm(vec)
        return vec / L

    def stiffness_matrix(self) -> np.ndarray:
        L = self.length()
        c = self.direction_cosines()
        k = (self.material.E * self.section.A / L) * np.array([
            [1, -1],
            [-1, 1]
        ])
        # Transform to global coordinates (2D/3D)
        T = np.zeros((2, self.model.ndof_per_node * 2))
        for i in range(self.model.ndof_per_node):
            T[0, i] = c[i]
            T[1, self.model.ndof_per_node + i] = c[i]
        return T.T @ k @ T

    def mass_matrix(self) -> np.ndarray:
        L = self.length()
        m = self.material.rho * self.section.A * L / 2
        M = m * np.eye(2)
        # Transform to global
        c = self.direction_cosines()
        T = np.zeros((2, self.model.ndof_per_node * 2))
        for i in range(self.model.ndof_per_node):
            T[0, i] = c[i]
            T[1, self.model.ndof_per_node + i] = c[i]
        return T.T @ M @ T

    def dof_indices(self) -> List[int]:
        indices = []
        for n in [self.n1, self.n2]:
            base = (n - 1) * self.model.ndof_per_node
            indices.extend(range(base, base + self.model.ndof_per_node))
        return indices

class BeamElement2D:
    """Euler-Bernoulli beam element in 2D (3 DOF per node: ux, uy, rz)"""
    def __init__(self, eid: int, n1: int, n2: int, material: Material, section: Section, model: FEModel):
        self.eid = eid
        self.n1 = n1
        self.n2 = n2
        self.material = material
        self.section = section
        self.model = model

    def length(self) -> float:
        x1 = self.model.nodes[self.n1]
        x2 = self.model.nodes[self.n2]
        return np.linalg.norm(x2 - x1)

    def stiffness_matrix(self) -> np.ndarray:
        L = self.length()
        E = self.material.E
        A = self.section.A
        I = self.section.I
        # Local stiffness matrix (6x6)
        k_local = np.zeros((6, 6))
        # Axial
        k_local[0,0] = k_local[3,3] = E*A/L
        k_local[0,3] = k_local[3,0] = -E*A/L
        # Bending
        k_local[1,1] = 12*E*I/L**3
        k_local[1,2] = 6*E*I/L**2
        k_local[1,4] = -12*E*I/L**3
        k_local[1,5] = 6*E*I/L**2
        k_local[2,1] = 6*E*I/L**2
        k_local[2,2] = 4*E*I/L
        k_local[2,4] = -6*E*I/L**2
        k_local[2,5] = 2*E*I/L
        k_local[4,1] = -12*E*I/L**3
        k_local[4,2] = -6*E*I/L**2
        k_local[4,4] = 12*E*I/L**3
        k_local[4,5] = -6*E*I/L**2
        k_local[5,1] = 6*E*I/L**2
        k_local[5,2] = 2*E*I/L
        k_local[5,4] = -6*E*I/L**2
        k_local[5,5] = 4*E*I/L
        # Symmetrize
        k_local = (k_local + k_local.T) / 2
        # Transformation to global (if needed, assuming aligned with x-axis for simplicity)
        return k_local

    def mass_matrix(self) -> np.ndarray:
        L = self.length()
        rho = self.material.rho
        A = self.section.A
        # Consistent mass matrix (simplified)
        m = rho * A * L / 420 * np.array([
            [140, 0, 0, 70, 0, 0],
            [0, 156, 22*L, 0, 54, -13*L],
            [0, 22*L, 4*L**2, 0, 13*L, -3*L**2],
            [70, 0, 0, 140, 0, 0],
            [0, 54, 13*L, 0, 156, -22*L],
            [0, -13*L, -3*L**2, 0, -22*L, 4*L**2]
        ])
        return m

    def dof_indices(self) -> List[int]:
        indices = []
        for n in [self.n1, self.n2]:
            base = (n - 1) * self.model.ndof_per_node
            indices.extend(range(base, base + self.model.ndof_per_node))
        return indices

# ==============================================================================
# 2. STRESS & STRAIN TRANSFORMATIONS
# ==============================================================================

def stress_transform(sigma_x: float, sigma_y: float, tau_xy: float, theta: float) -> Tuple[float, float, float]:
    """Transform stress components by angle theta (radians)."""
    c = np.cos(2*theta)
    s = np.sin(2*theta)
    sigma_xp = (sigma_x + sigma_y)/2 + (sigma_x - sigma_y)/2 * c + tau_xy * s
    sigma_yp = (sigma_x + sigma_y)/2 - (sigma_x - sigma_y)/2 * c - tau_xy * s
    tau_xyp = -(sigma_x - sigma_y)/2 * s + tau_xy * c
    return sigma_xp, sigma_yp, tau_xyp

def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
    """Return principal stresses (sigma1, sigma2) and angle theta_p (radians)."""
    avg = (sigma_x + sigma_y) / 2
    R = np.sqrt(((sigma_x - sigma_y)/2)**2 + tau_xy**2)
    sigma1 = avg + R
    sigma2 = avg - R
    theta_p = 0.5 * np.arctan2(2*tau_xy, sigma_x - sigma_y)
    return sigma1, sigma2, theta_p

def von_mises_stress(sigma_x: float, sigma_y: float, tau_xy: float, sigma_z: float = 0.0) -> float:
    """Von Mises equivalent stress for plane stress or 3D."""
    return np.sqrt(0.5 * ((sigma_x - sigma_y)**2 + (sigma_y - sigma_z)**2 + (sigma_z - sigma_x)**2 + 6*tau_xy**2))

def tresca_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Tresca equivalent stress (max shear stress * 2)."""
    sigma1, sigma2, _ = principal_stresses(sigma_x, sigma_y, tau_xy)
    return max(abs(sigma1 - sigma2), abs(sigma1), abs(sigma2))

def mohr_circle_data(sigma_x: float, sigma_y: float, tau_xy: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate points for Mohr's circle plot."""
    avg = (sigma_x + sigma_y) / 2
    R = np.sqrt(((sigma_x - sigma_y)/2)**2 + tau_xy**2)
    theta = np.linspace(0, 2*np.pi, n_points)
    sigma = avg + R * np.cos(theta)
    tau = R * np.sin(theta)
    return sigma, tau

# ==============================================================================
# 3. MECHANICAL VIBRATIONS
# ==============================================================================

class SDOFSystem:
    def __init__(self, m: float, c: float, k: float):
        self.m = m
        self.c = c
        self.k = k
        self.wn = np.sqrt(k/m)
        self.zeta = c / (2*np.sqrt(k*m))
        self.wd = self.wn * np.sqrt(1 - self.zeta**2) if self.zeta < 1 else 0

    def free_response(self, t: np.ndarray, x0: float, v0: float) -> np.ndarray:
        if self.zeta < 1:  # Underdamped
            A = x0
            B = (v0 + self.zeta*self.wn*x0) / self.wd
            return np.exp(-self.zeta*self.wn*t) * (A*np.cos(self.wd*t) + B*np.sin(self.wd*t))
        elif self.zeta == 1:  # Critically damped
            A = x0
            B = v0 + self.wn*x0
            return (A + B*t) * np.exp(-self.wn*t)
        else:  # Overdamped
            s1 = -self.zeta*self.wn + self.wn*np.sqrt(self.zeta**2 - 1)
            s2 = -self.zeta*self.wn - self.wn*np.sqrt(self.zeta**2 - 1)
            A = (v0 - s2*x0) / (s1 - s2)
            B = x0 - A
            return A*np.exp(s1*t) + B*np.exp(s2*t)

    def forced_response_harmonic(self, t: np.ndarray, F0: float, omega: float) -> np.ndarray:
        """Steady-state response to F0*sin(omega*t)."""
        r = omega / self.wn
        denom = (1 - r**2)**2 + (2*self.zeta*r)**2
        X = F0 / self.k / np.sqrt(denom)
        phi = np.arctan2(2*self.zeta*r, 1 - r**2)
        return X * np.sin(omega*t - phi)

    def state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[0, 1], [-self.k/self.m, -self.c/self.m]])
        B = np.array([[0], [1/self.m]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        return A, B, C, D

class MDOFSystem:
    def __init__(self, M: np.ndarray, C: np.ndarray, K: np.ndarray):
        self.M = M
        self.C = C
        self.K = K
        self.n = M.shape[0]

    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        eigvals, eigvecs = la.eigh(self.K, self.M)
        freqs = np.sqrt(eigvals) / (2*np.pi)
        return freqs, eigvecs

    def state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        Z = np.zeros((n, n))
        I = np.eye(n)
        Minv = la.inv(self.M)
        A = np.block([[Z, I], [-Minv @ self.K, -Minv @ self.C]])
        B = np.block([[Z], [Minv]])
        C = np.block([[I, Z]])
        D = np.zeros((n, n))
        return A, B, C, D

    def forced_response_modal(self, t: np.ndarray, F: Callable[[float], np.ndarray], 
                              x0: np.ndarray, v0: np.ndarray, n_modes: Optional[int] = None) -> np.ndarray:
        """Modal superposition for forced response."""
        freqs, phi = self.modal_analysis()
        if n_modes is None:
            n_modes = self.n
        # Modal coordinates
        Phi = phi[:, :n_modes]
        M_modal = Phi.T @ self.M @ Phi
        K_modal = Phi.T @ self.K @ Phi
        C_modal = Phi.T @ self.C @ Phi
        # Initial conditions in modal space
        q0 = Phi.T @ self.M @ x0
        qd0 = Phi.T @ self.M @ v0
        # Time integration for each mode (using Duhamel integral or numerical)
        # Simplified: assume zero damping for modal superposition example
        q = np.zeros((n_modes, len(t)))
        for i in range(n_modes):
            m_i = M_modal[i, i]
            k_i = K_modal[i, i]
            w_i = np.sqrt(k_i / m_i)
            # Modal force
            F_modal = np.array([Phi[:, i] @ F(ti) for ti in t])
            # Solve SDOF for each mode (using numerical integration)
            # Simple Newmark-beta for demonstration
            dt = t[1] - t[0]
            beta = 0.25
            gamma = 0.5
            a = 1/(beta*dt**2)
            b = gamma/(beta*dt)
            c = 1/(beta*dt)
            d = 1/(2*beta) - 1
            k_hat = k_i + a*m_i
            q_i = np.zeros(len(t))
            qd_i = np.zeros(len(t))
            qdd_i = np.zeros(len(t))
            q_i[0] = q0[i]
            qd_i[0] = qd0[i]
            qdd_i[0] = (F_modal[0] - k_i*q_i[0]) / m_i
            for j in range(1, len(t)):
                F_eff = F_modal[j] + m_i*(a*q_i[j-1] + c*qd_i[j-1] + d*qdd_i[j-1])
                q_i[j] = F_eff / k_hat
                qdd_i[j] = a*(q_i[j] - q_i[j-1]) - c*qd_i[j-1] - d*qdd_i[j-1]
                qd_i[j] = qd_i[j-1] + dt*((1-gamma)*qdd_i[j-1] + gamma*qdd_i[j])
            q[i, :] = q_i
        # Transform back to physical coordinates
        x = Phi @ q
        return x

# ==============================================================================
# 4. FATIGUE & FRACTURE MECHANICS
# ==============================================================================

def sn_curve_stress_life(N: float, sigma_f_prime: float, b: float) -> float:
    """Basquin's equation: sigma_a = sigma_f' * (2N)^b"""
    return sigma_f_prime * (2*N)**b

def sn_curve_strain_life(N: float, E: float, sigma_f_prime: float, b: float, epsilon_f_prime: float, c: float) -> float:
    """Manson-Coffin: epsilon_a = sigma_f'/E*(2N)^b + epsilon_f'*(2N)^c"""
    return sigma_f_prime/E * (2*N)**b + epsilon_f_prime * (2*N)**c

def goodman_criterion(sigma_a: float, sigma_m: float, sigma_ut: float, sigma_e: float) -> float:
    """Goodman line: sigma_a/sigma_e + sigma_m/sigma_ut = 1. Returns factor of safety."""
    return 1 / (sigma_a/sigma_e + sigma_m/sigma_ut)

def gerber_criterion(sigma_a: float, sigma_m: float, sigma_ut: float, sigma_e: float) -> float:
    """Gerber parabola: sigma_a/sigma_e + (sigma_m/sigma_ut)^2 = 1."""
    return 1 / (sigma_a/sigma_e + (sigma_m/sigma_ut)**2)

def paris_law(dN: float, C: float, m: float, delta_K: float) -> float:
    """da/dN = C * (delta_K)^m. Returns crack growth per cycle."""
    return C * delta_K**m

def stress_intensity_factor(a: float, sigma: float, geometry: str = 'center_crack', 
                            width: float = None, height: float = None) -> float:
    """Stress intensity factor K for common geometries."""
    if geometry == 'center_crack':
        # Infinite plate with center crack length 2a
        return sigma * np.sqrt(np.pi * a)
    elif geometry == 'edge_crack':
        # Edge crack in semi-infinite plate
        return 1.12 * sigma * np.sqrt(np.pi * a)
    elif geometry == 'finite_width_center':
        # Center crack in finite width plate
        if width is None:
            raise ValueError("Width required for finite width geometry")
        return sigma * np.sqrt(np.pi * a) * np.sqrt(width/(np.pi*a) * np.tan(np.pi*a/width))
    elif geometry == 'finite_width_edge':
        if width is None:
            raise ValueError("Width required")
        return 1.12 * sigma * np.sqrt(np.pi * a) * (1 - 0.1*a/width)  # Approximation
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

# ==============================================================================
# 5. STRUCTURAL OPTIMIZATION
# ==============================================================================

def beam_weight_minimization(L: float, F: float, sigma_allow: float, E: float, rho: float,
                             bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.01, 0.5), (0.01, 0.5))) -> dict:
    """
    Minimize weight of a rectangular cantilever beam (width b, height h) with tip load F.
    Constraints: max bending stress <= sigma_allow, tip deflection <= L/200.
    """
    def objective(x):
        b, h = x
        return rho * b * h * L

    def stress_constraint(x):
        b, h = x
        M_max = F * L
        I = b * h**3 / 12
        sigma_max = M_max * (h/2) / I
        return sigma_allow - sigma_max

    def deflection_constraint(x):
        b, h = x
        I = b * h**3 / 12
        delta = F * L**3 / (3 * E * I)
        return L/200 - delta

    constraints = [
        {'type': 'ineq', 'fun': stress_constraint},
        {'type': 'ineq', 'fun': deflection_constraint}
    ]

    x0 = [(bounds[0][0]+bounds[0][1])/2, (bounds[1][0]+bounds[1][1])/2]
    res = opt.minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    return {'success': res.success, 'opt_dimensions': res.x, 'min_weight': res.fun, 'message': res.message}

# ==============================================================================
# 6. ROTORDYNAMICS / CONTACT MECHANICS / FAILURE THEORIES
# ==============================================================================

def jeffcott_rotor_critical_speed(m: float, k: float, e: float, omega: float) -> float:
    """Steady-state amplitude of Jeffcott rotor at speed omega."""
    wn = np.sqrt(k/m)
    r = omega / wn
    return e * r**2 / abs(1 - r**2)

def hertzian_contact_sphere(F: float, R1: float, R2: float, E1: float, nu1: float, E2: float, nu2: float) -> dict:
    """Hertzian contact between two spheres."""
    E_star = 1 / ((1-nu1**2)/E1 + (1-nu2**2)/E2)
    R_star = 1 / (1/R1 + 1/R2)
    a = (3*F*R_star / (4*E_star))**(1/3)
    p0 = 3*F / (2*np.pi*a**2)
    delta = a**2 / R_star
    return {'contact_radius': a, 'max_pressure': p0, 'approach': delta}

def max_shear_stress_theory(sigma1: float, sigma2: float, sigma3: float, sigma_y: float) -> float:
    """Tresca factor of safety."""
    tau_max = max(abs(sigma1-sigma2), abs(sigma2-sigma3), abs(sigma3-sigma1)) / 2
    tau_yield = sigma_y / 2
    return tau_yield / tau_max

def von_mises_theory(sigma1: float, sigma2: float, sigma3: float, sigma_y: float) -> float:
    """Von Mises factor of safety."""
    sigma_vm = np.sqrt(0.5*((sigma1-sigma2)**2 + (sigma2-sigma3)**2 + (sigma3-sigma1)**2))
    return sigma_y / sigma_vm

# ==============================================================================
# MAIN DEMONSTRATION
# ==============================================================================

if __name__ == '__main__':
    print("="*60)
    print("MECHANICAL ENGINEERING COMPUTATIONAL TOOLKIT DEMONSTRATION")
    print("="*60)

    # 1. FEA Example: Simple 2D Truss
    print("\n--- 1. FEA: 2D Truss Analysis ---")
    mat = Material(E=200e9, A=1e-4)
    sec = Section(A=1e-4)
    model = FEModel(ndof_per_node=2)
    model.add_node(1, (0, 0))
    model.add_node(2, (1, 0))
    model.add_node(3, (0.5, 1))
    model.add_element(TrussElement1D(1, 1, 3, mat, sec, model))
    model.add_element(TrussElement1D(2, 2, 3, mat, sec, model))
    model.add_element(TrussElement1D(3, 1, 2, mat, sec, model))
    model.apply_bc(0, 0); model.apply_bc(1, 0)  # Node 1 fixed
    model.apply_bc(2, 0); model.apply_bc(3, 0)  # Node 2 fixed
    model.apply_force(5, -1000)  # Vertical force at node 3 (dof 5 = uy of node 3)
    U = model.solve_static()
    print(f"Displacement at node 3: ux={U[4]:.2e} m, uy={U[5]:.2e} m")

    # 2. Stress Transformation
    print("\n--- 2. Stress Transformation & Failure Theories ---")
    sx, sy, txy = 100e6, 50e6, 30e6
    s1, s2, theta = principal_stresses(sx, sy, txy)
    print(f"Principal stresses: σ1={s1/1e6:.1f} MPa, σ2={s2/1e6:.1f} MPa, θ={np.degrees(theta):.1f}°")
    print(f"Von Mises stress: {von_mises_stress(sx, sy, txy)/1e6:.1f} MPa")
    print(f"Tresca stress: {tresca_stress(sx, sy, txy)/1e6:.1f} MPa")
    fs_tresca = max_shear_stress_theory(s1, s2, 0, 250e6)
    fs_vm = von_mises_theory(s1, s2, 0, 250e6)
    print(f"Factor of Safety (Tresca): {fs_tresca:.2f}")
    print(f"Factor of Safety (Von Mises): {fs_vm:.2f}")

    # 3. Vibrations: SDOF
    print("\n--- 3. Vibrations: SDOF System ---")
    sdof = SDOFSystem(m=10, c=10, k=1e4)
    print(f"Natural freq: {sdof.wn:.2f} rad/s ({sdof.wn/(2*np.pi):.2f} Hz)")
    print(f"Damping ratio: {sdof.zeta:.3f}")
    t = np.linspace(0, 10, 1000)
    x_free = sdof.free_response(t, x0=0.01, v0=0)
    x_forced = sdof.forced_response_harmonic(t, F0=100, omega=sdof.wn*0.8)
    print(f"Free response max amplitude: {np.max(np.abs(x_free)):.4f} m")
    print(f"Forced response amplitude (ω=0.8ωn): {np.max(np.abs(x_forced)):.4f} m")

    # 4. Fatigue
    print("\n--- 4. Fatigue & Fracture Mechanics ---")
    N = 1e6
    sigma_a = sn_curve_stress_life(N, 1000e6, -0.09)
    print(f"Basquin stress amplitude at {N:.0e} cycles: {sigma_a/1e6:.1f} MPa")
    fs_goodman = goodman_criterion(200e6, 50e6, 600e6, 300e6)
    fs_gerber = gerber_criterion(200e6, 50e6, 600e6, 300e6)
    print(f"Goodman FoS: {fs_goodman:.2f}")
    print(f"Gerber FoS: {fs_gerber:.2f}")
    da_dN = paris_law(1, 1e-12, 3, 20e6)
    print(f"Paris law crack growth (ΔK=20 MPa√m): {da_dN*1e6:.2f} μm/cycle")
    K = stress_intensity_factor(0.005, 200e6, 'center_crack')
    print(f"Stress intensity factor (a=5mm, σ=200MPa): {K/1e6:.2f} MPa√m")

    # 5. Optimization
    print("\n--- 5. Structural Optimization: Cantilever Beam ---")
    opt_result = beam_weight_minimization(L=1, F=1000, sigma_allow=250e6, E=200e9, rho=7850)
    if opt_result['success']:
        b_opt, h_opt = opt_result['opt_dimensions']
        print(f"Optimal dimensions: b={b_opt*1000:.1f} mm, h={h_opt*1000:.1f} mm")
        print(f"Minimum weight: {opt_result['min_weight']:.3f} kg")
    else:
        print(f"Optimization failed: {opt_result['message']}")

    # 6. Rotordynamics & Contact
    print("\n--- 6. Rotordynamics & Contact Mechanics ---")
    amp = jeffcott_rotor_critical_speed(m=5, k=1e6, e=1e-5, omega=400)
    print(f"Jeffcott rotor amplitude at 400 rad/s: {amp*1e6:.2f} μm")
    contact = hertzian_contact_sphere(F=1000, R1=0.02, R2=0.02, E1=200e9, nu1=0.3, E2=200e9, nu2=0.3)
    print(f"Hertz contact: radius={contact['contact_radius']*1e6:.1f} μm, p0={contact['max_pressure']/1e6:.1f} MPa, approach={contact['approach']*1e9:.1f} nm")

    # Plot Mohr's circle
    sigma_circle, tau_circle = mohr_circle_data(sx, sy, txy)
    plt.figure(figsize=(6,6))
    plt.plot(sigma_circle/1e6, tau_circle/1e6, 'b-', label="Mohr's Circle")
    plt.plot([s1/1e6, s2/1e6], [0, 0], 'ro', label='Principal Stresses')
    plt.plot([sx/1e6, sy/1e6], [txy/1e6, -txy/1e6], 'go', label='Given Stress State')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlabel('Normal Stress (MPa)')
    plt.ylabel('Shear Stress (MPa)')
    plt.title("Mohr's Circle for Stress Transformation")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('mohrs_circle.png', dpi=150)
    print("\nMohr's circle plot saved as 'mohrs_circle.png'")

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)