import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Callable, Optional

# ----------------------------------------------------------------------
# Physical constants and default material properties
# ----------------------------------------------------------------------
GRAVITY = 9.81                       # m/s²
STEEL_E = 200e9                      # Pa (Young's modulus)
STEEL_NU = 0.3                       # Poisson's ratio
STEEL_RHO = 7850                     # kg/m³ (density)
STEEL_YIELD = 250e6                  # Pa (yield strength)
STEEL_ULS = 400e6                    # Pa (ultimate tensile strength)
ALU_E = 70e9
ALU_NU = 0.33
ALU_RHO = 2700
ALU_YIELD = 200e6

# ----------------------------------------------------------------------
# Material class
# ----------------------------------------------------------------------
class Material:
    """Structural material with isotropic linear elastic properties."""
    def __init__(self, name: str, E: float, nu: float, rho: float,
                 yield_strength: Optional[float] = None,
                 ultimate_strength: Optional[float] = None):
        self.name = name
        self.E = E
        self.nu = nu
        self.rho = rho
        self.yield_strength = yield_strength
        self.ultimate_strength = ultimate_strength

    def shear_modulus(self) -> float:
        """Return shear modulus G = E/(2(1+nu))."""
        return self.E / (2 * (1 + self.nu))

    def bulk_modulus(self) -> float:
        """Return bulk modulus K = E/(3(1-2nu))."""
        return self.E / (3 * (1 - 2 * self.nu))

# ----------------------------------------------------------------------
# Stress & Strain transformations
# ----------------------------------------------------------------------
def stress_tensor(sigma_x: float, sigma_y: float, tau_xy: float) -> np.ndarray:
    """Return 2D stress tensor."""
    return np.array([[sigma_x, tau_xy],
                     [tau_xy, sigma_y]])

def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float]:
    """Compute principal stresses σ₁, σ₂ (σ₁ ≥ σ₂)."""
    avg = (sigma_x + sigma_y) / 2.0
    R = math.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
    sigma1 = avg + R
    sigma2 = avg - R
    return sigma1, sigma2

def mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
    """Return center, radius, and principal stresses of Mohr's circle."""
    center = (sigma_x + sigma_y) / 2.0
    radius = math.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
    sigma1 = center + radius
    sigma2 = center - radius
    return center, radius, (sigma1, sigma2)

def von_mises(sigma_x: float, sigma_y: float, sigma_z: float,
              tau_xy: float, tau_yz: float, tau_zx: float) -> float:
    """Compute von Mises equivalent stress for 3D stress state."""
    return math.sqrt(0.5 * ((sigma_x - sigma_y)**2 + (sigma_y - sigma_z)**2 +
                             (sigma_z - sigma_x)**2 +
                             6 * (tau_xy**2 + tau_yz**2 + tau_zx**2)))

def tresca(sigma_x: float, sigma_y: float, sigma_z: float) -> float:
    """Compute Tresca equivalent stress (max shear stress * 2)."""
    s = sorted([sigma_x, sigma_y, sigma_z])
    return s[2] - s[0]

def stress_rotation_matrix(theta: float) -> np.ndarray:
    """Return rotation matrix for stress transformation (2D)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c**2, s**2, 2*c*s],
                     [s**2, c**2, -2*c*s],
                     [-c*s, c*s, c**2 - s**2]])

def rotate_stress_2d(sigma_x: float, sigma_y: float, tau_xy: float,
                     theta: float) -> Tuple[float, float, float]:
    """Rotate 2D stress components by angle θ (counter-clockwise)."""
    R = stress_rotation_matrix(theta)
    vec = np.array([sigma_x, sigma_y, tau_xy])
    rotated = R @ vec
    return rotated[0], rotated[1], rotated[2]

# ----------------------------------------------------------------------
# Finite Element Analysis – 1D/2D elements
# ----------------------------------------------------------------------
class Truss2D:
    """2D truss element with two nodes."""
    def __init__(self, node_i: int, node_j: int, A: float, material: Material):
        self.node_i = node_i
        self.node_j = node_j
        self.A = A
        self.E = material.E
        self.rho = material.rho

    def stiffness_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute 4×4 element stiffness matrix in global coordinates.
        coords: (2,2) array of node coordinates.
        """
        x1, y1 = coords[self.node_i]
        x2, y2 = coords[self.node_j]
        L = math.hypot(x2 - x1, y2 - y1)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        k_local = (self.A * self.E / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        return k_local

    def mass_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Consistent mass matrix (lumped mass approximation)."""
        x1, y1 = coords[self.node_i]
        x2, y2 = coords[self.node_j]
        L = math.hypot(x2 - x1, y2 - y1)
        m = self.rho * self.A * L
        # lumped mass matrix (diagonal)
        M = np.zeros((4,4))
        M[0,0] = m/2
        M[1,1] = m/2
        M[2,2] = m/2
        M[3,3] = m/2
        return M

class Beam2D:
    """Euler–Bernoulli beam element with two nodes."""
    def __init__(self, node_i: int, node_j: int, I: float, A: float, material: Material):
        self.node_i = node_i
        self.node_j = node_j
        self.I = I          # second moment of area
        self.A = A
        self.E = material.E
        self.rho = material.rho

    def stiffness_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        6×6 element stiffness matrix (axial + bending) in global coordinates.
        coords: (2,2) array.
        """
        x1, y1 = coords[self.node_i]
        x2, y2 = coords[self.node_j]
        L = math.hypot(x2 - x1, y2 - y1)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        # axial stiffness
        k_axial = self.A * self.E / L
        # bending stiffness
        k_bend = self.E * self.I / L**3
        # transformation matrix
        T = np.zeros((6,6))
        T[0,0] = c; T[0,1] = s
        T[1,0] = -s; T[1,1] = c
        T[2,2] = 1
        T[3,3] = c; T[3,4] = s
        T[4,3] = -s; T[4,4] = c
        T[5,5] = 1
        # local stiffness (axial + bending)
        k_local = np.zeros((6,6))
        # axial contributions
        k_local[0,0] = k_axial
        k_local[0,3] = -k_axial
        k_local[3,0] = -k_axial
        k_local[3,3] = k_axial
        # bending contributions
        k_local[1,1] = 12*k_bend
        k_local[1,4] = -12*k_bend
        k_local[4,1] = -12*k_bend
        k_local[4,4] = 12*k_bend
        k_local[2,2] = 4*k_bend*L
        k_local[2,5] = 2*k_bend*L
        k_local[5,2] = 2*k_bend*L
        k_local[5,5] = 4*k_bend*L
        k_local[1,2] = 6*k_bend*L
        k_local[1,5] = 6*k_bend*L
        k_local[2,1] = 6*k_bend*L
        k_local[2,4] = -6*k_bend*L
        k_local[4,2] = -6*k_bend*L
        k_local[4,5] = -6*k_bend*L
        k_local[5,1] = 6*k_bend*L
        k_local[5,4] = -6*k_bend*L
        # transform to global
        k_global = T.T @ k_local @ T
        return k_global

    def mass_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Lumped mass matrix."""
        x1, y1 = coords[self.node_i]
        x2, y2 = coords[self.node_j]
        L = math.hypot(x2 - x1, y2 - y1)
        m = self.rho * self.A * L
        M = np.zeros((6,6))
        # lumped translational masses
        M[0,0] = m/2
        M[1,1] = m/2
        M[3,3] = m/2
        M[4,4] = m/2
        # rotational inertia (approx)
        I_rho = self.rho * self.I * L
        M[2,2] = I_rho/2
        M[5,5] = I_rho/2
        return M

class Plate4Node:
    """4-node isoparametric plate bending element (Kirchhoff)."""
    def __init__(self, nodes: List[int], thickness: float, material: Material):
        self.nodes = nodes
        self.h = thickness
        self.E = material.E
        self.nu = material.nu
        self.rho = material.rho

    def bending_rigidity(self) -> float:
        return self.E * self.h**3 / (12 * (1 - self.nu**2))

    def stiffness_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute 12×12 stiffness matrix using numerical integration.
        coords: (4,2) array of node coordinates.
        """
        D = self.bending_rigidity()
        # shape functions and derivatives in natural coordinates
        def shape_functions(xi, eta):
            N = np.array([(1+xi)*(1+eta)/4,
                          (1-xi)*(1+eta)/4,
                          (1-xi)*(1-eta)/4,
                          (1+xi)*(1-eta)/4])
            return N
        def dshape_dxi(xi, eta):
            dN_dxi = np.array([(1+eta)/4, -(1+eta)/4, -(1-eta)/4, (1-eta)/4])
            return dN_dxi
        def dshape_deta(xi, eta):
            dN_deta = np.array([(1+xi)/4, (1-xi)/4, -(1-xi)/4, -(1+xi)/4])
            return dN_deta

        # Gauss quadrature
        gauss_pts = [-1/math.sqrt(3), 1/math.sqrt(3)]
        gauss_w = [1, 1]
        K = np.zeros((12,12))
        for xi in gauss_pts:
            for eta in gauss_pts:
                # Jacobian
                dN_dxi = dshape_dxi(xi, eta)
                dN_deta = dshape_deta(xi, eta)
                J = np.zeros((2,2))
                for i in range(4):
                    J[0,0] += dN_dxi[i] * coords[i,0]
                    J[0,1] += dN_dxi[i] * coords[i,1]
                    J[1,0] += dN_deta[i] * coords[i,0]
                    J[1,1] += dN_deta[i] * coords[i,1]
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                # derivatives in global coordinates
                dN_dx = np.zeros(4)
                dN_dy = np.zeros(4)
                for i in range(4):
                    dN_dx[i] = invJ[0,0]*dN_dxi[i] + invJ[0,1]*dN_deta[i]
                    dN_dy[i] = invJ[1,0]*dN_dxi[i] + invJ[1,1]*dN_deta[i]
                # B matrix (3x12)
                B = np.zeros((3,12))
                for i in range(4):
                    col = 3*i
                    B[0, col] = dN_dx[i]
                    B[1, col] = dN_dy[i]
                    B[2, col] = dN_dx[i]*dN_dy[i]  # simplified curvature-displacement
                # constitutive matrix D
                Dmat = D * np.array([[1, self.nu, 0],
                                     [self.nu, 1, 0],
                                     [0, 0, (1-self.nu)/2]])
                K += (B.T @ Dmat @ B) * detJ * gauss_w[0] * gauss_w[1] * self.h**3 / 12
        return K

# ----------------------------------------------------------------------
# Mechanical Vibrations
# ----------------------------------------------------------------------
def modal_analysis(M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenvalue problem Kφ = λMφ.
    Returns eigenvalues (natural frequencies squared) and eigenvectors.
    """
    eigvals, eigvecs = la.eig(K, M)
    # sort by eigenvalue
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def sdof_damped_response(m: float, c: float, k: float, F: Callable[[float], float],
                         x0: float, v0: float, t_span: Tuple[float, float],
                         dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve SDOF damped forced vibration using state-space integration.
    Returns time array and displacement array.
    """
    def state_deriv(t, y):
        x, v = y
        dxdt = v
        dvdt = (F(t) - c*v - k*x) / m
        return [dxdt, dvdt]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = integrate.solve_ivp(state_deriv, t_span, [x0, v0],
                              t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0]

def mdo_forced_response(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                        F: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute MDOF forced response using modal superposition.
    M, C, K: n×n matrices
    F: n×nt force matrix
    t: time vector
    Returns displacement matrix (n×nt).
    """
    # modal decomposition
    eigvals, eigvecs = modal_analysis(M, K)
    omega_n = np.sqrt(eigvals)
    # modal damping (proportional)
    # assume C = αM + βK
    # compute α, β from first two modes
    if len(omega_n) > 1:
        # solve for α, β using first two frequencies
        A_mat = np.array([[1, omega_n[0]**2],
                          [1, omega_n[1]**2]])
        # assume modal damping ratios ζ = 0.01
        zeta = np.array([0.01, 0.01])
        b_vec = 2 * zeta * omega_n[:2]
        alpha_beta = la.solve(A_mat, b_vec)
        alpha, beta = alpha_beta
        C_mod = alpha * M + beta * K
    else:
        C_mod = C
    # transform to modal coordinates
    Phi = eigvecs
    n = M.shape[0]
    q = np.zeros((n, len(t)))
    # decoupled SDOF equations
    for i in range(n):
        mi = Phi[:, i].T @ M @ Phi[:, i]
        ci = Phi[:, i].T @ C_mod @ Phi[:, i]
        ki = Phi[:, i].T @ K @ Phi[:, i]
        fi = Phi[:, i].T @ F
        # solve ODE for each mode
        def force_interp(tt):
            return np.interp(tt, t, fi)
        # initial conditions
        q0 = Phi[:, i].T @ M @ np.zeros(n)  # assume zero initial displacement
        v0 = Phi[:, i].T @ M @ np.zeros(n)  # zero initial velocity
        # use state-space integration
        def ode(t, y):
            return [y[1], (force_interp(t) - ci*y[1] - ki*y[0]) / mi]
        sol = integrate.solve_ivp(ode, (t[0], t[-1]), [q0, v0], t_eval=t, method='RK45')
        q[i, :] = sol.y[0]
    # transform back to physical coordinates
    x = Phi @ q
    return x

# ----------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
def s_n_curve(N: np.ndarray, S_u: float, m: float = 9.0, C: float = 1e20) -> np.ndarray:
    """
    Basquin S-N curve: S = C * N^(-1/m)
    Returns stress amplitude for given cycles N.
    """
    return C * N**(-1/m)

def goodman_diagram(sigma_a: np.ndarray, sigma_m: np.ndarray,
                    S_u: float, S_e: float) -> np.ndarray:
    """
    Goodman correction: sigma_a / S_e + sigma_m / S_u = 1
    Returns equivalent alternating stress.
    """
    return sigma_a / (1 - sigma_m / S_u) * S_e

def gerber_diagram(sigma_a: np.ndarray, sigma_m: np.ndarray,
                   S_u: float, S_e: float) -> np.ndarray:
    """
    Gerber correction: (sigma_a / S_e) * (1 - (sigma_m / S_u)^2) = 1
    """
    return sigma_a / (1 - (sigma_m / S_u)**2) * S_e

def paris_law(da_dN: float, C: float, m: float, K_range: float) -> float:
    """
    Paris' law: da/dN = C * (ΔK)^m
    Given da/dN, solve for ΔK.
    """
    return (da_dN / C) ** (1/m)

def fatigue_life(S_a: float, S_u: float, S_e: float, m: float = 9.0,
                 C: float = 1e20) -> float:
    """
    Estimate cycles to failure using S-N curve and Goodman correction.
    """
    # equivalent alternating stress
    S_eq = goodman_diagram(np.array([S_a]), np.array([0]), S_u, S_e)[0]
    # solve S_eq = C * N^(-1/m) => N = (C / S_eq)^m
    N = (C / S_eq) ** m
    return N

# ----------------------------------------------------------------------
# Structural Optimization
# ----------------------------------------------------------------------
def beam_weight_minimization(length: float, load: float, material: Material,
                             bounds: Tuple[float, float, float, float]) -> float:
    """
    Minimize weight of a rectangular cross-section beam under bending.
    bounds: (b_min, b_max, h_min, h_max)
    Returns optimal width b, height h.
    """
    b_min, b_max, h_min, h_max = bounds
    def weight(x):
        b, h = x
        return material.rho * length * b * h
    def constraint_bending(x):
        b, h = x
        # max bending stress = M * y / I, M = load*length/4 (simply supported)
        M = load * length / 4
        I = b * h**3 / 12
        sigma_max = M * (h/2) / I
        return material.yield_strength - sigma_max
    cons = {'type': 'ineq', 'fun': constraint_bending}
    x0 = np.array([(b_min+b_max)/2, (h_min+h_max)/2])
    bounds_opt = [(b_min, b_max), (h_min, h_max)]
    res = opt.minimize(weight, x0, bounds=bounds_opt, constraints=cons)
    return res.x

def genetic_algorithm_population(pop_size: int, bounds: List[Tuple[float, float]],
                                 fitness_func: Callable) -> np.ndarray:
    """Generate random population within bounds."""
    pop = np.zeros((pop_size, len(bounds)))
    for i, (lb, ub) in enumerate(bounds):
        pop[:, i] = np.random.uniform(lb, ub, pop_size)
    return pop

def genetic_algorithm_selection(pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """Tournament selection."""
    selected = np.zeros_like(pop)
    for i in range(len(pop)):
        idx = np.random.choice(len(pop), 2, replace=False)
        if fitness[idx[0]] > fitness[idx[1]]:
            selected[i] = pop[idx[0]]
        else:
            selected[i] = pop[idx[1]]
    return selected

def genetic_algorithm_crossover(parent1: np.ndarray, parent2: np.ndarray,
                                 crossover_rate: float = 0.8) -> np.ndarray:
    """Single-point crossover."""
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
    else:
        child = parent1.copy()
    return child

def genetic_algorithm_mutation(child: np.ndarray, bounds: List[Tuple[float, float]],
                                mutation_rate: float = 0.1) -> np.ndarray:
    """Gaussian mutation."""
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            lb, ub = bounds[i]
            child[i] += np.random.normal(0, (ub-lb)*0.1)
            child[i] = np.clip(child[i], lb, ub)
    return child

def genetic_algorithm_optimize(fitness_func: Callable, bounds: List[Tuple[float, float]],
                                pop_size: int = 50, generations: int = 100,
                                crossover_rate: float = 0.8, mutation_rate: float = 0.1) -> Tuple[np.ndarray, float]:
    """Simple genetic algorithm for minimization."""
    pop = genetic_algorithm_population(pop_size, bounds, fitness_func)
    best_fitness = np.inf
    best_individual = None
    for gen in range(generations):
        fitness = np.array([fitness_func(ind) for ind in pop])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_individual = pop[min_idx].copy()
        # selection
        selected = genetic_algorithm_selection(pop, -fitness)  # maximize -fitness
        # create next generation
        next_pop = np.zeros_like(pop)
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < pop_size else selected[0]
            child1 = genetic_algorithm_crossover(parent1, parent2, crossover_rate)
            child2 = genetic_algorithm_crossover(parent2, parent1, crossover_rate)
            child1 = genetic_algorithm_mutation(child1, bounds, mutation_rate)
            child2 = genetic_algorithm_mutation(child2, bounds, mutation_rate)
            next_pop[i] = child1
            next_pop[i+1] = child2 if i+1 < pop_size else child1
        pop = next_pop
    return best_individual, best_fitness

# ----------------------------------------------------------------------
# Rotordynamics
# ----------------------------------------------------------------------
def jeffcott_rotor_critical_speed(m: float, k: float) -> float:
    """Critical speed (rad/s) of a simple Jeffcott rotor."""
    return math.sqrt(k / m)

def rotordynamics_unbalance_response(m: float, c: float, k: float,
                                     e: float, omega: float) -> Tuple[float, float]:
    """
    Steady-state unbalance response amplitude and phase.
    m: rotor mass, c: damping, k: stiffness, e: eccentricity, omega: spin speed.
    Returns (amplitude, phase).
    """
    omega_n = math.sqrt(k / m)
    r = omega / omega_n
    zeta = c / (2 * math.sqrt(k * m))
    X = (e * r**2) / math.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    phi = math.atan2(2*zeta*r, 1 - r**2)
    return X, phi

# ----------------------------------------------------------------------
# Contact Mechanics
# ----------------------------------------------------------------------
def hertz_contact_pressure(p_max: float, a: float, b: float, x: float, y: float) -> float:
    """
    Hertzian contact pressure distribution for elliptical contact.
    p_max: maximum pressure, a: semi-major axis, b: semi-minor axis.
    """
    if (x/a)**2 + (y/b)**2 > 1:
        return 0.0
    return p_max * math.sqrt(1 - (x/a)**2 - (y/b)**2)

def hertz_contact_dimensions(F: float, E1: float, E2: float, nu1: float, nu2: float,
                             R1: float, R2: float) -> Tuple[float, float, float]:
    """
    Compute Hertz contact dimensions for sphere-plane contact.
    Returns (a, b, p_max).
    """
    E_eq = 1 / ((1 - nu1**2)/E1 + (1 - nu2**2)/E2)
    R_eq = 1 / (1/R1 + 1/R2)
    a = (3 * F * R_eq / (4 * E_eq)) ** (1/3)
    b = a  # circular contact
    p_max = 3 * F / (2 * math.pi * a * b)
    return a, b, p_max

# ----------------------------------------------------------------------
# Failure Theories
# ----------------------------------------------------------------------
def maximum_normal_stress_failure(sigma1: float, sigma2: float, sigma3: float,
                                  S_ut: float, S_uc: float) -> bool:
    """Maximum normal stress theory (Rankine)."""
    if sigma1 > 0:
        return sigma1 < S_ut
    elif sigma1 < 0:
        return abs(sigma1) < S_uc
    return True

def maximum_shear_stress_failure(sigma1: float, sigma2: float, sigma3: float,
                                 S_y: float) -> bool:
    """Maximum shear stress theory (Tresca)."""
    tau_max = (max(sigma1, sigma2, sigma3) - min(sigma1, sigma2, sigma3)) / 2
    return tau_max < S_y / 2

def von_mises_failure(sigma1: float, sigma2: float, sigma3: float,
                      S_y: float) -> bool:
    """Von Mises (distortion energy) theory."""
    vm = math.sqrt(0.5 * ((sigma1 - sigma2)**2 + (sigma2 - sigma3)**2 + (sigma3 - sigma1)**2))
    return vm < S_y

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 1. Stress transformation
    # ------------------------------------------------------------------
    sx, sy, txy = 100e6, 50e6, 30e6
    s1, s2 = principal_stresses(sx, sy, txy)
    print(f"Principal stresses: σ1={s1:.2e}, σ2={s2:.2e}")
    vm = von_mises(sx, sy, 0, txy, 0, 0)
    print(f"Von Mises stress: {vm:.2e}")

    # ------------------------------------------------------------------
    # 2. Truss analysis
    # ------------------------------------------------------------------
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    truss = Truss2D(0, 1, A=1e-4, material=Material('Steel', STEEL_E, STEEL_NU, STEEL_RHO))
    k = truss.stiffness_matrix(coords)
    print("Truss element stiffness matrix (first 2 rows):")
    print(k[:2, :])

    # ------------------------------------------------------------------
    # 3. Modal analysis of a 2DOF system
    # ------------------------------------------------------------------
    M = np.array([[2.0, 0.0], [0.0, 1.0]])
    K = np.array([[300.0, -100.0], [-100.0, 200.0]])
    eigvals, eigvecs = modal_analysis(M, K)
    omega_n = np.sqrt(eigvals)
    print(f"Natural frequencies (rad/s): {omega_n}")

    # ------------------------------------------------------------------
    # 4. SDOF damped response
    # ------------------------------------------------------------------
    def force_func(t):
        return 1000 * np.sin(2 * np.pi * 5 * t)
    t, x = sdof_damped_response(10.0, 50.0, 2000.0, force_func, 0.0, 0.0, (0, 0.5), 0.001)
    print(f"SDOF response at t=0.2s: {np.interp(0.2, t, x):.4f} m")

    # ------------------------------------------------------------------
    # 5. Fatigue life estimation
    # ------------------------------------------------------------------
    S_a = 150e6
    S_u = 400e6
    S_e = 200e6
    N_f = fatigue_life(S_a, S_u, S_e)
    print(f"Estimated fatigue life: {N_f:.2e} cycles")

    # ------------------------------------------------------------------
    # 6. Beam weight optimization
    # ------------------------------------------------------------------
    mat = Material('Steel', STEEL_E, STEEL_NU, STEEL_RHO, STEEL_YIELD)
    b_opt, h_opt = beam_weight_minimization(2.0, 5000.0, mat, (0.05, 0.5, 0.1, 0.5))
    print(f"Optimal beam cross-section: b={b_opt:.4f} m, h={h_opt:.4f} m")

    # ------------------------------------------------------------------
    # 7. Genetic algorithm example
    # ------------------------------------------------------------------
    def sphere(x):
        return np.sum(x**2)
    bounds_ga = [(-5, 5), (-5, 5)]
    best_ind, best_fit = genetic_algorithm_optimize(sphere, bounds_ga, pop_size=30, generations=50)
    print(f"GA best solution: {best_ind}, fitness: {best_fit:.6f}")

    # ------------------------------------------------------------------
    # 8. Jeffcott rotor critical speed
    # ------------------------------------------------------------------
    m_rotor = 20.0
    k_shaft = 2e6
    omega_crit = jeffcott_rotor_critical_speed(m_rotor, k_shaft)
    print(f"Critical speed: {omega_crit:.2f} rad/s ({omega_crit*60/(2*np.pi):.2f} rpm)")

    # ------------------------------------------------------------------
    # 9. Hertz contact
    # ------------------------------------------------------------------
    a, b, p_max = hertz_contact_dimensions(1000, STEEL_E, STEEL_E, STEEL_NU, STEEL_NU, 0.05, 0.05)
    print(f"Hertz contact: a={a:.6f} m, p_max={p_max:.2e} Pa")

    # ------------------------------------------------------------------
    # 10. Failure theory check
    # ------------------------------------------------------------------
    sig1, sig2, sig3 = 200e6, 100e6, 0.0
    safe_rankine = maximum_normal_stress_failure(sig1, sig2, sig3, STEEL_ULS, STEEL_ULS)
    safe_tresca = maximum_shear_stress_failure(sig1, sig2, sig3, STEEL_YIELD)
    safe_vm = von_mises_failure(sig1, sig2, sig3, STEEL_YIELD)
    print(f"Rankine safe: {safe_rankine}, Tresca safe: {safe_tresca}, Von Mises safe: {safe_vm}")