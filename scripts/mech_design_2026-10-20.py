import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Constants and Material Defaults
# ----------------------------------------------------------------------
G_JUPITER = 1.0e11          # Young's modulus (Pa) for structural steel
NU_JUPITER = 0.3            # Poisson's ratio
RHO_STEEL = 7850.0          # Density (kg/m^3)
YIELD_STRESS_STEEL = 250e6  # Yield stress (Pa)
E_STEEL = 210e9             # Young's modulus (Pa) for structural steel
E_ALUMINUM = 69e9           # Young's modulus (Pa) for aluminum
RHO_ALUMINUM = 2700.0       # Density (kg/m^3)

# ----------------------------------------------------------------------
# Truss Analysis (1D)
# ----------------------------------------------------------------------
class TrussElement:
    """Element stiffness matrix for a 2-node axial truss member."""
    def __init__(self, A, L, E):
        self.A = A          # Cross-sectional area (m^2)
        self.L = L          # Length (m)
        self.E = E          # Young's modulus (Pa)
        self.k = self.E * self.A / self.L  # Axial stiffness (N/m)

    def stiffness(self):
        """Return 4x4 element stiffness matrix in global coordinates."""
        k = self.k
        T = np.array([[1, 0, -1, 0],
                      [0, 0,  0, 0],
                      [-1, 0,  1, 0],
                      [0, 0,  0, 0]], dtype=float)
        return k * T

class Truss:
    """Assembly and solution of a 1D truss."""
    def __init__(self, nodes, elements, area, E):
        """
        nodes: list of (x, y) coordinates (y ignored for 1D)
        elements: list of (node_i, node_j) indices (1‑based)
        area: cross‑sectional area (m^2)
        E: Young's modulus (Pa)
        """
        self.nodes = np.array(nodes)
        self.elements = np.array(elements) - 1  # zero‑based
        self.A = area
        self.E = E
        self K = None

    def assemble(self):
        n_nodes = len(self.nodes)
        self.K = np.zeros((2*n_nodes, 2*n_nodes))
        for conn in self.elements:
            i, j = conn
            k_elem = TrussElement(self.A, self.L(i, j), self.E).stiffness()
            dof_map = [2*i, 2*i+1, 2*j, 2*j+1]
            self.K[np.ix_(dof_map, dof_map)] += k_elem

    def L(self, i, j):
        """Geometric length between node i and node j."""
        return np.linalg.norm(self.nodes[j] - self.nodes[i])

    def apply_loads(self, F, u_fixed):
        """
        F: array of applied forces at nodes (positive tension)
        u_fixed: boolean array indicating which displacements are fixed (True = fixed)
        """
        self.assemble()
        n_dof = self.K.shape[0]
        free_dof = np.where(~u_fixed)[0]
        F_free = F[free_dof]
        K_ff = self.K[np.ix_(free_dof, free_dof)]
        u_free = la.solve(K_ff, F_free)
        u_full = np.zeros(n_dof)
        u_full[free_dof] = u_free
        return u_full

    def solve_example(self):
        """Simple 3‑node truss example."""
        nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5)]
        elements = [(1, 2), (2, 3)]  # node indices 1‑based
        truss = Truss(nodes, elements, area=0.01, E=E_STEEL)
        F = np.zeros(2*len(nodes))
        F[0] = -1000.0   # 1000 N tensile on node 1
        F[4] = 1000.0    # 1000 N tensile on node 3
        u_fixed = np.array([True, False, True, False, True, False])
        u = truss.apply_loads(F, u_fixed)
        print("Truss displacements (m):", u[::2])
        print("Truss axial forces (N):", u[1::2] * truss.k)  # approximate axial force

# ----------------------------------------------------------------------
# Beam Analysis (2D Euler‑Bernoulli)
# ----------------------------------------------------------------------
def beam_element_stiffness(L, E, I, A):
    """Return 12x12 element stiffness matrix for a 2‑node Euler‑Bernoulli beam."""
    # Axial part
    k_ax = E * A / L
    # Bending part
    k_b = E * I / L**3 * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])
    # Assemble
    dof = [0, 1, 2, 3]  # u, v, θ_i, θ_j (local)
    K = np.zeros((12, 12))
    for i, d in enumerate(dof):
        for j, e in enumerate(dof):
            K[i, j] = k_ax * (1 if i == j else 0)  # axial coupling only for u‑u and v‑v (ignored here)
            if i < 2 and j < 2:  # axial u‑u and v‑v (v is axial if no transverse)
                continue
            # bending coupling
            if i < 2:  # transverse displacement (v)
                if j < 2:
                    continue
                # v‑θ coupling
                K[i, j] = k_b[1, 2] if i == 1 and j == 2 else k_b[3, 2] if i == 3 and j == 2 else 0
                K[i, j] = k_b[1, 0] if i == 1 and j == 0 else k_b[3, 0] if i == 3 and j == 0 else 0
            else:  # rotation θ
                if j < 2:  # θ‑v coupling
                    K[i, j] = k_b[2, 1] if i == 2 and j == 1 else k_b[0, 1] if i == 0 and j == 1 else 0
                else:  # θ‑θ
                    K[i, j] = k_b[2, 2] if i == 2 and j == 2 else k_b[0, 0] if i == 0 and j == 0 else 0
    return K

class BeamElement:
    """Beam element with axial, shear, and bending DOFs."""
    def __init__(self, node_i, node_j, L, E, I, A):
        self.node_i = node_i
        self.node_j = node_j
        self.L = L
        self.E = E
        self.I = I
        self.A = A
        self.k = beam_element_stiffness(L, E, I, A)

    def assemble_global(self, K_global, u):
        """Add element stiffness to global matrix."""
        dof_i = [2*self.node_i, 2*self.node_i+1, 2*self.node_i+2]
        dof_j = [2*self.node_j, 2*self.node_j+1, 2*self.node_j+2]
        dof_all = dof_i + dof_j
        K_global[np.ix_(dof_all, dof_all)] += self.k
        return K_global

class Beam:
    """Assembly and solution of a 2D beam structure."""
    def __init__(self, nodes, elements, E, I, A):
        """
        nodes: (N,2) array of coordinates
        elements: list of (i, j) node indices (1‑based)
        E, I, A: Young's modulus, second moment, area (same for all elements)
        """
        self.nodes = np.array(nodes)
        self.elements = np.array(elements) - 1
        self.E = E
        self.I = I
        self.A = A
        self.K = None

    def assemble(self):
        n_nodes = len(self.nodes)
        self.K = np.zeros((2*n_nodes, 2*n_nodes))
        for conn in self.elements:
            i, j = conn
            L = np.linalg.norm(self.nodes[j] - self.nodes[i])
            elem = BeamElement(i, j, L, self.E, self.I, self.A)
            self.K = elem.assemble_global(self.K, None)

    def apply_loads(self, F, u_fixed):
        """
        F: force vector (size 2N) – positive in global directions
        u_fixed: boolean mask (size 2N) indicating fixed DOFs
        """
        self.assemble()
        n_dof = self.K.shape[0]
        free = np.where(~u_fixed)[0]
        F_free = F[free]
        K_ff = self.K[np.ix_(free, free)]
        u_free = la.solve(K_ff, F_free)
        u_full = np.zeros(n_dof)
        u_full[free] = u_free
        return u_full

    def solve_example(self):
        """Simple simply‑supported beam example."""
        L = 4.0  # m
        nodes = [(0.0, 0.0), (L, 0.0)]
        elements = [(1, 2)]
        beam = Beam(nodes, elements, E=E_STEEL, I=1e-6, A=0.01)
        F = np.zeros(4)
        F[1] = -1000.0   # vertical load at right node
        u_fixed = np.array([True, False, True, False])
        u = beam.apply_loads(F, u_fixed)
        print("Beam displacements (m):", u[0], u[1], u[2], u[3])
        # Extract axial and bending stresses at node 2
        u_node2 = u[2:4]  # θ_i, θ_j
        # Simple bending stress: sigma = M*z/I, M = V*L/2 (approx)
        V = -F[1]  # shear force
        M = V * L / 2
        z = 0.01  # distance from neutral axis (m)
        sigma = M * z / self.I
        print("Bending stress at node 2 (Pa):", sigma)

# ----------------------------------------------------------------------
# Stress & Strain Transformations
# ----------------------------------------------------------------------
def stress_transformation(sx, sy, tau_xy, theta):
    """
    Transform plane stress from global to rotated coordinates.
    theta: rotation angle (rad) from global x‑axis to new x'‑axis.
    Returns (sx_prime, sy_prime, tau_xy_prime).
    """
    c = np.cos(theta)
    s = np.sin(theta)
    sx_prime = sx * c**2 + sy * s**2 + 2 * tau_xy * s * c
    sy_prime = sx * s**2 + sy * c**2 - 2 * tau_xy * s * c
    tau_xy_prime = (sy - sx) * s * c + tau_xy * (c**2 - s**2)
    return sx_prime, sy_prime, tau_xy_prime

class MohrCircle:
    """Compute and plot Mohr's circle for plane stress."""
    def __init__(self, sx, sy, tau):
        self.sx = sx
        self.sy = sy
        self.tau = tau
        self.center = (sx + sy) / 2.0
        self.radius = np.sqrt(((sx - sy) / 2.0)**2 + tau**2)

    def plot(self):
        theta = np.linspace(0, np.pi, 200)
        sigma = self.center + self.radius * np.cos(theta)
        tau_circle = self.radius * np.sin(theta)
        plt.figure()
        plt.plot(sigma, tau_circle, 'b')
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        plt.title('Mohr\'s Circle')
        plt.xlabel('Normal Stress (Pa)')
        plt.ylabel('Shear Stress (Pa)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def von_mises(sx, sy, tau):
    """Von Mises equivalent stress for plane stress."""
    return np.sqrt(sx**2 + sy**2 - sx*sy + 3*tau**2)

def tresca(sx, sy, tau):
    """Tresca maximum shear stress criterion."""
    return np.max([abs(sx - sy), abs(sx + tau), abs(sy + tau)]) / 2.0

# ----------------------------------------------------------------------
# Mechanical Vibrations
# ----------------------------------------------------------------------
def sdo_damped_response(m, c, k, u0=0.0, v0=0.0, t_max=10.0, dt=0.001):
    """Solve m*u'' + c*u' + k*u = 0 for under‑damped SDOF."""
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    if zeta >= 1:
        raise ValueError("System is not under‑damped")
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    t = np.arange(0, t_max, dt)
    u = np.zeros_like(t)
    v = np.zeros_like(t)
    for i in range(1, len(t)):
        a = (-c * v[i-1] - k * u[i-1]) / m
        u[i] = u[i-1] + v[i-1] * dt + 0.5 * a * dt**2
        v[i] = v[i-1] + a * dt
    return t, u

def modal_analysis(K, M):
    """Return natural frequencies (rad/s) and mode shapes for MDOF system."""
    w_sq, V = la.eig(la.solve(M, K))
    w = np.sqrt(np.real(w_sq))
    idx = np.argsort(w)
    return w[idx], V[:, idx]

def sdof_damped_example():
    m = 100.0          # kg
    c = 10.0           # Ns/m
    k = 1000.0         # N/m
    t, u = sdo_damped_response(m, c, k)
    plt.figure()
    plt.plot(t, u)
    plt.title('SDOF Damped Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    plt.show()

# ----------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------
def s_n_curve(N_f, sigma_a):
    """Simple linear S‑N curve: N_f = C / sigma_a^b."""
    C = 1e9
    b = -0.1
    return C / (sigma_a**b)

def paris_law(a0, da_dn, C, m, n_cycles):
    """Integrate Paris' law da/dn = C * a^m for n_cycles."""
    a = a0
    da = C * (a**m) * n_cycles
    return a + da

def fatigue_example():
    sigma_a = 150e6  # Pa
    N_f = s_n_curve(1e6, sigma_a)
    print(f"Estimated fatigue life: {N_f:.0f} cycles")
    # Paris law parameters
    C = 6e-11
    m = 0.5
    a0 = 0.001  # m
    da_dn = C * (a0**m)
    a_final = paris_law(a0, da_dn, C, m, 1e5)
    print(f"Crack length after 1e5 cycles: {a_final:.6f} m")

# ----------------------------------------------------------------------
# Structural Optimization (Weight Minimization)
# ----------------------------------------------------------------------
def beam_weight(b, h, L, rho, E, yield):
    """Cross‑section weight (kg) for rectangular beam."""
    A = b * h
    V = A * L
    return rho * V

def beam_deflection(b, h, L, E, I, P):
    """Maximum deflection for simply supported beam with central load P."""
    return P * L**3 / (48 * E * I)

def optimization_objective(x, L, P, E, I, rho, yield_stress):
    """Objective: minimize weight subject to stress and deflection constraints."""
    b, h = x
    A = b * h
    sigma = P * L / (2 * A)  # approximate axial stress
    if sigma > yield_stress:
        return 1e6  # penalize infeasible
    I_val = b * h**3 / 12
    delta = beam_deflection(b, h, L, E, I_val, P)
    if delta > L / 250:  # deflection limit
        return 1e6
    return beam_weight(b, h, L, rho, E, yield_stress)

def optimization_example():
    L = 2.0          # m
    P = 5000.0       # N
    E = E_STEEL
    I = 1e-6         # initial I (m^4) – will be recomputed
    rho = RHO_STEEL
    yield_stress = YIELD_STRESS_STEEL
    # Initial guess
    b0, h0 = 0.1, 0.2
    bounds = [(0.01, 0.5), (0.01, 0.5)]
    res = opt.minimize(optimization_objective, x0=[b0, h0],
                       args=(L, P, E, I, rho, yield_stress),
                       bounds=bounds, method='SLSQP')
    b_opt, h_opt = res.x
    print(f"Optimized dimensions: b={b_opt:.4f} m, h={h_opt:.4f} m")
    print(f"Optimal weight: {beam_weight(b_opt, h_opt, L, rho, E, yield_stress):.2f} kg")

# ----------------------------------------------------------------------
# Rotordynamics (Critical Speed Approximation)
# ----------------------------------------------------------------------
def shaft_natural_frequency(E, I, m, L):
    """Critical speed for a simply supported shaft (Euler‑Bernoulli)."""
    omega = (np.pi**2) * np.sqrt(E * I / (m * L**4))
    return omega / (2 * np.pi)  # Hz

def rotor_example():
    E = E_STEEL
    I = 5e-6          # m^4
    m = 50.0          # kg
    L = 1.0           # m
    fn = shaft_natural_frequency(E, I, m, L)
    print(f"Critical shaft speed: {fn:.2f} Hz")

# ----------------------------------------------------------------------
# Contact Mechanics (Hertzian contact)
# ----------------------------------------------------------------------
def hertz_contact(R1, R2, P):
    """
    Compute contact radius a and maximum pressure p0 for two elastic spheres.
    R1, R2: radii (m)
    P: normal load (N)
    Returns a (m) and p0 (Pa).
    """
    E_star = 1 / ( (1 - np.square(0.3) / E_STEEL).sum() )  # simplified
    a = np.sqrt( (3 * P * R1 * R2) / (4 * E_star * (R1 + R2)) )
    p0 = (3 * P) / (2 * np.pi * a**2)
    return a, p0

# ----------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Truss example
    print("=== Truss Example ===")
    truss_example()

    # 2. Beam example
    print("\n=== Beam Example ===")
    beam_example()

    # 3. Stress transformation & Mohr's circle
    print("\n=== Stress Transformation & Mohr's Circle ===")
    sx, sy, tau = 100e6, 20e6, 15e6  # Pa
    theta = np.radians(30)
    sx_p, sy_p, tau_p = stress_transformation(sx, sy, tau, theta)
    print(f"Global: σx={sx/1e6:.1f} MPa, σy={sy/1e6:.1f} MPa, τxy={tau/1e6:.1f} MPa")
    print(f"Rotated: σx'={sx_p/1e6:.1f} MPa, σy'={sy_p/1e6:.1f} MPa, τxy'={tau_p/1e6:.1f} MPa")
    MohrCircle(sx, sy, tau).plot()

    # 4. Von Mises & Tresca criteria
    vm = von_mises(sx, sy, tau)
    tc = tresca(sx, sy, tau)
    print(f"\nVon Mises equivalent stress: {vm/1e6:.2f} MPa")
    print(f"Tresca limit: {tc/1e6:.2f} MPa")

    # 5. Vibration SDOF
    print("\n=== SDOF Damped Vibration Example ===")
    sdof_damped_example()

    # 6. Modal analysis (MDOF)
    print("\n=== MDOF Modal Analysis (synthetic) ===")
    n = 3
    K = np.random.rand(n, n) * 1e4 + 100
    M = np.diag(np.random.rand(n) * 10 + 1)
    w, V = modal_analysis(K, M)
    print("Natural frequencies (rad/s):", w)

    # 7. Fatigue example
    print("\n=== Fatigue Example ===")
    fatigue_example()

    # 8. Structural optimization
    print("\n=== Structural Optimization Example ===")
    optimization_example()

    # 9. Rotor critical speed
    print("\n=== Rotor Critical Speed Example ===")
    rotor_example()

    plt.close('all')
    print("\nAll demonstrations completed.")