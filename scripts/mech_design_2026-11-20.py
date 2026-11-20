import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, solve_banded
from scipy.integrate import solve_ivp

# ==================== FINITE ELEMENT ANALYSIS ====================

class TrussElement:
    """1D truss element stiffness matrix (axial)."""
    def __init__(self, node_i, node_j, area, young_modulus):
        self.node_i = node_i
        self.node_j = node_j
        self.A = area
        self.E = young_modulus
        self.L = np.linalg.norm(node_j - node_i)
        self.cos_theta = (node_j[0] - node_i[0]) / self.L
        self.sin_theta = (node_j[1] - node_i[1]) / self.L

    def stiffness(self):
        k_axial = self.E * self.A / self.L
        c = np.array([[ self.cos_theta,  self.sin_theta, 0, 0],
                      [ self.sin_theta, -self.cos_theta, 0, 0],
                      [ 0,              0,             0, 0],
                      [ 0,              0,             0, 0]], dtype=float)
        return k_axial * c

class BeamElement:
    """Euler-Bernoulli beam element (2 nodes, 2 DOFs each)."""
    def __init__(self, node_i, node_j, area, young_modulus, moment_inertia):
        self.node_i = node_i
        self.node_j = node_j
        self.L = np.linalg.norm(node_j - node_i)
        self.E = young_modulus
        self.A = area
        self.I = moment_inertia
        self.c = self.E * self.I / (self.L**3)
        self.c0 = self.E * self.I / (self.L**2)

    def stiffness(self):
        k = np.zeros((4, 4))
        k[0, 0] = 12 * self.c
        k[0, 1] = 6 * self.c0
        k[0, 2] = -12 * self.c
        k[0, 3] = -6 * self.c0
        k[1, 0] = 6 * self.c0
        k[1, 1] = 4 * self.E * self.A / self.L + 2 * self.c
        k[1, 2] = -6 * self.c0
        k[1, 3] = -4 * self.E * self.A / self.L + 2 * self.c
        k[2, 0] = -12 * self.c
        k[2, 1] = -6 * self.c0
        k[2, 2] = 12 * self.c
        k[2, 3] = 6 * self.c0
        k[3, 0] = -6 * self.c0
        k[3, 1] = -4 * self.E * self.A / self.L + 2 * self.c
        k[3, 2] = 6 * self.c0
        k[3, 3] = 12 * self.c
        return k

class PlateElement:
    """Simple 2D plate bending element (placeholder)."""
    def __init__(self, node_i, node_j, thickness, young_modulus, poisson_ratio):
        self.node_i = node_i
        self.node_j = node_j
        self.h = thickness
        self.E = young_modulus
        self.nu = poisson_ratio
        self.D = self.E * self.h**3 / (12 * (1 - self.nu**2))  # flexural rigidity

    def stiffness(self):
        # Simplified 4x4 matrix for a single element (very approximate)
        k = np.zeros((4, 4))
        k[0, 0] = 12 * self.D / self.L**4
        k[0, 1] = 6 * self.D / self.L**3
        k[0, 2] = -12 * self.D / self.L**4
        k[0, 3] = -6 * self.D / self.L**3
        k[1, 0] = 6 * self.D / self.L**3
        k[1, 1] = 4 * self.D / self.L**2 + 2 * self.D / self.L**2
        k[1, 2] = -6 * self.D / self.L**3
        k[1, 3] = -4 * self.D / self.L**2 + 2 * self.D / self.L**2
        k[2, 0] = -12 * self.D / self.L**4
        k[2, 1] = -6 * self.D / self.L**3
        k[2, 2] = 12 * self.D / self.L**4
        k[2, 3] = 6 * self.D / self.L**3
        k[3, 0] = -6 * self.D / self.L**3
        k[3, 1] = -4 * self.D / self.L**2 + 2 * self.D / self.L**2
        k[3, 2] = 6 * self.D / self.L**3
        k[3, 3] = 12 * self.D / self.L**4
        return k

class FEAModel:
    """Assembly of global stiffness matrix and solution of displacements."""
    def __init__(self, ndofs):
        self.ndofs = ndofs
        self K = np.zeros((ndofs, ndofs))

    def add_element(self, elem_stiffness, dof_map):
        """Add element stiffness to global matrix given DOF mapping."""
        for i in range(4):
            for j in range(4):
                self.K[dof_map[i], dof_map[j]] += elem_stiffness[i, j]

    def apply_loads(self, force_vector):
        self.F = force_vector.copy()

    def apply_boundary_conditions(self, fixed_dofs):
        free_dofs = np.setdiff1d(np.arange(self.ndofs), fixed_dofs)
        Kff = self.K[np.ix_(free_dofs, free_dofs)]
        Ff = self.F[free_dofs]
        uf = np.linalg.solve(Kff, Ff)
        u = np.zeros(self.ndofs)
        u[free_dofs] = uf
        return u

    def solve(self, fixed_dofs, force_vector):
        self.apply_loads(force_vector)
        u = self.apply_boundary_conditions(fixed_dofs)
        return u

# ==================== STRESS TRANSFORMATIONS ====================

def stress_transformation(sigma_x, sigma_y, tau_xy, theta):
    """Transform plane stress components from local to global (theta in radians)."""
    c = np.cos(theta)
    s = np.sin(theta)
    sigma_xx = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
    sigma_yy = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
    tau_xy_new = -sigma_x * s * c + sigma_y * s * c + tau_xy * (c**2 - s**2)
    return sigma_xx, sigma_yy, tau_xy_new

def mohr_circle(sigma_x, sigma_y, tau_xy):
    """Return center and radius of Mohr's circle."""
    center = (sigma_x + sigma_y) / 2.0
    radius = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
    return center, radius

def plot_mohr(sigma_x, sigma_y, tau_xy):
    center, radius = mohr_circle(sigma_x, sigma_y, tau_xy)
    theta = np.linspace(0, np.pi, 200)
    sigma1 = center + radius * np.cos(2 * theta)
    sigma2 = center - radius * np.cos(2 * theta)
    tau = radius * np.sin(2 * theta)
    plt.figure()
    plt.plot(sigma1, tau, 'b-')
    plt.plot(sigma2, -tau, 'b--')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(center - radius, color='r', linestyle='--')
    plt.axvline(center + radius, color='r', linestyle='--')
    plt.title('Mohr\'s Circle')
    plt.xlabel('Normal Stress')
    plt.ylabel('Shear Stress')
    plt.grid(True)
    plt.show()

# ==================== STRESS CRITERIA ====================

def von_mises(sigma_x, sigma_y, tau_xy):
    """Von Mises equivalent stress."""
    return np.sqrt(sigma_x**2 + sigma_y**2 - sigma_x*sigma_y + 3*tau_xy**2)

def tresca(sigma_x, sigma_y, tau_xy):
    """Tresca equivalent stress (maximum shear stress)."""
    return np.max([abs(sigma_x - sigma_y)/2, abs(sigma_x)/2 + abs(tau_xy)/2, abs(sigma_y)/2 + abs(tau_xy)/2])

def max_normal_stress(sigma_x, sigma_y):
    """Maximum normal stress (principal stresses)."""
    return max(abs(sigma_x), abs(sigma_y))

# ==================== VIBRATIONS ====================

class VibrationModel:
    """Simple beam model for modal analysis (Euler-Bernoulli)."""
    def __init__(self, length, mass_per_length, flexural rigidity):
        self.L = length
        self.m = mass_per_length
        self.EI = flexural_rigidity

    def natural_frequencies(self, n_modes=5):
        """Compute first n_modes natural frequencies (rad/s) using eigenvalue problem."""
        # Assemble consistent mass and stiffness matrices (lumped mass for simplicity)
        ndofs = 2 * 2  # 2 nodes, 2 DOFs each (deflection, rotation)
        # Element stiffness (same as BeamElement)
        k_elem = BeamElement([0, 0], [self.L, 0], 1.0, 1.0, self.EI/12.0).stiffness()
        # Element mass (lumped)
        m_elem = np.array([[self.m*self.L/2, 0],
                           [0, self.m*self.L/2]])
        # Assemble global matrices
        K = np.zeros((4, 4))
        M = np.zeros((4, 4))
        dof_map = [0, 1, 2, 3]
        K[np.ix_(dof_map, dof_map)] += k_elem
        M[np.ix_(dof_map, dof_map)] += m_elem
        # Eigenvalue problem K*u = w^2 * M*u
        w2, _ = eig(K, M)
        w = np.sqrt(np.real(w2))
        w.sort()
        return w[:n_modes]

    def damped_sdof_response(self, omega_n, zeta, t_span, y0, v0):
        """Solve SDOF damped system: m*y'' + c*y' + k*y = 0."""
        m = 1.0
        k = omega_n**2 * m
        c = 2 * zeta * omega_n * m
        def ode(t, y):
            return [y[1], (-c/m)*y[1] - (k/m)*y[0]]
        sol = solve_ivp(ode, t_span, y0 + [v0], t_eval=np.linspace(*t_span, 500))
        return sol.t, sol.y[0]

    def mdof_state_space(self, M, C, K):
        """Return state-space matrices for MDOF system."""
        n = M.shape[0]
        A = np.block([[np.zeros((n,n)), np.eye(n)],
                      [np.linalg.inv(K), -np.linalg.inv(M) @ C]])
        B = np.block([[np.zeros((n,1)), np.zeros((n,1))],
                      [np.zeros((n,1)), np.eye(n)]])
        return A, B

# ==================== FATIGUE & FRACTURE MECHANICS ====================

def s_n_curve(stress_amplitude, cycles):
    """Simple linear S-N curve (log-log)."""
    # Assume 10^6 cycles at 200 MPa, slope -0.1
    sigma_ref = 200.0
    N_ref = 1e6
    C = (sigma_ref / stress_amplitude)**0.1 * N_ref
    return C / (cycles**0.1)

def paris_law(a0, C, m, N):
    """Crack growth integration using Paris' law."""
    da_dN = C * (1e6 * a0)**m  # simplified: assume constant amplitude
    a = np.zeros_like(N)
    a[0] = a0
    for i in range(1, len(N)):
        da = da_dN * (N[i] - N[i-1])
        a[i] = a[i-1] + da
    return a

def plot_goodman(stress_amplitude, sigma_y, sigma_uts):
    """Plot Goodman and Gerber safety lines."""
    N = np.logspace(1, 6, 200)
    sigma_a = stress_amplitude * np.ones_like(N)
    sigma_m = N * 0  # zero mean stress for pure alternating
    sigma_a_goodman = sigma_a * (1 - sigma_m / sigma_uts)
    sigma_a_gerber = sigma_a * (1 - (sigma_m / sigma_uts)**2)
    plt.figure()
    plt.plot(N, sigma_a_goodman, label='Goodman')
    plt.plot(N, sigma_a_gerber, label='Gerber')
    plt.plot(N, sigma_uts * (1 - sigma_m / sigma_y), '--', label='Yield limit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cycles to failure')
    plt.ylabel('Stress amplitude')
    plt.title('Goodman / Gerber Diagrams')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()

# ==================== STRUCTURAL OPTIMIZATION ====================

def beam_weight(b, h, L, rho):
    """Weight of a rectangular beam."""
    return rho * b * h * L

def deflection_rectangular(b, h, L, w, E):
    """Mid-span deflection for simply supported beam with uniform load w."""
    I = b * h**3 / 12.0
    return (5 * w * L**4) / (384 * E * I)

def optimize_beam(b, L, w, E, rho, target_deflection=0.01):
    """Simple optimization: vary height h to meet deflection limit."""
    h_vals = np.linspace(0.01, 0.5, 500)  # meters
    weights = []
    deflections = []
    for h in h_vals:
        I = b * h**3 / 12.0
        delta = (5 * w * L**4) / (384 * E * I)
        weights.append(beam_weight(b, h, L, rho))
        deflections.append(delta)
    idx = np.argmin([d for d in deflections if d <= target_deflection])
    if idx == -1:
        idx = np.argmin(deflections)  # relax constraint
    h_opt = h_vals[idx]
    return h_opt, beam_weight(b, h_opt, L, rho), deflections[idx]

# ==================== ROTORDYNAMICS ====================

class JeffcottRotor:
    """Simple Jeffcott rotor with mass, stiffness, damping."""
    def __init__(self, m, k, c):
        self.m = m
        self.k = k
        self.c = c

    def critical_speed(self):
        """First natural frequency (rad/s)."""
        return np.sqrt(self.k / self.m)

# ==================== CONTACT MECHANICS ====================

def contact_pressure(load, area):
    """Average contact pressure."""
    return load / area

# ==================== FAILURE THEORIES ====================

def failure_theories(sigma_principal1, sigma_principal2, sigma_ys, sigma_uts):
    """Evaluate several failure criteria."""
    von_mises_eq = von_mises(sigma_principal1, sigma_principal2, 0)
    tresca_max = tresca(sigma_principal1, sigma_principal2, 0)
    max_normal = max_normal_stress(sigma_principal1, sigma_principal2)
    return {
        'von_mises': von_mises_eq <= sigma_ys,
        'tresca': tresca_max <= sigma_ys,
        'max_normal': max_normal <= sigma_ys,
        'von_mises_uts': von_mises_eq <= sigma_uts
    }

# ==================== MAIN DEMONSTRATION ====================

if __name__ == '__main__':
    # ---- 1. Truss Example ----
    node1 = np.array([0.0, 0.0])
    node2 = np.array([3.0, 0.0])
    node3 = np.array([3.0, 4.0])
    model = FEAModel(ndofs=6)  # 3 nodes * 2 DOFs
    truss1 = TrussElement(node1, node2, area=0.01, young_modulus=210e9)
    truss2 = TrussElement(node2, node3, area=0.01, young_modulus=210e9)
    dof_map1 = [0, 1, 2, 3]
    dof_map2 = [2, 3, 4, 5]
    model.add_element(truss1.stiffness(), dof_map1)
    model.add_element(truss2.stiffness(), dof_map2)
    # Apply boundary conditions: fix node1 (DOFs 0,1)
    fixed = [0, 1]
    force = np.zeros(6)
    force[2] = -1000.0  # vertical load at node2
    u = model.solve(fixed, force)
    print("Truss displacements:", u)

    # Stress in element 1 (node1->node2)
    sigma_x = 0.0
    sigma_y = 0.0
    tau_xy = 0.0
    # Simple axial stress
    axial_force = 1000.0  # N
    sigma_axial = axial_force / (0.01 * 0.01)  # simplified area = 0.01 m^2
    print("Axial stress:", sigma_axial)

    # Mohr's circle
    plot_mohr(sigma_axial, 0.0, 0.0)

    # ---- 2. Beam Vibration Example ----
    beam = VibrationModel(length=2.0, mass_per_length=1.0, flexural_rigidity=1e6)
    freqs = beam.natural_frequencies(n_modes=3)
    print("First 3 natural frequencies (rad/s):", freqs)
    # Plot mode shapes (simplified)
    t = np.linspace(0, 2*np.pi, 200)
    for i, w in enumerate(freqs[:2]):
        y, _ = beam.damped_sdof_response(omega_n=w, zeta=0.05, t_span=(0, 10), y0=(1.0, 0.0), v0=0.0)
        plt.plot(t, y, label=f'Mode {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.title('Damped SDOF Response')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- 3. Fatigue Example ----
    stress_amp = 150.0  # MPa
    cycles = 1e6
    N_to_failure = s_n_curve(stress_amp, cycles)
    print(f"Cycles to failure for {stress_amp} MPa: {N_to_failure:.2e}")
    plot_goodman(stress_amp, sigma_y=250.0, sigma_uts=400.0)

    # ---- 4. Structural Optimization ----
    b = 0.10  # m
    L = 5.0   # m
    w = 1000.0  # N/m (uniform load)
    E = 210e9  # Pa
    rho = 7850  # kg/m3
    h_opt, w_opt, delta_opt = optimize_beam(b, L, w, E, rho, target_deflection=0.01)
    print(f"Optimized height: {h_opt:.3f} m, Weight: {w_opt:.2f} kg, Deflection: {delta_opt:.4f} m")

    # ---- 5. Rotor Example ----
    rotor = JeffcottRotor(m=2.0, k=5e4, c=10.0)
    crit = rotor.critical_speed()
    print(f"Jeffcott critical speed: {crit:.2f} rad/s")

    # ---- 6. Contact Mechanics ----
    load = 5000.0  # N
    area = 0.02  # m^2
    p = contact_pressure(load, area)
    print(f"Contact pressure: {p:.2f} Pa")

    # ---- 7. Failure Theories ----
    sigma1 = 120.0  # MPa
    sigma2 = 80.0   # MPa
    sigma_ys = 250.0  # MPa
    sigma_uts = 400.0  # MPa
    results = failure_theories(sigma1, sigma2, sigma_ys, sigma_uts)
    print("Failure criteria results:", results)

    plt.show()
```