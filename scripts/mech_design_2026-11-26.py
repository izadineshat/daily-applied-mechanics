import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants and Material Defaults (Steel unless otherwise specified)
# ---------------------------------------------------------------------------
DEFAULT_STEEL_E = 210e9      # Pa, Young's modulus
DEFAULT_STEEL_RHO = 7850     # kg/m^3, density
DEFAULT_STEEL_NU = 0.3       # Poisson's ratio


# ---------------------------------------------------------------------------
# 1. Finite Element Analysis (1D Truss & Beam)
# ---------------------------------------------------------------------------

class TrussFEA:
    """Linear truss (bar) finite element analysis.
    
    Supports arbitrary 3D truss structures with axial deformation only.
    """
    def __init__(self, E=DEFAULT_STEEL_E, A=1e-4):
        self.E = E
        self.A = A
        self.nodes = []      # list of (x,y,z)
        self.elements = []     # list of (i, j, A_local) or use class A
        self.loads = {}        # node_idx -> (Fx, Fy, Fz)
        self.bc = {}           # node_idx -> (ux, uy, uz) True=fixed

    def add_node(self, x, y, z):
        self.nodes.append((x, y, z))
        return len(self.nodes) - 1

    def add_element(self, i, j, A=None):
        """Add truss element between node i and j. A can override default."""
        self.elements.append((i, j, A if A is not None else self.A))

    def apply_load(self, node_idx, Fx=0, Fy=0, Fz=0):
        self.loads[node_idx] = (Fx, Fy, Fz)

    def apply_bc(self, node_idx, ux=False, uy=False, uz=False):
        self.bc[node_idx] = (ux, uy, uz)

    def assemble_stiffness(self):
        n_nodes = len(self.nodes)
        K = np.zeros((3 * n_nodes, 3 * n_nodes))

        def dof(idx, comp):
            return 3 * idx + comp  # 0:x, 1:y, 2:z

        for el_i, el_j, A in self.elements:
            xi, yi, zi = self.nodes[el_i]
            xj, yj, zj = self.nodes[el_j]
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
            if L == 0:
                continue
            c = (xj - xi) / L
            s = (yj - yi) / L
            t = (zj - zi) / L

            k_local = (self.E * A / L) * np.array([
                [ c,  s,  t, -c, -s, -t],
                [ s,  c, -t, -s,  c, -t],
                [ t, -s,  c, -t,  s,  c],
                [-c, -s, -t,  c,  s,  t],
                [-s,  c,  s,  s, -c, -t],
                [ t, -s,  c, -t,  s,  c]
            ], dtype=float)
            # Map local DOFs to global
            g = [dof(el_i, 0), dof(el_i, 1), dof(el_i, 2),
                 dof(el_j, 0), dof(el_j, 1), dof(el_j, 2)]
            for r in range(6):
                for c_ in range(6):
                    K[g[r], g[c_]] += k_local[r, c_]
        return K

    def apply_boundary_conditions(self, K, F):
        n_nodes = len(self.nodes)
        F_vec = F.copy() if F.size else np.zeros(3 * n_nodes)
        # Identify free and fixed DOFs
        fixed = []
        free = []
        for i in range(3 * n_nodes):
            dof_idx = i
            node = i // 3
            comp = i % 3
            if node in self.bc and self.bc[node][comp]:
                fixed.append(i)
            else:
                free.append(i)
        # Partition K and F
        K_ff = K[np.ix_(free, free)]
        F_f = F_vec[free] - K[np.ix_(free, fixed)] @ np.zeros(len(fixed))  # no loads on fixed
        # Actually, distribute fixed reaction loads properly:
        # F_f already has applied loads on free DOFs; fixed DOFs have zero displacement
        # RHS for free: F_f - K_fixed_free * u_fixed, but u_fixed=0
        u_free = solve(K_ff, F_f)
        u = np.zeros(3 * n_nodes)
        u[free] = u_free
        # Reactions at fixed
        r = K @ u - F_vec
        return u, r, free, fixed

    def solve(self, F=None):
        K = self.assemble_stiffness()
        F_vec = np.zeros(3 * len(self.nodes)) if F is None else np.zeros(3 * len(self.nodes))
        for nid, (fx, fy, fz) in (F or {}).items():
            F_vec[3*nid]     += fx
            F_vec[3*nid + 1] += fy
            F_vec[3*nid + 2] += fz
        u, r, free, fixed = self.apply_boundary_conditions(K, F_vec)
        return u, r


class BeamFEA:
    """Euler-Bernoulli beam finite element analysis (2-node elements)."""
    def __init__(self, E=DEFAULT_STEEL_E, I=1e-8, A=1e-4, L_elem=None):
        self.E = E
        self.I = I
        self.A = A
        self.L_elem = L_elem
        self.nodes = []      # list of (x,) 1D coordinates
        self.elements = []   # list of (i, j) node indices
        self.loads = {}      # node_idx -> (fx, fy, mx) in local beam coords
        self.bc = {}         # node_idx -> (ux, theta, vy) constraints

    def add_node(self, x):
        self.nodes.append(x)
        return len(self.nodes) - 1

    def add_element(self, i, j):
        self.elements.append((i, j))

    def apply_load(self, node_idx, fx=0, fy=0, mx=0):
        self.loads[node_idx] = (fx, fy, mx)

    def apply_bc(self, node_idx, ux=False, theta=False, vy=False):
        self.bc[node_idx] = (ux, theta, vy)

    def element_stiffness(self, L):
        E, I = self.E, self.I
        k = (E * I / L**3) * np.array([
            [ 12,   6*L, -12,   6*L],
            [ 6*L,  4*L**2, -6*L,  2*L**2],
            [-12,  -6*L,  12,  -6*L],
            [ 6*L,  2*L**2, -6*L,  4*L**2]
        ], dtype=float)
        return k

    def assemble_stiffness(self):
        n_nodes = len(self.nodes)
        K = np.zeros((3 * n_nodes, 3 * n_nodes))  # each node: ux, theta, vy

        def dof(idx, comp):
            return 3 * idx + comp  # 0:ux, 1:theta, 2:vy

        for i, j in self.elements:
            xi, xj = self.nodes[i], self.nodes[j]
            L = abs(xj - xi)
            if L == 0:
                continue
            k = self.element_stiffness(L)
            g = [dof(i, 0), dof(i, 1), dof(i, 2),
                 dof(j, 0), dof(j, 1), dof(j, 2)]
            for r in range(6):
                for c_ in range(6):
                    K[g[r], g[c_]] += k[r, c_]
        return K

    def solve(self, F=None):
        K = self.assemble_stiffness()
        n_nodes = len(self.nodes)
        F_vec = np.zeros(3 * n_nodes) if F is None else np.zeros(3 * n_nodes)
        for nid, (fx, fy, mx) in (F or {}).items():
            F_vec[3*nid]     += fx
            F_vec[3*nid + 1] += fy
            F_vec[3*nid + 2] += mx
        u, r, free, fixed = self.apply_boundary_conditions(K, F_vec)
        # Extract deflections: ux, theta, vy
        return u, r


# ---------------------------------------------------------------------------
# 2. Stress & Strain Transformations & Failure Criteria
# ---------------------------------------------------------------------------

def transform_stress(sigma_x, sigma_y, tau_xy, theta_deg):
    """Transform plane stress state by angle theta (degrees).
    Returns (sigma_x', sigma_y', tau_xy') in new axes."""
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    sigma_x_prime = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
    sigma_y_prime = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
    tau_xy_prime = (sigma_y - sigma_x) * s * c + tau_xy * (c**2 - s**2)
    return sigma_x_prime, sigma_y_prime, tau_xy_prime


def principal_stresses(sigma_x, sigma_y, tau_xy):
    """Compute principal stresses sigma_1, sigma_2 (sigma_1 >= sigma_2)."""
    avg = (sigma_x + sigma_y) / 2
    R = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
    sigma_1 = avg + R
    sigma_2 = avg - R
    return sigma_1, sigma_2


def max_shear_stress(sigma_x, sigma_y, tau_xy):
    """Tresca maximum shear stress."""
    sigma_1, sigma_2 = principal_stresses(sigma_x, sigma_y, tau_xy)
    return (sigma_1 - sigma_2) / 2


def von_mises_sigma(sigma_x, sigma_y, tau_xy):
    """Von Mises equivalent stress for plane stress."""
    return np.sqrt(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3 * tau_xy**2)


def goodman_line(sigma_a, sigma_m, sigma_e, sigma_uts):
    """Goodman criterion: sigma_a/sigma_e + sigma_m/sigma_uts <= 1 for safety."""
    return sigma_a / sigma_e + sigma_m / sigma_uts


def gerber_curve(sigma_a, sigma_m, sigma_e, sigma_uts):
    """Gerber parabola criterion."""
    return (sigma_a / sigma_e) + (sigma_m / sigma_uts)**2


def mohr_circle_plot(sigma_x, sigma_y, tau_xy, save_path=None):
    """Plot Mohr's circle and return center, radius, principal stresses."""
    center = (sigma_x + sigma_y) / 2
    radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
    sigma_1 = center + radius
    sigma_2 = center - radius
    theta_max = np.arctan2(2*tau_xy, sigma_x - sigma_y) / 2

    theta = np.linspace(0, np.pi, 400)
    sigma = center + radius * np.cos(theta)
    tau = radius * np.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(sigma, tau, 'b-', label='Mohr\'s Circle')
    plt.plot([sigma_1, sigma_2], [0, 0], 'ro', markersize=8, label='Principal stresses')
    plt.plot([center, center], [-radius, radius], 'g--', label='Circle center')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Normal stress σ')
    plt.ylabel('Shear stress τ')
    plt.title('Mohr\'s Circle')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return center, radius, sigma_1, sigma_2


# ---------------------------------------------------------------------------
# 3. Mechanical Vibrations (SDOF & MDOF, State-Space)
# ---------------------------------------------------------------------------

class SDOF:
    """Single Degree of Freedom system: m*u'' + c*u' + k*u = F(t)
    Supports free vibration, damped response, and forced harmonic response."""
    def __init__(self, m, c, k, x0=0, v0=0):
        self.m = m
        self.c = c
        self.k = k
        self.x0 = x0
        self.v0 = v0
        omega_n = np.sqrt(k / m)
        zeta = c / (2 * np.sqrt(m * k))
        self.omega_n = omega_n
        self.zeta = zeta
        self.omega_d = omega_n * np.sqrt(1 - zeta**2) if zeta < 1 else 0

    def free_response(self, t):
        """Free vibration displacement response x(t)."""
        zeta, omega_d = self.zeta, self.omega_d
        omega_n = self.omega_n
        if zeta == 1:
            # Critically damped (simplified)
            return self.x0 * np.cos(omega_n * t) + (self.v0 + omega_n * self.x0) / omega_n * np.sin(omega_n * t) * np.exp(-omega_n * t)
        elif zeta < 1:
            omega_d = omega_n * np.sqrt(1 - zeta**2)
            phi = np.arct2(-omega_d * self.x0, zeta * omega_n * self.x0 + self.v0) if abs(self.x0) > 1e-12 else 0
            amp = np.sqrt(self.x0**2 + (self.v0 + zeta * omega_n * self.x0)**2 / omega_d**2)
            return amp * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t - phi)
        else:
            # Overdamped (not fully implemented, return exponential decay)
            return self.x0 * np.exp(-(zeta - np.sqrt(zeta**2 - 1)) * omega_n * t)

    def forced_harmonic(self, F0, omega, t):
        """Steady-state response to F(t) = F0*sin(omega*t)."""
        zeta, omega_n = self.zeta, self.omega_n
        denom = np.sqrt((1 - (omega/omega_n)**2)**2 + (2*zeta*omega/omega_n)**2)
        phase = np.arctan2(2*zeta*omega/omega_n, 1 - (omega/omega_n)**2)
        X = F0 / (self.k * denom)  # amplitude of displacement
        return X * np.sin(omega * t - phase)


class MDOF:
    """Multi-Degree of Freedom system using mass and stiffness matrices.
    Provides modal analysis (natural frequencies, mode shapes) and direct integration."""
    def __init__(self, M, C, K):
        """M, C, K: numpy arrays (nDOF x nDOF)."""
        self.M = np.array(M, dtype=float)
        self.C = np.array(C, dtype=float)
        self.K = np.array(K, dtype=float)
        n = M.shape[0]
        # Eigenvalue problem: (K - omega^2 M) phi = 0
        omega_sq, Phi = eig(self.K, self.M)
        omega = np.sqrt(omega_sq).real  # natural frequencies (rad/s)
        idx = np.argsort(omega)
        self.omega = omega[idx]
        self.Phi = Phi[:, idx]
        # Modal damping (proportional assumption)
        # alpha*M + beta*K diagonalized by Phi
        # Here we just store eigenvalues/eigenvectors; damping matrix used in state-space below

    def natural_frequencies_hz(self):
        return self.omega / (2 * np.pi)

    def mode_shapes(self):
        return self.Phi

    def state_space(self, x0, v0, t):
        """Simple state-space integration using matrix exponential (zero-order hold)."""
        n = self.M.shape[0]
        # State vector: [u; u_dot]
        A = np.zeros((2*n, 2*n))
        B = np.zeros((2*n, n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(self.M, self.K)
        A[n:, n:] = -np.linalg.solve(self.M, self.C)
        # Initial state
        z0 = np.zeros(2*n)
        z0[:n] = x0
        z0[n:] = v0
        # Time integration using scipy linalg expm
        from scipy.linalg import expm
        # For constant zero input, we can use state transition matrix
        # Here we just do simple Newmark-like step for demo; full integration would need time looping
        # Placeholder: return analytical modal solution
        # We'll just compute undamped modal participation for demonstration
        return "State-space matrix A constructed; use time-stepping loop for response"


def sdoe_damped_response():
    """Demo SDOF damped response."""
    m, c, k = 10, 200, 2000  # kg, Ns/m, N/m
    sdof = SDOF(m, c, k, x0=0.01, v0=0)
    t = np.linspace(0, 5, 500)
    x = np.array([sdof.free_response(ti) for ti in t])
    plt.figure()
    plt.plot(t, x)
    plt.title('SDOF Free Damped Response')
    plt.xlabel('time (s)')
    plt.ylabel('displacement (m)')
    plt.grid(True, alpha=0.3)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Fatigue & Fracture Mechanics
# ---------------------------------------------------------------------------

def s_n_curve_basquin(sigma_a, sigma_f_prime=1200, b=-0.088):
    """Basquin's law: sigma_a = sigma_f_prime * (2*N)^b -> cycles to failure N."""
    if sigma_a <= 0:
        return np.inf
    N = 0.5 * ((sigma_a / sigma_f_prime) ** (1 / b))
    return N


def miner_damage(n_cycles, n_applied, N_to_failure):
    """Miner's rule cumulative damage: D = sum(n_i / N_i). Failure when D >= 1."""
    D = np.sum(n_applied / N_to_failure)
    return D


def goodman_check(sigma_a, sigma_m, sigma_e, sigma_uts):
    """Return factor of safety on Goodman line."""
    ratio = sigma_a / sigma_e + sigma_m / sigma_uts
    return 1 / ratio if ratio > 0 else np.inf


def gerber_check(sigma_a, sigma_m, sigma_e, sigma_uts):
    """Return factor of safety on Gerber curve."""
    ratio = sigma_a / sigma_e + (sigma_m / sigma_uts) ** 2
    return 1 / np.sqrt(ratio) if ratio > 0 else np.inf


def paris_law_integration(a_init, a_crit, C, m, Delta_K_func, max_N=1e6, dt=1000):
    """Integrate Paris' law da/dN = C * (Delta K)^m from a_init to a_crit.
    Returns cycles to failure N_f and history arrays."""
    N = 0
    a = a_init
    history_N = [0]
    history_a = [a]
    while a < a_crit and N < max_N:
        Delta_K = Delta_K_func(a)
        da_dN = C * Delta_K ** m
        if da_dN <= 1e-15:
            break
        a += da_dN
        N += dt
        history_N.append(N)
        history_a.append(a)
    return N, np.array(history_N), np.array(history_a)


# ---------------------------------------------------------------------------
# 5. Structural Optimization (Beam Weight Minimization)
# ---------------------------------------------------------------------------

def beam_weight_minimization(L=2.0, F=1000, allowable_stress=250e6, 
                             E=DEFAULT_STEEL_E, rho=DEFAULT_STEEL_RHO,
                             height_range=(0.05, 0.5), width_range=(0.02, 0.2)):
    """Minimize beam weight = rho * A * L = rho * b * h * L
    subject to: sigma_max = F*L/(b*h^2/6) <= allowable_stress (rectangular, simply supported, center load)
    and deflection delta = F*L^3/(48*E*I) <= L/100 (typical serviceability).
    I = b*h^3/12.
    Uses scipy.optimize.minimize with bounds."""
    from scipy.optimize import Bounds, minimize

    def constraints(b_h):
        b, h = b_h
        I = b * h**3 / 12
        A = b * h
        sigma_max = 6 * F * L / (b * h**2)  # max bending stress at supports/center
        delta = F * L**3 / (48 * E * I)
        return [allowable_stress - sigma_max, L/100 - delta]

    def objective(b_h):
        b, h = b_h
        return rho * b * h * L  # weight

    # Initial guess
    b0 = (height_range[1] + height_range[0]) / 4  # rough
    h0 = (width_range[1] + width_range[0]) / 4
    x0 = [b0, h0]

    # Bounds
    bounds = Bounds(width_range[0], width_range[1], [height_range[0], height_range[1]])

    # Constraint dict for scipy
    cons = [{'type': 'ineq', 'fun': lambda x: constraints(x)[0]},
            {'type': 'ineq', 'fun': lambda x: constraints(x)[1]}]

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    b_opt, h_opt = result.x
    weight_opt = rho * b_opt * h_opt * L
    sigma_max = 6 * F * L / (b_opt * h_opt**2)
    delta = F * L**3 / (48 * E * (b_opt * h_opt**3 / 12))
    return {
        'width': b_opt,
        'height': h_opt,
        'cross_section_area': b_opt * h_opt,
        'weight': weight_opt,
        'max_stress': sigma_max,
        'deflection': delta,
        'constraints_satisfied': sigma_max <= allowable_stress and delta <= L/100
    }


# ---------------------------------------------------------------------------
# 6. Failure Theories & Rotordynamics (Simple Jeffcott Rotor)
# ---------------------------------------------------------------------------

class FailureTheories:
    """Compare failure predictions under plane stress."""
    @staticmethod
    def von_mises(sigma_x, sigma_y, tau_xy):
        return von_mises_sigma(sigma_x, sigma_y, tau_xy)

    @staticmethod
    def tresca(sigma_x, sigma_y, tau_xy):
        return max_shear_stress(sigma_x, sigma_y, tau_xy)

    @staticmethod
    mohr_analytical(sigma_x, sigma_y, tau_xy):
        """Maximum Normal Stress Theory (Rankine)."""
        sigma_1, sigma_2 = principal_stresses(sigma_x, sigma_y, tau_xy)
        return max(abs(sigma_1), abs(sigma_2))


def jeffcott_rotor(omega, J, K, C, Omega_crit):
    """Simple Jeffcott rotor: lateral response of a rotor with mass J, shaft stiffness K, damping C.
    Returns amplitude of center of mass vs shaft speed omega."""
    # Simplified: amplitude = J * omega^2 / sqrt((K - J*omega^2)**2 + (C*omega)**2)
    denom = np.sqrt((K - J * omega**2)**2 + (C * omega)**2)
    amplitude = J * omega**2 / denom
    return amplitude


# ---------------------------------------------------------------------------
# Main Demo Block
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("MECHANICAL ENGINEERING TOOLKIT DEMONSTRATION")
    print("=" * 60)

    # --- Truss FEA Demo ---
    print("\n[1] Truss FEA: 3-bar planar truss under load")
    T = TrussFEA(E=200e9, A=5e-4)
    n1 = T.add_node(0, 0, 0)
    n2 = T.add_node(2, 0, 0)
    n3 = T.add_node(1, 1.5, 0)
    T.add_element(n1, n2)
    T.add_element(n2, n3)
    T.add_element(n3, n1)
    T.apply_load(n1, Fx=-10000, Fy=0, Fz=0)
    T.apply_load(n2, Fx=5000, Fy=0, Fz=0)
    T.apply_bc(n3, ux=True, uy=True, uz=True)
    u, r = T.solve()
    print(f"  Displacements (m): ux={u[0]:.6f}, uy={u[3]:.6f}")
    print(f"  Reactions (N): Fx={r[0]:.1f}, Fy={r[3]:.1f}, Fz={r[6]:.1f}")

    # --- Beam FEA Demo ---
    print("\n[2] Beam FEA: Simply supported beam with central load")
    L_beam = 4.0
    n1b = BeamFEA(E=200e9, I=1e-4, A=0.01)
    nb1 = n1b.add_node(0)
    nb2 = n1b.add_node(L_beam)
    n1b.add_element(nb1, nb2)
    n1b.apply_load(nb1, fx=0, fy=-10000, mx=0)
    n1b.apply_bc(nb2, ux=True, theta=True, vy=True)  # fixed end
    u_beam, r_beam = n1b.solve()
    print(f"  Deflections: ux={u_beam[0]:.6f}, theta={u_beam[1]:.6f} rad, vy={u_beam[2]:.6f} m")
    print(f"  Reactions: Fx={r_beam[0]:.1f}, M={r_beam[3]:.1f}, Fy={r_beam[6]:.1f}")

    # --- Stress Transformation & Mohr's Circle ---
    print("\n[3] Stress transformation & Mohr's circle")
    sx, sy, txy = 80e6, -30e6, 50e6  # Pa
    center, rad, s1, s2 = mohr_circle_plot(sx, sy, txy, save_path='mohr_circle.png')
    print(f"  Center: {center/1e6:.2f} MPa, Radius: {rad/1e6:.2f} MPa")
    print(f"  Principal stresses: {s1/1e6:.2f} MPa, {s2/1e6:.2f} MPa")
    print(f"  Von Mises: {von_mises_sigma(sx, sy, txy)/1e6:.2f} MPa")
    print(f"  Tresca (max shear): {max_shear_stress(sx, sy, txy)/1e6:.2f} MPa")

    # --- SDOF Vibrations ---
    print("\n[4] SDOF damped free vibration")
    sdoe_damped_response()

    # --- Fatigue Demo ---
    print("\n[5] Fatigue: Goodman and Gerber checks")
    sigma_a, sigma_m = 150e6, 100e6
    sigma_e, sigma_uts = 250e6, 600e6
    g_factor = goodman_check(sigma_a, sigma_m, sigma_e, sigma_uts)
    ger_factor = gerber_check(sigma_a, sigma_m, sigma_e, sigma_uts)
    print(f"  Goodman FoS: {g_factor:.3f} (>=1 safe)")
    print(f"  Gerber FoS:  {ger_factor:.3f} (>=1 safe)")
    # S-N life
    N_to_fail = s_n_curve_basquin(sigma_a)
    print(f"  Basquin cycles to failure (sigma_a={sigma_a/1e6:.0f} MPa): {N_to_fail:.0e}")

    # --- Structural Optimization ---
    print("\n[6] Beam cross-section weight minimization")
    opt_result = beam_weight_minimization(L=3.0, F=5000, allowable_stress=250e6)
    print(f"  Optimal width (m):   {opt_result['width']:.5f}")
    print(f"  Optimal height (m):  {opt_result['height']:.5f}")
    print(f"  Min weight (kg):     {opt_result['weight']:.2f}")
    print(f"  Max stress (Pa):     {opt_result['max_stress']:.2e}")
    print(f"  Deflection (m):      {opt_result['deflection']:.6f}")

    # --- Failure Theories Comparison ---
    print("\n[7] Failure theories comparison")
    sigma_x, sigma_y, tau_xy = 120e6, 40e6, 60e6
    vm = FailureTheories.von_mises(sigma_x, sigma_y, tau_xy)
    tc = FailureTheories.tresca(sigma_x, sigma_y, tau_xy)
    mn = FailureTheories.mohr_analytical(sigma_x, sigma_y, tau_xy)
    sigma_ult = 600e6
    print(f"  Von Mises:      {vm/1e6:.2f} MPa")
    print(f"  Tresca (shear): {tc/1e6:.2f} MPa")
    print(f"  Max Normal stress (Rankine): {mn/1e6:.2f} MPa")
    print(f"  Material UTS:   {sigma_ult/1e6:.0f} MPa")
    print(f"  Von Mises > UTS? {vm > sigma_ult}")
    print(f"  Tresca > UTS?  {tc > sigma_ult/2}")  # Tresca compares to sigma_uts/2

    # --- Paris Law Demo ---
    print("\n[8] Paris' Law crack growth integration")
    a_init, a_crit = 0.01, 0.1  # m
    C_paris, m_paris = 1e-10, 3.0
    # Delta K for edge crack in tension: Delta K = sigma * sqrt(pi * a) * F; F~1.0
    def Delta_K(a, sigma=100e6):
        return sigma * np.sqrt(np.pi * a)
    N_f, hist_N, hist_a = paris_law_integration(a_init, a_crit, C_paris, m_paris,
                                                 lambda a: Delta_K(a, sigma=100e6),
                                                 max_N=1e6)
    print(f"  Initial crack size: {a_init*1000:.1f} mm")
    print(f"  Critical crack size: {a_crit*1000:.1f} mm")
    print(f"  Cycles to failure: {N_f:.0e}")

    # --- Rotordynamics Demo ---
    print("\n[9] Jeffcott rotor simplified response")
    omega_range = np.linspace(0, 200, 400)  # rad/s
    J, K_shaft, C_shaft = 50, 2e6, 500  # kg, N/m, Ns/m
    amplitudes = [jeffcott_rotor(w, J, K_shaft, C_shaft, 100) for w in omega_range]
    plt.figure()
    plt.plot(omega_range, amplitudes)
    plt.axvline(np.sqrt(K_shaft/J), 'r--', label=f'Critical speed ~{np.sqrt(K_shaft/J):.0f} rad/s')
    plt.xlabel('Shaft speed ω (rad/s)')
    plt.ylabel('Orbital amplitude (m)')
    plt.title('Jeffcott Rotor Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rotor_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Critical speed (theoretical): {np.sqrt(K_shaft/J):.1f} rad/s")

    print("\n" + "=" * 60)
    print("All demonstrations completed. Check saved plots: mohr_circle.png, rotor_response.png")
    print("=" * 60)