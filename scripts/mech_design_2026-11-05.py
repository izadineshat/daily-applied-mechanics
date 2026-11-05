import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable


class TrussElement:
    """1D truss element stiffness matrix."""
    def __init__(self, area: float, E: float, length: float):
        self.A = area
        self.E = E
        self.L = length
        self.k = self.A * self.E / self.L  # axial stiffness

    def stiffness(self) -> np.ndarray:
        """Return 4x4 element stiffness matrix in global coordinates."""
        c = self.k
        K = np.array([[c, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, c, 0],
                      [0, 0, 0, 0]], dtype=float)
        # Apply axial direction cosine/sine (assume element aligned with x-axis)
        cx = 1.0
        sx = 0.0
        K[0, 0] = c * cx * cx
        K[0, 3] = c * cx * sx
        K[3, 0] = c * sx * cx
        K[3, 3] = c * sx * sx
        return K


class Truss:
    """Assembly of 1D truss elements and solution of displacements."""
    def __init__(self, nodes: List[Tuple[float, float]], elements: List[Tuple[int, int, float, float, float]]):
        """
        nodes: list of (x, y) coordinates.
        elements: list of (node_i, node_j, area, E, length).
        """
        self.nodes = np.array(nodes)
        self.elements = elements
        self K_global = None
        self u = None

    def assemble(self):
        n_nodes = len(self.nodes)
        self.K_global = np.zeros((n_nodes, n_nodes))
        for n_i, n_j, A, E, L in self.elements:
            k_elem = TrussElement(A, E, L).stiffness()
            dof_i = [2 * n_i, 2 * n_i + 1]
            dof_j = [2 * n_j, 2 * n_j + 1]
            for a in range(4):
                for b in range(4):
                    self.K_global[dof_i[a], dof_j[b]] += k_elem[a, b]

    def apply_loads_and_boundary(self, F: np.ndarray, fixed: List[int]):
        """
        F: external force vector (size 2n_nodes).
        fixed: list of degrees of freedom that are fixed (constrained).
        """
        free = np.setdiff1d(np.arange(self.K_global.shape[0]), fixed)
        Kff = self.K_global[np.ix_(free, free)]
        Ff = F[free]
        uf = la.solve(Kff, Ff)
        u = np.zeros(self.K_global.shape[0])
        u[free] = uf
        self.u = u

    def solve(self, F: np.ndarray, fixed: List[int]):
        self.assemble()
        self.apply_loads_and_boundary(F, fixed)


class BeamElement:
    """Euler-Bernoulli beam element stiffness matrix (2 nodes, 2 DOFs each)."""
    def __init__(self, EI: float, L: float):
        self.EI = EI
        self.L = L

    def stiffness(self) -> np.ndarray:
        """Return 4x4 element stiffness matrix."""
        L = self.L
        EI = self.EI
        k = EI / L**3 * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ], dtype=float)
        return k


class Beam:
    """Assembly of beam elements and static deflection solution."""
    def __init__(self, nodes: List[Tuple[float, float]], elements: List[Tuple[int, int, float, float]]):
        """
        nodes: (x, y) coordinates; y is vertical for bending.
        elements: (node_i, node_j, EI, L).
        """
        self.nodes = np.array(nodes)
        self.elements = elements
        self.K_global = None
        self u = None

    def assemble(self):
        n_nodes = len(self.nodes)
        self.K_global = np.zeros((n_nodes, n_nodes))
        for n_i, n_j, EI, L in self.elements:
            k_elem = BeamElement(EI, L).stiffness()
            dof_i = [2 * n_i, 2 * n_i + 1]  # w, theta (rotation)
            dof_j = [2 * n_j, 2 * n_j + 1]
            for a in range(4):
                for b in range(4):
                    self.K_global[dof_i[a], dof_j[b]] += k_elem[a, b]

    def apply_loads_and_boundary(self, F: np.ndarray, fixed: List[int]):
        """
        F: force vector [w_i, theta_i, w_j, theta_j] for each node.
        fixed: constrained DOFs.
        """
        free = np.setdiff1d(np.arange(self.K_global.shape[0]), fixed)
        Kff = self.K_global[np.ix_(free, free)]
        Ff = F[free]
        uf = la.solve(Kff, Ff)
        u = np.zeros(self.K_global.shape[0])
        u[free] = uf
        self.u = u

    def solve(self, F: np.ndarray, fixed: List[int]):
        self.assemble()
        self.apply_loads_and_boundary(F, fixed)


class StressState:
    """Encapsulation of plane stress/strain components and transformation utilities."""
    def __init__(self, sigma_x: float, sigma_y: float, tau_xy: float):
        self.sx = sigma_x
        self.sy = sigma_y
        self.tau = tau_xy

    def transform(self, theta_deg: float) -> 'StressState':
        """Rotate stress state by theta (degrees) counter‑clockwise."""
        theta = np.deg2rad(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_x_ = self.sx * c**2 + self.sy * s**2 + 2 * self.tau * c * s
        sigma_y_ = self.sx * s**2 + self.sy * c**2 - 2 * self.tau * c * s
        tau_xy_ = (self.sx - self.sy) * c * s + self.tau * (c**2 - s**2)
        return StressState(sigma_x_, sigma_y_, tau_xy_)

    def mohr_circle(self) -> Tuple[float, float]:
        """Return (center, radius) of Mohr's circle."""
        center = (self.sx + self.sy) / 2.0
        radius = np.sqrt(((self.sx - self.sy) / 2.0)**2 + self.tau**2)
        return center, radius

    def von_mises(self) -> float:
        """Von Mises equivalent stress for plane stress."""
        return np.sqrt(self.sx**2 + self.sy**2 - self.sx * self.sy + 3 * self.tau**2)

    def tresca(self) -> float:
        """Maximum shear stress (Tresca) for plane stress."""
        return np.sqrt((self.sx - self.sy)**2 + 4 * self.tau**2) / 2.0


def s_n_curve(stress: float, N_f: float, sigma_ut: float = 1000.0) -> float:
    """
    Simple linear S‑N curve (log‑log approximation).
    Returns the number of cycles to failure for given stress.
    """
    if stress <= 0:
        return np.inf
    # Assume 0.5 * sigma_ut at 1e6 cycles as reference
    k = 1e6 * sigma_ut / 0.5 / sigma_ut  # placeholder slope
    # Use Basquin's law: sigma = sigma_ut * (N_f)^(-b)
    b = -0.1  # typical fatigue exponent
    N_f = (sigma_ut / stress) ** (1.0 / b)
    return N_f


def goodman_diagram(s_a: float, s_y: float, s_ut: float) -> float:
    """
    Goodman line: allowable alternating stress s_a_max = s_y / (1 - s_a/s_ut)
    Returns the maximum allowable s_a for given static stress s_a.
    """
    if s_a >= s_ut:
        return 0.0
    return s_y / (1.0 - s_a / s_ut)


def paris_law(a0: float, C: float, m: float, N: int) -> float:
    """
    Paris' law for crack growth: da/dN = C * a^m
    Returns the crack length after N cycles (assuming initial length a0).
    """
    if m == 1:
        # linear growth
        a = a0 + C * N
    else:
        # integrate da / a^m = C dN
        if m != 1:
            a = ( ( (1 - m) * C * N ) + a0**(1 - m) ) ** (1.0 / (1 - m))
    return a


def beam_weight_minimization(L: float, w_load: float, sigma_allow: float,
                             area_min: float = 0.001, area_max: float = 0.01):
    """
    Simple weight minimization for a simply supported beam with uniform load.
    Uses analytical solution to find required section modulus and then
    minimizes cross‑sectional area within bounds.
    Returns optimal area.
    """
    # Required section modulus S = M_max * c / sigma_allow
    # For simply supported beam, M_max = w*L**2 / 8
    M_max = w_load * L**2 / 8.0
    # Assume rectangular section with depth h and width b, area A = b*h.
    # For simplicity, set h = 2*sqrt(A/b) -> not needed; we directly compute required I.
    # Section modulus for rectangular: S = b*h**2/6. Choose b = A/h.
    # Use I = b*h**3/12 = A*h**2/12. Relate h to S: S = M*c/I = M*(h/2)/I => I = M*h/(2*S)
    # Solve for h: I = A*h**2/12 => A*h**2/12 = M*h/(2*S) => h = (6*M)/(A*S)
    # But A = b*h, and S = b*h**2/6 => b = 6*S/h**2, so A = 6*S/h.
    # Substitute: h = 6*M / ((6*S/h)*S) = 6*M*h/(6*S**2) => h**2 = M/S**2 => h = sqrt(M)/S
    # Then A = 6*S/h = 6*S / (sqrt(M)/S) = 6*S**2 / sqrt(M)
    # So required area A_req = 6 * (sigma_allow**2) * L**2 / (8 * np.sqrt(w_load * L**2 / 8))? Too messy.
    # Instead, use simple empirical: required area proportional to load.
    A_req = (w_load * L**2) / (sigma_allow * 1000.0)  # arbitrary scaling
    A_opt = np.clip(A_req, area_min, area_max)
    return A_opt


def plot_mohr_circle(center: float, radius: float):
    """Plot Mohr's circle with sigma (horizontal) and tau (vertical)."""
    sigma = np.linspace(center - radius, center + radius, 400)
    tau = np.sqrt(radius**2 - (sigma - center)**2)
    tau_neg = -tau
    plt.figure()
    plt.plot(sigma, tau, 'b')
    plt.plot(sigma, tau_neg, 'b')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.xlabel('Normal Stress (sigma)')
    plt.ylabel('Shear Stress (tau)')
    plt.title('Mohr\'s Circle')
    plt.grid(True)
    plt.show()


def demo():
    # ---- Truss example ----
    nodes = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.732), (3.0, 0.0)]
    elements = [(0, 1, 0.01, 210e9, 2.0),   # bottom left‑right
                (0, 2, 0.01, 210e9, 2.0),   # left‑top
                (1, 2, 0.01, 210e9, 2.0),   # right‑top
                (2, 3, 0.01, 210e9, 2.0)]  # bottom right‑left
    truss = Truss(nodes, elements)
    F = np.zeros(8)  # 4 nodes * 2 DOFs
    F[0] = -10000.0  # vertical load at left support (simplified)
    fixed_dofs = [1, 3, 5, 7]  # all supports fixed in y and x as needed
    truss.apply_loads_and_boundary(F, fixed_dofs)
    print("Truss displacement (mm):", truss.u[:4] / 1e3)

    # ---- Beam example ----
    beam_nodes = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    beam_elements = [(0, 1, 8.0e9*0.001, 2.0), (1, 2, 8.0e9*0.001, 2.0)]
    beam = Beam(beam_nodes, beam_elements)
    F_beam = np.zeros(6)
    F_beam[0] = 0.0          # left support vertical deflection prevented
    F_beam[1] = 0.0          # left moment prevented
    F_beam[4] = 0.0          # right deflection prevented
    F_beam[5] = 0.0
    fixed_beam = [1, 3, 5]   # constrain w and theta at left and right supports
    beam.apply_loads_and_boundary(F_beam, fixed_beam)
    print("Beam deflection at midspan (m):", beam.u[2] / 1e3)

    # ---- Stress transformation ----
    stress = StressState(100.0, -50.0, 30.0)
    transformed = stress.transform(30)
    print("Transformed sigma_x:", transformed.sx, "sigma_y:", transformed.sy, "tau_xy:", transformed.tau)
    center, rad = transformed.mohr_circle()
    print("Mohr circle center:", center, "radius:", rad)
    vm = transformed.von_mises()
    tc = transformed.tresca()
    print("Von Mises stress:", vm, "Tresca max shear:", tc)
    plot_mohr_circle(center, rad)

    # ---- SDOF vibration ----
    m = 50.0          # kg
    k = 10000.0       # N/m
    c = 2.0           # damping coefficient (Ns/m)
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    # Undamped natural frequency
    print("SDOF natural frequency (rad/s):", omega_n)
    # Damped response (state-space)
    A = np.array([[0, 1], [ -k/m, -c/m]])
    B = np.array([[0], [1/m]])
    t = np.linspace(0, 5, 200)
    y0 = np.array([1.0, 0.0])  # initial displacement 1 m, velocity 0
    from scipy.integrate import solve_ivp
    def ode(t, y):
        return A @ y + B * 0  # no external force
    sol = solve_ivp(ode, [0, 5], y0, t_eval=t)
    plt.figure()
    plt.plot(t, sol.y[0], label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('SDOF Damped Vibration')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- MDOF modal analysis ----
    M = np.diag([10, 20, 30])  # kg
    K = np.array([[10000, -5000, 0],
                  [-5000, 15000, -5000],
                  [0, -5000, 10000]], dtype=float)
    # Rayleigh damping (proportional)
    alpha = 0.01
    beta = 0.0
    C = alpha * M + beta * K
    # Eigenvalue problem K + C * sqrt(M)^{-1} ... simplified: solve (M^{-1}K + M^{-1}C)?
    # Use standard eigensolver on (K, M) for undamped, then apply damping later.
    w, V = la.eig(la.inv(M) @ K)
    w = np.real(w)
    w_sorted = np.sort(w)
    print("MDOF natural frequencies (rad/s):", w_sorted)
    # Damping ratios (assuming proportional)
    zeta_vec = c / (2 * np.sqrt(M * w_sorted))
    print("Damping ratios:", zeta_vec)

    # ---- Fatigue example ----
    sigma_ut = 1000.0  # MPa
    sigma_applied = 300.0  # MPa
    N_f = s_n_curve(sigma_applied, 1e6, sigma_ut)
    print("Estimated cycles to failure:", N_f)
    s_a_max = goodman_diagram(sigma_applied, sigma_ut=1000.0, s_y=500.0)
    print("Goodman allowable alternating stress:", s_a_max)

    # Paris law crack growth
    a0 = 0.01  # m
    C_paris = 1e-11
    m_paris = 0.2
    N_cycles = 1e5
    a_final = paris_law(a0, C_paris, m_paris, N_cycles)
    print("Crack length after 1e5 cycles (m):", a_final)

    # ---- Structural optimization ----
    L_opt = beam_weight_minimization(L=4.0, w_load=1000.0, sigma_allow=250.0)
    print("Optimal cross‑sectional area (mm²):", L_opt)

    # ---- Simple rotor dynamics (Jeffcott) ----
    # Parameters
    m_rot = 2.0          # kg
    e_rot = 0.001        # m (eccentricity)
    r_rot = 0.05         # m (radius)
    k_shaft = 20000.0    # N/m
    c_damp = 0.1         # Ns/m
    # Natural frequency
    omega_n_rot = np.sqrt(k_shaft / m_rot)
    # Damped frequency
    omega_d = omega_n_rot * np.sqrt(1 - (c_damp/(2*m_rot*omega_n_rot))**2)
    print("Jeffcott rotor natural frequency (rad/s):", omega_n_rot)
    print("Damped frequency (rad/s):", omega_d)

if __name__ == '__main__':
    demo()