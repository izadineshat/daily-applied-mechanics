import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    """2D rotation matrix for angle theta (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

class TrussMember:
    """Axial truss member with nodal coordinates."""
    def __init__(self, node1, node2, E, A, L):
        self.node1 = node1
        self.node2 = node2
        self.E = E
        self.A = A
        self.L = L
        self.theta = np.arctan2(self.node2[1] - self.node1[1],
                                 self.node2[0] - self.node1[0])
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
        self.k = self.E * self.A / self.L  # axial stiffness

    def local_stiffness(self):
        """2×2 local axial stiffness matrix in element coordinates."""
        k = self.k
        return np.array([[k, -k],
                         [-k,  k]])

    def global_stiffness(self, dof_map):
        """
        Return global 4×4 stiffness matrix.
        dof_map: dict {local_dof: global_index}
        where local DOFs are [u_i, u_j] for axial displacement.
        """
        C = self.local_stiffness()
        T = np.array([[ self.cos_t,  self.sin_t, 0, 0],
                      [-self.sin_t,  self.cos_t, 0, 0],
                      [0, 0, self.cos_t,  self.sin_t],
                      [0, 0, -self.sin_t, self.cos_t]])
        return T @ C @ T.T

class BeamElement:
    """2D Euler‑Bernoulli beam element (2 nodes, 2 DOFs each)."""
    def __init__(self, node_i, node_j, E, I, L):
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.I = I
        self.L = L
        self.k = E * I / (L**3)  # bending stiffness factor

    def global_stiffness(self, dof_i_v, dof_i_theta, dof_j_v, dof_j_theta):
        """
        Assemble 8×8 global stiffness matrix.
        DOF ordering: [v_i, θ_i, v_j, θ_j] → global indices.
        """
        factor = self.k
        Ke = factor * np.array([[ 1, -1,  0,  0],
                                [-1,  2,  0,  1],
                                [ 0,  0,  1, -1],
                                [ 0,  1, -1,  2]])
        Kg = np.zeros((8, 8))
        dof_elem = [dof_i_v, dof_i_theta, dof_j_v, dof_j_theta]
        for i in range(4):
            for j in range(4):
                Kg[dof_elem[i], dof_elem[j]] += Ke[i, j]
        return Kg

class StressState:
    """Container for plane stress/strain components."""
    def __init__(self, sigma_x, sigma_y, tau_xy):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.tau_xy = tau_xy

    def principal_stresses(self):
        """Return principal stresses σ1, σ2."""
        avg = (self.sigma_x + self.sigma_y) / 2.0
        radius = np.sqrt(((self.sigma_x - self.sigma_y) / 2.0)**2 + self.tau_xy**2)
        return avg + radius, avg - radius

    def von_mises(self):
        """Von Mises equivalent stress."""
        return np.sqrt(self.sigma_x**2 + self.sigma_y**2 -
                       self.sigma_x * self.sigma_y + 3.0 * self.tau_xy**2)

    def tresca(self):
        """Maximum shear stress (Tresca criterion)."""
        return max(abs(self.sigma_x - self.sigma_y),
                   abs(self.sigma_x + self.sigma_y) / 2.0,
                   abs(self.sigma_y + self.tau_xy) / 2.0,
                   abs(self.sigma_x - self.tau_xy) / 2.0)

def mohr_circle(stress):
    """Return center and radius of Mohr's circle."""
    sigma_x, sigma_y, tau = stress.sigma_x, stress.sigma_y, stress.tau_xy
    center = (sigma_x + sigma_y) / 2.0
    radius = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau**2)
    return center, radius

def sdo_damped_response(m, c, k, x0, v0, t_span, t_eval):
    """Solve m·x¨ + c·x˙ + k·x = 0 for given initial conditions."""
    def ode(t, y):
        x, v = y
        return [v, (-c/m)*v - (k/m)*x]
    sol = solve_ivp(ode, t_span, [x0, v0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0], sol.y[1]

def modal_analysis(K, M):
    """Generalized eigenvalue problem Kφ = ω²Mφ. Returns ω (rad/s) and Φ (mode shapes)."""
    w_sq, Phi = eigh(K, M)
    w = np.sqrt(np.abs(w_sq))  # ensure positive frequencies
    # Normalize modes: φᵀMφ = 1
    for i in range(len(w)):
        Phi[:, i] = Phi[:, i] / np.sqrt(Phi[:, i] @ M @ Phi[:, i])
    return w, Phi

def fatigue_cycles(stress_amp, sigma_f, b=0.08, N_ref=1e6):
    """Simple linear S‑N curve (log‑log)."""
    return N_ref * (sigma_f / stress_amp)**(1.0/b)

def goodman_limit(Sy, sigma_a):
    """Goodman relation for alternating stress limit."""
    return sigma_a * Sy / (Sy - sigma_a) if sigma_a < Sy else np.inf

def paris_law(delta_K, C=1e-6, m=3):
    """Paris law: da/dN = C·(ΔK)^m."""
    return C * (delta_K**m)

def integrate_paris(delta_K, C, m, N0, N_end):
    """Integrate Paris law assuming constant ΔK over cycles N0→N_end."""
    dN = N_end - N0
    return paris_law(delta_K, C, m) * dN

def stress_beam(b, h, M):
    """Bending stress σ = Mc/I for rectangular section."""
    I = b * h**3 / 12.0
    c = h / 2.0
    return M * c / I

def beam_weight(E, L, b, h, rho=7850.0):
    """Mass (kg) of a uniform beam."""
    A = b * h
    return rho * A * L

def optimize_beam_section(E, L, M_max, sigma_allow, rho=7850.0):
    """Minimize weight subject to bending stress constraint."""
    def objective(x):
        b, h = x
        return beam_weight(E, L, b, h, rho)
    def constraint(x):
        b, h = x
        sigma = stress_beam(b, h, M_max)
        return sigma_allow - sigma
    x0 = np.array([0.05, 0.1])  # initial guess (b, h) in meters
    bounds = [(1e-4, None), (1e-4, None)]
    cons = [{'type': 'ineq', 'fun': constraint}]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, res.fun, res.success

def rotor_critical_speed(E, I, L, m, d):
    """Jeffcott rotor: first natural frequency for a simply supported shaft."""
    omega1 = (np.pi**2) * np.sqrt(E * I / (m * L**4))
    return omega1 / (2.0 * np.pi)  # Hz

def main():
    # ----- Truss example -----
    nodeA = (0.0, 0.0)
    nodeB = (1.0, 0.0)
    truss = TrussMember(nodeA, nodeB, E=210e9, A=0.01, L=1.0)
    # DOF map: node i axial displacement → global index 0, node j → 2
    dof_map = {0: 0, 1: 1, 2: 2, 3: 3}
    K_truss = truss.global_stiffness({0:0, 2:2})
    print("Truss member global stiffness matrix (4×4):\n", K_truss)

    # ----- Beam element example -----
    node1 = (0.0, 0.0)
    node2 = (1.0, 0.0)
    b_sec = 0.05   # width (m)
    h_sec = 0.1    # height (m)
    I_val = b_sec * h_sec**3 / 12.0
    beam = BeamElement(node1, node2, E=210e9, I=I_val, L=1.0)
    # Global DOF indices: node_i (v,θ) → 0,1 ; node_j (v,θ) → 2,3
    K_beam = beam.global_stiffness(0, 1, 2, 3)
    print("\nBeam element global stiffness matrix (8×8):\n", K_beam)

    # ----- Stress state and transformations -----
    sigma_state = StressState(sigma_x=120e6, sigma_y=30e6, tau_xy=20e6)  # Pa
    sigma1, sigma2 = sigma_state.principal_stresses()
    vm = sigma_state.von_mises()
    tc = sigma_state.tresca()
    print("\nPrincipal stresses (Pa):", sigma1, sigma2)
    print("Von Mises (Pa):", vm)
    print("Tresca max shear (Pa):", tc)
    center, radius = mohr_circle(sigma_state)
    print("Mohr's circle center (Pa):", center, "radius (Pa):", radius)

    # ----- SDOF damped vibration -----
    m_sdo = 1000.0   # kg
    c_sdo = 20.0     # Ns/m
    k_sdo = 8000.0   # N/m
    t_span = (0, 5)
    t_eval = np.linspace(*t_span, 500)
    t, x, v = sdo_damped_response(m_sdo, c_sdo, k_sdo, x0=0.1, v0=0.0, t_span=t_span, t_eval=t_eval)
    plt.figure()
    plt.plot(t, x, label='Displacement')
    plt.plot(t, v, label='Velocity')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title('SDOF Damped Vibration')
    plt.grid(True)

    # ----- Modal analysis of a simple 2‑DOF system -----
    M_demo = np.diag([m_sdo, m_sdo])
    K_demo = np.array([[k_sdo, -k_sdo],
                       [-k_sdo, k_sdo]])
    w_demo, Phi_demo = modal_analysis(K_demo, M_demo)
    print("\nModal frequencies (rad/s):", w_demo)
    print("Mode shapes (normalized):\n", Phi_demo)

    # ----- Fatigue example -----
    sigma_f = 300e6          # Pa (fatigue strength)
    N_f = fatigue_cycles(vm, sigma_f)
    print("\nEstimated fatigue cycles to failure:", N_f)

    # Goodman limit
    Sy = 250e6               # Pa (yield strength)
    sigma_goodman = goodman_limit(Sy, vm)
    print("Goodman alternating stress limit (Pa):", sigma_goodman)

    # ----- Fracture mechanics (Paris law) -----
    delta_K = 50e6 * np.sqrt(0.01)   # example stress intensity range
    da_dN = paris_law(delta_K)
    N_needed = integrate_paris(delta_K, C=1e-12, m=3, N0=0, N_end=1e6)
    print("Paris law da/dN (m/cycle):", da_dN)
    print("Total crack growth over 1e6 cycles (m):", N_needed)

    # ----- Structural optimization -----
    M_max = 100.0          # N·m bending moment
    sigma_allow = 150e6    # Pa
    b_opt, h_opt = optimize_beam_section(E=210e9, L=1.0, M_max=M_max,
                                         sigma_allow=sigma_allow)
    weight_opt = beam_weight(E=210e9, L=1.0, b=b_opt, h=h_opt)
    print("\nOptimized dimensions - b (m):", b_opt, "h (m):", h_opt)
    print("Minimized weight (kg):", weight_opt)

    # ----- Jeffcott rotor critical speed -----
    d_rotor = 0.05          # m (diameter)
    I_rotor = np.pi * d_rotor**4 / 64.0
    m_rotor = 20.0
    L_rotor = 0.5
    f_crit = rotor_critical_speed(E=210e9, I=I_rotor, L=L_rotor,
                                  m=m_rotor, d=d_rotor)
    print("\nJeffcott rotor critical frequency (Hz):", f_crit)

    plt.show()

if __name__ == '__main__':
    main()