import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt

class TrussMember:
    """
    Represents a 2D truss member connecting two nodes.
    """
    def __init__(self, node_i, node_j, area, E):
        """
        Parameters
        ----------
        node_i, node_j : tuple of (x, y) coordinates
        area : float, cross-sectional area (m^2)
        E : float, Young's modulus (Pa)
        """
        self.node_i = np.array(node_i, dtype=float)
        self.node_j = np.array(node_j, dtype=float)
        self.area = float(area)
        self.E = float(E)

    def length(self):
        return np.linalg.norm(self.node_j - self.node_i)

    def direction_cosine(self):
        L = self.length()
        return (self.node_j - self.node_i) / L

    def local_stiffness(self):
        """
        Returns the 4x4 local stiffness matrix for an axial truss member.
        """
        L = self.length()
        cos_theta = self.direction_cosine()
        k = self.E * self.area / L
        c = np.outer(cos_theta, cos_theta)
        return k * c

class Truss:
    """
    Assembles a 2D truss from a list of TrussMember objects.
    """
    def __init__(self, members, nodes):
        """
        Parameters
        ----------
        members : list of TrussMember
        nodes : dict mapping node index to (x, y) coordinates
        """
        self.members = members
        self.nodes = nodes
        self.nnodes = max(nodes.keys()) + 1
        self.global_K = None

    def assemble(self):
        """Assemble global stiffness matrix."""
        K = np.zeros((self.nnodes * 2, self.nnodes * 2))
        for member in self.members:
            L = member.length()
            c = member.direction_cosine()
            k_local = member.local_stiffness()
            # assemble into global matrix
            for i in range(2):
                for j in range(2):
                    row = member.node_i * 2 + i
                    col = member.node_j * 2 + j
                    K[row, col] += k_local[i, j]
                    K[col, row] += k_local[i, j]
        self.global_K = K
        return K

    def apply_loads(self, force_vector):
        """
        Apply external nodal forces.
        force_vector: list of (fx, fy) for each node (length = nnodes)
        """
        n_nodes = self.nnodes
        F = np.zeros(n_nodes * 2)
        for i, (fx, fy) in enumerate(force_vector):
            F[i * 2] = fx
            F[i * 2 + 1] = fy
        return F

    def solve(self, force_vector):
        """Solve K * u = F and return displacements."""
        K = self.assemble()
        F = self.apply_loads(force_vector)
        u = la.solve(K, F)
        return u.reshape(-1, 2)

class BeamElement:
    """
    Euler-Bernoulli beam element (2 nodes, each with 2 transverse DOFs).
    """
    def __init__(self, node_i, node_j, area, E, I, L=None):
        """
        Parameters
        ----------
        node_i, node_j : tuple of (x, y) coordinates
        area : float, cross-sectional area
        E : float, Young's modulus
        I : float, second moment of area about neutral axis
        L : float, element length (if None, computed from coordinates)
        """
        self.node_i = np.array(node_i, dtype=float)
        self.node_j = np.array(node_j, dtype=float)
        self.area = float(area)
        self.E = float(E)
        self.I = float(I)
        if L is None:
            self.L = self.length()
        else:
            self.L = float(L)

    def length(self):
        return np.linalg.norm(self.node_j - self.node_i)

    def local_stiffness(self):
        """
        Returns the 4x4 local stiffness matrix for a beam element (transverse bending only).
        """
        L = self.L
        EI = self.E * self.I
        k = (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

class PlateElement:
    """
    Simplified 2D plate bending element (Kirchhoff plate theory).
    For demonstration, we use a 4x4 matrix representing simply supported square plate.
    """
    def __init__(self, side_length, D):
        """
        Parameters
        ----------
        side_length : float, side length of square plate (m)
        D : float, flexural rigidity (E*I/(1-nu^2))
        """
        self.L = float(side_length)
        self.D = float(D)

    def stiffness(self):
        """
        Returns the 4x4 plate bending stiffness matrix (simplified).
        """
        # For a simply supported square plate, analytical stiffness is:
        # k = D * pi^4 / L^4 * [[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]]
        factor = self.D * np.pi**4 / self.L**4
        return factor * np.array([
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, -1, 0, 1]
        ])

class StressState:
    """
    Represents a 2D stress state (sigma_x, sigma_y, tau_xy).
    """
    def __init__(self, sigma_x, sigma_y, tau_xy):
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.tau_xy = float(tau_xy)

    def transform(self, theta):
        """
        Rotate stress state by angle theta (radians) using transformation equations.
        Returns new StressState.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        sigma_x_new = self.sigma_x * c**2 + self.sigma_y * s**2 + 2 * self.tau_xy * s * c
        sigma_y_new = self.sigma_x * s**2 + self.sigma_y * c**2 - 2 * self.tau_xy * s * c
        tau_xy_new = (self.sigma_y - self.sigma_x) * s * c + self.tau_xy * (c**2 - s**2)
        return StressState(sigma_x_new, sigma_y_new, tau_xy_new)

    def mohr_circle(self):
        """
        Returns center and radius of Mohr's circle.
        """
        center = (self.sigma_x + self.sigma_y) / 2.0
        radius = np.sqrt(((self.sigma_x - self.sigma_y) / 2.0)**2 + self.tau_xy**2)
        return center, radius

    def von_mises(self):
        """Von Mises equivalent stress."""
        return np.sqrt(self.sigma_x**2 + self.sigma_y**2 - self.sigma_x * self.sigma_y + 3 * self.tau_xy**2)

    def tresca(self):
        """Maximum shear stress (Tresca)."""
        return np.abs(self.sigma_x - self.sigma_y) / 2.0 + np.abs(self.tau_xy)

class VibrationSDOF:
    """
    Simple damped SDOF system (mass-spring-damper).
    """
    def __init__(self, m, c, k, x0=0.0, v0=0.0):
        """
        Parameters
        ----------
        m : float, mass (kg)
        c : float, damping coefficient (Ns/m)
        k : float, stiffness (N/m)
        x0 : float, initial displacement (m)
        v0 : float, initial velocity (m/s)
        """
        self.m = m
        self.c = c
        self.k = k
        self.x0 = x0
        self.v0 = v0

    def equations(self, t, y):
        """State vector derivative: y = [x, v]"""
        x, v = y
        dxdt = v
        dvdt = (-self.c / self.m) * v + (-self.k / self.m) * x
        return [dxdt, dvdt]

    def solve(self, t_span=(0, 10), n_points=1000):
        """Numerically integrate and return time, displacement, velocity."""
        sol = integrate.solve_ivp(self.equations, t_span, [self.x0, self.v0], t_eval=np.linspace(*t_span, n_points))
        return sol.t, sol.y[0], sol.y[1]

class VibrationMDOF:
    """
    Linear MDOF system with proportional damping.
    """
    def __init__(self, M, C, K, x0=None, v0=None):
        """
        Parameters
        ----------
        M, C, K : numpy arrays (n,n)
        x0, v0 : initial displacement and velocity vectors (optional)
        """
        self.M = M
        self.C = C
        self.K = K
        n = M.shape[0]
        if x0 is None:
            x0 = np.zeros(n)
        if v0 is None:
            v0 = np.zeros(n)
        self.x0 = x0
        self.v0 = v0

    def state_space(self, t):
        """Return state derivative function for scipy.integrate."""
        def deriv(t, y):
            x = y[:self.M.shape[0]]
            v = y[self.M.shape[0]:]
            dxdt = v
            dvdt = la.solve(self.M, la.solve(self.M, self.C @ v + self.K @ x))
            return np.concatenate([dxdt, dvdt])
        return deriv

    def solve(self, t_span=(0, 5), n_points=500):
        deriv = self.state_space(0)
        sol = integrate.solve_ivp(deriv, t_span, np.concatenate([self.x0, self.v0]), t_eval=np.linspace(*t_span, n_points))
        return sol.t, sol.y[:self.M.shape[0]], sol.y[self.M.shape[0]:]

class Fatigue:
    """
    Fatigue life estimation using S-N curve and Paris' law for crack growth.
    """
    def __init__(self, sigma_amp, N_cycles_ref, b=0.08, c=10**12, da_dN_ref=1e-6):
        """
        Parameters
        ----------
        sigma_amp : float, stress amplitude (Pa)
        N_cycles_ref : float, reference cycles to failure at sigma_ref (Pa)
        b, c : material constants for Basquin's law (sigma = sigma_ref * (N/N_ref)^b)
        da_dN_ref : float, reference crack growth rate (m/cycle) at reference stress intensity
        """
        self.sigma_amp = sigma_amp
        self.N_cycles_ref = N_cycles_ref
        self.b = b
        self.c = c
        self.da_dN_ref = da_dN_ref

    def cycles_to_failure(self):
        """Compute cycles to failure using Basquin's law."""
        # Basquin: sigma_amp = sigma_f' * (2N_f)^b  => N_f = (sigma_f' / sigma_amp)^(1/b) / 2
        # We approximate sigma_f' using reference point
        Nf = (self.N_cycles_ref) * (self.sigma_amp / self.N_cycles_ref)**(1.0 / self.b) / 2.0
        return Nf

    def crack_growth_life(self, Delta_K):
        """
        Paris' law: da/dN = C * (Delta_K)^m (m ~ 3 for steel)
        For simplicity, assume m=3 and C = da_dN_ref * (Delta_K_ref)^-3.
        """
        # Assume reference Delta_K_ref = 1e5 Pa*sqrt(m)
        Delta_K_ref = 1e5
        C = self.da_dN_ref * (Delta_K_ref**3)
        # Integrate da/C = (Delta_K)^3 dN => N = (a_final - a_initial) / (C * Delta_K^3)
        # Assume initial crack length a0=0.001 m, final a=0.01 m
        a0 = 0.001
        a_final = 0.01
        N_crack = (a_final - a0) / (C * (Delta_K**3))
        return N_crack

class StructuralOptimization:
    """
    Beam cross-section weight minimization with stress constraint.
    """
    def __init__(self, length, material, allowable_stress, density):
        """
        Parameters
        ----------
        length : float, beam length (m)
        material : dict with 'E' and 'nu'
        allowable_stress : float, max stress (Pa)
        density : float, material density (kg/m^3)
        """
        self.L = length
        self.E = material['E']
        self.nu = material['nu']
        self.allowable_stress = allowable_stress
        self.density = density

    def section_properties(self, b, h):
        """Rectangular section: area = b*h, I = b*h**3/12, section modulus Z = b*h**2/6."""
        A = b * h
        I = b * h**3 / 12.0
        Z = b * h**2 / 6.0
        return A, I, Z

    def stress_max(self, b, h):
        """
        Maximum bending stress for simply supported beam with central point load P.
        P = 1 N for normalization.
        """
        _, I, Z = self.section_properties(b, h)
        sigma = 1.0 * self.L / (4.0 * Z)  # Mmax = P*L/4, sigma = M/Z
        return sigma

    def objective(self, vars):
        """Objective: minimize weight = density * area * length."""
        b, h = vars[0], vars[1]
        if b <= 0 or h <= 0:
            return 1e6
        A, _, _ = self.section_properties(b, h)
        weight = self.density * A * self.L
        return weight

    def constraint_stress(self, vars):
        """Constraint: max stress <= allowable."""
        b, h = vars[0], vars[1]
        sigma = self.stress_max(b, h)
        return self.allowable_stress - sigma

    def solve(self):
        """Run optimization."""
        # Initial guess
        b0, h0 = 0.05, 0.1
        x0 = np.array([b0, h0])
        bounds = [(0.01, 0.2), (0.01, 0.2)]  # reasonable bounds
        cons = [{'type': 'ineq', 'fun': self.constraint_stress}]
        res = opt.minimize(self.objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        return res

def plot_truss(u):
    """Plot truss displacements."""
    plt.figure()
    for i in range(u.shape[0]):
        x, y = u[i]
        plt.plot([x], [y], 'bo')
    plt.title('Truss Nodal Displacements')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_beam_deflection(t, x, v):
    """Plot beam deflection over time."""
    plt.figure()
    plt.plot(t, x, label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Deflection (m)')
    plt.title('SDOF Damped Vibration')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stress_mohr(sigma_state):
    """Plot Mohr's circle."""
    center, radius = sigma_state.mohr_circle()
    theta = np.linspace(0, np.pi, 200)
    sigma_n = center + radius * np.cos(theta)
    tau_n = radius * np.sin(theta)
    plt.figure()
    plt.plot(sigma_n, tau_n, 'b-')
    plt.plot(center, 0, 'ro')
    plt.xlabel('Normal Stress (Pa)')
    plt.ylabel('Shear Stress (Pa)')
    plt.title('Mohr\'s Circle')
    plt.grid(True)
    plt.show()

def plot_fatigue_life(N_cycles):
    """Simple bar chart of fatigue life."""
    plt.figure()
    plt.bar(['Fatigue Life'], [N_cycles], color='orange')
    plt.ylabel('Cycles')
    plt.title('Fatigue Life Estimate')
    plt.grid(axis='y')
    plt.show()

def plot_optimization_result(res):
    """Plot optimized cross-section dimensions."""
    plt.figure()
    b_opt, h_opt = res.x
    plt.bar(['Optimized'], [b_opt, h_opt], tick_label=['b (m)', 'h (m)'], color='green')
    plt.title('Optimized Beam Cross-Section')
    plt.show()

def main():
    # ----- Truss Example -----
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.5), 3: (0.0, 0.5)}
    members = [
        TrussMember((0,0), (1,0), 0.01, 210e9),
        TrussMember((1,0), (1,0.5), 0.01, 210e9),
        TrussMember((1,0.5), (0,0.5), 0.01, 210e9),
        TrussMember((0,0.5), (0,0), 0.01, 210e9)
    ]
    truss = Truss(members, nodes)
    force_vec = [(0,0), (0,0), (-1000,0), (0,0)]  # -1000 N at node 2
    u = truss.solve(force_vec)
    plot_truss(u)

    # ----- Beam Example -----
    beam = BeamElement((0,0), (1.0,0), area=0.005, E=210e9, I=1e-6)  # rectangular b=0.05, h=0.02
    beam_K = beam.local_stiffness()
    print("Beam local stiffness matrix:\n", beam_K)

    # ----- Stress Transformation Example -----
    stress = StressState(sigma_x=150e6, sigma_y=50e6, tau_xy=20e6)
    sigma_rot = stress.transform(np.radians(30))
    print(f"Rotated stress: sigma_x={sigma_rot.sigma_x:.2e}, sigma_y={sigma_rot.sigma_y:.2e}, tau_xy={sigma_rot.tau_xy:.2e}")
    center, radius = stress.mohr_circle()
    print(f"Mohr's circle center={center:.2e}, radius={radius:.2e}")
    print(f"Von Mises stress={stress.von_mises():.2e} Pa")
    print(f"Tresca stress={stress.tresca():.2e} Pa")
    plot_stress_mohr(stress)

    # ----- Vibration SDOF Example -----
    sdof = VibrationSDOF(m=1.0, c=0.1, k=100.0, x0=0.1, v0=0.0)
    t, x, v = sdof.solve(t_span=(0, 10), n_points=500)
    plot_beam_deflection(t, x, v)

    # ----- MDOF Vibration Example -----
    M = np.array([[1.0, 0.0], [0.0, 1.5]])
    C = np.array([[0.05, 0.0], [0.0, 0.07]])
    K = np.array([[200.0, -30.0], [-30.0, 150.0]])
    mdof = VibrationMDOF(M, C, K, x0=np.array([0.05, 0.0]), v0=np.array([0.0, 0.0]))
    t_m, x_m, v_m = mdof.solve(t_span=(0, 5), n_points=300)
    # Simple plot of first DOF displacement
    plt.figure()
    plt.plot(t_m, x_m[:,0])
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('MDOF System - First DOF')
    plt.grid(True)
    plt.show()

    # ----- Fatigue Example -----
    fatigue = Fatigue(sigma_amp=150e6, N_cycles_ref=1e6)
    N_f = fatigue.cycles_to_failure()
    print(f"Cycles to failure (Basquin): {N_f:.2e}")
    Delta_K = 1e5  # Pa*sqrt(m)
    N_crack = fatigue.crack_growth_life(Delta_K)
    print(f"Crack growth life (Paris): {N_crack:.2e}")
    plot_fatigue_life(N_crack)

    # ----- Structural Optimization Example -----
    opt_problem = StructuralOptimization(length=1.0, material={'E': 210e9, 'nu': 0.3},
                                         allowable_stress=250e6, density=7850)
    res = opt_problem.solve()
    print("Optimization result:", res.x, "Weight:", res.fun)
    plot_optimization_result(res)

if __name__ == '__main__':
    main()