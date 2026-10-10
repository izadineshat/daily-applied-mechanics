import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

# ======================
# Finite Element Analysis
# ======================
class TrussFE:
    """Simple 2D truss FEM (linear elastic)."""
    def __init__(self):
        self.nodes = {}      # node_id: [x, y]
        self.elements = []   # list of (node_i, node_j, E, A)
        self.dofs = {}       # node_id: [dof_x, dof_y]
        self.K = None
        self.F = None
        self.U = None
        self.next_dof = 0

    def add_node(self, nid, x, y):
        self.nodes[nid] = np.array([x, y], dtype=float)
        self.dofs[nid] = [self.next_dof, self.next_dof+1]
        self.next_dof += 2

    def add_element(self, nid_i, nid_j, E, A):
        self.elements.append((nid_i, nid_j, E, A))

    def _element_stiffness(self, xi, xj, E, A):
        dx = xj - xi
        L = np.linalg.norm(dx)
        if L == 0:
            raise ValueError("Zero length element")
        c = dx / L
        k_local = E * A / L * np.array([[c[0]*c[0], c[0]*c[1], -c[0]*c[0], -c[0]*c[1]],
                                        [c[0]*c[1], c[1]*c[1], -c[0]*c[1], -c[1]*c[1]],
                                        [-c[0]*c[0], -c[0]*c[1], c[0]*c[0], c[0]*c[1]],
                                        [-c[0]*c[1], -c[1]*c[1], c[0]*c[1], c[1]*c[1]]])
        # map to global dofs
        dofs = self.dofs[self.elements[-1][0]] + self.dofs[self.elements[-1][1]]
        return k_local, dofs

    def assemble_stiffness(self):
        ndof = self.next_dof
        K = np.zeros((ndof, ndof))
        for (ni, nj, E, A) in self.elements:
            xi = self.nodes[ni]
            xj = self.nodes[nj]
            k_local, dofs = self._element_stiffness(xi, xj, E, A)
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += k_local[i, j]
        self.K = K
        return K

    def apply_force(self, fid, fx, fy):
        if self.F is None:
            ndof = self.next_dof
            self.F = np.zeros(ndof)
        dofs = self.dofs[fid]
        self.F[dofs[0]] += fx
        self.F[dofs[1]] += fy

    def solve(self):
        if self.K is None:
            self.assemble_stiffness()
        # apply simple boundary conditions: fix first two DOFs (node 0)
        fixed = [0, 1]
        free = np.setdiff1d(np.arange(self.next_dof), fixed)
        Kff = self.K[np.ix_(free, free)]
        Fc = self.F[free] - self.K[np.ix_(free, fixed)] @ np.zeros(len(fixed))
        Ufree = la.solve(Kff, Fc)
        U = np.zeros(self.next_dof)
        U[free] = Ufree
        self.U = U
        return U

# ======================
# Beam Element (2D Euler-Bernoulli)
# ======================
def beam_stiffness_2d(E, I, L):
    """Return 4x4 stiffness matrix for a 2-node beam element (vertical DOF & rotation)."""
    k = E * I / L**3 * np.array([[12, 6*L, -12, 6*L],
                                 [6*L, 4*L**2, -6*L, 2*L**2],
                                 [-12, -6*L, 12, -6*L],
                                 [6*L, 2*L**2, -6*L, 4*L**2]])
    return k

# ======================
# Stress & Strain Transformations
# ======================
def principal_stresses(sx, sy, txy):
    """Return principal stresses (s1, s2) and max shear stress."""
    avg = (sx + sy) / 2
    R = np.sqrt(((sx - sy) / 2)**2 + txy**2)
    s1 = avg + R
    s2 = avg - R
    tau_max = R
    return s1, s2, tau_max

def von_mises_stress(sx, sy, txy):
    """Von Mises equivalent stress for plane stress."""
    return np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)

def tresca_stress(sx, sy, txy):
    """Tresca equivalent stress (max shear*2)."""
    _, _, tau_max = principal_stresses(sx, sy, txy)
    return 2 * tau_max

def mohr_circle(sx, sy, txy):
    """Return center, radius, and principal stresses for Mohr's circle."""
    C = (sx + sy) / 2
    R = np.sqrt(((sx - sy) / 2)**2 + txy**2)
    s1 = C + R
    s2 = C - R
    return C, R, s1, s2

def plot_mohr_circle(sx, sy, txy, ax=None):
    C, R, s1, s2 = mohr_circle(sx, sy, txy)
    if ax is None:
        fig, ax = plt.subplots()
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(C + R*np.cos(theta), R*np.sin(theta), 'b')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(C, color='k', linewidth=0.5)
    ax.plot([s1, s2], [0, 0], 'ro')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Normal Stress')
    ax.set_ylabel('Shear Stress')
    ax.set_title("Mohr's Circle")
    return ax

# ======================
# Mechanical Vibrations
# ======================
def modal_analysis(M, K):
    """Solve generalized eigenproblem K*phi = omega^2 * M*phi."""
    evals, evecs = la.eig(K, M)
    idx = np.argsort(evals)
    omega = np.sqrt(evals[idx])
    phi = evecs[:, idx]
    return omega, phi

def sdof_damped_response(m, c, k, F0, omega_f, t):
    """Steady-state harmonic response magnitude and phase for SDOF."""
    wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    r = omega_f / wn
    H = 1 / ((1 - r**2)**2 + (2*zeta*r)**2)**0.5
    X = F0 * H / k
    phi = np.arctan2(2*zeta*r, 1 - r**2)
    x_t = X * np.cos(omega_f * t - phi)
    return x_t, wn, zeta

def mdof_state_space(A, B, C, D, U, t):
    """Linear time-invariant state-space simulation using scipy.signal.lsim."""
    from scipy import signal
    sys = signal.lti(A, B, C, D)
    T, yout, xout = signal.lsim(sys, U, t)
    return T, yout, xout

# ======================
# Fatigue & Fracture
# ======================
def basquin_sn(N, sigma_f_prime, b):
    """Basquin equation: sigma_a = sigma'_f * (2N)^b."""
    return sigma_f_prime * (2.0 * N) ** b

def goodman(sigma_a, sigma_m, Sut, Se):
    """Goodman diagram: return factor of safety."""
    if sigma_a <= 0:
        return np.inf
    return 1.0 / (sigma_a/Se + sigma_m/Sut)

def gerber(sigma_a, sigma_m, Sut, Se):
    """Gerber diagram: return factor of safety."""
    if sigma_a <= 0:
        return np.inf
    term = sigma_a/Se + (sigma_m/Sut)**2
    return 1.0 / term if term > 0 else np.inf

def paris_law(delta_K, C, m):
    """Paris' law: da/dn = C * (delta_K)^m."""
    return C * (delta_K) ** m

# ======================
# Structural Optimization
# ======================
def beam_weight_minimization(L, M_max, sigma_allow, rho, b_bounds, h_bounds):
    """
    Minimize weight of a rectangular beam under bending stress limit.
    Design variables: width b, height h.
    Stress: sigma = M*c/I, c = h/2, I = b*h^3/12 => sigma = 6*M/(b*h^2).
    Weight = rho * b * h * L.
    """
    def weight(x):
        b, h = x
        return rho * b * h * L

    def stress_constr(x):
        b, h = x
        sigma = 6.0 * M_max / (b * h**2)
        return sigma_allow - sigma  # >=0

    cons = ({'type': 'ineq', 'fun': stress_constr})
    bounds = (b_bounds, h_bounds)
    x0 = [(b_bounds[0]+b_bounds[1])/2, (h_bounds[0]+h_bounds[1])/2]
    res = opt.minimize(weight, x0, bounds=bounds, constraints=cons, method='SLSQP')
    return res.x, res.fun

# ======================
# Rotordynamics (Jeffcott rotor)
# ======================
def jeffcott_critical_speed(m, k):
    """Undamped natural frequency (rad/s) and critical speed (Hz)."""
    wn = np.sqrt(k / m)
    fc = wn / (2*np.pi)
    return wn, fc

# ======================
# Contact Mechanics (Hertzian sphere-sphere)
# ======================
def hertz_contact_sphere(R1, R2, E1, nu1, E2, nu2, F):
    """
    Hertzian contact for two spheres.
    Returns contact radius a, maximum pressure p0, and approach delta.
    """
    # Effective modulus
    E_eff = 1.0 / ((1-nu1**2)/E1 + (1-nu2**2)/E2)
    # Effective radius
    R_eff = 1.0 / (1.0/R1 + 1.0/R2)
    a = (3.0 * F * R_eff / (4.0 * E_eff))**(1/3.0)
    p0 = (3.0 * F) / (2.0 * np.pi * a**2)
    delta = a**2 / R_eff
    return a, p0, delta

# ======================
# Failure Theories
# ======================
def von_mises_eq(sx, sy, sz, txy, tyz, tzx):
    """3D Von Mises equivalent stress."""
    return np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2) + 3*(txy**2 + tyz**2 + tzx**2))

def tresca_eq(sx, sy, sz):
    """Tresca equivalent stress (max principal - min principal)."""
    s1, s2, s3 = np.sort([sx, sy, sz])[::-1]  # descending
    return s1 - s3

def max_principal_stress(sx, sy, sz):
    """Maximum principal stress."""
    return np.max([sx, sy, sz])

# ======================
# Demonstration / Main
# ======================
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("=== Finite Element Truss Example ===")
    truss = TrussFE()
    truss.add_node(0, 0.0, 0.0)
    truss.add_node(1, 1.0, 0.0)
    truss.add_node(2, 2.0, 0.0)
    truss.add_element(0, 1, E=210e9, A=0.01)
    truss.add_element(1, 2, E=210e9, A=0.01)
    truss.apply_force(2, 0.0, -10000.0)  # downward force at node 2
    truss.K = truss.assemble_stiffness()
    U = truss.solve()
    print("Nodal displacements (ux, uy):")
    for nid in truss.nodes:
        dofs = truss.dofs[nid]
        print(f"Node {nid}: {U[dofs[0]]:.6e}, {U[dofs[1]]:.6e}")

    print("\n=== Beam Stiffness (2-node) ===")
    E = 210e9
    I = 8.33e-6  # m^4
    L = 2.0
    Kbeam = beam_stiffness_2d(E, I, L)
    print("Beam stiffness matrix:\n", Kbeam)

    print("\n=== Stress Transformation (Mohr's Circle) ===")
    sx, sy, txy = 50e6, 20e6, 30e6
    s1, s2, tau_max = principal_stresses(sx, sy, txy)
    vm = von_mises_stress(sx, sy, txy)
    tres = tresca_stress(sx, sy, txy)
    print(f"Principal stresses: s1={s1/1e6:.2f} MPa, s2={s2/1e6:.2f} MPa")
    print(f"Max shear stress: {tau_max/1e6:.2f} MPa")
    print(f"Von Mises stress: {vm/1e6:.2f} MPa")
    print(f"Tresca stress: {tres/1e6:.2f} MPa")
    fig, ax = plt.subplots()
    plot_mohr_circle(sx, sy, txy, ax=ax)
    plt.show()

    print("\n=== Modal Analysis (2-DOF spring-mass) ===")
    m1 = m2 = 1.0
    k1 = k2 = 20000.0
    M = np.diag([m1, m2])
    K = np.array([[k1+k2, -k2],
                  [-k2, k2]])
    omega, phi = modal_analysis(M, K)
    print("Natural frequencies (rad/s):", omega)
    print("Mode shapes (columns):\n", phi)

    print("\n=== SDOF Damped Harmonic Response ===")
    m = 10.0
    k = 50000.0
    c = 200.0
    F0 = 1000.0
    omega_f = 15.0
    t = np.linspace(0, 5, 500)
    x_t, wn, zeta = sdof_damped_response(m, c, k, F0, omega_f, t)
    print(f"Natural frequency: {wn:.2f} rad/s, Damping ratio: {zeta:.3f}")
    plt.figure()
    plt.plot(t, x_t)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('SDOF Harmonic Response')
    plt.grid()
    plt.show()

    print("\n=== Fatigue: S-N Curve (Basquin) ===")
    N = np.logspace(3, 7, 10)
    sigma_f_prime = 1000e6
    b = -0.12
    sigma_a = basquin_sn(N, sigma_f_prime, b)
    plt.figure()
    plt.loglog(N, sigma_a/1e6, 'o-')
    plt.xlabel('Cycles to failure N')
    plt.ylabel('Stress amplitude (MPa)')
    plt.title('Basquin S-N Curve')
    plt.grid(which='both')
    plt.show()

    print("\n=== Goodman & Gerber Diagrams ===")
    Sut = 600e6
    Se = 250e6
    sigma_m = np.linspace(0, Sut, 100)
    sigma_a_goodman = Se * (1 - sigma_m/Sut)
    sigma_a_gerber = Se * (1 - (sigma_m/Sut)**2)
    plt.figure()
    plt.plot(sigma_m/1e6, sigma_a_goodman/1e6, label='Goodman')
    plt.plot(sigma_m/1e6, sigma_a_gerber/1e6, label='Gerber')
    plt.xlabel('Mean stress (MPa)')
    plt.ylabel('Allowable alternating stress (MPa)')
    plt.title('Goodman & Gerber Criteria')
    plt.legend()
    plt.grid()
    plt.show()

    print("\n=== Paris Law Crack Growth Example ===")
    delta_K = np.linspace(5, 30, 50)  # MPa√m
    C = 1e-12
    m = 3.0
    da_dn = paris_law(delta_K, C, m)
    plt.figure()
    plt.plot(delta_K, da_dn, 'r-')
    plt.xlabel(r'$\Delta K$ (MPa√m)')
    plt.ylabel(r'da/dn (m/cycle)')
    plt.title('Paris Law')
    plt.grid()
    plt.show()

    print("\n=== Structural Optimization: Rectangular Beam ===")
    L = 2.0
    M_max = 5000.0  # Nm
    sigma_allow = 250e6  # Pa
    rho = 7850.0  # kg/m3 (steel)
    b_bounds = (0.02, 0.2)
    h_bounds = (0.02, 0.4)
    (b_opt, h_opt), weight_opt = beam_weight_minimization(L, M_max, sigma_allow, rho, b_bounds, h_bounds)
    print(f"Optimal width: {b_opt:.4f} m")
    print(f"Optimal height: {h_opt:.4f} m")
    print(f"Minimum weight: {weight_opt:.2f} kg")
    # Verify stress
    sigma_opt = 6.0 * M_max / (b_opt * h_opt**2)
    print(f"Resulting bending stress: {sigma_opt/1e6:.2f} MPa (allowable {sigma_allow/1e6:.2f} MPa)")

    print("\n=== Rotordynamics: Jeffcott Critical Speed ===")
    m_rotor = 5.0  # kg
    k_shaft = 1e5  # N/m
    wn, fc = jeffcott_critical_speed(m_rotor, k_shaft)
    print(f"Natural frequency: {wn:.2f} rad/s")
    print(f"Critical speed: {fc:.2f} Hz ({fc*60:.0f} RPM)")

    print("\n=== Hertzian Contact (Two Steel Spheres) ===")
    R1 = R2 = 0.05  # m
    E1 = E2 = 210e9
    nu1 = nu2 = 0.3
    F = 1000.0  # N
    a, p0, delta = hertz_contact_sphere(R1, R2, E1, nu1, E2, nu2, F)
    print(f"Contact radius: {a*1e3:.3f} mm")
    print(f"Maximum pressure: {p0/1e6:.2f} MPa")
    print(f"Approach: {delta*1e6:.3f} µm")

    print("\n=== Failure Theories Example ===")
    sx, sy, sz = 100e6, 50e6, 0.0
    txy, tyz, tzx = 30e6, 0.0, 0.0
    vm3d = von_mises_eq(sx, sy, sz, txy, tyz, tzx)
    tres3d = tresca_eq(sx, sy, sz)
    maxp = max_principal_stress(sx, sy, sz)
    print(f"3D Von Mises stress: {vm3d/1e6:.2f} MPa")
    print(f"3D Tresca stress: {tres3d/1e6:.2f} MPa")
    print(f"Maximum principal stress: {maxp/1e6:.2f} MPa")

    print("\nAll demonstrations completed.")