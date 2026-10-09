import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
from math import pi, sqrt


# --------------------------
# Finite Element Utilities
# --------------------------

class TrussElement:
    """2-node 2D truss element."""
    def __init__(self, node_i, node_j, A, E):
        self.node_i = node_i  # tuple (x, y)
        self.node_j = node_j
        self.A = A            # cross-sectional area
        self.E = E            # Young's modulus
        self.length = sqrt((node_j[0] - node_i[0])**2 +
                           (node_j[1] - node_i[1])**2)
        self.c = (node_j[0] - node_i[0]) / self.length
        self.s = (node_j[1] - node_i[1]) / self.length

    def local_stiffness(self):
        """Element stiffness matrix in local coordinates (2x2)."""
        k = self.E * self.A / self.length
        return k * np.array([[1, -1],
                             [-1, 1]])

    def transformation_matrix(self):
        """Transformation matrix from local to global DOFs (2x4)."""
        T = np.array([[self.c, self.s, 0, 0],
                      [0, 0, self.c, self.s]])
        return T

    def global_stiffness(self):
        """Element stiffness matrix in global coordinates (4x4)."""
        k_local = self.local_stiffness()
        T = self.transformation_matrix()
        return T.T @ k_local @ T


class BeamElement2D:
    """2-node Euler‑Bernoulli beam element in the xy‑plane."""
    def __init__(self, node_i, node_j, E, I, A=None):
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.I = I
        self.A = A if A is not None else I  # placeholder for axial
        self.L = sqrt((node_j[0] - node_i[0])**2 +
                      (node_j[1] - node_i[1])**2)
        self.c = (node_j[0] - node_i[0]) / self.L
        self.s = (node_j[1] - node_i[1]) / self.L

    def local_stiffness(self):
        """Element stiffness matrix in local coordinates (4x4)."""
        E, I, L = self.E, self.I, self.L
        k = E * I / L**3
        return k * np.array([[12, 6*L, -12, 6*L],
                             [6*L, 4*L**2, -6*L, 2*L**2],
                             [-12, -6*L, 12, -6*L],
                             [6*L, 2*L**2, -6*L, 4*L**2]])

    def transformation_matrix(self):
        """Transformation matrix for beam (4x4)."""
        T = np.array([[self.c, self.s, 0, 0],
                      [-self.s, self.c, 0, 0],
                      [0, 0, self.c, self.s],
                      [0, 0, -self.s, self.c]])
        return T

    def global_stiffness(self):
        """Element stiffness matrix in global coordinates (4x4)."""
        k_local = self.local_stiffness()
        T = self.transformation_matrix()
        return T.T @ k_local @ T


def assemble_global(elements, ndof):
    """Assemble global stiffness matrix from a list of elements."""
    K = np.zeros((ndof, ndof))
    for elem in elements:
        k_elem = elem.global_stiffness()
        # map element DOFs to global DOFs (assumes ordering stored in elem.dof_map)
        dofs = elem.dof_map
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += k_elem[i, j]
    return K


def apply_boundary(K, F, fixed_dofs):
    """Apply Dirichlet boundary conditions by modifying K and F."""
    K_mod = K.copy()
    F_mod = F.copy()
    for dof in fixed_dofs:
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = 1.0
        F_mod[dof] = 0.0
    return K_mod, F_mod


# --------------------------
# Stress & Strain Transformations
# --------------------------

def principal_stresses(sx, sy, txy):
    """Return principal stresses and maximum shear stress."""
    avg = (sx + sy) / 2
    r = sqrt(((sx - sy) / 2)**2 + txy**2)
    s1 = avg + r
    s2 = avg - r
    tau_max = r
    return s1, s2, tau_max


def von_mises_stress(sx, sy, sz, txy, tyz, tzx):
    """Von Mises equivalent stress for 3D stress state."""
    return sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 +
                       6 * (txy**2 + tyz**2 + tzx**2)))


def tresca_yield(sx, sy, sz, txy, tyz, tzx):
    """Tresca yield criterion: max shear stress."""
    # principal stresses from stress deviator
    s1, s2, s3 = np.linalg.eigvals(np.array([[sx, txy, tzx],
                                             [txy, sy, tyz],
                                             [tzx, tyz, sz]]))
    tau_max = 0.5 * (np.max([s1, s2, s3]) - np.min([s1, s2, s3]))
    return tau_max


def mohr_circle(sx, sy, txy):
    """Return center, radius, and angle to principal plane."""
    C = (sx + sy) / 2
    R = sqrt(((sx - sy) / 2)**2 + txy**2)
    theta_p = 0.5 * np.arctan2(2*txy, sx - sy)  # radians
    return C, R, theta_p


# --------------------------
# Mechanical Vibrations
# --------------------------

class SDOF:
    """Single‑Degree‑Of‑Freedom system."""
    def __init__(self, m, k, c=0.0):
        self.m = m
        self.k = k
        self.c = c
        self.wn = sqrt(k / m)          # undamped natural frequency (rad/s)
        self.zeta = c / (2 * sqrt(m * k))  # damping ratio

    def damped_frequency(self):
        return self.wn * sqrt(1 - self.zeta**2) if self.zeta < 1 else 0.0

    def impulse_response(self, t):
        """Response to unit impulse (assuming zero initial conditions)."""
        if self.zeta < 1:
            wd = self.damped_frequency()
            return (1 / (self.m * wd)) * np.exp(-self.zeta * self.wn * t) * np.sin(wd * t)
        elif self.zeta == 1:
            return (1 / self.m) * t * np.exp(-self.wn * t)
        else:
            w1 = self.wn * (self.zeta + sqrt(self.zeta**2 - 1))
            w2 = self.wn * (self.zeta - sqrt(self.zeta**2 - 1))
            return (1 / (self.m * (w1 - w2))) * (np.exp(-w2 * t) - np.exp(-w1 * t))

    def forced_response(self, t, F0, omega):
        """Steady‑state amplitude for harmonic force F0*sin(omega*t)."""
        r = omega / self.wn
        denom = sqrt((1 - r**2)**2 + (2 * self.zeta * r)**2)
        return (F0 / self.k) * (1 / denom)


class MDOF:
    """Multi‑Degree‑Of‑Freedom system with proportional (Rayleigh) damping."""
    def __init__(self, M, K, alpha=0.0, beta=0.0):
        self.M = np.array(M, dtype=float)
        self.K = np.array(K, dtype=float)
        self.alpha = alpha
        self.beta = beta
        self.C = alpha * self.M + beta * self.K
        self.eigvals, self.phi = la.eig(K, M)
        # sort eigenvalues and eigenvectors
        idx = np.argsort(self.eigvals)
        self.eigvals = self.eigvals[idx]
        self.phi = self.phi[:, idx]
        self.wn = np.sqrt(self.eigvals.real)  # natural frequencies (rad/s)
        self.zeta = 0.5 * (alpha / self.wn + beta * self.wn)  # modal damping ratios

    def modal_superposition(self, t, F_vec, u0=None, v0=None):
        """Compute displacement response using modal superposition (assuming zero initial conditions)."""
        n = self.M.shape[0]
        if u0 is None:
            u0 = np.zeros(n)
        if v0 is None:
            v0 = np.zeros(n)
        q = np.zeros((len(t), n))  # modal coordinates
        for i in range(n):
            wd = self.wn[i] * sqrt(1 - self.zeta[i]**2) if self.zeta[i] < 1 else 0.0
            phi_i = self.phi[:, i]
            # modal force
            Fi = phi_i @ F_vec
            # Duhamel integral approximated via numeric integration (simple trapezoidal)
            # For brevity we use analytical solution for zero initial conditions and harmonic force.
            # Here we assume F_vec is constant in time (step load).
            if self.zeta[i] < 1:
                q[:, i] = (Fi / (self.M[i, i] * wd)) * np.exp(-self.zeta[i] * self.wn[i] * t) * np.sin(wd * t)
            else:
                q[:, i] = (Fi / self.M[i, i]) * t * np.exp(-self.wn[i] * t)
        # physical displacement
        U = q @ self.phi.T
        return U


# --------------------------
# Fatigue & Fracture Mechanics
# --------------------------

class S_N_Curve:
    """Basquin-type S-N curve: S = S_f' * (2N)^b."""
    def __init__(self, S_f_prime, b):
        self.S_f_prime = S_f_prime
        self.b = b

    def stress(self, N):
        return self.S_f_prime * (2 * N) ** self.b

    def cycles(self, S):
        return 0.5 * (S / self.S_f_prime) ** (1 / self.b)


def goodman_diagram(S_mean, S_alt, S_ut, S_y):
    """Goodman criterion: S_alt/S_e + S_mean/S_ut <= 1 (using S_e ≈ S_y/2 as endurance limit)."""
    S_e = S_y / 2
    return S_alt / S_e + S_mean / S_ut


def gerber_diagram(S_mean, S_alt, S_ut):
    """Gerber criterion: (S_alt/S_e)^2 + S_mean/S_ut <= 1."""
    S_e = S_ut / 2  # approximate
    return (S_alt / S_e) ** 2 + S_mean / S_ut


def paris_law(C, m, delta_K):
    """da/dN = C * (delta_K)^m."""
    return C * (delta_K) ** m

    # Integration to get cycles for crack growth from a0 to af
def paris_cycles(C, m, delta_K_func, a0, af):
    """
    Integrate da/dN = C * (ΔK)^m from a0 to af.
    delta_K_func(a) must return ΔK for a given crack length a.
    """
    from scipy.integrate import quad
    integrand = lambda a: 1.0 / (C * (delta_K_func(a)) ** m)
    N, _ = quad(integrand, a0, af)
    return N


# --------------------------
# Structural Optimization
# --------------------------

def beam_weight(rect_width, height, length, density):
    """Weight of a rectangular beam."""
    return density * rect_width * height * length

def beam_stress(rect_width, height, length, load, moment_factor=1.0):
    """Maximum bending stress for a cantilever with end load."""
    M = load * length * moment_factor
    I = rect_width * height**3 / 12
    c = height / 2
    return M * c / I

def optimize_beam_height(width, length, load, density, allowable_stress, max_iter=100):
    """Minimize weight subject to stress <= allowable_stress using gradient-free optimization."""
    def objective(h):
        if h <= 0:
            return 1e9
        w = beam_weight(width, h[0], length, density)
        sigma = beam_stress(width, h[0], length, load)
        penalty = 0.0 if sigma <= allowable_stress else 1e6 * (sigma - allowable_stress)
        return w + penalty
    res = opt.minimize(objective, x0=[length/10], bounds=[(1e-3, length)],
                       method='L-BFGS-B', options={'maxiter': max_iter})
    return res.x[0], res.fun


# --------------------------
# Rotordynamics (Jeffcott Rotor)
# --------------------------

def jeffcott_critical_speed(m, k, e=0.0):
    """Critical speed (rad/s) of a simple Jeffcott rotor with unbalance e."""
    wn = sqrt(k / m)
    return wn  # undamped natural frequency equals critical speed

def jeffcott_unbalance_response(m, k, c, e, omega):
    """Steady-state displacement amplitude due to unbalance."""
    wn = sqrt(k / m)
    zeta = c / (2 * sqrt(m * k))
    r = omega / wn
    denom = sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
    return m * e * omega**2 / (k * denom)


# --------------------------
# Contact Mechanics (Hertzian)
# --------------------------

def hertzian_sphere_flat(R, E1, nu1, E2, nu2, F):
    """Hertzian contact for a sphere on a flat."""
    # Effective modulus
    E_eff = 1 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)
    a = (3 * F * R / (4 * E_eff)) ** (1/3)          # contact radius
    p_max = (3 * F) / (2 * pi * a**2)               # max pressure
    return a, p_max


# --------------------------
# Failure Theories
# --------------------------

def von_mises_failure(sx, sy, sz, txy, tyz, tzx, S_y):
    """Return von Mises equivalent stress and yield check."""
    sigma_vm = von_mises_stress(sx, sy, sz, txy, tyz, tzx)
    return sigma_vm, sigma_vm <= S_y

def tresca_failure(sx, sy, sz, txy, tzx, tyz, S_y):
    """Return max shear stress and Tresca yield check."""
    tau_max = tresca_yield(sx, sy, sz, txy, tyz, tzx)
    return tau_max, tau_max <= S_y / 2


# --------------------------
# Demonstration / Main
# --------------------------

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("=== 1D/2D Truss Example ===")
    nodes = [(0, 0), (1, 0), (0, 1)]
    elements = [
        TrussElement(nodes[0], nodes[1], A=0.01, E=200e9),
        TrussElement(nodes[0], nodes[2], A=0.01, E=200e9)
    ]
    # assign DOF maps: each node has 2 DOFs (ux, uy)
    for i, elem in enumerate(elements):
        ni, nj = nodes.index(elem.node_i), nodes.index(elem.node_j)
        elem.dof_map = [2*ni, 2*ni+1, 2*nj, 2*nj+1]
    ndof = len(nodes) * 2
    K = assemble_global(elements, ndof)
    F = np.zeros(ndof)
    F[3] = -1000.0  # downward load at node 2 (vertical DOF)
    fixed = [0, 1]   # fix node 0 (ux, uy)
    Kbc, Fbc = apply_boundary(K, F, fixed)
    U = la.solve(Kbc, Fbc)
    print("Nodal displacements:\n", U.reshape(-1, 2))
    # compute axial stress in first element
    e0 = elements[0]
    u_local = e0.transformation_matrix() @ U[e0.dof_map]
    sigma = e0.E * (u_local[1] - u_local[0]) / e0.length
    print(f"Axial stress in element 1: {sigma/1e6:.2f} MPa")

    print("\n=== Beam Bending Example ===")
    Lb = 2.0
    beam = BeamElement2D((0, 0), (Lb, 0), E=210e9, I=8.33e-6, A=0.01)
    # assign DOF map: each node has 3 DOFs (ux, uy, theta)
    beam.dof_map = [0, 1, 2, 3, 4, 5]
    Kb = beam.global_stiffness()
    Fb = np.zeros(6)
    Fb[4] = -5000.0  # vertical force at free end
    fixed_beam = [0, 1, 2]  # clamp left node
    Kbc_b, Fbc_b = apply_boundary(Kb, Fb, fixed_beam)
    Ub = la.solve(Kbc_b, Fbc_b)
    print("Tip vertical displacement (m):", Ub[4])
    print("Tip rotation (rad):", Ub[5])

    print("\n=== Stress Transformation Example ===")
    sx, sy, txy = 50e6, -20e6, 30e6
    s1, s2, tau_max = principal_stresses(sx, sy, txy)
    print(f"Principal stresses: s1={s1/1e6:.2f} MPa, s2={s2/1e6:.2f} MPa")
    print(f"Maximum shear stress: {tau_max/1e6:.2f} MPa")
    C, R, theta_p = mohr_circle(sx, sy, txy)
    print(f"Mohr circle center: {C/1e6:.2f} MPa, radius: {R/1e6:.2f} MPa, angle to principal plane: {np.degrees(theta_p):.2f} deg")
    sigma_vm, ok_vm = von_mises_failure(sx, sy, 0, txy, 0, 0, 250e6)
    print(f"Von Mises stress: {sigma_vm/1e6:.2f} MPa, yield check: {ok_vm}")
    tau_tresca, ok_tresca = tresca_failure(sx, sy, 0, txy, 0, 0, 250e6)
    print(f"Tresca max shear: {tau_tresca/1e6:.2f} MPa, yield check: {ok_tresca}")

    print("\n=== SDOF Vibration Example ===")
    sdof = SDOF(m=10.0, k=20000.0, c=20.0)
    print(f"Natural frequency: {sdof.wn:.2f} rad/s, damping ratio: {sdof.zeta:.3f}")
    t = np.linspace(0, 5, 500)
    resp = sdof.impulse_response(t)
    plt.figure()
    plt.plot(t, resp)
    plt.title('SDOF Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)

    print("\n=== MDOF Modal Analysis Example ===")
    M = np.diag([2.0, 2.0])
    K = np.array([[30000, -10000],
                  [-10000, 30000]])
    mdof = MDOF(M, K, alpha=0.0, beta=0.001)
    print("Natural frequencies (rad/s):", mdof.wn)
    print("Mode shapes:\n", mdof.phi)
    # simulate impulse on first DOF
    F_vec = np.array([1000.0, 0.0])
    t = np.linspace(0, 5, 500)
    U = mdof.modal_superposition(t, F_vec)
    plt.figure()
    plt.plot(t, U[:, 0], label='DOF 1')
    plt.plot(t, U[:, 1], label='DOF 2')
    plt.title('MDOF Response to Impulse')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.legend()
    plt.grid(True)

    print("\n=== Fatigue S-N Curve Example ===")
    sn = S_N_Curve(S_f_prime=900e6, b=-0.12)
    Ns = np.logspace(3, 7, 5)
    Ss = sn.stress(Ns)
    plt.figure()
    plt.loglog(Ns, Ss/1e6, 'o-')
    plt.title('S-N Curve (Basquin)')
    plt.xlabel('Cycles N')
    plt.ylabel('Stress amplitude (MPa)')
    plt.grid(True, which='both')

    print("\n=== Goodman Diagram Example ===")
    S_mean = np.linspace(0, 500, 100)
    S_alt_goodman = 250e6 * (1 - S_mean/600e6)  # S_e=250 MPa, S_ut=600 MPa
    plt.figure()
    plt.plot(S_mean/1e6, S_alt_goodman/1e6, label='Goodman')
    plt.xlabel('Mean stress (MPa)')
    plt.ylabel('Allowable alternating stress (MPa)')
    plt.title('Goodman Diagram')
    plt.grid(True)
    plt.legend()

    print("\n=== Paris Law Crack Growth Example ===")
    C = 1e-12
    m = 3.0
    def delta_K(a):  # simple geometry factor, assume constant stress intensity factor range
        return 50e6 * sqrt(pi * a)  # MPa*sqrt(m) -> Pa*sqrt(m)
    a0 = 0.001  # 1 mm
    af = 0.01   # 10 mm
    N_cycle = paris_cycles(C, m, delta_K, a0, af)
    print(f"Cycles to grow crack from {a0*1e3:.2f} mm to {af*1e3:.2f} mm: {N_cycle:.0e}")

    print("\n=== Beam Cross-Section Optimization ===")
    width = 0.05
    length = 2.0
    load = 10000.0
    density = 7850.0
    allowable = 250e6
    h_opt, weight_opt = optimize_beam_height(width, length, load, density, allowable)
    print(f"Optimal height: {h_opt*1e3:.2f} mm, Weight: {weight_opt:.2f} N")

    print("\n=== Jeffcott Rotor Critical Speed ===")
    m_rotor = 5.0
    k_rotor = 1e6
    omega_cr = jeffcott_critical_speed(m_rotor, k_rotor)
    print(f"Critical speed: {omega_cr:.2f} rad/s ({omega_cr*60/(2*pi):.1f} RPM)")

    print("\n=== Hertzian Contact (Sphere on Flat) ===")
    R = 0.01
    E1 = E2 = 210e9
    nu1 = nu2 = 0.3
    F = 500.0
    a, p_max = hertzian_sphere_flat(R, E1, nu1, E2, nu2, F)
    print(f"Contact radius: {a*1e3:.3f} mm")
    print(f"Maximum pressure: {p_max/1e6:.2f} MPa")

    plt.show()