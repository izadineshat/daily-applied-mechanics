import numpy as np
from scipy import linalg, integrate, optimize
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Stress & Strain Transformations
# ----------------------------------------------------------------------


def stress_transform(sigma_x, sigma_y, tau_xy, theta_deg):
    """Transform stresses from global to local coordinate system rotated by theta_deg.
    Returns (sigma_x_local, sigma_y_local, tau_xy_local)."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([[c**2, s**2, 2*s*c],
                  [s**2, c**2, -2*s*c],
                  [-s*c, s*c, c**2 - s**2]])
    sigma = T @ np.array([sigma_x, sigma_y, tau_xy])
    return sigma[0], sigma[1], sigma[2]


def principal_stresses(sigma_x, sigma_y, tau_xy):
    """Compute principal stresses and orientation.
    Returns (sigma1, sigma2, theta_p_deg)."""
    sigma_avg = (sigma_x + sigma_y) / 2.0
    R = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
    sigma1 = sigma_avg + R
    sigma2 = sigma_avg - R
    theta_p = np.rad2deg(0.5 * np.arctan2(2.0 * tau_xy, sigma_x - sigma_y))
    return sigma1, sigma2, theta_p


def von_mises(sigma_x, sigma_y, tau_xy):
    """Von Mises equivalent stress for plane stress state."""
    return np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3.0*tau_xy**2)


def tresca(sigma_x, sigma_y, tau_xy):
    """Tresca maximum shear stress (difference of principal stresses)."""
    s1, s2, _ = principal_stresses(sigma_x, sigma_y, tau_xy)
    return s1 - s2


# ----------------------------------------------------------------------
# Finite Element: Truss & Beam Elements
# ----------------------------------------------------------------------


class TrussElement:
    """1D truss element assembling the local axial stiffness matrix.
    ke = (A*E/L) * [[1, -1], [-1, 1]]"""
    def __init__(self, A, E):
        self.A = A   # cross-sectional area [m^2]
        self.E = E   # Young's modulus [Pa]

    def stiffness(self, L):
        """Return the 2x2 local stiffness matrix for truss element of length L [m]."""
        k = (self.A * self.E / L) * np.array([[1, -1], [-1, 1]])
        return k


class BeamElement:
    """2D Euler-Bernoulli beam element (4 DOF: transverse deflection + rotation at each node).
    Local 4x4 bending stiffness matrix ke = (E*I/L^3) * [[12, 6L, -12, 6L], ...]"""
    def __init__(self, E, I):
        self.E = E   # Young's modulus [Pa]
        self.I = I   # area moment of inertia [m^4]

    def stiffness(self, L):
        """Return the 4x4 local bending stiffness matrix for beam of length L [m]."""
        E, I = self.E, self.I
        k = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k


# ----------------------------------------------------------------------
# Mechanical Vibrations
# ----------------------------------------------------------------------


class SDOF:
    """Single Degree of Freedom damped system: m*x'' + c*x' + k*x = F(t)."""
    def __init__(self, m, c, k):
        self.m = m   # mass [kg]
        self.c = c   # damping coefficient [Ns/m]
        self.k = k   # stiffness [N/m]

    def omega_n(self):
        """Undamped natural frequency [rad/s]."""
        return np.sqrt(self.k / self.m)

    def zeta(self):
        """Damping ratio."""
        return self.c / (2.0 * np.sqrt(self.m * self.k))

    def damped_omega(self):
        """Damped natural frequency [rad/s]; returns 0 if overdamped."""
        z = self.zeta()
        if z < 1.0:
            return self.omega_n() * np.sqrt(1.0 - z**2)
        return 0.0

    def ode(self, t, y):
        """Right-hand side of first-order ODE system: y = [x, x']."""
        x, v = y
        dxdt = v
        dvdt = (-self.c * v - self.k * x) / self.m  # free vibration, F=0
        return [dxdt, dvdt]

    def free_response(self, t_span=(0, 10), y0=(1.0, 0.0), npts=200):
        """Solve free vibration and return time history of displacement."""
        t = np.linspace(*t_span, npts)
        sol = integrate.solve_ivp(self.ode, t_span, y0, t_eval=t)
        return sol.t, sol.y[0]


class MDOF:
    """Undamped modal analysis for an N-DOF system [K]{x} = w^2[M]{x}.
    Uses scipy.linalg.eigh for symmetric positive-definite generalized eigenvalue problem."""
    def __init__(self, K, M):
        self.K = np.asarray(K, dtype=float)
        self.M = np.asarray(M, dtype=float)
        n = self.K.shape[0]
        eigvals, eigvecs = linalg.eigh(self.K, self.M)
        self.omega_n = np.sqrt(eigvals)           # rad/s
        self.f_n = self.omega_n / (2.0 * np.pi)   # Hz
        self.phi = eigvecs                        # mode shapes (columns)

    def natural_frequencies_hz(self):
        """Return natural frequencies in Hertz."""
        return self.f_n


# ----------------------------------------------------------------------
# Fatigue & Fracture Mechanics
# ----------------------------------------------------------------------


class Fatigue:
    """S-N curves, Goodman/Gerber diagrams, and Paris' law for fatigue crack growth."""

    @staticmethod
    def basquin_coeff(sigma_f_prime, b):
        """Basquin's equation: sigma_a = sigma_f_prime * N^b. Returns (a, b)."""
        return sigma_f_prime, b

    @staticmethod
    def goodman_fos(sigma_a, sigma_m, sigma_ut, sigma_e):
        """Factor of safety using the Goodman failure line.
        n = 1 / (sigma_a/sigma_e + sigma_m/sigma_ut)"""
        if sigma_a <= 0:
            return np.inf
        return 1.0 / (sigma_a / sigma_e + sigma_m / sigma_ut)

    @staticmethod
    def gerber_fos(sigma_a, sigma_m, sigma_ut, sigma_e):
        """Factor of safety using the Gerber parabola.
        n = sigma_a / [sigma_e * (1 - (sigma_m/sigma_ut)^2)]"""
        if sigma_a <= 0:
            return np.inf
        denom = sigma_e * (1.0 - (sigma_m / sigma_ut)**2)
        if denom <= 0:
            return 0.0 if sigma_a > 0 else np.inf
        return sigma_a / denom

    @staticmethod
    def paris(Delta_K, C, m):
        """Paris' law for fatigue crack growth: da/dN = C * (Delta_K)^m."""
        return C * Delta_K**m


# ----------------------------------------------------------------------
# Structural Optimization
# ----------------------------------------------------------------------


def beam_weight_min(L, P, E, sigma_allow, rho=7800.0, w=0.1):
    """Minimum-weight simply supported beam with central point load P [N].
    Rectangular cross-section of fixed width w [m]; height h [m] optimized
    such that sigma_max <= sigma_allow [Pa]. Returns dict with h, A, weight."""
    # M_max = P*L/4; sigma_max = 3*P*L/(2*w*h^2)  =>  h = sqrt(3*P*L/(2*w*sigma_allow))
    h = np.sqrt(3.0 * P * L / (2.0 * w * sigma_allow))
    A = w * h
    weight = rho * A * L
    return {"height": h, "area": A, "weight": weight, "status": "optimal"}


# ----------------------------------------------------------------------
# Failure Theories
# ----------------------------------------------------------------------


def rankine_failure(sigma1, sigma2, sigma3=0.0, sigma_allow=1e6):
    """Rankine (maximum normal stress) failure criterion.
    Returns True if any principal stress exceeds sigma_allow."""
    return max(sigma1, sigma2, sigma3) > sigma_allow


def tresca_failure(sigma1, sigma2, sigma3=0.0, sigma_allow=1e6):
    """Tresca (maximum shear stress) failure criterion.
    Returns True if max shear stress > sigma_allow/2."""
    tau_max = (max(sigma1, sigma2, sigma3) - min(sigma1, sigma2, sigma3)) / 2.0
    return tau_max > sigma_allow / 2.0


# ----------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("Mechanical Engineering Toolkit — Demo\n")

    # 1. Stress & strain transformation
    sx, sy, txy = 100e6, 40e6, 30e6
    sx2, sy2, txy2 = stress_transform(sx, sy, txy, 30)
    s1, s2, th = principal_stresses(sx, sy, txy)
    vm = von_mises(sx, sy, txy)
    tc = tresca(sx, sy, txy)
    print(f"Transformed @ 30°: sigma_x={sx2/1e6:.1f} MPa, sigma_y={sy2/1e6:.1f} MPa, tau_xy={txy2/1e6:.1f} MPa")
    print(f"Principal: sigma1={s1/1e6:.1f} MPa, sigma2={s2/1e6:.1f} MPa @ {th:.1f}°")
    print(f"Von Mises: {vm/1e6:.1f} MPa,  Tresca: {tc/1e6:.1f} MPa\n")

    # 2. Truss FEA
    tr = TrussElement(A=0.01, E=210e9)
    K = tr.stiffness(L=2.0)
    print(f"Truss 2x2 local KE (GPa):\n{K/1e9:.4f}\n")

    # 3. Beam FEA
    beam = BeamElement(E=210e9, I=8.33e-6)
    Kb = beam.stiffness(L=4.0)
    print(f"Beam 4x4 bending KE (scaled):\n{Kb/1e9:.4f} GPa\n")

    # 4. SDOF free vibration
    sdof = SDOF(m=2.0, c=0.5, k=200.0)
    t, x = sdof.free_response(y0=(0.05, 0.0))
    print(f"SDOF nat freq: {sdof.omega_n()/2/np.pi:.2f} Hz")
    plt.figure()
    plt.plot(t, x)
    plt.title("SDOF Free Vibration")
    plt.xlabel("t [s]"); plt.ylabel("x [m]"); plt.grid(True)

    # 5. MDOF modal analysis
    Km = np.array([[20e6, -5e6], [-5e6, 20e6]])
    Mm = np.array([[5000., 0.], [0., 5000.]])
    mof = MDOF(Km, Mm)
    print(f"MDOF natural frequencies: {mof.natural_frequencies_hz()} Hz\n")

    # 6. Fatigue & fracture
    sigma_a, sigma_m = 150e6, 50e6
    sigma_ut, sigma_e = 600e6, 300e6
    n_g = Fatigue.goodman_fos(sigma_a, sigma_m, sigma_ut, sigma_e)
    n_r = Fatigue.gerber_fos(sigma_a, sigma_m, sigma_ut, sigma_e)
    da_dN = Fatigue.paris(Delta_K=50.0, C=1e-10, m=3.0)
    print(f"Goodman FoS: {n_g:.3f}")
    print(f"Gerber FoS:  {n_r:.3f}")
    print(f"Paris law da/dN: {da_dN:.2e} m/cycle\n")

    # 7. Beam weight optimization
    opt = beam_weight_min(L=4.0, P=5000.0, E=210e9, sigma_allow=250e6, w=0.1)
    print(f"Beam opt height: {opt['height']:.4f} m, cross-sectional area: {opt['area']:.4f} m^2, weight: {opt['weight']:.1f} N\n")

    # 8. Failure theories
    s1, s2, _ = principal_stresses(100e6, 50e6, 20e6)[:3]
    print(f"Rankine failure (sigma_allow=150 MPa): {rankine_failure(s1, s2, sigma_allow=150e6)}")
    print(f"Tresca failure (sigma_allow=150 MPa): {tresca_failure(s1, s2, sigma_allow=150e6)}")

    plt.show()
    print("\nDemo finished.")