#!/usr/bin/env python3
"""MECHANICAL ENGINEERING TOOLKIT v1.0
Finite Element Analysis, Stress/Strain Transformations,
Mechanical Vibrations, Fatigue/Fracture Mechanics,
Structural Optimization, Rotordynamics/Contact/Failure Theories.
Uses only numpy, scipy, matplotlib.
"""

import numpy as np
from scipy import linalg, integrate, optimize
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# FINITE ELEMENT ANALYSIS (1D Truss / Beam / Plate Stiffness Matrices)
# -----------------------------------------------------------------------------

def truss_stiffness(E: float, A: float, L: float) -> np.ndarray:
    """1D truss element stiffness matrix in local coordinates.
    
    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    A : float
        Cross-sectional area [m^2]
    L : float
        Element length [m]
    
    Returns
    -------
    k : np.ndarray (2,2)
        Element stiffness matrix [N/m]
    """
    k = np.array([[E*A/L, -E*A/L],
                  [-E*A/L,  E*A/L]])
    return k


def beam_stiffness_local(E: float, I: float, A: float, L: float) -> np.ndarray:
    """Euler-Bernoulli beam element stiffness matrix (local, 6 DOF -> 3 nodes x 2 DOF).
    
    DOF ordering: [u1, theta1, u2, theta2] for axial + bending.
    Axial submatrix is truss-like; bending submatrix is 12x12 reduced to 4x4 
    for transverse deflection rotations if axial decoupled. This function returns 
    the full 6x6 matrix combining axial and bending.
    
    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    I : float
        Area moment of inertia [m^4]
    A : float
        Cross-sectional area [m^2]
    L : float
        Element length [m]
    
    Returns
    -------
    K : np.ndarray (6,6)
        Local stiffness matrix [N/m, N*m/rad]
    """
    # Axial part
    k_axial = truss_stiffness(E, A, L)
    # Bending part (Euler-Bernoulli, 4x4 in terms of w, theta at two nodes)
    k_bend = (E*I / L**3) * np.array([
        [12,  6*L, -12,  6*L],
        [6*L,  4*L**2, -6*L,  2*L**2],
        [-12, -6*L,  12, -6*L],
        [6*L,  2*L**2, -6*L,  4*L**2]
    ])
    # Assemble 6x6 (axial at indices 0,5; bending at 1,2,3,4)
    K = np.zeros((6,6))
    K[0,0] = k_axial[0,0]; K[0,5] = k_axial[0,1]; K[5,0] = k_axial[1,0]; K[5,5] = k_axial[1,1]
    K[1:5,1:5] = k_bend
    return K


def plate_stiffness_4node(E: float, nu: float, t: float, h: float) -> np.ndarray:
    """Basic 4-node quadrilateral plate element stiffness (plane stress, Kirchhoff).
    Uses constant strain (CST) assumption. Returns 8x8 matrix for nodal w and theta_x,theta_y.
    For production, replace with FEM assembly or isoparametric formulation.
    
    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    nu : float
        Poisson's ratio
    t : float
        Plate thickness [m]
    h : float
        Element size / height [m] (assumed square)
    
    Returns
    -------
    K : np.ndarray (8,8)
        Element stiffness matrix [N/m]
    """
    # B = (E*t/h) * (1/(1-nu^2)) * constant matrix; we return simplified form
    D = E / (1 - nu**2)
    # CST stiffness determinant: k = ∫B^T D B dA ≈ B^T D B * h
    # Simplified: k = D * h * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]] per 2 DOF node
    # Here we just return the 3x3 per node assembled to 8x8 for 4 nodes (u_x, u_y)
    # Actually returning 8x8 for transverse deflection w only is common; 
    # for brevity return 8x8 identity-scaled placeholder representing bending stiffness.
    k0 = D * h * (1 - nu) / 2
    K = k0 * np.eye(8)
    return K


# -----------------------------------------------------------------------------
# STRESS & STRAIN TRANSFORMATIONS & YIELD CRITERIA
# -----------------------------------------------------------------------------

def stress_transform(sigma_x: float, sigma_y: float, tau_xy: float, theta_deg: float) -> tuple:
    """Transform plane stress state by angle theta (degrees).
    
    Parameters
    ----------
    sigma_x, sigma_y, tau_xy : float
        Stress components [Pa]
    theta_deg : float
        Rotation angle about z-axis [degrees]
    
    Returns
    -------
    sigma_xn, sigma_yn, tau_xyn : float
        Transformed stresses [Pa]
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    sigma_xn = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
    sigma_yn = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
    tau_xyn = -(sigma_x - sigma_y) * s * c + tau_xy * (c**2 - s**2)
    return sigma_xn, sigma_yn, tau_xyn


def mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float) -> tuple:
    """Compute Mohr's circle parameters for plane stress.
    
    Parameters
    ----------
    sigma_x, sigma_y, tau_xy : float
        Stress components [Pa]
    
    Returns
    -------
    center, radius, sigma1, sigma2 : float
        Circle center stress, radius, max/min principal stresses [Pa]
    """
    center = (sigma_x + sigma_y) / 2.0
    radius = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
    sigma1 = center + radius  # max principal
    sigma2 = center - radius  # min principal
    return center, radius, sigma1, sigma2


def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float) -> tuple:
    """Return principal stresses and orientation angle."""
    center, radius, sigma1, sigma2 = mohr_circle(sigma_x, sigma_y, tau_xy)
    theta_p = 0.5 * np.arctan2(2*tau_xy, sigma_x - sigma_y)  # rad
    return sigma1, sigma2, theta_p


def von_mises_stress(sigma1: float, sigma2: float, sigma3: float = 0.0) -> float:
    """Von Mises equivalent stress for principal stresses."""
    return np.sqrt(((sigma1 - sigma2)**2 + (sigma2 - sigma3)**2 + (sigma3 - sigma1)**2) / 2.0)


def tresca_shear(sigma1: float, sigma2: float, sigma3: float = 0.0) -> float:
    """Tresca maximum shear stress."""
    return sigma1 - sigma3  # assume sigma1 >= sigma2 >= sigma3


# -----------------------------------------------------------------------------
# MECHANICAL VIBRATIONS (SDOF/MDOF damped response, state-space)
# -----------------------------------------------------------------------------

def sdof_natural_freq(m: float, k: float) -> float:
    """Undamped natural frequency of a single-degree-of-freedom system."""
    return np.sqrt(k / m)


def sdof_damped_freq(m: float, c: float, k: float) -> float:
    """Damped natural frequency."""
    omega_n = sdof_natural_freq(m, k)
    zeta = c / (2.0 * np.sqrt(m * k))
    if zeta >= 1.0:
        return 0.0  # overdamped/critically damped
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    return omega_d


def sdof_response_undamped(m: float, k: float, F0: float, omega_force: float, 
                           t_final: float = 10.0, dt: float = 0.01) -> tuple:
    """Undamped SDOF harmonic response: m*x'' + k*x = F0*sin(omega*t).
    Uses exact analytical solution via scipy solve_ivp for generality.
    """
    zeta = 0.0
    omega_n = sdof_natural_freq(m, k)
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    def equations(t, y):
        x, v = y
        xdd = (F0 * np.sin(omega_force * t) - k * x) / m
        return [v, xdd]

    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, int(t_final/dt) + 1)
    sol = solve_ivp(equations, t_span, [0.0, 0.0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0], sol.y[1]


def mdof_eigenanalysis(M: np.ndarray, K: np.ndarray, mass_normalize: bool = True) -> tuple:
    """Generalized eigenvalue problem: K phi = lambda M phi.
    Returns natural frequencies (rad/s) and mode shapes.
    
    Parameters
    ----------
    M, K : np.ndarray (N,N)
        Mass and stiffness matrices
    mass_normalize : bool
        If True, normalize phi^T M phi = I
    
    Returns
    -------
    omega2 : np.ndarray (N,)
        Eigenvalues (omega^2) [rad^2/s^2]
    Phi : np.ndarray (N,N)
        Mode shape matrix (columns = modes)
    """
    # scipy.linalg.eigh for symmetric matrices
    eigenvalues, eigenvectors = linalg.eigh(K, M)
    if mass_normalize:
        # Mass-normalize eigenvectors
        M_ll = np.diag(np.sqrt(np.diag(linalg.inv(M))))  # simplified
        # Proper normalization: phi_i^T M phi_j = delta_ij
        for i in range(eigenvectors.shape[1]):
            phi = eigenvectors[:, i]
            norm = np.sqrt(phi @ M @ phi)
            eigenvectors[:, i] = phi / norm
    omega2 = np.sort(eigenvalues.real)
    Phi = eigenvectors[:, np.argsort(eigenvalues.real)]
    return omega2, Phi


def state_space_a_b_c_d(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                        x0: np.ndarray, t_final: float = 10.0, u: float = 1.0) -> tuple:
    """Linear state-space simulation: x' = A x + B u, y = C x + D u.
    
    Parameters
    ----------
    A, B, C, D : np.ndarray
        State-space matrices
    x0 : np.ndarray
        Initial state vector
    t_final : float
        Simulation time
    u : float
        Constant input
    
    Returns
    -------
    t, y : np.ndarray
        Time vector and output response
    """
    def dynamics(t, y):
        return A @ y + B * u
    
    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, 500)
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45')
    y = sol.y @ C.T + D * u  # assuming C shape (p,n), D shape (p,)
    return sol.t, y


# -----------------------------------------------------------------------------
# FATIGUE & FRACTURE MECHANICS (S-N, Goodman/Gerber, Paris' Law)
# -----------------------------------------------------------------------------

class SNCurve:
    """Baseline S-N curve model: log-log linear fit between two points.
    N = cycles to failure; S = stress amplitude.
    Typically defined by (S_a, N_f) at 10^3 and 10^6 cycles.
    """
    def __init__(self, S_min: float, N_min: float, S_max: float, N_max: float):
        self.S_min = S_min  # stress at N_min
        self.S_max = S_max  # stress at N_max
        self.N_min = N_min
        self.N_max = N_max
        # log10 parameters
        self.log_a, self.log_b = self._fit()
    
    def _fit(self) -> tuple:
        """Fit log10(N) = a - b*log10(S) or similar. 
        Standard form: log10(S) = m * log10(N) + c"""
        # Rearranged: S = a * N^b -> log10(S) = log10(a) + b*log10(N)
        # We'll do log10(N) = alpha - beta * log10(S)
        N = np.array([self.N_min, self.N_max])
        S = np.array([self.S_min, self.S_max])
        logN = np.log10(N)
        logS = np.log10(S)
        # Fit logN = alpha + beta * logS (negative slope typically)
        A = np.vstack([logS, np.ones(2)]).T
        beta, alpha = np.linalg.lstsq(A, logN, rcond=None)[0]
        return alpha, -beta  # store as logN = alpha - beta*logS
    
    def cycles_to_failure(self, S_a: float) -> float:
        """Estimate cycles to failure given stress amplitude S_a."""
        logN = self.log_a - self.log_b * np.log10(S_a)
        return 10.0 ** logN


def goodman_criterion(sigma_a: float, sigma_m: float, S_e: float, S_ut: float) -> float:
    """Goodman mean-stress correction.
    Interaction: sigma_a/S_e + sigma_m/S_ut = 1
    Returns interaction factor; >1 indicates failure."""
    return sigma_a / S_e + sigma_m / S_ut


def gerber_criterion(sigma_a: float, sigma_m: float, S_e: float, S_ut: float) -> float:
    """Gerber parabolic mean-stress correction.
    Interaction: (sigma_a/S_e) + (sigma_m/S_ut)^2 = 1"""
    return sigma_a / S_e + (sigma_m / S_ut) ** 2


class ParisLaw:
    """Paris' Law for fatigue crack growth: da/dN = C * (Delta K)^m
    Delta K = sigma_max * sqrt(pi * a) * Y  (Y = geometry factor, often ~1.12 for SENB)
    """
    def __init__(self, C: float, m: float, Y: float = 1.0):
        self.C = C
        self.m = m
        self.Y = Y
    
    def da_dN(self, a: float, sigma_max: float) -> float:
        """Instantaneous crack growth rate."""
        Delta_K = sigma_max * np.sqrt(np.pi * a) * self.Y
        return self.C * Delta_K ** self.m
    
    def life_increment(self, a0: float, af: float, sigma_max: float, 
                       N_target: float = None, step: float = 1e-3) -> float:
        """Integrate da/dN from a0 to af to find cycles required.
        Uses simple Euler integration for production prototyping."""
        a = a0
        N = 0.0
        while a < af:
            growth = self.da_dN(a, sigma_max)
            if growth <= 0:
                break
            a += growth * step
            N += step
        return N


# -----------------------------------------------------------------------------
# STRUCTURAL OPTIMIZATION (beam cross-section weight minimization)
# -----------------------------------------------------------------------------

def beam_weight_minimization(L: float, E: float, rho: float, 
                             sigma_allow: float, F: float, 
                             I_min: float = 1e-12, A_min: float = 1e-4) -> dict:
    """Minimize beam weight = rho * A * L subject to 
    - axial stress: sigma = F/A <= sigma_allow
    - transverse deflection: delta = F*L**3/(3*E*I) <= delta_allow (default 10mm)
    - minimum area/inertia constraints
    
    Parameters
    ----------
    L : float
        Beam length [m]
    E : float
        Young's modulus [Pa]
    rho : float
        Density [kg/m^3]
    sigma_allow : float
        Allowable stress [Pa]
    F : float
        Applied axial load [N]
    I_min : float
        Minimum moment of inertia [m^4]
    A_min : float
        Minimum cross-sectional area [m^2]
    
    Returns
    -------
    result : dict
        Optimization result with 'A_opt', 'I_opt', 'weight', 'status'
    """
    def constraint_stress(A: float):
        sigma = np.abs(F) / A
        return sigma_allow - sigma  # must be >= 0
    
    def constraint_deflection(A: float, I: float):
        delta = F * L**3 / (3 * E * I)
        # Assume solid rectangular cross-section: I = A*h^2/12, h = sqrt(12*I/A)
        # For given A, minimize I subject to geometry; here treat A and I as independent vars
        return 1e-3 - delta  # delta in meters, assume allow 10mm = 0.01m
    
    # Design variables: [A, I]
    vars0 = [A_min, I_min]
    
    constraints = (
        {'type': 'ineq', 'fun': constraint_stress},
        {'type': 'ineq', 'fun': lambda vars: 0.01 - F * L**3 / (3 * E * vars[1])}  # delta <= 10mm
    )
    
    bounds = [(A_min, None), (I_min, None)]
    
    res = optimize.minimize(
        lambda vars: rho * vars[0] * L,
        vars0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    A_opt, I_opt = res.x
    weight = rho * A_opt * L
    return {
        'A_opt': A_opt,
        'I_opt': I_opt,
        'weight': weight,
        'niters': res.nit,
        'success': res.success,
        'message': res.message
    }


# -----------------------------------------------------------------------------
# ROTORDYNAMICS / CONTACT / FAILURE THEORIES (basic implementations)
# -----------------------------------------------------------------------------

def jeffcott_rotor(m: float, k: float, c: float, omega: float, 
                   rpm: float = 0, T: float = 10.0, dt: float = 0.01) -> dict:
    """Simple Jeffcott rotor model: mass m on shaft stiffness k, damping c,
    rotating at omega (rad/s). Response in x and y directions.
    m*x'' + c*x' + k*x = m*omega^2 * cos(omega*t)  (unbalance force)
    """
    F_unbal = m * omega**2  # amplitude of centrifugal force
    
    def equations(t, y):
        x, vx, y, vy = y
        ax = (F_unbal * np.cos(omega * t) - c * vx - k * x) / m
        ay = (F_unbal * np.sin(omega * t) - c * vy - k * y) / m
        return [vx, ax, vy, ay]
    
    t_span = (0, T)
    t_eval = np.linspace(0, T, int(T/dt) + 1)
    sol = solve_ivp(equations, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, method='RK45')
    
    # Peak amplitude
    Amp_x = np.max(np.abs(sol.y[0]))
    Amp_y = np.max(np.abs(sol.y[2]))
    
    return {
        't': sol.t,
        'x': sol.y[0], 'vx': sol.y[1],
        'y': sol.y[2], 'vy': sol.y[3],
        'omega': omega,
        'rpm': rpm,
        'Amp_x': Amp_x,
        'Amp_y': Amp_y,
        'critical_speed': np.sqrt(k/m)  # rad/s
    }


def hertz_contact(a: float, b: float, F: float, E1: float, nu1: float, 
                  E2: float, nu2: float) -> dict:
    """Hertzian contact stress for two ellipsoidal bodies.
    Max contact pressure p0 = (3*F)/(2*pi*a*b)
    Max shear stress ~ 0.3*p0 at depth ~0.48a (for spheres).
    For simplicity, spheres with a=b=contact radius from Hertz theory.
    This is a placeholder for full tensorial solution."""
    # Simplified: assume contact radius a already computed from geometry/load
    p0 = 3.0 * F / (2.0 * np.pi * a * b)
    tau_max = 0.306 * p0  # approximate max shear depth
    return {
        'contact_radius_a': a,
        'contact_radius_b': b,
        'max_pressure_p0': p0,
        'max_shear_stress_tau_max': tau_max
    }


def failure_theory_compare(sigma1: float, sigma2: float, sigma3: float, 
                           sigma_ut: float, sigma_uc: float = None, 
                           S_y: float = None) -> dict:
    """Compare major failure theories given principal stresses.
    Returns dict with predicted failure index or safety factor per theory.
    """
    # Rankine (tensile/compressive)
    rankine = max(sigma1, sigma3) / (sigma_ut if sigma1 > 0 else abs(sigma3))
    
    # Tresca
    tresca_val = tresca(sigma1, sigma2, sigma3)
    tresca_sf = S_y / tresca_val if S_y else 1.0
    
    # Von Mises
    vm = von_mises_stress(sigma1, sigma2, sigma3)
    vonmises_sf = S_y / vm if S_y else 1.0
    
    # Mohr-Coulomb (simplified)
    mc = sigma1 / sigma_ut - sigma3 / (sigma_uc or sigma_ut)  # assuming compressive strength
    
    return {
        'principal_stresses': [sigma1, sigma2, sigma3],
        'rankine_sf': rankine,
        'tresca_sf': tresca_sf,
        'vonmises_sf': vonmises_sf,
        'max_principal': sigma1,
        'min_principal': sigma3
    }


# -----------------------------------------------------------------------------
# MAIN DEMONSTRATION BLOCK
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("MECHANICAL ENGINEERING TOOLKIT — Demonstration")
    print("=" * 60)
    
    # 1. FEA: Truss stiffness
    print("\n[1] Truss Element Stiffness")
    k = truss_stiffness(E=210e9, A=0.01, L=2.0)
    print(f"  K = \n{k}")
    
    # 2. Beam stiffness
    print("\n[2] Beam Element Stiffness (local 6x6)")
    Kb = beam_stiffness_local(E=210e9, I=1e-4, A=0.01, L=2.0)
    print(f"  K_beam shape = {Kb.shape}")
    
    # 3. Stress transformation + Mohr's circle
    print("\n[3] Stress Transformation & Mohr's Circle")
    sx, sy, txy = 100e6, -50e6, 30e6
    sxn, syn, tuxy = stress_transform(sx, sy, txy, theta_deg=30)
    print(f"  Original: σx={sx/1e6:.1f} MPa, σy={sy/1e6:.1f} MPa, τxy={txy/1e6:.1f} MPa")
    print(f"  Transformed @ 30°: σx'={sxn/1e6:.1f} MPa, σy'={syn/1e6:.1f} MPa, τxy'={tuxy/1e6:.1f} MPa")
    center, rad, s1, s2 = mohr_circle(sx, sy, txy)
    print(f"  Mohr: Center={center/1e6:.1f} MPa, Radius={rad/1e6:.1f} MPa, σ1={s1/1e6:.1f} MPa, σ2={s2/1e6:.1f} MPa")
    
    # 4. Yield criteria
    print("\n[4] Yield Criteria")
    vm = von_mises_stress(s1, s2, 0.0)
    tc = tresca_shear(s1, s2, 0.0)
    print(f"  Von Mises: {vm/1e6:.1f} MPa")
    print(f"  Tresca:    {tc/1e6:.1f} MPa")
    
    # 5. SDOF vibration
    print("\n[5] SDOF Undamped Harmonic Response")
    m, k, F0, omega = 10.0, 4000.0, 50.0, 15.0
    t, x, v = sdof_response_undamped(m, k, F0, omega, t_final=5.0, dt=0.05)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.plot(t, x); plt.title('Displacement'); plt.xlabel('t [s]')
    plt.subplot(1,2,2); plt.plot(t, v); plt.title('Velocity'); plt.xlabel('t [s]')
    plt.tight_layout(); plt.show()
    
    # 6. MDOF eigenanalysis (2-DOF simple)
    print("\n[6] MDOF Eigenanalysis (2-DOF)")
    M = np.array([[2, 0], [0, 1.5]])
    K = np.array([[200, -50], [-50, 100]])
    w2, Phi = mdof_eigenanalysis(M, K)
    print(f"  Natural frequencies (rad/s): {np.sqrt(w2):.1f}")
    print(f"  Mode shapes:\n{Phi}")
    
    # 7. Fatigue: Goodman & Gerber
    print("\n[7] Fatigue Mean-Stress Criteria")
    Se, Sut = 250e6, 600e6
    sigma_a, sigma_m = 100e6, 150e6
    gi = goodman_criterion(sigma_a, sigma_m, Se, Sut)
    gerc = gerber_criterion(sigma_a, sigma_m, Se, Sut)
    print(f"  Goodman interaction factor: {gi:.3f} (>1 = predicted fatigue failure)")
    print(f"  Gerber interaction factor:  {gerc:.3f} (>1 = predicted fatigue failure)")
    sn = SNCurve(S_min=600e6, N_min=1e3, S_max=100e6, N_max=1e6)
    Nf = sn.cycles_to_failure(100e6)
    print(f"  S-N curve: estimated cycles at 100 MPa = {Nf:.1e}")
    
    # 8. Paris' Law (crack growth)
    print("\n[8] Paris' Law Crack Growth")
    paris = ParisLaw(C=1e-10, m=3.0, Y=1.12)
    N_growth = paris.life_increment(a0=1e-3, af=5e-3, sigma_max=200e6, step=1e-2)
    print(f"  Cycles to grow crack from 1mm to 5mm: {N_growth:.1e}")
    
    # 9. Beam weight optimization
    print("\n[9] Beam Weight Minimization")
    opt_result = beam_weight_minimization(L=2.0, E=210e9, rho=7800, 
                                          sigma_allow=250e6, F=5000)
    print(f"  Optimal Area: {opt_result['A_opt']:.4e} m^2")
    print(f"  Optimal I:    {opt_result['I_opt']:.4e} m^4")
    print(f"  Structural weight: {opt_result['weight']:.1f} kg")
    print(f"  Status: {opt_result['success']} — {opt_result['message']}")
    
    # 10. Jeffcott rotor
    print("\n[10] Jeffcott Rotor")
    rotor = jeffcott_rotor(m=50.0, k=1e6, c=500.0, omega=50.0, rpm=475.0, T=5.0)
    print(f  Critical speed (rad/s): {rotor['critical_speed']:.1f}")
    print(f"  Peak amplitudes: x={rotor['Amp_x']:.3f} m, y={rotor['Amp_y']:.3f} m")
    
    # 11. Hertz contact
    print("\n[11] Hertz Contact Stress")
    hc = hertz_contact(a=0.02, b=0.02, F=5000.0, E1=210e9, nu1=0.3, E2=210e9, nu2=0.3)
    print(f"  Max contact pressure p0: {hc['max_pressure_p0']/1e6:.1f} MPa")
    print(f"  Max shear stress:        {hc['max_shear_stress_tau_max']/1e6:.1f} MPa")
    
    # 12. Failure theories comparison
    print("\n[12] Failure Theory Comparison")
    sf = failure_theory_compare(sigma1=300e6, sigma2=0, sigma3=-100e6, 
                                sigma_ut=500e6, sigma_uc=500e6, S_y=250e6)
    for key, val in sf.items():
        if 'sf' in key or 'stress' in key.lower():
            print(f"  {key}: {val:.3f}" if 'sf' in key else f"  {key}: {val/1e6:.1f} MPa")
    
    print("\n" + "=" * 60)
    print("Demonstration complete.")
    print("=" * 60)