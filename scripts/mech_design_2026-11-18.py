import numpy as np
from scipy.linalg import eigh, inv, det
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable

# ============================================================
# Finite Element Analysis (1D/2D Truss & Beam)
# ============================================================

@dataclass
class TrussElement:
    """1D truss element with two nodes"""
    E: float  # Young's modulus (Pa)
    A: float  # Cross-sectional area (m^2)
    L: float  # Length (m)
    
    def stiffness_matrix(self) -> np.ndarray:
        """Return 4x4 global stiffness matrix for 1D truss element"""
        k = (self.E * self.A / self.L) * np.array([
            [1, -1],
            [-1, 1]
        ])
        return np.kron(np.eye(2), k)  # Expand for 2D (x,y)

@dataclass
class BeamElement:
    """Euler-Bernoulli beam element with two nodes"""
    E: float  # Young's modulus (Pa)
    I: float  # Second moment of area (m^4)
    A: float  # Cross-sectional area (m^2)
    L: float  # Length (m)
    
    def stiffness_matrix(self) -> np.ndarray:
        """Return 12x12 stiffness matrix for 3D beam element"""
        k_local = np.zeros((12, 12))
        
        # Axial stiffness
        axial = self.E * self.A / self.L
        k_local[0, 0] = axial
        k_local[0, 6] = -axial
        k_local[6, 0] = -axial
        k_local[6, 6] = axial
        
        # Bending about y-axis (xz-plane)
        by = 12 * self.E * self.I / (self.L**3)
        k_local[1, 1] = by
        k_local[1, 7] = -by
        k_local[7, 1] = -by
        k_local[7, 7] = by
        
        # Bending about z-axis (xy-plane)
        bz = 12 * self.E * self.I / (self.L**3)
        k_local[2, 2] = bz
        k_local[2, 8] = -bz
        k_local[8, 2] = -bz
        k_local[8, 8] = bz
        
        # Torsion
        torsion = self.E * self.I / (self.L)  # Simplified
        k_local[3, 3] = torsion
        k_local[3, 9] = -torsion
        k_local[9, 3] = -torsion
        k_local[9, 9] = torsion
        
        return k_local

def assemble_global_stiffness(elements: List, nodes: np.ndarray) -> np.ndarray:
    """Assemble global stiffness matrix from element matrices"""
    n_nodes = len(nodes)
    n_dof = n_nodes * 6  # 6 DOF per node in 3D
    K_global = np.zeros((n_dof, n_dof))
    
    for i, element in enumerate(elements):
        k_elem = element.stiffness_matrix()
        # Simple assembly (assuming sequential node numbering)
        start_idx = i * 6
        end_idx = (i + 2) * 6
        K_global[start_idx:end_idx, start_idx:end_idx] += k_elem
    
    return K_global

# ============================================================
# Stress & Strain Transformations
# ============================================================

def mohrs_circle(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
    """
    Calculate Mohr's circle parameters
    Returns: (center, radius, max_shear_stress)
    """
    center = (sigma_x + sigma_y) / 2
    radius = np.sqrt(((sigma_x - sigma_y)/2)**2 + tau_xy**2)
    return center, radius, radius

def von_mises_stress(sigma_1: float, sigma_2: float, sigma_3: float) -> float:
    """Calculate Von Mises equivalent stress"""
    return np.sqrt(0.5 * ((sigma_1 - sigma_2)**2 + 
                         (sigma_2 - sigma_3)**2 + 
                         (sigma_3 - sigma_1)**2))

def tresca_stress(sigma_1: float, sigma_2: float, sigma_3: float) -> float:
    """Calculate Tresca equivalent stress"""
    return max(abs(sigma_1 - sigma_2), abs(sigma_2 - sigma_3), abs(sigma_3 - sigma_1))

def stress_transformation_2d(sigma_x: float, sigma_y: float, tau_xy: float, theta: float) -> Tuple[float, float, float]:
    """Transform 2D stress state by angle theta (radians)"""
    c = np.cos(theta)
    s = np.sin(theta)
    
    sigma_xp = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
    sigma_yp = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
    tau_xyp = -(sigma_x - sigma_y) * s * c + tau_xy * (c**2 - s**2)
    
    return sigma_xp, sigma_yp, tau_xyp

# ============================================================
# Mechanical Vibrations
# ============================================================

@dataclass
class SDOFSystem:
    """Single Degree of Freedom system"""
    m: float  # Mass (kg)
    c: float  # Damping coefficient (N·s/m)
    k: float  # Spring stiffness (N/m)
    
    def natural_frequency(self) -> float:
        """Undamped natural frequency (rad/s)"""
        return np.sqrt(self.k / self.m)
    
    def damping_ratio(self) -> float:
        """Damping ratio"""
        return self.c / (2 * np.sqrt(self.k * self.m))
    
    def frequency_response(self, omega: float, F0: float) -> Tuple[float, float]:
        """Steady-state response amplitude and phase"""
        wn = self.natural_frequency()
        zeta = self.damping_ratio()
        r = omega / wn
        
        X = F0 / (self.k * np.sqrt((1 - r**2)**2 + (2*zeta*r)**2))
        phi = np.arctan(2*zeta*r / (1 - r**2))
        return X, phi

@dataclass
class MDOFSystem:
    """Multi-Degree of Freedom system"""
    M: np.ndarray  # Mass matrix
    K: np.ndarray  # Stiffness matrix
    C: Optional[np.ndarray] = None  # Damping matrix
    
    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform modal analysis - returns natural frequencies and mode shapes"""
        eigenvalues, eigenvectors = eigh(self.K, self.M)
        frequencies = np.sqrt(eigenvalues)
        return frequencies, eigenvectors
    
    def state_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to state-space representation"""
        n = len(self.M)
        A = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-inv(self.M) @ self.K, -inv(self.M) @ self.C]
        ])
        B = np.block([
            [np.zeros((n, n))],
            [inv(self.M)]
        ])
        return A, B

def solve_sdof_response(sys: SDOFSystem, F: Callable[[float], float], 
                       t_span: Tuple[float, float], x0: float = 0.0, v0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Solve SDOF system response using numerical integration"""
    def ode(t, y):
        x, v = y
        dxdt = v
        dvdt = (F(t) - sys.c * v - sys.k * x) / sys.m
        return [dxdt, dvdt]
    
    sol = solve_ivp(ode, t_span, [x0, v0], t_eval=np.linspace(t_span[0], t_span[1], 1000))
    return sol.t, sol.y[0]

# ============================================================
# Fatigue & Fracture Mechanics
# ============================================================

@dataclass
class SN_Curve:
    """S-N curve parameters for fatigue analysis"""
    fatigue_limit: float  # Endurance limit (MPa)
    fatigue_exponent: float  # Basquin exponent
    reference_cycles: float = 1e6  # Reference cycles for fatigue limit
    
    def cycles_to_failure(self, stress: float) -> float:
        """Calculate cycles to failure for given stress amplitude"""
        if stress <= self.fatigue_limit:
            return np.inf
        return self.reference_cycles * (stress / self.fatigue_limit) ** (-1/self.fatigue_exponent)

def goodman_diagram(sigma_m: float, sigma_a: float, S_u: float, S_e: float) -> float:
    """Goodman failure criterion - returns safety factor"""
    return 1 / (sigma_a / S_e + sigma_m / S_u)

def gerber_diagram(sigma_m: float, sigma_a: float, S_u: float, S_e: float) -> float:
    """Gerber failure criterion - returns safety factor"""
    return 1 / (sigma_a / S_e + (sigma_m / S_u)**2)

@dataclass
class ParisLaw:
    """Paris' Law for fatigue crack growth"""
    C: float  # Material constant
    m: float  # Material exponent
    threshold: float = 4.0  # Delta K threshold (MPa√m)
    
    def crack_growth_rate(self, delta_K: float) -> float:
        """Calculate da/dN for given stress intensity factor range"""
        if delta_K < self.threshold:
            return 0.0
        return self.C * (delta_K ** self.m)

# ============================================================
# Structural Optimization
# ============================================================

@dataclass
class BeamCrossSection:
    """Rectangular beam cross-section parameters"""
    width: float
    height: float
    density: float = 7850.0  # Steel density (kg/m^3)
    
    def area(self) -> float:
        return self.width * self.height
    
    def weight_per_unit_length(self) -> float:
        return self.area() * self.density
    
    def section_modulus(self) -> float:
        return (self.width * self.height**2) / 6
    
    def moment_of_inertia(self) -> float:
        return (self.width * self.height**3) / 12

def optimize_beam_section(max_stress: float, max_deflection: float, 
                         length: float, load: float) -> BeamCrossSection:
    """Optimize beam cross-section for minimum weight"""
    
    def objective(x):
        width, height = x
        section = BeamCrossSection(width, height)
        return section.weight_per_unit_length()
    
    def stress_constraint(x):
        width, height = x
        section = BeamCrossSection(width, height)
        actual_stress = load * length / (4 * section.section_modulus())
        return max_stress - actual_stress
    
    def deflection_constraint(x):
        width, height = x
        section = BeamCrossSection(width, height)
        actual_deflection = (load * length**3) / (48 * 200e9 * section.moment_of_inertia())
        return max_deflection - actual_deflection
    
    constraints = [
        {'type': 'ineq', 'fun': stress_constraint},
        {'type': 'ineq', 'fun': deflection_constraint}
    ]
    
    bounds = [(0.01, 0.5), (0.01, 0.5)]  # Width and height bounds (m)
    
    result = minimize(objective, [0.1, 0.2], bounds=bounds, constraints=constraints)
    return BeamCrossSection(result.x[0], result.x[1])

# ============================================================
# Rotordynamics
# ============================================================

@dataclass
class RotorSystem:
    """Simple Jeffcott rotor system"""
    m: float  # Rotor mass (kg)
    k: float  # Shaft stiffness (N/m)
    c: float  # Damping coefficient (N·s/m)
    omega: float = 0.0  # Operating speed (rad/s)
    
    def critical_speed(self) -> float:
        """Calculate critical speed"""
        return np.sqrt(self.k / self.m)
    
    def unbalance_response(self, e: float, omega: float) -> Tuple[float, float]:
        """Calculate unbalance response amplitude and phase"""
        wn = self.critical_speed()
        zeta = self.c / (2 * np.sqrt(self.k * self.m))
        r = omega / wn
        
        X = (e * r**2) / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
        phi = np.arctan(2*zeta*r / (1 - r**2))
        return X, phi

# ============================================================
# Contact Mechanics (Hertz Contact)
# ============================================================

def hertz_contact(R1: float, R2: float, E1: float, E2: float, 
                 nu1: float, nu2: float, F: float) -> Tuple[float, float]:
    """
    Hertz contact between two spheres
    Returns: (contact_radius, max_pressure)
    """
    # Equivalent radius
    R = (1/R1 + 1/R2)**(-1)
    
    # Equivalent modulus
    E_star = (1/E1 + 1/E2)**(-1)
    
    # Contact radius
    a = (3 * F * R / (4 * E_star))**(1/3)
    
    # Maximum pressure
    p_max = 3 * F / (2 * np.pi * a**2)
    
    return a, p_max

# ============================================================
# Failure Theories
# ============================================================

def maximum_normal_stress_theory(s1: float, s2: float, s3: float, S_ut: float) -> float:
    """Maximum Normal Stress Theory (Rankine)"""
    return S_ut / max(abs(s1), abs(s2), abs(s3))

def maximum_shear_stress_theory(s1: float, s2: float, s3: float, S_y: float) -> float:
    """Maximum Shear Stress Theory (Tresca)"""
    tau_max = max(abs(s1-s2)/2, abs(s2-s3)/2, abs(s3-s1)/2)
    return S_y / (2 * tau_max)

def distortion_energy_theory(s1: float, s2: float, s3: float, S_y: float) -> float:
    """Distortion Energy Theory (Von Mises)"""
    s_vm = von_mises_stress(s1, s2, s3)
    return S_y / s_vm

# ============================================================
# Main Demonstration
# ============================================================

if __name__ == '__main__':
    print("=== Mechanical Engineering Analysis Suite ===\n")
    
    # 1. Finite Element Analysis
    print("1. Finite Element Analysis")
    truss = TrussElement(E=200e9, A=0.01, L=2.0)
    print(f"Truss element stiffness matrix shape: {truss.stiffness_matrix().shape}")
    
    beam = BeamElement(E=200e9, I=1e-4, A=0.01, L=2.0)
    print(f"Beam element stiffness matrix shape: {beam.stiffness_matrix().shape}\n")
    
    # 2. Stress Transformations
    print("2. Stress Transformations")
    sigma_x, sigma_y, tau_xy = 100e6, 50e6, 30e6
    center, radius, max_shear = mohrs_circle(sigma_x, sigma_y, tau_xy)
    print(f"Mohr's circle center: {center/1e6:.1f} MPa, radius: {radius/1e6:.1f} MPa")
    
    s_vm = von_mises_stress(sigma_x, sigma_y, 0)
    print(f"Von Mises stress: {s_vm/1e6:.1f} MPa\n")
    
    # 3. Vibrations
    print("3. Mechanical Vibrations")
    sdof = SDOFSystem(m=100, c=50, k=10000)
    wn = sdof.natural_frequency()
    print(f"SDOF natural frequency: {wn:.2f} rad/s")
    
    # 4. Fatigue Analysis
    print("\n4. Fatigue Analysis")
    sn_curve = SN_Curve(fatigue_limit=200, fatigue_exponent=3)
    cycles = sn_curve.cycles_to_failure(300)
    print(f"Cycles to failure at 300 MPa: {cycles:.0f}")
    
    # 5. Optimization
    print("\n5. Structural Optimization")
    optimal_section = optimize_beam_section(max_stress=200e6, max_deflection=0.01, 
                                           length=2.0, load=10000)
    print(f"Optimal beam section: {optimal_section.width:.3f} m x {optimal_section.height:.3f} m")
    print(f"Weight per unit length: {optimal_section.weight_per_unit_length():.1f} kg/m")
    
    # 6. Rotordynamics
    print("\n6. Rotordynamics")
    rotor = RotorSystem(m=10, k=1e6, c=100)
    critical_speed = rotor.critical_speed()
    print(f"Critical speed: {critical_speed:.2f} rad/s")
    
    # 7. Contact Mechanics
    print("\n7. Contact Mechanics")
    a, p_max = hertz_contact(R1=0.1, R2=0.1, E1=200e9, E2=200e9, 
                            nu1=0.3, nu2=0.3, F=1000)
    print(f"Contact radius: {a*1000:.2f} mm, Max pressure: {p_max/1e6:.1f} MPa")
    
    print("\n=== Analysis Complete ===")
    
    # Generate sample plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: SDOF Frequency Response
    frequencies = np.logspace(0, 2, 100)
    amplitudes = []
    for freq in frequencies:
        X, _ = sdof.frequency_response(freq, 1000)
        amplitudes.append(X)
    
    axes[0, 0].semilogx(frequencies, amplitudes)
    axes[0, 0].set_xlabel('Frequency (rad/s)')
    axes[0, 0].set_ylabel('Amplitude (m)')
    axes[0, 0].set_title('SDOF Frequency Response')
    axes[0, 0].grid(True)
    
    # Plot 2: Goodman Diagram
    sigmas = np.linspace(0, 200, 50)
    goodman_line = []
    for sigma_m in sigmas:
        goodman_line.append(goodman_diagram(sigma_m, 50, 400, 200))
    
    axes[0, 1].plot(sigmas, goodman_line)
    axes[0, 1].set_xlabel('Mean Stress (MPa)')
    axes[0, 1].set_ylabel('Allowable Stress Amplitude (MPa)')
    axes[0, 1].set_title('Goodman Diagram')
    axes[0, 1].grid(True)
    
    # Plot 3: S-N Curve
    stresses = np.linspace(100, 500, 100)
    cycles = [sn_curve.cycles_to_failure(s) for s in stresses]
    
    axes[1, 0].semilogy(stresses, cycles)
    axes[1, 0].set_xlabel('Stress Amplitude (MPa)')
    axes[1, 0].set_ylabel('Cycles to Failure')
    axes[1, 0].set_title('S-N Curve')
    axes[1, 0].grid(True)
    
    # Plot 4: Rotor Unbalance Response
    speeds = np.linspace(0, 2*rotor.critical_speed(), 100)
    responses = []
    for speed in speeds:
        X, _ = rotor.unbalance_response(0.01, speed)
        responses.append(X)
    
    axes[1, 1].plot(speeds, responses)
    axes[1, 1].set_xlabel('Operating Speed (rad/s)')
    axes[1, 1].set_ylabel('Vibration Amplitude (m)')
    axes[1, 1].set_title('Rotor Unbalance Response')
    axes[1, 1].axvline(rotor.critical_speed(), color='r', linestyle='--', label='Critical Speed')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()