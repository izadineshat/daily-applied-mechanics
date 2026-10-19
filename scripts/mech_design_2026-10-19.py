"""
Mechanical Engineering Toolkit: Applied Mechanics, FEA, Vibrations, and Design
Production-ready Python module for mechanical engineering calculations.
"""

import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable

# ==============================================================================
# Finite Element Analysis (FEA) Module
# ==============================================================================

class TrussElement:
    """1D truss element with stiffness matrix formulation."""
    
    def __init__(self, nodes: np.ndarray, E: float, A: float):
        """
        Initialize truss element.
        
        Args:
            nodes: Array of shape (2, 2) containing [x, y] coordinates for nodes 1 and 2
            E: Young's modulus (Pa)
            A: Cross-sectional area (m²)
        """
        self.nodes = np.asarray(nodes)
        self.E = E
        self.A = A
        self.length = self._compute_length()
        self.direction_cosines = self._compute_direction_cosines()
        self.local_stiffness = self._compute_local_stiffness()
        self.global_stiffness = self._compute_global_stiffness()
    
    def _compute_length(self) -> float:
        return np.linalg.norm(self.nodes[1] - self.nodes[0])
    
    def _compute_direction_cosines(self) -> np.ndarray:
        diff = self.nodes[1] - self.nodes[0]
        return diff / self.length
    
    def _compute_local_stiffness(self) -> np.ndarray:
        k_local = (self.E * self.A / self.length) * np.array([
            [1, -1],
            [-1, 1]
        ])
        return k_local
    
    def _compute_global_stiffness(self) -> np.ndarray:
        c, s = self.direction_cosines
        T = np.array([
            [c, s, 0, 0],
            [0, 0, c, s]
        ])
        k_global = T.T @ self.local_stiffness @ T
        return k_global

class BeamElement:
    """Euler-Bernoulli beam element for bending analysis."""
    
    def __init__(self, length: float, E: float, I: float, A: float):
        """
        Initialize beam element.
        
        Args:
            length: Element length (m)
            E: Young's modulus (Pa)
            I: Second moment of area (m⁴)
            A: Cross-sectional area (m²)
        """
        self.length = length
        self.E = E
        self.I = I
        self.A = A
        self.stiffness = self._compute_stiffness()
    
    def _compute_stiffness(self) -> np.ndarray:
        L = self.length
        k = (self.E * self.I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

def assemble_truss_structure(elements: List[TrussElement], n_nodes: int) -> np.ndarray:
    """
    Assemble global stiffness matrix for truss structure.
    
    Args:
        elements: List of TrussElement instances
        n_nodes: Total number of nodes in structure
    
    Returns:
        Global stiffness matrix (2*n_nodes × 2*n_nodes)
    """
    K_global = np.zeros((2 * n_nodes, 2 * n_nodes))
    
    for i, element in enumerate(elements):
        node1, node2 = i, i + 1  # Assuming sequential numbering
        idx = np.array([
            2*node1, 2*node1+1,
            2*node2, 2*node2+1
        ])
        K_global[np.ix_(idx, idx)] += element.global_stiffness
    
    return K_global

# ==============================================================================
# Stress & Strain Transformations Module
# ==============================================================================

@dataclass
class StressState:
    """Represents a 3D stress state."""
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    sigma_z: float = 0.0
    tau_xy: float = 0.0
    tau_yz: float = 0.0
    tau_xz: float = 0.0
    
    def to_tensor(self) -> np.ndarray:
        return np.array([
            [self.sigma_x, self.tau_xy, self.tau_xz],
            [self.tau_xy, self.sigma_y, self.tau_yz],
            [self.tau_xz, self.tau_yz, self.sigma_z]
        ])
    
    def principal_stresses(self) -> np.ndarray:
        """Compute principal stresses using eigenvalue decomposition."""
        stress_tensor = self.to_tensor()
        eigenvalues = eig(stress_tensor, left=False, right=False)[0]
        return np.sort(eigenvalues)[::-1]  # Sort descending

def mohrs_circle_2d(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
    """
    Compute Mohr's circle parameters for 2D stress state.
    
    Returns:
        center: Center of Mohr's circle
        radius: Radius of Mohr's circle
        max_shear: Maximum shear stress
    """
    center = (sigma_x + sigma_y) / 2
    radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
    max_shear = radius
    return center, radius, max_shear

def von_mises_stress(stress_state: StressState) -> float:
    """Compute von Mises equivalent stress."""
    s = stress_state
    return np.sqrt(0.5 * ((s.sigma_x - s.sigma_y)**2 + 
                          (s.sigma_y - s.sigma_z)**2 + 
                          (s.sigma_z - s.sigma_x)**2 + 
                          6 * (s.tau_xy**2 + s.tau_yz**2 + s.tau_xz**2)))

def tresca_stress(stress_state: StressState) -> float:
    """Compute Tresca (maximum shear) equivalent stress."""
    principal = stress_state.principal_stresses()
    return principal[0] - principal[2]

def stress_transformation_2d(sigma_x: float, sigma_y: float, tau_xy: float, 
                            theta: float) -> Tuple[float, float, float]:
    """
    Transform 2D stress components by rotation angle theta (radians).
    
    Returns:
        Transformed (sigma_x', sigma_y', tau_xy')
    """
    c = np.cos(2*theta)
    s = np.sin(2*theta)
    
    sigma_xp = (sigma_x + sigma_y)/2 + (sigma_x - sigma_y)/2 * c + tau_xy * s
    sigma_yp = (sigma_x + sigma_y)/2 - (sigma_x - sigma_y)/2 * c - tau_xy * s
    tau_xyp = -(sigma_x - sigma_y)/2 * s + tau_xy * c
    
    return sigma_xp, sigma_yp, tau_xyp

# ==============================================================================
# Mechanical Vibrations Module
# ==============================================================================

@dataclass
class MassSpringDamper:
    """SDOF mass-spring-damper system."""
    mass: float
    stiffness: float
    damping: float = 0.0
    
    def natural_frequency(self) -> float:
        return np.sqrt(self.stiffness / self.mass)
    
    def damping_ratio(self) -> float:
        return self.damping / (2 * np.sqrt(self.stiffness * self.mass))
    
    def sdof_response(self, F0: float, omega: float, t: float) -> float:
        """
        Compute steady-state response to harmonic excitation F0*cos(omega*t).
        """
        wn = self.natural_frequency()
        zeta = self.damping_ratio()
        r = omega / wn
        
        # Static deflection
        x_static = F0 / self.stiffness
        
        # Magnification factor
        M = 1 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
        
        # Phase angle
        phi = np.arctan2(2*zeta*r, 1 - r**2)
        
        return x_static * M * np.cos(omega*t - phi)

class MDOFSystem:
    """Multi-degree-of-freedom vibration system."""
    
    def __init__(self, mass_matrix: np.ndarray, stiffness_matrix: np.ndarray,
                 damping_matrix: Optional[np.ndarray] = None):
        """
        Initialize MDOF system.
        
        Args:
            mass_matrix: Mass matrix (n×n)
            stiffness_matrix: Stiffness matrix (n×n)
            damping_matrix: Damping matrix (n×n), optional (default: proportional)
        """
        self.M = np.asarray(mass_matrix)
        self.K = np.asarray(stiffness_matrix)
        self.n_dof = self.M.shape[0]
        
        if damping_matrix is None:
            # Rayleigh damping (proportional to M and K)
            self.C = 0.001 * self.M + 0.001 * self.K
        else:
            self.C = np.asarray(damping_matrix)
    
    def modal_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform modal analysis.
        
        Returns:
            natural_frequencies: Array of natural frequencies (Hz)
            mode_shapes: Mode shape matrix (n×n)
        """
        eigenvalues, eigenvectors = eig(np.linalg.solve(self.M, self.K))
        natural_frequencies = np.sqrt(np.real(eigenvalues)) / (2 * np.pi)
        
        # Sort by frequency
        idx = np.argsort(natural_frequencies)
        return natural_frequencies[idx], eigenvectors[:, idx]
    
    def state_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to state-space representation.
        
        Returns:
            A: State matrix (2n×2n)
            B: Input matrix (2n×m)
        """
        n = self.n_dof
        M_inv = np.linalg.inv(self.M)
        
        A = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-M_inv @ self.K, -M_inv @ self.C]
        ])
        
        B = np.block([
            [np.zeros((n, n))],
            [M_inv]
        ])
        
        return A, B

# ==============================================================================
# Fatigue & Fracture Mechanics Module
# ==============================================================================

@dataclass
class SNParameters:
    """S-N curve parameters for fatigue analysis."""
    fatigue_limit: float  # Endurance limit (MPa)
    fatigue_exponent: float  # Basquin exponent
    fatigue_coefficient: float  # Fatigue strength coefficient

def GoodmanDiagram(sigma_m: float, sigma_a: float, S_u: float, S_e: float) -> float:
    """
    Goodman fatigue criterion.
    
    Args:
        sigma_m: Mean stress (MPa)
        sigma_a: Alternating stress amplitude (MPa)
        S_u: Ultimate tensile strength (MPa)
        S_e: Endurance limit (MPa)
    
    Returns:
        Safety factor against fatigue failure
    """
    if sigma_m <= 0:  # Compressive mean stress
        return S_e / sigma_a if sigma_a > 0 else np.inf
    
    return 1 / (sigma_a/S_e + sigma_m/S_u)

def GerberDiagram(sigma_m: float, sigma_a: float, S_u: float, S_e: float) -> float:
    """Gerber parabolic fatigue criterion."""
    if sigma_m <= 0:
        return S_e / sigma_a if sigma_a > 0 else np.inf
    
    term = (sigma_m / S_u)**2
    return 1 / (sigma_a/S_e * (1 - term) + term)

def ParisLaw(C: float, m: float, delta_K: float, delta_K_th: float, K_c: float) -> float:
    """
    Paris' Law for fatigue crack growth.
    
    Args:
        C: Paris coefficient
        m: Paris exponent
        delta_K: Stress intensity factor range (MPa√m)
        delta_K_th: Threshold stress intensity range
        K_c: Fracture toughness
    
    Returns:
        Crack growth rate da/dN (m/cycle)
    """
    if delta_K < delta_K_th:
        return 0.0
    elif delta_K > K_c:
        return np.inf  # Unstable fracture
    
    return C * (delta_K)**m

# ==============================================================================
# Structural Optimization Module
# ==============================================================================

class BeamCrossSectionOptimizer:
    """Optimize beam cross-section for minimum weight under constraints."""
    
    def __init__(self, length: float, load: float, E: float, sigma_allow: float):
        """
        Initialize beam optimizer.
        
        Args:
            length: Beam span (m)
            load: Maximum bending load (N)
            E: Young's modulus (Pa)
            sigma_allow: Allowable stress (Pa)
        """
        self.length = length
        self.load = load
        self.E = E
        self.sigma_allow = sigma_allow
    
    def rectangular_section_objective(self, x: np.ndarray) -> float:
        """Objective function for rectangular cross-section (minimize area)."""
        b, h = x
        return b * h  # Area to minimize
    
    def rectangular_section_constraint(self, x: np.ndarray) -> float:
        """Stress constraint for rectangular section."""
        b, h = x
        I = b * h**3 / 12  # Second moment of area
        M_max = self.load * self.length / 4  # Maximum bending moment (simply supported)
        sigma_max = M_max * (h/2) / I
        return self.sigma_allow - sigma_max
    
    def optimize_rectangular(self) -> Tuple[np.ndarray, float]:
        """Optimize rectangular cross-section dimensions."""
        # Initial guess
        x0 = np.array([0.1, 0.1])
        
        # Bounds (minimum practical dimensions)
        bounds = [(0.01, 1.0), (0.01, 1.0)]
        
        # Constraints
        constraints = {'type': 'ineq', 'fun': self.rectangular_section_constraint}
        
        result = minimize(
            self.rectangular_section_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x, result.fun

# ==============================================================================
# Rotordynamics & Contact Mechanics Module
# ==============================================================================

class RotorSystem:
    """Simple Jeffcott rotor (Laval rotor) model."""
    
    def __init__(self, mass: float, stiffness: float, damping: float, 
                 disk_mass: float, disk_radius: float):
        """
        Initialize rotor system.
        
        Args:
            mass: Total rotor mass (kg)
            stiffness: Shaft stiffness (N/m)
            damping: Damping coefficient (Ns/m)
            disk_mass: Disk mass (kg)
            disk_radius: Disk radius (m)
        """
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.disk_mass = disk_mass
        self.disk_radius = disk_radius
    
    def critical_speed(self) -> float:
        """Compute critical speed (rad/s)."""
        return np.sqrt(self.stiffness / self.mass)
    
    def unbalance_response(self, omega: float, eccentricity: float) -> float:
        """
        Compute steady-state amplitude due to unbalance.
        
        Args:
            omega: Operating speed (rad/s)
            eccentricity: Mass unbalance eccentricity (m)
        
        Returns:
            Vibration amplitude (m)
        """
        wn = self.critical_speed()
        r = omega / wn
        zeta = self.damping / (2 * np.sqrt(self.stiffness * self.mass))
        
        # Magnification factor for unbalance
        M = r**2 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
        return M * (self.disk_mass * eccentricity) / self.mass

def HertzianContact(E1: float, E2: float, nu1: float, nu2: float, 
                   R1: float, R2: float, F: float) -> Tuple[float, float, float]:
    """
    Compute Hertzian line contact parameters.
    
    Args:
        E1, E2: Young's moduli (Pa)
        nu1, nu2: Poisson's ratios
        R1, R2: Radii of curvature (m)
        F: Applied force per unit length (N/m)
    
    Returns:
        contact_half_width: Half-width of contact area (m)
        max_pressure: Maximum contact pressure (Pa)
        contact_length: Contact length (m)
    """
    # Equivalent modulus
    E_prime = 1 / ((1 - nu1**2)/E1 + (1 - nu2**2)/E2)
    
    # Equivalent radius
    R_prime = 1 / (1/R1 + 1/R2)
    
    # Contact half-width
    a = np.sqrt(4 * F * R_prime / (np.pi * E_prime))
    
    # Maximum pressure
    p_max = 2 * F / (np.pi * a)
    
    # Contact length (for line contact)
    L = 2 * a  # This is actually the contact width, but for line contact it's the same
    
    return a, p_max, L

# ==============================================================================
# Main Demonstration
# ==============================================================================

def main():
    """Demonstrate the mechanical engineering toolkit."""
    
    print("=" * 60)
    print("Mechanical Engineering Toolkit Demonstration")
    print("=" * 60)
    
    # 1. Truss Analysis
    print("\n1. Truss Element Analysis")
    nodes = np.array([[0, 0], [1, 0]])  # 1m horizontal truss
    truss = TrussElement(nodes, E=200e9, A=0.01)  # Steel, 100cm² area
    print(f"Truss length: {truss.length:.2f} m")
    print(f"Local stiffness:\n{truss.local_stiffness}")
    print(f"Global stiffness:\n{truss.global_stiffness}")
    
    # 2. Stress Transformation
    print("\n2. Stress State Analysis")
    stress = StressState(sigma_x=100, sigma_y=50, tau_xy=30)
    principal = stress.principal_stresses()
    vm = von_mises_stress(stress)
    tc = tresca_stress(stress)
    
    print(f"Principal stresses: {principal} MPa")
    print(f"Von Mises stress: {vm:.2f} MPa")
    print(f"Tresca stress: {tc:.2f} MPa")
    
    center, radius, max_shear = mohrs_circle_2d(100, 50, 30)
    print(f"Mohr's circle center: {center:.2f} MPa")
    print(f"Mohr's circle radius: {radius:.2f} MPa")
    print(f"Max shear stress: {max_shear:.2f} MPa")
    
    # 3. Vibration Analysis
    print("\n3. Mechanical Vibrations")
    sdof = MassSpringDamper(mass=10.0, stiffness=10000, damping=100)
    wn = sdof.natural_frequency()
    zeta = sdof.damping_ratio()
    print(f"SDOF natural frequency: {wn:.2f} rad/s")
    print(f"Damping ratio: {zeta:.4f}")
    
    # MDOF system
    M = np.array([[1, 0], [0, 1]])
    K = np.array([[2000, -1000], [-1000, 2000]])
    mdof = MDOFSystem(M, K)
    freqs, modes = mdof.modal_analysis()
    print(f"MDOF natural frequencies: {freqs} Hz")
    
    # 4. Fatigue Analysis
    print("\n4. Fatigue Analysis")
    sigma_m, sigma_a = 50, 100  # Mean and alternating stresses
    S_u, S_e = 400, 200  # Ultimate and endurance limits
    safety_goodman = GoodmanDiagram(sigma_m, sigma_a, S_u, S_e)
    safety_gerber = GerberDiagram(sigma_m, sigma_a, S_u, S_e)
    print(f"Goodman safety factor: {safety_goodman:.3f}")
    print(f"Gerber safety factor: {safety_gerber:.3f}")
    
    # 5. Optimization
    print("\n5. Structural Optimization")
    optimizer = BeamCrossSectionOptimizer(length=2.0, load=1000, E=200e9, sigma_allow=150e6)
    optimal_dims, min_area = optimizer.optimize_rectangular()
    print(f"Optimal rectangular section: {optimal_dims[0]:.3f} m × {optimal_dims[1]:.3f} m")
    print(f"Minimum area: {min_area:.4f} m²")
    
    # 6. Rotordynamics
    print("\n6. Rotordynamics")
    rotor = RotorSystem(mass=50, stiffness=1e6, damping=100, disk_mass=10, disk_radius=0.1)
    critical_speed = rotor.critical_speed()
    print(f"Rotor critical speed: {critical_speed:.2f} rad/s")
    
    # 7. Contact Mechanics
    print("\n7. Contact Mechanics")
    a, p_max, L = HertzianContact(E1=200e9, E2=200e9, nu1=0.3, nu2=0.3,
                                 R1=0.05, R2=0.05, F=1000)
    print(f"Contact half-width: {a*1000:.2f} mm")
    print(f"Maximum pressure: {p_max/1e6:.2f} MPa")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

if __name__ == '__main__':
    main()