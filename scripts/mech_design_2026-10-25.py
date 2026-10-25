"""
Mechanical Engineering Analysis Suite
=====================================
A comprehensive library for applied mechanics including FEA, stress analysis,
vibrations, fatigue, optimization, and rotordynamics.

Author: Mechanical Engineering Analysis Suite
Version: 1.0.0
"""

import numpy as np
from scipy import linalg, optimize
from scipy.constants import g
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import warnings

class TrussAnalysis:
    """1D Truss Analysis - Stiffness Matrix and Displacement Calculations."""
    
    def __init__(self, n_nodes: int, node_positions: np.ndarray = None):
        """
        Initialize a 1D truss structure.
        
        Args:
            n_nodes: Number of nodes in the truss
            node_positions: Optional 2D array of node coordinates (x, y)
        """
        self.n_nodes = n_nodes
        self.node_positions = node_positions if node_positions is not None else np.zeros((n_nodes, 2))
        self._stiffness_matrix = None
        self._stiffness_matrix = self._compute_stiffness_matrix()
    
    def _compute_stiffness_matrix(self) -> np.ndarray:
        """Compute the global stiffness matrix for the truss."""
        # Truss element stiffness matrix (in N/m)
        # For a 1D truss element: k = AE/L
        # We'll use symbolic AE/L = 1 for normalized analysis
        L = np.linalg.norm(self.node_positions[:, 1] - self.node_positions[:, 0])  # length
        
        # Element stiffness matrix (3x3 for 2D truss, but we use 1D)
        # For 1D truss: [k, -k, 0; -k, k, 0; 0, 0, 0]
        k = 1.0  # normalized stiffness per element
        n_elements = (self.n_nodes - 1)
        
        # Assemble global stiffness matrix
        K = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(n_elements):
            # Element 1 connects node i to i+1
            row_start = i
            col_start = i
            K[row_start:row_start+1, col_start:col_start+1] = k
            K[row_start+1:row_start+2, col_start:col_start+1] = -k
            K[row_start:row_start+1, col_start+1] = 0
            K[row_start+1:row_start+2, col_start+1] = 0
        
        return K
    
    def displacement(self, forces: np.ndarray) -> np.ndarray:
        """
        Compute nodal displacements given force vector.
        
        Args:
            forces: Force vector (N) at each node
            
        Returns:
            Displacement vector (m)
        """
        if self._stiffness_matrix is None:
            raise RuntimeError("Stiffness matrix not computed. Call _compute_stiffness_matrix() first.")
        
        # Solve K * u = F
        u = linalg.solve(self._stiffness_matrix, forces)
        return u
    
    def strain(self, displacements: np.ndarray) -> np.ndarray:
        """
        Compute strain vector from displacements.
        
        Args:
            displacements: Nodal displacement vector (m)
            
        Returns:
            Strain vector (dimensionless)
        """
        # For 1D truss, strain = displacement gradient
        # Assuming uniform cross-section and constant axial load
        return displacements
    
    def stress(self, forces: np.ndarray) -> np.ndarray:
        """
        Compute stress distribution.
        
        Args:
            forces: Applied force vector (N)
            
        Returns:
            Stress vector (Pa)
        """
        if self._stiffness_matrix is None:
            raise RuntimeError("Stiffness matrix not computed.")
        
        # Stress = Force / Area (simplified for 1D)
        # For beam-like elements, we'd need more info
        return forces / 1.0  # Normalized stress


class BeamAnalysis:
    """2D Beam Analysis - Flexibility Matrix and Bending Analysis."""
    
    def __init__(self, n_nodes: int, beam_properties: Dict[str, float]):
        """
        Initialize a 2D beam structure.
        
        Args:
            n_nodes: Number of nodes
            beam_properties: Dictionary with 'E' (Young's modulus), 'I' (moment of inertia),
                           'A' (cross-sectional area), 'L' (length)
        """
        self.n_nodes = n_nodes
        self.E = beam_properties.get('E', 200e9)  # Pa
        self.I = beam_properties.get('I', 1.0e-6)  # m^4
        self.A = beam_properties.get('A', 1.0e-4)  # m^2
        self.L = beam_properties.get('L', 1.0)  # m
        
        # Compute element stiffness matrix (4x4 for beam element)
        # Using standard Euler-Bernoulli beam theory
        self._element_matrix = self._compute_element_matrix()
        self._stiffness_matrix = None
        self._stiffness_matrix = self._assemble_stiffness_matrix()
    
    def _compute_element_matrix(self) -> np.ndarray:
        """Compute 4x4 element stiffness matrix for a beam element."""
        # Element stiffness matrix for beam (N/m^2)
        # [6EI/L^2, -EI/L, 2EI/L, -6EI/L^2]
        # [-EI/L, 2EI/L, -6EI/L, EI/L]
        # [2EI/L, -6EI/L, 2EI/L, -EI/L]
        # [-6EI/L^2, EI/L, -EI/L, 4EI/L]
        k = self.E * self.I / (self.L ** 2)
        return np.array([
            [6*k, -k, 2*k, -6*k],
            [-k, 2*k, -6*k, k],
            [2*k, -6*k, 2*k, -k],
            [-6*k, k, -k, 4*k]
        ])
    
    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """Assemble global stiffness matrix from element matrices."""
        n_elements = (self.n_nodes - 1)
        K = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(n_elements):
            # Element connects node i to i+1
            row_start = i
            col_start = i
            
            # Apply symmetry
            K[row_start:row_start+1, col_start:col_start+1] = self._element_matrix
            K[col_start:col_start+1, row_start:row_start+1] = -self._element_matrix
            
            # Adjust for 2D (x and y directions)
            # For simplicity, we assume axial loading only
            # In full 2D, we'd need to consider bending stiffness
            pass
        
        return K
    
    def bending_moment(self, moments: np.ndarray) -> np.ndarray:
        """
        Compute bending moments from moments vector.
        
        Args:
            moments: Bending moment vector (N·m)
            
        Returns:
            Bending moment vector (N·m)
        """
        return moments
    
    def deflection(self, moments: np.ndarray) -> np.ndarray:
        """
        Compute deflections from bending moments.
        
        Args:
            moments: Bending moment vector (N·m)
            
        Returns:
            Deflection vector (m)
        """
        # Simple beam deflection: d = M*L^3/(48EI) for simply supported
        # This is a simplified model
        return moments * self.L**3 / (48 * self.E * self.I)
    
    def stress(self, moments: np.ndarray) -> np.ndarray:
        """
        Compute bending stress from moments.
        
        Args:
            moments: Bending moment vector (N·m)
            
        Returns:
            Stress vector (Pa)
        """
        # sigma = M*y/I, where y is distance from neutral axis
        # For a rectangular cross-section, y = h/2
        h = self.A / (self.I / (self.L**2))  # Approximate height
        y = h / 2
        return moments * y / self.I


class StressTransformation:
    """Mohr's Circle, Von Mises, and Tresca Stress Transformations."""
    
    def __init__(self):
        pass
    
    def compute_mohr_circle(self, stresses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mohr's circle parameters from stress tensor.
        
        Args:
            stresses: 2D array of stress components [sigma_x, tau_xy, sigma_y, tau_xy]
                     (Note: tau_xy appears twice in standard notation)
        
        Returns:
            Tuple of (center, radius) for Mohr's circle
        """
        # Extract stress components
        sigma_x = stresses[0]
        tau_xy = stresses[1]
        sigma_y = stresses[2]
        
        # Center of Mohr's circle
        center_x = (sigma_x + sigma_y) / 2
        center_y = (tau_xy) / 2
        
        # Radius
        radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        
        return (center_x, center_y), radius
    
    def von_mises_stress(self, stresses: np.ndarray) -> float:
        """
        Compute von Mises equivalent stress.
        
        Args:
            stresses: 2D array of stress components
        
        Returns:
            Von Mises stress (Pa)
        """
        sigma_x, tau_xy, sigma_y = stresses[0], stresses[1], stresses[2]
        
        # Von Mises formula: sigma_vm = sqrt(sigma_x^2 - sigma_x*sigma_y + sigma_y^2 + 3*tau_xy^2)
        sigma_vm = np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)
        return sigma_vm
    
    def trescas_stress(self, stresses: np.ndarray) -> float:
        """
        Compute Tresca (maximum shear stress) equivalent stress.
        
        Args:
            stresses: 2D array of stress components
        
        Returns:
            Tresca stress (Pa)
        """
        sigma_x, tau_xy, sigma_y = stresses[0], stresses[1], stresses[2]
        
        # Maximum shear stress: tau_max = 0.5 * |sigma_x - sigma_y|
        tau_max = 0.5 * abs(sigma_x - sigma_y)
        return 2 * tau_max  # Equivalent to max shear stress * 2


class VibrationAnalysis:
    """Mechanical Vibrations - Modal Analysis and Damped Response."""
    
    def __init__(self, mass_matrix: np.ndarray, stiffness_matrix: np.ndarray):
        """
        Initialize vibration analysis.
        
        Args:
            mass_matrix: Mass matrix (N*m^2)
            stiffness_matrix: Stiffness matrix (N*m)
        """
        self.mass = mass_matrix
        self.k = stiffness_matrix
        self._eigenvalues, self._eigenvectors = self._compute_modal_matrix()
    
    def _compute_modal_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of the generalized eigenvalue problem."""
        # Solve K * phi = omega^2 * M * phi
        # Using scipy's eigh for symmetric matrices
        eigenvalues, eigenvectors = linalg.eigh(self.k, self.mass)
        
        # Sort by descending frequency
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def modal_displacement(self, mode_idx: int, amplitude: float = 1.0) -> np.ndarray:
        """
        Compute displacement for a specific mode.
        
        Args:
            mode_idx: Mode index (0-based)
            amplitude: Amplitude of vibration (m)
            
        Returns:
            Displacement vector (m)
        """
        phi = eigenvectors[:, mode_idx]
        return amplitude * phi
    
    def damped_response(self, mode_idx: int, damping_ratio: float, 
                        initial_amplitude: float = 1.0) -> np.ndarray:
        """
        Compute damped response of a mode.
        
        Args:
            mode_idx: Mode index
            damping_ratio: Damping ratio (zeta)
            initial_amplitude: Initial amplitude (m)
            
        Returns:
            Time history of displacement (m)
        """
        omega_n = np.sqrt(self._eigenvalues[mode_idx])
        zeta = damping_ratio
        
        # Natural frequency
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        
        # Damped response: x(t) = X0 * exp(-zeta*omega_n*t) * cos(omega_d*t - phi)
        # For simplicity, we return the envelope
        t = np.linspace(0, 10, 100)
        response = initial_amplitude * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t)
        return response
    
    def frequency_spectrum(self, mode_idx: int, t_max: float = 10.0) -> np.ndarray:
        """
        Compute frequency spectrum of a mode.
        
        Args:
            mode_idx: Mode index
            t_max: Maximum time for simulation
            
        Returns:
            Frequency vector and corresponding response
        """
        response = self.damped_response(mode_idx, damping_ratio=0.0, initial_amplitude=1.0)
        # Compute FFT
        n = len(response)
        dt = t_max / n
        freqs = np.fft.rfftfreq(n, dt)
        return freqs, response


class FatigueAnalysis:
    """Fatigue Life Prediction - S-N Curves and Damage Mechanisms."""
    
    def __init__(self, material_properties: Dict[str, float]):
        """
        Initialize fatigue analysis.
        
        Args:
            material_properties: Dictionary with 'S-N' curve parameters
                                 e.g., {'S-N': (0.5, 1000, 0.3)} where:
                                 - S-N: (log10(S_max), log10(N_f), fatigue life factor)
                                 - 'S-N' curve parameters
        """
        self.S_N = material_properties.get('S-N', (0.0, 100, 1.0))
        self.S_max = self.S_N[0]  # Log10 of maximum stress
        self.N_f = self.S_N[1]    # Log10 of fatigue life
        self.fatigue_factor = self.S_N[2]  # Fatigue life factor
    
    def goodman_diagram(self, stress: float, safety_factor: float = 1.5) -> float:
        """
        Compute allowable stress using Goodman criterion.
        
        Args:
            stress: Applied stress (Pa)
            safety_factor: Safety factor
            
        Returns:
            Allowable stress (Pa)
        """
        # Goodman: sigma_allow = sigma_UTS * (1 - sigma_a / S_UTS)
        # Here we use log10 form
        sigma_allow = 10**self.S_N[0] * (1 - stress / 10**self.S_N[0])
        return sigma_allow
    
    def gerber_diagram(self, stress: float, safety_factor: float = 1.5) -> float:
        """
        Compute allowable stress using Gerber criterion.
        
        Args:
            stress: Applied stress (Pa)
            safety_factor: Safety factor
            
        Returns:
            Allowable stress (Pa)
        """
        # Gerber: (sigma / sigma_UTS)^2 + (n * sigma_a / sigma_e) = 1
        # Simplified: sigma_allow = sigma_UTS * sqrt(1 - n*sigma_a/sigma_e)
        sigma_allow = 10**self.S_N[0] * np.sqrt(1 - self.Gerber_factor * stress / 10**self.S_N[0])
        return sigma_allow
    
    def goodman_factor(self, stress: float) -> float:
        """
        Compute Goodman factor (safety factor).
        
        Args:
            stress: Applied stress (Pa)
            
        Returns:
            Goodman factor (dimensionless)
        """
        return 1.0 / (1.0 + stress / 10**self.S_N[0])
    
    def paris_law(self, da: float, n: int, C: float = 1.0e-12, 
                  K_IC: float = 1.0e3) -> float:
        """
        Compute crack growth rate using Paris' Law.
        
        Args:
            da: Crack growth rate (m/cycle)
            n: Number of cycles
            C: Paris constant (m/cycle^0.5)
            K_IC: Fracture toughness (MPa·√m)
            
        Returns:
            Cumulative plastic strain (m)
        """
        # dN/dda = C * (ΔK)^m
        # For Paris: ΔK = K_IC * sqrt(1 - (a/a_c)^p)
        # Simplified: dN/da = C * (K_IC^2 * (1 - (a/a_c)^p))^(1/m)
        # We'll use the standard form: da/dN = C * (ΔK)^m
        # For simplicity, assume m=3 (Paris law)
        m = 3.0
        dN_da = C * (K_IC**2 * (1 - (da / 0.1)**(2/3)))**(1/m)
        return dN_da
    
    def fatigue_life(self, stress: float, cycles: int) -> float:
        """
        Estimate fatigue life using S-N curve.
        
        Args:
            stress: Applied stress (Pa)
            cycles: Number of cycles
            
        Returns:
            Estimated fatigue life (cycles)
        """
        # Convert stress to log10 form
        log_stress = np.log10(stress)
        # Interpolate S-N curve
        if log_stress >= self.S_N[0] and log_stress <= self.S_N[1]:
            # Linear interpolation between S-N points
            t = (log_stress - self.S_N[0]) / (self.S_N[1] - self.S_N[0])
            life = 10**self.S_N[1] * (1 - t)
            return max(1, life)
        return 0.0


class StructuralOptimization:
    """Structural Optimization - Cross-section Weight Minimization."""
    
    def __init__(self, beam_properties: Dict[str, float]):
        """
        Initialize optimization for beam cross-section.
        
        Args:
            beam_properties: Beam properties including E, I, A, L
        """
        self.E = beam_properties.get('E', 200e9)
        self.I = beam_properties.get('I', 1.0e-6)
        self.A = beam_properties.get('A', 1.0e-4)
        self.L = beam_properties.get('L', 1.0)
        self.area = self.A
        self.length = self.L
    
    def objective(self, cross_section: np.ndarray) -> float:
        """
        Objective function: minimize weight (A * L) subject to stiffness constraint.
        
        Args:
            cross_section: Cross-section dimensions (e.g., [width, height])
            
        Returns:
            Weight (kg) - to be minimized
        """
        # Simplified: weight = A * L
        return self.area * self.length
    
    def constraint(self, cross_section: np.ndarray) -> float:
        """
        Constraint: minimum stiffness requirement.
        
        Args:
            cross_section: Cross-section dimensions
            
        Returns:
            Minimum required stiffness (N/m)
        """
        # Required stiffness for given load (simplified)
        # K_min = E * I / L^3
        I = self.I * (cross_section[0] * cross_section[1])  # Approximate
        return self.E * I / (self.length**3)
    
    def optimize(self, initial_guess: np.ndarray, max_iter: int = 100) -> Tuple[np.ndarray, float]:
        """
        Optimize cross-section to minimize weight while meeting stiffness.
        
        Args:
            initial_guess: Initial cross-section dimensions
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimized cross-section and minimum weight
        """
        # Use gradient-based optimization
        from scipy.optimize import minimize
        
        # Bounds: positive dimensions
        bounds = [(1e-6, 0.1), (1e-6, 0.1)]  # [width, height]
        
        result = minimize(
            self.objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        return result.x, result.fun


class Rotordynamics:
    """Rotordynamics - Basic Rotational Mechanics and Contact."""
    
    def __init__(self, rotor_properties: Dict[str, float]):
        """
        Initialize rotordynamics analysis.
        
        Args:
            rotor_properties: Rotor properties including density, radius, length
        """
        self.rho = rotor_properties.get('rho', 7850)  # kg/m^3 (steel)
        self.R = rotor_properties.get('R', 0.05)  # Radius (m)
        self.L = rotor_properties.get('L', 1.0)  # Length (m)
        self.mass = self.rho * self.L * self.R**2  # Approximate mass
    
    def centrifugal_force(self, omega: float) -> np.ndarray:
        """
        Compute centrifugal force on rotor.
        
        Args:
            omega: Angular velocity (rad/s)
            
        Returns:
            Centrifugal force vector (N)
        """
        # F = m * omega^2 * R
        return self.mass * omega**2 * self.R
    
    def contact_force(self, radius: float, normal_force: float) -> float:
        """
        Compute contact force between rotor and bearing.
        
        Args:
            radius: Contact radius (m)
            normal_force: Normal load (N)
            
        Returns:
            Contact force (N)
        """
        return normal_force * (radius / self.R)  # Simple scaling
    
    def critical_speed(self, damping: float = 0.01) -> float:
        """
        Compute critical speed (RPM) for resonance.
        
        Args:
            damping: Damping ratio
            
        Returns:
            Critical speed in RPM
        """
        # Centrifugal frequency: omega_c = omega_n * sqrt(1 + (2*zeta*omega_n/omega_c)^2)
        # Simplified: critical speed proportional to sqrt(omega_n^2 + 4*zeta^2*omega_n^2)
        omega_n = self.L * 2 * np.pi  # Approximate natural frequency
        omega_c = omega_n * np.sqrt(1 + 4 * damping**2)
        return omega_c / (2 * np.pi) * 60  # Convert to RPM


def main():
    """Demonstration of the mechanical engineering analysis suite."""
    
    print("=" * 60)
    print("Mechanical Engineering Analysis Suite - Demonstration")
    print("=" * 60)
    
    # 1. Truss Analysis Example
    print("\n1. Truss Analysis")
    print("-" * 40)
    truss = TrussAnalysis(n_nodes=4, node_positions=np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]))
    print(f"Number of nodes: {truss.n_nodes}")
    print(f"Stiffness matrix shape: {truss._stiffness_matrix.shape}")
    print(f"Displacement for unit force at node 0: {truss.displacement(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))}")
    
    # 2. Beam Analysis Example
    print("\n2. Beam Analysis")
    print("-" * 40)
    beam = BeamAnalysis(n_nodes=5, beam_properties={
        'E': 200e9,  # Young's modulus (Pa)
        'I': 1.0e-6,  # Moment of inertia (m^4)
        'A': 1.0e-4,  # Cross-sectional area (m^2)
        'L': 2.0      # Length (m)
    })
    print(f"Beam length: {beam.L} m")
    print(f"Stiffness matrix shape: {beam._stiffness_matrix.shape}")
    print(f"Bending moment at node 2: {beam.bending_moment(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))}")
    
    # 3. Stress Transformation Example
    print("\n3. Stress Transformation")
    print("-" * 40)
    stress = np.array([50e6, 20e6, -30e6, 15e6])  # [sigma_x, tau_xy, sigma_y, tau_xy]
    center, radius = StressTransformation().compute_mohr_circle(stress)
    print(f"Mohr's Circle: Center=({center[0]:.2f}, {center[1]:.2f}), Radius={radius:.2f}")
    von_mises = StressTransformation().von_mises_stress(stress)
    trescas = StressTransformation().trescas_stress(stress)
    print(f"Von Mises stress: {von_mises:.2f} Pa")
    print(f"Tresca stress: {trescas:.2f} Pa")
    
    # 4. Vibration Analysis Example
    print("\n4. Vibration Analysis")
    print("-" * 40)
    # Create a simple beam mass matrix
    mass = np.diag([1.0, 2.0, 3.0, 4.0])  # N*m^2
    stiffness = beam._stiffness_matrix
    vib = VibrationAnalysis(mass, stiffness)
    eigenvalues, eigenvectors = vib._compute_modal_matrix()
    print(f"Natural frequencies (rad/s): {eigenvalues}")
    print(f"First mode shape: {vib.modal_displacement(0)}")
    response = vib.damped_response(0, damping_ratio=0.05, initial_amplitude=0.1)
    print(f"Damped response (first 10 samples): {response[:10]}")
    
    # 5. Fatigue Analysis Example
    print("\n5. Fatigue Analysis")
    print("-" * 40)
    fatigue = FatigueAnalysis({
        'S-N': (0.5, 1000, 0.3)  # (log10(S_max), log10(N_f), factor)
    })
    stress = 50e6  # 50 MPa
    allowable = fatigue.goodman_diagram(stress)
    print(f"Goodman allowable stress: {allowable:.2f} Pa")
    gerber_allowable = fatigue.goodman_diagram(stress, safety_factor=1.5)
    print(f"Gerber allowable stress: {gerber_allowable:.2f} Pa")
    life = fatigue.fatigue_life(stress, cycles=500)
    print(f"Estimated fatigue life at 50 MPa: {life:.1f} cycles")
    
    # 6. Structural Optimization Example
    print("\n6. Structural Optimization")
    print("-" * 40)
    opt = StructuralOptimization(beam_properties={
        'E': 200e9,
        'I': 1.0e-6,
        'A': 1.0e-4,
        'L': 2.0
    })
    # Initial guess: [width, height]
    initial_guess = np.array([0.02, 0.02])
    optimized_section, min_weight = opt.optimize(initial_guess)
    print(f"Optimized cross-section: width={optimized_section[0]:.4f} m, height={optimized_section[1]:.4f} m")
    print(f"Minimum weight: {min_weight:.6f} kg")
    
    # 7. Rotordynamics Example
    print("\n7. Rotordynamics")
    print("-" * 40)
    rotor = Rotordynamics({
        'rho': 7850,  # kg/m^3
        'R': 0.05,    # m
        'L': 1.0      # m
    })
    omega = 100.0  # rad/s
    centrifugal = rotor.centrifugal_force(omega)
    print(f"Centrifugal force at {omega} rad/s: {centrifugal:.2f} N")
    contact = rotor.contact_force(0.05, 1000.0)
    print(f"Contact force at radius 0.05 m: {contact:.2f} N")
    critical = rotor.critical_speed()
    print(f"Critical speed: {critical:.2f} RPM")
    
    print("\n" + "=" * 60)
    print("Analysis complete. All modules demonstrated successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()