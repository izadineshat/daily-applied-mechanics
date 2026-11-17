"""
Mechanical Engineering Analysis Suite
====================================
A comprehensive Python library for mechanical engineering calculations including:
- Finite Element Analysis (1D/2D truss/beam)
- Stress & Strain Transformations
- Mechanical Vibrations (SDOF/MDOF)
- Fatigue & Fracture Mechanics
- Structural Optimization
- Rotordynamics & Failure Theories
"""

import numpy as np
from scipy import linalg, optimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Callable
import warnings

# ============================================================
# FINITE ELEMENT ANALYSIS
# ============================================================

class TrussElement:
    """2D truss element with stiffness matrix formulation"""
    
    def __init__(self, node_i: int, node_j: int, area: float, elasticity: float):
        self.node_i = node_i
        self.node_j = node_j
        self.area = area
        self.elasticity = elasticity
    
    def stiffness_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute element stiffness matrix in global coordinates"""
        xi, yi = coords[self.node_i]
        xj, yj = coords[self.node_j]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        c = (xj - xi) / L
        s = (yj - yi) / L
        
        k_local = (self.area * self.elasticity / L) * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
        return k_local

class BeamElement:
    """Euler-Bernoulli beam element with bending and axial stiffness"""
    
    def __init__(self, node_i: int, node_j: int, area: float, inertia: float, 
                 elasticity: float, length: float):
        self.node_i = node_i
        self.node_j = node_j
        self.area = area
        self.inertia = inertia
        self.elasticity = elasticity
        self.length = length
    
    def stiffness_matrix(self) -> np.ndarray:
        """Compute beam element stiffness matrix"""
        E = self.elasticity
        A = self.area
        I = self.inertia
        L = self.length
        
        k_axial = E * A / L
        k_bend = 2 * E * I / L**3
        
        return np.array([
            [k_axial, 0, 0, -k_axial, 0, 0],
            [0, 12*k_bend, 6*L*k_bend, 0, -12*k_bend, 6*L*k_bend],
            [0, 6*L*k_bend, 4*L**2*k_bend, 0, -6*L*k_bend, 2*L**2*k_bend],
            [-k_axial, 0, 0, k_axial, 0, 0],
            [0, -12*k_bend, -6*L*k_bend, 0, 12*k_bend, -6*L*k_bend],
            [0, 6*L*k_bend, 2*L**2*k_bend, 0, -6*L*k_bend, 4*L**2*k_bend]
        ])

class TrussAnalyzer:
    """Finite element analysis for 2D truss structures"""
    
    def __init__(self, nodes: np.ndarray, elements: List[TrussElement]):
        self.nodes = nodes
        self.elements = elements
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes
    
    def assemble_global_stiffness(self) -> np.ndarray:
        """Assemble global stiffness matrix"""
        K = np.zeros((self.n_dof, self.n_dof))
        
        for elem in self.elements:
            ke = elem.stiffness_matrix(self.nodes)
            idx = np.array([2*elem.node_i, 2*elem.node_i+1, 
                           2*elem.node_j, 2*elem.node_j+1])
            
            for i in range(4):
                for j in range(4):
                    K[idx[i], idx[j]] += ke[i, j]
        
        return K
    
    def solve(self, forces: np.ndarray, fixed_dofs: List[int]) -> np.ndarray:
        """Solve truss system with boundary conditions"""
        K = self.assemble_global_stiffness()
        
        # Apply boundary conditions
        free_dofs = [i for i in range(self.n_dof) if i not in fixed_dofs]
        K_free = K[np.ix_(free_dofs, free_dofs)]
        f_free = forces[free_dofs]
        
        u_free = linalg.solve(K_free, f_free, assume_a='pos')
        
        # Reconstruct full displacement vector
        u = np.zeros(self.n_dof)
        u[free_dofs] = u_free
        
        return u

# ============================================================
# STRESS & STRAIN TRANSFORMATIONS
# ============================================================

@dataclass
class StressState:
    """Represents a 3D stress state"""
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    sigma_z: float = 0.0
    tau_xy: float = 0.0
    tau_yz: float = 0.0
    tau_xz: float = 0.0
    
    def stress_tensor(self) -> np.ndarray:
        """Return stress tensor as 3x3 matrix"""
        return np.array([
            [self.sigma_x, self.tau_xy, self.tau_xz],
            [self.tau_xy, self.sigma_y, self.tau_yz],
            [self.tau_xz, self.tau_yz, self.sigma_z]
        ])
    
    def principal_stresses(self) -> np.ndarray:
        """Calculate principal stresses"""
        stress_tensor = self.stress_tensor()
        eigenvalues = linalg.eigvals(stress_tensor)
        return np.sort(eigenvalues)[::-1]  # Sort descending
    
    def mohr_circle_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Mohr's circle data for 2D stress state"""
        sigma_avg = (self.sigma_x + self.sigma_y) / 2
        R = np.sqrt(((self.sigma_x - self.sigma_y)/2)**2 + self.tau_xy**2)
        
        theta = np.linspace(0, 2*np.pi, 100)
        sigma = sigma_avg + R * np.cos(theta)
        tau = R * np.sin(theta)
        
        return sigma, tau
    
    def von_mises(self) -> float:
        """Calculate von Mises equivalent stress"""
        s1, s2, s3 = self.principal_stresses()
        return np.sqrt(0.5 * ((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2))
    
    def tresca(self) -> float:
        """Calculate Tresca equivalent stress"""
        s1, s2, s3 = self.principal_stresses()
        return s1 - s3
    
    def factor_of_safety(self, yield_strength: float, 
                        theory: str = 'von_mises') -> float:
        """Calculate factor of safety using specified failure theory"""
        if theory.lower() == 'von mises':
            equivalent_stress = self.von_mises()
        elif theory.lower() == 'tresca':
            equivalent_stress = self.tresca()
        else:
            raise ValueError("Theory must be 'von_mises' or 'tresca'")
        
        return yield_strength / equivalent_stress if equivalent_stress > 0 else np.inf

# ============================================================
# MECHANICAL VIBRATIONS
# ============================================================

class MechanicalVibrations:
    """Analysis of mechanical vibration systems"""
    
    @staticmethod
    def sdof_response(mass: float, stiffness: float, damping: float,
                     force: Callable[[float], float],
                     initial_conditions: List[float],
                     t_span: Tuple[float, float],
                     t_eval: Optional[np.ndarray] = None) -> Dict:
        """Single Degree of Freedom system response"""
        def ode_system(t, y):
            x, v = y
            dxdt = v
            dvdt = (force(t) - stiffness*x - damping*v) / mass
            return [dxdt, dvdt]
        
        sol = solve_ivp(ode_system, t_span, initial_conditions,
                       t_eval=t_eval, method='RK45')
        
        return {
            'time': sol.t,
            'displacement': sol.y[0],
            'velocity': sol.y[1],
            'natural_frequency': np.sqrt(stiffness/mass),
            'damping_ratio': damping / (2*np.sqrt(stiffness*mass))
        }
    
    @staticmethod
    def modal_analysis(mass_matrix: np.ndarray, stiffness_matrix: np.ndarray) -> Dict:
        """Perform modal analysis for MDOF system"""
        eigenvalues, eigenvectors = linalg.eig(
            linalg.solve(mass_matrix, stiffness_matrix)
        )
        
        natural_freqs = np.sqrt(np.real(eigenvalues))
        mode_shapes = np.real(eigenvectors)
        
        # Sort by frequency
        idx = np.argsort(natural_freqs)
        natural_freqs = natural_freqs[idx]
        mode_shapes = mode_shapes[:, idx]
        
        return {
            'natural_frequencies': natural_freqs,
            'mode_shapes': mode_shapes,
            'mass_matrix': mass_matrix,
            'stiffness_matrix': stiffness_matrix
        }
    
    @staticmethod
    def state_space(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                   initial_state: np.ndarray, u: Callable[[float], np.ndarray],
                   t_span: Tuple[float, float], t_eval: np.ndarray) -> Dict:
        """State-space representation for MDOF systems"""
        def state_ode(t, x):
            return A @ x + B @ u(t)
        
        sol = solve_ivp(state_ode, t_span, initial_state, t_eval=t_eval)
        
        # Calculate output
        y = np.array([C @ sol.y[:, i] + D @ u(sol.t[i]) 
                     for i in range(len(sol.t))]).T
        
        return {
            'time': sol.t,
            'states': sol.y,
            'output': y,
            'A': A, 'B': B, 'C': C, 'D': D
        }

# ============================================================
# FATIGUE & FRACTURE MECHANICS
# ============================================================

class FatigueAnalysis:
    """Fatigue life calculations and analysis"""
    
    @staticmethod
    def s_n_curve(stress_range: float, fatigue_limit: float, 
                 exponent: float, cycles: float) -> float:
        """Calculate cycles to failure using S-N curve"""
        return (fatigue_limit / stress_range) ** exponent
    
    @staticmethod
    def goodman_diagram(ultimate_strength: float, endurance_limit: float,
                       mean_stress: float, stress_amplitude: float) -> float:
        """Goodman correction for mean stress effects"""
        return endurance_limit * (1 - mean_stress / ultimate_strength)
    
    @staticmethod
    def gerber_diagram(ultimate_strength: float, endurance_limit: float,
                      mean_stress: float, stress_amplitude: float) -> float:
        """Gerber correction for mean stress effects"""
        return endurance_limit * (1 - (mean_stress / ultimate_strength)**2)
    
    @staticmethod
    def paris_law(crack_growth_rate: float, stress_intensity_range: float,
                 c: float, m: float) -> float:
        """Paris' law for fatigue crack growth"""
        return c * (stress_intensity_range ** m)
    
    @staticmethod
    def palmgren_miner_rule(damage: List[float]) -> float:
        """Calculate cumulative damage using Palmgren-Miner rule"""
        return sum(damage)

# ============================================================
# STRUCTURAL OPTIMIZATION
# ============================================================

class BeamOptimizer:
    """Optimization of beam cross-sections for minimum weight"""
    
    def __init__(self, length: float, load: float, material_strength: float,
                 density: float):
        self.length = length
        self.load = load
        self.material_strength = material_strength
        self.density = density
    
    def rectangular_section_optimization(self) -> Dict:
        """Optimize rectangular cross-section for minimum weight"""
        def objective(x):
            width, height = x
            return self.density * self.length * width * height
        
        def constraint(x):
            width, height = x
            # Bending stress constraint
            section_modulus = width * height**2 / 6
            max_stress = self.load * self.length / (4 * section_modulus)
            return self.material_strength - max_stress
        
        constraints = [{'type': 'ineq', 'fun': constraint}]
        bounds = [(0.01, 1.0), (0.01, 1.0)]  # Reasonable bounds for dimensions
        
        result = optimize.minimize(
            objective, [0.1, 0.1], method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        return {
            'optimal_dimensions': result.x,
            'min_weight': result.fun,
            'success': result.success,
            'max_stress': self.load * self.length / (4 * (result.x[0] * result.x[1]**2 / 6))
        }

# ============================================================
# ROTORDYNAMICS & FAILURE THEORIES
# ============================================================

class RotorDynamics:
    """Rotordynamics analysis for rotating machinery"""
    
    @staticmethod
    def critical_speed(mass: float, stiffness: float) -> float:
        """Calculate critical speed of rotating shaft"""
        return np.sqrt(stiffness / mass)
    
    @staticmethod
    def unbalance_response(mass: float, stiffness: float, damping: float,
                          unbalance: float, speed: float) -> Dict:
        """Calculate response due to rotor unbalance"""
        omega_n = np.sqrt(stiffness / mass)
        zeta = damping / (2 * np.sqrt(stiffness * mass))
        r = speed / omega_n
        
        amplitude = unbalance * r**2 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
        phase = np.arctan2(2*zeta*r, 1 - r**2)
        
        return {
            'amplitude': amplitude,
            'phase': phase,
            'natural_frequency': omega_n,
            'speed_ratio': r
        }

class FailureTheories:
    """Various failure theories for ductile materials"""
    
    @staticmethod
    def maximum_normal_stress(s1: float, s2: float, s3: float, 
                             yield_strength: float) -> float:
        """Maximum normal stress theory"""
        return yield_strength / max(abs(s1), abs(s2), abs(s3))
    
    @staticmethod
    def maximum_shear_stress(s1: float, s2: float, s3: float,
                            yield_strength: float) -> float:
        """Maximum shear stress theory (Tresca)"""
        max_shear = max(abs(s1-s2), abs(s2-s3), abs(s3-s1)) / 2
        return yield_strength / (2 * max_shear)
    
    @staticmethod
    def distortion_energy(s1: float, s2: float, s3: float,
                         yield_strength: float) -> float:
        """Distortion energy theory (von Mises)"""
        von_mises = np.sqrt(0.5 * ((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2))
        return yield_strength / von_mises

# ============================================================
# MAIN APPLICATION EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mechanical Engineering Analysis Suite")
    print("=" * 60)
    
    # Example 1: Truss Analysis
    print("\n1. TRUSS ANALYSIS EXAMPLE")
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elements = [
        TrussElement(0, 1, 0.001, 200e9),
        TrussElement(1, 2, 0.001, 200e9),
        TrussElement(2, 3, 0.001, 200e9),
        TrussElement(3, 0, 0.001, 200e9),
        TrussElement(0, 2, 0.001, 200e9)
    ]
    
    truss = TrussAnalyzer(nodes, elements)
    forces = np.zeros(8)
    forces[4] = 1000  # Apply force at node 2 in x-direction
    fixed_dofs = [0, 1, 6, 7]  # Fix nodes 0 and 3
    
    displacements = truss.solve(forces, fixed_dofs)
    print(f"Maximum displacement: {np.max(np.abs(displacements)):.6f} m")
    
    # Example 2: Stress Analysis
    print("\n2. STRESS ANALYSIS EXAMPLE")
    stress_state = StressState(sigma_x=100, sigma_y=50, tau_xy=30)
    principal_stresses = stress_state.principal_stresses()
    von_mises = stress_state.von_mises()
    fos = stress_state.factor_of_safety(250, 'von_mises')
    
    print(f"Principal stresses: {principal_stresses}")
    print(f"Von Mises stress: {von_mises:.2f} MPa")
    print(f"Factor of safety: {fos:.2f}")
    
    # Example 3: Vibration Analysis
    print("\n3. VIBRATION ANALYSIS EXAMPLE")
    mass = 10.0  # kg
    stiffness = 1000.0  # N/m
    damping = 50.0  # Ns/m
    
    def force_func(t):
        return 100 * np.sin(2 * np.pi * 0.5 * t)
    
    response = MechanicalVibrations.sdof_response(
        mass, stiffness, damping, force_func,
        [0, 0], (0, 10), np.linspace(0, 10, 100)
    )
    
    print(f"Natural frequency: {response['natural_frequency']:.2f} rad/s")
    print(f"Damping ratio: {response['damping_ratio']:.3f}")
    print(f"Max displacement: {np.max(np.abs(response['displacement'])):.4f} m")
    
    # Example 4: Fatigue Analysis
    print("\n4. FATIGUE ANALYSIS EXAMPLE")
    cycles = FatigueAnalysis.s_n_curve(50, 200, 3.0, 1e6)
    goodman_life = FatigueAnalysis.goodman_diagram(400, 200, 100, 80)
    
    print(f"Cycles to failure: {cycles:.0f}")
    print(f"Goodman corrected endurance limit: {goodman_life:.2f} MPa")
    
    # Example 5: Beam Optimization
    print("\n5. BEAM OPTIMIZATION EXAMPLE")
    optimizer = BeamOptimizer(length=2.0, load=5000, 
                            material_strength=250e6, density=7850)
    optimal = optimizer.rectangular_section_optimization()
    
    print(f"Optimal dimensions: {optimal['optimal_dimensions']}")
    print(f"Minimum weight: {optimal['min_weight']:.2f} kg")
    print(f"Maximum stress: {optimal['max_stress']/1e6:.2f} MPa")
    
    # Example 6: Rotor Dynamics
    print("\n6. ROTORDYNAMICS EXAMPLE")
    critical_speed = RotorDynamics.critical_speed(10, 1e6)
    unbalance_resp = RotorDynamics.unbalance_response(10, 1e6, 100, 0.001, 100)
    
    print(f"Critical speed: {critical_speed:.2f} rad/s")
    print(f"Unbalance amplitude: {unbalance_resp['amplitude']*1000:.3f} mm")
    
    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)
    
    # Generate plots for key results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Mohr's Circle
    plt.subplot(2, 2, 1)
    sigma, tau = stress_state.mohr_circle_2d()
    plt.plot(sigma, tau, 'b-', label='Mohr\'s Circle')
    plt.plot(sigma, -tau, 'b-')
    plt.xlabel('Normal Stress (MPa)')
    plt.ylabel('Shear Stress (MPa)')
    plt.title('Mohr\'s Circle')
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 2: Vibration Response
    plt.subplot(2, 2, 2)
    plt.plot(response['time'], response['displacement']*1000)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (mm)')
    plt.title('SDOF Vibration Response')
    plt.grid(True)
    
    # Plot 3: S-N Curve
    plt.subplot(2, 2, 3)
    stresses = np.linspace(10, 200, 50)
    cycles_life = [FatigueAnalysis.s_n_curve(s, 200, 3.0, 1e6) for s in stresses]
    plt.loglog(stresses, cycles_life, 'r-')
    plt.xlabel('Stress Range (MPa)')
    plt.ylabel('Cycles to Failure')
    plt.title('S-N Curve')
    plt.grid(True)
    
    # Plot 4: Goodman Diagram
    plt.subplot(2, 2, 4)
    mean_stresses = np.linspace(0, 400, 50)
    goodman_limits = [FatigueAnalysis.goodman_diagram(400, 200, ms, 0) 
                     for ms in mean_stresses]
    plt.plot(mean_stresses, goodman_limits, 'g-', label='Goodman')
    plt.xlabel('Mean Stress (MPa)')
    plt.ylabel('Allowable Stress Amplitude (MPa)')
    plt.title('Goodman Diagram')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()