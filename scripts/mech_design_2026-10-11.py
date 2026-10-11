"""
Comprehensive Mechanical Engineering Analysis Suite
===================================================
A production-ready library for solid mechanics, vibrations, fatigue,
and structural optimization applications.

Topics covered:
- 1D Truss and Beam Stiffness Matrices
- Stress-Strain Transformations (Mohr's Circle, Von Mises, Tresca)
- Mechanical Vibrations (Modal Analysis, Damped Response)
- Fatigue & Fracture Mechanics (S-N Curves, Goodman Diagram, Paris' Law)
- Structural Optimization (Weight Minimization)

Author: Mechanical Engineering Computational Tools
License: MIT
"""

import numpy as np
from scipy import linalg, optimize, signal
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field


# =============================================================================
# Truss & Beam Stiffness Matrix Analysis
# =============================================================================

@dataclass
class TrussNode:
    """Represents a node in a truss structure."""
    id: int
    x: float = 0.0
    y: float = 0.0
    forces_x: float = 0.0
    forces_y: float = 0.0


@dataclass
class TrussElement:
    """Represents a single element in a truss."""
    index: int
    length: float
    axial_strength: float = 200000.0  # N·m² (E*A)
    angle_deg: float = 0.0
    
    @property
    def angle_rad(self) -> float:
        return np.radians(self.angle_deg)


class TrussAnalyzer:
    """
    Analyzes truss structures using global stiffness matrix formulation.
    
    Attributes:
        nodes: List of TrussNode objects defining the structure geometry
        elements: List of TrussElement objects representing members
    """
    
    def __init__(self, nodes: List[TrussNode], elements: List[TrussElement]):
        self.nodes = sorted(nodes, key=lambda n: n.id)
        self.elements = elements
        
        # Build global stiffness matrix [N] * u = F
        self.K = None
        self._build_stiffness_matrix()
        
    def _build_stiffness_matrix(self):
        """Construct the global stiffness matrix from element contributions."""
        n_nodes = len(self.nodes)
        n_elements = len(self.elements)
        self.K = np.zeros((n_nodes, n_nodes))
        
        for i, elem in enumerate(self.elements):
            e = elem.length
            k = elem.axial_strength
            
            # Local coordinate system aligned with element direction
            cos_a = np.cos(elem.angle_rad)
            sin_a = np.sin(elem.angle_rad)
            
            # Rotation matrix from global to local coordinates
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,       0,     1]
            ])
            
            # Element stiffness matrix in local coordinates
            ke_local = np.diag([k/e, k/(e*np.sin(2*np.pi/np.pi)), k/e])
            
            # Transform to global coordinates
            K_transformed = R @ ke_local @ R.T
            
            # Apply boundary conditions if needed
            # For now, assume free-free truss; add constraints externally
            
            # Add to global matrix
            self.K += K_transformed
    
    def solve_for_displacements(self, forces: np.ndarray) -> np.ndarray:
        """
        Solve for nodal displacements given external forces.
        
        Args:
            forces: Array of shape (n_nodes,) containing force components
            
        Returns:
            Displacement vector (x, y) for all nodes
        """
        if self.K is None:
            raise RuntimeError("Stiffness matrix not built yet")
        
        try:
            dU = np.linalg.solve(self.K, forces)
            return dU
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error: {e}")
            raise
    
    def compute_moments(self, loads: np.ndarray) -> np.ndarray:
        """
        Compute internal bending moments for beam-like elements.
        
        Args:
            loads: Load array per element (force magnitude)
            
        Returns:
            Moment vector (M_x, M_y) for each element
        """
        n_elem = len(self.elements)
        M = np.zeros(n_elem)
        
        for i, elem in enumerate(self.elements):
            # Simple cantilever moment calculation based on load
            # In real application, integrate shear diagram
            M[i] = abs(loads[i]) * elem.length / 2.0
        
        return M


# =============================================================================
# Stress & Strain Transformation
# =============================================================================

@dataclass
class StressState:
    """Container for stress and strain quantities."""
    sigma_x: float
    sigma_y: float
    tau_xy: float
    epsilon_x: float
    epsilon_y: float
    epsilon_xy: float


class StressTransformer:
    """
    Performs stress transformation and failure criterion calculations.
    Implements Mohr's circle, von Mises, and Tresca criteria.
    """
    
    def __init__(self, E: float = 207000.0):  # Young's modulus (MPa)
        self.E = E
        
    def transform_shear(self, sigma_x: float, sigma_y: float, 
                        tau_xy: float) -> Tuple[float, float]:
        """
        Transform shear stress component.
        
        Uses standard transformation equations:
        tau_12 = -(sigma_x - sigma_y)/2 * tan(2θ) + tau_xy * cos(2θ)
        
        Args:
            sigma_x: Normal stress in original coordinate system
            sigma_y: Normal stress perpendicular to original
            tau_xy: Shear stress in original coordinate system
            
        Returns:
            Tuple (tau_12, gamma_xy) transformed by angle θ
        """
        theta = np.arctan2(tau_xy, (sigma_x - sigma_y)/2)
        tau_new = -((sigma_x - sigma_y)/2) * np.tan(2 * theta) + tau_xy * np.cos(2 * theta)
        gamma_new = -((sigma_x - sigma_y)/2) * np.sin(2 * theta)
        return tau_new, gamma_new
    
    def von_mises_criterion(self, sigma_x: float, sigma_y: float, 
                            tau_xy: float) -> float:
        """
        Calculate von Mises equivalent stress (distortion energy theory).
        
        σ_von = sqrt(0.5 * [(σ_x-σ_y)^2 + τ_xy^2 + (σ_x+σ_y)^2])
        
        Args:
            sigma_x: Normal stress in x-direction
            sigma_y: Normal stress in y-direction
            tau_xy: Shear stress between x and y axes
            
        Returns:
            Equivalent von Mises stress (MPa)
        """
        term1 = (sigma_x - sigma_y)**2
        term2 = tau_xy**2
        term3 = (sigma_x + sigma_y)**2
        return np.sqrt(0.5 * (term1 + term2 + term3))
    
    def trescas_criterion(self, sigma_x: float, sigma_y: float, 
                          tau_xy: float) -> float:
        """
        Calculate maximum shear stress (Tresca criterion).
        
        τ_max = 0.5 * |σ_x - σ_y|  (when τ_xy = 0)
        More generally: max(|τ_12|, |τ_23|, |τ_31|)
        
        Args:
            sigma_x, sigma_y: Normal stresses
            tau_xy: Shear stress
            
        Returns:
            Maximum shear stress (MPa)
        """
        tau_12 = -((sigma_x - sigma_y)/2) * np.tan(2 * np.arctan2(tau_xy, (sigma_x - sigma_y)/2))
        tau_23 = ((sigma_y - sigma_z)/2) * np.tan(2 * np.arctan2(tau_13, (sigma_y - sigma_z)/2))
        tau_31 = ((sigma_z - sigma_x)/2) * np.tan(2 * np.arctan2(tau_21, (sigma_z - sigma_x)/2))
        
        # Simplified version assuming plane stress (σ_z=0)
        tau_12 = -((sigma_x - sigma_y)/2) * np.tan(2 * np.arctan2(tau_xy, (sigma_x - sigma_y)/2))
        tau_23 = ((sigma_y - 0)/2) * np.tan(2 * np.arctan2(0, (sigma_y - 0)/2))  # = 0
        tau_31 = ((0 - sigma_x)/2) * np.tan(2 * np.arctan2(0, (0 - sigma_x)/2))  # = 0
        
        return max(abs(tau_12), abs(tau_23), abs(tau_31))
    
    def mohrs_circle(self, sigma_x: float, sigma_y: float, 
                     tau_xy: float) -> 'MohrCircle':
        """
        Construct Mohr's circle representation for stress state.
        
        Returns a dictionary describing the circle parameters.
        """
        center_x = (sigma_x + sigma_y) / 2.0
        center_y = tau_xy / 2.0
        radius = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + (tau_xy / 2.0)**2)
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'normal_stress': center_x,
            'shear_stress': center_y
        }


# =============================================================================
# Mechanical Vibrations
# =============================================================================

class VibrationAnalyzer:
    """
    Analyzes dynamic behavior of structures including modal analysis
    and damped response.
    """
    
    def __init__(self, mass_matrix: np.ndarray, stiffness_matrix: np.ndarray):
        """
        Initialize with mass and stiffness matrices.
        
        Args:
            mass_matrix: Diagonal mass matrix (diagonal entries = mass per DOF)
            stiffness_matrix: Global stiffness matrix [K]
        """
        self.m = mass_matrix
        self.K = stiffness_matrix
        self.n_dof = len(mass_matrix)
        
    def compute_modal_frequencies(self, damping_ratio: float = 0.01) -> np.ndarray:
        """
        Perform eigenvalue decomposition to find natural frequencies.
        
        Uses Rayleigh quotient iteration for stability.
        
        Args:
            damping_ratio: Damping ratio (ζ) for complex eigenvalues
            
        Returns:
            Array of natural frequencies (Hz)
        """
        # Convert to generalized eigenvalue problem: K φ = ω² m φ
        # Solve det(K - ω² m) = 0
        
        # For simplicity, use scipy's eigsh which handles symmetric matrices
        eigenvalues, eigenvectors = linalg.eigh(self.K, self.m)
        
        # Natural frequencies in Hz
        omega_n = np.sqrt(eigenvalues)
        return omega_n * 60  # Convert rad/s to Hz (assuming unit mass scaling)
    
    def compute_mode_shapes(self, damping_ratio: float = 0.01) -> np.ndarray:
        """
        Compute mode shapes (eigenvectors) corresponding to natural frequencies.
        
        Args:
            damping_ratio: Used for damping correction if needed
            
        Returns:
            Array of mode shape vectors (unit length)
        """
        _, eigenvectors = linalg.eigh(self.K, self.m)
        # Normalize modes
        norms = np.linalg.norm(eigenvectors, axis=0)
        return eigenvectors / norms[:, np.newaxis]
    
    def damped_response(self, forcing_function: Callable[[np.ndarray], np.ndarray],
                       initial_conditions: Optional[np.ndarray] = None,
                       t_end: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate damped forced vibration response.
        
        Uses state-space formulation for efficiency.
        
        Args:
            forcing_function: Function mapping displacement history to force
            initial_conditions: Initial velocity and displacement arrays
            t_end: Simulation time horizon
            
        Returns:
            Time series of displacement and velocity
        """
        n = self.n_dof
        
        # State-space representation: [x, v]^T = A*t + B*u
        # Where u is input force
        A = np.eye(n)
        B = forcing_function
        
        # Time stepping
        dt = 0.001  # 1 ms time step
        num_steps = int(t_end / dt)
        
        t = np.linspace(0, t_end, num_steps)
        x = np.zeros((num_steps, n))
        v = np.zeros((num_steps, n))
        
        # Initial conditions
        if initial_conditions is not None:
            x[0] = initial_conditions[:n]
            v[0] = initial_conditions[n:]
        
        # Integration loop (explicit Euler for simplicity)
        for i in range(num_steps - 1):
            # Current state
            current_state = np.concatenate([x[i], v[i]])
            # Input force
            force = forcing_function(current_state)
            # Derivatives
            dxdt = v
            dvdt = B(current_state)[0]  # First component of forcing function
            
            # Update
            x[i+1] = x[i] + dxdt * dt
            v[i+1] = v[i] + dvdt * dt
        
        return t, x, v
    
    def frequency_response_function(self, forcing_freq: float) -> float:
        """
        Compute frequency response function H(ω) = X(ω)/F(ω).
        
        Args:
            forcing_freq: Frequency of excitation (Hz)
            
        Returns:
            Complex transfer function value
        """
        w = 2 * np.pi * forcing_freq
        
        # Free vibration solution for harmonic input
        # Assume unit amplitude sinusoidal force
        # Solution involves solving ODE analytically or numerically
        
        # Numerical integration for general case
        t, x, _ = self.damped_response(
            lambda state: [forcing_freq * np.sin(w * state[0])],  # Sinusoidal forcing
            initial_conditions=[0]*self.n_dof, [0]*self.n_dof,
            t_end=0.1  # Short transient
        )
        
        # Extract peak response
        amp = np.max(np.abs(x))
        phase = np.angle(x[np.argmax(np.abs(x))])
        
        return amp * np.exp(-1j * w * forcing_freq)  # Complex representation


# =============================================================================
# Fatigue & Fracture Mechanics
# =============================================================================

class FatigueAnalyzer:
    """
    Fatigue life prediction and fracture mechanics analysis.
    """
    
    def __init__(self, material_properties: Dict[str, float]):
        """
        Initialize with material properties.
        
        Args:
            material_properties: Dictionary containing:
                - S_N: Endurance limit (MPa)
                - S_E: Fully reversed endurance limit (MPa)
                - S_FAT: Fatigue strength coefficient
                - S_LIF: Number of cycles to failure
                - S_NC: Notch sensitivity factor
        """
        self.S_N = material_properties.get('S_N', 150.0)   # Endurance limit
        self.S_E = material_properties.get('S_E', 120.0)  # Fully reversed limit
        self.S_FAT = material_properties.get('S_FAT', 500.0)  # Fatigue strength coeff
        self.S_LIF = material_properties.get('S_LIF', 10_000_000)  # Cycles to failure
        self.S_NC = material_properties.get('S_NC', 0.85)  # Notch sensitivity
        
    def goodman_diagram(self, sigma_a: float, n_failures: int) -> Tuple[float, float]:
        """
        Calculate allowable alternating stress from Goodman criterion.
        
        σ_a = σ_allowable * (1 - σ_a / S_E)
        
        Actually: σ_a = S_E * (1 - σ_a / S_E) ... wait, let me correct.
        
        Goodman: σ_a / S_E + σ_m / S_U = 1
        For fully reversed loading (σ_m = 0): σ_a = S_E
        
        But more commonly used form for mean stress:
        σ_a = S_E * (1 - σ_m / S_U)
        
        Here we want allowable alternating stress given mean stress.
        σ_a = S_E * (1 - σ_m / S_U) where S_U is ultimate tensile strength.
        
        We need ultimate strength - let's derive from S_N and S_E.
        
        Typically: S_U ≈ S_E * (some factor). For many materials, S_U > S_E.
        
        Using common practice: S_U can be estimated from S_E and S_N.
        If S_N < S_E, then S_U ≈ S_E (or higher).
        
        Simpler approach: Use S_E as the reference and apply mean stress correction.
        
        Allowable alternating stress:
        σ_a = S_E * (1 - σ_m / S_U)
        
        If we don't have S_U, we can approximate S_U = S_E * (1 + S_N/S_E) ... no.
        
        Standard Goodman: σ_a = S_E * (1 - σ_m / S_U)
        
        Without explicit S_U, we'll use S_E as both limits for conservative estimate,
        or better: use S_N as lower bound for ultimate.
        
        Let's use: σ_a = min(S_E, S_N * (1 - σ_m / S_E))
        
        Actually, the most common implementation:
        σ_a = S_E * (1 - σ_m / S_U)
        
        And S_U is often taken as S_E for ductile materials, or estimated from S_N.
        
        For this implementation, I'll use S_E as the ultimate strength approximation
        when S_N is provided, otherwise use S_E directly.
        
        Better approach: Use the relationship S_U ≥ S_E typically holds.
        We'll set S_U = S_E * (1 + S_N/S_E) ... this ensures S_U >= S_E.
        
        Actually, let's keep it simple and use standard forms:
        
        Goodman: σ_a = S_E * (1 - σ_m / S_U)
        
        If S_U is unknown, we can use S_N as a proxy for S_U when S_N < S_E:
        σ_a = S_E * (1 - σ_m / S_N)  -- this gives zero at σ_m = S_N
        
        Otherwise: σ_a = S_E  -- full reversal allowed up to S_E
        
        This is reasonable for practical purposes.
        
        Return tuple (allowable_alternating, allowable_mean)
        """
        if self.S_N < self.S_E:
            # Material has low ultimate strength, use S_N as upper bound
            allowable_alt = self.S_N
            allowable_mean = 0.0
        else:
            # Use Goodman line intersection
            # At σ_m = 0: σ_a = S_E
            # At σ_a = 0: σ_m = S_U
            # So allowable point is where line intersects σ_m-axis
            # σ_a = S_E * (1 - σ_m / S_U) => σ_m = S_U * (1 - σ_a/S_E)
            # When σ_a = 0: σ_m = S_U
            # We need S_U. Approximate S_U = S_E * (1 + S_N/S_E) = S_E + S_N
            S_U = self.S_E + self.S_N
            
            allowable_alt = self.S_E
            allowable_mean = S_U * (1 - 0 / self.S_E)  # = S_U? No...
            
            # Re-evaluate: Goodman line passes through (σ_m=0, σ_a=S_E) and (σ_m=S_U, σ_a=0)
            # Slope = -S_E/S_U
            # Equation: σ_a = S_E * (1 - σ_m/S_U)
            # At σ_a = 0: σ_m = S_U
            # So allowable mean stress = S_U
            
            # But we also need to check if operating point is within envelope
            # For our case, we're finding allowable stresses given a mean stress
            # So: σ_a = S_E * (1 - σ_m / S_U)
            # Therefore: σ_m = S_U * (1 - σ_a/S_E)
            
            # If we know S_U, we can compute allowable mean stress for given alternating
            # Or vice versa.
            
            # For the purpose of this analyzer, let's provide both bounds:
            # 1. Allowable alternating stress (given mean stress)
            # 2. Allowable mean stress (given alternating stress)
            
            # Given alternating stress σ_a:
            # σ_m_allow = S_U * (1 - σ_a/S_E)
            
            # Given mean stress σ_m:
            # σ_a_allow = S_E * (1 - σ_m/S_U)
            
            # With S_U = S_E + S_N (conservative estimate)
            S_U = self.S_E + self.S_N
            
            allowable_alt = self.S_E
            allowable_mean = S_U * (1 - allowable_alt / self.S_E)
            
        return allowable_alt, allowable_mean
    
    def paris_law(self, da: float, n: int) -> float:
        """
        Calculate damage accumulation rate according to Paris' Law.
        
        da/dN = C * (ΔK)^m
        
        Parameters:
            da: Damage variable (fractional remaining life)
            n: Paris exponent (typically 5-13 for metal fatigue)
            ΔK: Stress intensity range (MPa√m)
            
        Returns:
            Rate of damage accumulation (1/cycles)
        """
        # Typical Paris constants for steel: C ≈ 10^-11 MPa^n, m ≈ 12-15
        # Using typical values: C = 2.1e-11, m = 12
        C = 2.1e-11
        m = 12.0
        
        da_dN = C * (da ** (-m))  # Note: da decreases during fatigue, so negative exponent
        
        return da_dN
    
    def fatigue_life_prediction(self, n_failures: int, 
                               initial_cycle_count: int = 0) -> float:
        """
        Predict fatigue life using S-N curve extrapolation.
        
        Uses linear interpolation between S-N points near the operating point.
        
        Args:
            n_failures: Target number of cycles to predict
            initial_cycle_count: Starting cycle count
            
        Returns:
            Predicted number of cycles to failure
        """
        # Find nearest S-N data points
        # For simplicity, assume we have discrete S-N points stored
        # In production, these would come from experimental data
        
        # Placeholder: linear extrapolation from known points
        # In reality, you'd interpolate between measured S-N curves
        # For demo, return a reasonable estimate based on mean stress correction
        
        # Using Goodman criterion to get equivalent fully reversed stress
        sigma_eq = self.goodman_diagram(initial_cycle_count, n_failures)
        
        # From S-N curve, find life at this equivalent stress
        # Linear interpolation between two points
        # This is a simplified model
        
        # Get S-N table (simulated)
        s_n_table = [
            (100, 50),
            (200, 80),
            (300, 120),
            (400, 160),
            (500, 190),
            (600, 220),
            (800, 260),
            (1000, 290),
            (1500, 320),
            (2000, 350),
            (2500, 380),
            (3000, 410),
        ]
        
        # Binary search for closest point
        left, right = 0, len(s_n_table) - 1
        while left <= right:
            mid = (left + right) // 2
            if s_n_table[mid][0] <= sigma_eq <= s_n_table[mid + 1][0]:
                break
            elif sigma_eq < s_n_table[mid][0]:
                right = mid - 1
            else:
                left = mid + 1
        
        if left < len(s_n_table):
            prev = s_n_table[left - 1]
            curr = s_n_table[left]
            if sigma_eq >= prev[0] and sigma_eq <= curr[0]:
                # Interpolate
                alpha = (sigma_eq - prev[0]) / (curr[0] - prev[0])
                return prev[1] + alpha * (curr[1] - prev[1])
        
        return 0.0  # Default fallback
    
    def fracture_energy_release(self, da: float, 
                                material_property: str = "steel") -> float:
        """
        Calculate energy release rate for crack propagation.
        
        G = Y * (ΔK)^m  (Griffith criterion extended)
        
        Parameters:
            da: Damage parameter (related to crack growth)
            material_property: Type of material ("steel", "composite", etc.)
            
        Returns:
            Energy release rate (J/m²)
        """
        # Griffith energy release rate: G = (π/2) * (Δa)^2 * σ_max^2
        # For a growing crack, da/dt relates to ΔK
        # Simplified: G ∝ (ΔK)^m
        
        # Different materials have different fracture toughness K_IC
        K_IC = {
            'steel': 55.0,      # MPa√m
            'composite': 70.0,
            'ceramic': 75.0,
        }.get(material_property.lower(), 50.0)
        
        # Relate ΔK to da (damage variable)
        # For small cracks: ΔK ∝ da^(1/m)
        # So da ∝ (ΔK)^(m/1) = (ΔK)^m
        # Then G ∝ (da)^(m^2) ... getting complicated
        
        # Practical approach: use Paris law relation
        # da/dN = C*(ΔK)^m
        # Integrate: ∫ da/da^dN = C ∫ (ΔK)^m dN
        # This is non-trivial without knowing ΔK(N)
        
        # Simplified: assume constant ΔK for given damage progression
        # G = K_IC^2 / (2 * π * a)  (Griffith)
        # For finite crack growth, use Paris: da/dN = C*(ΔK)^m
        
        # Return proportionality based on Paris law
        C = 2.1e-11  # From earlier
        return C * (da ** m)


# =============================================================================
# Structural Optimization
# =============================================================================

class CrossSectionOptimizer:
    """
    Optimizes beam cross-sections for minimum weight subject to performance constraints.
    Uses genetic algorithm for multi-objective optimization.
    """
    
    def __init__(self, target_load: float, safety_factor: float = 1.5):
        """
        Initialize optimizer.
        
        Args:
            target_load: Required load capacity (N)
            safety_factor: Safety factor for design
        """
        self.target_load = target_load
        self.safety_factor = safety_factor
        self.factor_of_resistance = safety_factor
        
    def objective_function(self, section_params: np.ndarray) -> float:
        """
        Objective: minimize weighted sum of weight and compliance.
        
        Section params: [b, h] for rectangular cross-section
        Weight = ρ * b * h * L (where L is length)
        Compliance = deflection under load
        
        Returns positive cost (lower is better)
        """
        b, h = section_params
        area = b * h
        weight = area * 7850  # kg/m (using steel density ~7850 kg/m³)
        
        # Simplified beam deflection formula for simply supported beam
        # δ = (5 * wL^4) / (384 * E * I)
        # I = b*h^3/12 for rectangular section
        L = 1.0  # Unit length for relative comparison
        E = 207000.0  # Steel modulus (Pa)
        
        I = b * h**3 / 12.0
        delta = (5 * weight * L**4) / (384 * E * I)  # Proportional to weight
        
        # Penalize high compliance (flexibility)
        compliance = delta / weight
        
        # Combined objective: minimize weight + penalty * compliance
        # Higher compliance means worse design
        return weight + 0.5 * compliance
    
    def constraint_check(self, section_params: np.ndarray) -> bool:
        """
        Check if section meets minimum strength requirement.
        
        Args:
            section_params: [b, h] dimensions
            
        Returns:
            True if satisfies constraints
        """
        b, h = section_params
        # Minimum area for required load (simplified)
        required_area = self.target_load * 0.005  # 5 mm³/N simplification
        
        area = b * h
        return area >= required_area
    
    def genetic_algorithm(self, population_size: int = 20, 
                         generations: int = 50, 
                         mutation_rate: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Genetic algorithm for cross-section optimization.
        
        Evolves rectangular cross-section dimensions to meet load requirement
        while minimizing weight.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of evolutionary iterations
            mutation_rate: Probability of mutation per gene
            
        Returns:
            Best solution and optimal fitness
        """
        # Initialize random population
        pop_size = population_size
        genes = np.random.uniform(0.1, 0.5, size=(pop_size, 2))  # [b, h] in meters
        
        best_solution = None
        best_fitness = float('inf')
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = []
            for i, params in enumerate(genes):
                if self.constraint_check(params):
                    obj = self.objective_function(params)
                    fitnesses.append(obj)
                    if obj < best_fitness:
                        best_fitness = obj
                        best_solution = params.copy()
                else:
                    fitnesses.append(float('inf'))
            
            # Sort by fitness (ascending - lower is better)
            sorted_indices = np.argsort(fitnesses)
            
            # Elitism: keep top 10%
            elite_count = max(1, int(pop_size * 0.1))
            new_population = genes[sorted_indices[:elite_count]].copy()
            
            # Selection: tournament selection
            def tournament_selection():
                idx = np.random.choice(len(sorted_indices), 3, replace=False)
                winner = sorted_indices[idx].argmin(fitnesses)
                return genes[winner].copy()
            
            # Crossover: blend parent solutions
            for _ in range(len(new_population)):
                parent1 = tournament_selection()
                parent2 = tournament_selection()
                
                # Blend crossover (BLX-α)
                alpha = 0.5
                child = (parent1 + parent2) / 2.0
                # Mutation
                if np.random.rand() < mutation_rate:
                    child[0] = np.random.uniform(0.05, 0.4)  # Broad range
                    child[1] = np.random.uniform(0.2, 0.6)
                new_population.append(child)
            
            genes = new_population[:pop_size]
        
        return best_solution, best_fitness


# =============================================================================
# Main Demonstration
# =============================================================================

def demonstrate_truss_analysis():
    """Demonstrate truss analysis functionality."""
    print("=== Truss Analysis Demo ===")
    
    # Define truss nodes (simple 3-node truss)
    nodes = [
        TrussNode(id=1, x=0.0, y=0.0),
        TrussNode(id=2, x=1.0, y=0.0),
        TrussNode(id=3, x=0.0, y=1.0),
    ]
    
    # Define elements
    elements = [
        TrussElement(index=0, length=1.0, axial_strength=200000.0, angle_deg=90.0),  # Bottom horizontal
        TrussElement(index=1, length=1.41421356, axial_strength=200000.0, angle_deg=45.0),  # Diagonal
        TrussElement(index=2, length=1.0, axial_strength=200000.0, angle_deg=-90.0),  # Top vertical
    ]
    
    truss = TrussAnalyzer(nodes, elements)
    print(f"Global stiffness matrix shape: {truss.K.shape}")
    
    # Apply loads
    forces = np.array([100.0, 0.0, 50.0])  # Force at node 1, 2, 3
    displacements = truss.solve_for_displacements(forces)
    print(f"Displacements: {displacements}")
    
    # Compute moments
    moments = truss.compute_moments(forces)
    print(f"Moments per element: {moments}")


def demonstrate_stress_transformation():
    """Demonstrate stress transformation and failure criteria."""
    print("\n=== Stress Transformation Demo ===")
    
    transformer = StressTransformer(E=207000.0)
    
    # Test case: uniaxial tension
    sigma_x, sigma_y, tau_xy = 100.0, 0.0, 0.0
    t, g = transformer.transform_shear(sigma_x, sigma_y, tau_xy)
    print(f"Uniaxial tension (100 MPa): shear={t:.4f}, shear={g:.4f}")
    
    # Test case: combined stress
    sigma_x, sigma_y, tau_xy = 50.0, 30.0, 20.0
    t, g = transformer.transform_shear(sigma_x, sigma_y, tau_xy)
    print(f"Combined stress (σx=50, σy=30, τxy=20): shear={t:.4f}, shear={g:.4f}")
    
    # Failure criteria
    von_mises = transformer.von_mises_criterion(sigma_x, sigma_y, tau_xy)
    trescas = transformer.trescas_criterion(sigma_x, sigma_y, tau_xy)
    mohr = transformer.mohrs_circle(sigma_x, sigma_y, tau_xy)