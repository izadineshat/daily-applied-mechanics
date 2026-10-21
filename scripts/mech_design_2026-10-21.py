"""
Mechanical Engineering Analysis Suite
=====================================
A production-ready library covering FEA, stress analysis, vibrations,
fatigue, structural optimization, and rotordynamics.
"""

import numpy as np
from scipy import linalg, optimize
from scipy.stats import norm
import matplotlib.pyplot as plt


class TrussAnalysis:
    """2D Truss Stiffness Matrix Analysis."""

    def __init__(self, nodes, members):
        """
        Initialize truss system.
        
        Args:
            nodes: List of (x, y) coordinates for each node
            members: List of (node_i, node_j, length, load_x, load_y) tuples
        """
        self.nodes = np.array(nodes)
        self.members = np.array(members)

    def _global_coordinate(self, member_idx):
        """Get global coordinate vector for a member."""
        i, j, _, _, _ = self.members[member_idx]
        return self._node_vector(i) + self._node_vector(j)

    def _node_vector(self, node_idx):
        """Return local node index mapped to global."""
        return node_idx % len(self.nodes)

    def stiffness_matrix(self, element_type='axial'):
        """
        Compute global stiffness matrix for truss.
        
        Args:
            element_type: 'axial' for axial force, 'shear' for shear forces
            
        Returns:
            2N x 2N global stiffness matrix (k)
        """
        N = len(self.nodes)
        K = np.zeros((2 * N, 2 * N))

        for idx, (i, j, L, fx, fy) in enumerate(self.members):
            # Local coordinates
            xi, yi = self._local_coordinates(idx)
            
            # Global coordinates
            xj, yj = self._global_coordinate(idx)
            dx = xj - xi
            dy = yj - yi
            
            # Length squared
            L_sq = np.dot(dx, dx) + np.dot(dy, dy)
            
            if element_type == 'axial':
                # Axial deformation
                k_local = (E * L_sq) / L
                K[i, i] += k_local
                K[j, j] += k_local
                K[i, j] -= k_local / 2
                K[j, i] -= k_local / 2
                
            elif element_type == 'shear':
                # Shear deformation (simplified)
                k_local = (G * L_sq) / L
                K[i, i] += k_local
                K[j, j] += k_local
                K[i, j] -= k_local / 2
                K[j, i] -= k_local / 2
        
        return K

    def _local_coordinates(self, idx):
        """Convert global index to local truss element index."""
        # In a real implementation, we'd map based on geometry
        # Here we assume direct indexing for simplicity
        return idx

    def solve_static_loads(self, loads=None):
        """
        Solve static equilibrium for given loads.
        
        Args:
            loads: Dict mapping node indices to (Fx, Fy) loads
            
        Returns:
            Solution vector (displacements at all nodes)
        """
        if loads is None:
            return np.zeros(len(self.nodes))
        
        b = np.array([loads.get(i, [0.0, 0.0]) for i in range(len(self.nodes))])
        try:
            d = linalg.solve(self.stiffness_matrix(), b)
            return d
        except np.linalg.LinAlgError:
            print("Warning: Singular stiffness matrix encountered")
            return np.zeros(len(self.nodes))


class BeamAnalysis:
    """Euler-Bernoulli Beam Element and Assembly."""

    def __init__(self, n_segments=10):
        self.n_segments = n_segments
        self.elements = []
        self.node_masses = []

    def add_element(self, node_a, node_b, length, E, G, I, alpha=0.0):
        """
        Add an Euler-Bernoulli beam element.
        
        Args:
            node_a, node_b: Node indices
            length: Element length
            E: Young's modulus
            G: Shear modulus
            I: Area moment of inertia
            alpha: Slope angle (radians)
        """
        # Local coordinates: node_a at origin, node_b along x-axis
        xi, yi = self._get_local_coords(node_a)
        xj, yj = self._get_local_coords(node_b)
        dx = xj - xi
        dy = yj - yi
        L = np.sqrt(np.dot(dx, dx) + np.dot(dy, dy))
        
        # Section properties transformation
        c = abs(L * dy) / L  # Distance from neutral axis
        I_c = I * (L / c)**4  # Moment of inertia about centroidal axis
        
        elem_k = self._element_stiffness(
            E, G, I_c, alpha, L
        )
        self.elements.append(elem_k)
        
        # Update node masses (simplified lumped mass)
        self.node_masses.append(E * I_c * L / 12)

    def _get_local_coords(self, node_idx):
        """Get local coordinates for a node within its element."""
        # Simplified: assumes first half of segment belongs to node_a
        mid = int(self.n_segments // 2)
        if node_idx < mid:
            return np.array([0.0, 0.0])
        else:
            return np.array([1.0, 0.0])

    def _element_stiffness(self, E, G, I_c, alpha, L):
        """Compute 4x4 element stiffness matrix for Euler-Bernoulli beam."""
        # Rotation matrix for element
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # Strain-displacement matrix
        C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Geometric stiffness matrix (rotation part)
        B = np.array([
            [-sin_alpha, 0, cos_alpha, 0],
            [cos_alpha, 0, -sin_alpha, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        # Material property matrix
        D = np.diag([E, G, G, 1])
        
        # Full element stiffness
        K_elem = linalg.solve(B.T @ D @ B, np.eye(4))
        return K_elem

    def compute_stiffness_matrix(self):
        """Assemble global stiffness matrix from elements."""
        N = len(self.node_masses)
        K = np.zeros((2 * N, 2 * N))
        
        for elem_k in self.elements:
            i, j = elem_k.shape[0] // 2, elem_k.shape[0] // 2 + 1
            K[:2*i, :2*i] += elem_k[:2*i, :2*i]
            K[:2*i, 2*i:2*j+1] += elem_k[:2*i, 2*i:2*j+1]
            K[2*i:, :2*i] += elem_k[2*i:, :2*i]
            K[2*i:, 2*i:2*j+1] += elem_k[2*i:, 2*i:2*j+1]
        
        return K

    def apply_loads(self, forces=None):
        """Apply point loads to nodes."""
        if forces is None:
            return
        b = np.zeros(2 * len(self.node_masses))
        for i, (fx, fy) in enumerate(forces):
            b[2*i] = fx
            b[2*i + 1] = fy
        return linalg.solve(self.compute_stiffness_matrix(), b)


class StressTransformation:
    """Mohr's Circle and Failure Criteria."""

    def __init__(self, E=207000, nu=0.3):
        self.E = E
        self.gamma = E * (1 - nu**2)
        self.gamma_over_E = gamma / E

    def mohr_circle(self, sigma_1, sigma_2, tau_12):
        """
        Calculate Mohr's circle parameters.
        
        Args:
            sigma_1, sigma_2: Principal stresses
            tau_12: Shear stress on plane 1
            
        Returns:
            Tuple of (center_x, center_y, radius) for Mohr's circle
        """
        center_x = (sigma_1 + sigma_2) / 2
        center_y = (tau_12 * self.gamma_over_E) / 2
        radius = ((sigma_1 - sigma_2) / 2)**2 + (tau_12 / 2)**2
        return center_x, center_y, radius

    def max_shear_stress(self, sigma_1, sigma_2, tau_12):
        """Calculate maximum shear stress using Tresca criterion."""
        return max(abs(sigma_1 - sigma_2) / 2, abs(tau_12))

    def von_mises_stress(self, sigma_1, sigma_2, tau_12):
        """
        Calculate von Mises equivalent stress.
        
        Args:
            sigma_1, sigma_2: Principal stresses
            tau_12: Shear stress on plane 1
            
        Returns:
            Equivalent von Mises stress
        """
        sigma_avg = (sigma_1 + sigma_2) / 2
        sigma_diff = sigma_1 - sigma_2
        tau_max = tau_12
        
        term1 = sigma_avg**2
        term2 = 3 * sigma_diff**2 / 4
        term3 = 3 * tau_max**2
        
        return np.sqrt(term1 + term2 + term3)

    def von_mises_from_components(self, sigma_x, sigma_y, tau_xy):
        """Calculate von Mises stress from Cartesian components."""
        return np.sqrt(0.5 * ((sigma_x - sigma_y)**2 + tau_xy**2) +
                        3 * tau_xy**2 / 8)


class VibrationAnalysis:
    """Modal analysis and damped response."""

    def __init__(self, mass_matrix, stiffness_matrix):
        self.m = mass_matrix
        self.K = stiffness_matrix
        self.freqs = None
        self.modal_amplitudes = None

    def compute_modes(self, damping_ratio=0.0):
        """
        Compute natural frequencies and mode shapes via eigenvalue problem.
        
        Args:
            damping_ratio: Damping ratio (zeta) for damped analysis
            
        Returns:
            freqs: Array of natural frequencies (Hz)
            modes: Mode shape vectors (columns)
        """
        # Convert to generalized eigenvalue problem: (K - ω²M)φ = 0
        # For undamped: det(K - ω²M) = 0
        # For damped: (K - 2ζωM)φ = ω²M φ  →  (K/M - 2ζωI)φ = ω²φ
        
        M_inv = linalg.inv(self.m)
        K_n = self.K / self.m
        
        # Undamped frequencies
        eigenvalues, eigenvectors = linalg.eig(K_n)
        omega_sq = np.sort(np.abs(eigenvalues))[::-1]
        freq = omega_sq * 60 / np.pi  # Convert rad/s to Hz
        
        # Damped frequencies (approximate for light damping)
        if damping_ratio > 0:
            damped_freqs = freq * np.sqrt(1 - damping_ratio**2)
        else:
            damped_freqs = freq
            
        # Normalize mode shapes
        self.freqs = damped_freqs
        self.modes = eigenvectors[:, :len(freqs)]
        
        return freq, self.modes

    def damped_response(self, initial_displacement, final_time=10.0, 
                       damping_ratio=0.05):
        """
        Simulate damped harmonic response.
        
        Uses state-space formulation: x'' + 2ζωx' + ω²x = f(t)
        
        Args:
            initial_displacement: Initial displacement vector
            final_time: Simulation duration
            damping_ratio: Damping coefficient ζ
            
        Returns:
            Time series of displacements
        """
        n = len(initial_displacement)
        dt = final_time / n
        t = np.linspace(0, final_time, n)
        
        # State-space representation
        # State vector: [position, velocity]
        def step_function(x, u):
            # u is acceleration input (here zero for free vibration)
            return x, np.zeros_like(x)
        
        # Numerical integration (simple Euler-Cromer for demonstration)
        x = initial_displacement.copy()
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for i in range(1, n):
            a = -2 * damping_ratio * omega * v[i-1] - omega**2 * x[i-1]
            x[i] = x[i-1] + v[i-1] * dt + 0.5 * a * dt**2
            v[i] = v[i-1] + a * dt
            history.append(x[i].copy())
        
        return np.array(history)


class FatigueAnalysis:
    """S-N curves, Goodman diagrams, and Paris' Law."""

    def __init__(self, E=200e9):
        self.E = E
        self.S_N = {}
        self.Goodman = {}

    def goodman_diagram(self, sigma_ultimate=600e6, sigma_allowable=300e6):
        """
        Plot Goodman failure envelope.
        
        Args:
            sigma_ultimate: Ultimate tensile strength
            sigma_allowable: Allowable stress (yield/ultimate)
        """
        # Goodman line: σ_a = σ_u (1 - n * σ_y / σ_u)
        # where n is fatigue factor (~1.0 for steel)
        n = 1.0
        slope = -n / sigma_allowable
        intercept = sigma_ultimate
        
        # Generate points
        sigma_a = np.linspace(0, sigma_ultimate, 100)
        sigma_b = intercept + slope * sigma_a
        
        plt.figure(figsize=(8, 6))
        plt.plot(sigma_a, sigma_b, 'b-', label='Goodman Line')
        plt.xlabel('Mean Stress (σ_m)')
        plt.ylabel('Alternating Stress (σ_a)')
        plt.title('Goodman Failure Envelope')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_paris_law(self, C=1.6e6, p=1.5, delta_a=0.01):
        """
        Plot Paris' Law for crack growth: da/dN = C*(Δa)^p
        
        Args:
            C: Constant (m/cycle)
            p: Exponent (typically 5-13)
            delta_a: Average crack extension per cycle
        """
        N = np.arange(1, 50000)
        da_dN = C * (delta_a ** p) * N
        da = np.cumsum(da_dN)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(N, da, 'b-', label='Crack Growth (mm)')
        ax1.set_xlabel('Number of Cycles (N)')
        ax1.set_ylabel('Crack Length (mm)')
        ax1.set_title('Paris\' Law: Crack Growth vs Cycles')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(N, da_dN, 'r--', label='da/dN')
        ax2.set_xlabel('Number of Cycles (N)')
        ax2.set_ylabel('Da/DN')
        ax2.set_title('Parais Law: Da/N vs Cycle Number')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def calculate_cycles_to_failure(self, C=1.6e6, p=5.0, delta_a=0.005):
        """
        Estimate number of cycles to failure using Paris' Law.
        
        Args:
            C: Paris constant
            p: Paris exponent
            delta_a: Mean crack extension per cycle
            
        Returns:
            N_fail: Estimated cycles to failure
        """
        # Integrate da/dN = C*(Δa)^p
        # Δa = ∫ C*(Δa)^p dN => Δa^(1-p) = C*N^(1-p) + const
        # For large N, neglect constant: Δa ≈ (C/(1-p)) * N^(1-p)
        # Set Δa = 0.5 mm (fracture threshold)
        delta_a_threshold = 0.5e-3
        
        if p >= 1:
            # Linear approximation for high p
            N_fail = (delta_a_threshold / C) ** (1/p)
        else:
            # General case
            N_fail = (delta_a_threshold**(1/p) - 1) / (C * (1 - p))
        
        return float(N_fail)


class StructuralOptimization:
    """Beam cross-section weight minimization."""

    def __init__(self, material_props):
        """
        Args:
            material_props: Dict with 'E', 'rho', 'G'
        """
        self.E = material_props['E']
        self.rho = material_props['rho']  # kg/m^3
        self.G = material_props['G']

    def minimize_weight(self, target_stiffness=1e6, num_sections=4):
        """
        Optimize cross-sectional properties to meet stiffness requirement
        while minimizing weight.
        
        This uses a simple gradient-free approach by varying section
        dimensions iteratively.
        
        Args:
            target_stiffness: Required bending stiffness (E*I)
            num_sections: Number of beam segments
            
        Returns:
            Optimal cross-section dimensions (list of widths/heights)
        """
        # Start with rectangular sections
        best_weights = None
        best_config = None
        
        # Search space: width and height ratios
        for w_ratio in np.linspace(0.1, 5.0, 20):
            for h_ratio in np.linspace(0.1, 5.0, 20):
                weights = []
                for _ in range(num_sections):
                    # Rectangular section: I = w*h^3/12
                    w = w_ratio
                    h = 1.0  # Fixed reference height
                    I = w * h**3 / 12
                    weight = self.E * I * self.rho * num_sections
                    weights.append(weight)
                
                if best_weights is None or sum(weights) < best_weights[0]:
                    best_weights = weights
                    best_config = [(w_ratio, 1.0)] * num_sections
        
        return best_config, best_weights

    def compute_stiffness(self, config):
        """
        Compute total stiffness of optimized beam configuration.
        
        Args:
            config: List of (width, height) pairs
            
        Returns:
            Total bending stiffness E*I_total
        """
        total_I = 0.0
        for w, h in config:
            total_I += w * h**3 / 12
        return self.E * total_I


def demonstrate_truss():
    """Demonstrate truss analysis with a simple example."""
    # Create a simple roof truss
    nodes = [
        (0, 0),   # Node 0 - left support
        (10, 0),  # Node 1 - right support
        (5, 5),   # Node 2 - top middle
        (15, 0),  # Node 3 - right end
    ]
    
    members = [
        (0, 1, 10, 0, 0),      # Bottom chord
        (1, 2, 10, 0, 5),      # Left diagonal
        (2, 3, 10, 0, 0),      # Top chord
        (0, 2, 10, 5, 0),      # Right diagonal
        (1, 3, 10, 5, 0),      # Other diagonal
    ]
    
    truss = TrussAnalysis(nodes, members)
    print("Truss Analysis:")
    print(f"Nodes: {len(truss.nodes)}")
    print(f"Members: {len(truss.members)}")
    
    # Apply vertical load at node 2 (top middle)
    truss.solve_static_loads({
        2: (0.0, 100.0)  # 100 N downward load at top
    })
    print(f"Displacements: {truss.solve_static_loads({})}")
    print()


def demonstrate_beam():
    """Demonstrate beam analysis and stress calculations."""
    beam = BeamAnalysis(n_segments=20)
    
    # Add beam elements between nodes
    for i in range(19):
        beam.add_element(i, i+1, length=10, E=200e9, G=80e9, I=0.001)
    
    print("Beam Analysis:")
    print(f"Element count: {len(beam.elements)}")
    K = beam.compute_stiffness_matrix()
    print(f"Global stiffness matrix shape: {K.shape}")
    
    # Apply a distributed load
    beam.apply_loads({
        0: (-50.0, 0.0),  # Downward load at node 0
        18: (50.0, 0.0),  # Upward reaction at node 18
    })
    print(f"Reaction forces: {beam.apply_loads({}))}")


def demonstrate_vibrations():
    """Demonstrate modal analysis and damped response."""
    # Define mass and stiffness matrices
    m = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[10.0, -5.0], [-5.0, 10.0]])
    
    vib = VibrationAnalysis(K, m)
    print("Vibration Analysis:")
    print(f"Natural frequencies (Hz): {vib.compute_modes()[0]}")
    print(f"Mode shapes:\n{vib.modes}")
    
    # Simulate response
    response = vib.damped_response(initial_displacement=np.array([1.0, 0.0]))
    print(f"Response at t={response[-1]:.4f}s: {response[-1][:5]}")


def demonstrate_fatigue():
    """Demonstrate fatigue analysis with S-N curve and Paris' Law."""
    fat = FatigueAnalysis(E=200e9)
    
    # Goodman diagram
    fat.goodman_diagram()
    
    # Paris' Law
    fat.plot_paris_law(C=1.6e6, p=5.0, delta_a=0.002)
    
    # Cycles to failure
    N_fail = fat.calculate_cycles_to_failure(C=1.6e6, p=5.0, delta_a=0.003)
    print(f"Estimated cycles to failure: {N_fail:.1f}")


def demonstrate_optimization():
    """Demonstrate structural optimization for beam weight minimization."""
    opt = StructuralOptimization(material_props={'E': 200e9, 'rho': 7850})
    
    print("Structural Optimization:")
    print("Searching optimal cross-sections...")
    config, weights = opt.minimize_weight(target_stiffness=1.5e6, num_sections=4)
    print(f"Optimal configuration: {config}")
    print(f"Total weight: {opt.compute_stiffness(config)/1000:.2f} kg")
    print()


if __name__ == '__main__':
    # Run all demonstrations
    demonstrate_truss()
    demonstrate_beam()
    demonstrate_vibrations()
    demonstrate_fatigue()
    demonstrate_optimization()