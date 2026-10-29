import numpy as np
from scipy.linalg import eigh, eigvals, solve
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable, Union
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MaterialProperties:
    """Material properties for structural analysis."""
    E: float = 200e9          # Young's modulus (Pa)
    nu: float = 0.3           # Poisson's ratio
    rho: float = 7850         # Density (kg/m^3)
    Sy: float = 250e6         # Yield strength (Pa)
    Su: float = 400e6         # Ultimate tensile strength (Pa)
    Se: float = 200e6         # Endurance limit (Pa)
    K_IC: float = 50e6        # Fracture toughness (Pa*sqrt(m))
    C: float = 1e-12          # Paris law coefficient
    m: float = 3.0            # Paris law exponent


class StressStrainAnalysis:
    """Stress and strain transformation, failure criteria."""
    
    @staticmethod
    def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
        """Return principal stresses (sigma1, sigma2, sigma3) and orientation angle."""
        avg = 0.5 * (sigma_x + sigma_y)
        radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        sigma1 = avg + radius
        sigma2 = avg - radius
        sigma3 = 0.0
        theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_x - sigma_y)
        return sigma1, sigma2, sigma3, theta_p
    
    @staticmethod
    def mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float, theta: float) -> Tuple[float, float]:
        """Stress transformation using Mohr's circle at angle theta (rad)."""
        sigma_avg = 0.5 * (sigma_x + sigma_y)
        R = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        sigma_n = sigma_avg + R * np.cos(2 * theta)
        tau_nt = -R * np.sin(2 * theta)
        return sigma_n, tau_nt
    
    @staticmethod
    def von_mises(sigma_x: float, sigma_y: float, tau_xy: float, sigma_z: float = 0.0) -> float:
        """Von Mises equivalent stress for plane stress/strain."""
        return np.sqrt(
            0.5 * ((sigma_x - sigma_y)**2 + (sigma_y - sigma_z)**2 + (sigma_z - sigma_x)**2)
            + 3 * tau_xy**2
        )
    
    @staticmethod
    def tresca(sigma_x: float, sigma_y: float, tau_xy: float, sigma_z: float = 0.0) -> float:
        """Tresca equivalent stress (max shear stress * 2)."""
        s1, s2, s3, _ = StressStrainAnalysis.principal_stresses(sigma_x, sigma_y, tau_xy)
        if sigma_z != 0:
            principal = np.array([s1, s2, s3, sigma_z])
        else:
            principal = np.array([s1, s2, s3])
        return np.max(principal) - np.min(principal)
    
    @staticmethod
    def strain_transformation(eps_x: float, eps_y: float, gamma_xy: float, theta: float) -> Tuple[float, float, float]:
        """Strain transformation at angle theta."""
        eps_avg = 0.5 * (eps_x + eps_y)
        R = np.sqrt(((eps_x - eps_y) / 2)**2 + (gamma_xy / 2)**2)
        eps_n = eps_avg + R * np.cos(2 * theta)
        gamma_nt = -2 * R * np.sin(2 * theta)
        eps_t = eps_avg - R * np.cos(2 * theta)
        return eps_n, eps_t, gamma_nt


class FEA_1D:
    """1D Finite Element Analysis for truss and beam elements."""
    
    @staticmethod
    def truss_stiffness(E: float, A: float, L: float, theta: float) -> np.ndarray:
        """Local-to-global stiffness matrix for 2D truss element (4x4)."""
        c = np.cos(theta)
        s = np.sin(theta)
        k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
        T = np.array([[c, s, 0, 0], [0, 0, c, s]])
        return T.T @ k_local @ T
    
    @staticmethod
    def beam_stiffness(E: float, I: float, L: float) -> np.ndarray:
        """Euler-Bernoulli beam element stiffness (4x4: v, theta at each node)."""
        k = np.zeros((4, 4))
        k[0, 0] = 12 * E * I / L**3
        k[0, 1] = 6 * E * I / L**2
        k[0, 2] = -12 * E * I / L**3
        k[0, 3] = 6 * E * I / L**2
        k[1, 0] = 6 * E * I / L**2
        k[1, 1] = 4 * E * I / L
        k[1, 2] = -6 * E * I / L**2
        k[1, 3] = 2 * E * I / L
        k[2, 0] = -12 * E * I / L**3
        k[2, 1] = -6 * E * I / L**2
        k[2, 2] = 12 * E * I / L**3
        k[2, 3] = -6 * E * I / L**2
        k[3, 0] = 6 * E * I / L**2
        k[3, 1] = 2 * E * I / L
        k[3, 2] = -6 * E * I / L**2
        k[3, 3] = 4 * E * I / L
        return k
    
    @staticmethod
    def beam_mass(rho: float, A: float, L: float) -> np.ndarray:
        """Consistent mass matrix for Euler-Bernoulli beam (4x4)."""
        m = (rho * A * L / 420) * np.array([
            [156, 22*L, 54, -13*L],
            [22*L, 4*L**2, 13*L, -3*L**2],
            [54, 13*L, 156, -22*L],
            [-13*L, -3*L**2, -22*L, 4*L**2]
        ])
        return m
    
    @staticmethod
    def assemble_global(n_nodes: int, elements: List[dict], dof_per_node: int) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global K and M matrices."""
        ndof = n_nodes * dof_per_node
        K = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))
        for elem in elements:
            conn = elem['connectivity']
            ke = elem['ke']
            me = elem.get('me', np.zeros_like(ke))
            dofs = []
            for node in conn:
                dofs.extend([node * dof_per_node + i for i in range(dof_per_node)])
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += ke[i, j]
                    M[gi, gj] += me[i, j]
        return K, M
    
    @staticmethod
    def apply_bc(K: np.ndarray, F: np.ndarray, fixed_dofs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Dirichlet boundary conditions (penalty method)."""
        K_mod = K.copy()
        F_mod = F.copy()
        penalty = 1e12 * np.max(np.abs(np.diag(K)))
        for dof in fixed_dofs:
            K_mod[dof, dof] += penalty
            F_mod[dof] += penalty * 0.0
        return K_mod, F_mod
    
    @staticmethod
    def solve_static(K: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Solve K*u = F."""
        return solve(K, F)


class VibrationAnalysis:
    """Mechanical vibrations: SDOF, MDOF, modal analysis, state-space."""
    
    @staticmethod
    def sdof_response(m: float, c: float, k: float, F0: float, omega: float, 
                      t: np.ndarray, x0: float = 0.0, v0: float = 0.0) -> np.ndarray:
        """SDOF forced harmonic response (analytical)."""
        wn = np.sqrt(k / m)
        zeta = c / (2 * np.sqrt(k * m))
        wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
        
        # Particular solution
        H = 1 / np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
        phi = np.arctan2(c * omega, k - m * omega**2)
        x_p = F0 * H * np.cos(omega * t - phi)
        
        # Homogeneous solution
        if zeta < 1:
            A = x0 - F0 * H * np.cos(-phi)
            B = (v0 + F0 * H * omega * np.sin(-phi) + zeta * wn * A) / wd
            x_h = np.exp(-zeta * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
        elif zeta == 1:
            A = x0 - F0 * H * np.cos(-phi)
            B = v0 + F0 * H * omega * np.sin(-phi) + wn * A
            x_h = np.exp(-wn * t) * (A + B * t)
        else:
            s1 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
            s2 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
            A = x0 - F0 * H * np.cos(-phi)
            B = (v0 + F0 * H * omega * np.sin(-phi) - s1 * A) / (s2 - s1)
            A = A - B
            x_h = A * np.exp(s1 * t) + B * np.exp(s2 * t)
        
        return x_p + x_h
    
    @staticmethod
    def modal_analysis(K: np.ndarray, M: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Solve generalized eigenvalue problem K*phi = lambda*M*phi."""
        eigvals_, eigvecs = eigh(K, M, subset_by_index=[0, min(n_modes, K.shape[0]) - 1])
        return eigvals_, eigvecs
    
    @staticmethod
    def modal_superposition(K: np.ndarray, M: np.ndarray, C: np.ndarray, 
                            F_t: Callable[[float], np.ndarray], t_span: Tuple[float, float],
                            t_eval: np.ndarray, x0: np.ndarray, v0: np.ndarray,
                            n_modes: int = None) -> np.ndarray:
        """MDOF transient response using modal superposition."""
        n = K.shape[0]
        if n_modes is None:
            n_modes = n
        
        lambdas, Phi = VibrationAnalysis.modal_analysis(K, M, n_modes)
        omega_n = np.sqrt(lambdas)
        
        # Modal damping (assume proportional or given)
        # Here we use modal damping ratios
        zeta = np.ones(n_modes) * 0.02  # 2% default
        
        # Transform initial conditions
        q0 = Phi.T @ M @ x0
        qd0 = Phi.T @ M @ v0
        
        # Time integration for each mode
        q = np.zeros((n_modes, len(t_eval)))
        for i in range(n_modes):
            wn = omega_n[i]
            z = zeta[i]
            wd = wn * np.sqrt(1 - z**2) if z < 1 else 0
            
            def modal_force(t):
                F = F_t(t)
                return Phi[:, i] @ F
            
            if z < 1:
                # Underdamped
                def ode(t, y):
                    qi, qdi = y
                    return [qdi, modal_force(t) - 2*z*wn*qdi - wn**2*qi]
                sol = solve_ivp(ode, t_span, [q0[i], qd0[i]], t_eval=t_eval, method='RK45')
                q[i, :] = sol.y[0]
            else:
                # Overdamped - use numerical integration
                def ode(t, y):
                    qi, qdi = y
                    return [qdi, modal_force(t) - 2*z*wn*qdi - wn**2*qi]
                sol = solve_ivp(ode, t_span, [q0[i], qd0[i]], t_eval=t_eval, method='RK45')
                q[i, :] = sol.y[0]
        
        # Transform back to physical coordinates
        x = Phi @ q
        return x
    
    @staticmethod
    def state_space(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                    u: Callable[[float], np.ndarray], t_span: Tuple[float, float],
                    x0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """Linear state-space integration dx/dt = Ax + Bu, y = Cx + Du."""
        def ode(t, x):
            return A @ x + B @ u(t)
        sol = solve_ivp(ode, t_span, x0, t_eval=t_eval, method='RK45')
        return sol.y


class FatigueFracture:
    """Fatigue and fracture mechanics analysis."""
    
    @staticmethod
    def sn_curve_stress_life(Sa: float, Se: float, Su: float, b: float = -0.1) -> float:
        """Basquin's equation: Sa = Se * (2N)^b for high-cycle fatigue."""
        if Sa <= Se:
            return 1e9  # Infinite life
        return 0.5 * (Sa / Se) ** (1 / b)
    
    @staticmethod
    def goodman_correction(Sa: float, Sm: float, Se: float, Su: float) -> float:
        """Goodman mean stress correction: Sa_eq = Sa / (1 - Sm/Su)."""
        if Sm >= Su:
            return np.inf
        return Sa / (1 - Sm / Su)
    
    @staticmethod
    def gerber_correction(Sa: float, Sm: float, Se: float, Su: float) -> float:
        """Gerber parabolic mean stress correction."""
        if Sm >= Su:
            return np.inf
        return Sa / (1 - (Sm / Su)**2)
    
    @staticmethod
    def soderberg_correction(Sa: float, Sm: float, Se: float, Sy: float) -> float:
        """Soderberg mean stress correction (conservative)."""
        if Sm >= Sy:
            return np.inf
        return Sa / (1 - Sm / Sy)
    
    @staticmethod
    def paris_law(da_dN: float, C: float, m: float, delta_K: float) -> float:
        """Paris' law: da/dN = C * (ΔK)^m. Returns ΔK for given da/dN."""
        return (da_dN / C) ** (1 / m)
    
    @staticmethod
    def crack_growth_life(a0: float, af: float, C: float, m: float, 
                          delta_sigma: float, geometry_factor: Callable[[float], float]) -> float:
        """Integrate Paris law for crack growth life (cycles)."""
        def integrand(a):
            K = geometry_factor(a) * delta_sigma * np.sqrt(np.pi * a)
            return 1 / (C * K**m)
        
        # Numerical integration
        a_vals = np.linspace(a0, af, 1000)
        integrand_vals = [integrand(a) for a in a_vals]
        return np.trapz(integrand_vals, a_vals)
    
    @staticmethod
    def stress_intensity_factor(sigma: float, a: float, geometry: str = 'center_crack') -> float:
        """Stress intensity factor K_I for common geometries."""
        if geometry == 'center_crack':
            # Infinite plate with center crack length 2a
            return sigma * np.sqrt(np.pi * a)
        elif geometry == 'edge_crack':
            # Edge crack in semi-infinite plate
            return 1.12 * sigma * np.sqrt(np.pi * a)
        elif geometry == 'surface_crack':
            # Semi-elliptical surface crack (simplified)
            return 1.12 * sigma * np.sqrt(np.pi * a)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")


class StructuralOptimization:
    """Structural optimization for beam cross-sections."""
    
    @staticmethod
    def beam_weight_minimization(L: float, P: float, sigma_allow: float, 
                                  rho: float, bounds: dict,
                                  section_type: str = 'rectangular') -> dict:
        """Minimize weight of simply supported beam with central load."""
        
        def objective(x):
            if section_type == 'rectangular':
                b, h = x
                A = b * h
                I = b * h**3 / 12
                S = I / (h / 2)  # Section modulus
            elif section_type == 'circular':
                d, = x
                A = np.pi * d**2 / 4
                I = np.pi * d**4 / 64
                S = I / (d / 2)
            elif section_type == 'I_beam':
                b, h, tw, tf = x
                A = 2*b*tf + (h-2*tf)*tw
                I = (b*h**3 - (b-tw)*(h-2*tf)**3) / 12
                S = I / (h / 2)
            else:
                raise ValueError("Unknown section type")
            
            weight = rho * A * L
            M_max = P * L / 4
            sigma_max = M_max / S
            
            # Penalty for constraint violation
            penalty = 1e6 * max(0, sigma_max - sigma_allow)
            return weight + penalty
        
        if section_type == 'rectangular':
            x0 = [(bounds['b'][0] + bounds['b'][1])/2, (bounds['h'][0] + bounds['h'][1])/2]
            bnds = [bounds['b'], bounds['h']]
        elif section_type == 'circular':
            x0 = [(bounds['d'][0] + bounds['d'][1])/2]
            bnds = [bounds['d']]
        elif section_type == 'I_beam':
            x0 = [(bounds['b'][0]+bounds['b'][1])/2, (bounds['h'][0]+bounds['h'][1])/2,
                  (bounds['tw'][0]+bounds['tw'][1])/2, (bounds['tf'][0]+bounds['tf'][1])/2]
            bnds = [bounds['b'], bounds['h'], bounds['tw'], bounds['tf']]
        
        result = minimize(objective, x0, bounds=bnds, method='SLSQP')
        return {
            'optimal_dimensions': result.x,
            'minimum_weight': result.fun,
            'success': result.success,
            'message': result.message
        }
    
    @staticmethod
    def genetic_algorithm_optimize(objective: Callable, bounds: List[Tuple[float, float]], 
                                    pop_size: int = 50, generations: int = 100) -> dict:
        """Simple genetic algorithm for global optimization."""
        n_vars = len(bounds)
        pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, n_vars))
        
        for gen in range(generations):
            fitness = np.array([objective(ind) for ind in pop])
            idx = np.argsort(fitness)
            pop = pop[idx]
            
            # Elitism: keep top 20%
            n_elite = max(1, pop_size // 5)
            new_pop = pop[:n_elite].copy()
            
            # Crossover and mutation
            while len(new_pop) < pop_size:
                parent1 = pop[np.random.randint(n_elite)]
                parent2 = pop[np.random.randint(n_elite)]
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Mutation
                if np.random.random() < 0.1:
                    mut_idx = np.random.randint(n_vars)
                    child[mut_idx] += np.random.normal(0, 0.1 * (bounds[mut_idx][1] - bounds[mut_idx][0]))
                    child[mut_idx] = np.clip(child[mut_idx], bounds[mut_idx][0], bounds[mut_idx][1])
                
                new_pop = np.vstack([new_pop, child])
            
            pop = new_pop
        
        best = pop[0]
        return {'optimal_design': best, 'objective_value': objective(best)}


class FailureTheories:
    """Failure theories for ductile and brittle materials."""
    
    @staticmethod
    def max_normal_stress(sigma1: float, sigma2: float, sigma3: float, 
                           Sut: float, Suc: float) -> float:
        """Rankine / Maximum Normal Stress Theory (brittle)."""
        max_tensile = max(sigma1, sigma2, sigma3, 0)
        max_compressive = min(sigma1, sigma2, sigma3, 0)
        FS_tension = Sut / max_tensile if max_tensile > 0 else np.inf
        FS_compression = abs(Suc / max_compressive) if max_compressive < 0 else np.inf
        return min(FS_tension, FS_compression)
    
    @staticmethod
    def max_shear_stress(sigma1: float, sigma2: float, sigma3: float, Sy: float) -> float:
        """Tresca / Maximum Shear Stress Theory (ductile)."""
        tau_max = 0.5 * (max(sigma1, sigma2, sigma3) - min(sigma1, sigma2, sigma3))
        return Sy / (2 * tau_max) if tau_max > 0 else np.inf
    
    @staticmethod
    def von_mises_fs(sigma1: float, sigma2: float, sigma3: float, Sy: float) -> float:
        """Von Mises / Distortion Energy Theory (ductile)."""
        sigma_eq = np.sqrt(0.5 * ((sigma1 - sigma2)**2 + (sigma2 - sigma3)**2 + (sigma3 - sigma1)**2))
        return Sy / sigma_eq if sigma_eq > 0 else np.inf
    
    @staticmethod
    def coulomb_mohr(sigma1: float, sigma2: float, sigma3: float, 
                      Sut: float, Suc: float) -> float:
        """Coulomb-Mohr theory (brittle with different tension/compression strength)."""
        # Simplified for principal stresses ordered sigma1 >= sigma2 >= sigma3
        sigmas = sorted([sigma1, sigma2, sigma3], reverse=True)
        s1, s2, s3 = sigmas
        
        if s1 >= 0 and s3 >= 0:
            return Sut / s1
        elif s1 <= 0 and s3 <= 0:
            return abs(Suc / s3)
        else:
            return 1 / (s1/Sut - s3/Suc)
    
    @staticmethod
    def modified_mohr(sigma1: float, sigma2: float, sigma3: float, 
                       Sut: float, Suc: float) -> float:
        """Modified Mohr theory."""
        sigmas = sorted([sigma1, sigma2, sigma3], reverse=True)
        s1, s2, s3 = sigmas
        
        if s1 >= 0 and s3 >= 0:
            return Sut / s1
        elif s1 <= 0 and s3 <= 0:
            return abs(Suc / s3)
        elif s1 >= 0 and s3 <= 0:
            if abs(s3) <= s1 * Suc / Sut:
                return Sut / s1
            else:
                return 1 / (s1/Sut - s3/Suc)
        return np.inf


class Rotordynamics:
    """Basic rotordynamics analysis."""
    
    @staticmethod
    def critical_speeds(K: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Undamped critical speeds (natural frequencies) of rotor."""
        eigvals, _ = eigh(K, M)
        return np.sqrt(eigvals) / (2 * np.pi)  # Convert to Hz
    
    @staticmethod
    def unbalance_response(M: np.ndarray, K: np.ndarray, C: np.ndarray,
                            unbalance: np.ndarray, omega: float) -> np.ndarray:
        """Steady-state response to unbalance force at speed omega (rad/s)."""
        # Force vector: M * e * omega^2
        F = unbalance * omega**2
        # Complex dynamic stiffness
        Z = K - omega**2 * M + 1j * omega * C
        X = solve(Z, F)
        return X
    
    @staticmethod
    def campbell_diagram(K: np.ndarray, M: np.ndarray, G: np.ndarray,
                          omega_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Campbell diagram for gyroscopic system (K, M, G = skew-symmetric gyroscopic)."""
        n = K.shape[0]
        freqs = np.zeros((len(omega_range), n))
        for i, omega in enumerate(omega_range):
            # State space for gyroscopic system
            A = np.block([
                [np.zeros((n, n)), np.eye(n)],
                [-solve(M, K), -solve(M, (C + omega * G) if 'C' in locals() else omega * G)]
            ])
            eigvals = eigvals(A)
            # Extract positive imaginary parts (forward whirl)
            freq = np.sort(np.abs(np.imag(eigvals)))[:n]
            freqs[i, :] = freq / (2 * np.pi)
        return omega_range, freqs


def demo_stress_strain():
    """Demonstrate stress/strain analysis."""
    print("=" * 60)
    print("STRESS/STRAIN TRANSFORMATION DEMO")
    print("=" * 60)
    
    # Given stress state
    sigma_x, sigma_y, tau_xy = 100e6, 50e6, 30e6  # Pa
    
    s1, s2, s3, theta_p = StressStrainAnalysis.principal_stresses(sigma_x, sigma_y, tau_xy)
    print(f"Principal stresses: σ1={s1/1e6:.1f} MPa, σ2={s2/1e6:.1f} MPa, σ3={s3/1e6:.1f} MPa")
    print(f"Principal angle: {np.degrees(theta_p):.1f}°")
    
    vm = StressStrainAnalysis.von_mises(sigma_x, sigma_y, tau_xy)
    tr = StressStrainAnalysis.tresca(sigma_x, sigma_y, tau_xy)
    print(f"Von Mises stress: {vm/1e6:.1f} MPa")
    print(f"Tresca stress: {tr/1e6:.1f} MPa")
    
    # Mohr's circle at 30°
    theta = np.radians(30)
    sigma_n, tau_nt = StressStrainAnalysis.mohr_circle(sigma_x, sigma_y, tau_xy, theta)
    print(f"At θ=30°: σ_n={sigma_n/1e6:.1f} MPa, τ_nt={tau_nt/1e6:.1f} MPa")
    
    # Strain transformation
    eps_x, eps_y, gamma_xy = 0.001, 0.0005, 0.0008
    eps_n, eps_t, gamma_nt = StressStrainAnalysis.strain_transformation(eps_x, eps_y, gamma_xy, theta)
    print(f"Strains at θ=30°: ε_n={eps_n:.6f}, ε_t={eps_t:.6f}, γ_nt={gamma_nt:.6f}")


def demo_fea_1d():
    """Demonstrate 1D FEA for truss and beam."""
    print("\n" + "=" * 60)
    print("1D FEA DEMO: TRUSS AND BEAM")
    print("=" * 60)
    
    # Truss example: 2-bar truss
    E = 200e9
    A = 1e-4
    L = 2.0
    theta1, theta2 = np.radians(45), np.radians(-45)
    
    k1 = FEA_1D.truss_stiffness(E, A, L, theta1)
    k2 = FEA_1D.truss_stiffness(E, A, L, theta2)
    
    # Global assembly (3 nodes, 2 DOF each)
    K_global = np.zeros((6, 6))
    # Element 1: nodes 0-1
    dofs1 = [0, 1, 2, 3]
    for i, gi in enumerate(dofs1):
        for j, gj in enumerate(dofs1):
            K_global[gi, gj] += k1[i, j]
    # Element 2: nodes 1-2
    dofs2 = [2, 3, 4, 5]
    for i, gi in enumerate(dofs2):
        for j, gj in enumerate(dofs2):
            K_global[gi, gj] += k2[i, j]
    
    # Boundary conditions: node 0 and 2 fixed
    fixed = [0, 1, 4, 5]
    F = np.zeros(6)
    F[2] = -10000  # Vertical load at node 1
    F[3] = 0
    
    K_mod, F_mod = FEA_1D.apply_bc(K_global, F, fixed)
    u = FEA_1D.solve_static(K_mod, F_mod)
    print(f"Node 1 displacement: ux={u[2]:.6e} m, uy={u[3]:.6e} m")
    
    # Beam example: cantilever with tip load
    L_beam = 5.0
    I = 1e-6
    n_elem = 10
    n_nodes = n_elem + 1
    
    elements = []
    for i in range(n_elem):
        ke = FEA_1D.beam_stiffness(E, I, L_beam/n_elem)
        me = FEA_1D.beam_mass(7850, 0.01, L_beam/n_elem)
        elements.append({
            'connectivity': [i, i+1],
            'ke': ke,
            'me': me
        })
    
    K, M = FEA_1D.assemble_global(n_nodes, elements, 2)
    
    # Fixed at node 0
    fixed = [0, 1]
    F = np.zeros(2 * n_nodes)
    F[-2] = -1000  # Tip vertical force
    
    K_mod, F_mod = FEA_1D.apply_bc(K, F, fixed)
    u = FEA_1D.solve_static(K_mod, F_mod)
    print(f"Tip deflection: {u[-2]:.6e} m")
    
    # Modal analysis
    lambdas, modes = VibrationAnalysis.modal_analysis(K, M, n_modes=3)
    freqs_hz = np.sqrt(lambdas) / (2 * np.pi)
    print(f"First 3 natural frequencies: {freqs_hz} Hz")


def demo_vibrations():
    """Demonstrate vibration analysis."""
    print("\n" + "=" * 60)
    print("VIBRATION ANALYSIS DEMO")
    print("=" * 60)
    
    # SDOF forced response
    m, c, k = 10.0, 5.0, 10000.0
    F0, omega = 100.0, 30.0
    t = np.linspace(0, 10, 1000)
    x = VibrationAnalysis.sdof_response(m, c, k, F0, omega, t)
    print(f"SDOF steady-state amplitude: {np.max(np.abs(x[-100:])):.6f} m")
    
    # MDOF example: 3-DOF system
    M = np.diag([2.0, 1.5, 1.0])
    K = np.array([[2000, -1000, 0],
                  [-1000, 2000, -1000],
                  [0, -1000, 1000]])
    C = 0.02 * K + 0.001 * M  # Rayleigh damping
    
    lambdas, Phi = VibrationAnalysis.modal_analysis(K, M, 3)
    print(f"Natural frequencies: {np.sqrt(lambdas)/(2*np.pi)} Hz")
    print(f"Mode shapes:\n{Phi}")
    
    # Impulse response via modal superposition
    def impulse_force(t):
        return np.array([1000.0 if t < 0.01 else 0.0, 0.0, 0.0])
    
    t_eval = np.linspace(0, 5, 500)
    x0 = np.zeros(3)
    v0 = np.zeros(3)
    x_response = VibrationAnalysis.modal_superposition(
        K, M, C, impulse_force, (0, 5), t_eval, x0, v0, n_modes=3
    )
    print(f"Max displacement at DOF 1: {np.max(np.abs(x_response[0, :])):.6f} m")


def demo_fatigue_fracture():
    """Demonstrate fatigue and fracture analysis."""
    print("\n" + "=" * 60)
    print("FATIGUE & FRACTURE MECHANICS DEMO")
    print("=" * 60)
    
    mat = MaterialProperties()
    
    # S-N curve
    Sa = 300e6
    Nf = FatigueFracture.sn_curve_stress_life(Sa, mat.Se, mat.Su)
    print(f"Cycles to failure at Sa={Sa/1e6:.0f} MPa: {Nf:.0f}")
    
    # Mean stress corrections
    Sa, Sm = 200e6, 100e6
    Sa_goodman = FatigueFracture.goodman_correction(Sa, Sm, mat.Se, mat.Su)
    Sa_gerber = FatigueFracture.gerber_correction(Sa, Sm, mat.Se, mat.Su)
    Sa_soderberg = FatigueFracture.soderberg_correction(Sa, Sm, mat.Se, mat.Sy)
    print(f"Equivalent alternating stress (Goodman): {Sa_goodman/1e6:.1f} MPa")
    print(f"Equivalent alternating stress (Gerber): {Sa_gerber/1e6:.1f} MPa")
    print(f"Equivalent alternating stress (Soderberg): {Sa_soderberg/1e6:.1f} MPa")
    
    # Paris law crack growth
    a0, af = 1e-3, 10e-3  # m
    delta_sigma = 150e6
    geometry_factor = lambda a: 1.12 * np.sqrt(np.pi * a)  # Edge crack
    
    life = FatigueFracture.crack_growth_life(a0, af, mat.C, mat.m, delta_sigma, geometry_factor)
    print(f"Crack growth life (a0={a0*1000:.1f}mm → af={af*1000:.1f}mm): {life:.0f} cycles")
    
    # Stress intensity factor
    K_I = FatigueFracture.stress_intensity_factor(200e6, 5e-3, 'edge_crack')
    print(f"K_I for edge crack a=5mm at 200 MPa: {K_I/1e6:.2f} MPa√m")


def demo_optimization():
    """Demonstrate structural optimization."""
    print("\n" + "=" * 60)
    print("STRUCTURAL OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Rectangular beam optimization
    bounds = {'b': (0.02, 0.2), 'h': (0.05, 0.5)}
    result = StructuralOptimization.beam_weight_minimization(
        L=3.0, P=10000, sigma_allow=150e6, rho=7850,
        bounds=bounds, section_type='rectangular'
    )
    print(f"Rectangular beam: b={result['optimal_dimensions'][0]*1000:.1f}mm, "
          f"h={result['optimal_dimensions'][1]*1000:.1f}mm")
    print(f"Minimum weight: {result['minimum_weight']:.2f} kg")
    
    # Circular beam
    bounds_circ = {'d': (0.03, 0.2)}
    result_circ = StructuralOptimization.beam_weight_minimization(
        L=3.0, P=10000, sigma_allow=150e6, rho=7850,
        bounds=bounds_circ, section_type='circular'
    )
    print(f"Circular beam: d={result_circ['optimal_dimensions'][0]*1000:.1f}mm")
    print(f"Minimum weight: {result_circ['minimum_weight']:.2f} kg")
    
    # Genetic algorithm example
    def test_objective(x):
        return x[0]**2 + x[1]**2 + 10 * np.sin(x[0]) * np.cos(x[1])
    
    ga_result = StructuralOptimization.genetic_algorithm_optimize(
        test_objective, [(-5, 5), (-5, 5)], pop_size=30, generations=50
    )
    print(f"GA optimum: x={ga_result['optimal_design']}, f={ga_result['objective_value']:.4f}")


def demo_failure_theories():
    """Demonstrate failure theories."""
    print("\n" + "=" * 60)
    print("FAILURE THEORIES DEMO")
    print("=" * 60)
    
    # Stress state
    sigma_x, sigma_y, tau_xy = 150e6, 50e6, 60e6
    s1, s2, s3, _ = StressStrainAnalysis.principal_stresses(sigma_x, sigma_y, tau_xy)
    print(f"Principal stresses: σ1={s1/1e6:.1f}, σ2={s2/1e6:.1f}, σ3={s3/1e6:.1f} MPa")
    
    Sy, Sut, Suc = 250e6, 400e6, 600e6
    
    fs_tresca = FailureTheories.max_shear_stress(s1, s2, s3, Sy)
    fs_vm = FailureTheories.von_mises_fs(s1, s2, s3, Sy)
    fs_rankine = FailureTheories.max_normal_stress(s1, s2, s3, Sut, Suc)
    fs_coulomb = FailureTheories.coulomb_mohr(s1, s2, s3, Sut, Suc)
    fs_mod_mohr = FailureTheories.modified_mohr(s1, s2, s3, Sut, Suc)
    
    print(f"Tresca FS: {fs_tresca:.2f}")
    print(f"Von Mises FS: {fs_vm:.2f}")
    print(f"Max Normal Stress FS: {fs_rankine:.2f}")
    print(f"Coulomb-Mohr FS: {fs_coulomb:.2f}")
    print(f"Modified Mohr FS: {fs_mod_mohr:.2f}")


def demo_rotordynamics():
    """Demonstrate rotordynamics."""
    print("\n" + "=" * 60)
    print("ROTOR DYNAMICS DEMO")
    print("=" * 60)
    
    # Simple Jeffcott rotor
    m = 10.0  # kg
    k = 1e6   # N/m
    c = 100.0 # Ns/m
    
    M = np.array([[m]])
    K = np.array([[k]])
    C = np.array([[c]])
    
    crit = Rotordynamics.critical_speeds(K, M)
    print(f"Critical speed: {crit[0]:.1f} Hz ({crit[0]*60:.0f} RPM)")
    
    # Unbalance response
    unbalance = np.array([0.001])  # kg*m
    omega_range = np.linspace(0, 200, 200)  # rad/s
    response = np.zeros_like(omega_range, dtype=complex)
    
    for i, omega in enumerate(omega_range):
        X = Rotordynamics.unbalance_response(M, K, C, unbalance, omega)
        response[i] = X[0]
    
    print(f"Peak response at resonance: {np.max(np.abs(response))*1000:.2f} mm")


def plot_results():
    """Generate demonstration plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Mohr's Circle
    ax = axes[0, 0]
    sigma_x, sigma_y, tau_xy = 100, 50, 30  # MPa
    avg = (sigma_x + sigma_y) / 2
    R = np.sqrt(((sigma_x - sigma_y)/2)**2 + tau_xy**2)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = avg + R * np.cos(theta)
    circle_y = R * np.sin(theta)
    ax.plot(circle_x, circle_y, 'b-', linewidth=2)
    ax.plot([sigma_x, sigma_y], [-tau_xy, tau_xy], 'ro', markersize=8)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Normal Stress (MPa)')
    ax.set_ylabel('Shear Stress (MPa)')
    ax.set_title("Mohr's Circle")
    ax.grid(True, alpha=0.3)
    
    # 2. S-N Curve
    ax = axes[0, 1]
    Se, Su = 200, 400  # MPa
    b = -0.1
    N = np.logspace(3, 7, 100)
    Sa = Se * (2*N)**b
    ax.loglog(N, Sa, 'b-', linewidth=2)
    ax.axhline(y=Se, color='r', linestyle='--', label=f'Endurance limit = {Se} MPa')
    ax.set_xlabel('Cycles to Failure (N)')
    ax.set_ylabel('Alternating Stress (MPa)')
    ax.set_title('S-N Curve (Basquin)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # 3. Mean Stress Corrections
    ax = axes[0, 2]
    Sm = np.linspace(0, 400, 100)
    Sa_applied = 200
    Sa_goodman = [FatigueFracture.goodman_correction(Sa_applied, s, Se, Su) for s in Sm]
    Sa_gerber = [FatigueFracture.gerber_correction(Sa_applied, s, Se, Su) for s in Sm]
    Sa_soderberg = [FatigueFracture.soderberg_correction(Sa_applied, s, Se, 250) for s in Sm]
    ax.plot(Sm, Sa_goodman, 'b-', label='Goodman', linewidth=2)
    ax.plot(Sm, Sa_gerber, 'r-', label='Gerber', linewidth=2)
    ax.plot(Sm, Sa_soderberg, 'g-', label='Soderberg', linewidth=2)
    ax.axhline(y=Se, color='k', linestyle=':', label='Se')
    ax.set_xlabel('Mean Stress (MPa)')
    ax.set_ylabel('Equivalent Alternating Stress (MPa)')
    ax.set_title('Mean Stress Corrections')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 500)
    
    # 4. Crack Growth (Paris Law)
    ax = axes[1, 0]
    C, m = 1e-12, 3.0
    delta_K = np.linspace(10, 50, 100)  # MPa√m
    da_dN = C * (delta_K * 1e6)**m  # m/cycle
    ax.loglog(delta_K, da_dN * 1e6, 'b-', linewidth=2)  # mm/cycle
    ax.set_xlabel('ΔK (MPa√m)')
    ax.set_ylabel('da/dN (mm/cycle)')
    ax.set_title("Paris' Law Crack Growth")
    ax.grid(True, which='both', alpha=0.3)
    
    # 5. Beam Mode Shapes
    ax = axes[1, 1]
    L = 5.0
    n_elem = 20
    E, I, rho, A = 200e9, 1e-6, 7850, 0.01
    
    elements = []
    for i in range(n_elem):
        ke = FEA_1D.beam_stiffness(E, I, L/n_elem)
        me = FEA_1D.beam_mass(rho, A, L/n_elem)
        elements.append({'connectivity': [i, i+1], 'ke': ke, 'me': me})
    
    K, M = FEA_1D.assemble_global(n_elem+1, elements, 2)
    fixed = [0, 1]
    free = [i for i in range(2*(n_elem+1)) if i not in fixed]
    K_free = K[np.ix_(free, free)]
    M_free = M[np.ix_(free, free)]
    
    lambdas, modes = VibrationAnalysis.modal_analysis(K_free, M_free, 3)
    x_coords = np.linspace(0, L, n_elem+1)
    
    for i in range(3):
        mode_full = np.zeros(2*(n_elem+1))
        mode_full[free] = modes[:, i]
        v_disp = mode_full[::2]  # Vertical DOFs
        # Normalize
        v_disp = v_disp / np.max(np.abs(v_disp))
        ax.plot(x_coords, v_disp, label=f'Mode {i+1} ({np.sqrt(lambdas[i])/(2*np.pi):.1f} Hz)')
    
    ax.set_xlabel('Position along beam (m)')
    ax.set_ylabel('Normalized displacement')
    ax.set_title('Cantilever Beam Mode Shapes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    # 6. SDOF Frequency Response
    ax = axes[1, 2]
    m, k = 10.0, 10000.0
    zeta_vals = [0.01, 0.05, 0.1, 0.2]
    omega = np.linspace(0, 2*np.sqrt(k/m), 200)
    
    for zeta in zeta_vals:
        c = 2 * zeta * np.sqrt(k * m)
        H = 1 / np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
        ax.plot(omega / np.sqrt(k/m), H * k, label=f'ζ = {zeta}')
    
    ax.set_xlabel('Frequency ratio ω/ω_n')
    ax.set_ylabel('Dynamic Magnification Factor')
    ax.set_title('SDOF Frequency Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mechanical_analysis_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'mechanical_analysis_results.png'")
    plt.close()


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    
    print("MECHANICAL ENGINEERING COMPUTATIONAL TOOLKIT")
    print("=" * 60)
    
    demo_stress_strain()
    demo_fea_1d()
    demo_vibrations()
    demo_fatigue_fracture()
    demo_optimization()
    demo_failure_theories()
    demo_rotordynamics()
    
    # Generate plots
    try:
        plot_results()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)