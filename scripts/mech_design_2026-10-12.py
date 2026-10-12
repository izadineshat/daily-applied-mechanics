import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

class TrussElement:
    """1D Truss Element for Stiffness Matrix Calculations"""
    def __init__(self, E, A, L, nodes):
        self.E = E  # Young's modulus (Pa)
        self.A = A  # Cross-sectional area (m²)
        self.L = L  # Length (m)
        self.nodes = nodes  # Global node numbers [i, j]

    def stiffness_matrix(self):
        """Calculate element stiffness matrix in local coordinates"""
        k = (self.E * self.A / self.L) * np.array([[1, -1], [-1, 1]])
        return k

class TrussStructure:
    """Assemble truss structure from elements"""
    def __init__(self, num_nodes, elements):
        self.num_nodes = num_nodes
        self.elements = elements
        self.ndof = 2 * num_nodes

    def assemble_global_k(self, node_dof=None):
        """Assemble global stiffness matrix"""
        K_global = np.zeros((self.ndof, self.ndof))
        for elem in self.elements:
            k_local = elem.stiffness_matrix()
            if node_dof is not None:
                nodes = node_dof[elem.nodes]
            else:
                nodes = [2*n-2, 2*n-1 for n in elem.nodes]
            for i in range(2):
                for j in range(2):
                    K_global[nodes[i], nodes[j]] += k_local[i, j]
        return K_global

def stress_transformation(sigma_x, sigma_y, tau_xy):
    """Calculate principal stresses and max shear stress from stress components"""
    sigma_avg = (sigma_x + sigma_y) / 2
    R = np.sqrt((sigma_x - sigma_avg)**2 + tau_xy**2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R
    tau_max = R
    return sigma_1, sigma_2, tau_max

def mohrs_circle(sigma_x, sigma_y, tau_xy, num_points=100):
    """Generate Mohr's circle data for stress state visualization"""
    sigma_avg = (sigma_x + sigma_y) / 2
    R = np.sqrt((sigma_x - sigma_avg)**2 + tau_xy**2)
    theta = np.linspace(0, np.pi, num_points)
    sigma = sigma_avg + R * np.cos(theta)
    tau = R * np.sin(theta)
    return sigma, tau, sigma_avg, R, sigma_1, sigma_2

def von_mises_stress(sigma_x, sigma_y, sigma_z=0, tau_xy=0, tau_yz=0, tau_zx=0):
    """Calculate von Mises equivalent stress"""
    term1 = sigma_x**2 + sigma_y**2 + sigma_z**2
    term2 = -(sigma_x*sigma_y + sigma_y*sigma_z + sigma_z*sigma_x)
    term3 = 3*(tau_xy**2 + tau_yz**2 + tau_zx**2)
    return np.sqrt(term1 + term2 + term3)

def tresca_criterion(sigma_1, sigma_2, sigma_3=0):
    """Calculate Tresca equivalent stress"""
    sigmas = sorted([sigma_1, sigma_2, sigma_3])
    return (sigmas[2] - sigmas[0])

class SDOFVibration:
    """Single Degree of Freedom Vibration Analysis"""
    def __init__(self, m, c, k, F0, omega):
        self.m = m  # Mass (kg)
        self.c = c  # Damping coefficient (N·s/m)
        self.k = k  # Stiffness (N/m)
        self.F0 = F0  # Force amplitude (N)
        self.omega = omega  # Forcing frequency (rad/s)
        
    def natural_frequency(self):
        """Calculate undamped natural frequency and ratio"""
        omega_n = np.sqrt(self.k / self.m)
        zeta = self.c / (2 * np.sqrt(self.k * self.m))
        return omega_n, zeta

    def steady_state_amplitude(self):
        """Calculate steady-state amplitude of vibration"""
        omega_n, zeta = self.natural_frequency()
        omega_r = self.omega
        denom = np.sqrt((1 - (omega_r/omega_n)**2)**2 + (2*zeta*omega_r/omega_n)**2)
        X0 = (self.F0 / self.k) / denom
        return X0

    def phase_angle(self):
        """Calculate phase angle of response"""
        _, zeta = self.natural_frequency()
        phi = np.arctan2(2*zeta*(self.omega/np.sqrt(self.k/self.m)), 
                        1 - (self.omega / np.sqrt(self.k/self.m))**2)
        return phi

class MDOFVibration:
    """Multiple Degree of Freedom Vibration Analysis"""
    def __init__(self, M, K, C=None, F_vec=None):
        self.M = M  # Mass matrix
        self.K = K  # Stiffness matrix
        self.C = C  # Damping matrix (optional)
        self.F_vec = F_vec  # Force vector
        
    def modal_analysis(self):
        """Perform modal analysis and return frequencies and modes"""
        omega2, Phi = linalg.eig(self.K, self.M)
        omega = np.sqrt(omega2.real)
        idx = np.argsort(omega)
        return omega[idx], Phi[:,idx]

    def state_space_response(self, x0, v0, t_span=[0,10], n_points=100):
        """Calculate state-space response"""
        n = self.M.shape[0]
        if self.C is None:
            self.C = np.zeros_like(self.K)
            
        A = np.block([[np.zeros((n,n)), np.eye(n)],
                      [-linalg.inv(self.M)@self.K, -linalg.inv(self.M)@self.C]])
        
        def dxdt(t, x):
            return A @ x
            
        from scipy.integrate import solve_ivp
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(dxdt, t_span, np.concatenate([x0, v0]), t_eval=t_eval)
        return sol.t, sol.y[:n,:], sol.y[n:,:]

def sn_curve(N, Sf=200e6, b=-0.1, C=80e6, n=0.1):
    """Calculate stress amplitude using Basquin equation"""
    # High-cycle fatigue parameters
    if N >= C/Sf:  # Below endurance limit
        return Sf * (C/(Sf * N))**(1/n) if N < 1e4 else Sf/2
    return (C)**0.1 * (N)**b

def goodman_line(Sa, Sm, Su_end=800e6):
    """Calculate safety factor using Goodman approach"""
    return Su_end / (Sa + Sm * (Sa / Su_end))

def gerber_line(Sa, Sm, Su_end=800e6):
    """Calculate safety factor using Gerber parabolic approach"""
    return 1 / (Sa/Su_end + (Sm/Su_end)**2)

def paris_law(delta_a, C=10e-11, m=3, N_ref=1e3, a_ref=1e-5):
    """Calculate crack growth rate using Paris Law"""
    delta_sigma = Sn, 800e6, 200e6)[1] - 0  # Example value, replace with appropriate
    return C * (delta_a**m)

def beam_optimization(weight_per_unit_length_func, optimize='gradient', 
                      constraints=None, n=1):
    """Structural optimization of beam cross-section for weight minimization"""
    from scipy.optimize import minimize
    
    def objective(x):
        # Example beam parameters, x contains cross-sectional dimensions
        # This is a simplified example
       L, E, M_max = 10, 200e9, 5000
        section_area = np.pi * x[0]**2 /4  # Solid circular
        I = np.pi * x[0]**4 /64
        stress = M_max * x[0]/(2*I)
        if stress > 250e6:
            return 1e10
        return L * weight_per_unit_length_func(x, E)
    
    x0 = [0.05]  # Initial guess: diameter 50 mm
    if optimize == 'gradient':
        res = minimize(objective, x0, method='L-BFGS-B')
    return res

class RotordynamicsSystem:
    """Simplified Rotordynamics Analysis"""
    def __init__(self, m, k, ecc=0.001):
        self.m = m  # Mass (kg)
        self.k = k  # Stiffness (N/m)
        self.ecc = ecc  # Eccentricity (m)

    def critical_speed(self):
        """Calculate critical speed (rad/s)"""
        return np.sqrt(self.k/self.m)
    
    def whirl_amplitude(self):
        """Calculate vibration amplitude due to unbalance"""
        omega_n = self.critical_speed()
        # Assume slight damping
        zeta = 0.01
        denom = np.sqrt((1 - (omega_n/omega_n)**2)**2 + (2*zeta*omega_n/omega_n)**2)
        return self.ecc / denom

def contact_forces(displacement, k_spring=1e9, force_threshold=50e3):
    """Calculate contact forces and pressure distribution"""
    F_contact = k_spring * displacement
    F_contact = np.where(F_contact > force_threshold, force_threshold, F_contact)
    pressure = F_contact / displacement  # Simplified
    return F_contact, pressure

if __name__ == '__main__':
    # Example Usage
    
    # 1D Truss FEA
    # Create truss elements
    elem1 = TrussElement(E=200e9, A=2e-4, L=1.0, nodes=[1, 2])
    elem2 = TrussElement(E=200e9, A=1.5e-4, L=0.8, nodes=[2, 3])
    structure = TrussStructure(num_nodes=3, elements=[elem1, elem2])
    K_global = structure.assemble_global_k()
    print("Global Stiffness Matrix:\n", K_global)
    
    # Stress Transformation Examples
    sigma_x, sigma_y, tau_xy = 100e6, 50e6, 30e6
    sigma1, sigma2, tau_max = stress_transformation(sigma_x, sigma_y, tau_xy)
    print(f"Principal Stresses: {sigma1:.2e} Pa, {sigma2:.2e} Pa")
    
    # Mohr's Circle Plot
    sigma, tau, xc, R, sig1, sig2 = mohrs_circle(sigma_x, sigma_y, tau_xy)
    plt.figure(figsize=(8,6))
    plt.plot(sigma, tau, 'b-', label='Mohr\'s Circle')
    plt.axvline(xc, color='r', linestyle='--', label='Center')
    plt.xlabel('Stress (Pa)')
    plt.ylabel('Shear Stress (Pa)')
    plt.grid(True)
    plt.legend()
    plt.title('Mohr\'s Circle')
    plt.show()
    
    # Von Mises and Tresca
    Sv = von_mises_stress(sigma_x, sigma_y, sigma_z=70e6, tau_xy=30e6)
    St = tresca_criterion(sigma1, sigma2)
    print(f"Von Mises Stress: {Sv/1e6:.2f} MPa\nTresca Equivalent: {St/1e6:.2f} MPa")
    
    # Mechanical Vibrations
    vib = SDOFVibration(m=100, c=5, k=20000, F0=500, omega=20)
    omega_n, zeta = vib.natural_frequency()
    print(f"Natural Frequency: {omega_n:.2f} rad/s\nDamping Ratio: {zeta:.4f}")
    print(f"Amplitude at omega={vib.omega}: {vib.steady_state_amplitude():.3e} m\n")
    
    # MDOF Vibration
    M = np.diag([100, 50])
    K = np.array([[30000, -15000], [-15000, 20000]])
    mdof_vib = MDOFVibration(M, K)
    freqs, modes = mdof_vib.modal_analysis()
    print("Natural Frequencies (Hz) and Modes:")
    print(f"  {freqs/(2*np.pi)}\n  {modes}")
    
    # Fatigue Analysis
    N_test = 1e5
    Sa = sn_curve(N_test)
    print(f"Stress Amplitude at N={N_test:.0e}: {Sa/1e6:.2f} MPa")
    
    # Goodman and Gerber
    Sm_test = 100e6
    FS_goodman = goodman_line(Sa, Sm_test)
    FS_gerber = gerber_line(Sa, Sm_test)
    print(f"Safety Factors: Goodman={FS_goodman:.2f}, Gerber={FS_gerber:.2f}")
    
    # Structural Optimization
    def density_based_weight(x, E):
        density = 7800  # kg/m³
        return density * (np.pi*x[0]**2/4) * L
    res = beam_optimization(weight_per_unit_length_func=weight=lambda x,E: x[0]*9e3,
                           optimize='gradient')
    print(f"Optimal Diameter: {res.x[0]*1e3:.1f} mm")

    # Additional Notes:
    # 1. All parameters should be verified against actual material properties.
    # 2. For complex FEA, consider using full node coordinate systems.
    # 3. Safety factors in fatigue use different criteria based on material properties.
    # 4. MATLAB-style plotting can be enabled where necessary.