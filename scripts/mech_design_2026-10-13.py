import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Material:
    """Material property container."""
    E: float  # Young's Modulus [Pa]
    nu: float # Poisson's ratio
    rho: float # Density [kg/m^3]
    yield_strength: float # [Pa]

@dataclass
class Section:
    """Cross-section geometric properties."""
    A: float  # Area [m^2]
    I: float  # Second moment of area [m^4]
    J: float  # Polar moment of inertia [m^4]

class StressAnalyzer:
    """Tools for Stress/Strain transformations and Failure Theories."""
    
    @staticmethod
    def mohr_circle(sigma_x: float, sigma_y: float, tau_xy: float) -> dict:
        """Computes principal stresses and max shear stress."""
        center = (sigma_x + sigma_y) / 2.0
        radius = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
        sigma_1 = center + radius
        sigma_2 = center - radius
        tau_max = radius
        return {
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "center": center,
            "radius": radius,
            "tau_max": tau_max
        }

    @staticmethod
    def von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Computes Von Mises equivalent stress for plane stress."""
        return np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)

    @staticmethod
    def tresca(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """Computes Tresca (Max Shear) equivalent stress."""
        res = StressAnalyzer.mohr_circle(sigma_x, sigma_y, tau_xy)
        return res["tau_max"] * 2.0

class VibrationEngine:
    """Handles Modal Analysis and MDOF dynamics."""

    @staticmethod
    def solve_modal(M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the generalized eigenvalue problem: K * phi = omega^2 * M * phi
        Returns:
            eigenvalues: squared natural frequencies (rad/s)^2
            eigenvectors: mode shapes
        """
        eigenvalues, eigenvectors = eigh(K, M)
        return eigenvalues, eigenvectors

    @staticmethod
    def state_space_response(A: np.ndarray, B: np.ndarray, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Simulates linear time-invariant system response.
        Standard form: dx/dt = Ax + Bu
        """
        from scipy.integrate import odeint
        def system_dynamics(x, t_val):
            return A @ x + B @ np.array([1.0]) # Assuming unit step/impulse logic if B is scaled
            
        # Simplified for demo: using odeint to solve the ODE
        # In production, use scipy.signal.lsim for better performance
        from scipy.signal import lsim
        # Constructing a transfer function approach for lsim
        # We need the system to be represented as A, B, C, D
        # For a generic state space, we define a dummy C and D
        C = np.eye(A.shape[0], 1) 
        D = np.zeros((1, 1))
        t, y, x = lsim((A, B, C, D), U=np.ones_like(t), T=t, X0=x0)
        return y

class FatigueModel:
    """Predicts fatigue life and safety factors."""

    @staticmethod
    def goodman_criterion(sigma_a: float, sigma_m: float, Se: float, Sut: float) -> float:
        """
        Calculates safety factor using Goodman relation.
        sigma_a: Alternating stress
        sigma_m: Mean stress
        Se: Endurance limit
        Sut: Ultimate tensile strength
        """
        return 1.0 / ( (sigma_a / Se) + (sigma_m / Sut) )

    @staticmethod
    def paris_law_crack_growth(delta_k: float, C: float, m: float, delta_N: float) -> float:
        """
        Calculates crack increment per cycle.
        da/dN = C * (delta_K)^m
        """
        return C * (delta_k**m) * delta_N

class StructuralOptimizer:
    """Basic optimization routines for beam design."""

    @staticmethod
    def optimize_beam_weight(length: float, load: float, E: float, I_func, A_func):
        """
        A simple gradient-descent-like approach to minimize weight 
        subject to a deflection constraint (delta < delta_max).
        Note: This is a skeleton for complex optimization.
        """
        from scipy.optimize import minimize

        def objective(dim):
            # dimension is a proxy for geometric scaling
            # e.g., height of a rectangular beam
            return A_func(dim) * length # minimize volume/weight

        def constraint(dim):
            # deflection = (P * L^3) / (48 * E * I)
            deflection = (load * length**3) / (48 * E * I_func(dim))
            return 0.01 - deflection # Constraint: deflection < 10mm

        cons = {'type': 'ineq', 'fun': constraint}
        res = minimize(objective, x0=[0.1], constraints=cons, bounds=[(0.001, 1.0)])
        return res

def run_demonstration():
    print("--- Mechanical Engineering Computational Suite ---\n")

    # 1. Stress Transformation Demo
    print("[1] Stress Transformation & Failure Theory")
    s_x, s_y, t_xy = 50e6, -10e6, 20e6 # Pa
    mohr = StressAnalyzer.mohr_circle(s_x, s_y, t_xy)
    vm = StressAnalyzer.von_mises(s_x, s_y, t_xy)
    print(f"    Principal Stresses: {mohr['sigma_1']/1e6:.2f} MPa, {mohr['sigma_2']/1e6:.2f} MPa")
    print(f"    Von Mises Stress: {vm/1e6:.2f} MPa")
    print(f"    Tresca Stress: {StressAnalyzer.tresca(s_x, s_y, t_xy)/1e6:.2f} MPa\n")

    # 2. Modal Analysis Demo (2-DOF Mass-Spring System)
    print("[2] MDOF Modal Analysis (2-DOF System)")
    m1, m2 = 2.0, 2.0  # kg
    k1, k2 = 1000.0, 1000.0  # N/m
    M = np.array([[m1, 0], [0, m2]])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    
    eigenvalues, eigenvectors = VibrationEngine.solve_modal(M, K)
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    print(f"    Natural Frequencies: {frequencies} Hz")
    print(f"    First Mode Shape: {eigenvectors[:, 0]}\n")

    # 3. Fatigue Analysis Demo
    print("[3] Fatigue Analysis (Goodman)")
    se = 250e6  # Endurance limit
    sut = 500e6 # UTS
    sa = 100e6  # Alternating stress
    sm = 150e6  # Mean stress
    sf = FatigueModel.goodman_criterion(sa, sm, se, sut)
    print(f"    Goodman Safety Factor: {sf:.3f}\n")

    # 4. Visualization of Stress State
    print("[4] Generating Visualizations...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mohr's Circle Plot
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = mohr['center'] + mohr['radius'] * np.cos(theta)
    circle_y = mohr['radius'] * np.sin(theta)
    
    ax[0].plot(circle_x, circle_y, 'b-', label="Mohr's Circle")
    ax[0].axhline(0, color='black', lw=1)
    ax[0].axvline(0, color='black', lw=1)
    ax[0].scatter([sigma_x, sigma_y], [0, 0], color='red', label="States")
    ax[0].set_title("Mohr's Circle (Stress)")
    ax[0].set_xlabel("$\sigma$ (Pa)")
    ax[0].set_ylabel("$\tau$ (Pa)")
    ax[0].legend()
    ax[0].grid(True)

    # Modal Frequencies Plot
    ax[1].stem(range(1, len(frequencies)+1), frequencies)
    ax[1].set_title("Natural Frequencies")
    ax[1].set_xlabel("Mode Number")
    ax[1].set_ylabel("Freq (Hz)")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demonstration()