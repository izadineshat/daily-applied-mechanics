from __future__ import annotations

import numpy as np
from scipy import linalg, optimize, integrate, signal
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Run headless by default so the script is usable without a display
plt.switch_backend("Agg")


# =============================================================================
# 1. FINITE ELEMENT ANALYSIS (TRUSS / BEAM / PLATE STIFFNESS)
# =============================================================================
class TrussFEA:
    """1D/2D truss and 2D Euler-Bernoulli beam finite element utilities."""

    @staticmethod
    def local_stiffness_1d(E: float, A: float, L: float) -> np.ndarray:
        """Return the 2x2 local stiffness matrix of a 1D bar element."""
        k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        return k

    @staticmethod
    def solve_truss(
        nodes: Sequence[Sequence[float]],
        elements: Sequence[Tuple[int, int]],
        E: float | Sequence[float],
        A: float | Sequence[float],
        loads: Dict[int, float],
        fixed_dofs: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Assemble and solve a linear elastic 2D truss.

        Parameters
        ----------
        nodes : (n_nodes, 2) array-like of nodal coordinates [x, y]
        elements : list of (node_i, node_j), zero-based indices
        E : float or list of Young's moduli per element
        A : float or list of cross-sectional areas per element
        loads : dict {global_dof: force_value}
        fixed_dofs : list of constrained global DOFs

        Returns
        -------
        u : (ndof,) displacement vector
        reactions : (ndof,) reaction vector (nonzero at fixed DOFs)
        stresses : list of axial stresses per element
        """
        nodes = np.asarray(nodes, dtype=float)
        n_nodes = nodes.shape[0]
        ndof = 2 * n_nodes
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        for dof, val in loads.items():
            F[dof] = val

        E_list = E if np.ndim(E) else [E] * len(elements)
        A_list = A if np.ndim(A) else [A] * len(elements)

        for el, (i, j) in enumerate(elements):
            Ee = E_list[el]
            Ae = A_list[el]
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            dx = x2 - x1
            dy = y2 - y1
            L = float(np.hypot(dx, dy))
            c = dx / L
            s = dy / L

            k_local = Ee * Ae / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
            T = np.array([[c, s, 0.0, 0.0], [0.0, 0.0, c, s]])
            k_el = T.T @ k_local @ T

            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += k_el[a, b]

        fixed = np.asarray(sorted(set(fixed_dofs)), dtype=int)
        free = np.array([d for d in range(ndof) if d not in fixed], dtype=int)

        u_free = linalg.solve(K[np.ix_(free, free)], F[free], assume_a="sym")
        u = np.zeros(ndof)
        u[free] = u_free
        reactions = K @ u - F

        stresses: List[float] = []
        for el, (i, j) in enumerate(elements):
            Ee = E_list[el]
            ue = u[[2 * i, 2 * i + 1, 2 * j, 2 * j + 1]]
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            dx = x2 - x1
            dy = y2 - y1
            L = float(np.hypot(dx, dy))
            c = dx / L
            s = dy / L
            u_ax1 = c * ue[0] + s * ue[1]
            u_ax2 = c * ue[2] + s * ue[3]
            strain = (u_ax2 - u_ax1) / L
            stresses.append(Ee * strain)

        return u, reactions, stresses

    @staticmethod
    def beam_stiffness_2d(
        E: float, I: float, L: float, A: Optional[float] = None
    ) -> np.ndarray:
        """
        Return the 6x6 stiffness matrix of a 2D Euler-Bernoulli beam element.

        DOF ordering per element: [u1, v1, theta1, u2, v2, theta2]
        where u = axial, v = transverse, theta = rotation (z-axis).
        """
        if A is None:
            A = 1.0
        k = np.zeros((6, 6))

        # Axial terms
        k[0, 0] = k[3, 3] = E * A / L
        k[0, 3] = k[3, 0] = -E * A / L

        # Bending terms
        k[1, 1] = k[4, 4] = 12.0 * E * I / L**3
        k[1, 2] = k[2, 1] = 6.0 * E * I / L**2
        k[1, 4] = k[4, 1] = -12.0 * E * I / L**3
        k[1, 5] = k[5, 1] = 6.0 * E * I / L**2
        k[2, 2] = k[5, 5] = 4.0 * E * I / L
        k[2, 4] = k[4, 2] = -6.0 * E * I / L**2
        k[2, 5] = k[5, 2] = 2.0 * E * I / L
        k[4, 5] = k[5, 4] = -6.0 * E * I / L**2

        # Symmetrize
        k = np.triu(k) + np.triu(k, 1).T
        return k

    @staticmethod
    def cst_stiffness_plane_stress(
        nodes_xy: Sequence[Sequence[float]],
        E: float,
        nu: float,
        t: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Constant Strain Triangle (CST) element stiffness for plane stress.

        Parameters
        ----------
        nodes_xy : [(x1,y1), (x2,y2), (x3,y3)]
        E : Young's modulus
        nu : Poisson's ratio
        t : thickness

        Returns
        -------
        k_e : (6, 6) element stiffness matrix
        area : element area
        """
        (x1, y1), (x2, y2), (x3, y3) = nodes_xy
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])

        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i] = c[i]
            B[2, 2 * i + 1] = b[i]
        B /= (2.0 * area)

        D = E / (1.0 - nu**2) * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )
        k_e = t * area * (B.T @ D @ B)
        return k_e, area


# =============================================================================
# 2. STRESS & STRAIN TRANSFORMATIONS (MOHR, VON MISES, TRESCA)
# =============================================================================
class StressStrain:
    """Stress transformations, failure theories, and Mohr's circle utilities."""

    @staticmethod
    def transform_2d(
        sx: float, sy: float, txy: float, theta_deg: float
    ) -> Tuple[float, float, float]:
        """Transform 2D plane stress by angle theta (degrees, CCW)."""
        th = np.deg2rad(theta_deg)
        c = np.cos(2.0 * th)
        sn = np.sin(2.0 * th)
        sxp = (sx + sy) / 2.0 + (sx - sy) / 2.0 * c + txy * sn
        syp = (sx + sy) / 2.0 - (sx - sy) / 2.0 * c - txy * sn
        txyp = -(sx - sy) / 2.0 * sn + txy * c
        return sxp, syp, txyp

    @staticmethod
    def principal_stresses_2d(
        sx: float, sy: float, txy: float
    ) -> Tuple[float, float, float, float]:
        """
        Return in-plane principal stresses s1, s2, max in-plane shear, and
        principal angle (degrees) for the first principal direction.
        """
        avg = (sx + sy) / 2.0
        R = np.sqrt(((sx - sy) / 2.0) ** 2 + txy**2)
        s1 = avg + R
        s2 = avg - R
        tau_max = R
        theta_p = 0.5 * np.rad2deg(np.arctan2(2.0 * txy, sx - sy))
        return s1, s2, tau_max, theta_p

    @staticmethod
    def von_mises(
        sx: float,
        sy: float,
        sz: float = 0.0,
        txy: float = 0.0,
        tyz: float = 0.0,
        txz: float = 0.0,
    ) -> float:
        """General 3D von Mises equivalent stress."""
        return float(
            np.sqrt(
                0.5
                * (
                    (sx - sy) ** 2
                    + (sy - sz) ** 2
                    + (sz - sx) ** 2
                    + 6.0 * (txy**2 + tyz**2 + txz**2)
                )
            )
        )

    @staticmethod
    def von_mises_2d(sx: float, sy: float, txy: float, sz: float = 0.0) -> float:
        """Plane-stress von Mises stress."""
        return StressStrain.von_mises(sx, sy, sz, txy, 0.0, 0.0)

    @staticmethod
    def tresca_max_shear(
        sx: float,
        sy: float,
        sz: float = 0.0,
        txy: float = 0.0,
        tyz: float = 0.0,
        txz: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Maximum shear stress (Tresca) from the full stress tensor.

        Returns (tau_max, s_max, s_min) where s_max/min are algebraic extrema
        of the three principal stresses.
        """
        T = np.array([[sx, txy, txz], [txy, sy, tyz], [txz, tyz, sz]])
        principal = linalg.eigvalsh(T)  # ascending
        s_min, _, s_max = principal
        tau_max = 0.5 * (s_max - s_min)
        return tau_max, s_max, s_min

    @staticmethod
    def mohr_circle_2d(
        sx: float, sy: float, txy: float
    ) -> Dict[str, float]:
        """Mohr's circle parameters for 2D plane stress."""
        center = (sx + sy) / 2.0
        radius = np.sqrt(((sx - sy) / 2.0) ** 2 + txy**2)
        s1, s2, tau_max, theta_p = StressStrain.principal_stresses_2d(sx, sy, txy)
        return {
            "center": center,
            "radius": radius,
            "s1": s1,
            "s2": s2,
            "tau_max": tau_max,
            "theta_p_deg": theta_p,
        }

    @staticmethod
    def failure_check(
        sx: float, sy: float, txy: float, syield: float, sut: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate common static failure theories under plane stress.

        Returns factors of safety for von Mises, Tresca, and Rankine (max
        principal). If sut is provided, a simple max-normal-stress check
        against ultimate is also reported.
        """
        vm = StressStrain.von_mises_2d(sx, sy, txy)
        tau_max, s_max, s_min = StressStrain.tresca_max_shear(sx, sy, 0.0, txy)
        tresca_eq = 2.0 * tau_max  # equivalent uniaxial stress for Tresca

        fos = {
            "von_mises_stress": vm,
            "fos_von_mises": syield / vm if vm > 0 else np.inf,
            "tresca_equiv_stress": tresca_eq,
            "fos_tresca": syield / tresca_eq if tresca_eq > 0 else np.inf,
            "max_principal": s_max,
            "fos_rankine": syield / s_max if s_max > 0 else np.inf,
        }
        if sut is not None:
            fos["fos_ultimate_rankine"] = sut / s_max if s_max > 0 else np.inf
        return fos


# =============================================================================
# 3. MECHANICAL VIBRATIONS (SDOF / MDOF / STATE-SPACE)
# =============================================================================
class Vibrations:
    """Vibration analysis: SDOF response, modal analysis, MDOF state-space."""

    @staticmethod
    def sdof_natural_frequency(k: float, m: float) -> float:
        return float(np.sqrt(k / m))

    @staticmethod
    def sdof_response(
        m: float,
        c: float,
        k: float,
        t: np.ndarray,
        x0: float,
        v0: float,
        force_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SDOF linear response via state-space simulation.

        m*xdd + c*xd + k*x = f(t)

        Returns time, displacement, velocity.
        """
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        B = np.array([[0.0], [1.0 / m]])
        C_mat = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])

        if force_func is None:
            U = np.zeros((len(t), 1))
        else:
            f = np.asarray(force_func(t), dtype=float)
            if f.ndim == 1:
                U = f.reshape(-1, 1)
            else:
                U = f

        t_out, y, _ = signal.lsim((A, B, C_mat, D), U=U, T=t, X0=np.array([x0, v0]))
        x = y[:, 0]
        xd = np.gradient(x, t_out)
        return t_out, x, xd

    @staticmethod
    def modal_analysis(
        M: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undamped modal analysis: solve K*phi = omega^2 * M*phi.

        Returns natural frequencies (rad/s) and mass-normalized mode shapes.
        """
        w2, phi = linalg.eigh(K, M)
        w2 = np.maximum(w2, 0.0)
        omega = np.sqrt(w2)
        for i in range(phi.shape[1]):
            norm = np.sqrt(phi[:, i] @ M @ phi[:, i])
            phi[:, i] /= norm
        return omega, phi

    @staticmethod
    def mdof_state_space_response(
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        t: np.ndarray,
        x0: np.ndarray,
        v0: np.ndarray,
        F_func: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linear MDOF response in state-space form.

        States: [x; xd].  Input: force vector F(t).

        Returns time, displacement history (n_t, n_dof), velocity history.
        """
        n = M.shape[0]
        Mi = linalg.inv(M)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Mi @ K
        A[n:, n:] = -Mi @ C
        B = np.zeros((2 * n, n))
        B[n:, :] = Mi
        C_mat = np.hstack([np.eye(n), np.zeros((n, n))])
        D = np.zeros((n, n))

        U = np.asarray(F_func(t), dtype=float)
        if U.ndim == 1:
            U = np.tile(U, (len(t), 1))

        X0 = np.concatenate([np.asarray(x0), np.asarray(v0)])
        t_out, y, _ = signal.lsim((A, B, C_mat, D), U=U, T=t, X0=X0)
        return t_out, y[:, :n], y[:, n:]

    @staticmethod
    def frequency_response_function(
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        omega: np.ndarray,
        input_dof: int = 0,
        output_dof: int = 0,
    ) -> np.ndarray:
        """Compute magnitude of the receptance H(omega) between two DOFs."""
        n = M.shape[0]
        H = np.zeros_like(omega, dtype=complex)
        for i, w in enumerate(omega):
            Z = -w**2 * M + 1j * w * C + K
            H[i] = linalg.inv(Z)[output_dof, input_dof]
        return np.abs(H)


# =============================================================================
# 4. FATIGUE & FRACTURE MECHANICS
# =============================================================================
class FatigueFracture:
    """S-N curves, mean-stress criteria, and linear elastic fracture mechanics."""

    @staticmethod
    def basquin_stress(
        N: float | np.ndarray, sigma_f_prime: float, b: float
    ) -> float | np.ndarray:
        """Basquin equation: sigma_a = sigma_f' * (2N)^b."""
        return sigma_f_prime * (2.0 * N) ** b

    @staticmethod
    def basquin_life(sigma_a: float, sigma_f_prime: float, b: float) -> float:
        """Invert Basquin to estimate cycles to failure N."""
        return 0.5 * (sigma_a / sigma_f_prime) ** (1.0 / b)

    @staticmethod
    def goodman_factor_of_safety(sa: float, sm: float, se: float, sut: float) -> float:
        """Goodman: sa/se + sm/sut = 1/n."""
        return 1.0 / (sa / se + sm / sut)

    @staticmethod
    def gerber_factor_of_safety(sa: float, sm: float, se: float, sut: float) -> float:
        """Gerber: sa/se + (sm/sut)^2 = 1/n."""
        return 1.0 / (sa / se + (sm / sut) ** 2)

    @staticmethod
    def sodberg_factor_of_safety(
        sa: float, sm: float, se: float, sy: float
    ) -> float:
        """Soderberg: sa/se + sm/sy = 1/n."""
        return 1.0 / (sa / se + sm / sy)

    @staticmethod
    def stress_intensity_factor(
        sigma: float, a: float, Y: float = 1.0
    ) -> float:
        """Mode-I stress intensity factor K = Y * sigma * sqrt(pi * a)."""
        return float(Y * sigma * np.sqrt(np.pi * a))

    @staticmethod
    def paris_da_dN(C: float, m: float, delta_K: float) -> float:
        """Paris' law crack growth rate da/dN = C * (delta_K)^m."""
        return float(C * delta_K**m)

    @staticmethod
    def delta_K_constant_amplitude(
        sigma: float, a: float, Y: float = 1.0
    ) -> float:
        """Delta-K for constant amplitude remote stress."""
        return FatigueFracture.stress_intensity_factor(sigma, a, Y)

    @staticmethod
    def crack_growth_cycles(
        a0: float,
        ac: float,
        C: float,
        m: float,
        delta_K_func: Callable[..., float],
        *args,
        **kwargs,
    ) -> float:
        """
        Numerically integrate Paris' law from initial crack a0 to critical ac.

        N = integral_{a0}^{ac} da / [ C * (delta_K(a))^m ]
        """
        integrand = lambda a: 1.0 / (
            C * (delta_K_func(a, *args, **kwargs)) ** m
        )
        N, _ = integrate.quad(integrand, a0, ac, epsabs=1e-12, epsrel=1e-10, limit=200)
        return float(N)


# =============================================================================
# 5. STRUCTURAL OPTIMIZATION (GA / GRADIENT)
# =============================================================================
class StructuralOptimization:
    """Weight minimization of structural members with stress constraints."""

    @staticmethod
    def beam_weight_rect(b: float, h: float, L: float, rho: float) -> float:
        return float(rho * L * b * h)

    @staticmethod
    def beam_bending_stress_rect(b: float, h: float, M_max: float) -> float:
        """Max bending stress in a rectangular section: sigma = 6M / (b h^2)."""
        return float(6.0 * M_max / (b * h**2))

    @staticmethod
    def objective_penalty(
        x: np.ndarray,
        L: float,
        rho: float,
        M_max: float,
        sigma_allow: float,
        penalty: float = 1e9,
    ) -> float:
        b, h = x
        if b <= 0.0 or h <= 0.0:
            return 1e12
        w = StructuralOptimization.beam_weight_rect(b, h, L, rho)
        sigma = StructuralOptimization.beam_bending_stress_rect(b, h, M_max)
        viol = max(0.0, sigma - sigma_allow)
        return float(w + penalty * viol**2)

    @staticmethod
    def gradient_based_optimize(
        L: float,
        rho: float,
        M_max: float,
        sigma_allow: float,
        x0: Sequence[float],
        bounds: Sequence[Tuple[float, float]],
    ) -> Tuple[np.ndarray, float]:
        """Gradient-based minimization using L-BFGS-B with bound constraints."""
        res = optimize.minimize(
            StructuralOptimization.objective_penalty,
            x0=np.array(x0, dtype=float),
            args=(L, rho, M_max, sigma_allow),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        return res.x, float(res.fun)

    @staticmethod
    def genetic_algorithm(
        func: Callable[[np.ndarray], float],
        bounds: Sequence[Tuple[float, float]],
        pop_size: int = 60,
        generations: int = 120,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Real-coded genetic algorithm with blend crossover and Gaussian mutation.

        Returns best individual, best fitness, and convergence history.
        """
        rng = np.random.default_rng(seed)
        dim = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        pop = rng.uniform(lower, upper, size=(pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])
        history: List[float] = []

        for _ in range(generations):
            # Elitism: keep best individuals
            order = np.argsort(fitness)
            elites = pop[order[:elitism]].copy()
            elite_fit = fitness[order[:elitism]].copy()

            # Tournament selection
            def select() -> int:
                idx = rng.integers(0, pop_size, size=3)
                best = idx[np.argmin(fitness[idx])]
                return best

            new_pop: List[np.ndarray] = []
            for _ in range((pop_size - elitism) // 2):
                p1, p2 = pop[select()], pop[select()]
                if rng.random() < crossover_rate:
                    alpha = 0.5
                    c1 = p1 + alpha * (p2 - p1)
                    c2 = p2 + alpha * (p1 - p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.extend([c1, c2])

            # Fill if odd
            while len(new_pop) < pop_size - elitism:
                new_pop.append(pop[select()].copy())

            new_pop = np.array(new_pop[: pop_size - elitism])

            # Mutation
            for ind in new_pop:
                if rng.random() < mutation_rate:
                    mut = rng.normal(0.0, (upper - lower) * 0.05, size=dim)
                    ind += mut
                    ind = np.clip(ind, lower, upper)

            # Recombine with elites
            pop = np.vstack([elites, new_pop])
            fitness = np.array([func(ind) for ind in pop])
            history.append(float(np.min(fitness)))

        best_idx = int(np.argmin(fitness))
        return pop[best_idx], float(fitness[best_idx]), history

    @staticmethod
    def optimize_rectangular_beam(
        L: float,
        rho: float,
        M_max: float,
        sigma_allow: float,
        b_bounds: Tuple[float, float],
        h_bounds: Tuple[float, float],
        method: str = "ga",
    ) -> Dict[str, object]:
        """
        Minimize weight of a rectangular beam subject to bending stress.

        Constraint: 6*M_max / (b*h^2) <= sigma_allow
        """
        bounds = [b_bounds, h_bounds]

        if method.lower() == "ga":
            best_x, best_f, hist = StructuralOptimization.genetic_algorithm(
                lambda x: StructuralOptimization.objective_penalty(
                    x, L, rho, M_max, sigma_allow
                ),
                bounds=bounds,
                pop_size=50,
                generations=100,
                seed=42,
            )
        else:
            best_x, best_f = StructuralOptimization.gradient_based_optimize(
                L, rho, M_max, sigma_allow, x0=[0.05, 0.1], bounds=bounds
            )
            hist = [best_f]

        b_opt, h_opt = best_x
        w_opt = StructuralOptimization.beam_weight_rect(b_opt, h_opt, L, rho)
        sigma_opt = StructuralOptimization.beam_bending_stress_rect(
            b_opt, h_opt, M_max
        )
        return {
            "method": method,
            "b": b_opt,
            "h": h_opt,
            "weight": w_opt,
            "max_stress": sigma_opt,
            "allowable": sigma_allow,
            "convergence": hist,
        }


# =============================================================================
# 6. ROTORDYNAMICS (JEFFCOTT ROTOR)
# =============================================================================
class Rotordynamics:
    """Simple Jeffcott rotor model for critical speed and unbalance response."""

    @staticmethod
    def critical_speed(k: float, m: float) -> float:
        return float(np.sqrt(k / m))

    @staticmethod
    def jeffcott_response(
        m: float, k: float, c: float, e: float, omega_range: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Steady-state unbalance response of a Jeffcott rotor.

        X(omega) = m * e * omega^2 / sqrt((k - m*omega^2)^2 + (c*omega)^2)
        """
        wn = Rotordynamics.critical_speed(k, m)
        denom = np.sqrt((k - m * omega_range**2) ** 2 + (c * omega_range) ** 2)
        X = m * e * omega_range**2 / denom
        return wn, X


# =============================================================================
# 7. CONTACT MECHANICS (HERTZIAN)
# =============================================================================
class ContactMechanics:
    """Hertzian contact stress estimates for sphere and cylinder contacts."""

    @staticmethod
    def _effective_modulus(
        E1: float, nu1: float, E2: float, nu2: float
    ) -> float:
        return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)

    @staticmethod
    def hertzian_sphere(
        F: float, R1: float, R2: float, E1: float, nu1: float, E2: float, nu2: float
    ) -> Dict[str, float]:
        """
        Hertzian contact between two spheres.

        Returns contact radius a, max pressure p0, effective modulus/radius,
        and approximate max shear stress (~0.31 p0).
        """
        Estar = ContactMechanics._effective_modulus(E1, nu1, E2, nu2)
        Rstar = 1.0 / (1.0 / R1 + 1.0 / R2)
        a = (3.0 * F * Rstar / (4.0 * Estar)) ** (1.0 / 3.0)
        p0 = (6.0 * F * Estar**2 / (np.pi**3 * Rstar**2)) ** (1.0 / 3.0)
        tau_max = 0.31 * p0
        return {
            "Estar": Estar,
            "Rstar": Rstar,
            "contact_radius": a,
            "max_pressure": p0,
            "approx_max_shear": tau_max,
        }

    @staticmethod
    def hertzian_cylinder(
        F_per_length: float,
        R1: float,
        R2: float,
        E1: float,
        nu1: float,
        E2: float,
        nu2: float,
    ) -> Dict[str, float]:
        """
        Hertzian line contact between two cylinders (per unit length).

        Returns half-width b and max pressure p0.
        """
        Estar = ContactMechanics._effective_modulus(E1, nu1, E2, nu2)
        Rstar = 1.0 / (1.0 / R1 + 1.0 / R2)
        b = np.sqrt(4.0 * F_per_length * Rstar / (np.pi * Estar))
        p0 = 2.0 * F_per_length / (np.pi * b)
        return {"Estar": Estar, "Rstar": Rstar, "half_width": b, "max_pressure": p0}


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def plot_mohr_circle(sx: float, sy: float, txy: float) -> None:
    data = StressStrain.mohr_circle_2d(sx, sy, txy)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = data["center"] + data["radius"] * np.cos(theta)
    circle_y = data["radius"] * np.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(circle_x, circle_y, "b-", label="Mohr's Circle")
    plt.plot([data["center"]], [0], "ko", label="Center")
    plt.plot([data["s1"], data["s2"]], [0, 0], "ro", label="Principal Stresses")
    plt.plot([sx, sy], [txy, -txy], "gs", label="Stress States")
    plt.xlabel("Normal Stress (MPa)")
    plt.ylabel("Shear Stress (MPa)")
    plt.title("Mohr's Circle (2D Plane Stress)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_sn_curve(sigma_f_prime: float, b: float) -> None:
    N = np.logspace(3, 7, 200)
    sigma = FatigueFracture.basquin_stress(N, sigma_f_prime, b)
    plt.figure(figsize=(6, 4))
    plt.loglog(N, sigma, "r-", linewidth=2)
    plt.xlabel("Cycles to Failure N")
    plt.ylabel("Stress Amplitude (MPa)")
    plt.title("Basquin S-N Curve")
    plt.grid(True, which="both", alpha=0.3)


def plot_goodman_gerber(sut: float, sy: float, se: float) -> None:
    sm = np.linspace(0, sut, 200)
    sa_goodman = se * (1.0 - sm / sut)
    sa_gerber = se * (1.0 - (sm / sut) ** 2)
    sa_soderberg = se * (1.0 - sm / sy)

    plt.figure(figsize=(6, 4))
    plt.plot(sm, sa_goodman, "b-", label="Goodman")
    plt.plot(sm, sa_gerber, "g--", label="Gerber")
    plt.plot(sm, sa_soderberg, "m:", label="Soderberg")
    plt.xlabel("Mean Stress (MPa)")
    plt.ylabel("Allowable Alternating Stress (MPa)")
    plt.title("Mean-Stress Failure Diagrams")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_jeffcott(wn: float, omega: np.ndarray, X: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(omega, X, "k-", linewidth=2, label="Unbalance Response")
    plt.axvline(wn, color="r", linestyle="--", label=f"Critical Speed $\\omega_n$ = {wn:.1f} rad/s")
    plt.xlabel("Excitation Frequency (rad/s)")
    plt.ylabel("Displacement Amplitude (m)")
    plt.title("Jeffcott Rotor Response")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_ga_convergence(history: List[float], title: str = "GA Convergence") -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history, "k-", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Objective (Weight)")
    plt.title(title)
    plt.grid(True, alpha=0.3)


# =============================================================================
# DEMONSTRATIONS
# =============================================================================
def demo_truss() -> None:
    print("\n--- Truss FEA Demo ---")
    nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    elements = [(0, 1), (1, 2), (2, 0)]
    E = 200e9  # Pa (steel)
    A = 1e-3   # m^2
    loads = {3: -10000.0}  # downward load at node 1, y-dof
    fixed_dofs = [0, 1, 4]  # node0 x,y fixed; node2 x fixed

    u, R, stresses = TrussFEA.solve_truss(nodes, elements, E, A, loads, fixed_dofs)
    print("Displacements (m):")
    print(np.round(u, 6))
    print("Reactions (N):")
    print(np.round(R, 3))
    print("Element axial stresses (Pa):")
    for i, s in enumerate(stresses):
        print(f"  Element {i}: {s:.3e}")


def demo_beam_stiffness() -> None:
    print("\n--- Beam Stiffness Demo ---")
    E = 200e9
    I = 1e-6
    L = 1.0
    k = TrussFEA.beam_stiffness_2d(E, I, L)
    np.set_printoptions(precision=3, suppress=True)
    print("2D Beam element stiffness matrix (6x6):")
    print(k)


def demo_stress() -> None:
    print("\n--- Stress Transformation Demo ---")
    sx, sy, txy = 100.0, 40.0, 60.0  # MPa
    sxp, syp, txyp = StressStrain.transform_2d(sx, sy, txy, 30.0)
    print(f"Transformed at 30 deg: sx'={sxp:.2f}, sy'={syp:.2f}, txy'={txyp:.2f} MPa")
    s1, s2, tau_max, theta_p = StressStrain.principal_stresses_2d(sx, sy, txy)
    print(f"Principal: s1={s1:.2f}, s2={s2:.2f}, tau_max={tau_max:.2f}, theta_p={theta_p:.2f} deg")
    print(f"von Mises: {StressStrain.von_mises_2d(sx, sy, txy):.2f} MPa")
    print(
        f"Failure FoS (sy=250 MPa): {StressStrain.failure_check(sx, sy, txy, 250.0)}"
    )
    plot_mohr_circle(sx, sy, txy)


def demo_vibrations() -> None:
    print("\n--- Vibrations Demo ---")
    # SDOF free vibration
    m, c, k = 10.0, 20.0, 10000.0
    t = np.linspace(0, 5, 1000)
    tt, x, xd = Vibrations.sdof_response(m, c, k, t, x0=0.01, v0=0.0)
    print(f"SDOF natural frequency: {Vibrations.sdof_natural_frequency(k, m):.2f} rad/s")
    plt.figure(figsize=(6, 4))
    plt.plot(tt, x, "b-", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("SDOF Free Vibration Response")
    plt.grid(True, alpha=0.3)

    # Modal analysis 2-DOF
    M = np.diag([2.0, 2.0])
    K = np.array([[2000.0, -1000.0], [-1000.0, 2000.0]])
    omega, phi = Vibrations.modal_analysis(M, K)
    print("Natural frequencies (rad/s):", np.round(omega, 3))
    print("Mode shapes:\n", np.round(phi, 3))

    # MDOF damped response with harmonic force
    C = 0.05 * K
    F_func = lambda tt: np.column_stack([100.0 * np.sin(20.0 * tt), np.zeros_like(tt)])
    t2, x2, xd2 = Vibrations.mdof_state_space_response(M, C, K, t2 := np.linspace(0, 2, 500), [0, 0], [0, 0], F_func)
    plt.figure(figsize=(6, 4))
    plt.plot(t2, x2[:, 0], "b-", label="DOF 1")
    plt.plot(t2, x2[:, 1], "r-", label="DOF 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("MDOF Damped Harmonic Response")
    plt.grid(True, alpha=0.3)
    plt.legend()


def demo_fatigue() -> None:
    print("\n--- Fatigue & Fracture Demo ---")
    # S-N curve
    sigma_f_prime, b = 1000.0, -0.1
    plot_sn_curve(sigma_f_prime, b)
    N_example = FatigueFracture.basquin_life(300.0, sigma_f_prime, b)
    print(f"Basquin life at 300 MPa: {N_example:.2f} cycles")

    # Goodman / Gerber
    sut, sy, se = 600.0, 480.0, 200.0
    sa, sm = 100.0, 150.0
    print(f"Goodman FoS: {FatigueFracture.goodman_factor_of_safety(sa, sm, se, sut):.3f}")
    print(f"Gerber FoS:  {FatigueFracture.gerber_factor_of_safety(sa, sm, se, sut):.3f}")
    print(f"Soderberg FoS: {FatigueFracture.sodberg_factor_of_safety(sa, sm, se, sy):.3f}")
    plot_goodman_gerber(sut, sy, se)

    # Paris law
    C, m_exp = 1e-12, 3.0
    sigma, Y, a0, ac = 200.0, 1.0, 0.001, 0.05
    N_crack = FatigueFracture.crack_growth_cycles(
        a0, ac, C, m_exp, FatigueFracture.delta_K_constant_amplitude, sigma, Y
    )
    print(f"Paris cycles to grow crack {a0}->{ac} m: {N_crack:.2f}")


def demo_optimization() -> None:
    print("\n--- Structural Optimization Demo ---")
    L, rho = 1.0, 7850.0
    M_max, sigma_allow = 10000.0, 250e6
    res = StructuralOptimization.optimize_rectangular_beam(
        L, rho, M_max, sigma_allow, b_bounds=(0.01, 0.2), h_bounds=(0.01, 0.4), method="ga"
    )
    print(f"Method: {res['method']}")
    print(f"Optimal b={res['b']:.4f} m, h={res['h']:.4f} m")
    print(f"Weight={res['weight']:.3f} kg, Max stress={res['max_stress']/1e6:.1f} MPa")
    plot_ga_convergence(res["convergence"], title="Beam Weight Optimization (GA)")


def demo_rotordynamics() -> None:
    print("\n--- Rotordynamics Demo ---")
    m, k, c, e = 50.0, 1e6, 500.0, 0.001
    wn = Rotordynamics.critical_speed(k, m)
    omega = np.linspace(0, 2 * wn, 400)
    wn_calc, X = Rotordynamics.jeffcott_response(m, k, c, e, omega)
    print(f"Critical speed: {wn_calc:.1f} rad/s")
    plot_jeffcott(wn_calc, omega, X)


def demo_contact() -> None:
    print("\n--- Contact Mechanics Demo ---")
    res = ContactMechanics.hertzian_sphere(
        F=1000.0, R1=0.05, R2=0.05, E1=200e9, nu1=0.3, E2=200e9, nu2=0.3
    )
    print(f"Contact radius: {res['contact_radius']*1e3:.3f} mm")
    print(f"Max pressure: {res['max_pressure']/1e6:.2f} MPa")
    print(f"Approx max shear: {res['approx_max_shear']/1e6:.2f} MPa")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    demo_truss()
    demo_beam_stiffness()
    demo_stress()
    demo_vibrations()
    demo_fatigue()
    demo_optimization()
    demo_rotordynamics()
    demo_contact()
    plt.show()