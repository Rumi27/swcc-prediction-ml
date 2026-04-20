
import numpy as np
import time
from scipy.interpolate import interp1d

class RichardsSolver1D:
    """
    1D Finite Difference Solver for Richards Equation (Mixed Form).
    
    Equation:
        d(theta)/dt = d/dz [ K(psi) * (d(psi)/dz + 1) ]
        
    Discretization:
        Implicit Backward Euler in time
        Finite Difference in space (Cell-centered or Vertex-centered)
        Linearized via Newton-Raphson
    """
    
    def __init__(self, L=200.0, nz=100, swcc_func=None, k_func=None, verbose=False):
        """
        Args:
            L: Column height [cm]
            nz: Number of nodes
            swcc_func: Callable(psi) -> theta, C (specific moisture capacity)
            k_func: Callable(psi) -> K (hydraulic conductivity)
            verbose: Print debug info
        """
        self.L = L
        self.nz = nz
        self.dz = L / (nz - 1)
        self.z = np.linspace(0, L, nz)  # z=0 is bottom, z=L is top
        
        self.swcc_func = swcc_func
        self.k_func = k_func
        self.verbose = verbose
        
        # State variables
        self.psi = None  # Suction head (negative for unsaturated) [cm]
        self.theta = None # Volumetric water content [-]
        
        # Metrics
        self.stats = {
            'total_steps': 0,
            'failed_steps': 0,
            'total_newton_iters': 0,
            'max_newton_iters': 0,
            'cpu_time': 0.0,
            'mass_balance_error': 0.0
        }

    def initialize(self, psi_init):
        """Set initial condition (psi profile)"""
        if np.isscalar(psi_init):
            self.psi = np.full(self.nz, psi_init)
        else:
            self.psi = np.array(psi_init)
            
        self.theta, _ = self.swcc_func(self.psi)
        
    def solve(self, t_max, dt_init=1e-3, dt_min=1e-8, dt_max=1.0, 
             top_bc_type='flux', top_bc_val=0.0, 
             bottom_bc_type='fixed', bottom_bc_val=0.0,
             max_newton=20, tol_newton=1e-4):
        """
        Run simulation from t=0 to t_max.
        
        Args:
            t_max: End time [hours]
            dt_init: Initial time step [hours]
            top_bc_val: Flux [cm/h] (negative for infiltration)
        """
        t = 0.0
        dt = dt_init
        
        start_time = time.time()
        
        psi_current = self.psi.copy()
        
        history = {
            'time': [0.0],
            'psi_profiles': [psi_current.copy()],
            'theta_profiles': [self.theta.copy()],
            'cumulative_infl': [0.0]
        }
        
        cumulative_infl = 0.0
        
        print(f"Starting simulation: L={self.L}cm, T={t_max}h")
        
        while t < t_max:
            # Adjust dt to land exactly on t_max
            if t + dt > t_max:
                dt = t_max - t
                
            # Newton Iteration
            psi_new = psi_current.copy()
            converged = False
            
            for iter in range(max_newton):
                # 1. Compute properties at current iteration guess
                theta_new, C_new = self.swcc_func(psi_new)
                K_new = self.k_func(psi_new)
                
                # Arithmetic mean for K at interfaces (i-1/2, i+1/2)
                # K_{i+1/2} = (K_{i+1} + K_i) / 2
                K_mid = 0.5 * (K_new[1:] + K_new[:-1]) 
                
                # 2. Build Residual R and Jacobian J
                # Discretized eqn:
                # Resid_i = (theta_i^{n+1} - theta_i^n)/dt - (Flux_{i+1/2} - Flux_{i-1/2})/dz = 0
                # Flux = -K * ( (psi_{i+1} - psi_i)/dz + 1 )
                
                R = np.zeros(self.nz)
                J_diag = np.zeros(self.nz)
                J_upper = np.zeros(self.nz-1)
                J_lower = np.zeros(self.nz-1)
                
                flux_in = 0.0
                flux_out = 0.0
                
                # Interior Nodes (1 to nz-2)
                for i in range(1, self.nz - 1):
                    # Fluxes
                    q_plus = -K_mid[i] * ((psi_new[i+1] - psi_new[i])/self.dz + 1.0)
                    q_minus = -K_mid[i-1] * ((psi_new[i] - psi_new[i-1])/self.dz + 1.0)
                    
                    # Residual
                    store_term = (theta_new[i] - self.theta[i]) / dt
                    flux_term = -(q_plus - q_minus) / self.dz
                    R[i] = store_term + flux_term
                    
                    # Jacobian
                    # dR_i / dpsi_i
                    dq_plus_dpsi_i = -K_mid[i] * (-1.0/self.dz) # Neglecting dK/dpsi for simple Picard-like J
                    dq_minus_dpsi_i = -K_mid[i-1] * (1.0/self.dz)
                    
                    J_diag[i] = C_new[i]/dt - (-(dq_plus_dpsi_i - dq_minus_dpsi_i)/self.dz)
                    
                    # dR_i / dpsi_{i+1}
                    dq_plus_dpsi_plus = -K_mid[i] * (1.0/self.dz)
                    J_upper[i] = -(-dq_plus_dpsi_plus)/self.dz
                    
                    # dR_i / dpsi_{i-1}
                    dq_minus_dpsi_minus = -K_mid[i-1] * (-1.0/self.dz)
                    J_lower[i-1] = -(-(-dq_minus_dpsi_minus))/self.dz
                
                # --- Boundary Conditions ---
                
                # Top Node (i = nz-1)
                if top_bc_type == 'flux':
                    # q_top is prescribed (e.g. rain)
                    # Flux enters at top interface imaginary boundary
                    # But simpler: use mass balance on top 1/2 cell
                    
                    # Flux at i-1/2
                    q_minus = -K_mid[-1] * ((psi_new[-1] - psi_new[-2])/self.dz + 1.0)
                    q_top = top_bc_val # Inward is negative in this sign convention if z is up? 
                                     # Standard: q = -K(dpsi/dz + 1). 
                                     # If rain (down), q_top < 0. 
                                     # Let's say top_bc_val is negative for infiltration.
                    
                    flux_in = -q_top # Net flow into domain
                    
                    # Residual (on half cell dz/2)
                    store_term = (theta_new[-1] - self.theta[-1]) / dt
                    flux_term = -(q_top - q_minus) / (0.5 * self.dz)
                    R[-1] = store_term + flux_term
                    
                    # Jacobian
                    dq_minus_dpsi_i = -K_mid[-1] * (1.0/self.dz)
                    J_diag[-1] = C_new[-1]/dt - (-(0 - dq_minus_dpsi_i)/(0.5*self.dz))
                    
                    dq_minus_dpsi_minus = -K_mid[-1] * (-1.0/self.dz)
                    J_lower[-1] = -(-(-(-dq_minus_dpsi_minus))/(0.5*self.dz))
                    
                # Bottom Node (i = 0)
                if bottom_bc_type == 'fixed':
                    # psi fixed
                    R[0] = psi_new[0] - bottom_bc_val
                    J_diag[0] = 1.0
                    J_upper[0] = 0.0
                    
                    # Calc flux for mass balance
                    q_plus = -K_mid[0] * ((psi_new[1] - psi_new[0])/self.dz + 1.0)
                    flux_out = -q_plus # Out of domain
                    
                elif bottom_bc_type == 'free_drain':
                    # Gradient dpsi/dz = 0 => q = -K
                    q_bot = -K_new[0] 
                    q_plus = -K_mid[0] * ((psi_new[1] - psi_new[0])/self.dz + 1.0)
                    
                    flux_out = -q_bot
                    
                    store_term = (theta_new[0] - self.theta[0]) / dt
                    flux_term = -(q_plus - q_bot) / (0.5 * self.dz)
                    R[0] = store_term + flux_term
                    
                    # Jacobian terms (simplified)
                    J_diag[0] = C_new[0]/dt + K_mid[0]/(0.5*self.dz*self.dz)
                    J_upper[0] = -K_mid[0]/(0.5*self.dz*self.dz)
                
                # 3. Solve Linear System
                # Tri-diagonal solver
                try:
                    # Construct sparse matrix or just use numpy solve for dense (N=100 is small)
                    # For speed/robustness with small N, dense is fine.
                    A = np.diag(J_diag) + np.diag(J_upper, k=1) + np.diag(J_lower, k=-1)
                    delta_psi = np.linalg.solve(A, -R)
                except np.linalg.LinAlgError:
                    if self.verbose: print(f"    Iter {iter}: Singular Matrix")
                    converged = False
                    break

                # 4. Update and Check Convergence
                psi_new += delta_psi
                norm_R = np.linalg.norm(R)
                norm_dpsi = np.linalg.norm(delta_psi)
                
                if self.verbose:
                    print(f"    Iter {iter}: |R|={norm_R:.2e}, |dpsi|={norm_dpsi:.2e}")
                
                if norm_R < tol_newton and norm_dpsi < tol_newton:
                    converged = True
                    self.stats['total_newton_iters'] += (iter + 1)
                    self.stats['max_newton_iters'] = max(self.stats['max_newton_iters'], iter + 1)
                    break
            
            # End of Time Step
            if converged:
                # Accept step
                self.psi = psi_new
                self.theta = theta_new
                t += dt
                self.stats['total_steps'] += 1
                
                # Mass Balance Check
                # Cumulative In = Influx * dt
                # mass_change = sum(theta_new - theta_old) * dz
                
                step_infl = flux_in * dt if top_bc_type=='flux' else 0 # Approximate
                cumulative_infl += step_infl
                
                # Save history periodically (approx every 0.1h or 10 steps)
                history['time'].append(t)
                history['psi_profiles'].append(self.psi.copy())
                history['theta_profiles'].append(self.theta.copy())
                history['cumulative_infl'].append(cumulative_infl)
                
                # Adaptive Time Step Increase
                if iter < 5:
                    dt *= 1.2
                elif iter > 10:
                    dt *= 0.8
                dt = min(dt, dt_max)
                
            else:
                # Reject step
                dt *= 0.5
                self.stats['failed_steps'] += 1
                if self.verbose:
                    print(f"  Step failed at t={t:.4f}, new dt={dt:.2e}")
                
                if dt < dt_min:
                    print("ERROR: dt too small. Simulation failed.")
                    break
        
        self.stats['cpu_time'] = time.time() - start_time
        print(f"Simulation done. CPU: {self.stats['cpu_time']:.3f}s, Steps: {self.stats['total_steps']}, Failed: {self.stats['failed_steps']}")
        
        return history, self.stats

# --- Helper Classes for SWCC Wrappers ---

class VGSWCCWrapper:
    """Wrapper for Analytical Van Genuchten"""
    def __init__(self, alpha, n, theta_r, theta_s, Ks):
        self.alpha = alpha
        self.n = n
        self.m = 1.0 - 1.0/n
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.Ks = Ks
        
    def swcc(self, psi):
        # Ensure input is float array
        psi_in = np.array(psi, dtype=float, copy=True)
        is_scalar = np.isscalar(psi)
        if is_scalar:
             psi_in = np.array([psi], dtype=float)
             
        # Handle psi > 0 (saturated)
        # We work with suction head (negative), so psi > 0 means pressure > 0
        # Formula usually expects suction head h > 0 (positive).
        # My Solver uses z-axis up, so unsaturated psi is negative.
        # VG formula: theta(h) where h = -psi.
        
        h = -psi_in
        h[h < 0] = 0 # Saturated
        
        Ah = self.alpha * h
        denom = (1.0 + Ah**self.n)**self.m
        Se = 1.0 / denom
        theta = self.theta_r + (self.theta_s - self.theta_r) * Se
        
        # Specific Moisture Capacity C = dtheta/dpsi = dtheta/d(-h) * (-1) = - dtheta/dh
        # dSe/dh = -m * (1+Ah^n)^(-m-1) * n * Ah^(n-1) * alpha
        dSe_dh = -self.m * (1.0 + Ah**self.n)**(-self.m - 1.0) * self.n * (Ah**(self.n-1.0)) * self.alpha
        
        # Avoid nan at h=0 if n < 1 (rare) or infinite derivative
        dSe_dh[h < 1e-5] = 0 
        
        C = (self.theta_s - self.theta_r) * np.abs(dSe_dh)
        
        if is_scalar:
            return float(theta[0]), float(C[0])
        return theta, C
        
    def conductivity(self, psi):
        # Mualem-Van Genuchten
        psi_in = np.array(psi, dtype=float, copy=True)
        is_scalar = np.isscalar(psi)
        if is_scalar:
             psi_in = np.array([psi], dtype=float)
             
        h = -psi_in
        h[h < 0] = 0
        
        Ah = self.alpha * h
        Se = (1.0 + Ah**self.n)**(-self.m)
        
        term = (1.0 - Se**(1.0/self.m))**self.m
        Kr = (Se**0.5) * (1.0 - term)**2
        
        K = self.Ks * Kr
        if is_scalar:
            return float(K[0])
        return K

class InterpolatedSWCCWrapper:
    """Wrapper for Discrete (Predicted) SWCC"""
    def __init__(self, suction_grid, theta_grid, Ks):
        self.Ks = Ks
        # Create interpolators
        # Predictors usually give log-spaced suction.
        # Log-linear interpolation is safer for SWCC.
        
        # Sort just in case
        idx = np.argsort(suction_grid)
        self.s_grid = suction_grid[idx]
        self.t_grid = theta_grid[idx]
        
        # Extend to 0 suction (saturation)
        if self.s_grid[0] > 1e-4:
            self.s_grid = np.insert(self.s_grid, 0, 0.0)
            self.t_grid = np.insert(self.t_grid, 0, self.t_grid[0]) # Assume saturated
            
        self.theta_s = self.t_grid[0]
        self.theta_r = self.t_grid[-1]
            
        # Linear interpolation in Log-Psi space for theta
        # But for robust C(psi), cubic spline might be too wiggly if points are noisy.
        # Let's use PchipInterpolator (monotonic) or UnivariateSpline?
        # Standard: simple linear interpolation.
        
        self.interp_theta = interp1d(self.s_grid, self.t_grid, kind='linear', fill_value='extrapolate')
        
    def swcc(self, psi):
        psi_abs = np.abs(psi)
        theta = self.interp_theta(psi_abs)
        
        # Numerical Derivative for C(psi)
        epsilon = 1e-3 * np.maximum(psi_abs, 0.1)
        theta_plus = self.interp_theta(psi_abs + epsilon)
        theta_minus = self.interp_theta(psi_abs - epsilon)
        C = np.abs(theta_plus - theta_minus) / (2 * epsilon)
        
        # If theta(psi) is non-monotone, C can be negative if we don't take abs().
        # But physically C should be dtheta/dpsi (negative).
        # In Richards eq term C * dpsi/dt, C is usually defined as |dtheta/dpsi|.
        # Let's assume C >= 0.
        
        return theta, C
        
    def conductivity(self, psi):
        # Using Mualem-VG with a "Best Fit" parameters or
        # Applying Mualem Integral to the discrete theta?
        # For GB comparison, usually we don't have K. 
        # But if we map theta->Se, we can use K = Ks * Se^0.5 * [...]
        # This assumes the GB curve implies a pore structure.
        
        theta, _ = self.swcc(psi)
        theta_e = (theta - self.theta_r) / (self.theta_s - self.theta_r + 1e-8)
        Se = np.clip(theta_e, 0, 1)
        
        # Simplified K for robustness if alpha/n unknown: 
        # K = Ks * Se^3 (Brooks-Corey like) or just Se^0.5 * ...
        # If we act like we don't know alpha/n (since it's GB), 
        # using a generic power law is common: K = Ks * Se^3.
        # OR: We can fit a local VG to get m? No that defeats the purpose.
        
        # Let's use simple power law K = Ks * Se^3 for the 'Predicted' curves without parameters.
        # This gives GB a fighting chance (smooth K) vs needing dK/dpsi from derivatives.
        Kr = Se**3.0 
        return self.Ks * Kr
