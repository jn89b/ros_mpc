import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim

class TrajectoryOptControlOuterLoop(OptimalControlProblem):
    """
    6-State Outer-Loop Kinematic Tracker.
    States: [x, y, h, V, psi, vz]
    Controls: [V_cmd, psi_cmd, vz_cmd]
    """
    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: ca.Opti,
                 target_x: float = 0.0,
                 target_y: float = 0.0,
                 target_alt: float = 100.0,
                 v_cruise: float = 18.0) -> None:
        super().__init__(mpc_params, casadi_model)
        
        # Spatial Targets (In a dynamic trajectory tracker, these would be 
        # passed dynamically via self.P instead of set at init)
        self.target_x = target_x
        self.target_y = target_y
        self.target_alt = target_alt
        self.v_cruise = v_cruise
        
        # Physical Airframe Limits
        self.v_stall_margin = 14.0 # Start heavily penalizing below 14 m/s

    def _parameter_length(self):
        return super()._parameter_length()
    
    def compute_dynamics_cost(self) -> ca.MX:
        cost = 0.0
        
        # We need a 4-element Q matrix: [Q_xy, Q_alt, Q_v, Q_vz_dampening]
        Q_xy  = float(self.mpc_params.Q[0]) # Horizontal tracking
        Q_alt = float(self.mpc_params.Q[2]) # Vertical tracking
        Q_v   = float(self.mpc_params.Q[3]) # Airspeed tracking
        Q_vz  = float(self.mpc_params.Q[4]) # Dampen actual climb rate oscillations
        
        # We need a 3-element R matrix for control slew rates: [R_V_cmd, R_psi_cmd, R_vz_cmd]
        R_slew_V   = float(self.mpc_params.R[0])
        R_slew_psi = float(self.mpc_params.R[1])
        R_slew_vz  = float(self.mpc_params.R[2])
        
        stall_weight:float = 5000.0 # Massive penalty multiplier for getting too slow
        
        for k in range(self.N):
            # Extract States (matching the ArduPlaneGuidanceModel)
            x   = self.X[0, k]
            y   = self.X[1, k]
            h   = self.X[2, k]
            V   = self.X[3, k]
            psi = self.X[4, k]
            vz  = self.X[5, k]
            
            # --- 1. SPATIAL TRACKING COST (Position) ---
            # Penalize distance from target waypoint
            cost += Q_xy * ((x - self.target_x)**2 + (y - self.target_y)**2)
            cost += Q_alt * (h - self.target_alt)**2  
            
            # --- 2. KINEMATIC TRACKING COST (Airspeed & Climb) ---
            # Try to hold the nominal cruise speed
            cost += Q_v * (V - self.v_cruise)**2
            
            # Keep actual vertical velocity smooth to prevent aggressive porpoising
            cost += Q_vz * (vz)**2 
            
            # --- 3. ANTI-STALL BARRIER (Soft Constraint) ---
            # ca.fmax returns 0 if V is safely above the margin.
            # If V drops below margin, it returns the difference, which is heavily penalized.
            stall_violation = ca.fmax(0.0, self.v_stall_margin - V)
            cost += stall_weight * (stall_violation**2)
            
            # --- 4. CONTROL SLEW RATES (Smooth Commands) ---
            if k > 0:
                delta_V_cmd   = self.U[0, k] - self.U[0, k-1]
                delta_psi_cmd = self.U[1, k] - self.U[1, k-1]
                delta_vz_cmd  = self.U[2, k] - self.U[2, k-1]
                
                # Heavily penalize massive jumps in commanded heading 
                # This acts as an implicit limit on the aircraft's bank angle roll rate
                cost += R_slew_V * (delta_V_cmd**2)
                cost += R_slew_psi * (delta_psi_cmd**2)
                cost += R_slew_vz * (delta_vz_cmd**2)

        return cost

    def compute_total_cost(self) -> ca.MX:
        return self.compute_dynamics_cost()