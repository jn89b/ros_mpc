import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim

import casadi as ca
from optitraj.models.casadi_model import CasadiModel

class ArduPlaneGuidanceModel(CasadiModel):
    """
    6-State Outer-Loop Kinematic Model for ArduPlane GUIDED Control.
    States: [x, y, h, V, psi, vz]
    Controls: [V_cmd, psi_cmd, vz_cmd]
    """
    def __init__(self, 
                 dt_val: float = 0.1, 
                 tau_V: float = 2.0, 
                 tau_psi: float = 0.3, 
                 tau_z: float = 0.5) -> None:
        super().__init__()
        self.dt_val = dt_val
        
        # Autopilot Response Time Constants (in Seconds)
        # These represent how long ArduPilot takes to achieve 63.2% of the commanded change.
        # Tune these by analyzing your flight logs!
        self.tau_V = tau_V     # Airspeed lag (TECS throttle response)
        self.tau_psi = tau_psi # Heading lag (L1 / roll controller response)
        self.tau_z = tau_z     # Climb rate lag (TECS pitch response)
        
        self.define_states()
        self.define_controls()
        self.define_state_space()

    def define_states(self) -> None:
        # Spatial Geometry
        self.x = ca.MX.sym('x')           # Easting/Local X (m)
        self.y = ca.MX.sym('y')           # Northing/Local Y (m)
        self.h = ca.MX.sym('h')           # Altitude (m)
        
        # Autopilot Actuals
        self.V = ca.MX.sym('V')           # Actual Airspeed (m/s)
        self.psi = ca.MX.sym('psi')       # Actual Heading (rad)
        self.vz = ca.MX.sym('vz')         # Actual Climb Rate (m/s)

        self.states = ca.vertcat(self.x, self.y, self.h, self.V, self.psi, self.vz)
        self.n_states = self.states.size()[0]

    def define_controls(self) -> None:
        self.V_cmd = ca.MX.sym('V_cmd')       # Commanded Airspeed (m/s)
        self.psi_cmd = ca.MX.sym('psi_cmd')   # Commanded Heading (rad)
        self.vz_cmd = ca.MX.sym('vz_cmd')     # Commanded Climb Rate (m/s)
        
        self.controls = ca.vertcat(self.V_cmd, self.psi_cmd, self.vz_cmd)
        self.n_controls = self.controls.size()[0]

    def define_state_space(self) -> None:
        # 1. Kinematic spatial derivatives (Dubins Coordinated Flight)
        self.x_dot = self.V * ca.cos(self.psi)
        self.y_dot = self.V * ca.sin(self.psi)
        self.h_dot = self.vz 
        
        # 2. First-order lag dynamics (Simulating ArduPilot's closed-loop response)
        self.V_dot = (1.0 / self.tau_V) * (self.V_cmd - self.V)
        self.psi_dot = (1.0 / self.tau_psi) * (self.psi_cmd - self.psi)
        self.vz_dot = (1.0 / self.tau_z) * (self.vz_cmd - self.vz)

        self.f_dot = ca.vertcat(self.x_dot, self.y_dot, self.h_dot, 
                                self.V_dot, self.psi_dot, self.vz_dot)
                                
        self.function = ca.Function('dynamics', 
                                    [self.states, self.controls],
                                    [self.f_dot])