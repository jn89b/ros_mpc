import casadi as ca
from optitraj import CasadiModel
from typing import List, Optional
import numpy as np
import math as m

def wrap_to_pi(angle: ca.SX) -> ca.SX:
    """
    Wrap an angle (CasADi SX expression) to the range [-pi, pi).

    Args:
        angle (ca.SX): Input angle in radians.

    Returns:
        ca.SX: The wrapped angle.
    """
    return angle - 2 * ca.pi * ca.floor((angle + ca.pi) / (2 * ca.pi))


class DataHandler:
    """
    Handles logging of simulation data including state variables, controls, wind, time, and rewards.
    """

    def __init__(self) -> None:
        """
        Initialize empty lists for storing simulation data.
        """
        self.x: List[float] = []
        self.y: List[float] = []
        self.z: List[float] = []
        self.phi: List[float] = []
        self.theta: List[float] = []
        self.psi: List[float] = []
        self.v: List[float] = []
        self.p: List[float] = []
        self.q: List[float] = []
        self.r: List[float] = []
        self.u_phi: List[float] = []
        self.u_theta: List[float] = []
        self.u_psi: List[float] = []
        self.v_cmd: List[float] = []
        self.wind_x: List[float] = []
        self.wind_y: List[float] = []
        self.wind_z: List[float] = []
        self.time: List[float] = []
        self.rewards: List[float] = []
        self.yaw: List[float] = []

    def update_states(self, info_array: np.ndarray) -> None:
        """
        Update the state variables from a given numpy array.

        The expected order is: [x, y, z, phi, theta, psi, v, p, q, r].

        Args:
            info_array (np.ndarray): Array containing state information.
        """
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.phi.append(info_array[3])
        self.theta.append(info_array[4])
        self.psi.append(info_array[5])
        self.v.append(info_array[6])

    def update_controls(self, control_array: np.ndarray) -> None:
        """
        Update the control inputs from a given numpy array.

        The expected order is: [u_phi, u_theta, u_psi, v_cmd].

        Args:
            control_array (np.ndarray): Array containing control information.
        """
        self.u_phi.append(control_array[0])
        self.u_theta.append(control_array[0])
        self.u_psi.append(control_array[1])
        self.v_cmd.append(control_array[2])

    def update_reward(self, reward: float) -> None:
        """
        Update the reward log.

        Args:
            reward (float): The reward value to log.
        """
        self.rewards.append(reward)

    def update_time(self, time: float) -> None:
        """
        Update the time log.

        Args:
            time (float): The current time stamp.
        """
        self.time.append(time)

    def update_wind(self, wind_array: np.ndarray) -> None:
        """
        Update the wind data from a given numpy array.

        The expected order is: [wind_x, wind_y, wind_z].

        Args:
            wind_array (np.ndarray): Array containing wind information.
        """
        self.wind_x.append(wind_array[0])
        self.wind_y.append(wind_array[1])
        self.wind_z.append(wind_array[2])

    def update(self, info_array: np.ndarray,
               control_array: np.ndarray,
               wind_array: np.ndarray,
               time: float,
               reward: float) -> None:
        """
        Update all logged data (states, controls, wind, time, and reward) at once.

        Args:
            info_array (np.ndarray): State information.
            control_array (np.ndarray): Control input information.
            wind_array (np.ndarray): Wind data.
            time (float): Time stamp.
            reward (float): Reward value.
        """
        self.update_states(info_array)
        self.update_controls(control_array)
        self.update_wind(wind_array)
        self.update_time(time)
        self.update_reward(reward)

class PlaneKinematicModel(CasadiModel):
    """
    A simple kinematic model of an aircraft operating in a North-East-Down (NED) frame.

    The model includes state and control definitions, wind effects, and numerical integration using
    a 4th order Runge-Kutta (RK45) method.
    """

    def __init__(self,
                 dt_val: float = 0.1,
                 tau_v: float = 0.15,
                 tau_phi: float = 0.07,
                 tau_theta: float = 0.1,
                 tau_psi: float = 0.10,
                 tau_p: float = 0.1,
                 tau_q: float = 0.1,
                 tau_r: float = 0.1) -> None:
        """
        Initialize the plane kinematic model with the specified time step and time constants.

        Args:
            dt_val (float): Integration time step.
            tau_v (float): Time constant for airspeed dynamics.
            tau_phi (float): Time constant for roll command tracking.
            tau_theta (float): Time constant for pitch command tracking.
            tau_psi (float): Time constant for yaw command tracking.
            tau_p (float): Time constant for roll rate dynamics.
            tau_q (float): Time constant for pitch rate dynamics.
            tau_r (float): Time constant for yaw rate dynamics.
        """
        self.dt_val: float = dt_val
        # Time constants
        self.tau_v: float = tau_v
        self.tau_phi: float = tau_phi
        self.tau_theta: float = tau_theta
        self.tau_psi: float = tau_psi
        self.tau_p: float = tau_p
        self.tau_q: float = tau_q
        self.tau_r: float = tau_r
        
        self.define_states()
        self.define_controls()
        self.define_wind()
        self.define_state_space()
        
        self.state_info: Optional[np.ndarray] = None
        self.data_handler: DataHandler = DataHandler()

    def update_state_info(self, state_info: np.ndarray) -> None:
        """
        Set the current state information and update the logged state data.

        Args:
            state_info (np.ndarray): Array containing the current state.
        """
        self.state_info = state_info
        self.data_handler.update_states(state_info)

    def update_controls(self, control: np.ndarray) -> None:
        """
        Args:
            control (np.ndarray): The control input to log.
        """
        self.data_handler.update_controls(control)

    def update_time_log(self, time: float) -> None:
        """
        Log the current time.

        Args:
            time (float): The current time.
        """
        self.data_handler.update_time(time)

    def define_states(self) -> None:
        """
        Define the symbolic state variables of the system in the NED frame.

        States include:
            x_f, y_f, z_f: Positions.
            phi_f, theta_f, psi_f: Euler angles.
            v_f: Airspeed.
            p_f, q_f, r_f: Angular rates (roll, pitch, yaw).
        """
        self.x_f: ca.SX = ca.SX.sym('x_f')
        self.y_f: ca.SX = ca.SX.sym('y_f')
        self.z_f: ca.SX = ca.SX.sym('z_f')
        self.phi_f: ca.SX = ca.SX.sym('phi_f')
        self.theta_f: ca.SX = ca.SX.sym('theta_f')
        self.psi_f: ca.SX = ca.SX.sym('psi_f')
        self.v_f: ca.SX = ca.SX.sym('v_f')
        # self.p_f: ca.SX = ca.SX.sym('p')  # roll rate
        # self.q_f: ca.SX = ca.SX.sym('q')  # pitch rate
        # self.r_f: ca.SX = ca.SX.sym('r')  # yaw rate

        self.states: ca.SX = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f,
            self.v_f,
        )

        self.n_states: int = int(self.states.size()[0])

    def define_controls(self) -> None:
        """
        Define the symbolic control input variables for the system.

        Controls include:
            u_phi, u_theta, u_psi: Attitude commands.
            v_cmd: Airspeed command.
        """
        self.u_phi: ca.SX = ca.SX.sym('u_phi')
        self.u_theta: ca.SX = ca.SX.sym('u_theta')
        self.u_psi: ca.SX = ca.SX.sym('u_psi')
        self.v_cmd: ca.SX = ca.SX.sym('v_cmd')

        self.controls: ca.SX = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls: int = int(self.controls.size()[0])

    def define_wind(self) -> None:
        """        return obs, reward, done, info

        Define the symbolic wind components in the inertial NED frame.
        """
        self.wind_x: ca.SX = ca.SX.sym('wind_x')
        self.wind_y: ca.SX = ca.SX.sym('wind_y')
        self.wind_z: ca.SX = ca.SX.sym('wind_z')

        self.wind: ca.SX = ca.vertcat(
            self.wind_x,
            self.wind_y,
            self.wind_z
        )

    def define_state_space(self, make_z_positive_up: bool = True) -> None:
        """
        Define the state space of the system and construct the ODE function.
        Args:
            make_z_positive (bool): If True, enforce that the z-coordinate is positive.
            This means the convention becomes NEU (North-East-Up).

        The ODE is defined using the kinematic equations for an aircraft in a NED frame,
        including wind effects and first-order lag dynamics for airspeed and angular rates.

        If frame is NED:
            Where x is North, y is East, z is Down
            Positive roll is right wing down
            Positive pitch is nose up
            Positive yaw is CW
        If frame is ENU:
            Where x is North, y is East, z is Up
            Positive roll is LEFT wing down
            Positive pitch is nose up
            Positive yaw is CCW 
        """

        # # Airspeed dynamics (airspeed does not include wind)
        self.v_dot: ca.SX = (self.v_cmd - self.v_f) / self.tau_v
        self.g = 9.81  # m/s^2
        # #body to inertia frame

        self.x_fdot = self.v_f * \
            ca.cos(self.theta_f) * ca.cos(self.psi_f)
        self.y_fdot = self.v_f * \
            ca.cos(self.theta_f) * ca.sin(self.psi_f)

        if make_z_positive_up:
            self.z_fdot = self.v_f * ca.sin(self.theta_f)
        else:
            self.z_fdot = -self.v_f * ca.sin(self.theta_f)

        phi_cmd = self.u_phi
        # self.phi_fdot: ca.SX = (self.u_phi - self.phi_f) / self.tau_phi
        self.phi_fdot: ca.SX = (phi_cmd - self.phi_f) / self.tau_phi

        # So a positive u_theta means we want the nose to be up
        self.theta_fdot: ca.SX = (
            -self.u_theta - self.theta_f) / self.tau_theta
    
        
        # This is the yaw rate command 
        if make_z_positive_up:
            self.psi_fdot: ca.SX = self.g * (ca.tan(self.phi_f) / self.v_f)
        else:
            self.psi_fdot = -self.g * (ca.tan(self.phi_f) / self.v_f)

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot,
            self.v_dot
        )

        # Define the ODE function: f(states, controls, wind) -> state derivatives
        self.function: ca.Function = ca.Function(
            'f', [self.states, self.controls], [self.z_dot])

    def update_reward(self, reward: float) -> None:
        """
        Update the reward data in the data handler.

        Args:
            reward (float): The reward value to x`log.
        """
        self.data_handler.update_reward(reward)

    def rk45(self, x: ca.SX, u: ca.SX, dt: float, use_numeric: bool = True,
             wind=np.array([0, 0, 0]),
             save_next_step: bool = False) -> np.ndarray:
        """
        Perform one integration step using the 4th order 
        Runge-Kutta (RK45) method.

        Args:
            x (ca.SX): Current state.
            u (ca.SX): Current control input.
            dt (float): Integration time step.
            use_numeric (bool): If True, returns a flattened numpy array; otherwise returns a CasADi expression.
            wind (np.ndarray): Wind vector. Default is [0, 0, 0].
            save_next_step (bool): If True, save the next state in the data handler.
        Returns:
            np.ndarray: Next state as a flattened numpy array if use_numeric is True.
        """
        # check if shape is correct
        if x.shape[0] != self.n_states:
            raise ValueError("input x does not match size of states: ",
                             x.shape[0], self.n_states)

        k1: ca.SX = self.function(x, u, wind)
        k2: ca.SX = self.function(x + dt / 2 * k1, u, wind)
        k3: ca.SX = self.function(x + dt / 2 * k2, u, wind)
        k4: ca.SX = self.function(x + dt * k3, u, wind)
        next_step: ca.SX = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if use_numeric:
            next_step_np: np.ndarray = np.array(next_step).flatten()
            # Wrap the yaw angle to be within [-pi, pi]
            yaw_idx: int = 5
            next_step_np[yaw_idx] = (
                next_step_np[yaw_idx] + np.pi) % (2 * np.pi) - np.pi
            if save_next_step:
                self.data_handler.update_states(next_step_np)
                self.data_handler.update_controls(u)
            return next_step_np
        else:
            if save_next_step:
                self.data_handler.update_states(next_step)
            return next_step

class SimpleKinematicModel(CasadiModel):
    def __init__(self):
        """
        A simpler kinematic model of an aircraft where states are defined as:
        - x : North position
        - y : East position
        - z : Up position
        - chi X: heading angle from North
        - gamma: Flight path angle
        - v: Airspeed
        
        Controls are defined as:
        - phi: roll angle
        - theta: pitch angle
        - Throttle command between 0 and 1
        
        """
        self.dt_val: float = 0.1
        self.mass_kg:float = 1.0
        self.g: float = 9.81 
        self.veloctiy_tau: float = 2.0
        self.k_speed: float = 1.2
        self.alpha: float = m.exp(-self.dt_val/self.veloctiy_tau)
        self.beta  = self.k_speed * self.veloctiy_tau * \
            (1 - m.exp(-self.dt_val/self.veloctiy_tau))
        self.define_states()
        self.define_controls()
        self.define_state_space()
        
        self.state_info: Optional[np.ndarray] = None
        self.data_handler: DataHandler = DataHandler()
        
    def define_states(self) -> None:
        """
        Define the symbolic state variables of the system in the NED frame
        """
        self.x : ca.SX = ca.SX.sym('x')
        self.y : ca.SX = ca.SX.sym('y')
        self.z : ca.SX = ca.SX.sym('z')
        self.chi_x : ca.SX = ca.SX.sym('chi_x')
        self.gamma : ca.SX = ca.SX.sym('gamma')
        self.v : ca.SX = ca.SX.sym('v')
        self.states: ca.SX = ca.vertcat(
            self.x,
            self.y,
            self.z,
            self.chi_x,
            self.gamma,
            self.v
        )
        
        self.n_states: int = int(self.states.size()[0])
    
    def define_controls(self) -> None:
        """
        Define the symbolic control input variables for the system
        """
        self.u_phi: ca.SX = ca.SX.sym('u_phi')
        self.u_theta: ca.SX = ca.SX.sym('u_theta')
        self.u_throttle: ca.SX = ca.SX.sym('u_throttle')

        self.controls: ca.SX = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_throttle
        )
        self.n_controls: int = int(self.controls.size()[0])
        
    def compute_roll_ref(self, vel:float, u_chi:float) -> float:
        """
        Compute the roll reference angle based on the target position
        https://aviation.stackexchange.com/questions/50078/what-is-the-maximum-angle-the-f-16-can-turn-in-x-seconds-while-flying-at-corne
        https://discuss.ardupilot.org/t/coordinated-turn/88982
        """

        roll_ref = np.arctan2(vel*u_chi, self.g) 
        return roll_ref

    def define_state_space(self) -> None:
        """
        Define the state space of the system and construct the ODE function
        """
        # Airspeed dynamics (airspeed does not include wind)
        self.x_dot : ca.SX = self.v * ca.cos(self.gamma) * ca.cos(self.chi_x)
        self.y_dot : ca.SX = self.v * ca.cos(self.gamma) * ca.sin(self.chi_x)
        self.z_dot : ca.SX = self.v * ca.sin(self.gamma)
        self.chi_x_dot: ca.SX = (self.g/self.v) * ca.tan(self.u_phi)
        self.gamma_dot: ca.SX = self.u_theta - self.gamma
        # self.v_dot: ca.SX = (self.alpha * self.v) + \
        #     (self.beta * self.u_throttle)
        self.v_dot: ca.SX = self.v
            
        self.z_dot = ca.vertcat(
            self.x_dot,
            self.y_dot,
            self.z_dot,
            self.chi_x_dot,
            self.gamma_dot,
            self.v_dot
        )
        
        # Define the ODE function: f(states, controls) -> state derivatives
        self.function: ca.Function = ca.Function(
            'f', [self.states, self.controls], [self.z_dot])
        
    def rk45(self, x: ca.SX, u: ca.SX, dt: float, use_numeric: bool = True,
             wind=np.array([0, 0, 0]),
             save_next_step: bool = False) -> np.ndarray:
        """
        Perform one integration step using the 4th order 
        Runge-Kutta (RK45) method.

        Args:
            x (ca.SX): Current state.
            u (ca.SX): Current control input.
            dt (float): Integration time step.
            use_numeric (bool): If True, returns a flattened numpy array; otherwise returns a CasADi expression.
            wind (np.ndarray): Wind vector. Default is [0, 0, 0].
            save_next_step (bool): If True, save the next state in the data handler.
        Returns:
            np.ndarray: Next state as a flattened numpy array if use_numeric is True.
        """
        # check if shape is correct
        if x.shape[0] != self.n_states:
            raise ValueError("input x does not match size of states: ",
                             x.shape[0], self.n_states)

        k1: ca.SX = self.function(x, u)
        k2: ca.SX = self.function(x + dt / 2 * k1, u)
        k3: ca.SX = self.function(x + dt / 2 * k2, u)
        k4: ca.SX = self.function(x + dt * k3, u)
        next_step: ca.SX = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if use_numeric:
            next_step_np: np.ndarray = np.array(next_step).flatten()
            # Wrap the yaw angle to be within [-pi, pi]
            yaw_idx: int = 3
            next_step_np[yaw_idx] = (
                next_step_np[yaw_idx] + np.pi) % (2 * np.pi) - np.pi
            if save_next_step:
                self.data_handler.update_states(next_step_np)
                self.data_handler.update_controls(u)
            return next_step_np
        else:
            if save_next_step:
                self.data_handler.update_states(next_step)
            return next_step
        