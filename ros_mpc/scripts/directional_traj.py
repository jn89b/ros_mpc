#!/usr/bin/env python3
import rclpy
import casadi as ca
import numpy as np
import time
import mavros
from mavros.base import SENSOR_QOS

from rclpy.node import Node
from drone_interfaces.msg import Telem, CtlTraj
from ros_mpc.models.MathModel import PlaneKinematicModel

from ros_mpc.rotation_utils import (ned_to_enu_states,
                                    yaw_enu_to_ned,
                                    enu_to_ned_states,
                                    euler_from_quaternion,
                                    convert_enu_state_sol_to_ned)

# from optitraj.mpc.PlaneOptControl import PlaneOptControl
from ros_mpc.PlaneOptControl import PlaneOptControl
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim

from typing import List, Dict, Any, Tuple

X_IDX = 0
Y_IDX = 1
Z_IDX = 2
PHI_IDX = 3
THETA_IDX = 4
PSI_IDX = 5
V_IDX = 6

U_PHI_IDX = 0
U_THETA_IDX = 1
U_PSI_IDX = 2
V_CMD_IDX = 3

class DirectionalTraj(Node):
    def __init__(self,
                 pub_freq: int = 100,
                 sub_freq: int = 100,
                 save_states: bool = False,
                 sub_to_mavros: bool = True):
        super().__init__('directional_traj')
        self.pub_freq = pub_freq
        self.sub_freq = sub_freq
        # intialize an array of nan
        self.num_states = 7
        self.enu_state: np.array = np.array([np.nan]*self.num_states)

        # self.sub_traj = self.create_subscription(
        #     Telem, 'telem', self.subscribe_telem, 10)

        self.state_sub = self.create_subscription(mavros.local_position.Odometry,
                                                  'mavros/local_position/odom',
                                                  self.mavros_state_callback,
                                                  qos_profile=SENSOR_QOS)
        self.pub_traj = self.create_publisher(
            CtlTraj, '/trajectory', 10)

        self.num_controls: int = 4
        self.current_enu_controls: np.array = np.array(
            [np.nan]*self.num_controls)

    def publish_traj(self,
                     solution: Dict[str, Any],
                     delta_sol_time: float,
                     idx_buffer:int = 0) -> None:
        """
        Trajectory published must be in NED frame
        Yaw control must be sent as relative NED command
        """
        # Solutions unpacked are in ENU frame
        # we need to convert to NED frame
        time_idx: int = get_time_idx(0.1, delta_sol_time,
                                     idx_buffer=idx_buffer)

        states, controls = unpack_optimal_control_results(solution)
        states: Dict[str, np.array] = states
        controls: Dict[str, np.array] = controls
        ned_states: Dict[str, np.array] = convert_enu_state_sol_to_ned(states)

        traj_msg: CtlTraj = CtlTraj()
        traj_msg.idx = time_idx
        traj_msg.x = ned_states['x'].tolist()
        traj_msg.y = ned_states['y'].tolist()
        #traj_msg.z = ned_states['z'].tolist()
        traj_msg.z = [-65.0, -65.0, -65.0, -65.0]
        traj_msg.roll = ned_states['phi'].tolist()
        traj_msg.pitch = ned_states['theta'].tolist()
        traj_msg.yaw = ned_states['psi'].tolist()
        traj_msg.vx = ned_states['v'].tolist()
        traj_msg.idx = time_idx + 1

        airspeed_error = states['v'][time_idx] - self.enu_state[6]        
        kp_airspeed:float = 0.25
        airspeed_cmd:float = kp_airspeed * airspeed_error
        min_thrust:float = 0.15
        max_thrust:float = 0.85
        thrust_cmd:float = np.clip(
            airspeed_cmd, min_thrust, max_thrust)
        traj_msg.thrust = [thrust_cmd,
                           thrust_cmd,
                           thrust_cmd,
                           thrust_cmd] 

        phi_cmd_rad: float = states['phi'][time_idx]
        theta_cmd_rad: float = states['theta'][time_idx]
        psi_cmd_rad: float = states['psi'][time_idx]
        vel_cmd: float = states['v'][time_idx]

        self.pub_traj.publish(traj_msg)
        self.update_controls(
            phi_cmd_rad=phi_cmd_rad,
            theta_cmd_rad=theta_cmd_rad,
            psi_cmd_float=psi_cmd_rad,
            vel_cmd=vel_cmd
        )

    def update_controls(self,
                        phi_cmd_rad: float,
                        theta_cmd_rad: float,
                        psi_cmd_float: float,
                        vel_cmd: float) -> None:
        """
        For the MPC controller we are sending the following commands
        roll, pitch, yaw (global), and airspeed commands
        Coordinate frame is ENU
        """
        self.current_enu_controls[U_PHI_IDX] = phi_cmd_rad
        self.current_enu_controls[U_THETA_IDX] = theta_cmd_rad
        self.current_enu_controls[U_PSI_IDX] = psi_cmd_float
        self.current_enu_controls[V_CMD_IDX] = vel_cmd

    def subscribe_telem(self, ned_msg: Telem) -> None:
        """
        State callbacks will be in NED frame
        need to convert to ENU frame
        """
        ned_state: np.array = np.array([
            ned_msg.x, ned_msg.y, ned_msg.z,
            ned_msg.roll, ned_msg.pitch, ned_msg.yaw,
            np.sqrt(ned_msg.vx**2 + ned_msg.vy**2 + ned_msg.vz**2)])

        self.enu_state: np.array = ned_to_enu_states(ned_state)

    def mavros_state_callback(self, msg: mavros.local_position.Odometry) -> None:
        """
        Converts NED to ENU and publishes the trajectory
          """
        self.enu_state[0] = msg.pose.pose.position.x
        self.enu_state[1] = msg.pose.pose.position.y
        self.enu_state[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = euler_from_quaternion(
            qx, qy, qz, qw)

        self.enu_state[3] = roll
        self.enu_state[4] = pitch
        self.enu_state[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        # get magnitude of velocity
        self.enu_state[6] = np.sqrt(vx**2 + vy**2 + vz**2)

def build_model(control_limits: Dict[str, Dict[str, float]],
                state_limits: Dict[str, Dict[str, float]]) -> PlaneKinematicModel:
    model: PlaneKinematicModel = PlaneKinematicModel()
    model.set_control_limits(control_limits)
    model.set_state_limits(state_limits)

    return model


def build_control_problem(mpc_params: MPCParams, casadi_model: PlaneKinematicModel) -> PlaneOptControl:
    plane_opt_control: PlaneOptControl = PlaneOptControl(
        mpc_params=mpc_params, casadi_model=casadi_model)
    return plane_opt_control


def custom_stop_criteria(state: np.ndarray,
                         final_state: np.ndarray) -> bool:
    distance = np.linalg.norm(state[0:2] - final_state[0:2])
    if distance < 5.0:
        return True


def unpack_optimal_control_results(
        optimal_control_results: Dict[str, Any]) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
    """
    Unpack the results of the optimal control problem
    """
    states: Dict[str, np.array] = optimal_control_results['states']
    controls: Dict[str, np.array] = optimal_control_results['controls']

    return states, controls


def get_time_idx(dt: float, solution_time: float,
                 idx_buffer: int = 0) -> int:
    """
    Args:
        dt (float): time step
        solution_time (float): time it took to solve the problem
        idx_buffer (int): buffer for the index
    Returns:
        int: index for the time step

    Returns the index of the time step that is closest to the solution time
    used to buffer the commands sent to the drone
    """
    time_rounded = round(solution_time, 1)

    if time_rounded <= 1:
        time_rounded = 1

    ctrl_idx = dt/time_rounded
    idx = int(round(ctrl_idx)) + idx_buffer

    return idx


def main(args=None):
    rclpy.init(args=args)
    traj_node = DirectionalTraj()
    rclpy.spin_once(traj_node)

    control_limits_dict: Dict[str, Dict[str, float]] = {
        'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'u_theta': {'min': -np.deg2rad(10), 'max': np.deg2rad(10)},
        'u_psi': {'min': -np.deg2rad(180), 'max': np.deg2rad(180)},
        'v_cmd': {'min': 10.0, 'max': 30.0}
    }
    state_limits_dict: Dict[str, Dict[str, float]] = {
        'x': {'min': -np.inf, 'max': np.inf},
        'y': {'min': -np.inf, 'max': np.inf},
        'z': {'min': 30, 'max': 100},
        'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'theta': {'min': -np.deg2rad(15), 'max': np.deg2rad(15)},
        'psi': {'min': -np.pi, 'max': np.pi},
        'v': {'min': 20, 'max': 30.0}
    }

    plane_model: PlaneKinematicModel = build_model(
        control_limits_dict, state_limits_dict)

    # now we will set the MPC weights for the plane
    # 0 means we don't care about the specific state variable 1 means we care about it
    Q: np.diag = np.diag([1.0, 1.0, 1.0, 0, 0, 0, 0])
    R: np.diag = np.diag([0.01, 0.01, 0.01, 1])

    # we will now slot the MPC weights into the MPCParams class
    mpc_params: MPCParams = MPCParams(Q=Q, R=R, N=15, dt=0.1)
    # formulate your optimal control problem
    plane_opt_control: PlaneOptControl = PlaneOptControl(
        mpc_params=mpc_params, casadi_model=plane_model)

    if np.all(np.isnan(traj_node.enu_state)):
        print("All elements are NaN")
    else:
        print("Not all elements are NaN")

    # now set your initial conditions for this case its the plane
    # x0: np.array = np.array([5, 5, 10, 0, 0, 0, 15])
    xF: np.array = np.array([0, 250, 60, 0, 0, 0, 15])
    u_0: np.array = np.array([0, 0, 0, 15])

    closed_loop_sim: CloseLoopSim = CloseLoopSim(
        optimizer=plane_opt_control,
        x_init=traj_node.enu_state,
        x_final=xF,
        print_every=1000,
        u0=u_0,
        N=100
    )

    # enu_traj: Dict[str, Any] = closed_loop_sim.run_single_step(
    #     xF=xF, u0=u_0)

    # time_duration: float = 20.0
    # time_start: float = time.time()

    # Initialize the node - X
    # Initialize optimization routine - X
    # Initiailze the closed loop simulation - X
    # recieve callback state information from the drone
    # set initial states and controls
    # set final states
    # run single step

    # In main loop:
    # callback information from drone
    # update the initial condition and initial control
    # Compute single step of the closed loop simulation
    # Get results that are ENU
    
    while rclpy.ok():
        try:
            rclpy.spin_once(traj_node, timeout_sec=0.1)
            start_sol_time: float = time.time()
            closed_loop_sim.x_init = traj_node.enu_state
            solution: Dict[str, Any] = closed_loop_sim.run_single_step(
                xF=xF,
                x0=traj_node.enu_state,
                u0=traj_node.current_enu_controls)
            delta_sol_time: float = time.time() - start_sol_time
            # distance
            distance = np.linalg.norm(
                np.array(traj_node.enu_state[0:3]) - np.array(xF[0:3]))
            print("Distance: ", distance)
            # publish the trajectory
            traj_node.publish_traj(solution, delta_sol_time,
                                idx_buffer=2)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    
    traj_node.destroy_node()
    rclpy.shutdown()
    return 

if __name__ == "__main__":
    main()