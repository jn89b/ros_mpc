#!/usr/bin/env python3
import time
import numpy as np
import math
import rclpy
from rclpy.node import Node

import mavros
from mavros.base import SENSOR_QOS
from nav_msgs.msg import Odometry

# Assuming you update your custom message to handle kinematic targets
from drone_interfaces.msg import CtlTraj

import ros_mpc.rotation_utils as rot_utils
from ros_mpc.models.tecs_model import ArduPlaneGuidanceModel
from ros_mpc.TecsOptimalControl import TrajectoryOptControlOuterLoop
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim

# Array Indices for 6-State Outer Loop
X_IDX = 0
Y_IDX = 1
Z_IDX = 2
V_IDX = 3
PSI_IDX = 4
VZ_IDX = 5

# Array Indices for 3-Control Input
U_V_IDX = 0
U_PSI_IDX = 1
U_VZ_IDX = 2

### Utility functions ##
def convert_yaw_enu_to_ned(yaw_enu_rad:float) -> float:
    return (math.pi / 2.0) - yaw_enu_rad

def wrap_to_2pi(angle_rad:float) -> float:
    return angle_rad % (2.0 * math.pi) # Wraps to [0, 360)

def compute_bank_angle(yaw_rate: float, airspeed: float) -> float:
    """
    Computes the required bank angle for a coordinated turn.
   
    Args:
        yaw_rate (float): Commanded turn rate in rad/s.
        airspeed (float): Current airspeed in m/s.

    Returns:
        float: Required bank angle in radians.
    """
    gravity: float = 9.81
    acc: float = yaw_rate * airspeed
   
    # acc is the numerator (y), gravity is the denominator (x)
    bank_angle_rad: float = np.arctan2(acc, gravity)
   
    return bank_angle_rad


def compute_max_yaw_rate(max_bank_angle_rad: float, airspeed: float) -> float:
    """
    Computes the maximum allowable yaw rate for a coordinated turn
    given a maximum bank angle capability and current airspeed.
   
    Args:
        max_bank_angle_rad (float): Maximum allowed bank angle in radians.
        airspeed (float): Current airspeed in m/s.
       
    Returns:
        float: Maximum allowable yaw rate in rad/s.
    """
    # Prevent division by zero if the aircraft is stationary or in a weird state
    if airspeed <= 0.1:
        return 0.0
       
    gravity: float = 9.81
   
    # yaw_rate = (g * tan(bank_angle)) / V
    max_yaw_rate: float = (gravity * np.tan(max_bank_angle_rad)) / airspeed
   
    return max_yaw_rate

class EngageTrajNodeOuter(Node):
    def __init__(self, pub_freq: int = 10):
        super().__init__('outer_loop_mpc_publisher')
        self.get_logger().info('Starting Outer-Loop Kinematic MPC Publisher')
       
        self.pub_freq = pub_freq
       
        # --- 6-STATE ARRAY: [x, y, h, V, psi, vz] ---
        self.state_info = np.full(6, np.nan)
       
        # --- 3-CONTROL ARRAY: [V_cmd, psi_cmd, vz_cmd] ---
        self.control_info = np.full(3, 0.0)
       
        # Publishers & Subscribers
        self.traj_pub = self.create_publisher(CtlTraj, '/trajectory', self.pub_freq)
       
        # MAVROS Local Odometry (Typically ENU Frame in ROS)
        self.state_sub = self.create_subscription(
            Odometry,
            'mavros/local_position/odom',
            self.mavros_state_callback,
            qos_profile=SENSOR_QOS
        )

    def mavros_state_callback(self, msg: Odometry) -> None:
        # 1. Extract Position (ENU)
        self.state_info[X_IDX] = msg.pose.pose.position.x
        self.state_info[Y_IDX] = msg.pose.pose.position.y
        self.state_info[Z_IDX] = msg.pose.pose.position.z

        # 2. Extract Heading (Yaw in ENU)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        _, _, yaw = rot_utils.euler_from_quaternion(qx, qy, qz, qw)
        self.state_info[PSI_IDX] = yaw

        # 3. Extract Velocities
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
       
        # Actual airspeed (using ground speed proxy) and true vertical rate
        self.state_info[V_IDX] = np.sqrt(vx**2 + vy**2 + vz**2)
        self.state_info[VZ_IDX] = vz

        # Ensure u0 starts with safe values if MPC hasn't initialized yet
        if np.isnan(self.control_info[U_V_IDX]) and not np.isnan(self.state_info[V_IDX]):
            self.control_info[U_V_IDX] = self.state_info[V_IDX]
            self.control_info[U_PSI_IDX] = self.state_info[PSI_IDX]
            self.control_info[U_VZ_IDX] = 0.0

    def compute_los(self, target_x:float, target_y:float) -> float:
        """_summary_

        Args:
            target_x (float): _description_
            target_y (float): _description_

        Returns:
            float: _description_
        """
        dx = target_x - self.state_info[X_IDX]
        dy = target_y - self.state_info[Y_IDX]
        los_heading_rad: float = np.arctan2(dy, dx)
       
        return los_heading_rad


    def publish_trajectory(self, solution: dict, 
                           delta_sol_time: float, 
                           mpc_params: MPCParams,
                           desired_alt_m: float, 
                           idx_buffer: int = 1) -> None:
        """Parses CloseLoopSim nested dict and publishes the kinematic trajectory."""
        states = solution['states']
        controls = solution['controls']
        
        # 1. Horizon Setup
        # Hardcoded to -1 to target the very end of the MPC horizon for maximum ArduPilot L1 lookahead
        idx_step = -1

        # 2. Extract Optimal States & Controls
        v_cmd_mps = states['V'][idx_step]  # Note: If you want to force 23.0m/s, do it in your MPC constraints, not here!
        vz_cmd_mps_enu = controls['vz_cmd'][idx_step]
        x_pred = states['x'][idx_step]
        y_pred = states['y'][idx_step]
        current_heading = self.state_info[PSI_IDX]

        # 3. Heading Calculation & Static Clipping (The "Macro-Target" approach)
        # Calculate where the MPC wants the nose to point
        raw_target_heading = self.compute_los(x_pred, y_pred)

        # Calculate the shortest angular distance [-pi, pi] to prevent 358-degree spins
        heading_error = np.arctan2(np.sin(raw_target_heading - current_heading), 
                                   np.cos(raw_target_heading - current_heading))
        
        # Clip the error to give ArduPilot a wide enough error window to trigger a hard bank
        static_turn_threshold = math.radians(45.0) 
        clipped_error = np.clip(heading_error, -static_turn_threshold, static_turn_threshold)
        
        # Apply the safe clipped error to generate the final ENU command
        psi_cmd_rad_enu = current_heading + clipped_error

        # 4. ArduPilot Coordinate & Unit Conversions
        # ENU to NED Heading
        psi_cmd_rad_ned = convert_yaw_enu_to_ned(psi_cmd_rad_enu)
        psi_cmd_rad_ned = wrap_to_2pi(psi_cmd_rad_ned) 
        psi_cmd_deg_ned = math.degrees(psi_cmd_rad_ned)

        # ENU m/s to ArduPilot GUIDED_CHANGE_ALTITUDE cm/s
        vz_cmd_cms = vz_cmd_mps_enu * 10.0  

        # 5. Debugging
        self.get_logger().debug(f"Target X,Y: {x_pred:.1f}, {y_pred:.1f}")
        self.get_logger().debug(f"Predicted State (deg): {np.rad2deg(states['psi'][idx_step]):.1f}")
        self.get_logger().debug(f"Actual State (deg): {np.rad2deg(current_heading):.1f}")

        # 6. Populate and Publish Command
        traj_msg = CtlTraj()
        traj_msg.vx = [v_cmd_mps] * 10
        traj_msg.z = [desired_alt_m] * 10
        traj_msg.yaw = [psi_cmd_deg_ned] * 10
        traj_msg.vz = [vz_cmd_cms] * 10 
        traj_msg.idx = 4

        self.traj_pub.publish(traj_msg)
        
        # 7. Update tracking for the next MPC step (u0)
        # MUST feed ENU values back into the MPC, not the NED values
        self.control_info[U_V_IDX] = v_cmd_mps
        self.control_info[U_PSI_IDX] = controls['r_cmd'][idx_step]
        self.control_info[U_VZ_IDX] = vz_cmd_mps_enu

    def get_time_idx(self, mpc_params: MPCParams, solution_time: float, idx_buffer: int = 0) -> int:
        time_rounded = max(round(solution_time, 1), 1.0)
        ctrl_idx = mpc_params.dt / time_rounded
        return int(round(ctrl_idx)) + idx_buffer

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main(args=None) -> None:
    rclpy.init(args=args)    
    dt = 0.1
    # Example Target: 200m North, 200m East, 100m Altitude
    target_x = 300.0
    target_y = 0.0
    target_alt = 80.0
    v_cruise = 23.0

    max_roll = np.deg2rad(45)
    max_yaw_rate = compute_max_yaw_rate(max_bank_angle_rad=max_roll,
                                     airspeed=v_cruise)
    print("max yaw rate", np.rad2deg(max_yaw_rate))
    # Initialize the 6-state Guidance Model with ArduPilot Time Constants
    plane_model = ArduPlaneGuidanceModel(dt_val=dt, tau_V=0.5, tau_psi=0.4,
                                      tau_z=0.5)
   
    # Set physical bounds for a typical fixed-wing
    plane_model.set_control_limits({
        'V_cmd':   {'min': -2.0, 'max': 2.0},
        'r_cmd': {'min': -max_yaw_rate, 'max': max_yaw_rate},
        'vz_cmd':  {'min': -2.0, 'max': 1.5} # SI limits (m/s)
    })

    plane_model.set_state_limits({
        'x':   {'min': -10000.0, 'max': 10000.0},
        'y':   {'min': -10000.0, 'max': 10000.0},
        'h':   {'min': 20.0,     'max': 100.0},
        'V':   {'min': 20.0,     'max': 25.0},
        'psi': {'min': -np.inf,   'max': np.inf},
        'vz':  {'min': -2.0,     'max': 2.0}
    })

    # Weight matrices: Q = [Q_xy, Q_alt, Q_v, Q_vz_dampening]  
    # Slew matrices: R = [R_V_cmd, R_psi_cmd, R_vz_cmd]
    Q_matrix = np.array([5.0, 5.0, 5.0, 0.0, 2.0, 1.0])
    R_matrix = np.array([1.0, 1.0, 1.0]) # Heavy penalty on heading/climb slew
   
    mpc_params = MPCParams(Q=Q_matrix, R=R_matrix, N=20, dt=dt)

    plane_opt_control = TrajectoryOptControlOuterLoop(
        mpc_params=mpc_params,
        casadi_model=plane_model,
        target_x=target_x,
        target_y=target_y,
        target_alt=target_alt,
        v_cruise=v_cruise
    )
   
    traj_node = EngageTrajNodeOuter()

    closed_loop_sim = CloseLoopSim(
        optimizer=plane_opt_control,
        x_init=traj_node.state_info,
        x_final=np.zeros(6),
        print_every=1000,
        u0=traj_node.control_info,
        N=100
    )

    print("Waiting for MAVROS odometry telemetry...")
    for _ in range(10):
        rclpy.spin_once(traj_node, timeout_sec=0.1)  

    while rclpy.ok():
        try:
            rclpy.spin_once(traj_node, timeout_sec=0.01)
           
            # --- SAFETY GUARD ---
            if np.any(np.isnan(traj_node.state_info)) or np.any(np.isnan(traj_node.control_info)):
                traj_node.get_logger().info("Waiting for valid MAVROS telemetry...", throttle_duration_sec=2.0)
                continue
           
            # [Target X, Target Y, Target Alt, Cruise Speed, Target Heading, Zero Climb]
            # (Note: In a pure pursuit MPC, the heading target is implicitly handled by the xy cost)
            los_target:float = traj_node.compute_los(target_x=target_x, target_y=target_y)
            xF = np.array([target_x, target_y, target_alt, v_cruise, los_target, 0.0])
           
            start_sol_time = time.time()
           
            closed_loop_sim.x_init = traj_node.state_info
           
            solution = closed_loop_sim.run_single_step(
                xF=xF,
                x0=traj_node.state_info,
                u0=traj_node.control_info
            )
           
            delta_sol_time = time.time() - start_sol_time
           
            dist_to_target = np.sqrt((target_x - traj_node.state_info[X_IDX])**2 + (target_y - traj_node.state_info[Y_IDX])**2)
            alt_error = abs(target_alt - traj_node.state_info[Z_IDX])
            traj_node.get_logger().info(f"Dist: {dist_to_target:.1f}m, Alt Err: {alt_error:.1f}m, Sol: {delta_sol_time*1000:.1f}ms")  

            traj_node.publish_trajectory(solution, delta_sol_time, mpc_params,
                                desired_alt_m=xF[2],idx_buffer=1)

        except KeyboardInterrupt:
            traj_node.get_logger().info('Keyboard Interrupt: Shutting Down Node')
            break

    traj_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()