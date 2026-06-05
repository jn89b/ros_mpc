#!/usr/bin/env python3
import time
import numpy as np
import math
import rclpy
from rclpy.node import Node

import mavros
from mavros.base import SENSOR_QOS
from nav_msgs.msg import Odometry
from typing import Optional, Dict

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
	gravity: float = 9.81
	acc: float = yaw_rate * airspeed
	bank_angle_rad: float = np.arctan2(acc, gravity)
	return bank_angle_rad

def compute_max_yaw_rate(max_bank_angle_rad: float, airspeed: float) -> float:
	if airspeed <= 0.1:
		return 0.0
	gravity: float = 9.81
	max_yaw_rate: float = (gravity * np.tan(max_bank_angle_rad)) / airspeed
	return max_yaw_rate


# ==========================================
# MISSION MANAGER
# ==========================================
class MissionManager:
	"""
	Manages a queue of kinematic waypoints and handles arrival logic.
	"""
	def __init__(self, acceptance_radius_m: float = 25.0):
		self.waypoints = []
		self.current_idx = 0
		self.acceptance_radius = acceptance_radius_m

	def add_waypoint(self, x: float, y: float, alt: float, speed: float) -> None:
		"""Appends a new target to the mission queue."""
		self.waypoints.append({
			'x': x,
			'y': y,
			'alt': alt,
			'speed': speed
		})

	def get_current_target(self) -> Optional[Dict[str, float]]:
		"""Returns the active waypoint dictionary, or None if the mission is complete."""
		if self.current_idx < len(self.waypoints):
			return self.waypoints[self.current_idx]
		return None 

	def check_and_advance(self, current_x: float, current_y: float) -> bool:
		"""
		Calculates distance to the current target. 
		If within the acceptance radius, advances the index.
		Returns True if a waypoint was just reached.
		"""
		target = self.get_current_target()
		if not target:
			return False

		# Calculate 2D Euclidean distance to target
		dist_to_target = math.hypot(target['x'] - current_x, target['y'] - current_y)
		
		if dist_to_target <= self.acceptance_radius:
			self.current_idx += 1
			return True
			
		return False


# ==========================================
# ROS 2 NODE
# ==========================================
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
		self.state_info[X_IDX] = msg.pose.pose.position.x
		self.state_info[Y_IDX] = msg.pose.pose.position.y
		self.state_info[Z_IDX] = msg.pose.pose.position.z

		qx = msg.pose.pose.orientation.x
		qy = msg.pose.pose.orientation.y
		qz = msg.pose.pose.orientation.z
		qw = msg.pose.pose.orientation.w
		_, _, yaw = rot_utils.euler_from_quaternion(qx, qy, qz, qw)
		self.state_info[PSI_IDX] = yaw

		vx = msg.twist.twist.linear.x
		vy = msg.twist.twist.linear.y
		vz = msg.twist.twist.linear.z
	   
		self.state_info[V_IDX] = np.sqrt(vx**2 + vy**2 + vz**2)
		self.state_info[VZ_IDX] = vz

		if np.isnan(self.control_info[U_V_IDX]) and not np.isnan(self.state_info[V_IDX]):
			self.control_info[U_V_IDX] = self.state_info[V_IDX]
			self.control_info[U_PSI_IDX] = self.state_info[PSI_IDX]
			self.control_info[U_VZ_IDX] = 0.0

	def compute_los(self, target_x:float, target_y:float) -> float:
		dx = target_x - self.state_info[X_IDX]
		dy = target_y - self.state_info[Y_IDX]
		los_heading_rad: float = np.arctan2(dy, dx)
		return los_heading_rad

	def publish_trajectory(self, solution: dict,
						delta_sol_time: float,
						mpc_params: MPCParams,
						desired_alt_m:float,
						idx_buffer: int = 1) -> None:
		"""Parses CloseLoopSim nested dict and publishes the kinematic trajectory."""
		states = solution['states']
		controls = solution['controls']
	   
		idx_step = self.get_time_idx(mpc_params, delta_sol_time, idx_buffer)
		max_idx = len(states['x']) - 1
		idx_step = min(idx_step, max_idx)
		idx_step = -1

		v_cmd_mps = states['V'][idx_step]
		v_cmd_mps = 23.0
		psi_cmd_rad_enu = states['psi'][idx_step]

		x_pred = states['x'][idx_step]
		y_pred = states['y'][idx_step]

		airspeed = self.state_info[V_IDX]
		safe_airspeed = max(airspeed, 1.0)

		# 1. Define the airframe's physical limits (Old setup left intact)
		max_bank_deg = 45.0 
		max_yaw_rate_rad_s = compute_max_yaw_rate(
			max_bank_angle_rad=np.deg2rad(max_bank_deg),
			airspeed=safe_airspeed)
		
		look_ahead_s:float = 1.5
		max_turn_threshold = max_yaw_rate_rad_s * look_ahead_s
	   
		# 2. Calculate the raw Line-of-Sight target
		raw_target_heading = self.compute_los(x_pred, y_pred)
		current_heading = self.state_info[PSI_IDX]

		# 3. Calculate the shortest angular difference mapped to [-pi, pi]
		heading_error = np.arctan2(np.sin(raw_target_heading - current_heading),
								   np.cos(raw_target_heading - current_heading))
	   
		# 4. Define your threshold (e.g., maximum 45 degrees of change per update)
		# max_turn_threshold = math.radians(55.0)

		# 5. Clip the error
		clipped_error = np.clip(heading_error, -max_turn_threshold, max_turn_threshold)
		# we will check if the clipped error will exceed our desired bank angle 
		attempted_yaw_rate:float = clipped_error/look_ahead_s
		predicted_roll_rad = compute_bank_angle(yaw_rate=attempted_yaw_rate,
			airspeed=safe_airspeed)
		print("yaw rate", math.degrees(attempted_yaw_rate))
		print("predicted roll deg", math.degrees(predicted_roll_rad))
		# 6. Apply the clipped delta to the current heading
		psi_cmd_rad_enu = current_heading + clipped_error
		
		vz_cmd_mps_enu = controls['vz_cmd'][idx_step]
		
		psi_cmd_rad_ned = convert_yaw_enu_to_ned(psi_cmd_rad_enu)
		psi_cmd_rad_ned = wrap_to_2pi(psi_cmd_rad_ned) 
		psi_cmd_deg_ned = math.degrees(psi_cmd_rad_ned)

		print("x,y", states['x'][idx_step], states['y'][idx_step])
		print("predicted state", np.rad2deg(states['psi'][PSI_IDX]))
		print("actual state", np.rad2deg(self.state_info[PSI_IDX]))
	   
		vz_cmd_cms = vz_cmd_mps_enu * 10.0  
		z_traj = desired_alt_m

		traj_msg = CtlTraj()
		traj_msg.vx = [v_cmd_mps] * 10
		traj_msg.z = [z_traj] * 10
		traj_msg.yaw = [psi_cmd_deg_ned] * 10
		traj_msg.vz = [vz_cmd_cms] * 10
		traj_msg.idx = 4

		self.traj_pub.publish(traj_msg)
	   
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
	
	# Initialize the Waypoint Manager
	wp_manager = MissionManager(acceptance_radius_m=25.0)
	
	# Load Mission Waypoints
	wp_manager.add_waypoint(0.0, 300.0, 80.0, 23.0)
	wp_manager.add_waypoint(-300.0, 0.0, 80.0, 23.0)
	wp_manager.add_waypoint(0.0, -300.0, 80.0, 23.0)

	# Get the initial target to setup MPC bounds
	initial_target = wp_manager.get_current_target()
	v_cruise = initial_target['speed']

	max_roll = np.deg2rad(55)
	max_yaw_rate = compute_max_yaw_rate(max_bank_angle_rad=max_roll,
									 airspeed=v_cruise)
	print("max yaw rate", np.rad2deg(max_yaw_rate))
	
	plane_model = ArduPlaneGuidanceModel(dt_val=dt, 
        tau_V=0.5, tau_psi=0.4, tau_z=0.5)
   
	plane_model.set_control_limits({
		'V_cmd':   {'min': -2.0, 'max': 2.0},
		'r_cmd': {'min': -max_yaw_rate, 'max': max_yaw_rate},
		'vz_cmd':  {'min': -2.0, 'max': 1.5} 
	})

	plane_model.set_state_limits({
		'x':   {'min': -10000.0, 'max': 10000.0},
		'y':   {'min': -10000.0, 'max': 10000.0},
		'h':   {'min': 20.0,     'max': 100.0},
		'V':   {'min': 20.0,     'max': 25.0},
		'psi': {'min': -np.inf,   'max': np.inf},
		'vz':  {'min': -2.0,     'max': 2.0}
	})

	Q_matrix = np.array([5.0, 5.0, 5.0, 0.0, 2.0, 1.0])
	R_matrix = np.array([1.0, 1.0, 1.0]) 
   
	mpc_params = MPCParams(Q=Q_matrix, R=R_matrix, N=20, dt=dt)

	plane_opt_control = TrajectoryOptControlOuterLoop(
		mpc_params=mpc_params,
		casadi_model=plane_model,
		target_x=initial_target['x'],
		target_y=initial_target['y'],
		target_alt=initial_target['alt'],
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
		   
			if np.any(np.isnan(traj_node.state_info)) or np.any(np.isnan(traj_node.control_info)):
				traj_node.get_logger().info("Waiting for valid MAVROS telemetry...", throttle_duration_sec=2.0)
				continue
		   
			# --- MISSION MANAGEMENT ---
			current_x = traj_node.state_info[X_IDX]
			current_y = traj_node.state_info[Y_IDX]
			
			# Check if we hit the waypoint and need to advance
			if wp_manager.check_and_advance(current_x, current_y):
				traj_node.get_logger().info(f"Waypoint Reached! Advancing to WP {wp_manager.current_idx}")
			
			# Fetch the active target
			active_target = wp_manager.get_current_target()
			
			if not active_target:
				traj_node.get_logger().info("Mission Complete. Holding last waypoint.")
				break 

			# --- DYNAMIC TARGETING ---
			target_x = active_target['x']
			target_y = active_target['y']
			target_alt = active_target['alt']
			v_cruise = active_target['speed']

			los_target = traj_node.compute_los(target_x=target_x, target_y=target_y)
			
			# Update the MPC array dynamically based on the active waypoint
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
			traj_node.get_logger().info(f"WP[{wp_manager.current_idx}] Dist: {dist_to_target:.1f}m, Alt Err: {alt_error:.1f}m, Sol: {delta_sol_time*1000:.1f}ms")  

			traj_node.publish_trajectory(solution, delta_sol_time, mpc_params,
								desired_alt_m=xF[2],idx_buffer=1)

		except KeyboardInterrupt:
			traj_node.get_logger().info('Keyboard Interrupt: Shutting Down Node')
			break

	traj_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()