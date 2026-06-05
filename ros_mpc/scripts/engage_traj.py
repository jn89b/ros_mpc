#!/usr/bin/env python3
import time
import numpy as np
import math
import rclpy
from rclpy.node import Node

import mavros
from mavros.base import SENSOR_QOS
from nav_msgs.msg import Odometry

# MAVROS Mission & Global Position Imports
from mavros_msgs.srv import WaypointPull
from mavros_msgs.msg import WaypointList, Waypoint, HomePosition

from typing import Optional, Dict, List, Tuple
from scipy.constants import g

# Assuming you update your custom message to handle kinematic targets
from drone_interfaces.msg import CtlTraj

import ros_mpc.rotation_utils as rot_utils
from ros_mpc.models.tecs_model import ArduPlaneGuidanceModel
from ros_mpc.TecsOptimalControl import TrajectoryOptControlOuterLoop
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim

# Earth radius in meters
EARTH_RADIUS = 6371000  

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

# ==========================================
# DRONE MATH UTILITIES
# ==========================================
class DroneMath():
	@staticmethod
	def compute_loiter_radius_m(mount_angle_phi_deg:float, 
								cam_range_m:float, 
								ccw_loiter:bool,
								roll_limit_deg:float) -> float:
		neg_roll_deg = ((-1) * roll_limit_deg)
		if mount_angle_phi_deg < neg_roll_deg:
			mount_angle_phi_deg = neg_roll_deg
		elif mount_angle_phi_deg > roll_limit_deg:
			mount_angle_phi_deg = roll_limit_deg
		
		if ccw_loiter:
			return (-1)*(math.sin(math.radians(mount_angle_phi_deg)) * cam_range_m)
		return (math.sin(math.radians(mount_angle_phi_deg)) * cam_range_m)
	
	@staticmethod
	def calc_velocity(mount_angle_phi_deg:float, cam_range_m:float, ccw_loiter:bool) -> float:
		loiter_radius:float = DroneMath.compute_loiter_radius_m(mount_angle_phi_deg=mount_angle_phi_deg,
																cam_range_m=cam_range_m,
																ccw_loiter=ccw_loiter,
																roll_limit_deg=45.0) # Assumed default for signature
		mount_angle_rad = math.radians(mount_angle_phi_deg)
		if ccw_loiter:
			return math.sqrt((-1)*(loiter_radius) * g * math.tan(mount_angle_rad))
		return math.sqrt(loiter_radius * g * math.tan(mount_angle_rad))
	
	@staticmethod
	def compute_altitude_m(cam_range_m_alt:float,
						   max_roll_tan:float,
						   max_alt_m:float,
						   min_alt_m:float) -> float:
		max_roll_rad = np.deg2rad(max_roll_tan)
		calc_alt = (cam_range_m_alt)*(math.cos(max_roll_rad))
		if calc_alt > max_alt_m:
			return max_alt_m
		elif calc_alt < min_alt_m:
			return min_alt_m
		else:
			return calc_alt

	@staticmethod
	def realtime_loiter_radius(mount_angle_phi_deg:float, 
								cam_range_m:float,
								roll_limit_deg:float) -> float:
		neg_roll_deg = ((-1) * roll_limit_deg)
		if mount_angle_phi_deg < neg_roll_deg:
			mount_angle_phi_deg = neg_roll_deg
		elif mount_angle_phi_deg > roll_limit_deg:
			mount_angle_phi_deg = roll_limit_deg
		
		return (math.sin(math.radians(mount_angle_phi_deg)) * cam_range_m)
	
	@staticmethod
	def calculate_loiter_time(num_loiters:int, loiter_radius: float, aircraft_velocity_mps: float) -> float: 
		loiter_circumference: float = (2) * (np.pi) * (loiter_radius)
		total_distance: float = loiter_circumference * num_loiters
		loiter_time = (total_distance) / aircraft_velocity_mps
		return loiter_time

	@staticmethod
	def deg2rad(deg: float) -> float:
		return deg * math.pi / 180 

	@staticmethod
	def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float: 
		dlat: float = DroneMath.deg2rad(lat2 - lat1)
		dlon: float = DroneMath.deg2rad(lon2 - lon1)
		lat1 = DroneMath.deg2rad(lat1)
		lat2 = DroneMath.deg2rad(lat2)

		a: float = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
		c: float = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
		return EARTH_RADIUS * c

	@staticmethod
	def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
		lat1 = DroneMath.deg2rad(lat1)
		lat2 = DroneMath.deg2rad(lat2)
		dlon = DroneMath.deg2rad(lon2 - lon1)

		x = math.sin(dlon) * math.cos(lat2)
		y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
		return math.atan2(x, y)

	@staticmethod
	def geodetic_to_cartesian(origin_lat: float, origin_lon: float, target_lat: float, target_lon: float) -> Tuple[float, float]:
		distance: float = DroneMath.haversine_distance(origin_lat, origin_lon, target_lat, target_lon)
		bearing: float = DroneMath.initial_bearing(origin_lat, origin_lon, target_lat, target_lon)
		x: float = distance * math.sin(bearing)
		y: float = distance * math.cos(bearing)
		return x, y

### Additional Utility functions ##
def convert_yaw_enu_to_ned(yaw_enu_rad:float) -> float:
	return (math.pi / 2.0) - yaw_enu_rad

def wrap_to_2pi(angle_rad:float) -> float:
	return angle_rad % (2.0 * math.pi) # Wraps to [0, 360)

def compute_bank_angle(yaw_rate: float, airspeed: float) -> float:
	acc: float = yaw_rate * airspeed
	bank_angle_rad: float = np.arctan2(acc, g)
	return bank_angle_rad

def compute_max_yaw_rate(max_bank_angle_rad: float, airspeed: float) -> float:
	if airspeed <= 0.1:
		return 0.0
	max_yaw_rate: float = (g * np.tan(max_bank_angle_rad)) / airspeed
	return max_yaw_rate


# ==========================================
# MISSION MANAGER
# ==========================================
class MissionManager:
	"""
	Manages a queue of kinematic waypoints and handles arrival logic.
	"""
	def __init__(self, acceptance_radius_m: float = 25.0, 
				 total_num_laps:int = 1):
		self.waypoints = []
		self.current_idx = 0
		self.acceptance_radius = acceptance_radius_m
		self.total_num_laps:int = total_num_laps
		self.current_lap_idx:int = 0

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
		if self.current_idx >= len(self.waypoints) and self.current_lap_idx <= self.total_num_laps:
			self.current_idx = 0
			self.current_lap_idx += 1
			return self.waypoints[self.current_idx]
		if self.current_idx >= len(self.waypoints) and self.current_lap_idx >= self.total_num_laps:
			return None
		if self.current_idx < len(self.waypoints):
			return self.waypoints[self.current_idx]

		return None 

	def check_and_advance(self, current_x: float, current_y: float) -> bool:
		target = self.get_current_target()
		if not target:
			return False

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
		self.state_info = np.full(6, np.nan)
		self.control_info = np.full(3, 0.0)
		
		# Publishers & Subscribers
		self.traj_pub = self.create_publisher(CtlTraj, '/trajectory', self.pub_freq)
		
		self.state_sub = self.create_subscription(
			Odometry,
			'mavros/local_position/odom',
			self.mavros_state_callback,
			qos_profile=SENSOR_QOS
		)

		# Home Position Subscriber (used as Cartesian Origin)
		self.home_lat = None
		self.home_lon = None
		# self.home_sub = self.create_subscription(
		#     HomePosition, 
		#     'mavros/home_position/home', 
		#     self.home_callback, 
		#     10
		# )

		# Waypoint Pull Client & Subscriber
		self.wp_client = self.create_client(WaypointPull, 'mavros/mission/pull')
		self.wp_sub = self.create_subscription(
			WaypointList,
			'mavros/mission/waypoints',
			self.waypoints_callback,
			10
		)
		self.mission_waypoints = []
		self.waypoints_received = False

	def home_callback(self, msg: HomePosition) -> None:
		if self.home_lat is None:
			self.home_lat = msg.geo.latitude
			self.home_lon = msg.geo.longitude
			self.get_logger().info(f"Home Position Locked: Lat {self.home_lat}, Lon {self.home_lon}")

	def request_waypoint_pull(self) -> None:
		while not self.wp_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('WaypointPull service not available, waiting again...')
		
		req = WaypointPull.Request()
		self.future = self.wp_client.call_async(req)
		self.future.add_done_callback(self.pull_response_callback)

	def pull_response_callback(self, future) -> None:
		try:
			response = future.result()
			if response.success:
				self.get_logger().info(f"Successfully pulled {response.wp_received} WPs from FCU.")
			else:
				self.get_logger().error("Failed to pull waypoints from FCU.")
		except Exception as e:
			self.get_logger().error(f"Service call failed: {e}")

	def waypoints_callback(self, msg: WaypointList) -> None:
		if not self.waypoints_received and len(msg.waypoints) > 0:
			for i, w in enumerate(msg.waypoints):
				w: Waypoint
				print("w", w)
				if i == 0:
					self.home_lat_dg = w.x_lat
					self.home_lon_dg = w.y_long
				else:
					self.mission_waypoints.append(w)
			print("waypoints are", self.mission_waypoints)
			self.waypoints_received = True
			self.get_logger().info(f"Received {len(msg.waypoints)} waypoints from MAVROS topic.")

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
						final_target:np.array = None,
						idx_buffer: int = 1) -> None:
		states = solution['states']
		controls = solution['controls']
		
		idx_step = self.get_time_idx(mpc_params, delta_sol_time, idx_buffer)
		max_idx = len(states['x']) - 1
		idx_step = min(idx_step, max_idx)
		idx_step = -1

		v_cmd_mps = 23.0
		psi_cmd_rad_enu = states['psi'][idx_step]

		if final_target is None:
			x_pred = states['x'][idx_step]
			y_pred = states['y'][idx_step]
		else:
			x_pred = final_target[X_IDX]
			y_pred = final_target[Y_IDX]

		airspeed = self.state_info[V_IDX]
		safe_airspeed = max(airspeed, 1.0)

		max_bank_deg = 45.0 
		max_yaw_rate_rad_s = compute_max_yaw_rate(
			max_bank_angle_rad=np.deg2rad(max_bank_deg),
			airspeed=safe_airspeed)
		
		look_ahead_s:float = 1.5
		max_turn_threshold = max_yaw_rate_rad_s * look_ahead_s
		
		raw_target_heading = self.compute_los(x_pred, y_pred)
		current_heading = self.state_info[PSI_IDX]

		heading_error = np.arctan2(np.sin(raw_target_heading - current_heading),
								   np.cos(raw_target_heading - current_heading))
		
		clipped_error = np.clip(heading_error, -max_turn_threshold, max_turn_threshold)
		
		attempted_yaw_rate:float = clipped_error/look_ahead_s
		predicted_roll_rad = compute_bank_angle(yaw_rate=attempted_yaw_rate,
			airspeed=safe_airspeed)
			
		psi_cmd_rad_enu = current_heading + clipped_error
		vz_cmd_mps_enu = controls['vz_cmd'][idx_step]
		
		psi_cmd_rad_ned = convert_yaw_enu_to_ned(psi_cmd_rad_enu)
		psi_cmd_rad_ned = wrap_to_2pi(psi_cmd_rad_ned) 
		psi_cmd_deg_ned = math.degrees(psi_cmd_rad_ned)

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
	
	wp_manager = MissionManager(acceptance_radius_m=5.0)
	traj_node = EngageTrajNodeOuter()

	# --- 1. PULL AND WAIT FOR WAYPOINTS ---
	traj_node.request_waypoint_pull()
	traj_node.get_logger().info("Waiting to receive waypoints from MAVROS...")
	while rclpy.ok() and not traj_node.waypoints_received:
		rclpy.spin_once(traj_node, timeout_sec=0.1)

	# --- 2. SET CARTESIAN ORIGIN FROM WP 0 ---
	if len(traj_node.mission_waypoints) < 2:
		traj_node.get_logger().error("Mission requires at least WP 0 (Home) and one target WP. Exiting.")
		return

	origin_lat = traj_node.home_lat_dg
	origin_lon = traj_node.home_lon_dg
	traj_node.get_logger().info(f"Using WP 0 as Cartesian Origin: Lat {origin_lat:.5f}, Lon {origin_lon:.5f}")

	# --- 3. CONVERT AND LOAD REMAINING WAYPOINTS ---
	default_cruise_speed = 23.0
	valid_wps_loaded = 0

	# Slice the list [1:] to skip the home waypoint
	for wp in traj_node.mission_waypoints[1:]:
		if wp.command == 16: # NAV_WAYPOINT
			# Project Geodetic coordinates to Local Cartesian relative to WP 0
			local_x, local_y = DroneMath.geodetic_to_cartesian(
				origin_lat=origin_lat, 
				origin_lon=origin_lon, 
				target_lat=wp.x_lat, 
				target_lon=wp.y_long
			)
			
			alt = wp.z_alt
			wp_manager.add_waypoint(local_x, local_y, alt, default_cruise_speed)
			valid_wps_loaded += 1
			traj_node.get_logger().info(f"Loaded WP {valid_wps_loaded}: Local(x={local_x:.1f}, y={local_y:.1f}), Alt={alt:.1f}m")

	if valid_wps_loaded == 0:
		pass # Note: Changed 'traj_node' standing alone to 'pass' to prevent a syntax error
		
	# --- 4. INITIALIZE MPC ---
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
		'h':   {'min': 20.0,     'max': 125.0},
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

	closed_loop_sim = CloseLoopSim(
		optimizer=plane_opt_control,
		x_init=traj_node.state_info,
		x_final=np.zeros(6),
		print_every=1000,
		u0=traj_node.control_info,
		N=20
	)

	print("Waiting for MAVROS odometry telemetry...")
	for _ in range(10):
		rclpy.spin_once(traj_node, timeout_sec=0.1)  

	# --- 5. MAIN CONTROL LOOP ---
	while rclpy.ok():
		try:
			rclpy.spin_once(traj_node, timeout_sec=0.01)
			
			if np.any(np.isnan(traj_node.state_info)) or np.any(np.isnan(traj_node.control_info)):
				traj_node.get_logger().info("Waiting for valid MAVROS telemetry...", throttle_duration_sec=2.0)
				continue
			
			current_x = traj_node.state_info[X_IDX]
			current_y = traj_node.state_info[Y_IDX]
			
			if wp_manager.check_and_advance(current_x, current_y):
				traj_node.get_logger().info(f"Waypoint Reached! Advancing to WP {wp_manager.current_idx}")
			
			active_target = wp_manager.get_current_target()
			
			if not active_target:
				traj_node.get_logger().info("Mission Complete. Holding last waypoint.", throttle_duration_sec=5.0)
				break 

			target_x = active_target['x']
			target_y = active_target['y']
			target_alt = active_target['alt']
			v_cruise = active_target['speed']

			los_target = traj_node.compute_los(target_x=target_x, target_y=target_y)
			
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
			traj_node.get_logger().info(f"WP[{wp_manager.current_idx}] Dist: {dist_to_target:.1f}m, Alt Err: {alt_error:.1f}m, Sol: {delta_sol_time*1000:.1f}ms", throttle_duration_sec=1.0)  
			if dist_to_target <= wp_manager.acceptance_radius * 4:
				final_target = xF
			else:
				final_target = None
	
			traj_node.publish_trajectory(solution, delta_sol_time, mpc_params,
								desired_alt_m=xF[2],idx_buffer=1, final_target=final_target)

		except KeyboardInterrupt:
			traj_node.get_logger().info('Keyboard Interrupt: Shutting Down Node')
			break

	traj_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()