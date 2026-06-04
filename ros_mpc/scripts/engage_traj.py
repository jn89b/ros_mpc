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

	def publish_trajectory(self, solution: dict, delta_sol_time: float, mpc_params: MPCParams, idx_buffer: int = 1) -> None:
		"""Parses CloseLoopSim nested dict and publishes the kinematic trajectory."""
		states = solution['states']
		controls = solution['controls']
		
		idx_step = self.get_time_idx(mpc_params, delta_sol_time, idx_buffer)
		max_idx = len(states['x']) - 1
		idx_step = min(idx_step, max_idx)
		idx_step = 5
		# idx_step = -1

		# 1. Extract MPC optimal controls (These are in ENU!)
		v_cmd_mps = controls['V_cmd'][idx_step]
		psi_cmd_rad_enu = controls['psi_cmd'][idx_step]
		#print("psi cmd enu", np.rad2deg(psi_cmd_rad_enu))
		vz_cmd_mps_enu = controls['vz_cmd'][idx_step]
		z_traj = 75.0#states['h'][-1] # use this as a projection reference
		#print("height", states['h'])
		# 2. Coordinate Frame Conversions (ENU -> NED)
		# Shift origin by 90 deg (pi/2) and invert rotation
		psi_cmd_rad_ned = (math.pi / 2.0) - psi_cmd_rad_enu
		
		# Wrap strictly to [0, 2*pi)
		psi_cmd_rad_ned = psi_cmd_rad_ned % (2.0 * math.pi)
		
		# Convert to degrees for ArduPilot
		psi_cmd_deg_ned = math.degrees(psi_cmd_rad_ned)

		# Apply the ArduPilot cm/s bug fix
		# Note: ArduPilot's GUIDED_CHANGE_ALTITUDE treats positive climb rate as UP.
		# Since ENU Z is also UP, the sign for vz does NOT need to be flipped here.
		vz_cmd_cms = vz_cmd_mps_enu * 100.0

		# Publish the fully converted NED targets
		traj_msg = CtlTraj()
		traj_msg.vx = [v_cmd_mps] * 10
		traj_msg.z = [z_traj] * 10
		traj_msg.yaw = [psi_cmd_deg_ned] * 10
		traj_msg.vz = [vz_cmd_cms] * 10 
		traj_msg.idx = 4

		self.traj_pub.publish(traj_msg)
		
		# Update current controls tracker for the next MPC step (u0)
		# Note: Feed the ENU values back into the MPC, not the NED values!
		self.control_info[U_V_IDX] = v_cmd_mps
		self.control_info[U_PSI_IDX] = psi_cmd_rad_enu
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
	target_y = 200.0
	target_alt = 75.0
	v_cruise = 21.0

	# Initialize the 6-state Guidance Model with ArduPilot Time Constants
	plane_model = ArduPlaneGuidanceModel(dt_val=dt, tau_V=0.5, tau_psi=0.3, 
                                      tau_z=0.5)
	
	# Set physical bounds for a typical fixed-wing
	plane_model.set_control_limits({
		'V_cmd':   {'min': 20.0, 'max': 25.0},
		'psi_cmd': {'min': -np.pi, 'max': np.pi},
		'vz_cmd':  {'min': -1.5, 'max': 1.0} # SI limits (m/s)
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
	Q_matrix = np.array([1.0, 1.0, 1.0, 1.0, 0.5]) 
	R_matrix = np.array([1.0, 1.0, 5.0]) # Heavy penalty on heading/climb slew
	
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
			xF = np.array([target_x, target_y, target_alt, v_cruise, 0.0, 0.0])
			
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

			traj_node.publish_trajectory(solution, delta_sol_time, mpc_params, idx_buffer=1)

		except KeyboardInterrupt:
			traj_node.get_logger().info('Keyboard Interrupt: Shutting Down Node') 
			break

	traj_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()