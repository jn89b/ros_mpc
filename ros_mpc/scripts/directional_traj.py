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
from ros_mpc.aircraft_config import get_time_idx
from ros_mpc.config import GOAL_X, GOAL_Y, GOAL_Z
from ros_mpc.rotation_utils import (ned_to_enu_states,
                                    yaw_enu_to_ned,
                                    enu_to_ned_states,
                                    euler_from_quaternion,
                                    convert_enu_state_sol_to_ned)

# from optitraj.mpc.PlaneOptControl import PlaneOptControl
from ros_mpc.PlaneOptControl import PlaneOptControl
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim
from apmuas_ros.drone_math import geodetic_to_cartesian, convert_all_to_cartesian

from rl_ros.PID import FirstOrderFilter, PID
from mavros_msgs.srv import WaypointPull
from mavros_msgs.msg import WaypointList
from mavros_msgs.msg import HomePosition, State, WaypointReached
from sensor_msgs.msg import NavSatFix
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
  
import threading
import hashlib

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


def wrap_to_pi(angle: float) -> float:
    """
    Wrap an angle in radians to the range [-pi, pi].

    Parameters:
        angle (float): Angle in radians.

    Returns:
        float: Angle wrapped to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_relative_ned_yaw_cmd(
        current_ned_yaw: float,
        inert_ned_yaw_cmd: float) -> float:

    yaw_cmd: float = inert_ned_yaw_cmd - current_ned_yaw

    # wrap the angle to [-pi, pi]
    return wrap_to_pi(yaw_cmd)



class DirectionalTraj(Node):
    """
    Remember Loose Coupling: 
        - We don't want to tie in MPC into this ROS2 node
        - This node is responsible for:
            - Publishing the trajectory 
            - Subscribing to the aircraft state
    - We will use our own libraries to compute the stuff
    - use this as a template for the other nodes
    - This node will be used to:
        - Subscribe to the aircraft state
        - Publish the trajectory
        - Convert between NED and ENU frames
        - Handle waypoint pulling from MAVROS
    """
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

        # self._lock = threading.RLock()
        self._init_timers()
        self._pull_in_flight = False
        self._last_pull_time = 0.0

      # --- Home / origin & local converter ---
        self.home_received = True
        self.home_lat_dg: float = 0.0
        self.home_lon_dg = float = 0.0
        self.local_cart = None
        self.origin_alt_msl = None  # to handle relative-alt frames
        
        # --- Path store (ENU) from mission waypoints ---
        self.path_gps: List[Tuple[float, float, float]] = []  # list of (lat,lon,alt)
        self.path_enu: List[Tuple[float, float, float]] = []  # list of (Ex,Ey,Ez)
        self.wp_idx:int = 0         # current progress along path
        self.lookahead_m:float = 25.0 # tune this

        # Periodic repull until we get a non-empty mission
        self._pull_timer = self.create_timer(1.0, self._maybe_pull_waypoints)
        self.waypoint_client = self.create_client(WaypointPull, '/mavros/mission/pull')        
        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = WaypointPull.Request()
        self.future = self.waypoint_client.call_async(self.req)
        
        self.dz_controller: PID = PID(
            kp=0.1, ki=0.0, kd=0.01,
            min_constraint=np.deg2rad(-12),
            max_constraint=np.deg2rad(10),
            use_derivative=True,
            dt = 0.025)
        
        # For Ardupilot when we query the waypoints 
        # the first waypoint is the home position
        # this will be used to index the commands
        self.home_waypoint_idx: int = 0 
        
        self.sub = self.create_subscription(
            WaypointList,
            '/mavros/mission/waypoints',
            self.waypoint_cb,
            10
        )
        self.state_sub = self.create_subscription(mavros.local_position.Odometry,
                                                  'mavros/local_position/odom',
                                                  self.mavros_state_callback,
                                                  qos_profile=SENSOR_QOS)
        self.pub_traj = self.create_publisher(
            CtlTraj, '/trajectory', 10)

        self.num_controls: int = 4
        self.current_enu_controls: np.array = np.array(
            [np.nan]*self.num_controls)

    def _init_timers(self)-> None:
        """
        Used to initialize timers for waypoint pulling and other periodic tasks
        """
        self._waiting_for_wps:bool = False
        self._last_wp_hash = None
        self._pull_timeout_s = 5.0
        self._pull_started_at = 0.0
        self._last_path_update = 0.0


    def _cache_gps_waypoints(self, msg: WaypointList) -> List[Tuple[float, float, float]]:
        """
        Cache the waypoints from the message
        Args:
            msg (WaypointList): Waypoint list message
        """
        new_path: List[Tuple[float, float, float]] = []
        for i, w in enumerate(msg.waypoints):
            if i == self.home_waypoint_idx:
                self.home_lat_dg = w.x_lat
                self.home_lon_dg = w.y_long    
                continue  # skip home waypoint. 
            # check if gps coordinate is 0,0
            if w.x_lat == 0.0 and w.y_long == 0.0:
                self.get_logger().warn(f"Skipping invalid waypoint {i} with lat,lon=0,0")
                continue                       
            lat, lon, alt = float(w.x_lat), float(w.y_long), float(w.z_alt)
            print(f"Waypoint: lat={lat}, lon={lon}, alt={alt}, frame={w.frame}, command={w.command}")
            new_path.append((lat, lon, alt))
        
        return new_path

    def _create_cartesian_waypoints(self, 
            gps_waypoints: List[Tuple[float, float, float]],
            minimum_altitude_m:float = 50,
            max_lat_range:float=1000) -> List[Tuple[float, float, float]]:
        """
        Returns a list of Cartesian coordinates (x, y, z) relative to the home position.
        """
        if self.home_lat_dg == None or self.home_lon_dg == None:
            self.get_logger().warn("Home position not set; cannot convert waypoints.")
            return []
        
        cartesian_path: List[Tuple[float, float, float]] = []
        for i, (wp) in enumerate(gps_waypoints):
            lat, lon, alt = wp
            x, y = geodetic_to_cartesian(
                self.home_lat_dg, self.home_lon_dg, lat, lon)
            print(f"Cartesian Waypoint {i}: x={x:.1f}, y={y:.1f}, alt={alt:.1f}")
            if abs(x) > max_lat_range or abs(y) > max_lat_range:
                self.get_logger().warn(f"Waypoint {i}, {wp} too far from home; skipping.")
                continue
            # this is a failsafe to ensure we don't go below a certain altitude
            if alt < minimum_altitude_m:
                alt = minimum_altitude_m
            cartesian_path.append((x, y, alt)) 
        
        return cartesian_path

    def waypoint_cb(self, msg: WaypointList) -> None:
        # QoS should be TRANSIENT_LOCAL so you get latched updates
        new_hash = self._hash_waypoints(msg)
        if new_hash == self._last_wp_hash:
            print("Duplicate waypoint message received; ignoring.")
            # Duplicate (e.g., first latched read or same mission) — ignore
            if self._waiting_for_wps:
                # We initiated a pull but cache didn’t change; let timer retry via timeout
                pass
            return
        
        self._last_wp_hash = new_hash
        self.path_gps: List[Tuple[float, float, float]] = self._cache_gps_waypoints(msg)
        print("GPS Waypoints: ", self.path_gps)
        self.path_enu: List[Tuple[float, float, float]] = self._create_cartesian_waypoints(self.path_gps)
        self.wp_idx = 0
        self._waiting_for_wps = False   # <- we got a fresh mission
        self._last_path_update = time.time()
        self.get_logger().info(f"Waypoints updated: {len(self.path_enu)} points.")
    
    def _hash_waypoints(self, msg: WaypointList) -> str:
        """
        We will use this to hash the waypoints to see if they have changed
        and we will only update the path if they have changed
        This is to avoid unnecessary path updates
        Args:
            msg (WaypointList): Waypoint list message
        Returns:
            str: Hash of the waypoints
        """
        h = hashlib.sha1()
        for w in msg.waypoints:
            h.update(f"{w.frame},{w.command},{w.x_lat:.8f},{w.y_long:.8f},{w.z_alt:.3f}".encode())
        return h.hexdigest()

    def _maybe_pull_waypoints(self) -> None:
        """
        We will use this to periodically pull waypoints if we don't have any
        or if we are waiting for a pull to complete
        """
        now = time.time()
        # If a pull is already in flight and taking too long, time it out
        if self._waiting_for_wps and (now - self._pull_started_at) > self._pull_timeout_s:
            self.get_logger().warn("Mission pull timed out; retrying.")
            self._waiting_for_wps = False  # allow a new attempt

        # Only pull if we have no path yet OR we're explicitly waiting and timed out, OR you detect staleness elsewhere
        if self._waiting_for_wps:
            return

        # Trigger pull
        if self.waypoint_client.service_is_ready():
            self._waiting_for_wps = True
            self._pull_started_at = now
            fut = self.waypoint_client.call_async(WaypointPull.Request())

            def _done_cb(f):
                try:
                    res = f.result()
                    self.get_logger().info(f"pull success={getattr(res,'success',True)} count={getattr(res,'wp_received',-1)}")
                except Exception as e:
                    self.get_logger().warn(f"pull exception: {e}")
                    self._waiting_for_wps = False  # allow retry
            fut.add_done_callback(_done_cb)
            

    def _home_cb(self, msg: HomePosition) -> None:
        # with self._lock:
        lat0 = msg.geo.latitude 
        lon0 = msg.geo.longitude 
        alt0 = msg.geo.altitude
        print("Home lat, lon, alt: ", lat0, lon0, alt0)
        # self.local_cart = LocalCartesian(lat0, lon0, alt0)
        self.origin_alt_msl = alt0
        self.home_received = True
        self.get_logger().info("Home set...")
        self._pull_in_flight = False
        
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
        traj_msg.z = ned_states['z'].tolist()
        traj_msg.roll = ned_states['phi'].tolist()
        dz:float = GOAL_Z - self.enu_state[2]
        if self.dz_controller.prev_error is None:
            self.dz_controller.prev_error = 0.0
            
        pitch_cmd = self.dz_controller.compute(
            setpoint=dz,
            current_value=0.0,
            dt=0.05
        )
        pitch_cmd = np.clip(pitch_cmd, -np.deg2rad(10), np.deg2rad(10))
        pitch_array = np.ones(len(ned_states['theta'])) * pitch_cmd
        traj_msg.pitch = pitch_array.tolist()
        current_ned_yaw: float = yaw_enu_to_ned(self.enu_state[5])
        # traj_msg.yaw = ned_states['psi'].tolist()
        ned_yaw = ned_states['psi'].tolist()
        ned_yaw_cmd = [get_relative_ned_yaw_cmd(
            current_ned_yaw, ned_yaw[i]) for i in range(len(ned_yaw))]
        traj_msg.yaw = ned_yaw_cmd

        traj_msg.vx = ned_states['v'].tolist()
        traj_msg.idx = time_idx + 1

        airspeed_error = states['v'][time_idx] - self.enu_state[6]        
        kp_airspeed:float = 0.25
        airspeed_cmd:float = kp_airspeed * airspeed_error
        min_thrust:float = 0.4
        max_thrust:float = 0.7
        thrust_cmd:float = np.clip(
            airspeed_cmd, min_thrust, max_thrust)
        thrust_cmd = np.ones(len(ned_states['v'])) * thrust_cmd
        traj_msg.thrust = thrust_cmd.tolist() 

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
    R: np.diag = np.diag([0.25, 0.25, 0.25, 1])

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
    xF: np.array = np.array([GOAL_X, GOAL_Y, GOAL_Z, 0, 0, 0, 20])
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
            distance = np.linalg.norm(
                np.array(traj_node.enu_state[0:2]) - np.array(xF[0:2]))
            # print("Distance: ", distance)
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