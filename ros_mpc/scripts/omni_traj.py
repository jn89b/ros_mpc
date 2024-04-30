#!/usr/bin/env python3

import casadi as ca
import rclpy
import numpy as np
import time 

from drone_interfaces.msg import Telem, CtlTraj
from rclpy.node import Node

from ros_mpc.PlaneOptControl import PlaneOptControl
from ros_mpc.Effector import Effector
from ros_mpc.aircraft_config import mpc_params, omni_effector_config, \
    control_constraints, state_constraints, obs_avoid_params
from ros_mpc.aircraft_config import GOAL_STATE, DONE_STATE, RADIUS_TARGET

import ros_mpc.rotation_utils as rot_utils
from ros_mpc.models.Plane import Plane
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

import ros_mpc.avoidance_tools as avoid_tools
import gpiod 
import mavros
from mavros.base import SENSOR_QOS

# this is very stupid but works for now 
USE_LED = True
if USE_LED:
    LED_PIN = 3
    chip = gpiod.Chip('../dev/gpiochip4')
    led_line = chip.get_line(LED_PIN)
    led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)

"""
- Set Effector Configurations
- Set Aircraft Configurations
- Set MPC Configurations

- Initialize Aircraft Model and 

"""

def to_avoid(current_position:np.ndarray, current_heading:float,
             obstacles:np.ndarray, r_threshold:float, 
             K:int=5, 
             dot_product_threshold:float=0.5, 
             distance_buffer_m:float=10, 
             use_nearest_ref:bool=True,
             ref_point:np.ndarray=None) -> tuple:
    
    nearest_obstacles,nearest_indices  = avoid_tools.knn_obstacles(
        obstacles, current_position, K=K, use_2d=True)

    #check if list is empty
    if len(nearest_obstacles) == 0:
        return False, None
    
    ego_unit_vector = np.array([np.cos(current_heading), np.sin(current_heading)])

    inline_obs, dot_products = avoid_tools.find_inline_obstacles(
        ego_unit_vector, 
        nearest_obstacles, current_position, 
        dot_product_threshold=dot_product_threshold,
        use_2d=True)
 
    if len(inline_obs) == 0:
        return False, None
 
    danger_zones,dot_product = avoid_tools.find_danger_zones(
        inline_obs, current_position, r_threshold, dot_products, 
        distance_buffer=distance_buffer_m, use_2d=True)
    
    # print("danger_zones:", danger_zones)	
 
    if len(danger_zones) == 0:
        return False, None
 
    # dot_product = [d[1] for d in danger_zones]
    # danger_zones = [d[0] for d in danger_zones]
    radius_obs = [d[-1] for d in danger_zones]    
    
    max_dot_obs = np.argmax(dot_product)
    #get max radius of obstacle
    max_radius = max(radius_obs)
    
    # if max_dot_obs < dot_product_threshold:
    #     return False, None
    max_dot_obs = danger_zones[max_dot_obs]
    robot_radius = obs_avoid_params['safe_distance'] + max_radius + distance_buffer_m
    driveby_position = avoid_tools.find_driveby_direction(
                                                        goal_position=max_dot_obs[:2], 
                                                        current_position=current_position[:2], 
                                                        heading_rad=current_heading,
                                                        obs_radius=max_radius, 
                                                        robot_radius=robot_radius, 
                                                        consider_obstacles=True, \
                                                        danger_zones=inline_obs,
                                                        use_nearest_ref=use_nearest_ref, 
                                                        ref_point=ref_point)
 
 
    return True, driveby_position

class OmniTrajNode(Node):
    def __init__(self, 
                 pub_freq:int=100, 
                 sub_freq:int=100,
                 save_states:bool=False,
                 sub_to_mavros:bool=True):
        super().__init__('omni_traj_fw_publisher')
        self.get_logger().info('Starting Omni Traj FW Publisher')
        
        self.pub_freq = pub_freq
        self.sub_freq = sub_freq
        
        #flag this to save states and cache it for later if needed
        self.save_states = save_states
        
        self.state_info =[
            None, #x
            None, #y
            None, #z
            None, #phi
            None, #theta
            None, #psi
            None, #airspeed
        ]
        
        self.control_info = [
            None, #u_phi
            None, #u_theta
            None, #u_psi
            None  #v_cmd
        ]

        
        self.traj_pub = self.create_publisher(
            CtlTraj, 
            'omni_trajectory', 
            self.pub_freq)
        
        if sub_to_mavros:
            self.state_sub = self.create_subscription(mavros.local_position.Odometry,
                                                    'mavros/local_position/odom', 
                                                    self.mavros_state_callback, 
                                                    qos_profile=SENSOR_QOS)
        else:        
            self.state_sub = self.create_subscription(Telem, 
                'telem', 
                self.state_callback, 
                self.sub_freq)


        self.cost_pub = self.create_publisher(Float64, 'waypoint_cost_val', 50)
        self.time_sol_pub = self.create_publisher(Float64, 'waypoint_time_sol', 50)
        self.driveby_pos_pub = self.create_publisher(PoseStamped, 
                                                     'driveby_position', 50)

    def publish_driveby_position(self, driveby_position:np.ndarray) -> None:
        driveby_msg = PoseStamped()
        driveby_msg.pose.position.x = driveby_position[0]
        driveby_msg.pose.position.y = driveby_position[1]
        driveby_msg.pose.position.z = driveby_position[2]
        self.driveby_pos_pub.publish(driveby_msg)

    def init_history(self) -> None:
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.phi_history = []
        self.theta_history = []
        self.psi_history = []
        self.u_phi_history = []
        self.u_theta_history = []
        self.u_psi_history = []
        self.v_cmd_history = []
        
    def mavros_state_callback(self, msg:mavros.local_position.Odometry) -> None:
        """
        Converts NED to ENU and publishes the trajectory
          """
        self.state_info[0] = msg.pose.pose.position.x
        self.state_info[1] = msg.pose.pose.position.y
        self.state_info[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = rot_utils.euler_from_quaternion(
            qx, qy, qz, qw)

        self.state_info[3] = roll
        self.state_info[4] = pitch
        self.state_info[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        #get magnitude of velocity
        self.state_info[6] = np.sqrt(vx**2 + vy**2 + vz**2)
        #self.state_info[6] = #msg.twist.twist.linear.x
        self.control_info[0] = msg.twist.twist.angular.x
        self.control_info[1] = msg.twist.twist.angular.y
        self.control_info[2] = msg.twist.twist.angular.z
        self.control_info[3] = msg.twist.twist.linear.x
    
        if self.save_states:
            self.x_history.append(self.state_info[0])
            self.y_history.append(self.state_info[1])
            self.z_history.append(self.state_info[2])
            self.phi_history.append(self.state_info[3])
            self.theta_history.append(self.state_info[4])
            self.psi_history.append(self.state_info[5])
            self.u_phi_history.append(self.control_info[0])
            self.u_theta_history.append(self.control_info[1])
            self.u_psi_history.append(self.control_info[2])
            self.v_cmd_history.append(self.control_info[3])


    def state_callback(self, msg:Telem) -> None:
        
        enu_coords = rot_utils.convertNEDToENU(
            msg.x, msg.y, msg.z)
        # positions
        self.state_info[0] = enu_coords[0]
        self.state_info[1] = enu_coords[1]
        self.state_info[2] = enu_coords[2]

        #wrap yaw to 0-360
        self.state_info[3] = msg.roll
        self.state_info[4] = msg.pitch
        self.state_info[5] = msg.yaw #flip the yaw to match ENU frame
        self.state_info[6] = np.sqrt(msg.vx**2 + msg.vy**2 + msg.vz**2)

        #rotate roll and pitch rates to ENU frame   
        self.control_info[0] = msg.roll_rate
        self.control_info[1] = msg.pitch_rate
        self.control_info[2] = msg.yaw_rate
        self.control_info[3] = np.sqrt(msg.vx**2 + msg.vy**2 + msg.vz**2) 

    def publish_trajectory(self, solution_results:dict, idx_step:int) -> None:
        x = solution_results['x']
        y = solution_results['y']
        z = solution_results['z']
        phi = solution_results['phi']
        theta = -solution_results['theta']#have to flip sign to match NED to ENU
        psi = solution_results['psi']
        
        u_phi = solution_results['u_phi']
        u_theta = -solution_results['u_theta']#have to flip sign to match NED to ENU
        u_psi = solution_results['u_psi']
        v_cmd = solution_results['v_cmd']
        
        x_ned = y
        y_ned = x
        z_ned = -z
        
        traj_msg = CtlTraj()
        #make sure its a list
        traj_msg.x = x_ned.tolist()
        traj_msg.y = y_ned.tolist()
        traj_msg.z = z_ned.tolist()
        traj_msg.roll = phi.tolist()
        traj_msg.pitch = theta.tolist()
        traj_msg.yaw = psi.tolist()
        traj_msg.roll_rate = u_phi.tolist()
        traj_msg.pitch_rate = u_theta.tolist()
        traj_msg.yaw_rate = u_psi.tolist()
        traj_msg.vx = v_cmd.tolist()

        traj_msg.idx = idx_step
                
        self.traj_pub.publish(traj_msg)

        if 'cost' in solution_results:
                cost_val = solution_results['cost']
                self.publish_cost(float(cost_val))


    def get_time_idx(self, mpc_params:dict, 
                     solution_time:float, idx_buffer:int=0) -> int:
        time_rounded = round(solution_time, 1)
        
        if time_rounded <= 1:
            time_rounded = 1
        
        ctrl_idx = mpc_params['dt']/time_rounded
        idx = int(round(ctrl_idx)) + idx_buffer
        
        return idx

    def publish_cost(self, cost:float) -> None:
        cost_msg = Float64()
        cost_msg.data = cost
        self.cost_pub.publish(cost_msg)
  
    def publish_time(self, time_sol:float) -> None:
        time_msg = Float64()
        time_msg.data = time_sol
        self.time_sol_pub.publish(time_msg)

def main(args=None) -> None:
    rclpy.init(args=args)    

    traj_node = OmniTrajNode()
    rclpy.spin_once(traj_node)
    plane = Plane()
 
    Q_val = 1E-2
    R_val = 1E-2
    omni_mpc_params = {
		'N': 15,
        'Q': ca.diag([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, Q_val]),
        'R': ca.diag([R_val, R_val, R_val, 0.1]),
		'dt': 0.1
	}
    
    plane_mpc = PlaneOptControl(
        control_constraints=control_constraints,
        state_constraints=state_constraints,
        mpc_params=omni_mpc_params,
        casadi_model=plane,
        use_pew_pew=True,
        pew_pew_params=omni_effector_config,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )
    plane_mpc.init_optimization_problem()

    #TODO: this is a hack, using another mpc formulation in case we're close to obstacles 
    # and need to avoid them resend a new trajectory in case we're close to obstacles
 
    avoid_mpc = PlaneOptControl(
        control_constraints=control_constraints,
        state_constraints=state_constraints,
        mpc_params=mpc_params,
        casadi_model=plane,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )
    avoid_mpc.init_optimization_problem()

    counter = 1  
    print_every = 10

    goal = GOAL_STATE
    idx_buffer = 2
    
    distance_tolerance = 5.0
    
    solution_results,end_time = plane_mpc.get_solution(traj_node.state_info, 
                                                        goal, traj_node.control_info,
                                                        get_cost=True)

    obs_x = obs_avoid_params['x']
    obs_y = obs_avoid_params['y']
    obs_z = obs_avoid_params['z']
    obs_radii = obs_avoid_params['radii']

    obs_x = np.append(obs_x, GOAL_STATE[0])
    obs_y = np.append(obs_y, GOAL_STATE[1])
    obs_z = np.append(obs_z, GOAL_STATE[2])
    obs_radii = np.append(obs_radii, RADIUS_TARGET)
        
    obstacles = np.array([obs_x, obs_y, obs_z, obs_radii]).T    
    min_velocity = state_constraints['airspeed_min'] + 1
    max_velocity = state_constraints['airspeed_max']
    max_phi = control_constraints['u_phi_max']
    max_psi = control_constraints['u_psi_max']
    
    min_radius = min_velocity**2 / (9.81*np.tan(max_phi))
 
    MISSION_COMPLETE = False
    goal_ref = GOAL_STATE
    JUST_AVOIDED = False
    
    CRASH = False
    
    while rclpy.ok():
        rclpy.spin_once(traj_node)

        start_time = time.time()

        distance_error = np.sqrt(
            (goal_ref[0] - traj_node.state_info[0])**2 + 
            (goal_ref[1] - traj_node.state_info[1])**2 
        )      
        
        #curr_pos = np.array([traj_node.state_info[0], traj_node.state_info[1]])
        curr_pos = np.array([solution_results['x'][idx_buffer], 
                             solution_results['y'][idx_buffer]], 
                             solution_results['z'][idx_buffer])
        
        start_time = time.time()
        if distance_error <= omni_effector_config['effector_range']:
            threshold = min_radius/1.5
            dot_product_threshold = 0.8
        else:
            threshold = min_radius/2.0
            dot_product_threshold = 0.6
            
        avoid, driveby_position = to_avoid(
            current_position=curr_pos,
            current_heading=traj_node.state_info[5],
            obstacles=obstacles,
            r_threshold=min_radius,
            dot_product_threshold=dot_product_threshold ,
            K=10,
            distance_buffer_m=threshold,
            use_nearest_ref=True,
            ref_point=goal_ref[:2])
        end_time = time.time()
  
        if avoid:
            print("Avoiding Obstacle")
            #TODO: might need to put this somewhere else and make it faster? 
            # profile performance of code and write in C++ if needed/Cython
            # print("avoid:", driveby_position, "time:", end_time-start_time)
            #compute phi desired
            #take cross product of the current heading and the driveby position
            ego_unit_vector = np.array([np.cos(traj_node.state_info[5]), 
                                        np.sin(traj_node.state_info[5])])
            los_to_driveby = curr_pos - driveby_position
            los_unit_driveby = los_to_driveby / np.linalg.norm(los_to_driveby)
            cross_product = np.cross(ego_unit_vector, los_unit_driveby)
            
            #abstract this to a method 
            #if negative turn right
            if cross_product < 0:
                #phi_desired = -max_phi
                phi_multiplier = -1.0
            else:
                #phi_desired = max_phi
                phi_multiplier = 1.0
                
            dlat = np.linalg.norm(driveby_position - curr_pos)
            dz = goal_ref[2] - traj_node.state_info[2]
            R = np.sqrt(dlat**2 + dz**2)
            phi = np.arctan(max_velocity**2 / (9.81*R))
            # phi_desired = phi * phi_multiplier
            #phi_desired = max_phi * phi_multiplier
            phi_desired = -9.81 * np.tan(phi) / max_velocity**2           
            yaw_desired = max_psi * phi_multiplier
            
            z_ref = traj_node.state_info[2]
            goal = [driveby_position[0], driveby_position[1], goal[2],
                                phi_desired, 
                                traj_node.state_info[4], 
                                yaw_desired, 
                                min_velocity]
            solution_results, end_time = avoid_mpc.get_solution(
                traj_node.state_info, goal, traj_node.control_info, get_cost=True)
            
            # JUST_AVOIDED = True
            
        else:
            goal = [goal_ref[0], goal_ref[1], goal_ref[2], 
                    solution_results['phi'][idx_buffer], 
                    solution_results['theta'][idx_buffer], 
                    np.deg2rad(-270),
                    solution_results['v_cmd'][idx_buffer]]
        
            solution_results,end_time = plane_mpc.get_solution(traj_node.state_info, 
                                                            goal, traj_node.control_info,
                                                            get_cost=True)
            JUST_AVOIDED = False
        
        # delta_time = time.time() - start_time
        idx_step = traj_node.get_time_idx(mpc_params, end_time - start_time, idx_buffer)
        traj_node.publish_time(end_time - start_time)
        if counter % print_every == 0:
            print('Distance Error: ', distance_error)

        if distance_error <= 40:
            led_line.set_value(1)
    
        if distance_error <= distance_tolerance:
            traj_node.get_logger().info('Goal Reached Shutting Down Node') 
            goal_ref = DONE_STATE
            led_line.set_value(0)
            led_line.release()
            # traj_node.destroy_node()
            # rclpy.shutdown()
            # return    
        
        traj_node.publish_trajectory(solution_results, idx_step)
        counter += 1
  
if __name__=='__main__':
    main()