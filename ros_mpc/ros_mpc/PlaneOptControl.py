from ros_mpc.OptimalControl import OptimalControlProblem
from ros_mpc.Effector import Effector
import casadi as ca
import numpy  as np
import time

class PlaneOptControl(OptimalControlProblem):
	def __init__(self, 
				 control_constraints:dict, 
				 state_constraints:dict, 
				 mpc_params:dict, casadi_model,
				 use_obstacle_avoidance:bool=False,
				 obs_params:dict=None,
				 use_dynamic_threats:bool=False,
				 dynamic_threat_params:dict=None, 
				 use_pew_pew:bool=False,
				 pew_pew_params:dict=None,
				 use_time_constraints:bool=False,
				 time_constraint_val:float=None,
				 use_time_intervals:bool=False) -> None:
		
		super().__init__(mpc_params, casadi_model)
		self.control_constraints = control_constraints
		self.state_constraints = state_constraints
		# self.update_bound_constraints()
		self.cost = 0
		
		self.use_obstacle_avoidance = use_obstacle_avoidance
		self.obs_params = obs_params
		
		self.use_dynamic_threats = use_dynamic_threats
		self.dynamic_threat_params = dynamic_threat_params
		
		self.Effector = None
		self.use_pew_pew = use_pew_pew
		self.pew_pew_params = pew_pew_params
		if self.use_pew_pew:
			self.Effector = Effector(self.pew_pew_params, use_casadi=True)
			if self.Effector.effector_type == 'omnidirectional':
				#we need to add an additional parameter P to the optimization problem
				self.driveby_location = ca.SX.sym('driveby_location', 3)

		#time constraint optimization
		#user sets some time, convert that to index and use that as a constraint
		self.use_time_constraints = use_time_constraints
		self.time_constraint_val = time_constraint_val
		self.initialize_time_constraints()
		#this parameter allows you to use time intervals for the optimization problem
		self.use_time_intervals = use_time_intervals 
		self.is_initialized = False
		
		self.gaussian_fn = self.gaussian_curve_function_casadi()
		
	def gaussian_curve_function_casadi(self, a:float=1, 
									   b:float=0, c:float=1/3) -> ca.SX:
		"""
		returns the symbolic representation of a gaussian curve
		a = peak height of gaussian curve default is 1
		b = center of the peak default is 0
		c = standard deviation default is 1/3
		"""
		x = ca.SX.sym('x')
		gaussian_eqn = a * ca.exp(-(x - b)**2 / (2 * c**2))
		gaussian_fn = ca.Function('gaussian_function', [x], [gaussian_eqn],
								['x'], ['y'])
		
		return gaussian_fn
	
	def reinit_start_goal(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray) -> None:
		"""reinitialize the start and goal states"""
		self.x0 = x0
		self.xF = xF
		self.u0 = u0
		
	
	def initialize_time_constraints(self) -> None:
		if self.use_time_constraints:
			if self.time_constraint_val is None:
				#set the constraint as the terminal point of the vehicle
				self.time_constraint_val = self.N*self.dt 
				self.time_constraint_idx = self.N
				print('Time constraint value not set, using default value:', self.time_constraint_val) 
			else:
				#make sure time constraint index is within the range of the vehicle
				self.time_constraint_val = self.time_constraint_val
				if self.time_constraint_val < 0:
					raise ValueError('Time constraint value must be greater than zero', 
									 self.time_constraint_val)
				if self.time_constraint_val > self.N*self.dt:
					raise ValueError('Time constraint value is greater than the total time of the vehicle', 
									 self.time_constraint_val, self.N*self.dt)
					
				#convert the time constraint to an index
				self.time_constraint_idx = int(self.time_constraint_val/self.dt)
				if self.time_constraint_idx > self.N:
					raise ValueError('Time constraint index is greater than the number of steps', 
									 self.time_constraint_idx, self.N)
					
				print("Time constraint value is set to:", self.time_constraint_val)
				print("Time constraint index is set to:", self.time_constraint_idx) 
					
	def update_bound_constraints(self) -> None:
		#add control constraints
		self.lbx['U'][0,:] = self.control_constraints['u_phi_min']
		self.ubx['U'][0,:] = self.control_constraints['u_phi_max']

		self.lbx['U'][1,:] = self.control_constraints['u_theta_min']
		self.ubx['U'][1,:] = self.control_constraints['u_theta_max']

		self.lbx['U'][2,:] = self.control_constraints['u_psi_min']
		self.ubx['U'][2,:] = self.control_constraints['u_psi_max']

		self.lbx['U'][3,:] = self.control_constraints['v_cmd_min']
		self.ubx['U'][3,:] = self.control_constraints['v_cmd_max']

		### State Constraints
		self.lbx['X'][2,:] = self.state_constraints['z_min']
		self.ubx['X'][2,:] = self.state_constraints['z_max']

		self.lbx['X'][3,:] = self.state_constraints['phi_min']
		self.ubx['X'][3,:] = self.state_constraints['phi_max']

		self.lbx['X'][4,:] = self.state_constraints['theta_min']
		self.ubx['X'][4,:] = self.state_constraints['theta_max']
		
		self.lbx['X'][5,:] = self.state_constraints['psi_min']
		self.ubx['X'][5,:] = self.state_constraints['psi_max']
		
		# self.lbx['X'][6,:] = self.state_constraints['airspeed_min']
		# self.ubx['X'][6,:] = self.state_constraints['airspeed_max']
		
		print('Bound constraints updated')

	def compute_dynamics_cost(self) -> ca.SX:
		"""compute the cost function"""
		n_states = self.model_casadi.n_states
		Q = self.Q
		R = self.R
		P = self.P
		x_final = P[n_states:]
		v_cmd = self.U[3, :]
		cost = 0
		if self.use_time_constraints == False:
			print("Computing dynamics cost without time constraint")
			for k in range(self.N):
				states = self.X[:, k]
				controls = self.U[:, k]
				cost += cost \
						+ (states - x_final).T @ Q @ (states - x_final) \
						+ controls.T @ R @ controls
				# cost += cost + controls.T @ R @ controls
										
		#add terminal cost
		else:
			# print('Using a time constraint cost function')
			if self.use_time_intervals:
				# pass
				# print('Using time intervals for cost function')
				for k in range(self.N):
					if k % self.time_constraint_idx == 0:
						states = self.X[:, k]
						controls = self.U[:, k]
						terminal_cost = (self.X[:, k] - x_final).T @ Q @ \
						(self.X[:, k] - x_final)
						#divide cost by velocity
						cost += (terminal_cost / v_cmd[k])
			else:
				time_constraint_idx = self.time_constraint_idx
				terminal_cost = (self.X[:, time_constraint_idx] - x_final).T @ Q @ \
					(self.X[:, time_constraint_idx] - x_final)
				#divide cost by velocity 
				cost += (terminal_cost / v_cmd[-1])
			
		return cost

	def compute_obstacle_avoidance_cost(self) -> ca.SX:
		obs_avoid_weight = self.obs_params['weight']
		obs_x_vector = self.obs_params['x']
		obs_y_vector = self.obs_params['y']
		obs_radii_vector = self.obs_params['radii']
		safe_distance = self.obs_params['safe_distance']
		
		x_position = self.X[0,:]
		y_position = self.X[1,:]
		psi = self.X[5,:]
		total_avoidance_cost = 0
		avoidance_cost = 0
		for i,x in enumerate(obs_x_vector):

			obs_x = obs_x_vector[i]
			obs_y = obs_y_vector[i]
			obs_radii = obs_radii_vector[i]
			# obstacle_constraints = self.opti.variable(self.N+1)
			obstacle_distance = -ca.sqrt((obs_x - x_position)**2 + \
				(obs_y - y_position)**2)
			
			diff = obstacle_distance + obs_radii + safe_distance
			
			#obstacle_distance = 1/obstacle_distance
			avoidance_cost += ca.sum2(diff)
			v_cmd = self.U[3, :]
			
			#want to slow down as we get closer to the obstacle
			avoidance_cost += diff[:-1]/v_cmd
			# los_theta = ca.atan2(obs_y - y_position, obs_x - x_position)
			# diff_psi = -los_theta + psi
			
			# #maximize the heading angle difference
			# avoidance_cost += ca.sum2(diff_psi + diff)
			
			self.g = ca.vertcat(self.g, diff[:-1].T)
			
			### this is used for the omnidirectional effector, we constraint it based on the 
			# radius of the obstacle, robot, and also the major and minor axis of the effector
			if self.use_pew_pew and i == len(obs_x_vector):
				if self.Effector.effector_type == 'omnidirectional':                    
					#constraint to make sure we don't get too close to the target and crash into it
					range_diff = (self.Effector.effector_config['effector_range'] - self.Effector.effector_config['minor_radius'])
					diff = -obstacle_distance + self.pew_pew_params['radius_target'] + safe_distance + range_diff
					self.g = ca.vertcat(self.g, diff[:-1].T)
		
		total_avoidance_cost = obs_avoid_weight * ca.sumsqr(avoidance_cost)
		if self.use_pew_pew:
			total_avoidance_cost = 0
		# else:
		# 	total_avoidance_cost = obs_avoid_weight * ca.sum2(avoidance_cost) 
		
		print('Obstacle avoidance cost computed')
		return total_avoidance_cost
	
	def compute_dynamic_threats_cost(self, cost:float) -> None:
		#ego vehicle
		# print('Computing dynamic threats cost')
		x_pos    = self.X[0, :]
		y_pos    = self.X[1, :]
		z_pos    = self.X[2, :]
		psi      = self.X[5, :]        
		#this is a list of dynamic threats refer to the Threat class
		total_threat_cost = 0
		threat_cost = 0
		threat_list = self.dynamic_threat_params['threats'] 
		for threat in threat_list:
			threat_x_traj = threat.x_traj
			threat_y_traj = threat.y_traj
			# threat_z_traj = threat.z_traj
			threat_psi_traj = threat.psi_traj
			#make sure that the threat trajectory is the same length as the ego vehicle
			if len(threat_x_traj) != self.N+1:
				raise ValueError('Threat trajectory must be the same length as the ego vehicle', 
								 len(threat_x_traj), self.N+1)
		
			if 'time_index' not in self.dynamic_threat_params:
				raise ValueError('Time index not found in dynamic threat parameters')
			
			time_index_check = self.dynamic_threat_params['time_index']
			
			for i in range(self.N):
				
				for j in range(time_index_check):
					idx = j + i
					if idx > self.N:
						break
					
					#calculate the distance between the ego vehicle and the threat
					distance = ca.sqrt((x_pos[i] - threat_x_traj[idx])**2 + \
						(y_pos[i] - threat_y_traj[idx])**2)
					#avoid division by zero
					#check if the distance vector has any negative values
					threat_radii = threat.safe_distance
					ca.if_else(distance>threat_radii, 1, distance)
					#get unit vector of ego vehicle
					u_x = ca.cos(psi[i])
					u_y = ca.sin(psi[i])
					#get unit vector of threat
					u_x_t = ca.cos(threat_psi_traj[idx])
					u_y_t = ca.sin(threat_psi_traj[idx])
					
					#calculate the dot product of the unit vectors
					dot_product = (u_x * u_x_t) + (u_y * u_y_t)
					dot_product_scale = (dot_product*0.5) + 0.5
					
					#if the value of the difference becomes positive that means we are in danger
					#this will set the distance cost to 1, which is a full penalty
					ca.if_else(distance<threat_radii, 1, distance)
					
					#this distance cost mathematically should never be greater than 1 
					distance_cost = 1/distance        
					threat_cost +=  distance_cost + dot_product_scale
					
			total_threat_cost = self.dynamic_threat_params['weight'] * ca.sumsqr(threat_cost)

		return total_threat_cost

	def compute_directional_pew_cost(self) -> ca.SX:
		"""compute the cost function for the directional effector"""
		if self.Effector is None:
			raise ValueError('Effector not initialized')
		
		n_states = self.model_casadi.n_states
		target_location = self.P[n_states:]

		total_effector_cost = 0
		effector_cost = 0
		
		v_max = self.control_constraints['v_cmd_max']
		v_min = self.control_constraints['v_cmd_min']
		k = 0.3#self.Effector.effector_config['k']
  
		for i in range(self.N):
			x_pos = self.X[0, i]
			y_pos = self.X[1, i]
			z_pos = self.X[2, i]
			roll  = self.X[3, i]
			pitch = self.X[4, i]
			yaw   = self.X[5, i]
			v_cmd = self.U[3, i]

			if i == self.N-1:
				v_cmd_next = self.U[3, i]
			else:
				v_cmd_next = self.U[3, i+1]

			###### DIRECTIONAL EFFECTOR COST FUNCTION MAXIMIZE TIME ON TARGET BY SLOWING DOWN APPROACH######
			#right now this is set up for the directional effector
			dx = target_location[0] - x_pos
			dy = target_location[1] - y_pos
			dz = target_location[2] - z_pos
					
			dtarget = ca.sqrt((dx)**2 + (dy)**2 + (dz)**2)
			lat_dist = ca.sqrt((dx)**2 + (dy)**2)
			#los_target = ca.vertcat(dx, dy)
			los_target = ca.atan2(dy, dx)
			  
			#these exponential functions will be used to account for the distance and angle of the target
			#the closer we are to the target the more the distance factor will be close to 1
			error_dist_factor = ca.exp(-k*lat_dist/self.Effector.effector_range)
			#error_dist = dtarget/self.Effector.effector_range
			
			#the closer we are to the target the more the angle factor will be close to 1
			error_psi = ca.fabs(los_target - yaw) #watch out for wrapping angles right ehre
			# error_psi_factor = ca.exp(-k*error_psi/self.Effector.effector_angle)
			
			#the closer we are to the target the more the angle factor will be close to 1
			los_theta = ca.atan2(dz, lat_dist)
			error_theta = ca.fabs(los_theta - pitch) 
			# error_theta_factor = ca.exp(-k*error_theta/self.Effector.effector_angle)  
			
			ratio_distance = (dtarget/self.Effector.effector_range)**2
			ratio_theta    = (error_theta/self.Effector.effector_angle)**2
			ratio_psi      = (error_psi/self.Effector.effector_angle)**2

			effector_dmg   = (ratio_distance + (ratio_theta * ratio_psi))
   
			#this is time on target
			#this velocity penalty will be used to slow down the vehicle as it gets closer to the target
			quad_v_max = (v_cmd - v_max)**2
			#quad_v_min = (v_cmd - v_min)**2
			quad_v_min = (v_cmd - v_min)**2
			vel_penalty = ca.if_else(error_dist_factor <= 0.1, 
									 quad_v_max, quad_v_min)

			#all other controls except for the velocity
			controls_cost = ca.sumsqr(self.U[:3, i])
			
			effector_cost +=  effector_dmg + vel_penalty + controls_cost 
			
			# constraint to make sure we don't get too close to the target and crash into it
			safe_distance = self.obs_params['safe_distance']
			diff = -lat_dist + self.pew_pew_params['radius_target'] + safe_distance 
			self.g = ca.vertcat(self.g, diff)
								
		total_effector_cost = self.pew_pew_params['weight'] * ca.sum2(effector_cost) #+ time_cost
		
		return total_effector_cost
	
	def compute_time_on_target_cost(self) -> ca.SX:
		"""
		General idea is if I'm off target speed up
		if I'm on target slow down
		"""
		n_states = self.model_casadi.n_states
		target_location = self.P[n_states:]
		
		v_max = self.control_constraints['v_cmd_max']
		v_min = self.control_constraints['v_cmd_min']
		
		total_tot_cost = 0
		tot_cost = 0
		
		for i in range(self.N):
			x_pos = self.X[0, i]
			y_pos = self.X[1, i]
			z_pos = self.X[2, i]
			v_cmd = self.U[3, i]
			
			dx = target_location[0] - x_pos
			dy = target_location[1] - y_pos
			dz = target_location[2] - z_pos
			
			dtarget = ca.sqrt((dx)**2 + (dy)**2 + (dz)**2)
			
			quad_v_max = (v_cmd - v_max)**2
			quad_v_min = (v_cmd - v_min)**2
			
			#if im far away allow me to go faster otherwise slow down
			vel_penalty = ca.if_else(dtarget >= self.Effector.effector_range, 
									 quad_v_max, quad_v_min)
			
			tot_cost += vel_penalty
			
		total_tot_cost = self.pew_pew_params['weight'] * ca.sum2(tot_cost)
		
		return total_tot_cost



	def compute_omni_pew_cost(self) -> ca.SX:
		"""compute the cost function for the directional effector"""
		if self.Effector is None:
			raise ValueError('Effector not initialized')
		
		n_states = self.model_casadi.n_states
		target_location = self.P[n_states:]

		total_effector_cost = 0
		effector_cost = 0
		
		v_max = self.control_constraints['v_cmd_max']
		v_min = self.control_constraints['v_cmd_min']

		k_range = 0.6
		k_elev = 5
		k_azim = 5

		for i in range(self.N):
			x_pos = self.X[0, i]
			y_pos = self.X[1, i]
			z_pos = self.X[2, i]
			roll  = self.X[3, i]
			pitch = self.X[4, i]
			yaw   = self.X[5, i]
			v_cmd = self.U[3, i]

			if i == self.N-1:
				v_cmd_next = self.U[3, i]
			else:
				v_cmd_next = self.U[3, i+1]

			###### DIRECTIONAL EFFECTOR COST FUNCTION MAXIMIZE TIME ON TARGET BY SLOWING DOWN APPROACH######
			#right now this is set up for the directional effector
			dx = target_location[0] - x_pos
			dy = target_location[1] - y_pos
			dz = target_location[2] - z_pos
					
			dtarget = ca.sqrt((dx)**2 + (dy)**2 + (dz)**2)
			#los_target = ca.vertcat(dx, dy)
			los_target = ca.atan2(dy, dx)
			
			#slow down cost  
			#normal unit vector of the target
			los_hat = ca.vertcat(dx, dy, dz) / dtarget
			
			#ego unit vector
			u_x = ca.cos(pitch) * ca.cos(yaw)
			u_y = ca.cos(pitch) * ca.sin(yaw)
			u_z = ca.sin(pitch)
			ego_unit_vector = ca.vertcat(u_x, u_y, u_z)
			
			#dot product of the unit vectors
			dot_product = ca.dot(los_hat, ego_unit_vector)
			abs_dot_product = ca.fabs(dot_product)
   
			# the idea of this is we want to be as perpendicular as possible to the target 
			# this will use the gaussian cuve where the peak is 1 if the dot product is 
			directional_cost = -self.gaussian_fn(dot_product)
							
			#these exponential functions will be used to account for the distance and angle of the target
			#the closer we are to the target the more the distance factor will be close to 1
			error_dist_factor = ca.exp(-k_range*dtarget/self.Effector.effector_range)
			#error_dist = dtarget/self.Effector.effector_range
			
			left_wing = ca.pi
			right_wing = -ca.pi
   
			#the closer we are to the target the more the angle factor will be close to 1
			error_psi = ca.fabs(los_target - (yaw + right_wing)) #watch out for wrapping angles right ehre
			error_psi_factor = ca.exp(-k_azim*error_psi/self.Effector.effector_angle)
			
			#the closer we are to the target the more the angle factor will be close to 1
			los_theta = ca.atan2(dz, dx)
			error_theta = ca.fabs(los_theta - pitch) 
			error_phi = ca.fabs(ca.atan2(dy, dx) - roll)
			
			ratio_distance = (dtarget/self.Effector.effector_range)**2
			ratio_theta = (error_theta/self.Effector.effector_angle)**2
			ratio_phi = (error_phi/self.Effector.effector_angle)**2
			ratio_psi = (error_psi/self.Effector.effector_angle)**2		
   
			effector_dmg = (ratio_distance + (ratio_theta * abs_dot_product * ratio_psi * ratio_phi))
			#this is time on target
			#this velocity penalty will be used to slow down the vehicle as it gets closer to the target
			quad_v_max = (v_cmd - v_max)**2
			#quad_v_min = (v_cmd - v_min)**2
			quad_v_min = (v_cmd - v_min)**2
			vel_penalty = ca.if_else(error_dist_factor <= 0.005, 
									 quad_v_max, quad_v_min)

			#all other controls except for the velocity
			controls_cost = ca.sumsqr(self.U[:3, i])
			
			effector_cost += effector_dmg+ vel_penalty #+ controls_cost
			
			# constraint to make sure we don't get too close to the target and crash into it
			safe_distance = self.obs_params['safe_distance']
			diff = -dtarget + self.pew_pew_params['radius_target'] + safe_distance 
			self.g = ca.vertcat(self.g, diff)
						
			###### TOROID EFFECTOR TURN IT INTO A TIME CONSTRAINT FUNCTION ######
		
		total_effector_cost = self.pew_pew_params['weight'] * ca.sum2(effector_cost) #+ time_cost
		
		return total_effector_cost
		

	def compute_total_cost(self) -> ca.SX:
		
		if self.use_obstacle_avoidance:
			self.cost += self.compute_obstacle_avoidance_cost()
			
		if self.use_dynamic_threats:			
			self.cost += self.compute_dynamic_threats_cost(self.cost)
					
		if self.use_pew_pew:
			if self.Effector.effector_type == 'directional_3d':
				self.cost += self.compute_directional_pew_cost()

			elif self.Effector.effector_type == 'omnidirectional':
				# self.cost += self.compute_dynamics_cost()
				self.cost += self.compute_omni_pew_cost()
				
		else:
			self.cost += self.compute_dynamics_cost()
			
		return self.cost

	def solve(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray) -> dict:
		"""solve the optimal control problem"""
		
		state_init = ca.DM(x0)
		state_final = ca.DM(xF)
		
		X0 = ca.repmat(state_init, 1, self.N + 1)
		U0 = ca.repmat(u0, 1, self.N)

		n_states = self.model_casadi.n_states
		n_controls = self.model_casadi.n_controls
		
		if self.use_obstacle_avoidance:
			# constraints lower bound added 
			if self.use_pew_pew and self.Effector.effector_type == 'directional_3d':
				num_obstacles = len(self.obs_params['x']) + 1
				num_constraints = num_obstacles * self.N
				lbg =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
				# -infinity to minimum margin value for obs avoidance  
				lbg[n_states*self.N+n_states:] = -ca.inf    
				# constraints upper bound
				ubg  =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
				#rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
				ubg[n_states*self.N+n_states:] = 0
			elif self.use_pew_pew and self.Effector.effector_type == 'omnidirectional':
				num_obstacles = len(self.obs_params['x']) + 1
				num_constraints = num_obstacles * self.N
				lbg =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
				# -infinity to minimum margin value for obs avoidance
				lbg[n_states*self.N+n_states:] = -ca.inf
				# constraints upper bound
				ubg  =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
				#rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
				ubg[n_states*self.N+n_states:] = 0
			else:
				num_obstacles = len(self.obs_params['x'])
				num_constraints = num_obstacles * self.N
				lbg =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
				# -infinity to minimum margin value for obs avoidance
				lbg[n_states*self.N+n_states:] = -ca.inf
				# constraints upper bound
				ubg  =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
				#rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
				ubg[n_states*self.N+n_states:] = 0

		else:
			lbg = ca.DM.zeros((n_states*(self.N+1), 1))
			ubg  =  ca.DM.zeros((n_states*(self.N+1), 1))
			
		args = {
			'lbg': lbg,
			'ubg': ubg,
			'lbx': self.pack_variables_fn(**self.lbx)['flat'],
			'ubx': self.pack_variables_fn(**self.ubx)['flat'],
		}
		
		args['p'] = ca.vertcat(
			state_init,    # current state
			state_final   # target state
		)
		
		args['x0'] = ca.vertcat(
			ca.reshape(X0, n_states*(self.N+1), 1),
			ca.reshape(U0, n_controls*self.N, 1)
		)

		sol = self.solver(
			x0=args['x0'],
			lbx=args['lbx'],
			ubx=args['ubx'],
			lbg=args['lbg'],
			ubg=args['ubg'],
			p=args['p']
		)
		
		return sol
	
	def unpack_solution(self, sol:dict) -> np.ndarray:
		"""
		This function unpacks the solution from the solver
		Original solution is a dictionary with keys 'x' and 'u' where 
		'x' is a matrix size of (n_states * (N+1), 1) and 
		'u' is a matrix size of (n_controls * N, 1) 
		
		This function reshapes the solution to the original state and control matrices
		where u is a matrix of size (n_controls, N) and x is a matrix of size (n_states, N+1)
		"""
		u = ca.reshape(sol['x'][self.model_casadi.n_states * (self.N + 1):], 
							self.model_casadi.n_controls, self.N)
		x = ca.reshape(sol['x'][: self.model_casadi.n_states * (self.N+1)], 
								self.model_casadi.n_states, self.N+1)

		return x, u
		
	def get_solution(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray,
					 get_cost:bool=False) -> np.ndarray:
		"""
		This function solves the optimization problem and returns the solution
		in a dictionary format based on the state and control variables
		"""
		solution = self.solve(x0, xF, u0)
		#P = x0 and xF
		p = ca.vertcat(x0, xF)

		x, u = self.unpack_solution(solution)
		
		if get_cost:
			nlp_grad_f = self.solver.get_function('nlp_grad_f')
			[f, df] = nlp_grad_f(solution['x'], p)
			solution_results = {
				'x': x[0,:].full().T[:,0],
				'y': x[1,:].full().T[:,0],
				'z': x[2,:].full().T[:,0],
				'phi': x[3,:].full().T[:,0],
				'theta': x[4,:].full().T[:,0],
				'psi': x[5,:].full().T[:,0],
				'v': x[6,:].full().T[:,0],
				'u_phi': u[0,:].full().T[:,0],
				'u_theta': u[1,:].full().T[:,0],
				'u_psi': u[2,:].full().T[:,0],
				'v_cmd': u[3,:].full().T[:,0],
				'cost': f,
				'grad': df
			}
			end_time = time.time()

			return solution_results,end_time
		else:
			solution_results = {
				'x': x[0,:].full().T[:,0],
				'y': x[1,:].full().T[:,0],
				'z': x[2,:].full().T[:,0],
				'phi': x[3,:].full().T[:,0],
				'theta': x[4,:].full().T[:,0],
				'psi': x[5,:].full().T[:,0],
				'v': x[6,:].full().T[:,0],
				'u_phi': u[0,:].full().T[:,0],
				'u_theta': u[1,:].full().T[:,0],
				'u_psi': u[2,:].full().T[:,0],
				'v_cmd': u[3,:].full().T[:,0]
			}
			end_time = time.time()
			return solution_results,end_time
			
			
	def init_optimization_problem(self) -> None:
		self.update_bound_constraints()
		self.cost = self.compute_total_cost()
		self.init_solver(self.cost)
		self.is_initialized = True
		print('Optimization problem initialized')

	def update_dynamic_threats(self, threat_params:dict) -> None:
		if self.is_initialized:
			print('Optimization problem already initialized')
			self.g = []
			self.cost = 0
			
		self.dynamic_threat_params = threat_params
		self.use_dynamic_threats = True
		self.cost = self.compute_total_cost()
		self.init_solver(self.cost)
		self.is_initialized = True
		print('Dynamic threats updated')
		

		