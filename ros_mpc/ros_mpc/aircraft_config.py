import numpy as np
import casadi as ca

"""
This file contains the aircraft configurations all in one place
Future work is to put this in a yaml file and load it for easy access
"""

RADIUS_TARGET = 5.0
N_OBSTACLES = 1
OBX_MIN_RANGE = 0
OBX_MAX_RANGE = 150
OBX_MIN_RADIUS = 20
OBX_MAX_RADIUS = 30
SEED_NUMBER = 0


# This is dumb but will work for now, should have a better way to do this
# probably make a service to update the new goal state 
GOAL_STATE = [
    200.0, #x 
    200.0, #y 
    50.0,  #z 
    0.0,   #phi 
    0.0,   #theta
    0.0,   #psi 
    20.0,  #airspeed
]

DONE_STATE = [
    0.0, #x 
    0.0, #y 
    20.0,  #z 
    0.0,   #phi 
    0.0,   #theta
    0.0,   #psi 
    20.0,  #airspeed
]

# This is dumb but will work for now, should have a better way to do this
np.random.seed(SEED_NUMBER)
obx = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_OBSTACLES)
oby = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_OBSTACLES)
obz = np.random.randint(0, 80, N_OBSTACLES)
radii = np.random.randint(OBX_MIN_RADIUS, OBX_MAX_RADIUS, N_OBSTACLES)

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(15),
    'u_theta_max': np.deg2rad(10),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   25
}

state_constraints = {
    'x_min': -np.inf,
    'x_max': np.inf,
    'y_min': -np.inf,
    'y_max': np.inf,
    'z_min': 40,
    'z_max': 95,
    'phi_min':  -np.deg2rad(50),
    'phi_max':   np.deg2rad(50),
    'theta_min':-np.deg2rad(10),
    'theta_max': np.deg2rad(10),
    'airspeed_min': 15,
    'airspeed_max': 30,
    'psi_min':  -np.deg2rad(np.inf),
    'psi_max':   np.deg2rad(np.inf),
}

mpc_params = {
    'N': 15,
    'Q': ca.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'R': ca.diag([0.5, 0.5, 0.5, 0.5]),
    'dt': 0.1
}

directional_effector_config = {
        'effector_range': 100, 
        'effector_power': 100, 
        'effector_type': 'directional_3d', 
        'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
        'weight': 1, 
        'radius_target': RADIUS_TARGET
        }

#might need to set the obstacles or update it later on 
obs_avoid_params = {
    'weight': 1E-10,
    'safe_distance': 3.0,
    'x': obx,
    'y': oby,
    'z': obz, 
    'radii': radii
}

omni_effector_config = {
        'effector_range': 50, 
        'effector_power': 1, 
        'effector_type': 'omnidirectional', 
        'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
        'weight': 1, 
        'radius_target': RADIUS_TARGET,
        'minor_radius': 28
        }


mpc_params_load = {
    'N': 30,
    'Q': ca.diag([1E-2, 1E-2, 1E-2, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.1, 0.1, 0.1, 0.1]),
    'dt': 0.1
}


control_constraints_load = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'load_z_min': -3,
    'load_z_max':  6,
    'load_x_min':  -2, #left 
    'load_x_max':   2, #right
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

state_constraints_load = {
    'x_min': -np.inf,
    'x_max': np.inf,
    'y_min': -np.inf,
    'y_max': np.inf,
    'z_min': 20,
    'z_max': 80,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(15),
    'theta_max': np.deg2rad(15),
    # 'psi_min':  -np.deg2rad(180),
    # 'psi_max':   np.deg2rad(180),
    'airspeed_min': 15,
    'airspeed_max': 30
}