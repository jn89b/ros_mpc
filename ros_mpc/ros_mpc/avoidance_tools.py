import time 
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from scipy.spatial import distance


def does_right_side_have_more_obstacles(ego_heading:float,
                                    danger_zones:np.ndarray, right_array:np.ndarray,
                                    left_array:np.ndarray) -> bool:
    """
    Given position and heading we will determine which side has more obstacles by 
    computing the cross product of the ego_heading and the vector to the obstacle
    We will then sum up the right_array and return a boolean value 
    it will return true if the right side has more obstacles and 
    false if the left side has more obstacles 
    """
    ego_unit_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    for i in range(len(danger_zones)):
        #compute the vector to the obstacle
        obs_position = danger_zones[i][:2]
        
        #compute los from ego to obstacle
        los_vector = ego_unit_vector - obs_position
        los_vector /= np.linalg.norm(los_vector)
        cross_product = np.cross(ego_unit_vector, los_vector)

        #if cross product is positive then obstacle is to the left 
        if cross_product >= 0:
            left_array[i] = 1
        #if cross product is negative then obstacle is to the right
        else:
            right_array[i] = 1 
    
    sum_right = np.sum(right_array)
    sum_left = np.sum(left_array)
    
    if sum_right > sum_left:
        print("Right side has more obstacles")
        return True
    else:
        print("Left side has more obstacles")
        return False

# @jit(nopython=True)
def find_driveby_direction(goal_position:np.ndarray, current_position:np.ndarray, 
                            heading_rad:float,
                            obs_radius:float,  
                            robot_radius:float, 
                            consider_obstacles:bool=False, 
                            danger_zones:np.ndarray=None,
                            use_nearest_ref:bool=False,
                            ref_point:np.ndarray=None) -> np.ndarray:
    """
    Finds the lateral offset given a goal location and the current position
    """    
    
    if goal_position.shape[0] != 2:
        goal_position = goal_position[:2]
    if current_position.shape[0] != 2:
        current_position = current_position[:2]
    
    
    range_total = obs_radius + robot_radius
    
    ego_unit_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    
    #swap the direction sign to get the normal vector
    drive_by_vector_one = np.array([ego_unit_vector[1], -ego_unit_vector[0]])
    drive_by_vector_two = np.array([-ego_unit_vector[1], ego_unit_vector[0]])
    
    drive_by_vector_one = drive_by_vector_one * range_total
    drive_by_vector_two = drive_by_vector_two * range_total
    
    #pick the one closer to the current position
    distance_one = np.linalg.norm(current_position - (goal_position + drive_by_vector_one))
    distance_two = np.linalg.norm(current_position - (goal_position + drive_by_vector_two))

    if consider_obstacles and use_nearest_ref==False:
        #make a decision baased on the obstacles
        is_right = does_right_side_have_more_obstacles(heading_rad, danger_zones,
                                                         np.zeros(len(danger_zones)),
                                                         np.zeros(len(danger_zones)))

        
        cross_product = np.cross(ego_unit_vector, drive_by_vector_one)
        if cross_product > 0:
            #if the cross product is positive then the right vector is the left vector
            left_vector = drive_by_vector_one
            right_vector = drive_by_vector_two
        else:
            left_vector = drive_by_vector_two
            right_vector = drive_by_vector_one
        
        if is_right:
            #if right side has more obstacles then pick the left vector
            #figure out which vector is the left vector
            drive_by_vector = left_vector
        else:
            drive_by_vector = right_vector
        # if distance_one < distance_two:
        #     drive_by_vector = drive_by_vector_two
        # else:
        #     drive_by_vector = drive_by_vector_one
    elif use_nearest_ref:
        
        #make a decision based on the nearest reference point
        distance_one = np.linalg.norm(current_position \
            - (ref_point + drive_by_vector_one))
        distance_two = np.linalg.norm(current_position \
            - (ref_point + drive_by_vector_two))
        
        if distance_one < distance_two:
            drive_by_vector = drive_by_vector_two
        else:
            drive_by_vector = drive_by_vector_one
            
    else:  
        if distance_one < distance_two:
            drive_by_vector = drive_by_vector_two
        else:
            drive_by_vector = drive_by_vector_one
    
        #check if danger zone is within the drive by vector
        if consider_obstacles:
            for obs in danger_zones:
                obs_position = obs[:2]
                obs_radius = obs[-1]
                distance_to_obstacle = np.linalg.norm(obs_position - (goal_position + drive_by_vector))
                delta_r = distance_to_obstacle - (obs_radius + robot_radius)
                if delta_r <= 0:
                    #if the obstacle is within the drive by vector then we need to 
                    #pick the other vector
                    if distance_one < distance_two:
                        drive_by_vector = drive_by_vector_one
                    else:
                        drive_by_vector = drive_by_vector_two
                    break
                
    #apply to goal position
    drive_by_position = goal_position + drive_by_vector
    
    return drive_by_position


def knn_obstacles(obs:np.ndarray, ego_pos:np.ndarray, 
                  K:int=3, use_2d:bool=False) -> tuple:
    """
    Find the K nearest obstacles to the ego vehicle and return the 
    obstacle positions and distances
    """
    if use_2d:
        ego_position = ego_pos[:2]
        obstacles = obs[:,:2]
    
    nearest_indices = distance.cdist([ego_position], obstacles).argsort()[:, :K]
    nearest_obstacles = obs[nearest_indices]
    return nearest_obstacles[0], nearest_indices


def find_inline_obstacles(ego_unit_vector:np.ndarray, obstacles:np.ndarray, 
                          ego_position:np.ndarray, 
                          dot_product_threshold:float=0.0, 
                          use_2d:bool=False) -> tuple:
    '''
    compute the obstacles that are inline with the ego
    '''
    #check size of ego_position
    if use_2d:
        if ego_position.shape[0] != 2:
            ego_position = ego_position[:2]
        current_obstacles = obstacles[:,:2]
        
    inline_obstacles = []
    dot_product_vals = []
    for i, obs in enumerate(current_obstacles):
        los_vector = obs - ego_position
        los_vector /= np.linalg.norm(los_vector)
        dot_product = np.dot(ego_unit_vector, los_vector)
        if dot_product >= dot_product_threshold:
            inline_obstacles.append(obstacles[i])
            dot_product_vals.append(dot_product)
            
    return inline_obstacles, dot_product_vals

def find_danger_zones(obstacles:np.ndarray, 
                      ego_position:np.ndarray, 
                      min_radius_turn:float, 
                      dot_products:np.ndarray,
                      distance_buffer:float=10.0, 
                      use_2d:bool=False) -> np.ndarray:
    '''
    compute the obstacles that are inline with the ego 
    we check if they are within the minimum radius of turn with the buffer
    '''
    #check size of ego_position
    if use_2d:
        if ego_position.shape[0] != 2:
            ego_position = ego_position[:2]
    
    danger_zones = []
    new_dot_products = []
    for i, obs, in enumerate(obstacles):
        obs_position = obs[:2]
        obs_radius = obs[-1]
        distance_to_obstacle = np.linalg.norm(obs_position - ego_position)
        delta_r = distance_to_obstacle - (min_radius_turn + obs_radius + distance_buffer)
        if delta_r <= distance_buffer:
            #compute the danger zone
            new_dot_products.append(dot_products[i])
            danger_zones.append(obs)
            
    return danger_zones, new_dot_products