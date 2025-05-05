
import numpy as np
import casadi as ca

from typing import Tuple, List
from dataclasses import dataclass
from casadi.casadi import MX
from optitraj.models.casadi_model import CasadiModel
from ros_mpc.OptimalControlProblem import OptimalControlProblem
from optitraj.utils.data_container import MPCParams


@dataclass
class Obstacle:
    center: Tuple[float, float]
    radius: float


class PlaneOptControl(OptimalControlProblem):
    """
    Example of a class that inherits from OptimalControlProblem
    for the Plane model using Casadi, can be used for 
    obstacle avoidance
    """

    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel,
                 use_obs_avoidance: bool = False,
                 obs_params: List[Obstacle] = None,
                 robot_radius: float = 3.0) -> None:
        super().__init__(mpc_params,
                         casadi_model)

        self.use_obs_avoidance: bool = use_obs_avoidance
        self.obs_params: List[Obstacle] = obs_params
        self.robot_radius: float = robot_radius
        if self.use_obs_avoidance:
            print("Using obstacle avoidance")
            self.is_valid_obs_params()
            self.set_obstacle_avoidance_constraints()

    def is_valid_obs_params(self) -> bool:
        """
        To use obstacle avoidance the parameters must be
        a list of Obstacle objects

        """
        if self.obs_params is None:
            raise ValueError("obs_params is None")

    def compute_dynamics_cost(self) -> MX:
        """
        Compute the dynamics cost for the optimal control problem
        """
        # initialize the cost
        cost = 0.0
        Q = self.mpc_params.Q
        R = self.mpc_params.R

        x_final = self.P[self.casadi_model.n_states:]

        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            cost += cost \
                + (states - x_final).T @ Q @ (states - x_final) \
                + controls.T @ R @ controls

        return cost

    def set_obstacle_avoidance_constraints(self) -> None:
        """
        Set the obstacle avoidance constraints for the optimal control problem
        """
        x_position = self.X[0, :]
        y_position = self.X[1, :]

        for i, obs in enumerate(self.obs_params):
            obs_center: Tuple[float] = ca.DM(obs.center)
            obs_radius: float = obs.radius
            distance = -ca.sqrt((x_position - obs_center[0])**2 +
                                (y_position - obs_center[1])**2)
            diff = distance + obs_radius + self.robot_radius
            self.g = ca.vertcat(self.g, diff[:-1].T)

    def compute_obstacle_avoidance_cost(self) -> MX:
        """
        Compute the obstacle avoidance cost for the optimal control problem
        We set g to an inequality constraint that satifies the following:
        -distance + radius <= 0 for each obstacle
        """
        cost = 0.0
        x_position = self.X[0, :]
        y_position = self.X[1, :]

        for i, obs in enumerate(self.obs_params):
            obs_center: Tuple[float] = ca.DM(obs.center)
            obs_radius: float = obs.radius
            distance = -ca.sqrt((x_position - obs_center[0])**2 +
                                (y_position - obs_center[1])**2)
            diff = distance + obs_radius + self.robot_radius
            cost += ca.sum1(diff[:-1].T)

        return 10*cost

    def compute_total_cost(self) -> MX:
        cost = self.compute_dynamics_cost()
        # cost = cost + self.compute_obstacle_avoidance_cost()
        return cost

    def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """
        Solve the optimal control problem for the given initial state and control

        """
        state_init = ca.DM(x0)
        state_final = ca.DM(xF)

        X0 = ca.repmat(state_init, 1, self.N+1)
        U0 = ca.repmat(u0, 1, self.N)

        n_states = self.casadi_model.n_states
        n_controls = self.casadi_model.n_controls
        # self.compute_obstacle_avoidance_cost()

        if self.use_obs_avoidance and self.obs_params is not None:
            # set the obstacle avoidance constraints
            num_obstacles = len(self.obs_params)  # + 1
            num_obstacle_constraints = num_obstacles * (self.N)
            # Constraints for lower and upp bound for state constraints
            # First handle state constraints
            lbg_states = ca.DM.zeros((n_states*(self.N+1), 1))
            ubg_states = ca.DM.zeros((n_states*(self.N+1), 1))

            # Now handle the obstacle avoidance constraints and add them at the bottom
            # Obstacles' lower bound constraints (-inf)
            # this is set up where -distance + radius <= 0
            lbg_obs = ca.DM.zeros((num_obstacle_constraints, 1))
            lbg_obs[:] = -ca.inf
            ubg_obs = ca.DM.zeros((num_obstacle_constraints, 1))
            # Concatenate state constraints and obstacle constraints (state constraints come first)
            # Concatenate state constraints and then obstacle constraints
            lbg = ca.vertcat(lbg_states, lbg_obs)
            ubg = ca.vertcat(ubg_states, ubg_obs)  # Same for the upper bounds
        else:
            num_constraints = n_states*(self.N+1)
            lbg = ca.DM.zeros((num_constraints, 1))
            ubg = ca.DM.zeros((num_constraints, 1))

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
        # init_time = time.time()
        solution = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        return solution
