"""
This module defines the OptimalControlProblem class for solving optimal control problems using 
Model Predictive Control (MPC). The class provides a generic structure for defining state and control 
constraints, dynamic equations, and cost functions.

Users must inherit from this class and implement the following methods:
    - `compute_total_cost`: Define the specific cost function of the control problem.
    - `solve`: Solve the optimization problem using the defined cost and constraints.
"""

import casadi as ca
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.utils.limits import Limits, validate_limits


class OptimalControlProblem(ABC):
    """
    Represents an abstract base class for solving optimal control problems.

    This class implements the general structure for an MPC problem, including the initialization 
    of decision variables, setting up constraints (state, control, dynamic), and preparing the solver.

    **How to Use:**
    - Inherit from this class.
    - Implement the `compute_total_cost` method to define the cost function.
    - Implement the `solve` method to formulate and solve the optimization problem.

    Parameters:
        mpc_params (MPCParams): Parameters for MPC (horizon, time step, etc.)
        casadi_model (CasadiModel): CasADi model that defines system dynamics.

    Attributes:
        nlp (Dict): Nonlinear programming problem dictionary (cost function, decision variables, constraints).
        solver (CasADi solver): Solver for the optimal control problem.
        is_initialized (bool): Flag to indicate if the solver is initialized.
        cost (float): Total cost of the control problem.
        N (np.ndarray): Time horizon for the MPC problem.
        dt (np.ndarray): Time step for the MPC problem.
        Q (np.ndarray): State weight matrix in the cost function.
        R (np.ndarray): Control weight matrix in the cost function.
        state_limits (Dict): State bounds for optimization.
        ctrl_limits (Dict): Control bounds for optimization.
    """

    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel) -> None:
        self.nlp: Dict = None
        self.solver = None
        self.is_initialized: bool = False
        self.cost: float = 0.0

        # Initialize MPC parameters
        self.mpc_params: MPCParams = mpc_params
        self.N: np.ndarray[float] = mpc_params.N
        self.dt: np.ndarray[float] = mpc_params.dt
        self.Q: np.ndarray[float] = mpc_params.Q
        self.R: np.ndarray[float] = mpc_params.R

        # Initialize CasADi model
        self.casadi_model: CasadiModel = casadi_model
        self.state_limits: dict = casadi_model.state_limits
        self.ctrl_limits: dict = casadi_model.control_limits

        if self.state_limits is None:
            raise ValueError("State limits not defined.")
        if self.ctrl_limits is None:
            raise ValueError("Control limits not defined.")

        # Validate the state and control limits
        validate_limits(self.state_limits, limit_type="state")
        validate_limits(self.ctrl_limits, limit_type="control")

        self.g: List[ca.SX] = []
        self._init_decision_variables()
        self._check_correct_dimensions(self.X, self.U)
        self.define_bound_constraints()
        self.set_dynamic_constraints()

    def _check_correct_dimensions(self, x: ca.MX, u: ca.MX) -> None:
        """
        Check that the dimensions of the states and controls are correct.
        Raises an exception if dimensions mismatch with the model's expectations.
        """
        if x.size()[0] != self.casadi_model.n_states:
            raise ValueError("The states do not have the correct dimensions.")

        if u.size()[0] != self.casadi_model.n_controls:
            raise ValueError(
                "The controls do not have the correct dimensions.")

    def _init_decision_variables(self) -> None:
        """
        Initialize decision variables (states X and controls U).
        These are symbolic variables used by the CasADi solver.
        """
        self.X: ca.MX = ca.MX.sym('X', self.casadi_model.n_states, self.N+1)
        self.U: ca.MX = ca.MX.sym('U', self.casadi_model.n_controls, self.N)

        # Initial and final state placeholders
        self.P = ca.MX.sym(
            'P', self.casadi_model.n_states + self.casadi_model.n_states)

        # Decision variables for optimization
        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),
            self.U.reshape((-1, 1)),
        )

    def define_bound_constraints(self):
        """Define the bound constraints for states and controls."""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']

        # Pack and unpack helper functions for decision variables
        self.pack_variables_fn = ca.Function(
            'pack_variables_fn', self.variables_list,
            [self.OPT_variables], self.variables_name, ['flat'])

        self.unpack_variables_fn = ca.Function(
            'unpack_variables_fn', [self.OPT_variables],
            self.variables_list, ['flat'], self.variables_name)

        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)

    def update_bound_constraints(self) -> None:
        """
        Update the bound constraints for states and controls based on the model's limits.
        Users can override this to define problem-specific constraints.
        """
        # Update control limits
        ctrl_keys = list(self.ctrl_limits.keys())
        for i, ctrl_name in enumerate(ctrl_keys):
            self.lbx['U'][i, :] = self.ctrl_limits[ctrl_name]['min']
            self.ubx['U'][i, :] = self.ctrl_limits[ctrl_name]['max']

        # Update state limits
        state_keys = list(self.state_limits.keys())
        for i, state_name in enumerate(state_keys):
            self.lbx['X'][i, :] = self.state_limits[state_name]['min']
            self.ubx['X'][i, :] = self.state_limits[state_name]['max']

    @abstractmethod
    def compute_total_cost(self) -> ca.MX:
        """
        Abstract method that must be implemented by the user to compute the total cost of the 
        optimal control problem. This should incorporate both state and control costs.
        """
        pass

    # @abstractmethod
    # def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Abstract method that must be implemented by the user to solve the optimal control problem.
    #     The method should return the optimal state and control trajectories.
    #     """
    #     pass

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

    def set_dynamic_constraints(self):
        """
        Define dynamic constraints for the system using a Runge-Kutta 4th-order (RK4) integration scheme.
        This ensures the system dynamics are respected.
        """
        self.g = self.X[:, 0] - self.P[:self.casadi_model.n_states]
        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            k1 = self.casadi_model.function(states, controls)
            k2 = self.casadi_model.function(states + self.dt/2 * k1, controls)
            k3 = self.casadi_model.function(states + self.dt/2 * k2, controls)
            k4 = self.casadi_model.function(states + self.dt * k3, controls)
            state_next_rk4 = states + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            # Add the dynamic constraint to the constraints list
            self.g = ca.vertcat(self.g, self.X[:, k+1] - state_next_rk4)

    # def init_solver(self, cost_fn: ca.MX, solver_opts: Dict = None,
    #                 max_wall_time_sec: float = 0.25) -> None:
    #     """Initialize the solver for the optimization problem using the defined cost function and constraints."""
    #     nlp_prob = {
    #         'f': cost_fn,
    #         'x': self.OPT_variables,
    #         'g': self.g,
    #         'p': self.P
    #     }

    #     if solver_opts is None:
    #         solver_opts = {
    #             'ipopt': {
    #                 'print_level': 0,
    #                 'warm_start_init_point': 'yes',
    #                 'acceptable_tol': 1e-2,
    #                 'acceptable_obj_change_tol': 1e-2,
    #                 'max_wall_time': max_wall_time_sec,
    #             },
    #             'print_time': 0,
    #             'expand': 1,
    #         }

    #     self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_opts)

    def init_solver(self, cost_fn: ca.MX, solver_opts: Dict = None,
        max_wall_time_sec: float = 0.1) -> None:
        print("Using ma27 solver")
        nlp_prob = {
            'f': cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }
        solver_opts = {
            'ipopt': {
                'print_level': 0,
                'warm_start_init_point': 'yes',
                'acceptable_tol': 1e-2,
                'acceptable_obj_change_tol': 1e-2,
                'max_wall_time': max_wall_time_sec,
                'print_level': 0,
                'warm_start_init_point': 'yes', #use the previous solution as initial guess
                # 'acceptable_tol': 1e-2,
                # 'acceptable_obj_change_tol': 1e-2,
                'hsllib': '/usr/local/lib/libcoinhsl.so', #need to set the optimizer library
                # 'hsllib': '/usr/local/lib/libfakemetis.so', #need to set the optimizer library
                'linear_solver': 'ma27',
                # 'hessian_approximation': 'limited-memory', # Changes the hessian calculation for a first order approximation.
            },
            'verbose': True,
            # 'jit':True,
            'print_time': 0,    
            'expand': 1
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_opts)


    def init_optimization(self, solver_opts: Dict = None) -> None:
        """Initialize the optimization problem, update constraints, and set up the cost function."""
        if self.is_initialized:
            self.g = []
            self.cost: float = 0.0

        self.update_bound_constraints()
        self.cost = self.compute_total_cost()
        self.init_solver(self.cost, solver_opts)
        self.is_initialized = True

    def unpack_solution(self, sol: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack the solution from the solver into state and control trajectories."""
        u = ca.reshape(sol['x'][self.casadi_model.n_states * (self.N + 1):],
                       self.casadi_model.n_controls, self.N)
        x = ca.reshape(sol['x'][: self.casadi_model.n_states * (self.N+1)],
                       self.casadi_model.n_states, self.N+1)

        return x, u

    def get_solution(self, solution: Dict) -> Dict:
        """
        Extract the state and control trajectories from the optimization solution.

        Returns:
            Dict: Dictionary containing state and control trajectories.
        """
        x, u = self.unpack_solution(solution)
        state_keys = list(self.state_limits.keys())
        ctrl_keys = list(self.ctrl_limits.keys())

        state_dict = {}
        ctrl_dict = {}

        for i, state_name in enumerate(state_keys):
            state_dict[state_name] = x[i, :].full().T[:, 0]

        for i, ctrl_name in enumerate(ctrl_keys):
            ctrl_dict[ctrl_name] = u[i, :].full().T[:, 0]

        return {
            "states": state_dict,
            "controls": ctrl_dict
        }

    def solve_and_get_solution(self, x0: np.ndarray,
                               xF: np.ndarray,
                               u0: np.ndarray) -> Dict:
        """
        Solve the optimization problem and return the state and control trajectories.

        Parameters:
            x0 (np.ndarray): Initial state.
            xF (np.ndarray): Final state.
            u0 (np.ndarray): Initial control input.

        Returns:
            Dict: Solution containing states and controls.
        """
        solution = self.solve(x0, xF, u0)
        return self.get_solution(solution)
