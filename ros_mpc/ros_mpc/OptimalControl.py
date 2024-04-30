import casadi as ca
import numpy as np
import abc 

class OptimalControlProblem():
    def __init__(self, mpc_params:dict, model_casadi) -> None:
        self.mpc_params = mpc_params
        self.N = self.mpc_params['N']
        self.Q = self.mpc_params['Q']
        self.R = self.mpc_params['R']
        self.dt = self.mpc_params['dt']
        self.model_casadi = model_casadi
        self.g = []        
        self.init_decision_variables()
        self.integrator = self.get_casadi_integrator()
        self.sample_shot = self.sample_shot_trajectory()
        
        self.define_bound_constraints()
        self.set_dynamic_constraints()
        self.solver = None
        
    def init_decision_variables(self):
        #decision variables
        """intialize decision variables for state space models"""
        model_casadi = self.model_casadi
        self.X = ca.SX.sym('X', model_casadi.n_states, self.N + 1)
        self.U = ca.SX.sym('U', model_casadi.n_controls, self.N)

        #column vector for storing initial and target locations
        self.P = ca.SX.sym('P', model_casadi.n_states + model_casadi.n_states)

        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        print('Decision variables initialized')
        
        
    def define_bound_constraints(self):
        """define bound constraints of system"""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']
        
        #function to turn decision variables into one long row vector
        self.pack_variables_fn = ca.Function('pack_variables_fn', self.variables_list, 
                                             [self.OPT_variables], self.variables_name, 
                                             ['flat'])
        
        #function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function('unpack_variables_fn', [self.OPT_variables], 
                                               self.variables_list, ['flat'], 
                                               self.variables_name)

        ##helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)
        print('Bound constraints defined')
    
    @abc.abstractmethod
    def update_bound_constraints(self) -> None:
        """update the bound constraints"""
        raise NotImplementedError('Must implement this function:', self.update_bound_constraints.__name__)

    def get_casadi_integrator(self) -> ca.Function:
        """
        get one step integrator for the system utilizes Runge-Kutta 4th order
        """
        states = ca.SX.sym('states', self.model_casadi.n_states)
        controls = ca.SX.sym('controls', self.model_casadi.n_controls)
        
        k1 = self.model_casadi.function(states, controls)
        k2 = self.model_casadi.function(states + self.dt/2 * k1, controls)
        k3 = self.model_casadi.function(states + self.dt/2 * k2, controls)
        k4 = self.model_casadi.function(states + self.dt * k3, controls)
        state_next_rk4 = states + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        integrator = ca.Function('integrator', [states, controls],
                                [state_next_rk4], ['states', 'controls'], ['x_next'])
        
        return integrator
    

    def sample_shot_trajectory(self) -> ca.Function:
        """
        shoot a trajectory using Runge-Kutta 4th order
        """
        states = ca.SX.sym('states', self.model_casadi.n_states, self.N + 1)
        controls = ca.SX.sym('controls', self.model_casadi.n_controls, self.N)
        X_next = ca.SX.sym('X_next', self.model_casadi.n_states)
        

        for k in range(self.N):
            X_next = self.integrator(states[:, k], controls[:, k])

            self.g = ca.vertcat(self.g, states[:, k+1] - X_next)

        sample_shot = ca.Function('sample_shot', 
                                  [states, controls], 
                                  [X_next], ['X', 'U'], ['X_next'])

        return sample_shot
        
    def set_dynamic_constraints(self) -> None:
        #dynamic constraints
        #equality constraint for initial condition
        self.g = self.X[:, 0] - self.P[:self.model_casadi.n_states]                  
        for k in range(self.N):
            # state_next_rk4 = self.integrator(self.X[:, k], self.U[:, k])
            # state_next_rk4 = self.integrator_fn(self.X[:, k], self.U[:, k])
            
            states = self.X[:, k]
            controls = self.U[:, k]
            k1 = self.model_casadi.function(states, controls)
            k2 = self.model_casadi.function(states + self.dt/2 * k1, controls)
            k3 = self.model_casadi.function(states + self.dt/2 * k2, controls)
            k4 = self.model_casadi.function(states + self.dt * k3, controls)
            state_next_rk4 = states + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            # constraint to make sure our dynamics are satisfied
            self.g = ca.vertcat(self.g, self.X[:, k+1] - state_next_rk4) 
        
        # self.g = self.X[:,0] - self.P[:self.model_casadi.n_states]        
        # Xn = self.sample_shot.map(self.N, 'openmp')(self.X, self.U)
        # for k in range(self.N):
        #     state_next_rk4 = Xn[:, k]
        #     self.g = ca.vertcat(self.g, self.X[:, k+1] - state_next_rk4)
            

        #self.g = ca.vertcat(self.g, gaps.T)
        
        # g = ca.vertcat(self.X[:, 0] - self.P[:self.model_casadi.n_states],
        #                   ca.reshape(Xn, -1, 1) - self.X[:, 1:])
        # self.g = g
        print('Dynamic constraints set')
        
        
    def compute_dynamics_cost(self, time_constraint:bool=False) -> ca.SX:
        """compute the cost function"""
        n_states = self.model_casadi.n_states   
        Q = self.Q
        R = self.R
        P = self.P
        x_final = P[n_states:]
        v_cmd = self.U[3, :]
        cost = 0
        if time_constraint:
            for k in range(self.N):
                states = self.X[:, k]
                controls = self.U[:, k]
                cost += cost \
                        + (states - x_final).T @ Q @ (states - x_final) \
                        + controls.T @ R @ controls
        #add terminal cost
        else:
            terminal_cost = (self.X[:, self.N] - x_final).T @ Q \
                @ (self.X[:, self.N] - x_final)
            #divide cost by velocity 
            cost += (terminal_cost / v_cmd[-1])
        
        print('Dynamics cost computed')
        return cost
    
    @abc.abstractmethod
    def compute_total_cost(self) -> ca.SX:
        """compute the total cost function"""
        # return self.compute_dynamics_cost(x_final)
        raise NotImplementedError(
            'Must implement this function:', 
            self.compute_total_cost.__name__)
    
    
    def init_solver(self, cost_fn:ca.SX) -> None:
        
        nlp_prob = {
            'f': cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }
        solver_opts = {
            'ipopt': {
                'max_iter': 150,
                'max_cpu_time': 0.15,
                'max_wall_time': 0.15,
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

        #create solver
        self.solver = ca.nlpsol('solver', 'ipopt', 
            nlp_prob, solver_opts)
        print('Solver initialized')

