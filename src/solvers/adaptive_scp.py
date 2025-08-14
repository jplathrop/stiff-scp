import numpy as np
import cvxpy as cp
from jax.scipy.linalg import expm as jax_expm

# for integer num_timesteps H, constructs a trajectory
# [x_0, x_1, ..., x_H] with inputs [u_0 u_1, ..., u_H] 
#   satisfying dynamics x_{k+1} = F(x_k, u_k) for k=0, ..., H-1
#   with stage rewards R(x_k, u_k) for k=0, ..., H-1
#   and terminal value V(x_H)
#   x_0 is treated as fixed problem data, and the input u_H is unused

class AdaptiveSCPSubproblem():
    def __init__(self, system, num_timesteps, dynamics_integration="forward_euler"):
        self.system = system
        self.num_timesteps = num_timesteps

        n, m = system.state_dim(), system.action_dim()

        assert dynamics_integration in ["forward_euler", "backward_euler", "matrix_exponential", "implicit"]
        self.dynamics_integration = dynamics_integration

        self.x0 = cp.Parameter((n))

        # dynamics Jacobians
        # xkp1 = F(xk, uk) \approx F(xnk, unk) 
        #                   + (I + dt * dfdx(xnk, unk)) @ (xk - xnk)
        #                   + (dt * dfdu(xnk, unk)) @ (uk - unk)
        #               for k=0, ..., H-1
        self.dfdx = [cp.Parameter((n, n)) for _ in range(num_timesteps)]
        self.dfdu = [cp.Parameter((n, m)) for _ in range(num_timesteps)]
        self.fnom = [cp.Parameter((n)) for _ in range(num_timesteps)]
        # Discrete-time state transition matrix for matrix exponential integration
        self.phi = [cp.Parameter((n, n)) for _ in range(num_timesteps)]

        # reward derivatives (index k suppresed)
        # R(x, u) \approx R(xn,un)
        #                   +dRdx(xn,un)@(x-xn) +dRdu(xn,un)@(u-un)
        #                   +1/2*(x-xn).T @ d2Rdx2(xn,un) @ (x-xn) 
        #                   + (x-xn).T @ d2Rdxdu(xn,un) @ (u-un)
        #                   +1/2*(u-un).T @ d2Rdu2(xn,un) @ (u-un)
        #       for k=0, ..., H-1
        self.Rnom = [cp.Parameter() for _ in range(num_timesteps)]
        self.dRdx = [cp.Parameter((n)) for _ in range(num_timesteps)]
        self.dRdu = [cp.Parameter((m)) for _ in range(num_timesteps)]
        # self.d2Rdx2 = [cp.Parameter((n, n), PSD = True) for _ in range(num_timesteps)]
        # self.d2Rdxdu = [cp.Parameter((n, m)) for _ in range(num_timesteps)]
        # self.d2Rdu2 = [cp.Parameter((m, m), PSD = True) for _ in range(num_timesteps)]

        self.d2Rdxdu2 = [cp.Parameter((n+m, n+m), NSD = True) for _ in range(num_timesteps)]

        # value deriatives
        # V(x_H) \approx V(xnH) + dVdx(xnH)@(xH-xnH) + 1/2*(xH-xnH).T@d2Vdx2(xnH)@(xH-xnH)
        self.Vnom = cp.Parameter()
        self.dVdx = cp.Parameter((n))
        self.d2Vdx2 = cp.Parameter((n, n), NSD = True)

        # nominal trajectory
        self.xnom = [cp.Parameter((n)) for _ in range(num_timesteps+1)]
        self.unom = [cp.Parameter((m)) for _ in range(num_timesteps)]

        # timestep
        self.dt = [cp.Parameter(nonneg=True) for _ in range(num_timesteps)]

        # decision variables
        self.x = [cp.Variable((n)) for _ in range(num_timesteps+1)]
        self.u = [cp.Variable((m)) for _ in range(num_timesteps)]

        # constraints
        constraints = []

        # initial condition
        constraints.append(self.x[0] == self.x0)

        # dynamics
        if self.dynamics_integration == "forward_euler":
            for k in range(num_timesteps):
                constraints.append(self.x[k+1] == self.xnom[k] + self.dt[k] * self.fnom[k] +
                                (np.eye(n) + self.dt[k] * self.dfdx[k]) @ (self.x[k] - self.xnom[k]) +
                                (self.dt[k] * self.dfdu[k]) @ (self.u[k] - self.unom[k]))
        elif self.dynamics_integration == "backward_euler":
            for k in range(num_timesteps):
                constraints.append(self.x[k+1] == self.xnom[k] + self.dt[k] * self.fnom[k] +
                                np.eye(n) @ (self.x[k] - self.xnom[k]) + 
                                (self.dt[k] * self.dfdx[k]) @ (self.x[k+1] - self.xnom[k+1]) +
                                (self.dt[k] * self.dfdu[k]) @ (self.u[k] - self.unom[k]))
        elif self.dynamics_integration == "matrix_exponential":
            # x_{k+1} ≈ x_nom[k] + dt * f_nom[k] + exp(A_k dt) (x_k - x_nom[k]) + dt * B_k (u_k - u_nom[k])
            for k in range(num_timesteps):
                constraints.append(
                    self.x[k+1] == self.xnom[k] + self.dt[k] * self.fnom[k]
                    + self.phi[k] @ (self.x[k] - self.xnom[k])
                    + (self.dt[k] * self.dfdu[k]) @ (self.u[k] - self.unom[k])
                )
        elif self.dynamics_integration == "implicit":
            for k in range(num_timesteps):
                if k < num_timesteps - 1:
                    constraints.append(
                        self.x[k+1] == self.x[k] + self.dt[k] * self.fnom[k]
                        + 0.5 * self.dt[k] * self.dfdx[k] @ (self.x[k+1] + self.x[k] - self.xnom[k+1] - self.xnom[k])
                        + 0.5 * self.dt[k] * self.dfdu[k] @ (self.u[k] + self.u[k+1] - self.unom[k] - self.unom[k+1])
                        + (self.xnom[k+1] - self.xnom[k] - self.dt[k] * self.fnom[k])
                    )
                else:
                    constraints.append(
                        self.x[k+1] == self.x[k] + self.dt[k] * self.fnom[k]
                        + 0.5 * self.dt[k] * self.dfdx[k] @ (self.x[k+1] + self.x[k] - self.xnom[k+1] - self.xnom[k])
                        + self.dt[k] * self.dfdu[k] @ (self.u[k] - self.unom[k])
                        + (self.xnom[k+1] - self.xnom[k] - self.dt[k] * self.fnom[k])
                    )
        else:
            raise RuntimeError("Unreachable code.")
        
        # state and action box constraints
        for k in range(num_timesteps+1):
            constraints.append(self.x[k] >= np.asarray(self.system.X()[:,0]))
            constraints.append(self.x[k] <= np.asarray(self.system.X()[:,1]))
            if k < num_timesteps:
                constraints.append(self.u[k] >= np.asarray(self.system.U()[:,0]))
                constraints.append(self.u[k] <= np.asarray(self.system.U()[:,1]))

        # trust region
        self.epsilon = cp.Parameter()
        for k in range(num_timesteps+1):
            constraints.append(cp.norm(self.x[k] - self.xnom[k]) <= self.epsilon)
            if k < num_timesteps:
                constraints.append(cp.norm(self.u[k] - self.unom[k]) <= self.epsilon)


        # value function
        value = 0
        for k in range(num_timesteps):
            xkuk = cp.hstack([self.x[k], self.u[k]])
            value += self.dt[k] * (self.Rnom[k] + self.dRdx[k] @ (self.x[k] - self.xnom[k]) + self.dRdu[k] @ (self.u[k] - self.unom[k]) + \
                0.5 * cp.quad_form(xkuk, self.d2Rdxdu2[k]))
                # 0.5 * cp.quad_form(self.x[k] - self.xnom[k], self.d2Rdx2[k]) + \
                # 1.0 * (self.x[k] - self.xnom[k]).T @ self.d2Rdxdu[k] @ (self.u[k] - self.unom[k]) + \
                # 0.5 * cp.quad_form(self.u[k] - self.unom[k], self.d2Rdu2[k])
        value += self.Vnom + self.dVdx @ (self.x[num_timesteps] - self.xnom[num_timesteps]) + \
            0.5 * cp.quad_form(self.x[num_timesteps] - self.xnom[num_timesteps], self.d2Vdx2)

        self.problem = cp.Problem(cp.Maximize(value), constraints)
        assert(self.problem.is_dcp())

    def solve(self, xnom, unom, dt, epsilon, cvxpy_kwargs={}):
        for k in range(self.num_timesteps):
            self.x0.save_value(np.asarray(xnom[0]))
            self.dt[k].save_value(dt[k])

            if self.dynamics_integration == "forward_euler":
                self.dfdx[k].save_value(np.asarray(self.system.dfdx(xnom[k], unom[k])))
                self.dfdu[k].save_value(np.asarray(self.system.dfdu(xnom[k], unom[k])))
                self.fnom[k].save_value(np.asarray(self.system.f(xnom[k], unom[k])))
            elif self.dynamics_integration == "backward_euler":
                self.dfdx[k].save_value(np.asarray(self.system.dfdx(xnom[k+1], unom[k])))
                self.dfdu[k].save_value(np.asarray(self.system.dfdu(xnom[k+1], unom[k])))
                self.fnom[k].save_value(np.asarray(self.system.f(xnom[k+1], unom[k])))
            elif self.dynamics_integration == "matrix_exponential":
                Ak = np.asarray(self.system.dfdx(xnom[k], unom[k]))
                Bk = np.asarray(self.system.dfdu(xnom[k], unom[k]))
                fk = np.asarray(self.system.f(xnom[k], unom[k]))
                self.dfdx[k].save_value(Ak)
                self.dfdu[k].save_value(Bk)
                self.fnom[k].save_value(fk)
                # Phi_k = exp(A_k * dt)
                Phi_k = np.asarray(jax_expm(Ak * self.dt[k].value))
                self.phi[k].save_value(Phi_k)
            elif self.dynamics_integration == "implicit":
                # midpoints of the *nominal* trajectory
                xbar = 0.5*(np.asarray(xnom[k]) + np.asarray(xnom[k+1]))
                ubar = 0.5*(np.asarray(unom[k]) + np.asarray(unom[k+1])) if k < self.num_timesteps - 1 else np.asarray(unom[k])
                Ak = np.asarray(self.system.dfdx(xbar, ubar))
                Bk = np.asarray(self.system.dfdu(xbar, ubar))
                fk = np.asarray(self.system.f(xbar, ubar))

                self.dfdx[k].save_value(Ak)
                self.dfdu[k].save_value(Bk)
                self.fnom[k].save_value(fk)
            else:
                raise RuntimeError("Unreachable code.")
            
            self.Rnom[k].save_value(np.asarray(self.system.R(xnom[k], unom[k])))
            self.dRdx[k].save_value(np.asarray(self.system.dRdx(xnom[k], unom[k])))
            self.dRdu[k].save_value(np.asarray(self.system.dRdu(xnom[k], unom[k])))

            d2Rdxdu2 = np.block([[self.system.d2Rdx2(xnom[k], unom[k]), self.system.d2Rdxdu(xnom[k], unom[k])], [self.system.d2Rdxdu(xnom[k], unom[k]).T, self.system.d2Rdu2(xnom[k], unom[k])]])
            self.d2Rdxdu2[k].save_value(d2Rdxdu2)
            # self.d2Rdx2[k].save_value(self.system.d2Rdx2(xnom[k], unom[k]))
            # self.d2Rdxdu[k].save_value(self.system.d2Rdxdu(xnom[k], unom[k]))
            # self.d2Rdu2[k].save_value(self.system.d2Rdu2(xnom[k], unom[k]))
            
            self.xnom[k].save_value(np.asarray(xnom[k]))
            self.unom[k].save_value(np.asarray(unom[k]))
        self.xnom[self.num_timesteps].save_value(np.asarray(xnom[self.num_timesteps]))

        self.Vnom.save_value(np.asarray(self.system.V(xnom[self.num_timesteps])))
        self.dVdx.save_value(np.asarray(self.system.dVdx(xnom[self.num_timesteps])))
        self.d2Vdx2.save_value(np.asarray(self.system.d2Vdx2(xnom[self.num_timesteps])))

        self.epsilon.save_value(epsilon)

        value = self.problem.solve(solver=cp.MOSEK, **cvxpy_kwargs)
        if cvxpy_kwargs is not None and cvxpy_kwargs.get("verbose", False):
            print(f"Problem status: {self.problem.status}")
        x_out = np.array([self.x[k].value for k in range(self.num_timesteps+1)])
        u_out = np.array([self.u[k].value for k in range(self.num_timesteps)])
        return x_out, u_out, value
    