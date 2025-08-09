from src.system import System

import jax.numpy as jnp
from typing import List


class AckermannCar(System):
    def __init__(self, xd: jnp.ndarray | None = None, wheelbase_meters: float = 2.5, steering_time_constant_s: float = 0.3):
        self.l = float(wheelbase_meters)
        self.tau = float(steering_time_constant_s)

        angle_limit_rad = jnp.deg2rad(60.0)

        self.state_bounds = jnp.array([
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-jnp.pi, jnp.pi],
            [-angle_limit_rad, angle_limit_rad],
        ])

        self.action_bounds = jnp.array([
            [-1.0, 1.0],
            [-angle_limit_rad, angle_limit_rad],
        ])

        self.Q_x = jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0]))
        self.Q_u = jnp.diag(jnp.array([0.001, 0.001]))
        self.Q_f = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0]))

        self.n = 4
        self.m = 2

        if xd is None:
            self.xd = jnp.zeros((self.n,))
        else:
            self.xd = xd

        super().__init__()
        return

    def f(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        v = action[0]
        eta = action[1]
        theta = state[2]
        delta = state[3]
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = v / self.l * jnp.tan(delta)
        delta_dot = (eta - delta) / self.tau
        return jnp.array([x_dot, y_dot, theta_dot, delta_dot])

    def R(self, state: jnp.ndarray, action: jnp.ndarray) -> float:
        sx = state - self.xd
        return - sx.T @ self.Q_x @ sx - action.T @ self.Q_u @ action

    def V(self, state: jnp.ndarray) -> float:
        sx = state - self.xd
        return - sx.T @ self.Q_f @ sx

    def state_labels(self) -> List[str]:
        return [r"$x$", r"$y$", r"$\\theta$", r"$\\delta$"]

    def state_dim(self) -> int:
        return self.n

    def action_dim(self) -> int:
        return self.m

    def X(self) -> jnp.ndarray:
        return self.state_bounds

    def U(self) -> jnp.ndarray:
        return self.action_bounds

