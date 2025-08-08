from src.system import System

import jax.numpy as jnp
from typing import List

class DubinsCar(System):
    def __init__(self, xd = jnp.zeros((3,))):
        self.state_bounds = jnp.array([[-10., 10.], [-10., 10.], [-jnp.pi, jnp.pi]])
        self.action_bounds = jnp.array([[-1., 1.], [-1., 1.]])

        self.Q_x = jnp.diag(jnp.array([0., 0., 0.]))
        self.Q_u = jnp.diag(jnp.array([0.001, 0.001]))
        self.Q_f = jnp.diag(jnp.array([1., 1., 1]))

        self.n = 3
        self.m = 2

        self.xd = xd

        super().__init__()
        return

    def f(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([
            action[0] * jnp.cos(state[2]),
            action[0] * jnp.sin(state[2]),
            action[1]])
    
    def R(self, state: jnp.ndarray, action: jnp.ndarray) -> float:
        return - jnp.dot((state - self.xd).T, jnp.dot(self.Q_x, state - self.xd)) - jnp.dot(action.T, jnp.dot(self.Q_u, action))
    
    def V(self, state: jnp.ndarray) -> float:
        return - jnp.dot((state - self.xd).T, jnp.dot(self.Q_f, state - self.xd))
    
    def state_labels(self) -> List[str]:
        return [r"$x$", r"$y$", r"$\theta$"]
    
    def state_dim(self) -> int:
        return self.n

    def action_dim(self) -> int:
        return self.m

    def X(self) -> jnp.ndarray:
        return self.state_bounds

    def U(self) -> jnp.ndarray:
        return self.action_bounds