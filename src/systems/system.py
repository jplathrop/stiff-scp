import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import List
import time

class System(ABC):
    
    def __init__(self):
        # print("Computing derivatives...")
        tic = time.time()

        self.dfdx_fn = jax.jit(jax.jacfwd(self.f, argnums=0))
        self.dfdu_fn = jax.jit(jax.jacfwd(self.f, argnums=1))

        self.dRdx_fn = jax.jit(jax.jacfwd(self.R, argnums=0))
        self.dRdu_fn = jax.jit(jax.jacfwd(self.R, argnums=1))

        self.d2Rdx2_fn = jax.jit(jax.hessian(self.R, argnums=0))
        self.d2Rdxdu_fn = jax.jit(jax.jacfwd(jax.jacfwd(self.R, argnums=0), argnums=1))
        self.d2Rdu2_fn = jax.jit(jax.hessian(self.R, argnums=1))

        self.dVdx_fn = jax.jit(jax.jacfwd(self.V, argnums=0))
        self.d2Vdx2_fn = jax.jit(jax.hessian(self.V, argnums=0))

        toc = time.time()
        # print(f"Time to differentiate: {toc - tic}")
        pass


    '''
    Continuous-time dynamics
    Returns 1D vector of size (n,)
    '''
    @abstractmethod
    def f(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        pass

    '''
    Jacobian of dynamics w.r.t. (A matrix)
    Returns 2D matrix of size (n,n)
    '''
    def dfdx(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.dfdx_fn(state, action)

    '''
    Jacobian of dynamics w.r.t. action (B matrix)
    Returns 2D matrix of size (n,m)
    '''
    def dfdu(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.dfdu_fn(state, action)

    '''
    Reward function
    Returns scalar value
    '''
    @abstractmethod
    def R(self, state: jnp.ndarray, action: jnp.ndarray) -> float:
        pass

    '''
    Jacobian of reward w.r.t. state
    Returns 1D vector of size (n,)
    '''
    def dRdx(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.dRdx_fn(state, action)

    '''
    Jacobian of reward w.r.t. action
    Returns 1D vector of size (m,)
    '''
    def dRdu(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.dRdu_fn(state, action)

    '''
    Hessian of reward w.r.t. state
    Returns 2D matrix of size (n,n)
    '''
    def d2Rdx2(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.d2Rdx2_fn(state, action)

    '''
    Second derivative cross terms of reward
    Returns 2D matrix of size (n,m)
    '''
    def d2Rdxdu(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.d2Rdxdu_fn(state, action)

    '''
    Hessian of reward w.r.t. action
    Returns 2D matrix of size (m,m)
    '''
    def d2Rdu2(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.d2Rdu2_fn(state, action)

    '''
    Terminal value function
    Returns scalar
    '''
    @abstractmethod
    def V(self, state: jnp.ndarray) -> float:
        pass

    '''
    Jacobian of value
    Returns 1D vector of size (n,)
    '''
    def dVdx(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.dVdx_fn(state)
    
    '''
    Hessian of value
    Returns 2D matrix of size (n,n)
    '''
    def d2Vdx2(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.d2Vdx2_fn(state)

    @abstractmethod
    def state_labels(self) -> List[str]:
        pass
    
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @abstractmethod
    def action_dim(self) -> int:
        pass

    @abstractmethod
    def X(self) -> jnp.ndarray:
        pass

    @abstractmethod
    def U(self) -> jnp.ndarray:
        pass