import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from collections.abc import Callable
from typing import List

from src.systems.system import System

def rollout(x0: np.ndarray, input_trajectory: np.ndarray, f: Callable, dfdx: Callable, dts: np.ndarray | List, dynamics_integration: str, tol: float = 1e-4):
    assert dynamics_integration in ["forward_euler", "backward_euler", "matrix_exponential", "implicit"]
    assert len(dts) == len(input_trajectory)
    horizon = len(dts)
    state_dim = x0.size

    state_trajectory = np.empty((horizon+1, x0.size))
    state_trajectory[0,:] = x0
    if dynamics_integration in ["forward_euler", "matrix_exponential"]:
        for k in range(horizon):
            state_trajectory[k+1,:] = state_trajectory[k,:] + dts[k] * f(state_trajectory[k,:], input_trajectory[k,:])
    if dynamics_integration == "backward_euler":
        for k in range(horizon):
            xk = state_trajectory[k,:]
            uk = input_trajectory[k,:]
            def backward_euler_residual(xkp1):
                return np.dot(
                    xkp1 - (xk + dts[k] * f(xkp1, uk)),
                    xkp1 - (xk + dts[k] * f(xkp1, uk))
                )
            def dbackward_euler_residualdx(xkp1):
                return (np.eye(state_dim) - dts[k] * dfdx(xkp1, uk)) @ (xkp1 - (xk + dts[k] * f(xkp1, uk)))
            res = minimize(backward_euler_residual, xk, jac=dbackward_euler_residualdx, method='BFGS', tol=tol)
            if res['success']:
                state_trajectory[k+1,:] = res['x']
            else:
                print(f"Backward euler minimize error. opt result: ", res)
                exit(1)
    if dynamics_integration == "implicit":
        for k in range(horizon):
            xk = state_trajectory[k,:]
            uk = input_trajectory[k,:]
            fxkuk = f(xk, uk)
            def implicit_trapz_residual(xkp1):
                return np.dot(
                    xkp1 - (xk + dts[k] * (fxkuk + f(xkp1, uk)) / 2.0),
                    xkp1 - (xk + dts[k] * (fxkuk + f(xkp1, uk)) / 2.0)
                )
            def dimplicit_trapz_residualdx(xkp1):
                return (np.eye(state_dim) - dts[k] * dfdx(xkp1, uk) / 2.0) @ (xkp1 - (xk + dts[k] * (fxkuk + f(xkp1, uk)) / 2.0))
            res = minimize(implicit_trapz_residual, xk, jac=dimplicit_trapz_residualdx, method='BFGS', tol=tol)
            if res['success']:
                state_trajectory[k+1,:] = res['x']
            else:
                print(f"Implicit trapezoidal minimize error. opt result: ", res)
                exit(1)

    return np.array(state_trajectory)

def resample_trajectory(traj: np.ndarray, old_dt: np.ndarray | List, new_dt: np.ndarray | List):
    dim = traj.shape[1]
    assert len(old_dt) == len(traj)
    old_ts = np.cumsum(old_dt)
    new_ts = np.cumsum(new_dt)
    new_traj = np.empty((len(new_dt), dim))
    for dim_idx in range(dim):
        new_traj[:, dim_idx] = np.interp(new_ts, old_ts, traj[:, dim_idx])
    return new_traj

def cumulative_cost(system: System, state_trajectory: np.ndarray, input_trajectory: np.ndarray, dts: np.ndarray | List) -> float:
    assert len(state_trajectory) == len(input_trajectory) + 1
    assert len(input_trajectory) == len(dts)
    for dt in dts:
        assert dt > 0

    horizon = len(input_trajectory)
    total = 0.0
    for k in range(horizon):
        total += dts[k] * float(np.asarray(system.R(state_trajectory[k], input_trajectory[k])))
    total += float(np.asarray(system.V(state_trajectory[horizon])))
    return total

def integrate_ground_truth(system: System, x0: np.ndarray, input_trajectory: np.ndarray, dts: np.ndarray | List):
    assert len(input_trajectory) == len(dts)
    for dt in dts:
        assert dt > 0

    times = np.cumsum(dts)
    def f(t, x):
        k = np.searchsorted(times, t)
        u = input_trajectory[k]
        return np.asarray(system.f(x, u), dtype=float)

    t0, tf = 0.0, times[-1]
    sol = solve_ivp(f, (t0, tf), x0, method="RK45", t_eval=np.hstack([0.0, times]), rtol=1e-8, atol=1e-10)
    return sol.y.T