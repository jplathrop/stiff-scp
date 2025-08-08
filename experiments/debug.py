import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax
from tqdm import tqdm
from scipy.integrate import solve_ivp

from src.dubins_car import DubinsCar
from src.scp import SCPSubproblem

def integrate_ground_truth(car, x0, us, dt):
    horizon = us.shape[0]

    def f(t, x):
        k = min(int(np.floor(t / dt)), horizon - 1)
        u = us[k]
        return np.asarray(car.f(x, u), dtype=float)

    t0, tf = 0.0, float(horizon) * dt
    t_eval = np.linspace(t0, tf, horizon + 1)
    sol = solve_ivp(f, (t0, tf), x0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.y.T

def debug_jacobians():
    car = DubinsCar(xd = jax.numpy.array([1.0, 1.0, 0.0]))
    x = np.zeros((3,))
    u = np.zeros((2,))
    r = car.R(x, u)
    drdx = car.dRdx(x, u)
    dr2dx2 = car.d2Rdx2(x, u)
    drdu = car.dRdu(x, u)
    dr2du2 = car.d2Rdu2(x, u)
    dr2dxdu = car.d2Rdxdu(x, u)
    print(f"r: {r}")
    print(f"drdx: {drdx}")
    print(f"drdu: {drdu}")
    print(f"dr2dx2: {dr2dx2}")
    print(f"dr2du2: {dr2du2}")
    print(f"d2rdxdu: {dr2dxdu}")

def main():
    horizon = 15
    dt = 0.2
    car = DubinsCar(xd = jax.numpy.array([1.0, 1.0, 0.0]))
    method = "implicit_midpoint"
    scp = SCPSubproblem(car, horizon, dt, dynamics_integration=method)

    # xnom = np.zeros((horizon+1, 3))
    unom = np.ones((horizon, 2)) * 0.01
    xnom = np.zeros((horizon+1, 3))
    def step(state, action):
        return state + dt * car.f(state, action)
    for k in range(horizon):
        xnom[k+1, :] = step(xnom[k, :], unom[k, :])

    num_iterations = 30
    xs = np.empty((num_iterations + 1, horizon+1, 3))
    us = np.empty((num_iterations + 1, horizon, 2))
    xs[0, :, :] = xnom
    us[0, :, :] = unom

    epsilon0 = 1.0
    beta = 0.9
    for i in tqdm(range(num_iterations), desc="Running SCP"):
        epsilon = epsilon0 * beta ** i
        xsoln, usoln, cost = scp.solve(xs[i, :, :], us[i, :, :], epsilon=epsilon, cvxpy_kwargs={"verbose": True})
        # rollout
        xsoln[0, :] = xs[0, 0, :]
        for k in range(horizon):
            xsoln[k+1, :] = step(xsoln[k, :], usoln[k, :])
        # print(f"xsoln: {xsoln}")
        # print(f"usoln: {usoln}")
        print(f"cost: {cost}")
        xs[i+1, :, :] = xsoln
        us[i+1, :, :] = usoln
    
    # Compute ground-truth rollout for the final optimized controls
    x_opt = xs[-1]
    u_opt = us[-1]
    x_gt = integrate_ground_truth(car, x_opt[0], u_opt, dt)
    mse = float(np.mean((x_opt - x_gt) ** 2))
    print(f"MSE(opt_traj vs RK45 ground truth): {mse:.6e}")

    fig, ax = plt.subplots()
    cmap = matplotlib.colormaps["plasma"]
    for i, xtraj in enumerate(xs):
        if i % 5 == 0 or i == len(xs) - 1:
            ax.plot(xtraj[:,0], xtraj[:,1], label=f"iteration {i}", c = cmap(i / len(xs)))
    # Overlay ground truth final trajectory
    ax.plot(x_gt[:,0], x_gt[:,1], "k--", linewidth=2.0, label="RK45 ground truth")
    ax.legend()
    ax.set_aspect("equal")
    plt.savefig(f"scp_trajectory_{method}.png")
    plt.show()

    return

if __name__ == "__main__":
    main()
    # debug_jacobians()