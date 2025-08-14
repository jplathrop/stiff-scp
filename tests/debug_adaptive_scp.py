import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax
from tqdm import tqdm
from scipy.integrate import solve_ivp

from src.systems.ackermann_car import AckermannCar
from src.solvers.adaptive_scp import AdaptiveSCPSubproblem
from src.util import rollout, resample_trajectory, cumulative_cost, integrate_ground_truth


def run_traj_opt(method: str):
    horizon = 25
    car = AckermannCar(xd=jax.numpy.array([1.0, 1.0, 0.0, 0.0]))
    scp = AdaptiveSCPSubproblem(car, horizon, dynamics_integration=method)

    x0 = np.array([0, 0, 0, 0])
    u_initial_guess = np.empty((horizon, 2))
    u_initial_guess[:,0] = 1.0
    u_initial_guess[:,1] = 0.5
    dt_initial_guess = 0.2 * np.ones((horizon))

    num_iterations = 25
    xs = np.empty((num_iterations + 1, horizon + 1, 4))
    us = np.empty((num_iterations + 1, horizon, 2))
    dts = np.empty((num_iterations + 1, horizon))

    us[0, :, :] = u_initial_guess
    dts[0, :] = dt_initial_guess
    xs[0, :, :] = rollout(x0, u_initial_guess, car.f, car.dfdx, dt_initial_guess, method)

    epsilon0 = 1.0
    beta = 0.95
    dt1 = 0.2 * np.ones((horizon))
    dt2 = np.hstack((0.18*np.ones((12)), np.array([0.2]), 0.22*np.ones((12))))
    for i in tqdm(range(num_iterations), desc=f"Running Adaptive SCP ({method})"):
        if i%2 == 0:
            dt = dt1
        else:
            dt = dt2
        epsilon = epsilon0 * beta ** i
        # print(np.hstack([[0], dts[i, :]]))
        # print(np.hstack([[0], dt]))
        xnom = resample_trajectory(xs[i, :, :], np.hstack([[0], dts[i, :]]), np.hstack([[0], dt]))
        unom = resample_trajectory(us[i, :, :], dts[i, :], dt)
        xsoln, usoln, cost = scp.solve(xnom, unom, dt, epsilon=epsilon, cvxpy_kwargs={"verbose": False})
        xs[i + 1, :, :] = rollout(x0, usoln, car.f, car.dfdx, dt, dynamics_integration=method)
        us[i + 1, :, :] = usoln
        dts[i + 1, :] = dt

    # Ground-truth rollout and MSE
    x_opt = xs[-1]
    u_opt = us[-1]
    x_gt = integrate_ground_truth(car, x0, u_opt, dt)
    mse = float(np.mean((x_opt - x_gt) ** 2))
    gt_cost = cumulative_cost(car, x_gt, u_opt, dt)
    print(f"[{method}] MSE(opt_traj vs RK45 ground truth): {mse:.6e}")
    print(f"  predicted cost:    {cumulative_cost(car, x_opt, u_opt, dt):.4e}")
    print(f"  ground truth cost: {gt_cost:.4e}")

    # Plot in XY
    fig, ax = plt.subplots()
    cmap = matplotlib.colormaps["plasma"]
    for i, xtraj in enumerate(xs):
        if i % 5 == 0 or i == len(xs) - 1:
            ax.plot(xtraj[:, 0], xtraj[:, 1], label=f"iter {i}", c=cmap(i / len(xs)))
    ax.plot(x_gt[:, 0], x_gt[:, 1], "k--", linewidth=2.0, label="RK45 ground truth")
    ax.legend()
    ax.set_aspect("equal")
    plt.title(f"Ackermann Adaptive SCP ({method}), MSE: {mse:.3e}, GT cost: {gt_cost:.3f}")
    plt.savefig(f"../plots/adaptive_scp_trajectory_ackermann_{method}.png")


if __name__ == "__main__":
    run_traj_opt("implicit")
    run_traj_opt("matrix_exponential")
    run_traj_opt("forward_euler")
    run_traj_opt("backward_euler")

