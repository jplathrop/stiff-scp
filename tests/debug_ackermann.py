import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax
from tqdm import tqdm

from src.systems.ackermann_car import AckermannCar
from src.solvers.scp import SCPSubproblem
from src.util import rollout, integrate_ground_truth, cumulative_cost


def run_traj_opt(method: str):
    horizon = 25
    dt = 0.2
    car = AckermannCar(xd=jax.numpy.array([1.0, 1.0, 0.0, 0.0]))
    scp = SCPSubproblem(car, horizon, dt, dynamics_integration=method)

    unom = np.empty((horizon, 2))
    unom[:, 0] = 1.0
    unom[:, 1] = 0.5
    xnom = np.zeros((horizon + 1, 4))

    def step(state, action):
        return state + dt * car.f(state, action)

    for k in range(horizon):
        xnom[k + 1, :] = step(xnom[k, :], unom[k, :])

    num_iterations = 24
    xs = np.empty((num_iterations + 1, horizon + 1, 4))
    us = np.empty((num_iterations + 1, horizon, 2))
    xs[0, :, :] = xnom
    us[0, :, :] = unom

    epsilon0 = 1.0
    beta = 0.95
    for i in tqdm(range(num_iterations), desc=f"Running SCP ({method})"):
        epsilon = epsilon0 * beta ** i
        xsoln, usoln, cost = scp.solve(xs[i, :, :], us[i, :, :], epsilon=epsilon, cvxpy_kwargs={"verbose": True})
        # rollout with the car dynamics for consistency
        xs[i + 1, :, :] = rollout(xs[0, 0, :], usoln, car.f, car.dfdx, [dt] * horizon, dynamics_integration=method)
        us[i + 1, :, :] = usoln

    # Ground-truth rollout and MSE
    x_opt = xs[-1]
    u_opt = us[-1]
    x_gt = integrate_ground_truth(car, x_opt[0], u_opt, [dt] * horizon)
    mse = float(np.mean((x_opt - x_gt) ** 2))
    gt_cost = cumulative_cost(car, x_gt, u_opt, [dt] * horizon)
    print(f"[{method}] MSE(opt_traj vs RK45 ground truth): {mse:.6e}")

    # Plot in XY
    fig, ax = plt.subplots()
    cmap = matplotlib.colormaps["plasma"]
    for i, xtraj in enumerate(xs):
        if i % 5 == 0 or i == len(xs) - 1:
            ax.plot(xtraj[:, 0], xtraj[:, 1], label=f"iter {i}", c=cmap(i / len(xs)))
    ax.plot(x_gt[:, 0], x_gt[:, 1], "k--", linewidth=2.0, label="RK45 ground truth")
    ax.legend()
    ax.set_aspect("equal")
    plt.title(f"Ackermann SCP ({method}), MSE: {mse:.3e}, GT cost: {gt_cost:.3f}")
    plt.savefig(f"../plots/scp_trajectory_ackermann_{method}.png")


if __name__ == "__main__":
    # run_traj_opt("implicit")
    # run_traj_opt("matrix_exponential")
    run_traj_opt("forward_euler")
    # run_traj_opt("backward_euler")

