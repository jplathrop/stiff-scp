import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax
from tqdm import tqdm

from src.systems.dubins_car import DubinsCar
from src.solvers.adaptive_scp import AdaptiveSCPSubproblem
from src.util import rollout, cumulative_cost, integrate_ground_truth

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

def run_traj_opt(method):
    horizon = 15
    dt = 0.2
    car = DubinsCar(xd = jax.numpy.array([1.0, 1.0, 0.0]))
    scp = AdaptiveSCPSubproblem(car, horizon, dynamics_integration=method)

    # xnom = np.zeros((horizon+1, 3))
    unom = np.ones((horizon, 2)) * 0.01
    xnom = np.zeros((horizon+1, 3))
    def step(state, action):
        return state + dt * car.f(state, action)
    for k in range(horizon):
        xnom[k+1, :] = step(xnom[k, :], unom[k, :])

    num_iterations = 25
    xs = np.empty((num_iterations + 1, horizon+1, 3))
    us = np.empty((num_iterations + 1, horizon, 2))
    xs[0, :, :] = xnom
    us[0, :, :] = unom

    epsilon0 = 1.0
    beta = 0.9 # wait this is actually important, 1.0 makes the traj opt not converge
    for i in tqdm(range(num_iterations), desc="Running SCP"):
        epsilon = epsilon0 * beta ** i
        xsoln, usoln, cost = scp.solve(xs[i, :, :], us[i, :, :], [dt] * horizon, epsilon=epsilon, cvxpy_kwargs={"verbose": False})
        # rollout
        xsoln[0, :] = xs[0, 0, :]
        xs[i + 1, :, :] = rollout(xs[0, 0, :], usoln, car.f, car.dfdx, [dt]*horizon, dynamics_integration=method)
        us[i + 1, :, :] = usoln
    
    # Compute ground-truth rollout for the final optimized controls
    x_opt = xs[-1]
    u_opt = us[-1]
    x_gt = integrate_ground_truth(car, x_opt[0], u_opt, [dt]*horizon)
    mse = float(np.mean((x_opt - x_gt) ** 2))
    gt_cost = cumulative_cost(car, x_gt, u_opt, [dt]*horizon)
    print(f"[{method}] MSE(opt_traj vs RK45 ground truth): {mse:.6e}")
    print(f"  predicted cost:    {cumulative_cost(car, x_opt, u_opt, [dt]*horizon):.4e}")
    print(f"  ground truth cost: {gt_cost:.4e}")

    fig, ax = plt.subplots()
    cmap = matplotlib.colormaps["plasma"]
    for i, xtraj in enumerate(xs):
        if i % 5 == 0 or i == len(xs) - 1:
            ax.plot(xtraj[:,0], xtraj[:,1], label=f"iteration {i}", c = cmap(i / len(xs)))
    # Overlay ground truth final trajectory
    ax.plot(x_gt[:,0], x_gt[:,1], "k--", linewidth=2.0, label="RK45 ground truth")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(f"SCP trajectory with {method} integration, MSE: {mse:.6e}")
    plt.savefig(f"../plots/scp_trajectory_{method}.png")
    return

if __name__ == "__main__":
    run_traj_opt("implicit")
    # run_traj_opt("matrix_exponential")
    # run_traj_opt("forward_euler")
    # run_traj_opt("backward_euler")
    # debug_jacobians()