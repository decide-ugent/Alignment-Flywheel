"""
Fitted Q-iteration on the learned MoE reward model.

Discretises the PointMaze (x, y) space into a grid.
Reward at each cell comes from the 4-D autoencoder (queried at a
moderate velocity) plus norm corrections (wall, velocity, path proximity).
Value iteration finds the optimal policy; the result is plotted as
arrows overlaid on the value landscape.

Produces q_iteration_policy.png.
"""

import os, sys
import numpy as np
import torch
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
sys.path.insert(0, ROOT)

from patching.environments.benchmark_layouts import LAYOUTS
from IIRL.models import MixtureOfExperts

# ── constants ────────────────────────────────────────────────────
GRID_SIZE = 8
CELL = 1.0
XY_RANGE = (-4.0, 4.0)

# Q-iteration params
GRID_RES = 80                 # cells per axis for the value grid
GAMMA = 0.97                  # discount factor
N_ITERS = 300                 # value-iteration sweeps
STEP_SIZE = 0.15              # how far one action moves the agent

# Norm params (same as plot_strictness.py)
WALL_CLEARANCE = 0.55
PATH_CLEARANCE = 0.2
PATH_SIGMA = 0.4
MAX_VEL_COMPONENT = 3.0
MIN_SPEED = 0.5

# Goal (tile (2,6) in layout-01)
GOAL_XY = np.array([2.5, 1.5], dtype=np.float32)
GOAL_RADIUS = 0.4
GOAL_REWARD = 5.0             # bonus for reaching goal

# Start region center (tiles (5-6, 1-2))
START_XY = np.array([-2.0, -1.5], dtype=np.float32)

# Query velocity for reward model (moderate, in the "PASS" range)
QUERY_VX, QUERY_VY = 1.5, 1.0

# 8 discrete actions (king moves)
ACTIONS = np.array([
    [ 0,  1],   # up
    [ 0, -1],   # down
    [-1,  0],   # left
    [ 1,  0],   # right
    [ 1,  1],   # up-right
    [-1,  1],   # up-left
    [ 1, -1],   # down-right
    [-1, -1],   # down-left
], dtype=np.float32)
ACTIONS = ACTIONS / np.linalg.norm(ACTIONS, axis=1, keepdims=True) * STEP_SIZE
ACTION_NAMES = ['U', 'D', 'L', 'R', 'UR', 'UL', 'DR', 'DL']


# ── helpers ──────────────────────────────────────────────────────
def tile_center(row, col):
    x = col - (GRID_SIZE - 1) / 2.0
    y = (GRID_SIZE - 1) / 2.0 - row
    return np.array([x, y], dtype=np.float32)


def wall_cells_xy(layout_map):
    cells = []
    for r, row in enumerate(layout_map):
        for c, v in enumerate(row):
            if v == 1:
                cells.append(tile_center(r, c))
    return np.array(cells, dtype=np.float32)


def is_wall(xy, walls, clearance):
    """Check if a single point is too close to any wall."""
    dists = np.sqrt(((xy - walls) ** 2).sum(axis=1))
    return dists.min() < clearance


def compute_reward_grid(model, walls, demo_pos, res=GRID_RES):
    """Compute per-cell reward from the MoE model + norms."""
    axis = np.linspace(*XY_RANGE, res)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    # raw model reward at query velocity
    vel = np.full((xy.shape[0], 2), [QUERY_VX, QUERY_VY], dtype=np.float32)
    inp = torch.from_numpy(np.concatenate([xy, vel], axis=1))
    with torch.no_grad():
        out = model(inp)
    mse = ((out - inp) ** 2).sum(dim=1).numpy()
    lo, hi = mse.min(), np.percentile(mse, 98)
    reward = 1.0 - np.clip((mse - lo) / (hi - lo + 1e-9), 0, 1)

    # wall mask
    dists_to_wall = cdist(xy, walls).min(axis=1)
    wall_mask = (dists_to_wall > WALL_CLEARANCE).astype(np.float32)
    reward *= wall_mask

    # path-proximity mask
    sub_demo = demo_pos[::5]
    dists_to_demo = cdist(xy, sub_demo).min(axis=1)
    excess = np.maximum(0.0, dists_to_demo - PATH_CLEARANCE)
    path_mask = np.exp(-(excess ** 2) / (PATH_SIGMA ** 2))
    reward *= path_mask

    # goal bonus
    dist_to_goal = np.sqrt(((xy - GOAL_XY) ** 2).sum(axis=1))
    goal_bonus = GOAL_REWARD * np.exp(-0.5 * (dist_to_goal / GOAL_RADIUS) ** 2)
    reward += goal_bonus

    return xx, yy, reward.reshape(res, res)


def run_value_iteration(reward_grid, walls, res, n_iters, gamma):
    """Tabular value iteration on the discretised grid.

    Returns V(s) and the greedy policy π(s) (action index per cell).
    """
    axis = np.linspace(*XY_RANGE, res)
    dx = axis[1] - axis[0]

    V = reward_grid.copy()
    policy = np.zeros((res, res), dtype=int)

    for it in range(n_iters):
        V_new = np.copy(reward_grid)
        for ri in range(res):
            for ci in range(res):
                x, y = axis[ci], axis[ri]
                # skip wall cells
                dw = np.sqrt(((np.array([x, y]) - walls) ** 2).sum(axis=1)).min()
                if dw < WALL_CLEARANCE:
                    V_new[ri, ci] = 0.0
                    continue

                best_val = -1e9
                best_a = 0
                for ai, action in enumerate(ACTIONS):
                    nx, ny = x + action[0], y + action[1]
                    # check if next state is in bounds and not a wall
                    if nx < XY_RANGE[0] or nx > XY_RANGE[1]:
                        continue
                    if ny < XY_RANGE[0] or ny > XY_RANGE[1]:
                        continue
                    nw = np.sqrt(((np.array([nx, ny]) - walls) ** 2).sum(axis=1)).min()
                    if nw < WALL_CLEARANCE:
                        continue
                    # bilinear interpolation of V at (nx, ny)
                    # grid indices
                    fi = (ny - XY_RANGE[0]) / dx
                    fj = (nx - XY_RANGE[0]) / dx
                    i0 = int(np.clip(np.floor(fi), 0, res - 2))
                    j0 = int(np.clip(np.floor(fj), 0, res - 2))
                    di = fi - i0
                    dj = fj - j0
                    v_next = (V[i0, j0] * (1 - di) * (1 - dj) +
                              V[i0, j0+1] * (1 - di) * dj +
                              V[i0+1, j0] * di * (1 - dj) +
                              V[i0+1, j0+1] * di * dj)
                    q_val = reward_grid[ri, ci] + gamma * v_next
                    if q_val > best_val:
                        best_val = q_val
                        best_a = ai

                V_new[ri, ci] = best_val
                policy[ri, ci] = best_a

        delta = np.abs(V_new - V).max()
        V = V_new
        if it % 50 == 0 or it == n_iters - 1:
            print(f"  iter {it:3d}  max_delta={delta:.6f}")
        if delta < 1e-5:
            print(f"  Converged at iter {it}")
            break

    return V, policy


def draw_walls(ax, layout_map, alpha=0.45):
    for r, row in enumerate(layout_map):
        for c, v in enumerate(row):
            if v == 1:
                cx, cy = tile_center(r, c)
                ax.add_patch(plt.Rectangle(
                    (cx - CELL/2, cy - CELL/2), CELL, CELL,
                    fc='grey', ec='none', alpha=alpha))


def draw_policy_arrows(ax, policy, res, stride=4):
    """Draw arrows showing the greedy policy."""
    axis = np.linspace(*XY_RANGE, res)
    for ri in range(0, res, stride):
        for ci in range(0, res, stride):
            x, y = axis[ci], axis[ri]
            a = policy[ri, ci]
            dx, dy = ACTIONS[a]
            # scale for visibility
            scale = 0.25
            ax.arrow(x, y, dx * scale / STEP_SIZE, dy * scale / STEP_SIZE,
                     head_width=0.06, head_length=0.03,
                     fc='white', ec='white', alpha=0.8, linewidth=0.5)


def simulate_trajectory(policy, walls, start, res, max_steps=200):
    """Roll out the greedy policy from a start point."""
    axis = np.linspace(*XY_RANGE, res)
    dx = axis[1] - axis[0]
    traj = [start.copy()]
    pos = start.copy()

    for _ in range(max_steps):
        # look up policy at current pos
        fi = (pos[1] - XY_RANGE[0]) / dx
        fj = (pos[0] - XY_RANGE[0]) / dx
        ri = int(np.clip(np.round(fi), 0, res - 1))
        ci = int(np.clip(np.round(fj), 0, res - 1))
        a = policy[ri, ci]
        new_pos = pos + ACTIONS[a]

        # wall check
        dw = np.sqrt(((new_pos - walls) ** 2).sum(axis=1)).min()
        if dw < WALL_CLEARANCE:
            break
        # bounds check
        if (new_pos < XY_RANGE[0]).any() or (new_pos > XY_RANGE[1]).any():
            break

        pos = new_pos
        traj.append(pos.copy())

        # goal check
        if np.sqrt(((pos - GOAL_XY) ** 2).sum()) < GOAL_RADIUS:
            break

    return np.array(traj)


# ── data loading ─────────────────────────────────────────────────
DEMO_DIR = os.path.join(HERE, "pointmaze_data", "pointmaze_benchmark_demos")


def load_demo_pos():
    per_layout = {}
    for lid in sorted(os.listdir(DEMO_DIR)):
        lp = os.path.join(DEMO_DIR, lid)
        if not os.path.isdir(lp):
            continue
        obs = []
        for ep in sorted(os.listdir(lp)):
            ep_path = os.path.join(lp, ep)
            if os.path.isdir(ep_path):
                obs.append(np.load(os.path.join(ep_path, "observations.npy")))
        arr = np.concatenate(obs, axis=0).astype(np.float32)
        per_layout[lid] = arr[:, :2]
    return per_layout


# ── main ─────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # load model
    ckpt = os.path.join(HERE, "checkpoints", "moe_5E_8B_full.pt")
    model = MixtureOfExperts(input_dim=4, bottleneck_dim=8, num_experts=5)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint {ckpt}")

    per_layout = load_demo_pos()
    all_demo_pos = np.concatenate(list(per_layout.values()), axis=0)

    ref_layout = LAYOUTS["layout-01"]["map"]
    walls = wall_cells_xy(ref_layout)

    # step 1: compute reward grid
    print("Computing reward grid …")
    xx, yy, reward_grid = compute_reward_grid(model, walls, all_demo_pos)
    print(f"  Reward: min={reward_grid.min():.3f}  max={reward_grid.max():.3f}  "
          f"mean={reward_grid.mean():.3f}")

    # step 2: value iteration
    print(f"Running value iteration ({N_ITERS} iters, γ={GAMMA}) …")
    V, policy = run_value_iteration(reward_grid, walls, GRID_RES, N_ITERS, GAMMA)
    print(f"  V: min={V[V > -1e8].min():.3f}  max={V.max():.3f}")

    # clip V for plotting (wall cells have -1e9)
    V_plot = V.copy()
    V_plot[V_plot < -1e8] = np.nan

    # step 3: simulate trajectories from several start positions
    starts = [
        START_XY,
        START_XY + np.array([0.3, 0.0]),
        START_XY + np.array([0.0, 0.3]),
        START_XY + np.array([-0.3, 0.3]),
    ]
    trajectories = []
    for s in starts:
        traj = simulate_trajectory(policy, walls, s, GRID_RES)
        trajectories.append(traj)
        reached = np.sqrt(((traj[-1] - GOAL_XY) ** 2).sum()) < GOAL_RADIUS
        print(f"  Start ({s[0]:.1f},{s[1]:.1f}): {len(traj)} steps, "
              f"reached goal: {reached}")

    # step 4: plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # panel 1: reward
    ax = axes[0]
    ax.contourf(xx, yy, reward_grid, levels=30, cmap='viridis')
    draw_walls(ax, ref_layout)
    ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=1.5, zorder=10)
    ax.set_xlim(*XY_RANGE); ax.set_ylim(*XY_RANGE); ax.set_aspect('equal')
    ax.set_title('Reward (MoE + norms + goal)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # panel 2: value function + policy arrows
    ax = axes[1]
    v_min = np.nanmin(V_plot)
    v_max = np.nanmax(V_plot)
    levels = np.linspace(v_min, v_max, 30)
    cf = ax.contourf(xx, yy, V_plot, levels=levels, cmap='inferno')
    draw_walls(ax, ref_layout)
    draw_policy_arrows(ax, policy, GRID_RES, stride=4)
    ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=1.5, zorder=10)
    ax.set_xlim(*XY_RANGE); ax.set_ylim(*XY_RANGE); ax.set_aspect('equal')
    ax.set_title(f'Value function (γ={GAMMA})', fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(cf, ax=ax, shrink=0.8, label='V(s)')

    # panel 3: trajectories overlaid on value
    ax = axes[2]
    ax.contourf(xx, yy, V_plot, levels=levels, cmap='inferno', alpha=0.5)
    draw_walls(ax, ref_layout)
    # demo scatter
    for _lid, obs in per_layout.items():
        ax.scatter(obs[:, 0], obs[:, 1], s=0.4, alpha=0.08, c='cyan',
                   edgecolors='none')
    # simulated trajectories
    colours = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
    for traj, col in zip(trajectories, colours):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=col, linewidth=2.5,
                alpha=0.9, zorder=5)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=col, markersize=8,
                markeredgecolor='white', zorder=6)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=col, markersize=8,
                markeredgecolor='white', zorder=6)
    ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=1.5, zorder=10)
    ax.set_xlim(*XY_RANGE); ax.set_ylim(*XY_RANGE); ax.set_aspect('equal')
    ax.set_title('Learned policy trajectories', fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.suptitle('Q-iteration on MoE reward model  (layout-01)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()

    outpath = os.path.join(HERE, "q_iteration_policy.png")
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    print(f"\nFigure saved → {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
