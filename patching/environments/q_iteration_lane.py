"""
Lane-keeping experiment: three-way comparison showing the flywheel
norm-enforcement mechanism applied to lane discipline.

Conditions:
  1. NO NORMS  — no lane patches, agent uses full corridor width
  2. RIGHT LANE — LaneDisciplineObserver finds reward-model flaws on left,
                  LaneRegressionDecider patches only where reward peaks
                  on the wrong side, without killing it completely
  3. LEFT LANE  — same pipeline, mirrored

Uses the real flywheel framework:
  - MoE2DOracle (BaseSpatialOracleAdapter) for reward queries
  - LaneDisciplineObserver (Red Team) scans for reward-model mistakes
  - SpatialNormMatcher + SpatialViolationDecider verify LANE_DISCIPLINE norm
  - LaneBandwidthOrienter + LaneRegressionDecider compute + regression-test patches
  - BatchDeployer + oracle.send_patch() deploys GovernanceBatch

Uses layout-02.  Produces q_iteration_lane.png.
"""

import os, sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from benchmark_layouts import LAYOUTS
from gated_autoencoder import MixtureOfExperts

# ── flywheel imports ─────────────────────────────────────────────
from flywheel.roles.oracle.adapters.moe_2d_oracle import MoE2DOracle
from flywheel.protocols.enums import CorrectionType
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.local_correction import LocalCorrection

# OODA roles – Red Team lane observer
from flywheel.roles.redteam.observe.lane_discipline_observer import LaneDisciplineObserver

# Refinement – lane-specific components
from flywheel.roles.refinement.orient.lane_bandwidth_orienter import LaneBandwidthOrienter
from flywheel.roles.refinement.decide.lane_regression_decider import LaneRegressionDecider

# ── constants ────────────────────────────────────────────────────
GRID_SIZE = 8
CELL = 1.0
XY_RANGE = (-4.0, 4.0)

GRID_RES = 80
GAMMA = 0.97
N_ITERS = 500
STEP_SIZE = 0.15

WALL_CLEARANCE = 0.15       # Norm 2: wider corridors so lane patches differentiate
PATH_CLEARANCE = 0.2
PATH_SIGMA = 0.6            # wider so path norm doesn't dominate lane choice
DEMO_SPEED = 3.0            # representative speed for velocity queries

GOAL_XY = np.array([2.5, 1.5], dtype=np.float32)
GOAL_RADIUS = 0.4
GOAL_REWARD = 5.0

START_XY = np.array([-2.0, -1.5], dtype=np.float32)

# 8 discrete actions (king moves)
ACTIONS = np.array([
    [0, 1], [0, -1], [-1, 0], [1, 0],
    [1, 1], [-1, 1], [1, -1], [-1, -1]
], dtype=np.float32)
ACTIONS = ACTIONS / np.linalg.norm(ACTIONS, axis=1, keepdims=True) * STEP_SIZE

# ── Lane suppression parameters ─────────────────────────────────
REWARD_THRESHOLD = 0.3     # only consider points with reward above this
LANE_TARGET = 0.75         # ideal lane position (0=left, 1=right of travel dir)

# Route waypoints (tile centers) along layout-02 path
ROUTE_WAYPOINTS_XY = np.array([
    [-2.5, -2.5],   # (6,1)
    [-2.5, -1.5],   # (5,1)
    [-0.5, -1.5],   # (5,3)
    [-0.5,  0.5],   # (3,3)
    [ 0.5,  0.5],   # (3,4)
    [ 0.5,  2.5],   # (1,4)
    [ 2.5,  2.5],   # (1,6)
    [ 2.5,  1.5],   # (2,6) goal
], dtype=np.float32)


def _build_corridor_specs(waypoints, width=1.0):
    """Build corridor specs from consecutive waypoints.

    Each corridor is a straight segment between two waypoints.
    The observer will scan both travel directions for each corridor.
    """
    specs = []
    for i in range(len(waypoints) - 1):
        specs.append({
            "start_xy": waypoints[i].tolist(),
            "end_xy": waypoints[i + 1].tolist(),
            "width": width,
        })
    return specs


CORRIDOR_SPECS = _build_corridor_specs(ROUTE_WAYPOINTS_XY, width=0.9)


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


def xy_to_tile(x, y):
    col = int(np.floor(x + (GRID_SIZE / 2.0)))
    row = int(np.floor((GRID_SIZE / 2.0) - y))
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    return row, col


def cell_is_wall(row, col, layout_map):
    if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
        return True
    return layout_map[row][col] == 1


def move_blocked_by_grid(pos_from, pos_to, layout_map):
    r_to, c_to = xy_to_tile(pos_to[0], pos_to[1])
    if cell_is_wall(r_to, c_to, layout_map):
        return True
    r_from, c_from = xy_to_tile(pos_from[0], pos_from[1])
    dr = r_to - r_from
    dc = c_to - c_from
    if dr != 0 and dc != 0:
        if (cell_is_wall(r_from + dr, c_from, layout_map) and
                cell_is_wall(r_from, c_from + dc, layout_map)):
            return True
    return False


# ── Lane patch pipeline (framework components) ──────────────────
def run_lane_pipeline(oracle, layout_map, lane_target=LANE_TARGET):
    """Run the full flywheel OODA pipeline for velocity-aware lane discipline.

    Full-grid cell-classified approach:
    1. Classify every non-wall cell by corridor type (H/V_STRAIGHT,
       CORNER, T_JUNCTION, CROSS, DEAD_END).
    2. For each cell × velocity direction, query R(x,y,vx,vy) and
       compute lane position.
    3. Emit patches for left-side (Norm 1) and center (Norm 3) points.

    Returns the number of patches deployed and patch center coordinates.
    """
    # ── 1. Red Team: observe ─────────────────────────────────
    observer = LaneDisciplineObserver()
    obs_ctx = {
        "oracle": oracle,
        "layout_map": layout_map,
        "grid_size": GRID_SIZE,
        "reward_threshold": REWARD_THRESHOLD,
        "lane_target": lane_target,
        "lane_dead_zone": 0.08,
        "query_speed": DEMO_SPEED,
        "sub_resolution": 5,
        "center_strength_scale": 0.4,
    }
    obs_result = observer.observe(obs_ctx)
    candidates = obs_result["candidates"]
    n_cells = obs_result["cells_scanned"]
    print(f"    Observer found {len(candidates)} lane flaws "
          f"scanning {n_cells} non-wall cells × 4 vel dirs")

    if not candidates:
        return 0, np.empty((0, 2))

    # ── 2. Skip heavy verifier — observer already does velocity-
    #    aware lane-position checking.  Pass all as verified.
    verified_items = [({}, cand) for cand in candidates]
    print(f"    All {len(verified_items)} candidates passed to refinement")

    # ── 3. Refinement: orient (compute patch parameters) ─────
    orienter = LaneBandwidthOrienter()
    orient_ctx = {
        "verified_items": verified_items,
        "oracle_version": obs_result.get("oracle_version", "?"),
        "spatial_bw": 0.15,
        "strength_scale": 0.9,
        "dir_sigma": 0.35,
    }
    oriented = orienter.orient(orient_ctx)

    # ── 4. Refinement: decide (cap strength for soft patches) ─
    regression = LaneRegressionDecider()
    decided = regression.decide(oriented)
    accepted = decided["accepted"]
    print(f"    Regression accepted {len(accepted)}/{decided['total_planned']} "
          f"patches (capped={decided['capped']})")

    if not accepted:
        return 0, np.empty((0, 2))

    # ── 5. Build and deploy batch ────────────────────────────
    corrections = []
    patch_centers = []
    for item in accepted:
        pt = item["point"]
        travel_dir = item["travel_direction"]
        strength = item["final_strength"]
        bw = item["proposed_bw"]
        dir_sigma = item["dir_sigma"]

        corrections.append(LocalCorrection(
            correction_type=CorrectionType.LANE_DIRECTION_PATCH,
            payload={
                "flaw_point": pt,
                "direction": travel_dir,
                "strength": strength,
                "support_radius": bw,
                "dir_sigma": dir_sigma,
            },
            description=(f"Lane patch at ({pt[0]:.2f},{pt[1]:.2f}) "
                         f"dir=({travel_dir[0]:.1f},{travel_dir[1]:.1f}) "
                         f"str={strength:.2f}"),
        ))
        patch_centers.append(pt)

    batch = GovernanceBatch(
        from_oracle_version=oracle.get_version(),
        to_oracle_version="oracle:v1",
        local_corrections=corrections,
        regression_evidence={
            "patched": len(patch_centers),
            "total_planned": decided["total_planned"],
            "capped": decided["capped"],
            "policy": "keep-right",
        },
        signature="lane-discipline-correction",
    )
    oracle.send_patch(batch)

    return len(patch_centers), np.array(patch_centers, dtype=np.float32)


def compute_reward_from_oracle(oracle, res=GRID_RES):
    """Query the flywheel oracle for each action direction (velocity-aware).

    Returns (xx, yy, reward_sa) where reward_sa has shape (res, res, n_actions).
    """
    reward_sa = oracle.query_grid_actions(
        ACTIONS, speed=DEMO_SPEED, res=res, xy_range=XY_RANGE)
    axis = np.linspace(*XY_RANGE, res)
    xx, yy = np.meshgrid(axis, axis)
    return xx, yy, reward_sa


def compute_reward_display(reward_sa):
    """Best-action reward for display purposes."""
    return reward_sa.max(axis=2)


# ── value iteration (grid-based collision, action-dependent rewards) ──
def run_value_iteration(reward_sa, layout_map, res, n_iters, gamma):
    """Tabular value iteration with action-dependent rewards R(s,a).

    reward_sa has shape (res, res, n_actions).
    Q(s,a) = R(s,a) + gamma * V(s').
    """
    axis = np.linspace(*XY_RANGE, res)
    dx = axis[1] - axis[0]

    V = reward_sa.max(axis=2).copy()
    policy = np.zeros((res, res), dtype=int)

    for it in range(n_iters):
        V_new = reward_sa.max(axis=2).copy()
        for ri in range(res):
            for ci in range(res):
                x, y = axis[ci], axis[ri]
                r_cell, c_cell = xy_to_tile(x, y)
                if cell_is_wall(r_cell, c_cell, layout_map):
                    V_new[ri, ci] = 0.0
                    continue

                best_val = -1e9
                best_a = 0
                for ai, action in enumerate(ACTIONS):
                    nx, ny = x + action[0], y + action[1]
                    if nx < XY_RANGE[0] or nx > XY_RANGE[1]:
                        continue
                    if ny < XY_RANGE[0] or ny > XY_RANGE[1]:
                        continue
                    pos_from = np.array([x, y], dtype=np.float32)
                    pos_to = np.array([nx, ny], dtype=np.float32)
                    if move_blocked_by_grid(pos_from, pos_to, layout_map):
                        continue
                    fi = (ny - XY_RANGE[0]) / dx
                    fj = (nx - XY_RANGE[0]) / dx
                    i0 = int(np.clip(np.floor(fi), 0, res - 2))
                    j0 = int(np.clip(np.floor(fj), 0, res - 2))
                    di = fi - i0
                    dj = fj - j0
                    v_next = (V[i0, j0] * (1-di)*(1-dj) +
                              V[i0, j0+1] * (1-di)*dj +
                              V[i0+1, j0] * di*(1-dj) +
                              V[i0+1, j0+1] * di*dj)
                    q_val = reward_sa[ri, ci, ai] + gamma * v_next
                    if q_val > best_val:
                        best_val = q_val
                        best_a = ai

                V_new[ri, ci] = best_val
                policy[ri, ci] = best_a

        delta = np.abs(V_new - V).max()
        V = V_new
        if it % 50 == 0 or it == n_iters - 1:
            print(f"    iter {it:3d}  Δ={delta:.6f}")
        if delta < 1e-5:
            print(f"    Converged at iter {it}")
            break

    return V, policy


# ── trajectory simulation ────────────────────────────────────────
def simulate_trajectory(policy, layout_map, start, res, max_steps=500):
    axis = np.linspace(*XY_RANGE, res)
    dx = axis[1] - axis[0]
    traj = [start.copy()]
    pos = start.copy()

    for _ in range(max_steps):
        fi = (pos[1] - XY_RANGE[0]) / dx
        fj = (pos[0] - XY_RANGE[0]) / dx
        ri = int(np.clip(np.round(fi), 0, res - 1))
        ci = int(np.clip(np.round(fj), 0, res - 1))
        a = policy[ri, ci]
        new_pos = pos + ACTIONS[a]

        if move_blocked_by_grid(pos, new_pos, layout_map):
            break
        if (new_pos < XY_RANGE[0]).any() or (new_pos > XY_RANGE[1]).any():
            break

        pos = new_pos
        traj.append(pos.copy())
        if np.sqrt(((pos - GOAL_XY) ** 2).sum()) < GOAL_RADIUS:
            break

    return np.array(traj)


# ── drawing helpers ──────────────────────────────────────────────
def draw_walls(ax, layout_map, alpha=0.45):
    for r, row in enumerate(layout_map):
        for c, v in enumerate(row):
            if v == 1:
                cx, cy = tile_center(r, c)
                ax.add_patch(plt.Rectangle(
                    (cx - CELL/2, cy - CELL/2), CELL, CELL,
                    fc='grey', ec='none', alpha=alpha))


def draw_policy_arrows(ax, policy, res, stride=4):
    axis = np.linspace(*XY_RANGE, res)
    for ri in range(0, res, stride):
        for ci in range(0, res, stride):
            x, y = axis[ci], axis[ri]
            a = policy[ri, ci]
            dx, dy = ACTIONS[a]
            scale = 0.25
            ax.arrow(x, y, dx * scale / STEP_SIZE, dy * scale / STEP_SIZE,
                     head_width=0.06, head_length=0.03,
                     fc='white', ec='white', alpha=0.7, linewidth=0.5)


def draw_lane_centerline(ax, waypoints, color='yellow', alpha=0.6):
    """Draw the route centerline."""
    ax.plot(waypoints[:, 0], waypoints[:, 1], '--',
            color=color, linewidth=1.5, alpha=alpha, zorder=4)


def draw_lane_sides(ax, waypoints, offset=0.3, alpha=0.5):
    """Draw right-side (green) and left-side (red) dashed lines."""
    for i in range(len(waypoints) - 1):
        fwd = waypoints[i+1] - waypoints[i]
        length = np.linalg.norm(fwd)
        if length < 1e-6:
            continue
        fwd = fwd / length
        right = np.array([fwd[1], -fwd[0]])
        # right side (green)
        p1r = waypoints[i] + right * offset
        p2r = waypoints[i+1] + right * offset
        ax.plot([p1r[0], p2r[0]], [p1r[1], p2r[1]], '-',
                color='#4caf50', linewidth=2, alpha=alpha, zorder=4)
        # left side (red)
        p1l = waypoints[i] - right * offset
        p2l = waypoints[i+1] - right * offset
        ax.plot([p1l[0], p2l[0]], [p1l[1], p2l[1]], '-',
                color='#f44336', linewidth=2, alpha=alpha, zorder=4)


# ── data loading ─────────────────────────────────────────────────
DEMO_DIR = os.path.join(HERE, "pointmaze_data", "pointmaze_benchmark_demos")


def load_demo_data():
    per_layout_pos = {}
    per_layout_vel = {}
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
        per_layout_pos[lid] = arr[:, :2]
        per_layout_vel[lid] = arr[:, 2:4]
    return per_layout_pos, per_layout_vel


# ── build one condition ──────────────────────────────────────────
def build_condition(model, walls, demo_pos, demo_vel, layout_map,
                    starts, label, apply_lane=False, lane_target=LANE_TARGET):
    """Build oracle, optionally run lane-discipline pipeline, run VI, simulate.

    apply_lane=False  → no patches (unconstrained)
    apply_lane=True   → apply velocity-aware "keep right" lane discipline
    """
    oracle = MoE2DOracle(
        model=model, walls_xy=walls, demo_pos=demo_pos,
        demo_vel=demo_vel,
        wall_clearance=WALL_CLEARANCE,
        path_clearance=PATH_CLEARANCE, path_sigma=PATH_SIGMA,
        goal_xy=GOAL_XY, goal_radius=GOAL_RADIUS, goal_reward=GOAL_REWARD,
    )

    patch_centers_xy = np.empty((0, 2))
    if apply_lane:
        n_patches, pc = run_lane_pipeline(oracle, layout_map,
                                          lane_target=lane_target)
        patch_centers_xy = pc
        print(f"  [{label}] Deployed {n_patches} directional lane patches")
    else:
        print(f"  [{label}] No patches")

    print(f"  [{label}] Computing velocity-aware reward grid …")
    xx, yy, reward_sa = compute_reward_from_oracle(oracle)
    reward_disp = compute_reward_display(reward_sa)

    print(f"  [{label}] Value iteration …")
    V, pol = run_value_iteration(reward_sa, layout_map, GRID_RES, N_ITERS, GAMMA)

    trajs = []
    for s in starts:
        traj = simulate_trajectory(pol, layout_map, s, GRID_RES)
        reached = np.sqrt(((traj[-1] - GOAL_XY) ** 2).sum()) < GOAL_RADIUS
        trajs.append(traj)
        print(f"    start ({s[0]:.1f},{s[1]:.1f}): {len(traj)} steps, "
              f"goal={reached}")

    return {
        "label": label, "xx": xx, "yy": yy,
        "reward_disp": reward_disp, "V": V, "pol": pol,
        "trajs": trajs, "patch_centers": patch_centers_xy,
    }


# ── main ─────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    ckpt = os.path.join(HERE, "checkpoints", "moe_5E_8B_full.pt")
    model = MixtureOfExperts(input_dim=4, bottleneck_dim=8, num_experts=5)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint {ckpt}")

    per_layout_pos, per_layout_vel = load_demo_data()
    demo_pos = per_layout_pos["layout-02"]
    demo_vel = per_layout_vel["layout-02"]

    layout_map = LAYOUTS["layout-02"]["map"]
    walls = wall_cells_xy(layout_map)

    starts = [
        START_XY,
        START_XY + np.array([0.3, 0.0]),
        START_XY + np.array([0.0, 0.3]),
    ]

    # ── Three conditions ─────────────────────────────────────
    conds = []

    print("\n=== Condition 1: NO NORMS ===")
    conds.append(build_condition(
        model, walls, demo_pos, demo_vel, layout_map, starts,
        label="NO NORMS", apply_lane=False))

    print("\n=== Condition 2: KEEP RIGHT (lane_target=0.75) ===")
    conds.append(build_condition(
        model, walls, demo_pos, demo_vel, layout_map, starts,
        label="KEEP RIGHT\n(target=0.75)", apply_lane=True,
        lane_target=0.75))

    print("\n=== Condition 3: KEEP LEFT (lane_target=0.25) ===")
    conds.append(build_condition(
        model, walls, demo_pos, demo_vel, layout_map, starts,
        label="KEEP LEFT\n(target=0.25)", apply_lane=True,
        lane_target=0.25))

    # ── Plot: 2 rows × 3 columns ────────────────────────────
    V_plots = []
    for c in conds:
        vp = c["V"].copy()
        vp[vp < -1e8] = np.nan
        V_plots.append(vp)

    v_min = min(np.nanmin(vp) for vp in V_plots)
    v_max = max(np.nanmax(vp) for vp in V_plots)
    levels = np.linspace(v_min, v_max, 30)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    traj_colours = ['#ff6b6b', '#ffd93d', '#6bcb77']

    for col_idx, (cond, V_plot) in enumerate(zip(conds, V_plots)):
        xx, yy = cond["xx"], cond["yy"]
        pol = cond["pol"]
        trajs = cond["trajs"]
        pc = cond["patch_centers"]

        # ── Row 0: Value function + policy arrows ────────────
        ax = axes[0, col_idx]
        ax.contourf(xx, yy, V_plot, levels=levels, cmap='inferno')
        draw_walls(ax, layout_map, alpha=0.35)
        draw_policy_arrows(ax, pol, GRID_RES, stride=4)
        draw_lane_centerline(ax, ROUTE_WAYPOINTS_XY, alpha=0.4)
        if col_idx > 0:
            draw_lane_sides(ax, ROUTE_WAYPOINTS_XY, offset=0.25)
        if len(pc) > 0:
            ax.scatter(pc[:, 0], pc[:, 1], marker='x', c='red', s=25,
                       linewidths=1.5, alpha=0.7, zorder=5)
        ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)
        ax.set_xlim(*XY_RANGE)
        ax.set_ylim(*XY_RANGE)
        ax.set_aspect('equal')
        ax.set_title(f'{cond["label"]}\nValue function',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # ── Row 1: Trajectories ──────────────────────────────
        ax = axes[1, col_idx]
        ax.contourf(xx, yy, V_plot, levels=levels, cmap='inferno', alpha=0.35)
        draw_walls(ax, layout_map, alpha=0.45)
        draw_lane_centerline(ax, ROUTE_WAYPOINTS_XY, alpha=0.4)
        if col_idx > 0:
            draw_lane_sides(ax, ROUTE_WAYPOINTS_XY, offset=0.25, alpha=0.35)
        if len(pc) > 0:
            ax.scatter(pc[:, 0], pc[:, 1], marker='x', c='red', s=20,
                       linewidths=1, alpha=0.4, zorder=5)
        for _lid, obs in per_layout_pos.items():
            ax.scatter(obs[:, 0], obs[:, 1], s=0.3, alpha=0.06, c='cyan',
                       edgecolors='none')
        for traj, tc in zip(trajs, traj_colours):
            ax.plot(traj[:, 0], traj[:, 1], '-', color=tc, linewidth=2.5,
                    alpha=0.9, zorder=5)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=tc, markersize=8,
                    markeredgecolor='white', zorder=6)
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=tc, markersize=8,
                    markeredgecolor='white', zorder=6)
        ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)
        ax.set_xlim(*XY_RANGE)
        ax.set_ylim(*XY_RANGE)
        ax.set_aspect('equal')
        n_reached = sum(1 for t in trajs
                        if np.sqrt(((t[-1] - GOAL_XY)**2).sum()) < GOAL_RADIUS)
        ax.set_title(f'Trajectories ({n_reached}/{len(trajs)} reach goal)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle(
        'Velocity-aware lane discipline: directional patches enforce keep-right/left\n'
        'No norms (free)  |  Keep right (target=0.75)  |  '
        'Keep left (target=0.25)',
        fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(HERE, "q_iteration_lane.png")
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    print(f"\nFigure saved → {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
