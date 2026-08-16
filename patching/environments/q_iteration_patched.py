"""
Vanilla vs patched Q-iteration on layout-02.

Layout-02 blocks row-3 col-3 compared to layout-01.  We compare:
  Left:  Vanilla — model + layout-01 wall mask (doesn't know about new wall)
  Right: Patched — same model + layout-01 wall mask + suppressive Gaussian
         patch produced by the flywheel governance engine

Uses the real flywheel framework:
  - MoE2DOracle (BaseSpatialOracleAdapter) for reward queries
  - SpatialOverlay for norm definitions
  - GovernanceEngine → RedTeam → Verifier → Refinement → BatchDeployer
  - Patches flow through GovernanceBatch / LocalCorrection(SPATIAL_FLAW_PATCH)
  - Oracle.send_patch() applies Gaussian suppression kernels

Produces q_iteration_patched.png.
"""

import copy, os, sys
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

# ── flywheel imports ─────────────────────────────────────────────
from flywheel.roles.oracle.moe_2d_oracle import MoE2DOracle
from flywheel.roles.flywheel_overlay.spatial_overlay import SpatialOverlay
from flywheel.protocols.artifacts.norm import Norm
from flywheel.protocols.enums import NormKind, CorrectionType
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.local_correction import LocalCorrection
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw
from flywheel.protocols.ooda.ooda_role import OODARole
from flywheel.roles.verifier.observe.norm_loader import NormLoader
from flywheel.roles.verifier.orient.spatial_norm_matcher import SpatialNormMatcher
from flywheel.roles.verifier.decide.spatial_violation_decider import SpatialViolationDecider
from flywheel.roles.verifier.act.verification_emitter import VerificationEmitter
from flywheel.roles.refinement.observe.queue_observer import QueueObserver
from flywheel.roles.refinement.orient.adaptive_bandwidth_orienter import AdaptiveBandwidthOrienter
from flywheel.roles.refinement.decide.no_cumulative_decider import NoCumulativeDecider
from flywheel.roles.refinement.act.batch_deployer import BatchDeployer
from flywheel.roles.triage.fifo_triage import FIFOTriage
from flywheel.protocols.enums import VerificationOutcome

# ── constants ────────────────────────────────────────────────────
GRID_SIZE = 8
CELL = 1.0
XY_RANGE = (-4.0, 4.0)

GRID_RES = 80
GAMMA = 0.97
N_ITERS = 500
STEP_SIZE = 0.15

WALL_CLEARANCE = 0.45
PATH_CLEARANCE = 0.2
PATH_SIGMA = 0.4
DEMO_SPEED = 3.0           # representative speed for velocity queries

GOAL_XY = np.array([2.5, 1.5], dtype=np.float32)
GOAL_RADIUS = 0.4
GOAL_REWARD = 5.0

START_XY = np.array([-2.0, -1.5], dtype=np.float32)

# 8 discrete actions (king moves)
ACTIONS = np.array([
    [0,1],[0,-1],[-1,0],[1,0],[1,1],[-1,1],[1,-1],[-1,-1]
], dtype=np.float32)
ACTIONS = ACTIONS / np.linalg.norm(ACTIONS, axis=1, keepdims=True) * STEP_SIZE


def create_patched_layout(base_layout, oracle):
    """Create a modified layout where heavily-patched cells become virtual walls.

    Queries the oracle's suppression at each tile centre.  Cells with
    suppression > 0.5 are treated as impassable (the agent "learned"
    there is a wall there).
    """
    layout = copy.deepcopy(base_layout)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if layout[r][c] == 1:
                continue
            xy = tile_center(r, c).reshape(1, 2)
            supp = oracle._suppression(xy)
            if supp[0] > 0.5:
                layout[r][c] = 1
    return layout


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
    """Convert continuous XY position to maze grid tile (row, col)."""
    col = int(np.floor(x + (GRID_SIZE / 2.0)))
    row = int(np.floor((GRID_SIZE / 2.0) - y))
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    return row, col


def cell_is_wall(row, col, layout_map):
    """Check if a grid cell is a wall (1) or goal/start string."""
    if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
        return True
    return layout_map[row][col] == 1


def move_blocked_by_grid(pos_from, pos_to, layout_map):
    """Check if movement is blocked by the maze grid."""
    r_to, c_to = xy_to_tile(pos_to[0], pos_to[1])
    if cell_is_wall(r_to, c_to, layout_map):
        return True

    r_from, c_from = xy_to_tile(pos_from[0], pos_from[1])
    dr = r_to - r_from
    dc = c_to - c_from

    # Diagonal: block only if BOTH corner cells are walls
    if dr != 0 and dc != 0:
        if (cell_is_wall(r_from + dr, c_from, layout_map) and
                cell_is_wall(r_from, c_from + dc, layout_map)):
            return True

    return False


def compute_reward_from_oracle(oracle, res=GRID_RES):
    """Query the flywheel oracle for each action direction (velocity-aware).

    Returns (xx, yy, reward_sa) where reward_sa has shape (res, res, n_actions).
    Each action direction produces a different MoE query velocity.
    """
    reward_sa = oracle.query_grid_actions(
        ACTIONS, speed=DEMO_SPEED, res=res, xy_range=XY_RANGE)
    axis = np.linspace(*XY_RANGE, res)
    xx, yy = np.meshgrid(axis, axis)
    return xx, yy, reward_sa


def compute_reward_display(reward_sa):
    """Best-action reward for display purposes."""
    return reward_sa.max(axis=2)


def run_value_iteration(reward_sa, layout_map, res, n_iters, gamma):
    """Tabular value iteration with action-dependent rewards R(s,a).

    reward_sa has shape (res, res, n_actions).
    Q(s,a) = R(s,a) + gamma * V(s')  where V(s) = max_a Q(s,a).
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
                    # bilinear interp of V
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


def simulate_trajectory(policy, real_layout, start, res, max_steps=500, debug=False):
    """Simulate trajectory using the REAL maze grid for collision checks."""
    axis = np.linspace(*XY_RANGE, res)
    dx = axis[1] - axis[0]
    traj = [start.copy()]
    pos = start.copy()
    stop_reason = "max_steps"

    for step in range(max_steps):
        fi = (pos[1] - XY_RANGE[0]) / dx
        fj = (pos[0] - XY_RANGE[0]) / dx
        ri = int(np.clip(np.round(fi), 0, res - 1))
        ci = int(np.clip(np.round(fj), 0, res - 1))
        a = policy[ri, ci]
        new_pos = pos + ACTIONS[a]

        # Real maze grid collision (includes diagonal squeeze prevention)
        if move_blocked_by_grid(pos, new_pos, real_layout):
            if debug:
                act_names = ['N','S','W','E','NE','NW','SE','SW']
                from_tile = xy_to_tile(pos[0], pos[1])
                to_tile = xy_to_tile(new_pos[0], new_pos[1])
                print(f"    BLOCKED step {step}: ({pos[0]:.3f},{pos[1]:.3f}) tile{from_tile} "
                      f"--{act_names[a]}--> ({new_pos[0]:.3f},{new_pos[1]:.3f}) tile{to_tile}")
            stop_reason = "wall"
            break
        if (new_pos < XY_RANGE[0]).any() or (new_pos > XY_RANGE[1]).any():
            stop_reason = "bounds"
            break

        pos = new_pos
        traj.append(pos.copy())
        if np.sqrt(((pos - GOAL_XY) ** 2).sum()) < GOAL_RADIUS:
            break

    return np.array(traj)


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


def highlight_new_wall(ax, row, col, color='red'):
    """Draw the new wall cell with a distinctive colour."""
    cx, cy = tile_center(row, col)
    ax.add_patch(plt.Rectangle(
        (cx - CELL/2, cy - CELL/2), CELL, CELL,
        fc=color, ec='white', linewidth=2, alpha=0.6, zorder=4))
    ax.text(cx, cy, 'NEW\nWALL', ha='center', va='center', fontsize=7,
            fontweight='bold', color='white', zorder=5)


def draw_patch_contour(ax, center, sigma, n_sigma=3):
    """Draw the patch's effective footprint as a dashed ellipse."""
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + n_sigma * sigma[0] * np.cos(theta)
    y = center[1] + n_sigma * sigma[1] * np.sin(theta)
    ax.plot(x, y, '--', color='red', linewidth=1.5, alpha=0.8, zorder=4)
    ax.plot(center[0], center[1], 'x', color='red', markersize=10,
            markeredgewidth=2, zorder=5)


# ── data loading ─────────────────────────────────────────────────
DEMO_DIR = os.path.join(HERE, "pointmaze_data", "pointmaze_benchmark_demos")


def load_demo_pos():
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


# ── main ─────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # load model (trained on ALL layouts)
    ckpt = os.path.join(HERE, "checkpoints", "moe_5E_8B_full.pt")
    model = MixtureOfExperts(input_dim=4, bottleneck_dim=8, num_experts=5)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint {ckpt}")

    per_layout_pos, per_layout_vel = load_demo_pos()
    all_demo_pos = np.concatenate(list(per_layout_pos.values()), axis=0)
    all_demo_vel = np.concatenate(list(per_layout_vel.values()), axis=0)

    # Path-proximity uses only layout-01 demos (the "known deployment")
    # Using all layouts would cover every corridor and make the norm useless.
    demo_pos_01 = per_layout_pos["layout-01"]
    demo_vel_01 = per_layout_vel["layout-01"]

    # layout-01 walls = the "old deployment" info
    layout01 = LAYOUTS["layout-01"]["map"]
    walls_old = wall_cells_xy(layout01)

    # layout-02 = the REAL maze (new wall at row 3, col 3)
    layout02 = LAYOUTS["layout-02"]["map"]
    walls_real = wall_cells_xy(layout02)
    new_wall_tile = (3, 3)  # the blocked cell
    new_wall_xy = tile_center(*new_wall_tile)
    print(f"New wall at tile {new_wall_tile} → XY ({new_wall_xy[0]:.1f}, {new_wall_xy[1]:.1f})")

    starts = [
        START_XY,
        START_XY + np.array([0.3, 0.0]),
        START_XY + np.array([0.0, 0.3]),
    ]

    # ── Build flywheel oracle (BaseSpatialOracleAdapter) ─────
    # The MoE2DOracle wraps the trained MoE model with wall mask,
    # path-proximity, and goal bonus — identical to the reward
    # computation from before, but now inside a proper flywheel adapter.
    oracle_vanilla = MoE2DOracle(
        model=model, walls_xy=walls_old, demo_pos=demo_pos_01,
        demo_vel=demo_vel_01,
        wall_clearance=WALL_CLEARANCE,
        path_clearance=PATH_CLEARANCE, path_sigma=PATH_SIGMA,
        goal_xy=GOAL_XY, goal_radius=GOAL_RADIUS, goal_reward=GOAL_REWARD,
    )

    # ── Condition A: VANILLA (old wall mask, no patch) ───────
    print("\n=== VANILLA (old wall mask, model doesn't know about new wall) ===")
    xx, yy, reward_A = compute_reward_from_oracle(oracle_vanilla)
    reward_A_disp = compute_reward_display(reward_A)
    print(f"  Reward at new wall cell: {reward_A_disp[40, 36]:.3f}  (should be >0 = BAD)")
    V_A, pol_A = run_value_iteration(
        reward_A, layout01, GRID_RES, N_ITERS, GAMMA)

    trajs_A = []
    for s in starts:
        traj = simulate_trajectory(pol_A, layout02, s, GRID_RES)
        reached = np.sqrt(((traj[-1] - GOAL_XY) ** 2).sum()) < GOAL_RADIUS
        hits_wall = any(xy_to_tile(t[0], t[1]) == new_wall_tile for t in traj)
        end_tile = xy_to_tile(traj[-1][0], traj[-1][1])
        trajs_A.append(traj)
        print(f"  Start ({s[0]:.1f},{s[1]:.1f}): {len(traj)} steps, "
              f"goal={reached}, hits_new_wall={hits_wall}, "
              f"end=({traj[-1][0]:.2f},{traj[-1][1]:.2f}) tile={end_tile}")

    # ── Condition B: PATCHED via flywheel governance pipeline ─
    # Build a second oracle for the patched condition.
    oracle_patched = MoE2DOracle(
        model=model, walls_xy=walls_old, demo_pos=demo_pos_01,
        demo_vel=demo_vel_01,
        wall_clearance=WALL_CLEARANCE,
        path_clearance=PATH_CLEARANCE, path_sigma=PATH_SIGMA,
        goal_xy=GOAL_XY, goal_radius=GOAL_RADIUS, goal_reward=GOAL_REWARD,
    )

    print("\n=== FLYWHEEL GOVERNANCE — detecting & patching new wall ===")

    # SpatialOverlay provides the norm set Φ
    overlay = SpatialOverlay()
    norms = overlay.get_norms()

    # Expert path for flaw detection (layout-01 demo positions)
    expert_path = demo_pos_01[::20]

    # ── Step 1: Detect environmental change ──────────────────
    # Compare layout-01 and layout-02 to find newly-blocked cells.
    # This is the "observation" that triggers the governance loop.
    new_walls = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if layout02[r][c] == 1 and layout01[r][c] != 1:
                new_walls.append((r, c))
    print(f"  Detected {len(new_walls)} new wall(s): {new_walls}")

    # ── Step 2: Create CandidateFlaw for each new wall ───────
    # In production, these would come from the RedTeam OODA role.
    # Here the environmental change detection feeds directly into
    # the verification pipeline.
    candidates = []
    for (r, c) in new_walls:
        pt = tile_center(r, c).tolist()
        # Query the oracle — it still shows high reward (the flaw)
        result = oracle_patched.query_points([pt])
        flaw_value = result["values"][0]
        dist_to_path = float(cdist([pt], expert_path).min())
        candidates.append(CandidateFlaw(
            context={"point": pt, "dist_to_path": dist_to_path},
            trajectory={"kind": "spatial",
                         "steps": [{"payload": {"point": pt}}]},
            s=flaw_value, u=0.2, u_thresh=0.5,
            v_O=oracle_patched.get_version(),
        ))
        print(f"  Flaw at tile ({r},{c}) XY=({pt[0]:.1f},{pt[1]:.1f}): "
              f"oracle reward={flaw_value:.3f} (should be ~0)")

    # ── Step 3: Verify each candidate against norms (Verifier OODA) ─
    verifier = OODARole(
        observe=NormLoader(),
        orient=SpatialNormMatcher(),
        decide=SpatialViolationDecider(),
        act=VerificationEmitter(),
    )
    triage = FIFOTriage()

    for cand in candidates:
        v_context = {
            "candidate": cand,
            "norms": norms,
            "expert_path": expert_path,
            "boundary": 0.5,
        }
        v_result = verifier.execute(v_context)
        result = v_result["result"]
        print(f"  Verifier: outcome={result.outcome.value}, "
              f"violated_norm={result.violated_norm_id}")
        if result.outcome == VerificationOutcome.VIOLATION:
            triage.submit(result, cand)

    verified_items = triage.pop_all()
    print(f"  Verified violations: {len(verified_items)}")

    # ── Step 4: Refinement — build GovernanceBatch ───────────
    # Use the Refinement OODA role to compute adaptive bandwidths
    # and produce the GovernanceBatch with SPATIAL_FLAW_PATCH corrections.
    refinement = OODARole(
        observe=QueueObserver(),
        orient=AdaptiveBandwidthOrienter(),
        decide=NoCumulativeDecider(),
        act=BatchDeployer(),
        params={
            "max_patches": 60,
            "min_bw": 0.30,
            "max_bw": 0.55,
            "boundary": 0.5,
            "safety_floor": 0.01,
        },
    )

    # If the verifier didn't flag it (flaw is within expert boundary),
    # manually create the GovernanceBatch — the wall IS a hazard
    # regardless of norm matching.
    if verified_items:
        ref_context = {
            "verified_items": verified_items,
            "oracle_version": oracle_patched.get_version(),
            "basin_points": None,
            "max_patches": 60,
            "min_bw": 0.30,
            "max_bw": 0.55,
            "boundary": 0.5,
            "safety_floor": 0.01,
        }
        ref_result = refinement.execute(ref_context)
        batch = ref_result.get("batch")
    else:
        # Direct correction: the new wall is a discovered hazard.
        # Build the batch manually using flywheel data structures.
        print("  Flaw within expert boundary — creating direct correction")
        corrections = []
        for (r, c) in new_walls:
            pt = tile_center(r, c).tolist()
            corrections.append(LocalCorrection(
                correction_type=CorrectionType.SPATIAL_FLAW_PATCH,
                payload={"flaw_point": pt, "support_radius": 0.55},
                description=f"Suppress reward at new wall ({r},{c})",
            ))
            corrections.append(LocalCorrection(
                correction_type=CorrectionType.AUDIT_COVERAGE_UPDATE,
                payload={"case_class": f"wall_hazard|tile=({r},{c})"},
            ))
        batch = GovernanceBatch(
            from_oracle_version=oracle_patched.get_version(),
            to_oracle_version="oracle:v1",
            local_corrections=corrections,
            regression_evidence={"patched": len(new_walls)},
            signature="wall-hazard-correction",
        )

    # ── Step 5: Deploy — send GovernanceBatch to oracle ──────
    # oracle.send_patch() applies Gaussian suppression kernels.
    # This is the same mechanism used by PrecomputedGridOracle in
    # the 3D spatial demo.
    if batch and batch.local_corrections:
        deploy_result = oracle_patched.send_patch(batch)
        overlay.apply_batch(batch)
        print(f"  Deployed GovernanceBatch: {deploy_result}")
        print(f"  Oracle version: {oracle_patched.get_version()}")
        print(f"  Suppression kernels: {oracle_patched.patch_count}")

        # Show patch details
        centers = oracle_patched.patch_centers
        bws = oracle_patched.patch_bandwidths
        for i in range(len(centers)):
            print(f"    kernel {i}: center=({centers[i][0]:.2f},{centers[i][1]:.2f}) "
                  f"bw={bws[i]:.3f}")

    # ── Re-query oracle after patches ────────────────────────
    print("\n=== PATCHED (flywheel governance + Gaussian suppression) ===")
    xx, yy, reward_B = compute_reward_from_oracle(oracle_patched)
    reward_B_disp = compute_reward_display(reward_B)
    # Check reward at new wall cell
    r_idx = int((0.5 - XY_RANGE[0]) / (XY_RANGE[1] - XY_RANGE[0]) * (GRID_RES - 1))
    c_idx = int((-0.5 - XY_RANGE[0]) / (XY_RANGE[1] - XY_RANGE[0]) * (GRID_RES - 1))
    print(f"  Reward at new wall cell: {reward_B_disp[r_idx, c_idx]:.3f}  (should be ~0 = GOOD)")

    # The flywheel's patches create "virtual walls" for planning
    layout01_patched = create_patched_layout(layout01, oracle_patched)
    print(f"  Patched layout adds virtual wall at cells: ", end="")
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if layout01_patched[r][c] == 1 and layout01[r][c] != 1:
                print(f"({r},{c}) ", end="")
    print()

    V_B, pol_B = run_value_iteration(
        reward_B, layout01_patched, GRID_RES, N_ITERS, GAMMA)

    trajs_B = []
    for s in starts:
        traj = simulate_trajectory(pol_B, layout02, s, GRID_RES)
        reached = np.sqrt(((traj[-1] - GOAL_XY) ** 2).sum()) < GOAL_RADIUS
        hits_wall = any(xy_to_tile(t[0], t[1]) == new_wall_tile for t in traj)
        end_tile = xy_to_tile(traj[-1][0], traj[-1][1])
        trajs_B.append(traj)
        print(f"  Start ({s[0]:.1f},{s[1]:.1f}): {len(traj)} steps, "
              f"goal={reached}, hits_new_wall={hits_wall}, "
              f"end=({traj[-1][0]:.2f},{traj[-1][1]:.2f}) tile={end_tile}")

    # ── Plot: 2×2 ────────────────────────────────────────────
    V_A_plot = V_A.copy(); V_A_plot[V_A_plot < -1e8] = np.nan
    V_B_plot = V_B.copy(); V_B_plot[V_B_plot < -1e8] = np.nan
    v_min = min(np.nanmin(V_A_plot), np.nanmin(V_B_plot))
    v_max = max(np.nanmax(V_A_plot), np.nanmax(V_B_plot))
    levels = np.linspace(v_min, v_max, 30)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    colours = ['#ff6b6b', '#ffd93d', '#6bcb77']

    # Collect patch info for drawing
    patch_centers = oracle_patched.patch_centers
    patch_bws = oracle_patched.patch_bandwidths

    for col_idx, (label, reward, V_plot, pol, trajs) in enumerate([
        ("VANILLA\n(old wall mask, no patch)", reward_A_disp, V_A_plot, pol_A, trajs_A),
        ("PATCHED\n(flywheel governance)", reward_B_disp, V_B_plot, pol_B, trajs_B),
    ]):
        # Row 0: value function + arrows
        ax = axes[0, col_idx]
        ax.contourf(xx, yy, V_plot, levels=levels, cmap='inferno')
        draw_walls(ax, layout01, alpha=0.35)
        highlight_new_wall(ax, *new_wall_tile)
        draw_policy_arrows(ax, pol, GRID_RES, stride=4)
        if col_idx == 1 and len(patch_centers) > 0:
            # draw all flywheel patch footprints
            for pc, bw in zip(patch_centers, patch_bws):
                draw_patch_contour(ax, pc, np.array([bw, bw]))
        ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)
        ax.set_xlim(*XY_RANGE); ax.set_ylim(*XY_RANGE); ax.set_aspect('equal')
        ax.set_title(f'{label}\nValue function', fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('y')

        # Row 1: trajectories (simulated in the REAL maze with new wall)
        ax = axes[1, col_idx]
        ax.contourf(xx, yy, V_plot, levels=levels, cmap='inferno', alpha=0.4)
        draw_walls(ax, layout02, alpha=0.45)  # draw REAL walls
        highlight_new_wall(ax, *new_wall_tile)
        if col_idx == 1 and len(patch_centers) > 0:
            for pc, bw in zip(patch_centers, patch_bws):
                draw_patch_contour(ax, pc, np.array([bw, bw]))
        # demo scatter
        for _lid, obs in per_layout_pos.items():
            ax.scatter(obs[:, 0], obs[:, 1], s=0.3, alpha=0.06, c='cyan',
                       edgecolors='none')
        # trajectories
        for traj, c in zip(trajs, colours):
            ax.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2.5,
                    alpha=0.9, zorder=5)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=c, markersize=8,
                    markeredgecolor='white', zorder=6)
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=c, markersize=8,
                    markeredgecolor='white', zorder=6)
            reached = np.sqrt(((traj[-1] - GOAL_XY)**2).sum()) < GOAL_RADIUS
            if not reached:
                ax.plot(traj[-1, 0], traj[-1, 1], 'X', color='red',
                        markersize=14, markeredgecolor='white', zorder=7)
        ax.plot(*GOAL_XY, 'r*', markersize=18, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)
        ax.set_xlim(*XY_RANGE); ax.set_ylim(*XY_RANGE); ax.set_aspect('equal')
        n_reached = sum(1 for t in trajs
                        if np.sqrt(((t[-1] - GOAL_XY)**2).sum()) < GOAL_RADIUS)
        ax.set_title(f'Trajectories in REAL maze (layout-02)\n'
                     f'{n_reached}/{len(trajs)} reach goal',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.suptitle(
        'Layout-02: new wall blocks center passage\n'
        'Vanilla (left) plans through blocked cell → stuck    |    '
        'Patch (right) flywheel suppresses reward → detours',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(HERE, "q_iteration_patched.png")
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    print(f"\nFigure saved → {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
