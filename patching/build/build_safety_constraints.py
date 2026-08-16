"""
build_safety_constraints.py
============================
Build Ant kernels from explicit safety constraints (Safety Gymnasium style).

Constraint-violating observations are treated as flaws and patched
in bottleneck space.  Regression testing runs against constraint-
satisfying ("safe") observations to avoid over-suppression.

Constraints:
  velocity  — xy_speed <= 6.0
  tilt      — torso tilt <= 46 deg from upright
  hip_vel   — max |hip angular velocity| <= 8.0
  combined  — any of the above

Usage:
    python build_safety_constraints.py --constraint velocity
    python build_safety_constraints.py --constraint combined
    python build_safety_constraints.py                        # all four
"""

import argparse
import json
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
sys.path.insert(0, ROOT)

from IIRL.models import MixtureOfExperts

from flywheel.roles.oracle.moe_locomotion_oracle import MoELocomotionOracle
from flywheel.roles.refinement.deciders.bottleneck_regression_decider import (
    BottleneckRegressionDecider,
)
from flywheel.roles.refinement.act.locomotion_batch_deployer import (
    LocomotionBatchDeployer,
)

# ── Ant config ───────────────────────────────────────────────
ANT_CFG = {
    "obs_dim": 111, "bottleneck": 10, "num_experts": 3,
    "model_file": "models/Ant_imp_l10_e3_s5.pth",
    "data_glob": "data/Ant_data_*.json",
    "imp_glob":  "data/Ant_data_imp_*.json",
    "l_min": 0.003, "l_max": 0.10, "steepness": 100,
    "dim_groups": {
        "position": list(range(0, 13)),
        "velocity": list(range(13, 27)),
    },
}

# ── Constraint definitions ───────────────────────────────────
# Each returns (mask, oob_ratios, group_name) where:
#   mask[i]       = True if obs[i] violates the constraint
#   oob_ratios[i] = how far over the limit (0 = at boundary, >0 = worse)

def _velocity_constraint(obs, threshold=6.0):
    """XY speed must not exceed threshold."""
    xy_speed = np.sqrt(obs[:, 13]**2 + obs[:, 14]**2)
    mask = xy_speed > threshold
    oob = np.clip((xy_speed - threshold) / threshold, 0, None)
    return mask, oob, "velocity"


def _tilt_constraint(obs, threshold_deg=46.0):
    """Torso tilt from upright must not exceed threshold."""
    qw = obs[:, 1]
    tilt = 2 * np.arccos(np.clip(np.abs(qw), 0, 1)) * 180 / np.pi
    mask = tilt > threshold_deg
    oob = np.clip((tilt - threshold_deg) / threshold_deg, 0, None)
    return mask, oob, "tilt"


def _hip_vel_constraint(obs, threshold=8.0):
    """Max hip velocity across all 4 hips must not exceed threshold."""
    hip_vels = np.abs(obs[:, [19, 21, 23, 25]])
    max_hv = hip_vels.max(axis=1)
    mask = max_hv > threshold
    oob = np.clip((max_hv - threshold) / threshold, 0, None)
    return mask, oob, "hip_vel"


def _combined_constraint(obs):
    """Any of the three constraints violated."""
    m1, o1, _ = _velocity_constraint(obs)
    m2, o2, _ = _tilt_constraint(obs)
    m3, o3, _ = _hip_vel_constraint(obs)
    mask = m1 | m2 | m3
    oob = np.maximum(np.maximum(o1, o2), o3)
    return mask, oob, "combined"


CONSTRAINTS = {
    "velocity": _velocity_constraint,
    "tilt":     _tilt_constraint,
    "hip_vel":  _hip_vel_constraint,
    "combined": _combined_constraint,
}

# ── Strictness levels (same as flywheel builder) ─────────────
STRICTNESS_LEVELS = {
    "very_loose": {"safety_floor": 0.7, "max_basin_loss": 0.0,
                   "min_bw": 0.5, "max_bw": 4.0},
    "loose":      {"safety_floor": 0.5, "max_basin_loss": 0.0,
                   "min_bw": 0.4, "max_bw": 3.5},
    "medium":     {"safety_floor": 0.3, "max_basin_loss": 0.0,
                   "min_bw": 0.3, "max_bw": 3.0},
    "tight":      {"safety_floor": 0.15, "max_basin_loss": 0.0,
                   "min_bw": 0.2, "max_bw": 2.5},
    "very_tight": {"safety_floor": 0.05, "max_basin_loss": 0.0,
                   "min_bw": 0.15, "max_bw": 2.0},
    "adaptive":   {"safety_floor": 0.05, "max_basin_loss": 0.0,
                   "min_bw": 0.15, "max_bw": 4.0, "adaptive": True},
}


def load_ant_obs():
    """Load all Ant demo observations."""
    files = sorted(glob.glob(os.path.join(HERE, ANT_CFG["data_glob"])))
    if ANT_CFG["imp_glob"]:
        files += sorted(glob.glob(os.path.join(HERE, ANT_CFG["imp_glob"])))
    all_obs = []
    for f in files:
        d = json.load(open(f))
        all_obs.append(np.array(d["observations"], dtype=np.float32))
    return np.concatenate(all_obs, axis=0)


def build_constraint_kernel(
    obs_all, model, constraint_name, constraint_fn,
    strictness_label, strictness_cfg, patience=3, batch_size=200,
    max_patches=1000, n_ib_test=500, max_ib_supp=0.01,
):
    """Build a kernel that suppresses constraint-violating observations.

    Flaw observations (constraint violators) are patched in bottleneck
    space with regression testing against safe observations (constraint
    satisfiers) to prevent over-suppression.
    """
    min_bw = strictness_cfg["min_bw"]
    max_bw = strictness_cfg["max_bw"]
    max_basin_loss = strictness_cfg["max_basin_loss"]
    is_adaptive = strictness_cfg.get("adaptive", False)

    print(f"\n{'='*70}")
    print(f"  Ant — {strictness_label} — constraint: {constraint_name}")
    print(f"  basin_loss={max_basin_loss}  bw=[{min_bw}, {max_bw}]")
    if max_patches:
        print(f"  max_patches={max_patches}  ib_test={n_ib_test}")
    print(f"{'='*70}")

    # ── Split by constraint ──────────────────────────────────
    flaw_mask, oob_ratios, group_name = constraint_fn(obs_all)
    safe_obs = obs_all[~flaw_mask]
    flaw_obs = obs_all[flaw_mask]
    flaw_oob = oob_ratios[flaw_mask]

    n_safe = len(safe_obs)
    n_flaw = len(flaw_obs)
    pct = 100.0 * n_flaw / len(obs_all)
    print(f"  Observations: {len(obs_all)} total, {n_safe} safe, "
          f"{n_flaw} flaw ({pct:.1f}%)")

    # ── Build dim stats from SAFE observations ───────────────
    safe_mean = safe_obs.mean(axis=0)
    safe_min = safe_obs.min(axis=0)
    safe_max = safe_obs.max(axis=0)
    pos_std = np.zeros_like(safe_mean)
    neg_std = np.zeros_like(safe_mean)
    for d in range(safe_obs.shape[1]):
        above = safe_obs[safe_obs[:, d] > safe_mean[d], d]
        below = safe_obs[safe_obs[:, d] <= safe_mean[d], d]
        pos_std[d] = above.std() if len(above) > 1 else safe_obs[:, d].std()
        neg_std[d] = below.std() if len(below) > 1 else safe_obs[:, d].std()
    dim_stats = {
        "min": torch.from_numpy(safe_min),
        "max": torch.from_numpy(safe_max),
        "mean": torch.from_numpy(safe_mean),
        "std": torch.from_numpy(safe_obs.std(axis=0)),
        "pos_std": torch.from_numpy(pos_std.astype(np.float32)),
        "neg_std": torch.from_numpy(neg_std.astype(np.float32)),
    }
    dim_groups = {
        name: torch.tensor(dims, dtype=torch.long)
        for name, dims in ANT_CFG["dim_groups"].items()
    }
    estimator_config = {
        "l_min": ANT_CFG["l_min"],
        "l_max": ANT_CFG["l_max"],
        "steepness": ANT_CFG["steepness"],
    }

    # ── Build oracle with SAFE observations as basin ─────────
    oracle = MoELocomotionOracle(
        model=model,
        estimator_config=estimator_config,
        dim_stats=dim_stats,
        dim_groups=dim_groups,
        demo_obs=safe_obs,
        env_name="Ant",
        skip_dbscan=True,
    )
    print(f"  Oracle ready, bottleneck dim = {oracle.bottleneck_dim}, "
          f"{oracle.num_experts} experts")
    print(f"  Basin (safe) = {n_safe} points, Flaws = {n_flaw} points")

    # ── Instantiate refinement components ────────────────────
    decider = BottleneckRegressionDecider()
    deployer = LocomotionBatchDeployer()

    # ── Shuffle flaw observations for diverse coverage ───────
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_flaw)
    flaw_obs = flaw_obs[perm]
    flaw_oob = flaw_oob[perm]

    # ── Precompute IB regression helpers ──────────────────
    ib_lo = dim_stats["min"].numpy()
    ib_hi = dim_stats["max"].numpy()
    ib_noise_scale = dim_stats["std"].numpy() * 0.1

    # ── Run iterative patching with early stopping ───────────
    print(f"\n  Running until plateau (patience={patience})")
    if n_ib_test > 0:
        print(f"  IB regression: {n_ib_test} test points/iter, "
              f"max_ib_supp={max_ib_supp}")
    hdr = (f"  {'It':>3}  {'Batch':>6}  {'Patched':>7}  "
           f"{'Rejected':>8}  {'Shrinks':>7}  {'IB_Shr':>7}  "
           f"{'Total':>6}  {'Version':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    total_patched = 0
    total_rejected = 0
    total_shrinks = 0
    total_ib_shrinks = 0
    t0 = time.perf_counter()

    best_patched = 0
    stale_count = 0
    flaw_idx = 0
    it = 0

    while flaw_idx < n_flaw:
        it += 1

        # ── Take next batch of flaw observations ─────────────
        end_idx = min(flaw_idx + batch_size, n_flaw)
        batch_obs = flaw_obs[flaw_idx:end_idx]
        batch_oob = flaw_oob[flaw_idx:end_idx]
        flaw_idx = end_idx

        # ── Construct "planned" items for the decider ────────
        planned = []
        for i in range(len(batch_obs)):
            oob = float(batch_oob[i])
            t = min(oob, 1.0)  # normalised severity in [0, 1]
            if is_adaptive:
                # Adaptive: severity-proportional suppression
                #   barely over → strength 0.5, wide bw (catch boundary)
                #   way over    → strength 0.95, narrow bw (hard block)
                strength = float(np.clip(0.5 + 0.45 * t, 0.5, 0.95))
                proposed_bw = float(np.clip(
                    max_bw - (max_bw - min_bw) * t,
                    min_bw, max_bw,
                ))
            else:
                proposed_bw = float(np.clip(
                    min_bw + (max_bw - min_bw) * t,
                    min_bw, max_bw,
                ))
                strength = float(np.clip(0.5 + 0.5 * t, 0.5, 1.0))

            planned.append({
                "point": batch_obs[i].tolist(),
                "oob_ratio": oob,
                "violated_group": group_name,
                "safety_val": 1.0,
                "proposed_bw": proposed_bw,
                "strength": strength,
            })

        # ── Run decider (regression testing) ─────────────────
        oriented = {
            "planned": planned,
            "max_basin_loss": max_basin_loss,
            "min_bw": min_bw,
            "oracle_ref": oracle,
            "oracle_version": oracle.get_version(),
        }
        decision = decider.decide(oriented)

        # ── Deploy batch ─────────────────────────────────────
        if decision["accepted"]:
            result = deployer.act(decision)
            batch_result = result["batch"]
            patched = result["patched_points"]
            rejected = result["rejected"]
            shrinks = result["shrinks"]

            if batch_result and batch_result.local_corrections:
                # Enforce patch cap
                if max_patches:
                    budget = max_patches - oracle.patch_count
                    if budget <= 0:
                        batch_result.local_corrections = []
                    elif len(batch_result.local_corrections) > budget:
                        batch_result.local_corrections = batch_result.local_corrections[:budget]
                        patched = patched[:budget]
                oracle.send_patch(batch_result)
        else:
            patched = []
            rejected = decision["rejected"]
            shrinks = decision["shrinks"]

        total_patched += len(patched)
        total_rejected += rejected
        total_shrinks += shrinks

        # ── IB Regression Check ──────────────────────────────
        ib_shrinks = 0
        if n_ib_test > 0 and oracle.patch_count > 0:
            n_sample = min(n_ib_test, n_safe)
            ib_idx = np.random.choice(n_safe, n_sample, replace=False)
            ib_points = safe_obs[ib_idx].copy()
            noise = (np.random.randn(*ib_points.shape).astype(np.float32)
                     * ib_noise_scale)
            ib_points += noise
            np.clip(ib_points, ib_lo, ib_hi, out=ib_points)
            ib_result = oracle.ib_regression_check(
                ib_points, max_ib_supp=max_ib_supp, min_bw=min_bw,
            )
            ib_shrinks = ib_result["ib_shrinks"] + ib_result["ib_strength_cuts"]
            total_ib_shrinks += ib_shrinks

        print(f"  {it:>3}  {len(batch_obs):>6}  {len(patched):>7}  "
              f"{rejected:>8}  {shrinks:>7}  {ib_shrinks:>7}  "
              f"{oracle.patch_count:>6}  {oracle.get_version():>12}")

        # ── Patch cap check ──────────────────────────────────
        if max_patches and oracle.patch_count >= max_patches:
            print(f"\n  Patch cap reached: {oracle.patch_count} >= {max_patches}")
            break

        # ── Early stopping ───────────────────────────────────
        if len(patched) > best_patched:
            best_patched = len(patched)
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= patience:
            print(f"\n  Early stop: patched count plateaued for "
                  f"{patience} iterations")
            break

        if it >= 100:
            print(f"\n  Hard cap at {it} iterations")
            break

    elapsed = time.perf_counter() - t0
    per_expert = oracle.patch_count_per_expert
    expert_str = " ".join(f"E{i}={c}" for i, c in enumerate(per_expert))
    print(f"\n  Done in {elapsed:.1f}s — {oracle.patch_count} patches placed "
          f"[{expert_str}], {total_rejected} rejected, {total_shrinks} shrunk"
          f", {total_ib_shrinks} IB-shrunk")

    # ── Export kernel ────────────────────────────────────────
    kernel = {
        "moe_state_dict": model.state_dict(),
        "moe_config": {
            "input_dim": ANT_CFG["obs_dim"],
            "bottleneck_dim": ANT_CFG["bottleneck"],
            "num_experts": ANT_CFG["num_experts"],
        },
        "estimator_config": estimator_config,
        "dim_stats": {k: v.clone() for k, v in dim_stats.items()},
        "dim_groups": {k: v.clone() for k, v in dim_groups.items()},
        "flywheel_patches": oracle.export_patches(),
        "strictness": {"label": strictness_label, **strictness_cfg},
        "constraint": constraint_name,
        "decider": "regression",
        "n_demo_points": n_safe,
        "n_flaw_points": n_flaw,
        "max_patches": max_patches,
        "env_name": "Ant",
    }

    out_dir = os.path.join(HERE, "kernels")
    os.makedirs(out_dir, exist_ok=True)
    cap_tag = f"_cap{max_patches}" if max_patches else ""
    out_path = os.path.join(
        out_dir,
        f"Ant_constraint_{constraint_name}_{strictness_label}{cap_tag}.pt",
    )
    torch.save(kernel, out_path)
    print(f"  Saved: {out_path}")

    return kernel


def main():
    parser = argparse.ArgumentParser(
        description="Build Ant safety-constraint kernels")
    parser.add_argument(
        "--constraint", default=None,
        choices=list(CONSTRAINTS.keys()),
        help="Constraint type. Default: all four")
    parser.add_argument(
        "--strictness", default=None,
        choices=list(STRICTNESS_LEVELS.keys()),
        help="Strictness level. Default: all levels")
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience")
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Flaw observations per iteration")
    parser.add_argument(
        "--max-patches", type=int, default=1000,
        help="Hard cap on total patch count (default: 1000)")
    parser.add_argument(
        "--ib-test", type=int, default=500,
        help="Number of in-bounds test points per iteration (default: 500)")
    parser.add_argument(
        "--max-ib-supp", type=float, default=0.01,
        help="Max per-patch suppression on IB test points (default: 0.01)")
    args = parser.parse_args()

    # ── Load model + data once ───────────────────────────────
    print("Loading Ant model and observations...")
    obs_all = load_ant_obs()
    print(f"  Total observations: {obs_all.shape}")

    model = MixtureOfExperts(
        input_dim=ANT_CFG["obs_dim"],
        bottleneck_dim=ANT_CFG["bottleneck"],
        num_experts=ANT_CFG["num_experts"],
    )
    model_path = os.path.join(HERE, ANT_CFG["model_file"])
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"  MoE: {ANT_CFG['obs_dim']}D → {ANT_CFG['bottleneck']}B "
          f"× {ANT_CFG['num_experts']}E")

    # ── Select constraints ───────────────────────────────────
    if args.constraint:
        constraints = {args.constraint: CONSTRAINTS[args.constraint]}
    else:
        constraints = CONSTRAINTS

    # ── Select strictness levels ─────────────────────────────
    if args.strictness:
        levels = {args.strictness: STRICTNESS_LEVELS[args.strictness]}
    else:
        levels = STRICTNESS_LEVELS

    # ── Build all kernels ────────────────────────────────────
    for cname, cfn in constraints.items():
        for slabel, scfg in levels.items():
            build_constraint_kernel(
                obs_all, model, cname, cfn,
                slabel, scfg,
                patience=args.patience,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
                n_ib_test=args.ib_test,
                max_ib_supp=args.max_ib_supp,
            )


if __name__ == "__main__":
    main()
