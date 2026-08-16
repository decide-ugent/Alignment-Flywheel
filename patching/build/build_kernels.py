"""
build_kernels_flywheel.py
=========================
Build locomotion kernels through the Flywheel governance loop.

Instead of static per-dim bounds, the flywheel iteratively:
  Red Team → discovers OOB obs with high safety reward (flaws)
  Verifier → confirms the flaw violates dim-group bounds
  Refinement → places bottleneck-space Gaussian suppression patches
               with regression testing against the demo basin

The accumulated patches ARE the kernel — exported as a .pt file.

Usage:
    python build_kernels_flywheel.py [--env Ant] [--iterations 10]
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

from flywheel.factory.registry import FactoryRegistry
from flywheel.protocols.ooda.ooda_role import OODARole
from flywheel.protocols.artifacts.norm import Norm
from flywheel.protocols.enums import NormKind, VerificationOutcome
from flywheel.roles.oracle.moe_locomotion_oracle import MoELocomotionOracle
from flywheel.roles.flywheel_overlay.spatial_overlay import SpatialOverlay
from flywheel.roles.triage.fifo_triage import FIFOTriage
from flywheel.core.knowledge_base.in_memory_knowledge_base import InMemoryKnowledgeBase

# ── Environment registry (same as build_kernel_constraints.py) ──
ENVS = {
    "Ant": {
        "obs_dim": 111, "bottleneck": 10, "num_experts": 3,
        "model_file": "models/Ant_imp_l10_e3_s5.pth",
        "data_glob": "data/Ant_data_*.json",
        "imp_glob":  "data/Ant_data_imp_*.json",
        "l_min": 0.003, "l_max": 0.10, "steepness": 100,
        "dim_groups": {
            "position": list(range(0, 13)),
            "velocity": list(range(13, 27)),
        },
    },
    "HalfCheetah": {
        "obs_dim": 17, "bottleneck": 7, "num_experts": 4,
        "model_file": "models/HalfCheetah_imp_l10_e4_s5.pth",
        "data_glob": "data/HalfCheetah_data_*.json",
        "imp_glob":  "data/HalfCheetah_imp_data_*.json",
        "l_min": 0.50, "l_max": 1.00, "steepness": 100,
        "dim_groups": {
            "position": list(range(0, 8)),
            "velocity": list(range(8, 17)),
        },
    },
    "Hopper": {
        "obs_dim": 11, "bottleneck": 4, "num_experts": 2,
        "model_file": "models/Hopper_l4_e2_s5.pth",
        "data_glob": "data/Hopper_data_*.json",
        "imp_glob":  None,
        "l_min": 0.13, "l_max": 0.15, "steepness": 25,
        "dim_groups": {
            "position": list(range(0, 5)),
            "velocity": list(range(5, 11)),
        },
    },
    "Swimmer": {
        "obs_dim": 8, "bottleneck": 3, "num_experts": 1,
        "model_file": "models/Swimmer_l1_e1_s5.pth",
        "data_glob": "data/Swimmer_data_*.json",
        "imp_glob":  None,
        "l_min": 0.11, "l_max": 0.20, "steepness": 25,
        "dim_groups": {
            "position": list(range(0, 3)),
            "velocity": list(range(3, 8)),
        },
    },
    "Walker2d": {
        "obs_dim": 17, "bottleneck": 7, "num_experts": 4,
        "model_file": "models/Walker_imp_l10_e4_s5.pth",
        "data_glob": "data/Walker2d_data_*.json",
        "imp_glob":  "data/Walker2d_data_imp_*.json",
        "l_min": 0.07, "l_max": 0.60, "steepness": 100,
        "dim_groups": {
            "position": list(range(0, 8)),
            "velocity": list(range(8, 17)),
        },
    },
    "AntMaze": {
        "obs_dim": 29, "bottleneck": 8, "num_experts": 3,
        "model_file": "models/AntMaze_l8_e3_s5.pth",
        "data_glob": "data/AntMaze_data_*.json",
        "imp_glob":  None,
        "l_min": 0.01, "l_max": 0.10, "steepness": 50,
        "dim_groups": {
            "spatial":  list(range(0, 2)),    # x, y
            "position": list(range(2, 15)),   # z, quat(4), joints(8)
            "velocity": list(range(15, 29)),  # linear/angular vel
        },
    },
    "Suction": {
        "obs_dim": 36, "bottleneck": 16, "num_experts": 10,
        "model_file": "models/suction.pth",
        "data_file": "data/suction.npz",
        "data_array": "episode_policy_obs",
        "data_max_rows": 5000,
        "valid_steps": (0, 13),
        "data_glob": None,
        "imp_glob":  None,
        "l_min": 0.00001, "l_max": 0.00004, "steepness": 25,
        "dim_groups": {
            "features": list(range(0, 36)),
        },
    },
}


def load_all_obs(data_glob, imp_glob=None):
    """Load and concatenate observations from all matching data files."""
    files = sorted(glob.glob(os.path.join(HERE, data_glob)))
    if imp_glob:
        files += sorted(glob.glob(os.path.join(HERE, imp_glob)))
    all_obs = []
    for f in files:
        d = json.load(open(f))
        all_obs.append(np.array(d["observations"], dtype=np.float32))
    return np.concatenate(all_obs, axis=0)


def load_suction_obs(cfg):
    """Load observations from a suction npz replay-buffer file."""
    from suction_dataset import load_suction_array
    filepath = os.path.join(HERE, cfg["data_file"])
    array_name = cfg.get("data_array", "episode_policy_obs")
    max_rows = cfg.get("data_max_rows")
    load_rows = (max_rows + 2000) if max_rows else None
    arr = load_suction_array(filepath, array_name, max_rows=load_rows)
    step_start, step_stop = cfg.get("valid_steps", (0, 13))
    arr = arr[:, step_start:step_stop, :]
    N, T, D = arr.shape
    steps = arr.reshape(N * T, D)
    valid = ~np.isnan(steps).any(axis=1)
    obs = steps[valid].astype(np.float32)
    print(f"  Loaded {N} episodes, {obs.shape[0]}/{N*T} valid timesteps")
    return obs


# ── Strictness levels ───────────────────────────────────────────
# Each level tunes: safety_floor, max_patches, max_basin_loss, bw range.
#
# safety_floor  — min reward to flag as flaw (lower = catches weaker flaws)
# max_patches   — budget (more = denser suppression coverage)
# max_basin_loss — tolerable demo suppression (higher = more aggressive)
# min_bw/max_bw — bandwidth range in bottleneck space
STRICTNESS_LEVELS = {
    "very_loose": {
        "safety_floor": 0.7,
        "max_basin_loss": 0.0,
        "min_bw": 0.5,
        "max_bw": 4.0,
    },
    "loose": {
        "safety_floor": 0.5,
        "max_basin_loss": 0.0,
        "min_bw": 0.4,
        "max_bw": 3.5,
    },
    "medium": {
        "safety_floor": 0.3,
        "max_basin_loss": 0.0,
        "min_bw": 0.3,
        "max_bw": 3.0,
    },
    "tight": {
        "safety_floor": 0.15,
        "max_basin_loss": 0.0,
        "min_bw": 0.2,
        "max_bw": 2.5,
    },
    "very_tight": {
        "safety_floor": 0.05,
        "max_basin_loss": 0.0,
        "min_bw": 0.15,
        "max_bw": 2.0,
    },
    "adaptive": {
        "safety_floor": 0.05,
        "max_basin_loss": 0.0,
        "min_bw": 0.15,
        "max_bw": 4.0,
        "adaptive": True,
    },
}


def build_env(env_name, cfg, strictness_cfg, strictness_label="medium",
              decider_name="BottleneckRegressionDecider",
              patience=3, min_flaws_frac=0.02,
              max_patches=None, oob_margin=0.0,
              n_ib_test=500, max_ib_supp=0.01):
    """Run the flywheel governance loop for one environment.

    Args:
        patience: stop after this many consecutive iterations where
                  new-flaw count doesn't improve by at least
                  *min_flaws_frac* of the first iteration's count.
        decider_name: which decide-step to use
                      ("BottleneckRegressionDecider" or "BottleneckClusterDecider")
        max_patches: hard cap on total patch count (None = no cap).
        oob_margin: number of asymmetric stds beyond demo [min, max]
                    before a point is considered OOB. 0 = strict demo range.
        n_ib_test: number of in-bounds test points generated each iteration
                   for IB regression testing. 0 = disable.
        max_ib_supp: per-patch suppression threshold on IB test points.
                     Patches exceeding this are shrunk.
    """
    safety_floor = strictness_cfg["safety_floor"]
    max_basin_loss = strictness_cfg["max_basin_loss"]
    min_bw = strictness_cfg["min_bw"]
    max_bw = strictness_cfg["max_bw"]
    is_adaptive = strictness_cfg.get("adaptive", False)

    tag = "cluster" if "Cluster" in decider_name else "regression"
    print(f"\n{'='*70}")
    print(f"  {env_name} — {strictness_label} ({tag})")
    print(f"  floor={safety_floor}  basin_loss={max_basin_loss}  bw=[{min_bw}, {max_bw}]")
    if max_patches:
        print(f"  max_patches={max_patches}  oob_margin={oob_margin}")
    print(f"{'='*70}")

    # ── Load data + model ────────────────────────────────────
    if cfg.get("data_file"):
        obs = load_suction_obs(cfg)
    else:
        obs = load_all_obs(cfg["data_glob"], cfg.get("imp_glob"))
    print(f"  Demo observations: {obs.shape}")

    model = MixtureOfExperts(
        input_dim=cfg["obs_dim"],
        bottleneck_dim=cfg["bottleneck"],
        num_experts=cfg["num_experts"],
    )
    model_path = os.path.join(HERE, cfg["model_file"])
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"  MoE: {cfg['obs_dim']}D -> {cfg['bottleneck']}B x {cfg['num_experts']}E")

    # ── Compute dim stats ────────────────────────────────────
    obs_mean = obs.mean(axis=0)
    obs_min = obs.min(axis=0)
    obs_max = obs.max(axis=0)
    # Asymmetric std: positive side (values > mean) and negative side separately
    pos_mask = obs > obs_mean[None, :]
    neg_mask = obs <= obs_mean[None, :]
    # Per-dim std of values above/below the mean
    pos_std = np.zeros_like(obs_mean)
    neg_std = np.zeros_like(obs_mean)
    for d in range(obs.shape[1]):
        above = obs[pos_mask[:, d], d]
        below = obs[neg_mask[:, d], d]
        pos_std[d] = above.std() if len(above) > 1 else obs[:, d].std()
        neg_std[d] = below.std() if len(below) > 1 else obs[:, d].std()
    dim_stats = {
        "min": torch.from_numpy(obs_min),
        "max": torch.from_numpy(obs_max),
        "mean": torch.from_numpy(obs_mean),
        "std": torch.from_numpy(obs.std(axis=0)),
        "pos_std": torch.from_numpy(pos_std.astype(np.float32)),
        "neg_std": torch.from_numpy(neg_std.astype(np.float32)),
    }
    # Effective OOB bounds: shift min/max by oob_margin × asymmetric std
    if oob_margin > 0:
        effective_lo = obs_min - oob_margin * neg_std
        effective_hi = obs_max + oob_margin * pos_std
        dim_stats["effective_lo"] = torch.from_numpy(effective_lo.astype(np.float32))
        dim_stats["effective_hi"] = torch.from_numpy(effective_hi.astype(np.float32))
        print(f"  OOB margin: {oob_margin} asymmetric stds beyond demo range")
    dim_groups = {
        name: torch.tensor(dims, dtype=torch.long)
        for name, dims in cfg["dim_groups"].items()
    }
    estimator_config = {
        "l_min": cfg["l_min"],
        "l_max": cfg["l_max"],
        "steepness": cfg["steepness"],
    }

    # ── Build oracle adapter ─────────────────────────────────
    skip_dbscan = "Cluster" not in decider_name
    oracle = MoELocomotionOracle(
        model=model,
        estimator_config=estimator_config,
        dim_stats=dim_stats,
        dim_groups=dim_groups,
        demo_obs=obs,
        env_name=env_name,
        skip_dbscan=skip_dbscan,
    )
    print(f"  Oracle ready, bottleneck dim = {oracle.bottleneck_dim}, "
          f"{oracle.num_experts} experts")

    # ── Build OODA roles via factory ─────────────────────────
    factory = FactoryRegistry()
    factory.auto_register()

    redteam = OODARole(
        observe=factory.create("LocomotionObserver"),
        orient=factory.create("LocomotionOrienter"),
        decide=factory.create("HighestCombinedDecider"),
        act=factory.create("LocomotionCandidateSubmitter"),
        params={
            "demo_obs": obs,
            "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
            "dim_groups": dim_groups,
        },
    )

    # Norms for verification
    oob_norm = Norm(
        id="OBS_DIM_BOUNDS",
        kind=NormKind.SPATIAL_BOUNDARY,
        spec={"require_within_demo_range": True},
        severity=1.0,
        description="Observations must stay within demo dim ranges",
    )

    verifier = OODARole(
        observe=factory.create("LocomotionNormLoader"),
        orient=factory.create("LocomotionNormMatcher"),
        decide=factory.create("LocomotionViolationDecider"),
        act=factory.create("VerificationEmitter"),
        params={
            "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
            "dim_groups": dim_groups,
        },
    )

    refinement = OODARole(
        observe=factory.create("LocomotionQueueObserver"),
        orient=factory.create("BottleneckBandwidthOrienter"),
        decide=factory.create(decider_name),
        act=factory.create("LocomotionBatchDeployer"),
        params={
            "max_basin_loss": max_basin_loss,
            "min_bw": min_bw,
            "max_bw": max_bw,
            "oracle_ref": oracle,
        },
    )

    # Flywheel overlay (for norm tracking)
    overlay = SpatialOverlay(norms=[oob_norm])
    triage = FIFOTriage()
    kb = InMemoryKnowledgeBase()

    # ── Precompute IB regression helpers ──────────────────
    ib_lo = dim_stats["min"].numpy()
    ib_hi = dim_stats["max"].numpy()
    ib_noise_scale = dim_stats["std"].numpy() * 0.1

    # ── Run governance loop with early stopping ────────────
    print(f"\n  Running until plateau (patience={patience})")
    if n_ib_test > 0:
        print(f"  IB regression: {n_ib_test} test points/iter, "
              f"max_ib_supp={max_ib_supp}")
    hdr = (f"  {'It':>3}  {'Found':>6}  {'Verified':>8}  {'Patched':>7}  "
           f"{'Rejected':>8}  {'Shrinks':>7}  {'IB_Shr':>7}  "
           f"{'Total':>6}  {'Version':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    active_flaws = []
    total_patched = 0
    total_rejected = 0
    total_shrinks = 0
    total_ib_shrinks = 0
    t0 = time.perf_counter()

    # Early stopping state
    best_found = 0
    stale_count = 0
    it = 0

    while True:
        it += 1
        # ── RED TEAM ─────────────────────────────────────────
        rt_context = {
            "oracle": oracle,
            "demo_obs": obs,
            "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
            "dim_groups": dim_groups,
            "iteration": it,
            "prior_flaws": active_flaws,
            "safety_floor": safety_floor,
        }
        rt_result = redteam.execute(rt_context)
        candidates = rt_result.get("candidates", [])

        # ── VERIFY ───────────────────────────────────────────
        norms = overlay.get_norms()
        for cand in candidates:
            v_context = {
                "candidate": cand,
                "norms": norms,
                "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
                "dim_groups": dim_groups,
            }
            v_result = verifier.execute(v_context)
            result = v_result["result"]
            kb.append_candidate(cand)
            if result.outcome == VerificationOutcome.VIOLATION:
                kb.append_verification(result)
                triage.submit(result, cand)

        # ── REFINEMENT ───────────────────────────────────────
        verified_items = triage.pop_all()
        patched_points = []
        rejected = 0
        shrinks = 0

        if verified_items:
            ref_context = {
                "verified_items": verified_items,
                "oracle_version": oracle.get_version(),
                "oracle_ref": oracle,
                "max_basin_loss": max_basin_loss,
                "min_bw": min_bw,
                "max_bw": max_bw,
                "adaptive": is_adaptive,
            }
            ref_result = refinement.execute(ref_context)

            batch = ref_result.get("batch")
            patched_points = ref_result.get("patched_points", [])
            rejected = ref_result.get("rejected", 0)
            shrinks = ref_result.get("shrinks", 0)

            if batch and batch.local_corrections:
                # Enforce patch cap within a single iteration
                if max_patches:
                    budget = max_patches - oracle.patch_count
                    if budget <= 0:
                        batch.local_corrections = []
                    elif len(batch.local_corrections) > budget:
                        batch.local_corrections = batch.local_corrections[:budget]
                        patched_points = patched_points[:budget]
                oracle.send_patch(batch)
                overlay.apply_batch(batch)
                kb.append_batch(batch)

        total_patched += len(patched_points)
        total_rejected += rejected
        total_shrinks += shrinks

        # ── IB Regression Check ──────────────────────────────
        ib_shrinks = 0
        if n_ib_test > 0 and oracle.patch_count > 0:
            n_sample = min(n_ib_test, len(obs))
            ib_idx = np.random.choice(len(obs), n_sample, replace=False)
            ib_points = obs[ib_idx].copy()
            noise = (np.random.randn(*ib_points.shape).astype(np.float32)
                     * ib_noise_scale)
            ib_points += noise
            np.clip(ib_points, ib_lo, ib_hi, out=ib_points)
            ib_result = oracle.ib_regression_check(
                ib_points, max_ib_supp=max_ib_supp, min_bw=min_bw,
            )
            ib_shrinks = ib_result["ib_shrinks"] + ib_result["ib_strength_cuts"]
            total_ib_shrinks += ib_shrinks

        # Track active flaws for next iteration
        active_flaws = [
            c.context.get("point") for c in candidates
            if c.context.get("point")
        ]

        print(f"  {it:>3}  {len(candidates):>6}  {len(verified_items):>8}  "
              f"{len(patched_points):>7}  {rejected:>8}  {shrinks:>7}  "
              f"{ib_shrinks:>7}  "
              f"{oracle.patch_count:>6}  {oracle.get_version():>12}")

        # ── Early stopping: plateau detection ────────────
        found_this_it = len(candidates)
        # Minimum improvement threshold = min_flaws_frac of first iteration
        if it == 1:
            threshold = max(1, int(found_this_it * min_flaws_frac))
        if found_this_it > best_found + threshold:
            best_found = found_this_it
            stale_count = 0
        else:
            stale_count += 1

        if max_patches and oracle.patch_count >= max_patches:
            print(f"\n  Patch cap reached: {oracle.patch_count} >= {max_patches}")
            break

        if stale_count >= patience:
            print(f"\n  Early stop: flaws plateaued for {patience} iterations")
            break

        # Safety cap to avoid runaway loops
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
            "input_dim": cfg["obs_dim"],
            "bottleneck_dim": cfg["bottleneck"],
            "num_experts": cfg["num_experts"],
        },
        "estimator_config": estimator_config,
        "dim_stats": {k: v.clone() for k, v in dim_stats.items()},
        "dim_groups": {k: v.clone() for k, v in dim_groups.items()},
        "flywheel_patches": oracle.export_patches(),
        "strictness": {"label": strictness_label, **strictness_cfg},
        "decider": tag,
        "oob_margin": oob_margin,
        "max_patches": max_patches,
        "n_demo_points": len(obs),
        "env_name": env_name,
    }

    out_dir = os.path.join(HERE, "kernels")
    os.makedirs(out_dir, exist_ok=True)
    margin_tag = f"_margin{oob_margin}" if oob_margin > 0 else ""
    cap_tag = f"_cap{max_patches}" if max_patches else ""
    out_path = os.path.join(
        out_dir,
        f"{env_name}_flywheel_{strictness_label}_{tag}{margin_tag}{cap_tag}.pt",
    )
    torch.save(kernel, out_path)
    print(f"  Saved: {out_path}")

    return kernel


def main():
    parser = argparse.ArgumentParser(
        description="Build locomotion kernels via the Flywheel governance loop")
    parser.add_argument(
        "--env", default=None,
        help="Build for a specific env (Ant, HalfCheetah, etc). Default: all")
    parser.add_argument(
        "--strictness", default=None,
        help="Strictness level: very_loose, loose, medium, tight, very_tight, "
             "adaptive. Default: all levels")
    parser.add_argument(
        "--decider", default=None,
        choices=["regression", "cluster"],
        help="Decider mode: regression (per-point) or cluster (DBSCAN). "
             "Default: both")
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience (consecutive stale iterations)")
    parser.add_argument(
        "--max-patches", type=int, default=None,
        help="Hard cap on total patch count (default: no cap)")
    parser.add_argument(
        "--oob-margin", type=float, default=0.0,
        help="Number of asymmetric stds beyond demo [min,max] for OOB. "
             "0 = strict demo range (default)")
    parser.add_argument(
        '--ib-test', type=int, default=500,
        help="Number of in-bounds test points per iteration for IB "
             "regression. 0 = disable (default: 500)")
    parser.add_argument(
        '--max-ib-supp', type=float, default=0.01,
        help="Max per-patch suppression on IB test points before "
             "shrinking (default: 0.01)")
    args = parser.parse_args()

    envs = {args.env: ENVS[args.env]} if args.env else ENVS
    if args.strictness:
        levels = {args.strictness: STRICTNESS_LEVELS[args.strictness]}
    else:
        levels = STRICTNESS_LEVELS

    decider_map = {
        "regression": "BottleneckRegressionDecider",
        "cluster": "BottleneckClusterDecider",
    }
    if args.decider:
        deciders = {args.decider: decider_map[args.decider]}
    else:
        deciders = decider_map

    for env_name, cfg in envs.items():
        model_path = os.path.join(HERE, cfg["model_file"])
        if not os.path.exists(model_path):
            print(f"  Skipping {env_name}: model not found at {model_path}")
            continue
        for label, scfg in levels.items():
            for dtag, dname in deciders.items():
                build_env(env_name, cfg, scfg, label,
                          decider_name=dname,
                          patience=args.patience,
                          max_patches=args.max_patches,
                          oob_margin=args.oob_margin,
                          n_ib_test=args.ib_test,
                          max_ib_supp=args.max_ib_supp)


if __name__ == "__main__":
    main()
