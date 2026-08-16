"""
build_kernel_constraints.py
===========================
For each MuJoCo environment (Ant, HalfCheetah, Hopper, Swimmer, Walker2d):

1.  Load *all* demo data files and the pre-trained MoE model.
2.  Compute per-dimension statistics from the demos:
        min, max, mean, std  (separately for positive and negative ranges)
3.  Build a gradient of **strictness levels** (0 = loose → 1 = tight):
        At strictness `s` the allowed range on each dimension shrinks from
        [demo_min − margin, demo_max + margin]   (s=0, loose)
        to
        [demo_min, demo_max]                      (s=1, tight)
    with different margins for the positive and negative sides:
        pos_margin = pos_range * (1 − s)
        neg_margin = neg_range * (1 − s)
    where pos_range = max − mean, neg_range = mean − min.
4.  Bundle everything (MoE state_dict, dim bounds, strictness configs,
    obs metadata) into a single `.pt` file per environment:
        kernels/<Env>_kernel.pt
    Each file contains a dict with:
        "moe_state_dict"  — the MoE weights
        "moe_config"      — {"input_dim", "bottleneck_dim", "num_experts"}
        "dim_stats"       — {"min", "max", "mean", "std"}  (per-dim arrays)
        "strictness_levels" — list of dicts, each with:
              {"level": float,                   # 0..1
               "label": str,                     # e.g. "loose", "medium", …
               "bounds_lo": array (D,),          # lower allowed bound
               "bounds_hi": array (D,)}          # upper allowed bound
        "obs_dim"         — int
        "env_name"        — str

Usage (after running this script):

    from patching.load.load_constraints import load_kernel, evaluate
    kernel = load_kernel("Ant")
    factor = evaluate(kernel, obs_vector, strictness="medium")
"""

import os, sys, json, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
sys.path.insert(0, ROOT)

from IIRL.models import MixtureOfExperts

# ── Environment registry ────────────────────────────────────────
# dim_groups maps group names to dimension indices.
# For locomotion tasks the obs is [qpos, qvel] (and possibly cfrc_ext).
# We split into "position" (body positions + joint angles) and "velocity".
ENVS = {
    "Ant": {
        "obs_dim": 111,
        "bottleneck": 10,
        "num_experts": 3,
        "model_file": "models/Ant_imp_l10_e3_s5.pth",
        "data_glob": "data/Ant_data_*.json",
        "imp_glob":  "data/Ant_data_imp_*.json",
        "l_min": 0.003,
        "l_max": 0.10,
        "steepness": 100,
        "dim_groups": {
            "position": list(range(0, 13)),   # z, quat(4), joints(8)
            "velocity": list(range(13, 27)),  # linear/angular vel
            # dims 27-110 are cfrc_ext, all zero — excluded
        },
    },
    "HalfCheetah": {
        "obs_dim": 17,
        "bottleneck": 7,
        "num_experts": 4,
        "model_file": "models/HalfCheetah_imp_l10_e4_s5.pth",
        "data_glob": "data/HalfCheetah_data_*.json",
        "imp_glob":  "data/HalfCheetah_imp_data_*.json",
        "l_min": 0.50,
        "l_max": 1.00,
        "steepness": 100,
        "dim_groups": {
            "position": list(range(0, 8)),    # rootz, rooty, joints(6)
            "velocity": list(range(8, 17)),   # rootx_vel, rootz_vel, rooty_vel, joint_vel(6)
        },
    },
    "Hopper": {
        "obs_dim": 11,
        "bottleneck": 4,
        "num_experts": 2,
        "model_file": "models/Hopper_l4_e2_s5.pth",
        "data_glob": "data/Hopper_data_*.json",
        "imp_glob":  None,
        "l_min": 0.13,
        "l_max": 0.15,
        "steepness": 25,
        "dim_groups": {
            "position": list(range(0, 5)),    # z, angle, thigh_joint, leg_joint, foot_joint
            "velocity": list(range(5, 11)),   # vel(6)
        },
    },
    "Swimmer": {
        "obs_dim": 8,
        "bottleneck": 3,
        "num_experts": 1,
        "model_file": "models/Swimmer_l1_e1_s5.pth",
        "data_glob": "data/Swimmer_data_*.json",
        "imp_glob":  None,
        "l_min": 0.11,
        "l_max": 0.20,
        "steepness": 25,
        "dim_groups": {
            "position": list(range(0, 3)),    # angle, joint0, joint1
            "velocity": list(range(3, 8)),    # vel(5)
        },
    },
    "Walker2d": {
        "obs_dim": 17,
        "bottleneck": 7,
        "num_experts": 4,
        "model_file": "models/Walker_imp_l10_e4_s5.pth",
        "data_glob": "data/Walker2d_data_*.json",
        "imp_glob":  "data/Walker2d_data_imp_*.json",
        "l_min": 0.07,
        "l_max": 0.60,
        "steepness": 100,
        "dim_groups": {
            "position": list(range(0, 8)),    # z, angle, joints(6)
            "velocity": list(range(8, 17)),   # vel(9)
        },
    },
}

# ── Strictness gradient ─────────────────────────────────────────
# Each level is (label, strictness_factor)
# strictness_factor=0 → widest bounds, =1 → tightest (exact demo range)
STRICTNESS_LEVELS = [
    ("very_loose", 0.0),
    ("loose",      0.2),
    ("mild",       0.4),
    ("medium",     0.6),
    ("tight",      0.8),
    ("very_tight", 1.0),
]


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


def compute_dim_stats(obs):
    """Per-dimension statistics."""
    return {
        "min":  obs.min(axis=0),
        "max":  obs.max(axis=0),
        "mean": obs.mean(axis=0),
        "std":  obs.std(axis=0),
    }


def build_strictness_bounds(stats, levels):
    """Build per-dim [lo, hi] bounds for each strictness level.

    At strictness s=0 the bounds extend beyond [min, max] by the full
    positive/negative half-range as margin.
    At s=1 the bounds equal [min, max] exactly (tightest).

    The margin is ASYMMETRIC:
        negative margin = (mean − min) * (1 − s)
        positive margin = (max − mean) * (1 − s)
    so the positive and negative sides shrink independently.
    """
    lo = stats["min"]
    hi = stats["max"]
    mean = stats["mean"]
    neg_range = mean - lo   # always ≥ 0
    pos_range = hi - mean   # always ≥ 0

    result = []
    for label, s in levels:
        margin_neg = neg_range * (1.0 - s)
        margin_pos = pos_range * (1.0 - s)
        bounds_lo = lo - margin_neg
        bounds_hi = hi + margin_pos
        result.append({
            "level": float(s),
            "label": label,
            "bounds_lo": bounds_lo,
            "bounds_hi": bounds_hi,
        })
    return result


def build_kernel(env_name, cfg):
    """Build and save the kernel constraint file for one environment."""
    print(f"\n{'='*60}")
    print(f"  {env_name}")
    print(f"{'='*60}")

    # Load data
    obs = load_all_obs(cfg["data_glob"], cfg.get("imp_glob"))
    print(f"  Observations: {obs.shape}")
    assert obs.shape[1] == cfg["obs_dim"], (
        f"Expected obs_dim={cfg['obs_dim']}, got {obs.shape[1]}")

    # Compute stats
    stats = compute_dim_stats(obs)
    print(f"  Per-dim min range: [{stats['min'].min():.3f}, {stats['min'].max():.3f}]")
    print(f"  Per-dim max range: [{stats['max'].min():.3f}, {stats['max'].max():.3f}]")

    # Load MoE model
    model_path = os.path.join(HERE, cfg["model_file"])
    model = MixtureOfExperts(
        input_dim=cfg["obs_dim"],
        bottleneck_dim=cfg["bottleneck"],
        num_experts=cfg["num_experts"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  Loaded MoE: {cfg['obs_dim']}D → {cfg['bottleneck']}B × {cfg['num_experts']}E")

    # Compute reconstruction loss stats on training data
    with torch.no_grad():
        inp = torch.from_numpy(obs)
        out = model(inp)
        mse = ((out - inp) ** 2).mean(dim=1).numpy()
    loss_min = float(mse.min())
    loss_p50 = float(np.median(mse))
    loss_p98 = float(np.percentile(mse, 98))
    loss_max = float(mse.max())
    print(f"  Recon MSE: min={loss_min:.4f}  median={loss_p50:.4f}  "
          f"p98={loss_p98:.4f}  max={loss_max:.4f}")

    # Tuned reward-mapping hyperparameters (stored as estimator_config below)
    l_min = cfg["l_min"]
    l_max = cfg["l_max"]
    steepness = cfg["steepness"]
    print(f"  Estimator: l_min={l_min}, l_max={l_max}, steepness={steepness}")

    # Build strictness bounds
    levels = build_strictness_bounds(stats, STRICTNESS_LEVELS)
    for lv in levels:
        print(f"    {lv['label']:>12} (s={lv['level']:.1f}):  "
              f"lo=[{lv['bounds_lo'].min():.2f}..{lv['bounds_lo'].max():.2f}]  "
              f"hi=[{lv['bounds_hi'].min():.2f}..{lv['bounds_hi'].max():.2f}]")

    # Dim groups (position/angle vs velocity)
    dim_groups = cfg.get("dim_groups", {})
    dim_groups_tensor = {
        name: torch.tensor(dims, dtype=torch.long)
        for name, dims in dim_groups.items()
    }
    active_dims = []
    for dims in dim_groups.values():
        active_dims.extend(dims)
    active_dims = sorted(set(active_dims))
    print(f"  Dim groups: {', '.join(f'{k}={len(v)}' for k,v in dim_groups.items())}  "
          f"active={len(active_dims)}/{cfg['obs_dim']}")

    # Bundle
    kernel = {
        "env_name":          env_name,
        "obs_dim":           cfg["obs_dim"],
        "moe_config": {
            "input_dim":     cfg["obs_dim"],
            "bottleneck_dim": cfg["bottleneck"],
            "num_experts":   cfg["num_experts"],
        },
        "moe_state_dict":    model.state_dict(),
        "estimator_config": {
            "l_min":     cfg["l_min"],
            "l_max":     cfg["l_max"],
            "steepness": cfg["steepness"],
        },
        "loss_stats": {
            "min":   loss_min,
            "median": loss_p50,
            "p98":   loss_p98,
            "max":   loss_max,
        },
        "dim_stats": {
            "min":  torch.from_numpy(stats["min"]),
            "max":  torch.from_numpy(stats["max"]),
            "mean": torch.from_numpy(stats["mean"]),
            "std":  torch.from_numpy(stats["std"]),
        },
        "dim_groups":        dim_groups_tensor,
        "strictness_levels": [
            {
                "level":     lv["level"],
                "label":     lv["label"],
                "bounds_lo": torch.from_numpy(lv["bounds_lo"]),
                "bounds_hi": torch.from_numpy(lv["bounds_hi"]),
            }
            for lv in levels
        ],
        "n_demo_points":     len(obs),
    }

    out_dir = os.path.join(HERE, "kernels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env_name}_kernel.pt")
    torch.save(kernel, out_path)
    print(f"  Saved → {out_path}")


def main():
    for env_name, cfg in ENVS.items():
        build_kernel(env_name, cfg)
    print(f"\nDone. Kernel files are in {os.path.join(HERE, 'kernels')}/")


if __name__ == "__main__":
    main()
