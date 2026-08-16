"""Thorough evaluation of kernel constraints across all environments.

For each env, sample observations at various perturbation levels and check
whether the bound constraint + MoE reward correctly suppress OOB states.
"""
import os, sys, json, glob, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))

from patching.load.load_constraints import (
    load_kernel, evaluate, moe_reward, get_bounds, bound_factor,
)
from IIRL.models import MixtureOfExperts


def load_raw_obs(env_name):
    """Load all demo observations for an environment."""
    patterns = {
        "Ant":         ("data/Ant_data_*.json", "data/Ant_data_imp_*.json"),
        "HalfCheetah": ("data/HalfCheetah_data_*.json", "data/HalfCheetah_imp_data_*.json"),
        "Hopper":      ("data/Hopper_data_*.json", None),
        "Swimmer":     ("data/Swimmer_data_*.json", None),
        "Walker2d":    ("data/Walker2d_data_*.json", "data/Walker2d_data_imp_*.json"),
    }
    main_glob, imp_glob = patterns[env_name]
    files = sorted(glob.glob(os.path.join(HERE, main_glob)))
    if imp_glob:
        files += sorted(glob.glob(os.path.join(HERE, imp_glob)))
    all_obs = []
    for f in files:
        d = json.load(open(f))
        all_obs.append(np.array(d["observations"], dtype=np.float32))
    return np.concatenate(all_obs, axis=0)


def test_env(env_name):
    print(f"\n{'='*70}")
    print(f"  {env_name}")
    print(f"{'='*70}")

    kernel = load_kernel(env_name)
    obs = load_raw_obs(env_name)
    D = obs.shape[1]
    N = len(obs)
    print(f"  {N} demo points, {D} dims")

    # --- 1. MoE reward on demo data ---
    moe_r = moe_reward(kernel, obs)
    print(f"\n  MoE reward on DEMO data:")
    print(f"    mean={moe_r.mean():.4f}  min={moe_r.min():.4f}  "
          f"max={moe_r.max():.4f}  %>0.5: {(moe_r>0.5).mean()*100:.1f}%")

    # --- 2. Bound factor on demo data at each strictness ---
    print(f"\n  Bound factor on DEMO data:")
    for lv in kernel["strictness_levels"]:
        f = evaluate(kernel, obs, strictness=lv["label"])
        print(f"    {lv['label']:>12}: mean={f.mean():.4f}  "
              f"min={f.min():.4f}  %>0.5: {(f>0.5).mean()*100:.1f}%  "
              f"<0.01: {(f<0.01).mean()*100:.1f}%")

    # --- 3. Random observations (uniform in wider range) ---
    lo_all = obs.min(axis=0)
    hi_all = obs.max(axis=0)
    rng = hi_all - lo_all + 1e-6
    n_rand = 10000

    print(f"\n  Random observations ({n_rand} samples):")
    for scale_label, scale in [("1× range", 1.0), ("2× range", 2.0),
                                ("3× range", 3.0), ("5× range", 5.0)]:
        center = (lo_all + hi_all) / 2
        half = rng / 2 * scale
        rand_obs = np.random.uniform(center - half, center + half,
                                     size=(n_rand, D)).astype(np.float32)
        moe_r = moe_reward(kernel, rand_obs)
        print(f"    {scale_label}:")
        print(f"      MoE reward: mean={moe_r.mean():.4f}  "
              f"max={moe_r.max():.4f}  %>0.5: {(moe_r>0.5).mean()*100:.1f}%")
        for sl in ["very_loose", "medium", "very_tight"]:
            bf = evaluate(kernel, rand_obs, strictness=sl)
            combined = moe_r * bf
            print(f"      {sl:>12} bound: mean_f={bf.mean():.4f}  "
                  f"combined_max={combined.max():.4f}  "
                  f"combined>0.5: {(combined>0.5).mean()*100:.1f}%")

    # --- 4. Per-dimension perturbation: push each dim to 1.5×, 2×, 3× max ---
    # Only test active dims (from dim_groups) to avoid noise from zero cfrc_ext dims
    dim_groups = kernel.get("dim_groups", {})
    active_dims = []
    for dims in dim_groups.values():
        if isinstance(dims, torch.Tensor):
            dims = dims.tolist()
        active_dims.extend(dims)
    active_dims = sorted(set(active_dims)) if active_dims else list(range(D))
    n_active = len(active_dims)
    group_names = {d: g for g, dims in dim_groups.items()
                   for d in (dims.tolist() if isinstance(dims, torch.Tensor) else dims)}

    print(f"\n  Per-dim perturbation (from demo mean, one dim at a time, "
          f"{n_active} active dims):")
    mean_obs = obs.mean(axis=0)
    for mult_label, mult in [("1.5× max", 1.5), ("2× max", 2.0), ("3× max", 3.0)]:
        passing_dims_loose = 0
        passing_dims_tight = 0
        leaking_dims = []
        for d_idx in active_dims:
            test = mean_obs.copy()
            test[d_idx] = hi_all[d_idx] * mult
            bf_loose = evaluate(kernel, test, strictness="very_loose")
            bf_tight = evaluate(kernel, test, strictness="very_tight")
            mr = moe_reward(kernel, test)
            if bf_loose > 0.5:
                passing_dims_loose += 1
            if bf_tight > 0.5:
                passing_dims_tight += 1
            if mr > 0.5 and bf_tight > 0.5:
                leaking_dims.append((d_idx, float(mr), float(bf_tight)))

        print(f"    {mult_label}: dims passing loose={passing_dims_loose}/{n_active}  "
              f"tight={passing_dims_tight}/{n_active}")
        if leaking_dims:
            print(f"      LEAKING (MoE>0.5 AND tight>0.5): "
                  f"{len(leaking_dims)} dims")
            for d_idx, mr, bf in leaking_dims[:10]:
                gn = group_names.get(d_idx, "?")
                print(f"        dim {d_idx} ({gn}): moe_r={mr:.4f}  tight_f={bf:.4f}")

    # --- 5. Negative-side perturbation ---
    print(f"\n  Negative-side perturbation (from demo mean, one dim at a time, "
          f"{n_active} active dims):")
    for mult_label, mult in [("1.5× min", 1.5), ("2× min", 2.0), ("3× min", 3.0)]:
        leaking_neg = []
        for d_idx in active_dims:
            test = mean_obs.copy()
            test[d_idx] = lo_all[d_idx] * mult
            bf_tight = evaluate(kernel, test, strictness="very_tight")
            mr = moe_reward(kernel, test)
            if mr > 0.5 and bf_tight > 0.5:
                leaking_neg.append((d_idx, float(mr), float(bf_tight)))
        print(f"    {mult_label}: LEAKING (MoE>0.5 AND tight>0.5): {len(leaking_neg)} dims")
        for d_idx, mr, bf in leaking_neg[:10]:
            gn = group_names.get(d_idx, "?")
            print(f"        dim {d_idx} ({gn}): moe_r={mr:.4f}  tight_f={bf:.4f}")

    # --- 6. Gaussian noise around demo points ---
    print(f"\n  Gaussian perturbation around demo points:")
    obs_std = obs.std(axis=0)
    subset = obs[np.random.choice(N, min(2000, N), replace=False)]
    for sigma_mult in [0.5, 1.0, 2.0, 3.0, 5.0]:
        noisy = subset + np.random.randn(*subset.shape).astype(np.float32) * obs_std * sigma_mult
        mr = moe_reward(kernel, noisy)
        for sl in ["medium", "very_tight"]:
            bf = evaluate(kernel, noisy, strictness=sl)
            combined = mr * bf
            print(f"    noise={sigma_mult:.1f}σ  {sl:>12}: "
                  f"moe_mean={mr.mean():.4f}  bound_mean={bf.mean():.4f}  "
                  f"combined>0.5: {(combined>0.5).mean()*100:.1f}%  "
                  f"combined_max={combined.max():.4f}")


if __name__ == "__main__":
    np.random.seed(42)
    for env in ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]:
        test_env(env)
