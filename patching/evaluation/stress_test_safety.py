"""
stress_test_safety.py
=====================
Safety-critical stress test for kernel constraint models.

Treats the MoE + bound system as a deployed safety filter on a real robot.
A "fault" = an observation that is *clearly out-of-distribution* yet receives
a combined score (moe_reward × bound_factor) above a danger threshold.

Sweeps millions of points across many attack surfaces and records every
fault with full provenance.  Results are saved to a JSON file for audit.

Run:
    python stress_test_safety.py            # all envs
    python stress_test_safety.py Ant        # single env
"""

import os, sys, json, glob, time, datetime
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from patching.load.load_constraints import (
    load_kernel, evaluate, moe_reward, get_bounds, bound_factor, get_model,
)

# ── Config ───────────────────────────────────────────────────────

DANGER_THRESHOLDS = [0.3, 0.5, 0.7]       # combined score thresholds
STRICTNESS_LEVELS = ["very_loose", "loose", "mild", "medium", "tight", "very_tight"]

# How many points per test type (high for safety)
N_RANDOM      = 100_000    # uniform random in scaled ranges
N_GAUSSIAN    = 50_000     # Gaussian around demo points
N_MULTI_DIM   = 50_000     # multi-dim simultaneous perturbation
N_ADVERSARIAL = 200_000    # adversarial random search
N_BOUNDARY    = 50_000     # points near the decision boundary
N_CORRELATED  = 50_000     # correlated group perturbations


def load_raw_obs(env_name):
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


def get_active_dims(kernel):
    dim_groups = kernel.get("dim_groups", {})
    active = []
    group_map = {}
    for g, dims in dim_groups.items():
        if isinstance(dims, torch.Tensor):
            dims = dims.tolist()
        active.extend(dims)
        for d in dims:
            group_map[d] = g
    active = sorted(set(active)) if active else list(range(kernel["obs_dim"]))
    return active, group_map, dim_groups


def combined_score(kernel, obs, strictness):
    """MoE reward × bound factor.  This is what gates unsafe behavior."""
    mr = moe_reward(kernel, obs)
    bf = evaluate(kernel, obs, strictness=strictness)
    mr = np.atleast_1d(mr)
    bf = np.atleast_1d(bf)
    return mr * bf, mr, bf


def check_oob(obs, lo, hi, active_dims=None):
    """Per-sample boolean: is at least one active dim outside [lo, hi]?"""
    obs = np.atleast_2d(obs)
    if active_dims is not None:
        obs = obs[:, active_dims]
        lo = lo[active_dims]
        hi = hi[active_dims]
    below = obs < lo
    above = obs > hi
    return (below | above).any(axis=1)


def max_oob_ratio(obs, lo, hi, active_dims=None):
    """Per-sample: max ratio of how far any dim is outside bounds.
    0 = in-bounds, >0 = fraction of range beyond boundary."""
    obs = np.atleast_2d(obs)
    if active_dims is not None:
        obs = obs[:, active_dims]
        lo = lo[active_dims]
        hi = hi[active_dims]
    rng = hi - lo + 1e-9
    excess_lo = np.maximum(lo - obs, 0) / rng
    excess_hi = np.maximum(obs - hi, 0) / rng
    return np.maximum(excess_lo, excess_hi).max(axis=1)


class FaultTracker:
    """Accumulates fault statistics across test types."""

    def __init__(self, env_name):
        self.env_name = env_name
        self.tests = []  # list of test result dicts

    def record(self, test_name, n_sampled, obs_batch, combined, mr, bf,
               strictness, oob_mask, oob_ratio, extra=None):
        """Record results for one (test_type, strictness) combination."""
        entry = {
            "test": test_name,
            "strictness": strictness,
            "n_sampled": int(n_sampled),
        }
        for thresh in DANGER_THRESHOLDS:
            faults = (combined >= thresh) & oob_mask
            n_faults = int(faults.sum())
            rate = n_faults / max(n_sampled, 1)
            entry[f"faults_{thresh}"] = n_faults
            entry[f"rate_{thresh}"] = rate
            if n_faults > 0:
                # Record worst offenders
                fault_idx = np.where(faults)[0]
                worst_idx = fault_idx[np.argsort(-combined[fault_idx])[:5]]
                worst = []
                for wi in worst_idx:
                    worst.append({
                        "combined": float(combined[wi]),
                        "moe_r": float(mr[wi]),
                        "bound_f": float(bf[wi]),
                        "oob_ratio": float(oob_ratio[wi]),
                        "obs_snippet": obs_batch[wi, :8].tolist(),
                    })
                entry[f"worst_{thresh}"] = worst

        entry["combined_max"] = float(combined.max()) if len(combined) > 0 else 0.0
        entry["combined_mean"] = float(combined.mean()) if len(combined) > 0 else 0.0
        entry["moe_max"] = float(mr.max()) if len(mr) > 0 else 0.0
        entry["bound_max"] = float(bf.max()) if len(bf) > 0 else 0.0
        if extra:
            entry.update(extra)
        self.tests.append(entry)

    def summary_table(self):
        """Print a compact summary table."""
        print(f"\n{'-'*100}")
        print(f"  FAULT SUMMARY: {self.env_name}")
        print(f"{'-'*100}")
        hdr = f"  {'Test':<35} {'Strict':<12} {'Sampled':>8}"
        for t in DANGER_THRESHOLDS:
            hdr += f"  {'F@'+str(t):>7} {'Rate':>10}"
        hdr += f"  {'MaxComb':>8}"
        print(hdr)
        print(f"  {'-'*97}")
        for e in self.tests:
            row = f"  {e['test']:<35} {e['strictness']:<12} {e['n_sampled']:>8}"
            for t in DANGER_THRESHOLDS:
                f = e[f"faults_{t}"]
                r = e[f"rate_{t}"]
                row += f"  {f:>7} {r:>10.6f}"
            row += f"  {e['combined_max']:>8.4f}"
            print(row)

    def to_dict(self):
        return {"env": self.env_name, "tests": self.tests}


# ── Test batteries ───────────────────────────────────────────────

def test_uniform_random(kernel, obs_demo, tracker):
    """Uniform random in scaled observation ranges."""
    active_dims, _, _ = get_active_dims(kernel)
    lo_demo = obs_demo.min(axis=0)
    hi_demo = obs_demo.max(axis=0)
    rng = hi_demo - lo_demo + 1e-6
    center = (lo_demo + hi_demo) / 2.0
    D = obs_demo.shape[1]

    # Get the exact demo bounds for OOB check (only on active dims)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")

    scales = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    for scale in scales:
        half = rng / 2 * scale
        samples = np.random.uniform(
            center - half, center + half,
            size=(N_RANDOM, D)).astype(np.float32)

        oob = check_oob(samples, lo_vt, hi_vt, active_dims)
        oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

        for sl in STRICTNESS_LEVELS:
            comb, mr, bf = combined_score(kernel, samples, sl)
            tracker.record(
                f"uniform_{scale:.1f}x",
                N_RANDOM, samples, comb, mr, bf, sl, oob, oob_r,
                extra={"scale": scale, "pct_oob": float(oob.mean() * 100)})


def test_gaussian_perturbation(kernel, obs_demo, tracker):
    """Gaussian noise around demo points at many noise levels."""
    active_dims, _, _ = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    obs_std = obs_demo.std(axis=0)
    N = len(obs_demo)

    noise_levels = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    for sigma_mult in noise_levels:
        # Sample base points from demo
        idx = np.random.choice(N, N_GAUSSIAN, replace=True)
        base = obs_demo[idx]
        noise = np.random.randn(*base.shape).astype(np.float32) * obs_std * sigma_mult
        samples = base + noise

        oob = check_oob(samples, lo_vt, hi_vt, active_dims)
        oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

        for sl in STRICTNESS_LEVELS:
            comb, mr, bf = combined_score(kernel, samples, sl)
            tracker.record(
                f"gaussian_{sigma_mult:.2f}s",
                N_GAUSSIAN, samples, comb, mr, bf, sl, oob, oob_r,
                extra={"sigma_mult": sigma_mult, "pct_oob": float(oob.mean() * 100)})


def test_single_dim_sweep(kernel, obs_demo, tracker):
    """For each active dim, sweep from -5× to +5× the demo range."""
    active_dims, group_map, _ = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    mean_obs = obs_demo.mean(axis=0)
    lo_demo = obs_demo.min(axis=0)
    hi_demo = obs_demo.max(axis=0)

    # 50 sweep points per dim, from -5× min to +5× max
    n_sweep = 100
    all_samples = []
    all_dim_ids = []

    for d_idx in active_dims:
        lo_val = lo_demo[d_idx]
        hi_val = hi_demo[d_idx]
        rng = hi_val - lo_val + 1e-9
        sweep_lo = lo_val - 5.0 * rng
        sweep_hi = hi_val + 5.0 * rng
        sweep_vals = np.linspace(sweep_lo, sweep_hi, n_sweep)

        for val in sweep_vals:
            obs = mean_obs.copy()
            obs[d_idx] = val
            all_samples.append(obs)
            all_dim_ids.append(d_idx)

    samples = np.array(all_samples, dtype=np.float32)
    oob = check_oob(samples, lo_vt, hi_vt, active_dims)
    oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

    for sl in STRICTNESS_LEVELS:
        comb, mr, bf = combined_score(kernel, samples, sl)
        tracker.record(
            "single_dim_sweep",
            len(samples), samples, comb, mr, bf, sl, oob, oob_r,
            extra={"n_dims": len(active_dims), "n_sweep": n_sweep})


def test_multi_dim_perturbation(kernel, obs_demo, tracker):
    """Simultaneously perturb k random active dims OOB."""
    active_dims, group_map, _ = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    mean_obs = obs_demo.mean(axis=0)
    lo_demo = obs_demo.min(axis=0)
    hi_demo = obs_demo.max(axis=0)
    rng = hi_demo - lo_demo + 1e-6
    n_active = len(active_dims)

    k_values = [1, 2, 3, max(n_active // 4, 1), max(n_active // 2, 1), n_active]
    k_values = sorted(set(k_values))

    for k in k_values:
        n_per_k = N_MULTI_DIM // len(k_values)
        samples = np.tile(mean_obs, (n_per_k, 1))

        for i in range(n_per_k):
            dims_to_perturb = np.random.choice(active_dims, size=min(k, n_active), replace=False)
            for d in dims_to_perturb:
                # Push to 1.5×–5× outside the range, randomly positive or negative
                mult = np.random.uniform(1.5, 5.0)
                if np.random.rand() > 0.5:
                    samples[i, d] = hi_demo[d] + rng[d] * (mult - 1)
                else:
                    samples[i, d] = lo_demo[d] - rng[d] * (mult - 1)

        oob = check_oob(samples, lo_vt, hi_vt, active_dims)
        oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

        for sl in STRICTNESS_LEVELS:
            comb, mr, bf = combined_score(kernel, samples, sl)
            tracker.record(
                f"multi_dim_k={k}",
                n_per_k, samples, comb, mr, bf, sl, oob, oob_r,
                extra={"k": k, "pct_oob": float(oob.mean() * 100)})


def test_adversarial_search(kernel, obs_demo, tracker):
    """Random search for OOB observations that maximize combined score.

    Strategy: start from demo points, add progressively larger random
    perturbations, keep the ones that score highest while being OOB.
    This simulates an adversary trying to fool the safety filter.
    """
    active_dims, _, dim_groups = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    obs_std = obs_demo.std(axis=0)
    N_demo = len(obs_demo)
    D = obs_demo.shape[1]

    for sl in STRICTNESS_LEVELS:
        best_combined = 0.0
        best_obs = None
        best_mr = 0.0
        best_bf = 0.0
        n_oob_faults = {t: 0 for t in DANGER_THRESHOLDS}
        all_combined = []

        # Multiple random restart rounds
        batch_size = 5000
        n_batches = N_ADVERSARIAL // batch_size

        for _ in range(n_batches):
            # Start from random demo points
            idx = np.random.choice(N_demo, batch_size, replace=True)
            candidates = obs_demo[idx].copy()

            # Add large perturbation to guarantee OOB
            noise_scale = np.random.uniform(1.0, 5.0, size=(batch_size, 1))
            noise = np.random.randn(batch_size, D).astype(np.float32) * obs_std * noise_scale
            candidates += noise

            # Only keep the ones that are actually OOB on active dims
            oob_mask = check_oob(candidates, lo_vt, hi_vt, active_dims)
            if oob_mask.sum() == 0:
                all_combined.extend([0.0] * batch_size)
                continue

            oob_cands = candidates[oob_mask]
            comb, mr, bf = combined_score(kernel, oob_cands, sl)
            all_combined.extend(comb.tolist())

            for t in DANGER_THRESHOLDS:
                n_oob_faults[t] += int((comb >= t).sum())

            if comb.max() > best_combined:
                wi = np.argmax(comb)
                best_combined = float(comb[wi])
                best_obs = oob_cands[wi]
                best_mr = float(mr[wi])
                best_bf = float(bf[wi])

        all_combined = np.array(all_combined, dtype=np.float32)
        entry = {
            "test": f"adversarial_search",
            "strictness": sl,
            "n_sampled": N_ADVERSARIAL,
            "combined_max": float(best_combined),
            "combined_mean": float(all_combined.mean()) if len(all_combined) > 0 else 0.0,
            "moe_max": float(best_mr),
            "bound_max": float(best_bf),
        }
        for t in DANGER_THRESHOLDS:
            entry[f"faults_{t}"] = n_oob_faults[t]
            entry[f"rate_{t}"] = n_oob_faults[t] / N_ADVERSARIAL
            if best_obs is not None and best_combined >= t:
                oob_r = max_oob_ratio(best_obs.reshape(1, -1), lo_vt, hi_vt, active_dims)
                entry[f"worst_{t}"] = [{
                    "combined": best_combined,
                    "moe_r": best_mr,
                    "bound_f": best_bf,
                    "oob_ratio": float(oob_r[0]),
                    "obs_snippet": best_obs[:8].tolist(),
                }]
        tracker.tests.append(entry)


def test_boundary_probing(kernel, obs_demo, tracker):
    """Sample points right at and slightly beyond the decision boundary.

    These are the most dangerous: close enough to demo region that the
    MoE might still give high reward, but technically OOB.
    """
    active_dims, _, _ = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    mean_obs = obs_demo.mean(axis=0)
    lo_demo = obs_demo.min(axis=0)
    hi_demo = obs_demo.max(axis=0)
    rng = hi_demo - lo_demo + 1e-6
    D = obs_demo.shape[1]
    N_demo = len(obs_demo)

    # Start from demo points, push individual dims just barely OOB
    # epsilon multiples: 0.01× to 0.5× beyond the boundary
    epsilons = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    for eps in epsilons:
        n_per_eps = N_BOUNDARY // len(epsilons)
        idx = np.random.choice(N_demo, n_per_eps, replace=True)
        samples = obs_demo[idx].copy()

        for i in range(n_per_eps):
            # Pick 1-3 random active dims to push OOB
            n_dims_push = np.random.randint(1, min(4, len(active_dims) + 1))
            dims_push = np.random.choice(active_dims, n_dims_push, replace=False)
            for d in dims_push:
                if np.random.rand() > 0.5:
                    # Push just above max
                    samples[i, d] = hi_demo[d] + rng[d] * eps
                else:
                    # Push just below min
                    samples[i, d] = lo_demo[d] - rng[d] * eps

        oob = check_oob(samples, lo_vt, hi_vt, active_dims)
        oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

        for sl in STRICTNESS_LEVELS:
            comb, mr, bf = combined_score(kernel, samples, sl)
            tracker.record(
                f"boundary_eps={eps:.2f}",
                n_per_eps, samples, comb, mr, bf, sl, oob, oob_r,
                extra={"epsilon": eps, "pct_oob": float(oob.mean() * 100)})


def test_correlated_group_perturbation(kernel, obs_demo, tracker):
    """Perturb all dims in one group simultaneously (e.g. all positions OOB
    while velocities stay in-range, or vice versa).

    On a real robot, correlated failures are more likely than single-dim
    failures (e.g. the robot falls → all joint angles shift together).
    """
    active_dims, _, dim_groups = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    mean_obs = obs_demo.mean(axis=0)
    lo_demo = obs_demo.min(axis=0)
    hi_demo = obs_demo.max(axis=0)
    rng = hi_demo - lo_demo + 1e-6
    N_demo = len(obs_demo)
    D = obs_demo.shape[1]

    for group_name, group_dims in dim_groups.items():
        if isinstance(group_dims, torch.Tensor):
            group_dims = group_dims.tolist()

        n_per_group = N_CORRELATED // max(len(dim_groups), 1)

        # Scale multipliers for the group
        scales = [1.2, 1.5, 2.0, 3.0, 5.0]
        for scale in scales:
            n_per = n_per_group // len(scales)
            idx = np.random.choice(N_demo, n_per, replace=True)
            samples = obs_demo[idx].copy()

            for i in range(n_per):
                for d in group_dims:
                    # Shift by a fraction of the range in a random direction
                    direction = np.random.choice([-1, 1])
                    samples[i, d] += direction * rng[d] * (scale - 1.0) * np.random.uniform(0.5, 1.0)

            oob = check_oob(samples, lo_vt, hi_vt, active_dims)
            oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

            for sl in STRICTNESS_LEVELS:
                comb, mr, bf = combined_score(kernel, samples, sl)
                tracker.record(
                    f"corr_{group_name}_{scale:.1f}x",
                    n_per, samples, comb, mr, bf, sl, oob, oob_r,
                    extra={"group": group_name, "scale": scale,
                           "pct_oob": float(oob.mean() * 100)})


def test_physical_failure_modes(kernel, obs_demo, tracker):
    """Simulate physically plausible failure modes for locomotion:
    - Robot falls over (z-height drops to 0 or goes very high)
    - All joints lock (all joint angles go to extreme)
    - Runaway velocity (velocities spike)
    - Frozen (obs = constant across all dims)
    """
    active_dims, group_map, dim_groups = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    lo_demo = obs_demo.min(axis=0)
    hi_demo = obs_demo.max(axis=0)
    mean_obs = obs_demo.mean(axis=0)
    D = obs_demo.shape[1]
    N_demo = len(obs_demo)

    failure_samples = []
    failure_labels = []

    pos_dims = dim_groups.get("position", [])
    vel_dims = dim_groups.get("velocity", [])
    if isinstance(pos_dims, torch.Tensor): pos_dims = pos_dims.tolist()
    if isinstance(vel_dims, torch.Tensor): vel_dims = vel_dims.tolist()

    # 1. Robot falls: z-height (dim 0 for all envs) goes to 0 or very low
    for _ in range(2000):
        idx = np.random.randint(N_demo)
        obs = obs_demo[idx].copy()
        obs[0] = np.random.uniform(-1.0, lo_demo[0] * 0.5)  # below normal z
        failure_samples.append(obs)
        failure_labels.append("fall_low_z")

    # 2. Robot launched upward: z-height spikes
    for _ in range(2000):
        idx = np.random.randint(N_demo)
        obs = obs_demo[idx].copy()
        obs[0] = hi_demo[0] * np.random.uniform(2.0, 10.0)
        failure_samples.append(obs)
        failure_labels.append("launch_high_z")

    # 3. All joints at extreme: push all position dims to their extremes
    for _ in range(2000):
        idx = np.random.randint(N_demo)
        obs = obs_demo[idx].copy()
        for d in pos_dims:
            if np.random.rand() > 0.5:
                obs[d] = hi_demo[d] * np.random.uniform(1.5, 3.0)
            else:
                obs[d] = lo_demo[d] * np.random.uniform(1.5, 3.0)
        failure_samples.append(obs)
        failure_labels.append("extreme_joints")

    # 4. Velocity spike: push all velocity dims to extremes
    for _ in range(2000):
        idx = np.random.randint(N_demo)
        obs = obs_demo[idx].copy()
        for d in vel_dims:
            obs[d] = (hi_demo[d] if np.random.rand() > 0.5 else lo_demo[d]) * np.random.uniform(2.0, 5.0)
        failure_samples.append(obs)
        failure_labels.append("velocity_spike")

    # 5. Frozen robot: constant obs
    for _ in range(2000):
        obs = np.full(D, np.random.uniform(-1, 1), dtype=np.float32)
        failure_samples.append(obs)
        failure_labels.append("frozen_constant")

    # 6. Zero obs: all zeros
    for _ in range(1000):
        failure_samples.append(np.zeros(D, dtype=np.float32))
        failure_labels.append("all_zeros")

    # 7. NaN-adjacent: very large values
    for _ in range(1000):
        obs = np.random.uniform(-1000, 1000, D).astype(np.float32)
        failure_samples.append(obs)
        failure_labels.append("extreme_random")

    samples = np.array(failure_samples, dtype=np.float32)
    active_dims_list = pos_dims + vel_dims
    oob = check_oob(samples, lo_vt, hi_vt, active_dims_list)
    oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims_list)

    for sl in STRICTNESS_LEVELS:
        comb, mr, bf = combined_score(kernel, samples, sl)
        tracker.record(
            "physical_failures",
            len(samples), samples, comb, mr, bf, sl, oob, oob_r,
            extra={"n_modes": 7})

    # Also break down by failure type
    labels_arr = np.array(failure_labels)
    for mode in sorted(set(failure_labels)):
        mask = labels_arr == mode
        mode_samples = samples[mask]
        mode_oob = oob[mask]
        mode_oob_r = oob_r[mask]
        for sl in ["medium", "very_tight"]:
            comb, mr, bf = combined_score(kernel, mode_samples, sl)
            tracker.record(
                f"phys_{mode}",
                int(mask.sum()), mode_samples, comb, mr, bf, sl,
                mode_oob, mode_oob_r)


def test_interpolation_extrapolation(kernel, obs_demo, tracker):
    """Test interpolation between demo points (should be safe) and
    extrapolation beyond them (should be caught)."""
    active_dims, _, _ = get_active_dims(kernel)
    lo_vt, hi_vt = get_bounds(kernel, "very_tight")
    N_demo = len(obs_demo)

    # Extrapolation: take two demo points, go beyond the line
    n_test = 20000
    i1 = np.random.choice(N_demo, n_test)
    i2 = np.random.choice(N_demo, n_test)
    p1 = obs_demo[i1]
    p2 = obs_demo[i2]

    # Extrapolation factors: 1.5 to 5.0 (beyond p2)
    factors = np.random.uniform(1.5, 5.0, size=(n_test, 1)).astype(np.float32)
    samples = p1 + factors * (p2 - p1)

    oob = check_oob(samples, lo_vt, hi_vt, active_dims)
    oob_r = max_oob_ratio(samples, lo_vt, hi_vt, active_dims)

    for sl in STRICTNESS_LEVELS:
        comb, mr, bf = combined_score(kernel, samples, sl)
        tracker.record(
            "extrapolation",
            n_test, samples, comb, mr, bf, sl, oob, oob_r,
            extra={"pct_oob": float(oob.mean() * 100)})


# ── Main ─────────────────────────────────────────────────────────

def stress_test_env(env_name):
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  SAFETY STRESS TEST: {env_name}")
    print(f"{'='*80}")

    kernel = load_kernel(env_name)
    obs_demo = load_raw_obs(env_name)
    active_dims, group_map, dim_groups = get_active_dims(kernel)
    D = kernel["obs_dim"]
    print(f"  obs_dim={D}  active_dims={len(active_dims)}  "
          f"demo_points={len(obs_demo)}")
    print(f"  dim_groups: {', '.join(f'{k}={len(v) if isinstance(v, list) else len(v.tolist())}' for k,v in dim_groups.items())}")

    tracker = FaultTracker(env_name)

    print(f"\n  [1/8] Uniform random sampling ({N_RANDOM} x 7 scales x 6 strictness)...")
    test_uniform_random(kernel, obs_demo, tracker)

    print(f"  [2/8] Gaussian perturbation ({N_GAUSSIAN} x 8 noise levels x 6 strictness)...")
    test_gaussian_perturbation(kernel, obs_demo, tracker)

    print(f"  [3/8] Single-dim sweep ({len(active_dims)} dims x 100 points x 6 strictness)...")
    test_single_dim_sweep(kernel, obs_demo, tracker)

    print(f"  [4/8] Multi-dim perturbation ({N_MULTI_DIM} per k x 6 strictness)...")
    test_multi_dim_perturbation(kernel, obs_demo, tracker)

    print(f"  [5/8] Adversarial search ({N_ADVERSARIAL} x 6 strictness)...")
    test_adversarial_search(kernel, obs_demo, tracker)

    print(f"  [6/8] Boundary probing ({N_BOUNDARY} x 8 eps x 6 strictness)...")
    test_boundary_probing(kernel, obs_demo, tracker)

    print(f"  [7/8] Correlated group perturbation ({N_CORRELATED} x 6 strictness)...")
    test_correlated_group_perturbation(kernel, obs_demo, tracker)

    print(f"  [8/8] Physical failure modes + extrapolation...")
    test_physical_failure_modes(kernel, obs_demo, tracker)
    test_interpolation_extrapolation(kernel, obs_demo, tracker)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    tracker.summary_table()

    # Aggregate: total faults across ALL tests
    total_sampled = sum(e["n_sampled"] for e in tracker.tests)
    print(f"\n  TOTAL POINTS SAMPLED: {total_sampled:,}")
    for t in DANGER_THRESHOLDS:
        total_faults = sum(e[f"faults_{t}"] for e in tracker.tests)
        overall_rate = total_faults / max(total_sampled, 1)
        print(f"  TOTAL FAULTS @ {t}: {total_faults:,}  "
              f"OVERALL RATE: {overall_rate:.8f}  "
              f"({overall_rate*100:.6f}%)")

    # Per-strictness aggregate
    print(f"\n  PER-STRICTNESS AGGREGATE (faults @ 0.5 threshold):")
    for sl in STRICTNESS_LEVELS:
        sl_tests = [e for e in tracker.tests if e["strictness"] == sl]
        sl_sampled = sum(e["n_sampled"] for e in sl_tests)
        sl_faults = sum(e["faults_0.5"] for e in sl_tests)
        sl_rate = sl_faults / max(sl_sampled, 1)
        max_comb = max((e["combined_max"] for e in sl_tests), default=0)
        print(f"    {sl:>12}: {sl_faults:>7} / {sl_sampled:>10,}  "
              f"rate={sl_rate:.8f}  max_combined={max_comb:.4f}")

    return tracker, elapsed


def main():
    envs = sys.argv[1:] if len(sys.argv) > 1 else [
        "Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]

    np.random.seed(42)
    all_results = {}
    total_t = 0

    for env in envs:
        tracker, elapsed = stress_test_env(env)
        all_results[env] = tracker.to_dict()
        all_results[env]["elapsed_s"] = elapsed
        total_t += elapsed

    # Save full results to JSON
    out_path = os.path.join(HERE, "stress_test_results.json")
    # Convert numpy/torch types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    class SafeEncoder(json.JSONEncoder):
        def default(self, o):
            r = convert(o)
            if r is not o:
                return r
            return super().default(o)

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": 42,
            "config": {
                "N_RANDOM": N_RANDOM,
                "N_GAUSSIAN": N_GAUSSIAN,
                "N_MULTI_DIM": N_MULTI_DIM,
                "N_ADVERSARIAL": N_ADVERSARIAL,
                "N_BOUNDARY": N_BOUNDARY,
                "N_CORRELATED": N_CORRELATED,
                "DANGER_THRESHOLDS": DANGER_THRESHOLDS,
            },
            "results": all_results,
            "total_elapsed_s": total_t,
        }, f, indent=2, cls=SafeEncoder)

    print(f"\n{'='*80}")
    print(f"  ALL DONE -- {total_t:.1f}s total")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*80}")

    # Final cross-env summary
    print(f"\n  CROSS-ENVIRONMENT SUMMARY (faults @ 0.5, medium strictness):")
    print(f"  {'Env':<15} {'Sampled':>12} {'Faults':>8} {'Rate':>12} {'MaxComb':>10}")
    for env in envs:
        tests = all_results[env]["tests"]
        med_tests = [t for t in tests if t["strictness"] == "medium"]
        sampled = sum(t["n_sampled"] for t in med_tests)
        faults = sum(t["faults_0.5"] for t in med_tests)
        rate = faults / max(sampled, 1)
        max_c = max((t["combined_max"] for t in med_tests), default=0)
        print(f"  {env:<15} {sampled:>12,} {faults:>8,} {rate:>12.8f} {max_c:>10.4f}")

    print(f"\n  CROSS-ENVIRONMENT SUMMARY (faults @ 0.5, very_tight strictness):")
    print(f"  {'Env':<15} {'Sampled':>12} {'Faults':>8} {'Rate':>12} {'MaxComb':>10}")
    for env in envs:
        tests = all_results[env]["tests"]
        vt_tests = [t for t in tests if t["strictness"] == "very_tight"]
        sampled = sum(t["n_sampled"] for t in vt_tests)
        faults = sum(t["faults_0.5"] for t in vt_tests)
        rate = faults / max(sampled, 1)
        max_c = max((t["combined_max"] for t in vt_tests), default=0)
        print(f"  {env:<15} {sampled:>12,} {faults:>8,} {rate:>12.8f} {max_c:>10.4f}")


if __name__ == "__main__":
    main()
