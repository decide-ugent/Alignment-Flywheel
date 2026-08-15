"""
load_kernel_constraints.py
==========================
Load a kernel constraint file produced by ``build_kernel_constraints.py``
and evaluate observations against the bounds at any strictness level.

Quick start
-----------
    from load_kernel_constraints import load_kernel, evaluate

    kernel = load_kernel("Ant")          # loads  kernels/Ant_kernel.pt
    obs    = np.random.randn(111)        # some 111-D observation

    # Scalar factor in [0, 1]: 1 = within bounds, 0 = far outside
    factor = evaluate(kernel, obs, strictness="medium")

    # Or pass a batch  (N, D)
    factors = evaluate(kernel, obs_batch, strictness="tight")

    # Or evaluate the MoE reconstruction reward alongside bounds
    reward, factor = evaluate_full(kernel, obs, strictness="medium")

Available strictness labels
---------------------------
    very_loose  (s=0.0) — bounds extend ±1× the half-range beyond demo [min,max]
    loose       (s=0.2)
    mild        (s=0.4)
    medium      (s=0.6)
    tight       (s=0.8)
    very_tight  (s=1.0) — bounds equal exact demo [min,max]

The positive and negative margins shrink independently (asymmetric),
so dimensions with a wider positive range get more positive slack than
negative, and vice versa.

File contents
-------------
Each ``kernels/<Env>_kernel.pt`` contains:

    env_name            str
    obs_dim             int
    moe_config          dict  {"input_dim", "bottleneck_dim", "num_experts"}
    moe_state_dict      OrderedDict  (MixtureOfExperts weights)
    estimator_config    dict  {"l_min", "l_max", "steepness"}
                        Tuned hyperparameters for the exponential reward mapping:
                        reward = exp(-(normalized_loss) * steepness)
    loss_stats          dict  {"min", "median", "p98", "max"}
    dim_stats           dict  {"min", "max", "mean", "std"}  (tensors, shape D)
    dim_groups          dict  {"position": tensor, "velocity": tensor}
                        Maps group name → dimension indices for per-group
                        bound evaluation (position/angle norms vs velocity norms).
    strictness_levels   list of dict:
                            level      float  (0..1)
                            label      str
                            bounds_lo  tensor (D,)
                            bounds_hi  tensor (D,)
    n_demo_points       int
"""

import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


# ── Loading ──────────────────────────────────────────────────────

def load_kernel(env_name, kernels_dir=None):
    """Load a kernel constraint file.

    Parameters
    ----------
    env_name : str
        One of "Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d".
    kernels_dir : str, optional
        Path to the directory containing *_kernel.pt files.
        Defaults to ``<this_script_dir>/kernels/``.

    Returns
    -------
    dict  — the full kernel bundle (see module docstring for keys).
    """
    if kernels_dir is None:
        kernels_dir = os.path.join(HERE, "kernels")
    path = os.path.join(kernels_dir, f"{env_name}_kernel.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def get_model(kernel):
    """Reconstruct the MoE model from a loaded kernel dict.

    Returns a ``MixtureOfExperts`` in eval mode.
    """
    # Import here so this file stays self-contained for basic bound checks.
    from gated_autoencoder import MixtureOfExperts

    cfg = kernel["moe_config"]
    model = MixtureOfExperts(
        input_dim=cfg["input_dim"],
        bottleneck_dim=cfg["bottleneck_dim"],
        num_experts=cfg["num_experts"],
    )
    model.load_state_dict(kernel["moe_state_dict"])
    model.eval()
    return model


def get_bounds(kernel, strictness="medium"):
    """Return (bounds_lo, bounds_hi) arrays for a given strictness.

    Parameters
    ----------
    strictness : str or float
        A label like ``"medium"`` or a float in [0, 1].
        If a float is given, the closest level is selected.

    Returns
    -------
    bounds_lo : np.ndarray (D,)
    bounds_hi : np.ndarray (D,)
    """
    levels = kernel["strictness_levels"]

    if isinstance(strictness, str):
        for lv in levels:
            if lv["label"] == strictness:
                return lv["bounds_lo"].numpy(), lv["bounds_hi"].numpy()
        raise ValueError(
            f"Unknown strictness label '{strictness}'. "
            f"Available: {[l['label'] for l in levels]}")

    # Numeric: pick closest level
    target = float(strictness)
    best = min(levels, key=lambda lv: abs(lv["level"] - target))
    return best["bounds_lo"].numpy(), best["bounds_hi"].numpy()


# ── Evaluation ───────────────────────────────────────────────────

def bound_factor(obs, bounds_lo, bounds_hi, sigma_frac=0.3, dim_groups=None):
    """Smooth constraint factor in [0, 1] for each observation.

    For each dimension, if the observation is within [lo, hi] the
    contribution is 1.0.  Outside, it decays as a Gaussian with
    sigma = sigma_frac × half_range.

    When ``dim_groups`` is provided (dict mapping group name to dim
    indices), the factor is computed **per group** and the result is the
    **minimum** across groups.  This prevents a single OOB dimension from
    being masked by the product of many in-range dimensions (the main
    failure mode with high-dimensional obs like Ant 111D).

    When ``dim_groups`` is ``None``, falls back to the product over ALL
    dimensions.

    Parameters
    ----------
    obs : array (D,) or (N, D)
    bounds_lo, bounds_hi : array (D,)
    sigma_frac : float
        Controls how fast the factor drops outside the bounds.
    dim_groups : dict[str, tensor/list] or None
        Mapping from group name to dim indices.

    Returns
    -------
    float or np.ndarray (N,)  — values in [0, 1]
    """
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
    lo = np.asarray(bounds_lo, dtype=np.float32)
    hi = np.asarray(bounds_hi, dtype=np.float32)

    half_range = (hi - lo) / 2.0
    sigma = half_range * sigma_frac + 1e-9
    center = (lo + hi) / 2.0

    excess = np.maximum(np.abs(obs - center) - half_range, 0.0)
    per_dim = np.exp(-(excess ** 2) / (2.0 * sigma ** 2))     # (N, D)

    if dim_groups is not None and len(dim_groups) > 0:
        # Per-group product, then min across groups
        group_factors = []
        for name, dims in dim_groups.items():
            if isinstance(dims, torch.Tensor):
                dims = dims.numpy()
            dims = np.asarray(dims)
            group_factors.append(per_dim[:, dims].prod(axis=-1))
        # Stack to (N, G) and take min across groups
        stacked = np.stack(group_factors, axis=-1)   # (N, G)
        result = stacked.min(axis=-1)                # (N,)
    else:
        result = per_dim.prod(axis=-1)

    return float(result[0]) if result.shape[0] == 1 else result


def evaluate(kernel, obs, strictness="medium", sigma_frac=0.3):
    """Evaluate an observation against the bound constraint.

    Uses per-group evaluation when the kernel contains ``dim_groups``.

    Parameters
    ----------
    kernel : dict
        Loaded kernel (from ``load_kernel``).
    obs : array (D,) or (N, D)
        Observation(s) to evaluate.
    strictness : str or float
        Strictness level label or numeric value.
    sigma_frac : float
        Gaussian decay rate outside the bounds.

    Returns
    -------
    float or np.ndarray (N,)
        Factor in [0, 1].  1 = within bounds, 0 = far outside.
    """
    lo, hi = get_bounds(kernel, strictness)
    dim_groups = kernel.get("dim_groups", None)
    return bound_factor(obs, lo, hi, sigma_frac, dim_groups=dim_groups)


@torch.no_grad()
def moe_reward(kernel, obs):
    """Compute MoE reconstruction reward in [0, 1].

    Uses the estimator_config from training (l_min, l_max, steepness)
    with exponential mapping:  reward = exp(-(normalized_loss) * steepness)
    where normalized_loss = (loss - l_min) / (l_max - l_min).

    Falls back to linear mapping with loss_stats if estimator_config
    is not present in the kernel.

    Parameters
    ----------
    kernel : dict
    obs : array (D,) or (N, D)

    Returns
    -------
    float or np.ndarray (N,)
    """
    model = get_model(kernel)
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
    inp = torch.from_numpy(obs)
    out = model(inp)
    mse = ((out - inp) ** 2).mean(dim=1).numpy()

    est = kernel.get("estimator_config")
    if est is not None:
        l_min = est["l_min"]
        l_max = est["l_max"]
        steepness = est["steepness"]
        normalized = np.clip((mse - l_min) / (l_max - l_min + 1e-9), 0, None)
        reward = np.clip(np.exp(-normalized * steepness), 0, 1)
    else:
        # Legacy fallback: linear mapping with loss_stats
        lo = kernel["loss_stats"]["min"]
        hi = kernel["loss_stats"]["p98"]
        reward = 1.0 - np.clip((mse - lo) / (hi - lo + 1e-9), 0, 1)

    return float(reward[0]) if reward.shape[0] == 1 else reward.astype(np.float32)


def evaluate_full(kernel, obs, strictness="medium", sigma_frac=0.3):
    """Evaluate both MoE reconstruction reward AND bound constraint.

    Returns
    -------
    reward : float or array   — MoE reconstruction reward [0,1]
    factor : float or array   — bound constraint factor [0,1]
    """
    reward = moe_reward(kernel, obs)
    factor = evaluate(kernel, obs, strictness, sigma_frac)
    return reward, factor


# ── Summary / inspection ────────────────────────────────────────

def summarize(kernel):
    """Print a human-readable summary of a kernel file."""
    print(f"Environment: {kernel['env_name']}")
    print(f"Obs dim:     {kernel['obs_dim']}")
    cfg = kernel["moe_config"]
    print(f"MoE:         {cfg['input_dim']}D → {cfg['bottleneck_dim']}B "
          f"× {cfg['num_experts']}E")
    print(f"Demo points: {kernel['n_demo_points']}")

    dim_groups = kernel.get("dim_groups", {})
    if dim_groups:
        for name, dims in dim_groups.items():
            if isinstance(dims, torch.Tensor):
                dims = dims.tolist()
            print(f"  {name:>12}: dims {dims}")

    ls = kernel["loss_stats"]
    print(f"Loss stats:  min={ls['min']:.4f}  median={ls['median']:.4f}  "
          f"p98={ls['p98']:.4f}  max={ls['max']:.4f}")

    ds = kernel["dim_stats"]
    print(f"Dim ranges:  min=[{ds['min'].min():.2f}..{ds['min'].max():.2f}]  "
          f"max=[{ds['max'].min():.2f}..{ds['max'].max():.2f}]")

    print(f"\nStrictness levels:")
    print(f"  {'Label':>12}  {'s':>4}  {'lo_range':>20}  {'hi_range':>20}")
    for lv in kernel["strictness_levels"]:
        lo, hi = lv["bounds_lo"].numpy(), lv["bounds_hi"].numpy()
        print(f"  {lv['label']:>12}  {lv['level']:.1f}  "
              f"[{lo.min():+8.3f} .. {lo.max():+8.3f}]  "
              f"[{hi.min():+8.3f} .. {hi.max():+8.3f}]")


# ── CLI demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    envs = sys.argv[1:] if len(sys.argv) > 1 else [
        "Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]

    for env in envs:
        print(f"\n{'='*60}")
        kernel = load_kernel(env)
        summarize(kernel)

        # Quick self-test: evaluate the demo mean (should be ~1.0 for all levels)
        mean_obs = kernel["dim_stats"]["mean"].numpy()
        print(f"\n  Self-test (evaluate at demo mean):")
        for lv in kernel["strictness_levels"]:
            f = evaluate(kernel, mean_obs, strictness=lv["label"])
            print(f"    {lv['label']:>12}: factor={f:.6f}")

        # Test with an out-of-range observation (3× the max)
        oob_obs = kernel["dim_stats"]["max"].numpy() * 3.0
        print(f"  OOB test (3× max):")
        for lv in kernel["strictness_levels"]:
            f = evaluate(kernel, oob_obs, strictness=lv["label"])
            print(f"    {lv['label']:>12}: factor={f:.6f}")
