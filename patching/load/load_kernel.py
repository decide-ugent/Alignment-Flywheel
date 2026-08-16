"""
load_kernel.py
==============
Load and evaluate flywheel kernels at inference time.

All heavy computation is precomputed at load time into contiguous
tensors.  At query time, ``evaluate()`` runs a single torch forward
pass with batched einsum — no Python loops, no numpy conversions in
the hot path.

Usage (new — single object):
    from patching.load.load_kernel import FlywheelKernel

    fk = FlywheelKernel("kernels/Ant_flywheel_tight_regression_cap1000.pt")
    safety = fk.safety(observations)        # (N,) in [0, 1]
    reward = fk.reward(observations)        # (N,) base MoE reward
    supp   = fk.suppression(observations)   # (N,) patch suppression

Usage (legacy dict API — still works):
    from patching.load.load_kernel import load_kernel, evaluate

    kernel = load_kernel("Ant")
    safety = evaluate(kernel, observations)
"""

import os
import numpy as np
import torch

from IIRL.models import MixtureOfExperts

HERE = os.path.dirname(os.path.abspath(__file__))


def _precompute_fast(model, patches_list, estimator_config):
    """Build contiguous numpy arrays for zero-overhead inference.

    All weight matrices are pre-transposed for optimal matmul layout.
    Everything is numpy — no torch dispatch in the hot path.
    """
    E = len(model.experts)
    B = model.experts[0].encoder[0].out_features
    D = model.experts[0].encoder[0].in_features

    with torch.no_grad():
        # Encoder: z_e = ReLU(x @ enc_WT[e] + enc_b[e])
        # enc_W is (E, B, D), transpose → (E, D, B) for x @ WT
        enc_WT = np.ascontiguousarray(
            torch.stack([e.encoder[0].weight for e in model.experts])
            .transpose(1, 2).numpy()                              # (E, D, B)
        )
        enc_b = np.ascontiguousarray(
            torch.stack([e.encoder[0].bias for e in model.experts])
            .numpy()                                               # (E, B)
        )

        # Decoder: recon_e = z_e @ dec_WT[e] + dec_b[e]
        # dec_W is (E, D, B), transpose → (E, B, D)
        dec_WT = np.ascontiguousarray(
            torch.stack([e.decoder[0].weight for e in model.experts])
            .transpose(1, 2).numpy()                              # (E, B, D)
        )
        dec_b = np.ascontiguousarray(
            torch.stack([e.decoder[0].bias for e in model.experts])
            .numpy()                                               # (E, D)
        )

        # Gating: g = softmax(x @ gate_WT + gate_b)
        gate_WT = np.ascontiguousarray(
            model.gating_network.fc.weight.T.numpy()              # (D, E)
        )
        gate_b = model.gating_network.fc.bias.numpy().copy()      # (E,)

    # Patch data — pad all experts to max_K
    max_K = max((p["n_patches"] for p in patches_list), default=0)
    has_patches = max_K > 0 and any(p["n_patches"] > 0 for p in patches_list)
    if max_K == 0:
        max_K = 1

    # centers transposed: (E, B, K) for z @ centers_T
    centers_T = np.zeros((E, B, max_K), dtype=np.float32)
    centers_sq = np.zeros((E, max_K), dtype=np.float32)
    strengths = np.zeros((E, max_K), dtype=np.float32)
    inv_2bw2 = np.zeros((E, max_K), dtype=np.float32)

    for p in patches_list:
        e = p["expert_idx"]
        K = p["n_patches"]
        if K > 0:
            c = np.array(p["centers_z"], dtype=np.float32)          # (K, B)
            centers_T[e, :, :K] = c.T                               # (B, K)
            centers_sq[e, :K] = (c ** 2).sum(axis=1)
            bw = np.array(p["bandwidths"], dtype=np.float32)
            inv_2bw2[e, :K] = 1.0 / (2.0 * bw ** 2 + 1e-12)
            strengths[e, :K] = np.array(p["strengths"], dtype=np.float32)

    # Estimator config
    l_min = np.float32(estimator_config["l_min"])
    l_max = np.float32(estimator_config["l_max"])
    steepness = np.float32(estimator_config["steepness"])

    return {
        "enc_WT": enc_WT, "enc_b": enc_b,
        "dec_WT": dec_WT, "dec_b": dec_b,
        "gate_WT": gate_WT, "gate_b": gate_b,
        "centers_T": centers_T, "centers_sq": centers_sq,
        "strengths": strengths, "inv_2bw2": inv_2bw2,
        "l_min": l_min, "l_max": l_max, "steepness": steepness,
        "has_patches": has_patches,
    }


# ── Locality-Sensitive Hashing for O(1) patch lookup ───────────

_GAUSS_CUTOFF = 3.72   # bw * 3.72 ≈ radius where Gaussian < 1e-3


def _build_lsh(fast, n_bits=None, n_tables=4, seed=42):
    """Build multi-table SimHash for O(1) patch lookup.

    Uses random hyperplane projections with data-dependent thresholds
    (median split per bit) so ReLU-positive bottleneck vectors hash
    uniformly.  n_bits auto-scales with patch count to keep bucket
    sizes approximately constant (~1-3 per bucket).

    Build time : O(K × n_tables × n_bits)  — one pass per table.
    Query time : O(n_tables)               — one dict lookup per table.
    Empty bucket ⇒ no patches apply ⇒ skip instantly.

    Parameters
    ----------
    fast : dict from ``_precompute_fast``
    n_bits : int or None
        Bits per hash code. If None, auto-scaled: max(14, ceil(log2(K))+4).
    n_tables : int
        Number of independent hash tables.
    seed : int
        RNG seed for reproducible hyperplanes.
    """
    E, B, K = fast["centers_T"].shape
    rng = np.random.RandomState(seed)

    # Recover bandwidths for stats
    all_bws = []
    total_patches = 0
    for e in range(E):
        mask = fast["strengths"][e] > 0
        n_active = int(mask.sum())
        total_patches += n_active
        if mask.any():
            bws = 1.0 / np.sqrt(2.0 * fast["inv_2bw2"][e][mask] + 1e-12)
            all_bws.extend(bws.tolist())
    median_bw = float(np.median(all_bws)) if all_bws else 0.0

    # Auto-scale bits: ensure hash space >> K per expert
    # Cap at B+2: in B-dim space, only B directions are independent
    max_K_e = max(int((fast["strengths"][e] > 0).sum()) for e in range(E))
    if n_bits is None:
        n_bits = max(14, int(np.ceil(np.log2(max(max_K_e, 1)))) + 4)
        n_bits = min(n_bits, B + 2)  # no benefit beyond B independent dims

    hyperplanes_list = []   # n_tables × (B, n_bits)
    thresholds_list = []    # n_tables × E × (n_bits,)
    tables = []             # n_tables × E × {hash_int → np.array(patch_indices)}

    for t in range(n_tables):
        # Random unit hyperplanes for this table
        hp = rng.randn(B, n_bits).astype(np.float32)
        hp /= np.linalg.norm(hp, axis=0, keepdims=True)
        hyperplanes_list.append(hp)

        expert_thresholds = []
        expert_tables = []
        for e in range(E):
            n_active = int((fast["strengths"][e] > 0).sum())
            if n_active == 0:
                expert_thresholds.append(np.zeros(n_bits, dtype=np.float32))
                expert_tables.append({})
                continue

            centers = fast["centers_T"][e, :, :n_active].T  # (K_e, B)

            proj = centers @ hp                               # (K_e, n_bits)
            # Data-dependent thresholds: median split per bit → 50/50
            thresh = np.median(proj, axis=0).astype(np.float32)  # (n_bits,)
            expert_thresholds.append(thresh)

            bits = (proj > thresh[None, :]).astype(np.int64)
            powers = 1 << np.arange(n_bits, dtype=np.int64)
            hash_vals = bits @ powers                         # (K_e,)

            # Vectorized grouping via argsort (no Python loop over K)
            order = np.argsort(hash_vals)
            sorted_hv = hash_vals[order]
            diffs = np.concatenate([[1], np.diff(sorted_hv)])
            boundaries = np.where(diffs != 0)[0]
            boundaries = np.concatenate([boundaries, [n_active]])

            ht = {}
            for i in range(len(boundaries) - 1):
                s, end = int(boundaries[i]), int(boundaries[i + 1])
                ht[int(sorted_hv[s])] = order[s:end].astype(np.int32)
            expert_tables.append(ht)

        thresholds_list.append(expert_thresholds)
        tables.append(expert_tables)

    total_cells = sum(
        len(ht) for expert_tables in tables for ht in expert_tables
    )

    return {
        "hyperplanes": hyperplanes_list,
        "thresholds": thresholds_list,
        "tables": tables,
        "n_bits": n_bits,
        "n_tables": n_tables,
        "total_patches": total_patches,
        "total_cells": total_cells,
        "median_bw": median_bw,
    }


def _hash_points(z, hyperplanes, thresholds, n_bits):
    """Compute SimHash codes for z (N, B) → (N,) int64."""
    proj = z @ hyperplanes            # (N, n_bits)
    bits = (proj > thresholds[None, :])
    powers = (1 << np.arange(n_bits, dtype=np.int64))
    return bits.astype(np.int64) @ powers


def evaluate_lsh(kernel, obs, top_k=None):
    """Evaluate with LSH-accelerated patch lookup.

    Same output as ``evaluate()`` but suppression uses SimHash to
    skip patches that are too far away.  For most observations
    (especially safe ones), all hash buckets are empty and suppression
    is zero with no distance computation.

    Parameters
    ----------
    kernel : dict from load_kernel()
    obs : np.ndarray, shape (N, D) or (D,)
    top_k : int or None
        If set, only the top-K experts (by gating weight) are used
        per observation.  The rest are zeroed out and skipped entirely.
        This makes cost O(top_k) instead of O(E).

    Complexity: O(n_tables × n_bits) per observation per active expert for
                hashing, plus O(|candidates|) for patches in matching buckets.
    """
    f = kernel["_fast"]
    lsh = kernel["_lsh"]
    x = np.atleast_2d(obs).astype(np.float32)
    N = len(x)

    # ── Gating + encode + reconstruct + reward  (same as evaluate) ──
    logits = x @ f["gate_WT"] + f["gate_b"]
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    gating = exp_l / exp_l.sum(axis=1, keepdims=True)

    E = gating.shape[1]

    # ── Top-K gating: zero out non-active experts ──
    if top_k is not None and top_k < E:
        top_idx = np.argpartition(-gating, top_k, axis=1)[:, :top_k]  # (N, top_k)
        mask = np.zeros_like(gating)
        np.put_along_axis(mask, top_idx, 1.0, axis=1)
        gating = gating * mask
        gating_sum = gating.sum(axis=1, keepdims=True)
        gating = gating / (gating_sum + 1e-12)  # renormalize over active
    else:
        top_idx = None

    z_all = np.maximum(0,
        np.einsum('nd,edb->neb', x, f["enc_WT"]) + f["enc_b"])

    recon_all = np.einsum('neb,ebd->ned', z_all, f["dec_WT"]) + f["dec_b"]
    recon = (gating[:, :, None] * recon_all).sum(axis=1)
    mse = ((recon - x) ** 2).mean(axis=1)
    norm = np.clip(
        (mse - f["l_min"]) / (f["l_max"] - f["l_min"] + 1e-9), 0, None)
    reward = np.clip(np.exp(-norm * f["steepness"]), 0, 1)

    if not f["has_patches"] or lsh["total_patches"] == 0:
        return reward.astype(np.float32)

    # ── LSH suppression (active experts only with top-K) ─────────────
    hyperplanes_list = lsh["hyperplanes"]
    thresholds_list = lsh["thresholds"]
    all_tables = lsh["tables"]
    n_bits = lsh["n_bits"]
    n_tables = lsh["n_tables"]
    E = z_all.shape[1]
    B = z_all.shape[2]
    max_K = f["centers_T"].shape[2]

    # Build active (obs, expert) pairs — top-K skips inactive experts
    if top_idx is not None:
        obs_active = np.repeat(np.arange(N), top_k)       # (N*top_k,)
        exp_active = top_idx.ravel()                       # (N*top_k,)
        z_hash = z_all[obs_active, exp_active, :]          # (N*top_k, B)
        active_experts = set(exp_active.tolist())
    else:
        obs_active = np.repeat(np.arange(N), E)
        exp_active = np.tile(np.arange(E), N)
        z_hash = z_all.reshape(N * E, B)
        active_experts = set(range(E))

    powers = 1 << np.arange(n_bits, dtype=np.int64)

    # Pre-allocate work arrays (generous estimate, grow if needed)
    est_work = len(obs_active) * 4
    obs_arr = np.empty(est_work, dtype=np.int64)
    exp_arr = np.empty(est_work, dtype=np.int32)
    pat_arr = np.empty(est_work, dtype=np.int32)
    pos = 0

    for t in range(n_tables):
        hp = hyperplanes_list[t]
        proj = z_hash @ hp                                 # (M, n_bits)
        thresh = np.stack(thresholds_list[t])[exp_active]  # (M, n_bits)
        hashes_flat = ((proj > thresh).astype(np.int64)
                       * powers[None, :]).sum(axis=1)       # (M,)

        for e in active_experts:
            ht = all_tables[t][e]
            if not ht:
                continue
            e_mask = exp_active == e
            e_indices = np.where(e_mask)[0]
            e_obs = obs_active[e_indices]
            e_hashes = hashes_flat[e_indices]

            unique_hv, inv = np.unique(e_hashes, return_inverse=True)
            for i in range(len(unique_hv)):
                cands = ht.get(int(unique_hv[i]))
                if cands is None:
                    continue
                obs_group = np.where(inv == i)[0]
                n_o, n_c = len(obs_group), len(cands)
                count = n_o * n_c
                # Grow arrays if needed
                if pos + count > len(obs_arr):
                    new_size = max(len(obs_arr) * 2, pos + count)
                    obs_arr = np.resize(obs_arr, new_size)
                    exp_arr = np.resize(exp_arr, new_size)
                    pat_arr = np.resize(pat_arr, new_size)
                obs_arr[pos:pos+count] = np.repeat(obs_group, n_c)
                exp_arr[pos:pos+count] = e
                pat_arr[pos:pos+count] = np.tile(cands, n_o)
                pos += count

    if pos == 0:
        return reward.astype(np.float32)

    obs_flat = obs_arr[:pos]
    exp_flat = exp_arr[:pos]
    pat_flat = pat_arr[:pos]

    # Deduplicate across tables with fast 1D key hash
    if n_tables > 1:
        keys = obs_flat * (E * max_K) + exp_flat * max_K + pat_flat
        _, unique_idx = np.unique(keys, return_index=True)
        obs_flat = obs_flat[unique_idx]
        exp_flat = exp_flat[unique_idx]
        pat_flat = pat_flat[unique_idx]

    # Step 3+4: ONE MATRIX — gather + vectorized Gaussian + scatter
    z_eval = z_all[obs_flat, exp_flat, :]                  # (total, B)
    c_eval = f["centers_T"][exp_flat, :, pat_flat]         # (total, B)
    s_eval = f["strengths"][exp_flat, pat_flat]             # (total,)
    inv_eval = f["inv_2bw2"][exp_flat, pat_flat]            # (total,)
    g_eval = gating[obs_flat, exp_flat]                     # (total,)

    d2 = ((z_eval - c_eval) ** 2).sum(axis=1)              # (total,)
    gauss = s_eval * np.exp(-d2 * inv_eval)
    gauss[gauss < 1e-3] = 0.0

    supp = np.bincount(obs_flat.astype(np.intp),
                       weights=g_eval * gauss,
                       minlength=N).astype(np.float32)

    supp = np.minimum(1.0, supp)
    return np.maximum(0.0, reward - supp).astype(np.float32)


def load_kernel(env_name, strictness="medium", decider="cluster", kernel_dir=None):
    """Load a flywheel kernel .pt file and return a ready-to-use dict.

    Parameters
    ----------
    env_name : str
        One of: Ant, HalfCheetah, Hopper, Swimmer, Walker2d
    strictness : str
        One of: very_loose, loose, medium, tight, very_tight
    decider : str
        One of: "regression", "cluster"
    kernel_dir : str, optional
        Directory containing the .pt files. Defaults to kernels/ next to this file.

    Returns
    -------
    dict with precomputed tensors for fast inference.
    """
    if kernel_dir is None:
        kernel_dir = os.path.join(HERE, "kernels")
    path = os.path.join(kernel_dir, f"{env_name}_flywheel_{strictness}_{decider}.pt")
    if not os.path.exists(path):
        path2 = os.path.join(kernel_dir, f"{env_name}_flywheel_{strictness}.pt")
        path3 = os.path.join(kernel_dir, f"{env_name}_flywheel_kernel.pt")
        if os.path.exists(path2):
            path = path2
        elif os.path.exists(path3):
            path = path3
    bundle = torch.load(path, map_location="cpu", weights_only=False)

    cfg = bundle["moe_config"]
    model = MixtureOfExperts(
        input_dim=cfg["input_dim"],
        bottleneck_dim=cfg["bottleneck_dim"],
        num_experts=cfg["num_experts"],
    )
    model.load_state_dict(bundle["moe_state_dict"])
    model.eval()

    dim_stats = {}
    for k, v in bundle["dim_stats"].items():
        dim_stats[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)

    dim_groups = {}
    for k, v in bundle["dim_groups"].items():
        dim_groups[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)

    patches = bundle["flywheel_patches"]["per_expert"]

    fast = _precompute_fast(model, patches, bundle["estimator_config"])
    lsh = _build_lsh(fast)

    return {
        "model": model,
        "estimator_config": bundle["estimator_config"],
        "dim_stats": dim_stats,
        "dim_groups": dim_groups,
        "patches": patches,
        "num_experts": cfg["num_experts"],
        "bottleneck_dim": cfg["bottleneck_dim"],
        "strictness": bundle.get("strictness", {"label": "unknown"}),
        "env_name": env_name,
        "_fast": fast,
        "_lsh": lsh,
    }


def evaluate(kernel, obs):
    """Evaluate observations — returns safety score in [0, 1].

    Pure numpy, fully vectorised across experts via einsum.
    No Python loops, no torch dispatch in the hot path.

    Parameters
    ----------
    kernel : dict from load_kernel()
    obs : np.ndarray, shape (N, D) or (D,)

    Returns
    -------
    np.ndarray, shape (N,) in [0, 1]
    """
    f = kernel["_fast"]
    x = np.atleast_2d(obs).astype(np.float32)  # (N, D)

    # ── Gating: softmax(x @ gate_WT + gate_b) ──
    logits = x @ f["gate_WT"] + f["gate_b"]           # (N, E)
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    gating = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N, E)

    # ── Encode all experts: (N,D)×(E,D,B) → (N,E,B) ──
    z_all = np.maximum(0,
        np.einsum('nd,edb->neb', x, f["enc_WT"]) + f["enc_b"])  # (N,E,B)

    # ── Reconstruct: (N,E,B)×(E,B,D) → (N,E,D) ──
    recon_all = np.einsum('neb,ebd->ned', z_all, f["dec_WT"]) + f["dec_b"]
    recon = (gating[:, :, None] * recon_all).sum(axis=1)        # (N,D)
    mse = ((recon - x) ** 2).mean(axis=1)

    # ── MoE reward ──
    norm = np.clip(
        (mse - f["l_min"]) / (f["l_max"] - f["l_min"] + 1e-9), 0, None
    )
    reward = np.clip(np.exp(-norm * f["steepness"]), 0, 1)

    # ── Suppression: (N,E,B)×(E,B,K) → (N,E,K) ──
    if f["has_patches"]:
        z_sq = (z_all ** 2).sum(axis=-1, keepdims=True)             # (N,E,1)
        dot = np.einsum('neb,ebk->nek', z_all, f["centers_T"])      # (N,E,K)
        d2 = z_sq + f["centers_sq"][None] - 2 * dot                 # (N,E,K)
        np.maximum(d2, 0, out=d2)
        gauss = f["strengths"][None] * np.exp(
            -d2 * f["inv_2bw2"][None])                               # (N,E,K)
        gauss[gauss < 1e-3] = 0.0                                   # clip negligible tails
        supp = np.minimum(1.0,
            (gating * gauss.sum(axis=-1)).sum(axis=-1))              # (N,)
    else:
        supp = np.zeros(len(x), dtype=np.float32)

    return np.maximum(0.0, reward - supp).astype(np.float32)


def moe_reward(kernel, obs):
    """Compute the base MoE reconstruction reward.  (N,) in [0, 1]."""
    f = kernel["_fast"]
    x = np.atleast_2d(obs).astype(np.float32)
    logits = x @ f["gate_WT"] + f["gate_b"]
    logits -= logits.max(axis=1, keepdims=True)
    el = np.exp(logits)
    gating = el / el.sum(axis=1, keepdims=True)
    z_all = np.maximum(0,
        np.einsum('nd,edb->neb', x, f["enc_WT"]) + f["enc_b"])
    recon_all = np.einsum('neb,ebd->ned', z_all, f["dec_WT"]) + f["dec_b"]
    recon = (gating[:, :, None] * recon_all).sum(axis=1)
    mse = ((recon - x) ** 2).mean(axis=1)
    norm = np.clip((mse - f["l_min"]) / (f["l_max"] - f["l_min"] + 1e-9), 0, None)
    return np.clip(np.exp(-norm * f["steepness"]), 0, 1).astype(np.float32)


def suppression(kernel, obs):
    """Compute per-expert gating-weighted suppression.  (N,) in [0, 1]."""
    f = kernel["_fast"]
    x = np.atleast_2d(obs).astype(np.float32)
    if not f["has_patches"]:
        return np.zeros(len(x), dtype=np.float32)
    logits = x @ f["gate_WT"] + f["gate_b"]
    logits -= logits.max(axis=1, keepdims=True)
    el = np.exp(logits)
    gating = el / el.sum(axis=1, keepdims=True)
    z_all = np.maximum(0,
        np.einsum('nd,edb->neb', x, f["enc_WT"]) + f["enc_b"])
    z_sq = (z_all ** 2).sum(axis=-1, keepdims=True)
    dot = np.einsum('neb,ebk->nek', z_all, f["centers_T"])
    d2 = np.maximum(0, z_sq + f["centers_sq"][None] - 2 * dot)
    gauss = f["strengths"][None] * np.exp(-d2 * f["inv_2bw2"][None])
    gauss[gauss < 1e-3] = 0.0  # clip negligible tails
    return np.minimum(1.0,
        (gating * gauss.sum(axis=-1)).sum(axis=-1)).astype(np.float32)


# ── FlywheelKernel: single-object API ───────────────────────────

class FlywheelKernel:
    """All-in-one flywheel inference object.

    Bundles the MoE autoencoder, estimator, Gaussian patches, and
    LSH tables into a single object with a one-call interface.

    Parameters
    ----------
    path : str
        Path to a ``.pt`` kernel file produced by ``build_kernels_flywheel.py``.
    top_k : int or None
        Default top-K expert count for LSH evaluation.  ``None`` = use all.

    Examples
    --------
    >>> fk = FlywheelKernel("kernels/Ant_flywheel_tight_regression_cap1000.pt")
    >>> safety = fk.safety(obs)          # (N,) in [0, 1]
    >>> reward = fk.reward(obs)          # (N,) base MoE reward (no patches)
    >>> supp   = fk.suppression(obs)     # (N,) patch suppression
    """

    def __init__(self, path, top_k=None):
        bundle = torch.load(path, map_location="cpu", weights_only=False)

        cfg = bundle["moe_config"]
        model = MixtureOfExperts(
            input_dim=cfg["input_dim"],
            bottleneck_dim=cfg["bottleneck_dim"],
            num_experts=cfg["num_experts"],
        )
        model.load_state_dict(bundle["moe_state_dict"])
        model.eval()

        patches = bundle["flywheel_patches"]["per_expert"]

        self._fast = _precompute_fast(model, patches, bundle["estimator_config"])
        self._lsh = _build_lsh(self._fast)
        self._top_k = top_k
        self.env_name = bundle.get("env_name", "unknown")
        self.num_experts = cfg["num_experts"]
        self.bottleneck_dim = cfg["bottleneck_dim"]
        self.input_dim = cfg["input_dim"]
        self.n_patches = bundle["flywheel_patches"].get("total_patches", 0)
        self.strictness = bundle.get("strictness", {}).get("label", "unknown")

        # Expose dim_stats/dim_groups for callers that need them
        self.dim_stats = {}
        for k, v in bundle.get("dim_stats", {}).items():
            self.dim_stats[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        self.dim_groups = {}
        for k, v in bundle.get("dim_groups", {}).items():
            self.dim_groups[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)

    # ── Public API ───────────────────────────────────────────

    def safety(self, obs, top_k=None):
        """Compute patched safety score.  (N,) in [0, 1].

        ``safety = max(0, moe_reward - suppression)``

        Uses LSH for O(1) patch lookup per observation.
        """
        k = top_k if top_k is not None else self._top_k
        kernel = {"_fast": self._fast, "_lsh": self._lsh}
        return evaluate_lsh(kernel, obs, top_k=k)

    def reward(self, obs):
        """Base MoE reconstruction reward (no patches).  (N,) in [0, 1]."""
        f = self._fast
        x = np.atleast_2d(obs).astype(np.float32)
        logits = x @ f["gate_WT"] + f["gate_b"]
        logits -= logits.max(axis=1, keepdims=True)
        el = np.exp(logits)
        gating = el / el.sum(axis=1, keepdims=True)
        z_all = np.maximum(0,
            np.einsum('nd,edb->neb', x, f["enc_WT"]) + f["enc_b"])
        recon_all = np.einsum('neb,ebd->ned', z_all, f["dec_WT"]) + f["dec_b"]
        recon = (gating[:, :, None] * recon_all).sum(axis=1)
        mse = ((recon - x) ** 2).mean(axis=1)
        norm = np.clip(
            (mse - f["l_min"]) / (f["l_max"] - f["l_min"] + 1e-9), 0, None)
        return np.clip(np.exp(-norm * f["steepness"]), 0, 1).astype(np.float32)

    def suppression(self, obs):
        """Patch suppression only.  (N,) in [0, 1]."""
        return np.maximum(0.0, self.reward(obs) - self.safety(obs))

    def __repr__(self):
        return (f"FlywheelKernel({self.env_name}, {self.strictness}, "
                f"{self.input_dim}D→{self.bottleneck_dim}B×{self.num_experts}E, "
                f"{self.n_patches} patches)")


# ── AntMaze Constraint Kernel ──────────────────────────────────

def _path_distance(xy, segments):
    """Min distance from (N,2) xy to nearest segment in (S,2,2)."""
    A = segments[:, 0, :]   # (S, 2)
    B = segments[:, 1, :]   # (S, 2)
    AB = B - A
    l2 = (AB ** 2).sum(axis=1)  # (S,)

    N = len(xy)
    S = len(segments)
    dists = np.empty((N, S), dtype=np.float32)

    for s in range(S):
        ax, ay = A[s]
        dx, dy = AB[s]
        seg_l2 = l2[s]
        if seg_l2 < 1e-10:
            dists[:, s] = np.sqrt((xy[:, 0] - ax) ** 2 +
                                  (xy[:, 1] - ay) ** 2)
        else:
            t = np.clip(((xy[:, 0] - ax) * dx + (xy[:, 1] - ay) * dy)
                        / seg_l2, 0, 1)
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dists[:, s] = np.sqrt((xy[:, 0] - proj_x) ** 2 +
                                  (xy[:, 1] - proj_y) ** 2)
    return dists.min(axis=1)


class AntMazeKernel:
    """Flywheel kernel for AntMaze with corridor path-distance constraint.

    Combines MoE reconstruction reward, Gaussian suppression patches,
    and spatial corridor constraint into a single safety score.

    Parameters
    ----------
    path : str
        Path to an AntMaze constraint ``.pt`` file produced by
        ``build_antmaze_constraints.py``, or a standard AntMaze flywheel
        kernel from ``build_kernels_flywheel.py``.
    top_k : int or None
        Default top-K expert count for LSH evaluation.

    Examples
    --------
    >>> ak = AntMazeKernel("kernels/AntMaze_constraint_path_tight_t1.0_cap1000.pt")
    >>> safety = ak.safety(obs)              # (N,) in [0, 1]
    >>> dists  = ak.corridor_distance(obs)   # (N,) MuJoCo units from centerline
    >>> mask   = ak.corridor_violation(obs)  # (N,) bool — True if outside corridor
    """

    def __init__(self, path, top_k=None):
        bundle = torch.load(path, map_location="cpu", weights_only=False)

        cfg = bundle["moe_config"]
        model = MixtureOfExperts(
            input_dim=cfg["input_dim"],
            bottleneck_dim=cfg["bottleneck_dim"],
            num_experts=cfg["num_experts"],
        )
        model.load_state_dict(bundle["moe_state_dict"])
        model.eval()

        patches = bundle["flywheel_patches"]["per_expert"]

        self._fast = _precompute_fast(model, patches, bundle["estimator_config"])
        self._lsh = _build_lsh(self._fast)
        self._top_k = top_k
        self.env_name = bundle.get("env_name", "AntMaze")
        self.num_experts = cfg["num_experts"]
        self.bottleneck_dim = cfg["bottleneck_dim"]
        self.input_dim = cfg["input_dim"]
        self.n_patches = bundle["flywheel_patches"].get("total_patches", 0)
        self.strictness = bundle.get("strictness", {}).get("label", "unknown")

        # Corridor geometry (present in constraint kernels)
        segs = bundle.get("corridor_segments")
        if segs is not None:
            self._segments = np.array(segs, dtype=np.float32)
        else:
            self._segments = None
        self.path_threshold = bundle.get("path_threshold", 1.0)

        # Dim stats / groups
        self.dim_stats = {}
        for k, v in bundle.get("dim_stats", {}).items():
            self.dim_stats[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        self.dim_groups = {}
        for k, v in bundle.get("dim_groups", {}).items():
            self.dim_groups[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)

    # ── Corridor geometry ────────────────────────────────────

    def corridor_distance(self, obs):
        """Min distance from ant (x,y) to corridor centerline.  (N,)"""
        if self._segments is None:
            raise ValueError("No corridor segments in this kernel. "
                             "Load a constraint kernel built by "
                             "build_antmaze_constraints.py.")
        x = np.atleast_2d(obs).astype(np.float32)
        return _path_distance(x[:, :2], self._segments)

    def corridor_violation(self, obs, threshold=None):
        """Bool mask: True where ant is outside corridor threshold.  (N,)"""
        t = threshold if threshold is not None else self.path_threshold
        return self.corridor_distance(obs) > t

    def corridor_factor(self, obs, threshold=None):
        """Smooth [0,1] factor: 1 inside corridor, decays outside.  (N,)

        Uses exponential decay: factor = exp(-excess / threshold)
        where excess = max(0, distance - threshold).
        """
        t = threshold if threshold is not None else self.path_threshold
        dists = self.corridor_distance(obs)
        excess = np.maximum(0, dists - t)
        return np.exp(-excess / (t + 1e-9)).astype(np.float32)

    # ── MoE reward & suppression ─────────────────────────────

    def reward(self, obs):
        """Base MoE reconstruction reward (no patches).  (N,) in [0, 1]."""
        f = self._fast
        x = np.atleast_2d(obs).astype(np.float32)
        logits = x @ f["gate_WT"] + f["gate_b"]
        logits -= logits.max(axis=1, keepdims=True)
        el = np.exp(logits)
        gating = el / el.sum(axis=1, keepdims=True)
        z_all = np.maximum(0,
            np.einsum('nd,edb->neb', x, f["enc_WT"]) + f["enc_b"])
        recon_all = np.einsum('neb,ebd->ned', z_all, f["dec_WT"]) + f["dec_b"]
        recon = (gating[:, :, None] * recon_all).sum(axis=1)
        mse = ((recon - x) ** 2).mean(axis=1)
        norm = np.clip(
            (mse - f["l_min"]) / (f["l_max"] - f["l_min"] + 1e-9), 0, None)
        return np.clip(np.exp(-norm * f["steepness"]), 0, 1).astype(np.float32)

    def patched_reward(self, obs, top_k=None):
        """MoE reward with patch suppression (no corridor).  (N,) in [0, 1]."""
        k = top_k if top_k is not None else self._top_k
        kernel = {"_fast": self._fast, "_lsh": self._lsh}
        return evaluate_lsh(kernel, obs, top_k=k)

    def safety(self, obs, top_k=None, threshold=None):
        """Combined safety: min(patched_reward, corridor_factor).  (N,) in [0, 1].

        If no corridor segments are present (standard flywheel kernel),
        returns just the patched reward.
        """
        patched = self.patched_reward(obs, top_k=top_k)
        if self._segments is not None:
            cf = self.corridor_factor(obs, threshold=threshold)
            return np.minimum(patched, cf).astype(np.float32)
        return patched

    def __repr__(self):
        seg_info = (f", {len(self._segments)} corridor segments"
                    if self._segments is not None else "")
        return (f"AntMazeKernel({self.strictness}, "
                f"{self.input_dim}D->{self.bottleneck_dim}B x {self.num_experts}E, "
                f"{self.n_patches} patches{seg_info})")


if __name__ == "__main__":
    # Quick sanity check
    for env in ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]:
        path = os.path.join(HERE, "kernels", f"{env}_flywheel_kernel.pt")
        if not os.path.exists(path):
            continue
        k = load_kernel(env)
        n_patches = sum(p["n_patches"] for p in k["patches"])
        expert_str = " ".join(
            f"E{p['expert_idx']}={p['n_patches']}" for p in k["patches"]
        )
        print(f"{env:>12}: {k['bottleneck_dim']}B × {k['num_experts']}E, "
              f"{n_patches} patches [{expert_str}]")

        # Test on random obs
        obs = np.random.randn(100, k["model"].experts[0].encoder[0].in_features).astype(np.float32)
        safety = evaluate(k, obs)
        print(f"             random obs safety: "
              f"mean={safety.mean():.4f} max={safety.max():.4f}")
