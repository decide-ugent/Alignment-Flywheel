"""FlywheelKernel — single-file inference: MoE + suppressive patches + LSH.

Load a .pt kernel file and call safety(obs) for patched reward.
Patches are defined in the MoE bottleneck (z-space) per expert,
looked up via SimHash LSH for O(1) per observation.

    fk = FlywheelKernel("kernels/Ant_flywheel_tight.pt")
    safety = fk.safety(obs)   # (N,) float32 in [0, 1]
"""

import numpy as np
import torch
from torch import nn

from IIRL.models import MixtureOfExperts


class FlywheelKernel:
    """All-in-one flywheel inference: MoE + estimator + patches + LSH.

    Load a ``.pt`` kernel file and call ``safety(obs)`` to get the
    reward the downstream agent sees.  Internally:

        reward     = clip(exp(-(MSE - l_min)/(l_max - l_min) * steepness))
        suppression = Σ_e  g_e · Σ_k  s_k · exp(-‖z_e - c_k‖²/(2·bw_k²))
        safety      = max(0, reward - suppression)

    Patches are looked up via SimHash LSH → O(1) per observation.

    Usage::

        fk = FlywheelKernel("kernels/Ant_flywheel_tight_regression_cap1000.pt")
        safety = fk.safety(obs)   # (N,) float32 in [0, 1]
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

        self._f = self._precompute(model, patches, bundle["estimator_config"])
        self._lsh = self._build_lsh(self._f)
        self._top_k = top_k

        self.env_name = bundle.get("env_name", "unknown")
        self.num_experts = cfg["num_experts"]
        self.bottleneck_dim = cfg["bottleneck_dim"]
        self.input_dim = cfg["input_dim"]
        self.n_patches = bundle["flywheel_patches"].get("total_patches", 0)
        self.strictness = bundle.get("strictness", {}).get("label", "unknown")

    # ── public ───────────────────────────────────────────────

    def safety(self, obs, top_k=None):
        """Patched reward the agent receives.  (N,) in [0, 1].

        ``safety = max(0, moe_reward - patch_suppression)``
        """
        k = top_k if top_k is not None else self._top_k
        return self._evaluate_lsh(obs, top_k=k)

    def reward(self, obs):
        """Base MoE reconstruction reward (ignoring patches).  (N,) in [0, 1]."""
        f = self._f
        x = np.atleast_2d(obs).astype(np.float32)
        gating = self._gating(x)
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

    # ── internals ────────────────────────────────────────────

    def _gating(self, x):
        f = self._f
        logits = x @ f["gate_WT"] + f["gate_b"]
        logits -= logits.max(axis=1, keepdims=True)
        el = np.exp(logits)
        return el / el.sum(axis=1, keepdims=True)

    @staticmethod
    def _precompute(model, patches_list, estimator_config):
        E = len(model.experts)
        B = model.experts[0].encoder[0].out_features
        with torch.no_grad():
            enc_WT = np.ascontiguousarray(
                torch.stack([e.encoder[0].weight for e in model.experts])
                .transpose(1, 2).numpy())
            enc_b = np.ascontiguousarray(
                torch.stack([e.encoder[0].bias for e in model.experts]).numpy())
            dec_WT = np.ascontiguousarray(
                torch.stack([e.decoder[0].weight for e in model.experts])
                .transpose(1, 2).numpy())
            dec_b = np.ascontiguousarray(
                torch.stack([e.decoder[0].bias for e in model.experts]).numpy())
            gate_WT = np.ascontiguousarray(
                model.gating_network.fc.weight.T.numpy())
            gate_b = model.gating_network.fc.bias.numpy().copy()

        max_K = max((p["n_patches"] for p in patches_list), default=0)
        has_patches = max_K > 0 and any(p["n_patches"] > 0 for p in patches_list)
        if max_K == 0:
            max_K = 1
        centers_T = np.zeros((E, B, max_K), dtype=np.float32)
        centers_sq = np.zeros((E, max_K), dtype=np.float32)
        strengths = np.zeros((E, max_K), dtype=np.float32)
        inv_2bw2 = np.zeros((E, max_K), dtype=np.float32)
        for p in patches_list:
            e, K = p["expert_idx"], p["n_patches"]
            if K > 0:
                c = np.array(p["centers_z"], dtype=np.float32)
                centers_T[e, :, :K] = c.T
                centers_sq[e, :K] = (c ** 2).sum(axis=1)
                bw = np.array(p["bandwidths"], dtype=np.float32)
                inv_2bw2[e, :K] = 1.0 / (2.0 * bw ** 2 + 1e-12)
                strengths[e, :K] = np.array(p["strengths"], dtype=np.float32)
        return {
            "enc_WT": enc_WT, "enc_b": enc_b,
            "dec_WT": dec_WT, "dec_b": dec_b,
            "gate_WT": gate_WT, "gate_b": gate_b,
            "centers_T": centers_T, "centers_sq": centers_sq,
            "strengths": strengths, "inv_2bw2": inv_2bw2,
            "l_min": np.float32(estimator_config["l_min"]),
            "l_max": np.float32(estimator_config["l_max"]),
            "steepness": np.float32(estimator_config["steepness"]),
            "has_patches": has_patches,
        }

    @staticmethod
    def _build_lsh(fast, n_tables=4, seed=42):
        E, B, K = fast["centers_T"].shape
        rng = np.random.RandomState(seed)
        total_patches = 0
        for e in range(E):
            total_patches += int((fast["strengths"][e] > 0).sum())
        max_K_e = max(int((fast["strengths"][e] > 0).sum()) for e in range(E))
        n_bits = max(14, int(np.ceil(np.log2(max(max_K_e, 1)))) + 4)
        n_bits = min(n_bits, B + 2)

        hyperplanes_list, thresholds_list, tables = [], [], []
        for t in range(n_tables):
            hp = rng.randn(B, n_bits).astype(np.float32)
            hp /= np.linalg.norm(hp, axis=0, keepdims=True)
            hyperplanes_list.append(hp)
            expert_thresholds, expert_tables = [], []
            for e in range(E):
                n_active = int((fast["strengths"][e] > 0).sum())
                if n_active == 0:
                    expert_thresholds.append(np.zeros(n_bits, dtype=np.float32))
                    expert_tables.append({})
                    continue
                centers = fast["centers_T"][e, :, :n_active].T
                proj = centers @ hp
                thresh = np.median(proj, axis=0).astype(np.float32)
                expert_thresholds.append(thresh)
                bits = (proj > thresh[None, :]).astype(np.int64)
                powers = 1 << np.arange(n_bits, dtype=np.int64)
                hash_vals = bits @ powers
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

        return {
            "hyperplanes": hyperplanes_list,
            "thresholds": thresholds_list,
            "tables": tables,
            "n_bits": n_bits,
            "n_tables": n_tables,
            "total_patches": total_patches,
        }

    def _evaluate_lsh(self, obs, top_k=None):
        f = self._f
        lsh = self._lsh
        x = np.atleast_2d(obs).astype(np.float32)
        N = len(x)

        gating = self._gating(x)
        E = gating.shape[1]

        if top_k is not None and top_k < E:
            top_idx = np.argpartition(-gating, top_k, axis=1)[:, :top_k]
            mask = np.zeros_like(gating)
            np.put_along_axis(mask, top_idx, 1.0, axis=1)
            gating = gating * mask
            gating = gating / (gating.sum(axis=1, keepdims=True) + 1e-12)
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

        # LSH patch lookup
        n_bits = lsh["n_bits"]
        n_tables = lsh["n_tables"]
        max_K = f["centers_T"].shape[2]

        if top_idx is not None:
            k_ = top_idx.shape[1]
            obs_active = np.repeat(np.arange(N), k_)
            exp_active = top_idx.ravel()
            z_hash = z_all[obs_active, exp_active, :]
            active_experts = set(exp_active.tolist())
        else:
            obs_active = np.repeat(np.arange(N), E)
            exp_active = np.tile(np.arange(E), N)
            z_hash = z_all.reshape(N * E, z_all.shape[2])
            active_experts = set(range(E))

        powers = 1 << np.arange(n_bits, dtype=np.int64)
        est_work = len(obs_active) * 4
        obs_arr = np.empty(est_work, dtype=np.int64)
        exp_arr = np.empty(est_work, dtype=np.int32)
        pat_arr = np.empty(est_work, dtype=np.int32)
        pos = 0

        for t in range(n_tables):
            hp = lsh["hyperplanes"][t]
            proj = z_hash @ hp
            thresh = np.stack(lsh["thresholds"][t])[exp_active]
            hashes_flat = ((proj > thresh).astype(np.int64)
                           * powers[None, :]).sum(axis=1)
            for e in active_experts:
                ht = lsh["tables"][t][e]
                if not ht:
                    continue
                e_mask = exp_active == e
                e_indices = np.where(e_mask)[0]
                e_hashes = hashes_flat[e_indices]
                unique_hv, inv = np.unique(e_hashes, return_inverse=True)
                for i in range(len(unique_hv)):
                    cands = ht.get(int(unique_hv[i]))
                    if cands is None:
                        continue
                    obs_group = np.where(inv == i)[0]
                    n_o, n_c = len(obs_group), len(cands)
                    count = n_o * n_c
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

        if n_tables > 1:
            keys = obs_flat * (E * max_K) + exp_flat * max_K + pat_flat
            _, unique_idx = np.unique(keys, return_index=True)
            obs_flat = obs_flat[unique_idx]
            exp_flat = exp_flat[unique_idx]
            pat_flat = pat_flat[unique_idx]

        z_eval = z_all[obs_flat, exp_flat, :]
        c_eval = f["centers_T"][exp_flat, :, pat_flat]
        s_eval = f["strengths"][exp_flat, pat_flat]
        inv_eval = f["inv_2bw2"][exp_flat, pat_flat]
        g_eval = gating[obs_flat, exp_flat]

        d2 = ((z_eval - c_eval) ** 2).sum(axis=1)
        gauss = s_eval * np.exp(-d2 * inv_eval)
        gauss[gauss < 1e-3] = 0.0

        supp = np.bincount(obs_flat.astype(np.intp),
                           weights=g_eval * gauss,
                           minlength=N).astype(np.float32)
        supp = np.minimum(1.0, supp)
        return np.maximum(0.0, reward - supp).astype(np.float32)
