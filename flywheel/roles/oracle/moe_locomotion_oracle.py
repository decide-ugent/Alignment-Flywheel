"""MoELocomotionOracle — oracle adapter for MuJoCo locomotion environments.

Wraps a trained MoE autoencoder, computes reconstruction reward using
the estimator config (l_min, l_max, steepness), and accumulates
suppressive patches through the flywheel governance loop.

Patches are **per-expert bottleneck-space kernels**.  Each MoE expert
has its own encoder producing its own latent space — these spaces are
NOT aligned, so patches must live in the specific expert's bottleneck.

When a flaw is discovered, the gating network identifies the dominant
expert (highest gating weight), and the patch is placed in that
expert's bottleneck space.  At query time, suppression is computed
per-expert and weighted by the gating network:

    suppression(x) = Σ_e  w_e(x) · Σ_{i ∈ patches_e}  α_i · exp(-‖z_e(x) - c_i‖²/(2σ_i²))

Each patch stores:
    expert_idx — which expert's bottleneck this patch lives in
    center_z   — center in that expert's bottleneck space
    obs_center — original flaw observation (for auditing)
    bandwidth  — kernel width in that expert's bottleneck space
    strength   — max suppression [0, 1]
    group_mask — which dim_groups the flaw violated

The oracle's corrected output:
    safety = max(0, moe_reward - suppression)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from flywheel.protocols.interfaces.base_spatial_oracle_adapter import (
    BaseSpatialOracleAdapter,
)
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.enums import CorrectionType


class MoELocomotionOracle(BaseSpatialOracleAdapter):
    """Oracle backed by a MoE autoencoder for MuJoCo locomotion tasks.

    Parameters
    ----------
    model : MixtureOfExperts
        Trained MoE model (eval mode).
    estimator_config : dict
        ``{"l_min": float, "l_max": float, "steepness": int}``
    dim_stats : dict
        ``{"min": tensor, "max": tensor, "mean": tensor, "std": tensor}``
    dim_groups : dict
        ``{"position": tensor, "velocity": tensor}``
    demo_obs : np.ndarray (N, D)
        Demonstration observations (used as basin for regression testing).
    env_name : str
    """

    def __init__(
        self,
        model,
        estimator_config: dict,
        dim_stats: dict,
        dim_groups: dict,
        demo_obs: np.ndarray,
        env_name: str = "unknown",
        skip_dbscan: bool = False,
    ):
        self._model = model
        self._model.eval()
        self._est = estimator_config
        self._dim_stats = dim_stats
        self._dim_groups = dim_groups
        self._demo_obs = demo_obs.astype(np.float32)
        self._env_name = env_name
        self._obs_dim = demo_obs.shape[1]
        self._num_experts = len(model.experts)
        self._version = 0

        # Precompute per-expert demo bottleneck reps for basin testing
        with torch.no_grad():
            inp = torch.from_numpy(self._demo_obs)
            self._demo_z_per_expert = []   # list of (N, B) numpy arrays
            for expert in self._model.experts:
                self._demo_z_per_expert.append(
                    expert.encoder(inp).numpy()
                )
            self._demo_gating = self._model.gating_network(inp).numpy()  # (N, E)

        z_dim = self._demo_z_per_expert[0].shape[1]
        self._z_dim = z_dim

        # ── DBSCAN safeguard regions per expert ──────────────
        # Cluster demo latents to find dense core regions.
        # ALL demo points are safeguarded — no point is discarded.
        # Core samples get density-based radii (covers interpolation
        # between them), non-core points get nearest-neighbor radii
        # (individual protection).
        self._safeguard_per_expert = []  # list of {centers, radii}
        if skip_dbscan:
            print("  DBSCAN skipped (regression-only mode)")
        else:
            self._build_dbscan_safeguards()

        # Per-expert patch storage
        self._patches_per_expert = []
        for _ in range(self._num_experts):
            self._patches_per_expert.append({
                "centers_z": np.empty((0, z_dim), dtype=np.float32),
                "bw": np.empty(0, dtype=np.float32),
                "strength": np.empty(0, dtype=np.float32),
                "obs_centers": np.empty((0, self._obs_dim), dtype=np.float32),
                "group_masks": [],
            })

    def _build_dbscan_safeguards(self):
        """Run DBSCAN clustering for safeguard regions (used by cluster decider)."""
        min_samples = max(5, len(self._demo_obs) // 500)
        for e in range(self._num_experts):
            z_e = self._demo_z_per_expert[e]
            N = len(z_e)

            # k-NN for every point (used for eps + individual radii)
            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(z_e)
            k_dists, _ = nn.kneighbors(z_e)
            # eps = 90th percentile of k-th neighbor distance
            eps = float(np.percentile(k_dists[:, -1], 90))

            db = DBSCAN(eps=eps, min_samples=min_samples).fit(z_e)
            core_mask = np.zeros(N, dtype=bool)
            if len(db.core_sample_indices_) > 0:
                core_mask[db.core_sample_indices_] = True

            # Radii: core samples get k-th neighbor distance (larger,
            # covers interpolation); non-core get 1st neighbor distance
            # (tight individual protection).
            radii = np.empty(N, dtype=np.float32)
            radii[core_mask] = k_dists[core_mask, -1]       # k-th neighbor
            radii[~core_mask] = k_dists[~core_mask, 0]      # nearest neighbor

            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            n_core = int(core_mask.sum())
            n_noise = int((db.labels_ == -1).sum())
            print(f"    Expert {e}: {n_clusters} clusters, "
                  f"{n_core} core + {N - n_core} non-core "
                  f"= {N} safeguarded, {n_noise} noise, "
                  f"eps={eps:.4f}")

            self._safeguard_per_expert.append({
                "centers": z_e,          # ALL demo latents
                "radii": radii,          # per-point radius
                "core_mask": core_mask,  # which are dense cores
                "eps": eps,
                "labels": db.labels_,
            })

    # ── MoE helpers ──────────────────────────────────────────────

    def _encode_per_expert(self, x: torch.Tensor):
        """Encode through each expert separately. Returns list of (N,B) tensors + gating."""
        with torch.no_grad():
            gating = self._model.gating_network(x)  # (N, E)
            expert_z = []
            for expert in self._model.experts:
                expert_z.append(expert.encoder(x))   # (N, B)
        return expert_z, gating

    def _raw_moe_reward(self, obs: np.ndarray) -> np.ndarray:
        """MoE reconstruction reward using estimator config."""
        inp = torch.from_numpy(obs.astype(np.float32))
        with torch.no_grad():
            out = self._model(inp)
            mse = ((out - inp) ** 2).mean(dim=1).numpy()

        l_min = self._est["l_min"]
        l_max = self._est["l_max"]
        steepness = self._est["steepness"]
        normalized = np.clip((mse - l_min) / (l_max - l_min + 1e-9), 0, None)
        return np.clip(np.exp(-normalized * steepness), 0, 1).astype(np.float32)

    def _bottleneck_suppression(self, obs: np.ndarray) -> np.ndarray:
        """Compute gating-weighted per-expert suppression."""
        total_patches = sum(
            len(p["bw"]) for p in self._patches_per_expert
        )
        if total_patches == 0:
            return np.zeros(len(obs), dtype=np.float32)

        inp = torch.from_numpy(obs.astype(np.float32))
        expert_z_list, gating = self._encode_per_expert(inp)
        gating_np = gating.numpy()  # (N, E)

        total_supp = np.zeros(len(obs), dtype=np.float32)

        for e in range(self._num_experts):
            patches = self._patches_per_expert[e]
            if len(patches["bw"]) == 0:
                continue

            z_e = expert_z_list[e].numpy()          # (N, B)
            centers = patches["centers_z"]           # (K_e, B)
            bw = patches["bw"]                       # (K_e,)
            strength = patches["strength"]           # (K_e,)

            inv = 1.0 / (2.0 * bw ** 2 + 1e-12)
            d2 = (
                np.sum(z_e ** 2, axis=1)[:, None]
                + np.sum(centers ** 2, axis=1)[None, :]
                - 2.0 * z_e @ centers.T
            )
            np.maximum(d2, 0, out=d2)

            per_patch = np.exp(-d2 * inv[None, :]) * strength[None, :]
            expert_supp = per_patch.sum(axis=1)     # (N,)

            # Weight by gating
            total_supp += gating_np[:, e] * expert_supp

        return np.minimum(total_supp, 1.0)

    def _oob_ratio(self, obs: np.ndarray) -> np.ndarray:
        """Max OOB ratio across dim_groups (for uncertainty estimate).

        Uses effective_lo/effective_hi (with oob_margin) if available,
        otherwise falls back to raw demo min/max.
        """
        if "effective_lo" in self._dim_stats:
            lo = self._dim_stats["effective_lo"]
            hi = self._dim_stats["effective_hi"]
        else:
            lo = self._dim_stats["min"]
            hi = self._dim_stats["max"]
        lo = lo.numpy() if hasattr(lo, 'numpy') else lo
        hi = hi.numpy() if hasattr(hi, 'numpy') else hi
        # Normalise by raw demo range
        raw_lo = self._dim_stats["min"].numpy()
        raw_hi = self._dim_stats["max"].numpy()
        rng = raw_hi - raw_lo + 1e-9

        below = np.maximum(lo - obs, 0) / rng
        above = np.maximum(obs - hi, 0) / rng
        max_oob = np.maximum(below, above)

        if self._dim_groups:
            group_max = []
            for name, dims in self._dim_groups.items():
                d = dims.numpy() if isinstance(dims, torch.Tensor) else np.array(dims)
                group_max.append(max_oob[:, d].max(axis=1))
            return np.stack(group_max, axis=1).max(axis=1)
        return max_oob.max(axis=1)

    # ── BaseSpatialOracleAdapter interface ───────────────────────

    def query_points(
        self,
        points: List[List[float]],
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        reward = self._raw_moe_reward(pts)
        suppression = self._bottleneck_suppression(pts)
        safety = np.maximum(0.0, reward - suppression)

        if include_uncertainty:
            oob = self._oob_ratio(pts)
            unc = np.clip(0.2 + oob * 0.8, 0.1, 0.95)
        else:
            unc = None

        return {
            "values": safety.tolist(),
            "uncertainties": unc.tolist() if unc is not None else None,
            "oracle_version": self.get_version(),
        }

    def send_patch(self, batch: GovernanceBatch) -> Dict[str, Any]:
        added = 0
        for lc in batch.local_corrections:
            if lc.correction_type == CorrectionType.SPATIAL_FLAW_PATCH:
                obs_center = lc.payload.get("flaw_point")
                bw = lc.payload.get("support_radius", 0.5)
                strength = lc.payload.get("strength", 1.0)
                group = lc.payload.get("group_mask")
                expert_idx = lc.payload.get("expert_idx")

                if obs_center is not None:
                    obs_np = np.array([obs_center], dtype=np.float32)

                    # Determine dominant expert if not specified
                    if expert_idx is None:
                        with torch.no_grad():
                            g = self._model.gating_network(
                                torch.from_numpy(obs_np)
                            ).numpy()
                        expert_idx = int(np.argmax(g[0]))

                    # Encode through that specific expert
                    with torch.no_grad():
                        z = self._model.experts[expert_idx].encoder(
                            torch.from_numpy(obs_np)
                        ).numpy()

                    p = self._patches_per_expert[expert_idx]
                    p["centers_z"] = np.vstack([p["centers_z"], z])
                    p["obs_centers"] = np.vstack([p["obs_centers"], obs_np])
                    p["bw"] = np.concatenate(
                        [p["bw"], np.array([bw], dtype=np.float32)]
                    )
                    p["strength"] = np.concatenate(
                        [p["strength"], np.array([strength], dtype=np.float32)]
                    )
                    p["group_masks"].append(group)
                    added += 1

        if added > 0:
            self._version += 1
        return {"applied": added > 0, "oracle_version": self.get_version()}

    def get_version(self) -> str:
        return f"oracle:v{self._version}"

    # ── Convenience ──────────────────────────────────────────────

    @property
    def demo_obs(self) -> np.ndarray:
        return self._demo_obs

    @property
    def demo_z_per_expert(self) -> List[np.ndarray]:
        """Per-expert bottleneck reps of demo observations."""
        return self._demo_z_per_expert

    @property
    def demo_gating(self) -> np.ndarray:
        """Gating weights for demo observations, shape (N, E)."""
        return self._demo_gating

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def patch_count(self) -> int:
        return sum(len(p["bw"]) for p in self._patches_per_expert)

    @property
    def patch_count_per_expert(self) -> List[int]:
        return [len(p["bw"]) for p in self._patches_per_expert]

    @property
    def bottleneck_dim(self) -> int:
        return self._z_dim

    @property
    def safeguard_per_expert(self) -> List[dict]:
        """Per-expert DBSCAN safeguard regions: {cores, radii, eps}."""
        return self._safeguard_per_expert

    def encode_expert(self, obs: np.ndarray, expert_idx: int) -> np.ndarray:
        """Encode through a specific expert's encoder."""
        inp = torch.from_numpy(np.atleast_2d(obs).astype(np.float32))
        with torch.no_grad():
            return self._model.experts[expert_idx].encoder(inp).numpy()

    def dominant_expert(self, obs: np.ndarray) -> np.ndarray:
        """Return the dominant expert index for each observation."""
        inp = torch.from_numpy(np.atleast_2d(obs).astype(np.float32))
        with torch.no_grad():
            g = self._model.gating_network(inp).numpy()
        return np.argmax(g, axis=1)

    def ib_regression_check(
        self,
        ib_points: np.ndarray,
        max_ib_supp: float = 0.01,
        min_bw: float = 0.15,
    ) -> dict:
        """Shrink existing patches that suppress in-bounds test points.

        For each expert, computes per-patch suppression (gating-weighted)
        on *ib_points*.  Any patch whose max single-point suppression
        exceeds *max_ib_supp* is shrunk via bisection (and if still too
        high, its strength is reduced).

        Returns ``{"ib_shrinks": int, "ib_strength_cuts": int}``.
        """
        inp = torch.from_numpy(ib_points.astype(np.float32))
        expert_z_list, gating = self._encode_per_expert(inp)
        gating_np = gating.numpy()  # (N, E)

        total_shrinks = 0
        total_strength_cuts = 0

        for e in range(self._num_experts):
            patches = self._patches_per_expert[e]
            n_k = len(patches["bw"])
            if n_k == 0:
                continue

            z_e = expert_z_list[e].numpy()       # (N, B)
            centers = patches["centers_z"]        # (K, B)
            g_e = gating_np[:, e]                 # (N,)

            # Pairwise squared distances: (N, K)
            d2 = (
                np.sum(z_e ** 2, axis=1)[:, None]
                + np.sum(centers ** 2, axis=1)[None, :]
                - 2.0 * z_e @ centers.T
            )
            np.maximum(d2, 0, out=d2)

            for k in range(n_k):
                bw = patches["bw"][k]
                strength = patches["strength"][k]
                d2_k = d2[:, k]  # (N,)

                supp_k = g_e * strength * np.exp(-d2_k / (2.0 * bw ** 2))
                if supp_k.max() <= max_ib_supp:
                    continue

                # Bisection search to shrink bandwidth
                lo_bw, hi_bw = min_bw, bw
                for _ in range(15):
                    mid = (lo_bw + hi_bw) / 2.0
                    test = g_e * strength * np.exp(-d2_k / (2.0 * mid ** 2))
                    if test.max() <= max_ib_supp:
                        lo_bw = mid
                    else:
                        hi_bw = mid

                patches["bw"][k] = lo_bw
                total_shrinks += 1

                # Re-check after shrinking
                final = g_e * strength * np.exp(-d2_k / (2.0 * lo_bw ** 2))
                if final.max() > max_ib_supp:
                    worst = int(np.argmax(final))
                    gauss_val = g_e[worst] * np.exp(
                        -d2_k[worst] / (2.0 * lo_bw ** 2)
                    )
                    if gauss_val > 1e-9:
                        new_str = float(max_ib_supp / gauss_val)
                        patches["strength"][k] = max(0.1, min(strength, new_str))
                        total_strength_cuts += 1

        if total_shrinks > 0 or total_strength_cuts > 0:
            self._version += 1

        return {
            "ib_shrinks": total_shrinks,
            "ib_strength_cuts": total_strength_cuts,
        }

    def export_patches(self) -> dict:
        """Export all per-expert patches as a serializable dict."""
        per_expert = []
        for e, p in enumerate(self._patches_per_expert):
            per_expert.append({
                "expert_idx": e,
                "n_patches": len(p["bw"]),
                "centers_z": p["centers_z"].tolist(),
                "centers_obs": p["obs_centers"].tolist(),
                "bandwidths": p["bw"].tolist(),
                "strengths": p["strength"].tolist(),
                "group_masks": p["group_masks"],
            })
        return {
            "env_name": self._env_name,
            "num_experts": self._num_experts,
            "total_patches": self.patch_count,
            "per_expert": per_expert,
            "bottleneck_dim": self.bottleneck_dim,
            "oracle_version": self.get_version(),
        }
