"""MoE2DOracle — 2D spatial oracle adapter backed by a trained MoE autoencoder.

Implements BaseSpatialOracleAdapter for 2D PointMaze environments.
The reward is the reconstruction quality of the MoE model queried at
a given velocity, plus wall clearance, path-proximity, velocity-range
norms and goal bonus.

Suppressive patches (Gaussian kernels) are accumulated via send_patch(),
matching the same mechanism used by PrecomputedGridOracle.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist

from flywheel.protocols.interfaces.base_spatial_oracle_adapter import BaseSpatialOracleAdapter
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.enums import CorrectionType


class MoE2DOracle(BaseSpatialOracleAdapter):
    """Oracle that wraps a trained MoE autoencoder for 2D spatial reward.

    Supports velocity-aware queries: when velocity is provided the model
    is queried at (x, y, vx, vy) and the VELOCITY_RANGE norm is applied.
    """

    def __init__(
        self,
        model,
        walls_xy: np.ndarray,
        demo_pos: np.ndarray,
        demo_vel: Optional[np.ndarray] = None,
        query_vel: tuple = (1.5, 1.0),
        wall_clearance: float = 0.3,
        path_clearance: float = 0.2,
        path_sigma: float = 0.4,
        vel_penalty_sigma: float = 1.0,
        goal_xy: np.ndarray = None,
        goal_radius: float = 0.4,
        goal_reward: float = 5.0,
    ):
        self._model = model
        self._walls_xy = walls_xy.astype(np.float32)
        self._demo_pos = demo_pos[::5].astype(np.float32)
        self._query_vel = query_vel
        self._wall_clearance = wall_clearance
        self._path_clearance = path_clearance
        self._path_sigma = path_sigma
        self._vel_penalty_sigma = vel_penalty_sigma
        self._goal_xy = goal_xy
        self._goal_radius = goal_radius
        self._goal_reward = goal_reward
        self._version = 0

        # Compute velocity bounds from demonstrations (VELOCITY_RANGE norm)
        if demo_vel is not None:
            self._vel_min = demo_vel.min(axis=0).astype(np.float32)
            self._vel_max = demo_vel.max(axis=0).astype(np.float32)
        else:
            self._vel_min = None
            self._vel_max = None

        # Suppressive patch accumulators (position-only)
        self._supp_centers = np.empty((0, 2), dtype=np.float32)
        self._supp_bw = np.empty((0,), dtype=np.float32)
        self._n_wall_patches = 0

        # Auto-deploy structural Gaussian patches to replace masks
        self._deploy_wall_patches()
        self._deploy_path_patches()

        # Directional lane patches: suppress based on (position + velocity)
        # Each has center, direction (unit vel), strength, spatial bw, dir_sigma
        # Plus half-space gate: lane_normal + lane_cell_center
        self._lane_centers = np.empty((0, 2), dtype=np.float32)
        self._lane_dirs = np.empty((0, 2), dtype=np.float32)
        self._lane_strengths = np.empty((0,), dtype=np.float32)
        self._lane_bw = np.empty((0,), dtype=np.float32)
        self._lane_dir_sigma = np.empty((0,), dtype=np.float32)
        self._lane_normals = np.empty((0, 2), dtype=np.float32)
        self._lane_cell_centers = np.empty((0, 2), dtype=np.float32)
        self._lane_cell_radii = np.empty((0,), dtype=np.float32)

    def _deploy_wall_patches(self):
        """Wall suppression is computed analytically in _wall_suppression()
        using Gaussian decay from the wall cell edge.  No point patches
        are added to _supp_centers; the full rectangular cell area is
        treated as the patch footprint.
        """
        self._n_wall_patches = 0
        self._wall_bw = max(self._wall_clearance * 0.75, 0.15)

    def _deploy_path_patches(self):
        """Deploy Gaussian suppression patches at off-path locations.

        Replaces the old multiplicative path_mask.  Patches are placed on
        a regular grid wherever the nearest demonstration point is farther
        than path_clearance.  Their summed Gaussians create a smooth
        boundary that decays toward the demo corridors.
        """
        # Sample a grid over the spatial domain
        lo = self._demo_pos.min(axis=0) - 1.5
        hi = self._demo_pos.max(axis=0) + 1.5
        # Tight bandwidth so Gaussian tails don't bleed into corridors
        bw_val = max(self._path_sigma * 0.35, 0.10)
        # Deploy threshold: patches start well outside the corridor
        # (path_clearance + 2.5*bw ensures <1% contribution at corridor edge)
        deploy_threshold = self._path_clearance + 2.5 * bw_val
        spacing = max(bw_val * 1.4, 0.15)  # overlap neighbours smoothly
        xs = np.arange(lo[0], hi[0] + spacing, spacing)
        ys = np.arange(lo[1], hi[1] + spacing, spacing)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        # Distance to nearest demo point
        dists = cdist(grid_pts, self._demo_pos).min(axis=1)
        far = dists > deploy_threshold
        pts = grid_pts[far]
        if len(pts) == 0:
            return
        bw = np.full(len(pts), bw_val, dtype=np.float32)
        self._supp_centers = np.vstack([self._supp_centers, pts])
        self._supp_bw = np.concatenate([self._supp_bw, bw])

    def _raw_reward(self, xy: np.ndarray,
                    vel: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute raw MoE reward at given 2D positions + velocity."""
        if vel is None:
            vel = np.full((len(xy), 2), self._query_vel, dtype=np.float32)
        inp = torch.from_numpy(
            np.concatenate([xy.astype(np.float32), vel.astype(np.float32)],
                           axis=1))
        with torch.no_grad():
            out = self._model(inp)
        mse = ((out - inp) ** 2).sum(dim=1).numpy()
        lo, hi = mse.min(), np.percentile(mse, 98)
        return 1.0 - np.clip((mse - lo) / (hi - lo + 1e-9), 0, 1)

    def _wall_mask(self, xy: np.ndarray) -> np.ndarray:
        """Smooth distance-to-wall-edge penalty (Gaussian decay)."""
        diff = xy[:, None, :] - self._walls_xy[None, :, :]
        clamped = np.maximum(np.abs(diff) - 0.5, 0.0)
        rect_dist = np.sqrt((clamped ** 2).sum(axis=2))
        min_dist = rect_dist.min(axis=1)
        # Hard zero inside clearance, smooth decay near boundary
        mask = np.where(
            min_dist < self._wall_clearance * 0.5, 0.0,
            np.clip((min_dist - self._wall_clearance * 0.5) /
                    (self._wall_clearance * 0.5 + 1e-9), 0.0, 1.0))
        return mask.astype(np.float32)

    def _path_mask(self, xy: np.ndarray) -> np.ndarray:
        """Path-proximity Gaussian decay."""
        dists = cdist(xy, self._demo_pos).min(axis=1)
        excess = np.maximum(0.0, dists - self._path_clearance)
        return np.exp(-(excess ** 2) / (self._path_sigma ** 2))

    def _velocity_mask(self, vel: np.ndarray) -> np.ndarray:
        """VELOCITY_RANGE norm: Gaussian penalty for out-of-bounds velocity."""
        if self._vel_min is None:
            return np.ones(len(vel), dtype=np.float32)
        # Compute excess beyond demo velocity bounds per component
        excess_lo = np.maximum(0.0, self._vel_min - vel)
        excess_hi = np.maximum(0.0, vel - self._vel_max)
        excess = excess_lo + excess_hi  # (N, 2)
        penalty = np.sqrt((excess ** 2).sum(axis=1))
        return np.exp(-(penalty ** 2) /
                      (self._vel_penalty_sigma ** 2)).astype(np.float32)

    def _goal_bonus(self, xy: np.ndarray) -> np.ndarray:
        """Gaussian goal bonus."""
        if self._goal_xy is None:
            return np.zeros(len(xy), dtype=np.float32)
        dist = np.sqrt(((xy - self._goal_xy) ** 2).sum(axis=1))
        return self._goal_reward * np.exp(-0.5 * (dist / self._goal_radius) ** 2)

    def _wall_suppression(self, xy: np.ndarray) -> np.ndarray:
        """Gaussian suppression using distance to wall cell EDGES.

        Each wall occupies a full 1×1 cell.  Distance is measured from
        the query point to the nearest edge of the nearest wall cell
        (rectangular distance), then a Gaussian kernel is applied.
        Inside any wall cell the distance is 0 → suppression = 1.0.
        Uses max over walls so multiple nearby walls don't accumulate.
        """
        if len(self._walls_xy) == 0:
            return np.zeros(len(xy), dtype=np.float32)
        pts = xy.astype(np.float32)
        diff = pts[:, None, :] - self._walls_xy[None, :, :]   # (N, W, 2)
        # Rectangular distance: 0 inside cell, positive outside
        edge_dist_sq = np.sum(
            np.maximum(np.abs(diff) - 0.5, 0.0) ** 2, axis=2)  # (N, W)
        bw2 = 2.0 * self._wall_bw ** 2
        per_wall = np.exp(-edge_dist_sq / bw2)                 # (N, W)
        cutoff = (4.0 * self._wall_bw) ** 2
        per_wall[edge_dist_sq > cutoff] = 0.0
        return per_wall.max(axis=1).astype(np.float32)

    def _suppression(self, xy: np.ndarray) -> np.ndarray:
        """Compute total suppression: wall cells + accumulated patches.

        Wall suppression uses rectangular-distance Gaussians (full cell
        coverage).  Path / flaw patches use point-source Gaussians with
        a 4-sigma hard cutoff.
        """
        wall_s = self._wall_suppression(xy)

        patch_s = np.zeros(len(xy), dtype=np.float32)
        if len(self._supp_centers) > 0:
            pts = xy.astype(np.float32)
            inv = 1.0 / (2.0 * self._supp_bw ** 2)
            d2 = (np.sum(pts ** 2, axis=1)[:, None]
                  + np.sum(self._supp_centers ** 2, axis=1)[None, :]
                  - 2.0 * pts @ self._supp_centers.T)
            np.maximum(d2, 0, out=d2)
            exponents = d2 * inv[None, :]
            contrib = np.exp(-exponents)
            contrib[exponents > 16.0] = 0.0
            patch_s = np.minimum(np.sum(contrib, axis=1), 1.0)

        return np.minimum(wall_s + patch_s, 1.0)

    def _lane_suppression(self, xy: np.ndarray,
                          vel: Optional[np.ndarray]) -> np.ndarray:
        """Velocity-dependent lane suppression with half-space + direction gates.

        Each lane patch fires only when ALL three conditions are met:
          1. The point is spatially close to the patch center (Gaussian).
          2. The point is on the WRONG side of the corridor centre line
             (half-space gate: dot(pt - cell_center, lane_normal) > 0).
             This is a hard boundary — zero bleed to the correct side.
          3. The velocity direction matches the patch direction
             (hard cutoff: dot(vel, patch_dir) > 0, then Gaussian).
             Opposite-direction traffic gets exactly zero suppression.

        Processes in chunks to stay within memory limits.
        """
        if len(self._lane_centers) == 0 or vel is None:
            return np.zeros(len(xy), dtype=np.float32)

        pts = xy.astype(np.float32)
        n_pts = len(pts)
        n_patches = len(self._lane_centers)

        # Pre-compute per-point velocity unit vectors
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
        vel_unit = vel / vel_norm  # (n_pts, 2)

        # Pre-compute per-patch constants
        inv_bw = 1.0 / (2.0 * self._lane_bw ** 2)  # (n_patches,)
        inv_ds = 1.0 / (2.0 * self._lane_dir_sigma ** 2)  # (n_patches,)
        patch_c2 = np.sum(self._lane_centers ** 2, axis=1)  # (n_patches,)
        has_normals = len(self._lane_normals) == n_patches

        # Choose chunk size to keep peak memory ~500 MB
        # Each chunk needs ~4 arrays of shape (chunk, n_patches) × 4 bytes
        max_bytes = 500_000_000
        chunk_size = max(1, max_bytes // (n_patches * 4 * 5))

        result = np.zeros(n_pts, dtype=np.float32)

        for start in range(0, n_pts, chunk_size):
            end = min(start + chunk_size, n_pts)
            p = pts[start:end]
            v = vel_unit[start:end]
            n = end - start

            # Spatial Gaussian
            d2 = (np.sum(p ** 2, axis=1)[:, None]
                  + patch_c2[None, :]
                  - 2.0 * p @ self._lane_centers.T)
            np.maximum(d2, 0, out=d2)
            spatial = np.exp(-d2 * inv_bw[None, :])

            # ── Spatial pre-filter: skip patches far from this chunk ──
            # A patch contributes only if its spatial Gaussian exceeds
            # threshold for at least one point in the chunk.
            active_mask = spatial.max(axis=0) > 1e-3
            if not active_mask.any():
                continue
            active_idx = np.where(active_mask)[0]
            spatial = spatial[:, active_idx]
            lc = self._lane_centers[active_idx]
            ld = self._lane_dirs[active_idx]
            ls = self._lane_strengths[active_idx]
            ids = inv_ds[active_idx]
            n_active = len(active_idx)

            # Half-space + directional cell boundary gate
            if has_normals:
                ln = self._lane_normals[active_idx]
                lcc = self._lane_cell_centers[active_idx]
                lcr = self._lane_cell_radii[active_idx]
                offset = p[:, None, :] - lcc[None, :, :]
                half_dot = (offset * ln[None, :, :]).sum(axis=2)
                par = (offset * ld[None, :, :]).sum(axis=2)
                perp_dirs = np.stack([-ld[:, 1], ld[:, 0]], axis=1)
                perp = (offset * perp_dirs[None, :, :]).sum(axis=2)
                radii = lcr[None, :]
                in_cell = ((np.abs(par) < radii) &
                           (np.abs(perp) < 0.5))
                half_gate = ((half_dot > 0) & in_cell).astype(np.float32)
                del offset, half_dot, par, perp, in_cell
            else:
                half_gate = np.ones((n, n_active), dtype=np.float32)

            # Directional gate: hard cutoff at dot <= 0
            dots = v @ ld.T
            dir_gate = np.where(
                dots > 0,
                np.exp(-((1.0 - dots) ** 2) * ids[None, :]),
                0.0,
            )

            per_patch = spatial * half_gate * dir_gate * ls[None, :]
            result[start:end] = np.minimum(per_patch.max(axis=1), 1.0)

        return result

    def query_points(
        self,
        points: List[List[float]],
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """Query oracle. Points can be [x,y] or [x,y,vx,vy]."""
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        xy = pts[:, :2]

        # Velocity-aware: use provided velocity or default
        if pts.shape[1] >= 4:
            vel = pts[:, 2:4]
        else:
            vel = None

        reward = self._raw_reward(xy, vel)
        # Wall and path proximity are handled by structural Gaussian
        # patches (deployed in __init__) via _suppression(), not masks.
        if vel is not None:
            reward *= self._velocity_mask(vel)
        reward += self._goal_bonus(xy)
        # Spatial suppression (structural + flaw patches): subtractive
        spatial_supp = self._suppression(xy)
        safety = np.maximum(0.0, reward - np.minimum(spatial_supp, 1.0))
        # Lane suppression: MULTIPLICATIVE so it scales with total reward
        # and cascades through value iteration.  supp=0.9 → keep 10%.
        lane_supp = self._lane_suppression(xy, vel)
        safety *= (1.0 - lane_supp)

        unc = np.clip(1.0 - np.abs(2 * reward - 1), 0.1, 0.9) * 0.5 + 0.1
        return {
            "values": safety.tolist(),
            "uncertainties": unc.tolist() if include_uncertainty else None,
            "oracle_version": self.get_version(),
        }

    def send_patch(self, batch: GovernanceBatch) -> Dict[str, Any]:
        added = 0
        for lc in batch.local_corrections:
            if lc.correction_type == CorrectionType.SPATIAL_FLAW_PATCH:
                pt = lc.payload.get("flaw_point")
                bw = lc.payload.get("support_radius", 0.1)
                if pt is not None:
                    pt_2d = np.array([pt[:2]], dtype=np.float32)
                    self._supp_centers = np.vstack([
                        self._supp_centers, pt_2d,
                    ])
                    self._supp_bw = np.concatenate([
                        self._supp_bw,
                        np.array([bw], dtype=np.float32),
                    ])
                    added += 1

            elif lc.correction_type == CorrectionType.LANE_DIRECTION_PATCH:
                pt = lc.payload.get("flaw_point")
                direction = lc.payload.get("direction")
                strength = lc.payload.get("strength", 0.5)
                bw = lc.payload.get("support_radius", 0.15)
                dir_sigma = lc.payload.get("dir_sigma", 0.5)
                cell_center = lc.payload.get("cell_center")
                lane_normal = lc.payload.get("lane_normal")
                if pt is not None and direction is not None:
                    pt_2d = np.array([pt[:2]], dtype=np.float32)
                    dir_2d = np.array([direction[:2]], dtype=np.float32)
                    dir_2d = dir_2d / (np.linalg.norm(dir_2d) + 1e-9)
                    self._lane_centers = np.vstack([
                        self._lane_centers, pt_2d])
                    self._lane_dirs = np.vstack([
                        self._lane_dirs, dir_2d])
                    self._lane_strengths = np.concatenate([
                        self._lane_strengths,
                        np.array([strength], dtype=np.float32)])
                    self._lane_bw = np.concatenate([
                        self._lane_bw,
                        np.array([bw], dtype=np.float32)])
                    self._lane_dir_sigma = np.concatenate([
                        self._lane_dir_sigma,
                        np.array([dir_sigma], dtype=np.float32)])
                    # Half-space gate data
                    if cell_center is not None and lane_normal is not None:
                        cc = np.array([cell_center[:2]], dtype=np.float32)
                        ln = np.array([lane_normal[:2]], dtype=np.float32)
                        ln = ln / (np.linalg.norm(ln) + 1e-9)
                    else:
                        # Fallback: use patch center, zero normal (no gate)
                        cc = pt_2d.copy()
                        ln = np.zeros((1, 2), dtype=np.float32)
                    self._lane_cell_centers = np.vstack([
                        self._lane_cell_centers, cc])
                    self._lane_normals = np.vstack([
                        self._lane_normals, ln])
                    cell_radius = lc.payload.get("cell_radius", 0.5)
                    self._lane_cell_radii = np.concatenate([
                        self._lane_cell_radii,
                        np.array([cell_radius], dtype=np.float32)])
                    added += 1

        if added > 0:
            self._version += 1
        return {"applied": added > 0, "oracle_version": self.get_version()}

    def get_version(self) -> str:
        return f"oracle:v{self._version}"

    def query_grid(self, res: int = 80, xy_range=(-4.0, 4.0)):
        """Query the full 2D grid (position-only, default velocity)."""
        axis = np.linspace(xy_range[0], xy_range[1], res)
        xx, yy = np.meshgrid(axis, axis)
        xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        result = self.query_points(xy.tolist())
        return xx, yy, np.array(result["values"]).reshape(res, res)

    def query_grid_with_vel(self, vel: np.ndarray,
                            res: int = 80, xy_range=(-4.0, 4.0)):
        """Query grid at a specific velocity (vx, vy) for all positions."""
        axis = np.linspace(xy_range[0], xy_range[1], res)
        xx, yy = np.meshgrid(axis, axis)
        xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        vel_arr = np.broadcast_to(vel, (len(xy), 2)).astype(np.float32)
        pts = np.concatenate([xy, vel_arr], axis=1)
        result = self.query_points(pts.tolist())
        return np.array(result["values"]).reshape(res, res)

    def query_grid_actions(self, actions: np.ndarray, speed: float = 3.0,
                           res: int = 80, xy_range=(-4.0, 4.0)):
        """Query grid for each action direction.

        Returns reward_grid of shape (res, res, n_actions).
        Each action's unit direction is scaled to `speed` for the MoE query.
        """
        n_actions = len(actions)
        reward = np.zeros((res, res, n_actions), dtype=np.float32)
        for ai in range(n_actions):
            direction = actions[ai] / (np.linalg.norm(actions[ai]) + 1e-9)
            vel = (direction * speed).astype(np.float32)
            reward[:, :, ai] = self.query_grid_with_vel(vel, res, xy_range)
        return reward

    @property
    def patch_count(self):
        return len(self._supp_centers)

    @property
    def patch_centers(self):
        return self._supp_centers.copy()

    @property
    def patch_bandwidths(self):
        return self._supp_bw.copy()
