"""PrecomputedGridOracle — spatial oracle adapter backed by a pre-computed grid."""

from typing import Any, Dict, List

import numpy as np

from flywheel.protocols.interfaces.base_spatial_oracle_adapter import BaseSpatialOracleAdapter
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.enums import CorrectionType


class PrecomputedGridOracle(BaseSpatialOracleAdapter):
    """Oracle backed by precomputed loss values on a 3D grid.

    Used to avoid re-running a trained model during demos.  In
    production this would be replaced by a NeuralIIRLOracle that
    wraps a live model.
    """

    def __init__(
        self,
        loss_values: np.ndarray,
        grid_resolution: int = 20,
        bounds: tuple = (-1.0, 1.0),
        loss_cap: float = 0.3,
        patch_bandwidth: float = 0.1,
    ):
        self._loss = loss_values.astype(np.float32)
        self._res = grid_resolution
        self._bounds = bounds
        self._loss_cap = loss_cap
        self._version = 0

        axis = np.linspace(bounds[0], bounds[1], grid_resolution)
        self._grid = np.array(
            [(x, y, z) for x in axis for y in axis for z in axis],
            dtype=np.float32,
        )
        self._rewards = 1.0 - np.clip(self._loss / loss_cap, 0.0, 1.0)

        self._supp_centers = np.empty((0, 3), dtype=np.float32)
        self._supp_bw = np.empty((0,), dtype=np.float32)

    def _nearest_indices(self, points):
        lo, hi = self._bounds
        n = self._res
        norm = (points - lo) / (hi - lo) * (n - 1)
        idx = np.clip(np.round(norm).astype(int), 0, n - 1)
        return idx[:, 0] * n * n + idx[:, 1] * n + idx[:, 2]

    def _suppression(self, points):
        if len(self._supp_centers) == 0:
            return np.zeros(len(points), dtype=np.float32)
        pts = points.astype(np.float32)
        inv = 1.0 / (2.0 * self._supp_bw ** 2)
        d2 = (np.sum(pts ** 2, axis=1)[:, None]
              + np.sum(self._supp_centers ** 2, axis=1)[None, :]
              - 2.0 * pts @ self._supp_centers.T)
        np.maximum(d2, 0, out=d2)
        return np.minimum(np.sum(np.exp(-d2 * inv[None, :]), axis=1), 1.0)

    def query_points(
        self,
        points: List[List[float]],
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        rewards = self._rewards[self._nearest_indices(pts)]
        safety = np.maximum(0.0, rewards - self._suppression(pts))
        unc = np.clip(1.0 - np.abs(2 * rewards - 1), 0.1, 0.9) * 0.5 + 0.1
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
                    self._supp_centers = np.vstack([
                        self._supp_centers,
                        np.array([pt], dtype=np.float32),
                    ])
                    self._supp_bw = np.concatenate([
                        self._supp_bw,
                        np.array([bw], dtype=np.float32),
                    ])
                    added += 1
        if added > 0:
            self._version += 1
        return {"applied": added > 0, "oracle_version": self.get_version()}

    def get_version(self) -> str:
        return f"oracle:v{self._version}"

    @property
    def grid_points(self):
        return self._grid

    @property
    def rewards(self):
        return self._rewards
