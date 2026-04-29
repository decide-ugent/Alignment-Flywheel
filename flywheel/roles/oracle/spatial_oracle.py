"""SpatialOracle — BaseOracle implementation backed by a spatial adapter."""

from typing import Any, Optional

from flywheel.protocols.interfaces.base_oracle import BaseOracle
from flywheel.protocols.interfaces.base_spatial_oracle_adapter import BaseSpatialOracleAdapter
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class SpatialOracle(BaseOracle):
    """Wraps a spatial adapter, exposing the standard BaseOracle interface."""

    def __init__(self, adapter: BaseSpatialOracleAdapter):
        self.adapter = adapter

    def predict(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> OracleRawOutput:
        point = None
        if trajectory.steps:
            payload = trajectory.steps[0].payload or {}
            if isinstance(payload, dict):
                point = payload.get("point")
        if point is None:
            point = context.data.get("point", [0, 0, 0])

        result = self.adapter.query_points([point])
        vals = result["values"]
        uncs = result.get("uncertainties") or [0.2]

        return OracleRawOutput(
            s=vals[0],
            u=uncs[0] if uncs else 0.2,
            u_thresh=0.5,
            v_O=self.adapter.get_version(),
        )

    def get_version(self) -> str:
        return self.adapter.get_version()

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        self.adapter.send_patch(batch)
        return True
