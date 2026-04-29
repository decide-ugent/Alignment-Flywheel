"""HTTPOracleClient — calls the Oracle service over HTTP for predict/apply_batch."""

from typing import Any, Optional

import requests

from flywheel.protocols.interfaces.base_oracle import BaseOracle
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class HTTPOracleClient(BaseOracle):
    """BaseOracle implementation routed through HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")

    def predict(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> OracleRawOutput:
        r = requests.post(
            f"{self.base_url}/oracle/query",
            json={"context": context.to_dict(),
                  "trajectory": trajectory.to_dict()},
            timeout=30,
        )
        r.raise_for_status()
        return OracleRawOutput.from_dict(r.json())

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        r = requests.post(
            f"{self.base_url}/oracle/apply_batch",
            json=batch.to_dict(),
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("applied", False)

    def get_version(self) -> str:
        r = requests.get(f"{self.base_url}/oracle/version", timeout=10)
        r.raise_for_status()
        return r.json()["version"]
