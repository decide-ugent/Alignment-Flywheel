"""HTTPSpatialOracleClient — calls the Oracle service over HTTP."""

from typing import Any, Dict, List

import requests

from flywheel.protocols.interfaces.base_spatial_oracle_adapter import BaseSpatialOracleAdapter
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class HTTPSpatialOracleClient(BaseSpatialOracleAdapter):
    """Spatial oracle adapter — every call routed over HTTP to the API."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")

    def query_points(
        self,
        points: List[List[float]],
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/oracle/query",
            json={"points": points, "include_uncertainty": include_uncertainty},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def send_patch(self, batch: GovernanceBatch) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/oracle/apply_batch",
            json=batch.to_dict(),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def get_version(self) -> str:
        r = requests.get(f"{self.base_url}/oracle/version", timeout=10)
        r.raise_for_status()
        return r.json()["version"]
