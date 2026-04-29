"""HTTPFlywheelClient — calls the Flywheel service over HTTP."""

from typing import Any, List, Optional

import requests

from flywheel.protocols.interfaces.base_flywheel_overlay import BaseFlywheelOverlay
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.norm import Norm


class HTTPFlywheelClient(BaseFlywheelOverlay):
    """BaseFlywheelOverlay implementation routed through HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")

    def overlay(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> FlywheelOverlay:
        r = requests.post(
            f"{self.base_url}/flywheel/overlay",
            json={"context": context.to_dict(),
                  "trajectory": trajectory.to_dict()},
            timeout=30,
        )
        r.raise_for_status()
        return FlywheelOverlay.from_dict(r.json())

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        r = requests.post(
            f"{self.base_url}/flywheel/apply_batch",
            json=batch.to_dict(),
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("applied", False)

    def get_version(self) -> str:
        r = requests.get(f"{self.base_url}/flywheel/version", timeout=10)
        r.raise_for_status()
        return r.json()["version"]

    def get_norms(self) -> List[Norm]:
        r = requests.get(f"{self.base_url}/flywheel/norms", timeout=10)
        r.raise_for_status()
        return [Norm.from_dict(d) for d in r.json().get("norms", [])]
