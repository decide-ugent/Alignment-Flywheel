"""HTTPProposerClient — calls the Proposer service over HTTP."""

import requests

from flywheel.protocols.interfaces.base_proposer import BaseProposer
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory


class HTTPProposerClient(BaseProposer):
    """BaseProposer implementation routed through HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")

    def propose(self, context: Context, **kwargs) -> Trajectory:
        r = requests.post(
            f"{self.base_url}/proposer/propose",
            json={"context": context.to_dict(), "kwargs": kwargs},
            timeout=30,
        )
        r.raise_for_status()
        return Trajectory.from_dict(r.json())
