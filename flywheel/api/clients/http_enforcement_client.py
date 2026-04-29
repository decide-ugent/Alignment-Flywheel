"""HTTPEnforcementClient — calls the Enforcement service over HTTP."""

import requests

from flywheel.protocols.interfaces.base_enforcement_policy import BaseEnforcementPolicy
from flywheel.protocols.artifacts.unified_query_result import UnifiedQueryResult
from flywheel.protocols.artifacts.enforcement_result import EnforcementResult


class HTTPEnforcementClient(BaseEnforcementPolicy):
    """BaseEnforcementPolicy implementation routed through HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")

    def decide(self, unified: UnifiedQueryResult) -> EnforcementResult:
        r = requests.post(
            f"{self.base_url}/enforcement/decide",
            json=unified.to_dict(),
            timeout=30,
        )
        r.raise_for_status()
        return EnforcementResult.from_dict(r.json())
