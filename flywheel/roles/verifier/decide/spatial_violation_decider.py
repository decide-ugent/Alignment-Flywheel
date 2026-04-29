"""SpatialViolationDecider — decides spatial boundary violations."""

from typing import Any, Dict

from flywheel.protocols.ooda.decide_step import DecideStep
from flywheel.protocols.enums import VerificationOutcome, NormKind


class SpatialViolationDecider(DecideStep):
    """Decide: mark as violation if distance exceeds boundary."""

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        candidate = oriented["candidate"]
        boundary = oriented["boundary"]

        for norm, dist in oriented["matched_norms"]:
            if norm.kind == NormKind.SPATIAL_BOUNDARY and dist is not None:
                if dist > boundary:
                    return {
                        "candidate": candidate,
                        "outcome": VerificationOutcome.VIOLATION,
                        "norm_id": norm.id,
                        "evidence": f"dist={dist:.3f}>boundary={boundary}",
                    }

        return {
            "candidate": candidate,
            "outcome": VerificationOutcome.NO_VIOLATION,
            "norm_id": None,
            "evidence": "",
        }
