"""VerificationEmitter — emits VerificationResult artifact."""

from typing import Any, Dict

from flywheel.protocols.ooda.act_step import ActStep
from flywheel.protocols.artifacts.verification_result import VerificationResult


class VerificationEmitter(ActStep):
    """Act: emit a VerificationResult artifact."""

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        candidate = decision["candidate"]
        result = VerificationResult(
            candidate_ref=candidate.id,
            outcome=decision["outcome"],
            violated_norm_id=decision.get("norm_id"),
            evidence_ref=decision.get("evidence", ""),
        )
        return {"result": result, "candidate": candidate}
