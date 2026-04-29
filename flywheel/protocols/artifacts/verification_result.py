"""VerificationResult artifact — verifier decision on a candidate."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from flywheel.protocols.artifacts._helpers import new_id, now_iso
from flywheel.protocols.enums import VerificationOutcome


@dataclass
class VerificationResult:
    id: str = field(default_factory=new_id)
    candidate_ref: str = ""
    outcome: VerificationOutcome = VerificationOutcome.NO_VIOLATION
    violated_norm_id: Optional[str] = None
    evidence_ref: str = ""
    reviewer_ref: str = "auto"
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerificationResult":
        return cls(
            id=d.get("id", new_id()),
            candidate_ref=d.get("candidate_ref", ""),
            outcome=VerificationOutcome(d.get("outcome", "no_violation")),
            violated_norm_id=d.get("violated_norm_id"),
            evidence_ref=d.get("evidence_ref", ""),
            reviewer_ref=d.get("reviewer_ref", "auto"),
            timestamp=d.get("timestamp", now_iso()),
        )
