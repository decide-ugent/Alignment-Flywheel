"""EnforcementResult artifact — final enforcement decision."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

from flywheel.protocols.enums import EnforcementAction


@dataclass
class EnforcementResult:
    action: EnforcementAction = EnforcementAction.ABSTAIN
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnforcementResult":
        return cls(
            action=EnforcementAction(d.get("action", d.get("decision", "abstain"))),
            reasons=d.get("reasons", []),
        )
