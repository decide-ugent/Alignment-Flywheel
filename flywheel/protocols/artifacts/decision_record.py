"""DecisionRecord artifact — complete runtime audit record."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from flywheel.protocols.artifacts._helpers import new_id, now_iso
from flywheel.protocols.enums import EnforcementAction


@dataclass
class DecisionRecord:
    id: str = field(default_factory=new_id)
    context: Dict[str, Any] = field(default_factory=dict)
    trajectory: Dict[str, Any] = field(default_factory=dict)
    action: EnforcementAction = EnforcementAction.ABSTAIN
    s: float = 0.0
    u: float = 0.0
    c_a: float = 1.0
    u_thresh: float = 0.5
    c_a_thresh: float = 0.6
    oracle_version: str = "oracle:v0"
    governance_version: str = "gov:v0"
    reasons: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRecord":
        return cls(
            id=d.get("id", new_id()),
            context=d.get("context", {}),
            trajectory=d.get("trajectory", {}),
            action=EnforcementAction(d.get("action", "abstain")),
            s=d.get("s", 0.0),
            u=d.get("u", 0.0),
            c_a=d.get("c_a", 1.0),
            u_thresh=d.get("u_thresh", 0.5),
            c_a_thresh=d.get("c_a_thresh", 0.6),
            oracle_version=d.get("oracle_version", "oracle:v0"),
            governance_version=d.get("governance_version", "gov:v0"),
            reasons=d.get("reasons", []),
            timestamp=d.get("timestamp", now_iso()),
        )
