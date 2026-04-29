"""FlywheelOverlay artifact — governance signals (c_a, c_a_thresh, audit)."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from flywheel.protocols.enums import AuditStatus


@dataclass
class FlywheelOverlay:
    c_a: float = 1.0
    c_a_thresh: float = 0.6
    v_G: str = "gov:v0"
    audit_status: AuditStatus = AuditStatus.INSUFFICIENT_COVERAGE
    audit_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["audit_status"] = self.audit_status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlywheelOverlay":
        return cls(
            c_a=d.get("c_a", 1.0),
            c_a_thresh=d.get("c_a_thresh", 0.6),
            v_G=d.get("v_G", "gov:v0"),
            audit_status=AuditStatus(d.get("audit_status", "insufficient_coverage")),
            audit_evidence=d.get("audit_evidence", {}),
        )
