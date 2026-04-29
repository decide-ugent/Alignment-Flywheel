"""UnifiedQueryResult — merged Oracle + Flywheel signals for enforcement."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from flywheel.protocols.enums import EvidenceStatus, AuditStatus


@dataclass
class UnifiedQueryResult:
    s: float = 0.0
    u: float = 0.0
    c_a: float = 1.0
    u_thresh: float = 0.5
    c_a_thresh: float = 0.6
    oracle_version: str = "oracle:v0"
    governance_version: str = "gov:v0"
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    audit_status: AuditStatus = AuditStatus.INSUFFICIENT_COVERAGE
    evidence_hooks: Dict[str, Any] = field(default_factory=dict)
    audit_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_status"] = self.evidence_status.value
        d["audit_status"] = self.audit_status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedQueryResult":
        return cls(
            s=d.get("s", 0.0),
            u=d.get("u", 0.0),
            c_a=d.get("c_a", 1.0),
            u_thresh=d.get("u_thresh", 0.5),
            c_a_thresh=d.get("c_a_thresh", 0.6),
            oracle_version=d.get("oracle_version", "oracle:v0"),
            governance_version=d.get("governance_version", "gov:v0"),
            evidence_status=EvidenceStatus(d.get("evidence_status", "unknown")),
            audit_status=AuditStatus(d.get("audit_status", "insufficient_coverage")),
            evidence_hooks=d.get("evidence_hooks", {}),
            audit_evidence=d.get("audit_evidence", {}),
        )
