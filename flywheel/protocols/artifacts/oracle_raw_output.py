"""OracleRawOutput artifact — Oracle signals (s, u, u_thresh, evidence)."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from flywheel.protocols.enums import EvidenceStatus


@dataclass
class OracleRawOutput:
    s: float = 0.0
    u: float = 0.0
    u_thresh: float = 0.5
    v_O: str = "oracle:v0"
    evidence_hooks: Dict[str, Any] = field(default_factory=dict)
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_status"] = self.evidence_status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OracleRawOutput":
        return cls(
            s=d.get("s", 0.0),
            u=d.get("u", 0.0),
            u_thresh=d.get("u_thresh", 0.5),
            v_O=d.get("v_O", "oracle:v0"),
            evidence_hooks=d.get("evidence_hooks", {}),
            evidence_status=EvidenceStatus(d.get("evidence_status", "unknown")),
        )
