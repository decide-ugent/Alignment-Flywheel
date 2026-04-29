"""CandidateFlaw artifact — Red Team discovery before verification."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from flywheel.protocols.artifacts._helpers import new_id, now_iso


@dataclass
class CandidateFlaw:
    id: str = field(default_factory=new_id)
    context: Dict[str, Any] = field(default_factory=dict)
    trajectory: Dict[str, Any] = field(default_factory=dict)
    s: float = 0.0
    u: float = 0.0
    u_thresh: float = 0.5
    v_O: str = "oracle:v0"
    seed_ref: Optional[str] = None
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateFlaw":
        return cls(
            id=d.get("id", new_id()),
            context=d.get("context", {}),
            trajectory=d.get("trajectory", {}),
            s=d.get("s", 0.0),
            u=d.get("u", 0.0),
            u_thresh=d.get("u_thresh", 0.5),
            v_O=d.get("v_O", "oracle:v0"),
            seed_ref=d.get("seed_ref"),
            timestamp=d.get("timestamp", now_iso()),
        )
