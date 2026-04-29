"""GovernanceBatch artifact — the deployment unit B_O."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

from flywheel.protocols.artifacts._helpers import new_id, now_iso
from flywheel.protocols.artifacts.local_correction import LocalCorrection


@dataclass
class GovernanceBatch:
    batch_id: str = field(default_factory=new_id)
    from_oracle_version: str = "oracle:v0"
    to_oracle_version: str = "oracle:v1"
    local_corrections: List[LocalCorrection] = field(default_factory=list)
    regression_evidence: Dict[str, Any] = field(default_factory=dict)
    rollout_metadata: Dict[str, Any] = field(default_factory=dict)
    signature: str = "unsigned"
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "from_oracle_version": self.from_oracle_version,
            "to_oracle_version": self.to_oracle_version,
            "local_corrections": [lc.to_dict() for lc in self.local_corrections],
            "regression_evidence": self.regression_evidence,
            "rollout_metadata": self.rollout_metadata,
            "signature": self.signature,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GovernanceBatch":
        return cls(
            batch_id=d.get("batch_id", new_id()),
            from_oracle_version=d.get("from_oracle_version", "oracle:v0"),
            to_oracle_version=d.get("to_oracle_version", "oracle:v1"),
            local_corrections=[LocalCorrection.from_dict(lc) for lc in d.get("local_corrections", [])],
            regression_evidence=d.get("regression_evidence", {}),
            rollout_metadata=d.get("rollout_metadata", {}),
            signature=d.get("signature", "unsigned"),
            timestamp=d.get("timestamp", now_iso()),
        )
