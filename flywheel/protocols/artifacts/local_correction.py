"""LocalCorrection artifact — single patch unit inside a GovernanceBatch."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from flywheel.protocols.artifacts._helpers import new_id
from flywheel.protocols.enums import CorrectionType


@dataclass
class LocalCorrection:
    correction_id: str = field(default_factory=new_id)
    correction_type: CorrectionType = CorrectionType.SPATIAL_FLAW_PATCH
    payload: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["correction_type"] = self.correction_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LocalCorrection":
        return cls(
            correction_id=d.get("correction_id", new_id()),
            correction_type=CorrectionType(d.get("correction_type", "spatial_flaw_patch")),
            payload=d.get("payload", {}),
            description=d.get("description", ""),
        )
