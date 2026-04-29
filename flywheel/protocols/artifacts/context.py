"""Context artifact — runtime input wrapper."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from flywheel.protocols.artifacts._helpers import new_id


@dataclass
class Context:
    context_id: str = field(default_factory=new_id)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Context":
        return cls(
            context_id=d.get("context_id", new_id()),
            data=d.get("data", {}),
        )
