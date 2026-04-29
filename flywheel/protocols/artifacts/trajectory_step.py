"""TrajectoryStep artifact — single element of a trajectory."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TrajectoryStep:
    step_index: int = 0
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step_index=d.get("step_index", 0),
            payload=d.get("payload"),
            metadata=d.get("metadata", {}),
        )
