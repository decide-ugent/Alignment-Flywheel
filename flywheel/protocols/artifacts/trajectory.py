"""Trajectory artifact — candidate behavior from the Proposer."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

from flywheel.protocols.artifacts._helpers import new_id
from flywheel.protocols.artifacts.trajectory_step import TrajectoryStep
from flywheel.protocols.enums import TrajectoryKind


@dataclass
class Trajectory:
    trajectory_id: str = field(default_factory=new_id)
    kind: TrajectoryKind = TrajectoryKind.ACTION
    steps: List[TrajectoryStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "kind": self.kind.value,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(
            trajectory_id=d.get("trajectory_id", new_id()),
            kind=TrajectoryKind(d.get("kind", "action")),
            steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
            metadata=d.get("metadata", {}),
        )
