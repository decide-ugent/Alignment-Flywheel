"""Norm artifact — typed normative constraint in Phi."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

from flywheel.protocols.artifacts._helpers import new_id
from flywheel.protocols.enums import NormKind


@dataclass
class Norm:
    id: str = field(default_factory=new_id)
    kind: NormKind = NormKind.PREDICATE
    spec: Dict[str, Any] = field(default_factory=dict)
    severity: float = 0.5
    weight: float = 1.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "spec": self.spec,
            "severity": self.severity,
            "weight": self.weight,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Norm":
        return cls(
            id=d.get("id", new_id()),
            kind=NormKind(d.get("kind", "predicate")),
            spec=d.get("spec", {}),
            severity=d.get("severity", 0.5),
            weight=d.get("weight", 1.0),
            description=d.get("description", ""),
        )
