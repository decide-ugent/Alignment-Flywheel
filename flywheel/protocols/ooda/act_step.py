"""ActStep — fourth OODA step, executes and emits artifacts."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ActStep(ABC):
    """Act: execute the decision and emit output artifacts."""

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute and return result artifacts."""
        ...
