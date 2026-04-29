"""DecideStep — third OODA step, chooses an action or strategy."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class DecideStep(ABC):
    """Decide: select a strategy or action based on oriented data."""

    @abstractmethod
    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        """Choose what to do."""
        ...
