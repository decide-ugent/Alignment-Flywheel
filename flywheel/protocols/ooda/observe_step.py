"""ObserveStep — first OODA step, acquires state from environment."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ObserveStep(ABC):
    """Observe: acquire relevant state (KB, oracle, external)."""

    @abstractmethod
    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read relevant state and return an observation dict.

        Args:
            context: runtime context (KB ref, oracle ref, iteration, prior results, config)

        Returns:
            observation passed to the Orient step
        """
        ...
