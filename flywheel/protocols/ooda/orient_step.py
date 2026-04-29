"""OrientStep — second OODA step, interprets observations."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class OrientStep(ABC):
    """Orient: interpret observations relative to the role's objective."""

    @abstractmethod
    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret raw observations into actionable context."""
        ...
