"""BaseSpatialOracleAdapter — interface for spatial (point-queryable) oracles."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class BaseSpatialOracleAdapter(ABC):
    """Spatial oracle adapter — supports batch point queries and patching."""

    @abstractmethod
    def query_points(
        self,
        points: List[List[float]],
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def send_patch(self, batch: GovernanceBatch) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_version(self) -> str:
        ...
