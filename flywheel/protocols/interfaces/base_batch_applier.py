"""BaseBatchApplier — abstract interface for routing a GovernanceBatch."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class BaseBatchApplier(ABC):
    """Applies a GovernanceBatch to Oracle and Flywheel simultaneously."""

    @abstractmethod
    def apply(self, batch: GovernanceBatch) -> Dict[str, Any]:
        ...
