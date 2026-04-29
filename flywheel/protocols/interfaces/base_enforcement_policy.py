"""BaseEnforcementPolicy — abstract interface for enforcement decision logic."""

from abc import ABC, abstractmethod

from flywheel.protocols.artifacts.unified_query_result import UnifiedQueryResult
from flywheel.protocols.artifacts.enforcement_result import EnforcementResult


class BaseEnforcementPolicy(ABC):
    """Converts a UnifiedQueryResult into an enforcement decision."""

    @abstractmethod
    def decide(self, unified: UnifiedQueryResult) -> EnforcementResult:
        ...
