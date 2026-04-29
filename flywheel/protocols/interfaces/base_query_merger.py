"""BaseQueryMerger — abstract interface for merging Oracle + Flywheel signals."""

from abc import ABC, abstractmethod

from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.unified_query_result import UnifiedQueryResult


class BaseQueryMerger(ABC):
    """Merges Oracle and Flywheel signals into a unified result."""

    @abstractmethod
    def merge(
        self,
        oracle: OracleRawOutput,
        flywheel: FlywheelOverlay,
    ) -> UnifiedQueryResult:
        ...
