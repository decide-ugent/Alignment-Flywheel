"""BaseOracle — abstract interface for Safety Oracles."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class BaseOracle(ABC):
    """Safety Oracle — evaluates trajectories and accepts governance patches."""

    @abstractmethod
    def predict(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> OracleRawOutput:
        ...

    @abstractmethod
    def get_version(self) -> str:
        ...

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        """Default: accept no corrections. Subclasses override."""
        return True
