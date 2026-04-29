"""BaseFlywheelOverlay — abstract interface for Flywheel overlay providers."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.norm import Norm


class BaseFlywheelOverlay(ABC):
    """Flywheel overlay — provides coverage signals, accepts governance patches."""

    @abstractmethod
    def overlay(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> FlywheelOverlay:
        ...

    @abstractmethod
    def apply_batch(self, batch: GovernanceBatch) -> bool:
        ...

    @abstractmethod
    def get_version(self) -> str:
        ...

    @abstractmethod
    def get_norms(self) -> List[Norm]:
        ...
