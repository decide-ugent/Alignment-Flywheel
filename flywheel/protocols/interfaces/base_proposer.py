"""BaseProposer — abstract interface for trajectory proposers."""

from abc import ABC, abstractmethod

from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory


class BaseProposer(ABC):
    """Produces a candidate Trajectory given a Context."""

    @abstractmethod
    def propose(self, context: Context, **kwargs) -> Trajectory:
        ...
