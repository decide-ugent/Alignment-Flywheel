"""BaseTriage — abstract interface for triage queue implementations."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw
from flywheel.protocols.artifacts.verification_result import VerificationResult


class BaseTriage(ABC):
    """Queues verified violations for refinement."""

    @abstractmethod
    def submit(self, result: VerificationResult, candidate: CandidateFlaw) -> None:
        ...

    @abstractmethod
    def pop_all(self) -> List[Tuple[VerificationResult, CandidateFlaw]]:
        ...

    @abstractmethod
    def size(self) -> int:
        ...
