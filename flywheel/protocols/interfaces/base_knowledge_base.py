"""BaseKnowledgeBase — abstract interface for append-only event stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from flywheel.protocols.artifacts.decision_record import DecisionRecord
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw
from flywheel.protocols.artifacts.verification_result import VerificationResult
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class BaseKnowledgeBase(ABC):
    """Knowledge Base — append-only store of all governance events."""

    @abstractmethod
    def append_decision(self, r: DecisionRecord) -> str: ...

    @abstractmethod
    def append_candidate(self, f: CandidateFlaw) -> str: ...

    @abstractmethod
    def append_verification(self, r: VerificationResult) -> str: ...

    @abstractmethod
    def append_batch(self, b: GovernanceBatch) -> str: ...

    @abstractmethod
    def get_decisions(self) -> List[DecisionRecord]: ...

    @abstractmethod
    def get_candidates(self) -> List[CandidateFlaw]: ...

    @abstractmethod
    def get_verifications(self) -> List[VerificationResult]: ...

    @abstractmethod
    def get_batches(self) -> List[GovernanceBatch]: ...

    @abstractmethod
    def get_ledger(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def summary(self) -> Dict[str, int]: ...
