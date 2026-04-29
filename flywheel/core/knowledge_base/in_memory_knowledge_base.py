"""InMemoryKnowledgeBase — append-only thread-safe in-memory store."""

import threading
from typing import Any, Dict, List

from flywheel.protocols.interfaces.base_knowledge_base import BaseKnowledgeBase
from flywheel.protocols.artifacts.decision_record import DecisionRecord
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw
from flywheel.protocols.artifacts.verification_result import VerificationResult
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class InMemoryKnowledgeBase(BaseKnowledgeBase):
    """Thread-safe append-only store of governance events."""

    def __init__(self):
        self._lock = threading.Lock()
        self._decisions: List[DecisionRecord] = []
        self._candidates: List[CandidateFlaw] = []
        self._verifications: List[VerificationResult] = []
        self._batches: List[GovernanceBatch] = []
        self._ledger: List[Dict[str, Any]] = []

    def append_decision(self, r: DecisionRecord) -> str:
        with self._lock:
            self._decisions.append(r)
            return r.id

    def append_candidate(self, f: CandidateFlaw) -> str:
        with self._lock:
            self._candidates.append(f)
            return f.id

    def append_verification(self, r: VerificationResult) -> str:
        with self._lock:
            self._verifications.append(r)
            return r.id

    def append_batch(self, b: GovernanceBatch) -> str:
        with self._lock:
            self._batches.append(b)
            self._ledger.append({
                "batch_id": b.batch_id,
                "from": b.from_oracle_version,
                "to": b.to_oracle_version,
                "corrections": len(b.local_corrections),
                "timestamp": b.timestamp,
            })
            return b.batch_id

    def get_decisions(self) -> List[DecisionRecord]:
        return list(self._decisions)

    def get_candidates(self) -> List[CandidateFlaw]:
        return list(self._candidates)

    def get_verifications(self) -> List[VerificationResult]:
        return list(self._verifications)

    def get_batches(self) -> List[GovernanceBatch]:
        return list(self._batches)

    def get_ledger(self) -> List[Dict[str, Any]]:
        return list(self._ledger)

    def summary(self) -> Dict[str, int]:
        return {
            "decisions": len(self._decisions),
            "candidates": len(self._candidates),
            "verifications": len(self._verifications),
            "batches": len(self._batches),
        }
