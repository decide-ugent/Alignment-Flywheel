"""FIFOTriage — simple first-in-first-out queue."""

from typing import List, Tuple

from flywheel.protocols.interfaces.base_triage import BaseTriage
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw
from flywheel.protocols.artifacts.verification_result import VerificationResult
from flywheel.protocols.enums import VerificationOutcome


class FIFOTriage(BaseTriage):
    """FIFO queue — violations processed in arrival order."""

    def __init__(self):
        self._queue = []

    def submit(self, result, candidate):
        if result.outcome == VerificationOutcome.VIOLATION:
            self._queue.append((result, candidate))

    def pop_all(self):
        items = list(self._queue)
        self._queue.clear()
        return items

    def size(self):
        return len(self._queue)
