"""PriorityTriage — priority queue by distance (spatial) or category (medical)."""

from flywheel.protocols.interfaces.base_triage import BaseTriage
from flywheel.protocols.enums import VerificationOutcome


CATEGORY_WEIGHT = {
    "missed_urgency": 1.0,
    "undertriaged_med": 0.7,
    "lab_no_context": 0.5,
    "vulnerable_patient": 0.6,
    "persistent_escalation": 0.6,
    "exploratory": 0.3,
}


class PriorityTriage(BaseTriage):
    """Priority queue — max(distance_to_path, category_weight)."""

    def __init__(self):
        self._queue = []

    def submit(self, result, candidate):
        if result.outcome == VerificationOutcome.VIOLATION:
            dist = candidate.context.get("dist_to_path", 0)
            cat = candidate.context.get("category", "")
            cat_w = CATEGORY_WEIGHT.get(cat, 0.3)
            score = max(dist, cat_w)
            self._queue.append((score, result, candidate))
            self._queue.sort(key=lambda x: -x[0])

    def pop_all(self):
        items = [(r, c) for _, r, c in self._queue]
        self._queue.clear()
        return items

    def size(self):
        return len(self._queue)
