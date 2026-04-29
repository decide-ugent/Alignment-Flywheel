"""MedicalPriorityDecider — orders cases by category severity."""

from typing import Any, Dict

from flywheel.protocols.ooda.decide_step import DecideStep


PRIORITY = {
    "missed_urgency": 1.0,
    "undertriaged_med": 0.7,
    "lab_no_context": 0.5,
    "vulnerable_patient": 0.6,
    "exploratory": 0.3,
}


class MedicalPriorityDecider(DecideStep):
    """Decide: priority ordering by failure category."""

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        cases = oriented["classified_cases"]
        scored = [(PRIORITY.get(c["category"], 0.3), c) for c in cases]
        scored.sort(key=lambda x: -x[0])
        return {"prioritised_cases": [c for _, c in scored]}
