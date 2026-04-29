"""MedicalCaseOrienter — filters and classifies generated medical cases."""

from typing import Any, Dict

from flywheel.protocols.ooda.orient_step import OrientStep


class MedicalCaseOrienter(OrientStep):
    """Orient: keep only plausibly problematic cases."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        cases = observation["generated_cases"]
        classified = []
        for case in cases:
            if case["category"] not in ("exploratory",) or case["evidence_status"] != "supported":
                classified.append(case)
        return {"classified_cases": classified}
