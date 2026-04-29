"""MedicalCandidateSubmitter — emits medical CandidateFlaw artifacts."""

from typing import Any, Dict

from flywheel.protocols.ooda.act_step import ActStep
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw


class MedicalCandidateSubmitter(ActStep):
    """Act: emit CandidateFlaw artifacts from prioritised medical cases."""

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        cases = decision["prioritised_cases"]
        candidates = []
        for case in cases:
            candidates.append(CandidateFlaw(
                context={
                    "case_type": case["case_type"],
                    "evidence_status": case["evidence_status"],
                    "acuity": case.get("acuity", "routine"),
                    "category": case["category"],
                    "proposed_disposition": case["proposed_disposition"],
                    "patient_age": case.get("patient_age", 45),
                },
                trajectory={
                    "kind": "message",
                    "steps": [{
                        "payload": {
                            "patient_message": case["patient_message"],
                            "draft_reply": case["draft_reply"],
                            "disposition": case["proposed_disposition"],
                        },
                        "metadata": {
                            "case_type": case["case_type"],
                            "evidence_status": case["evidence_status"],
                            "acuity": case.get("acuity", "routine"),
                            "patient_age": case.get("patient_age", 45),
                        },
                    }],
                },
                s=0.5, u=0.5, u_thresh=0.3, v_O="oracle:v0"))
        return {"candidates": candidates, "count": len(candidates)}
