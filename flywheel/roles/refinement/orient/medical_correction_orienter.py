"""MedicalCorrectionOrienter — maps flaw categories to correction types."""

from typing import Any, Dict

from flywheel.protocols.ooda.orient_step import OrientStep
from flywheel.protocols.artifacts.local_correction import LocalCorrection
from flywheel.protocols.enums import CorrectionType


WEAK_EVIDENCE = ("insufficient", "conflicting", "unknown")


def _extract_med_kw(text):
    for kw in ["increase dose", "stop taking", "start taking",
               "switch to", "half your current", "dosage", "tapering"]:
        if kw in text.lower():
            return kw
    return None


def _extract_urgency_kw(text):
    for kw in ["chest pain", "difficulty breathing", "severe headache",
               "vision loss", "shortness of breath", "seizure",
               "nothing to worry", "see if it improves",
               "schedule an appointment next week", "call back if"]:
        if kw in text.lower():
            return kw
    return None


class MedicalCorrectionOrienter(OrientStep):
    """Orient: build per-flaw correction lists by category."""

    # Map each category to its canonical case type so that
    # corrections target the correct case class regardless of
    # the red-team candidate's actual case_type.
    _CAT_CASE_TYPE = {
        "undertriaged_med": "medication",
        "lab_no_context": "lab_results",
        "missed_urgency": "symptom",
        "vulnerable_patient": "medication",
        "exploratory": "general",
    }

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        items = observation["verified_items"]
        planned = []

        for result, candidate in items:
            cat = candidate.context.get("category", "unknown")
            case_type = self._CAT_CASE_TYPE.get(
                cat, candidate.context.get("case_type", "general"))

            steps = candidate.trajectory.get("steps", [])
            draft = ""
            patient_msg = ""
            if steps:
                draft = steps[0].get("payload", {}).get("draft_reply", "")
                patient_msg = steps[0].get("payload", {}).get("patient_message", "")

            corrections = []

            if cat == "missed_urgency":
                combined = draft + " " + patient_msg
                kw = _extract_urgency_kw(combined)
                if kw:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.MEDICAL_HARD_BLOCK,
                        payload={"keyword": kw}))
                for ev in WEAK_EVIDENCE:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.THRESHOLD_ADJUSTMENT,
                        payload={"key": f"{case_type}|{ev}",
                                 "min_disposition": "urgent_escalation"}))

            elif cat == "undertriaged_med":
                kw = _extract_med_kw(draft)
                if kw:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.MEDICAL_HARD_BLOCK,
                        payload={"keyword": kw}))
                for ev in WEAK_EVIDENCE:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.THRESHOLD_ADJUSTMENT,
                        payload={"key": f"{case_type}|{ev}",
                                 "min_disposition": "clinician_review"}))

            elif cat == "lab_no_context":
                kw = _extract_med_kw(draft)
                if kw:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.MEDICAL_HARD_BLOCK,
                        payload={"keyword": kw}))
                for ev in WEAK_EVIDENCE:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.THRESHOLD_ADJUSTMENT,
                        payload={"key": f"{case_type}|{ev}",
                                 "min_disposition": "nurse_review"}))

            elif cat in ("vulnerable_patient", "exploratory"):
                kw = _extract_med_kw(draft) or _extract_urgency_kw(draft)
                if kw:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.MEDICAL_HARD_BLOCK,
                        payload={"keyword": kw}))
                for ev in WEAK_EVIDENCE:
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.THRESHOLD_ADJUSTMENT,
                        payload={"key": f"{case_type}|{ev}",
                                 "min_disposition": "clinician_review"}))

            acuity = candidate.context.get("acuity", "routine")
            for ev in WEAK_EVIDENCE:
                for acu in ("routine", "semi_urgent", "urgent"):
                    corrections.append(LocalCorrection(
                        correction_type=CorrectionType.AUDIT_COVERAGE_UPDATE,
                        payload={"case_class": f"{case_type}|{ev}|{acu}"}))

            planned.append({
                "candidate": candidate,
                "result": result,
                "corrections": corrections,
                "category": cat,
            })

        return {**observation, "planned": planned}
